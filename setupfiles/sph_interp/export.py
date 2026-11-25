"""Export an AMR interpolation result to other codes / mesh formats.

The library's input side converts any code *into* the canonical `Particles`
container (`from_tipsy`, `from_gadget`, ...). The output side is the mirror: an
AMR `GridResult` is a code-agnostic description of an octree mesh — a flat list
of leaf cells with centre, size, refinement level, base-cell linkage and one
value per field — and per-code **export adapters** translate *out of* it.

`to_cell_table` is the universal intermediate every exporter builds from: a plain
dict of equal-length arrays (one row per leaf). From it a writer for a specific
code is a thin function — group by `level` for patch/grid codes (Enzo/FLASH),
walk `base`+`level` for octree codes (RAMSES), or dump straight to HDF5/npz/VTK
for post-processing/visualization. `to_ramses` writes a self-describing HDF5 of
the RAMSES hydro variables now; `to_vtk`/`to_enzo` and the RAMSES native
amr_/hydro_ binary backend are declared as documented extension points.

Magnetic fields & RAMSES (constrained transport): RAMSES MHD stores B
FACE-CENTERED — six components per cell (left/right faces in x,y,z) — to keep
div B = 0. The AMR interpolation here is cell-CENTERED, so a CT-correct B export
needs a face-centered deposit (sample each B component at the relevant cell-face
centres) feeding six extra hydro variables. The hydro-variable list is
order-driven and extensible, so adding those is additive — no restructuring.
"""

import math
import struct

import numpy as np

from .targets import AMRGrid, UniformGrid, VoronoiGrid

#: mu_0 in Gaussian cgs (the MU0CGS in IC_writeascii.py / gasoline's Bfac).
MU0CGS = 4.0 * math.pi
#: Magnetic-field unit conversion: gasoline/tipsy code units (mu_0 = 1,
#: E_mag = 1/2 B^2 V -- "Heaviside-Lorentz") -> the GAUSS units the Gadget-family
#: codes (GIZMO, AREPO, OpenGadget3) expect for the IC MagneticField. All three
#: INTERPRET the IC B as Gauss and internally reduce the magnetic PRESSURE by 4pi
#: (GIZMO/AREPO divide the field by sqrt(4pi); OpenGadget3 keeps the field but uses
#: mu0_inv = 1/4pi), so a mu0=1 field fed 1:1 is 4pi too weak dynamically. The
#: suite generates each code's parameterfile with code units = 1 (UnitMass_in_g =
#: UnitLength_in_cm = UnitVelocity = 1, UnitMagneticField_in_gauss = 1), for which
#: the conversion is exactly sqrt(4pi) (= the codes' own UNIT_B_IN_GAUSS =
#: sqrt(4pi * UnitPressure_cgs)). Multiply the code B by this on WRITE; divide the
#: snapshot B by it on READ. (This is gasoline's CodetoGauss specialised to code
#: units = 1; the full physical CodetoGauss is only needed if a code's param uses
#: gasoline's kpc/Msol units instead.)
MU0_CODE_TO_GAUSS = math.sqrt(MU0CGS)


def scale_hdf5_bfield(path, factor):
    """Multiply PartType0/MagneticField in a GADGET-type-0 HDF5 IC by `factor`,
    in place. No-op returning False if the file carries no MagneticField (hydro
    IC); returns True after scaling an MHD IC."""
    import h5py
    with h5py.File(path, "r+") as f:
        g = f.get("PartType0")
        if g is None or "MagneticField" not in g:
            return False
        g["MagneticField"][...] = np.asarray(g["MagneticField"][:]) * float(factor)
    return True


def to_cell_table(result, fields=None):
    """Flatten a `GridResult` into a code-agnostic per-cell table.

    result : a `GridResult` whose target is an `AMRGrid` (one row per leaf) or a
             `UniformGrid` (one row per cell, flattened in C order).
    fields : field names to include (default: every data field except the
             internal `_mass`).

    Returns dict[str -> (Ncell,) ndarray] — the canonical intermediate for any
    mesh-code exporter. Geometry keys: 'x','y','z' (cell centres), 'dx','dy','dz'
    (cell sizes), 'volume', 'level' (0=base/uniform), 'base' (flat base-grid
    index — the octree-root linkage; the cell index for a uniform grid). Plus one
    column per requested field. Trivially saved with `np.savez`, written to HDF5,
    or turned into a structured array / DataFrame by the caller.
    """
    grid = result.target
    if isinstance(grid, AMRGrid):
        c = grid.centres()
        s = grid.cell_size
        table = {
            "x": np.ascontiguousarray(c[:, 0]),
            "y": np.ascontiguousarray(c[:, 1]),
            "z": np.ascontiguousarray(c[:, 2]),
            "dx": np.ascontiguousarray(s[:, 0]),
            "dy": np.ascontiguousarray(s[:, 1]),
            "dz": np.ascontiguousarray(s[:, 2]),
            "volume": grid.volumes(),
            "level": grid.cell_level,
            "base": grid.cell_base,
        }
        flat = (lambda a: np.asarray(a))            # already (Ncell,)
    elif isinstance(grid, UniformGrid):
        cx, cy, cz = grid.centres()
        X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        n = X.size
        pw = grid.pixwidth
        table = {
            "x": X.ravel(), "y": Y.ravel(), "z": Z.ravel(),
            "dx": np.full(n, pw[0]), "dy": np.full(n, pw[1]),
            "dz": np.full(n, pw[2]),
            "volume": np.full(n, grid.cell_volume),
            "level": np.zeros(n, dtype=np.int64),
            "base": np.arange(n, dtype=np.int64),
        }
        flat = (lambda a: np.asarray(a).ravel())    # (nx,ny,nz) -> (n,), C order
    elif isinstance(grid, VoronoiGrid):
        c = grid.centres()                          # generators (== cell seeds)
        v = grid.volumes()
        s = np.cbrt(v)                              # nominal cell size (vol^(1/3))
        n = c.shape[0]
        table = {
            "x": np.ascontiguousarray(c[:, 0]),
            "y": np.ascontiguousarray(c[:, 1]),
            "z": np.ascontiguousarray(c[:, 2]),
            "dx": s, "dy": s.copy(), "dz": s.copy(),
            "volume": v,
            "level": np.zeros(n, dtype=np.int64),
            "base": np.arange(n, dtype=np.int64),
        }
        flat = (lambda a: np.asarray(a))            # already (Ncell,) [central]
    else:
        raise TypeError("to_cell_table expects an AMRGrid, UniformGrid or "
                        f"VoronoiGrid GridResult (got target {type(grid).__name__})")
    if fields is None:
        fields = [k for k in result.data if not k.startswith("_")]
    for k in fields:
        table[k] = flat(result.data[k])
    return table


def to_structured_array(result, fields=None):
    """`to_cell_table` as a single numpy structured array (one record per leaf).

    Convenient for `np.save`, `tofile`, or as a drop-in mesh record array.
    """
    table = to_cell_table(result, fields=fields)
    dt = [(k, v.dtype) for k, v in table.items()]
    out = np.empty(next(iter(table.values())).shape[0], dtype=dt)
    for k, v in table.items():
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
#  Per-code export adapters — extension points (NOT yet implemented).
#  Each consumes `to_cell_table(result)` and writes the target format. The
#  canonical table already carries everything these need (centre, size, level,
#  base linkage, fields); implementing one is a format-serialization task, not a
#  re-derivation of the mesh.
# --------------------------------------------------------------------------- #

def to_vtk(result, path, fields=None):
    """Write the leaves as a VTK unstructured grid of hexahedra (ParaView/VisIt).

    Contract: 8 corner points per leaf from (x,y,z)±(dx,dy,dz)/2, VTK_HEXAHEDRON
    cells, fields as CELL_DATA. Pure ASCII `.vtk` needs no extra deps.
    """
    raise NotImplementedError(
        "to_vtk: build hexahedra from to_cell_table(result) and write a VTK "
        "UNSTRUCTURED_GRID with the fields as CELL_DATA.")


#: Canonical RAMSES hydro-variable order (ideal hydro, ndim=3). For MHD, RAMSES
#: inserts the SIX face-centered magnetic components (Bx_l,By_l,Bz_l,Bx_r,By_r,
#: Bz_r) BETWEEN the velocities and the pressure -> nvar becomes 11; provide them
#: as named fields once a face-centered deposit exists (see module note on the
#: constrained-transport representation). The writer is order-driven, so adding B
#: is just extending `hydro_fields`.
RAMSES_HYDRO_VARS = ("rho", "vx", "vy", "vz", "P")

#: RAMSES MHD variable order: the six FACE-centred B components (left x/y/z, then
#: right x/y/z) flank the pressure, as in RAMSES's hydro output. Produce the B
#: fields with `solenoidal.project_solenoidal` (cell-centred B -> div-free faces)
#: before exporting with this ordering.
RAMSES_MHD_VARS = ("rho", "vx", "vy", "vz",
                   "Bx_l", "By_l", "Bz_l", "P", "Bx_r", "By_r", "Bz_r")


def to_ramses(result, path, hydro_fields=RAMSES_HYDRO_VARS, gamma=5.0 / 3.0,
              boxlen=None, units=None, time=0.0, fmt="hdf5"):
    """Export an AMR interpolation to a RAMSES-flavored mesh file.

    result       : a `GridResult` whose target is an `AMRGrid`, carrying the
                   hydro fields (interpolate them first, e.g.
                   `interpolate(p, amr, values=["rho","vx","vy","vz","P"])`).
    hydro_fields : ordered RAMSES hydro variables to write (default
                   `RAMSES_HYDRO_VARS`). Each must be a field in `result.data`.
                   For MHD, extend this with the magnetic components in RAMSES
                   order (face-centered B between v and P).
    gamma,boxlen,units,time : run metadata. `boxlen` defaults to the max domain
                   side; `units` is an optional dict (e.g. unit_l/unit_d/unit_t).
    fmt          : "hdf5" (implemented; self-describing, reader-agnostic) or
                   "native" (RAMSES amr_/hydro_ Fortran binary — NOT yet
                   implemented; needs a RAMSES reader to validate against).

    HDF5 layout (cell-centered leaf list — the data, decoupled from RAMSES's
    on-disk oct nesting, which a native backend would rebuild from level+base):
      /cells/{x,y,z,dx,level,base,ramses_level}   leaf geometry
      /hydro/data                                 (Ncell, nvar) in var order
      /hydro  attrs: varnames, nvar
      /        attrs: ndim, boxlen, levelmin, levelmax, gamma, time, ncpu,
                      ordering, mhd, + any units
    """
    grid = result.target
    if isinstance(grid, AMRGrid):
        nb = grid.base_npx
    elif isinstance(grid, UniformGrid):
        nb = grid.npx
    else:
        raise TypeError("to_ramses expects an AMRGrid or UniformGrid GridResult "
                        f"(got target {type(grid).__name__})")
    if fmt == "native":
        raise NotImplementedError(
            "to_ramses(fmt='native'): RAMSES amr_/hydro_ Fortran-binary writer "
            "is not implemented (no RAMSES reader installed here to validate "
            "against). Use fmt='hdf5'; the native backend would rebuild the oct "
            "nesting from the leaves' level+base and serialize per-cpu records.")
    if fmt != "hdf5":
        raise ValueError(f"unknown fmt {fmt!r} (use 'hdf5' or 'native')")

    missing = [f for f in hydro_fields if f not in result.data]
    if missing:
        raise KeyError(f"result is missing hydro field(s) {missing}; interpolate "
                       f"them (have {sorted(result.data)})")

    import h5py
    table = to_cell_table(result, fields=list(hydro_fields))
    hydro = np.ascontiguousarray(
        np.stack([table[f] for f in hydro_fields], axis=1))   # (Ncell, nvar)

    # RAMSES levels: base level l0 = log2(base_npx) (RAMSES needs a power-of-two
    # cubic base grid); a leaf at refinement r sits at RAMSES level l0 + r. For a
    # UniformGrid every cell is at the base level (table 'level' is all 0).
    cell_level = table["level"]
    l0f = np.log2(nb[0])
    cubic_pow2 = (nb[0] == nb[1] == nb[2]) and float(l0f).is_integer()
    if cubic_pow2:
        l0 = int(round(l0f))
        ramses_level = l0 + cell_level
        levelmin, levelmax = l0, int(l0 + cell_level.max())
    else:
        ramses_level = cell_level.copy()
        levelmin = levelmax = -1            # not a RAMSES-compatible base grid

    if boxlen is None:
        boxlen = float(np.max(grid.box))

    with h5py.File(path, "w") as f:
        c = f.create_group("cells")
        c.create_dataset("x", data=table["x"])
        c.create_dataset("y", data=table["y"])
        c.create_dataset("z", data=table["z"])
        c.create_dataset("dx", data=table["dx"])
        c.create_dataset("level", data=table["level"])
        c.create_dataset("base", data=table["base"])
        c.create_dataset("ramses_level", data=ramses_level)
        hg = f.create_group("hydro")
        hg.create_dataset("data", data=hydro)
        hg.attrs["varnames"] = np.array(hydro_fields, dtype=h5py.string_dtype())
        hg.attrs["nvar"] = len(hydro_fields)
        f.attrs["ndim"] = 3
        f.attrs["ncell"] = int(hydro.shape[0])
        f.attrs["boxlen"] = boxlen
        f.attrs["levelmin"] = levelmin
        f.attrs["levelmax"] = levelmax
        f.attrs["gamma"] = gamma
        f.attrs["time"] = time
        f.attrs["ncpu"] = 1
        f.attrs["ordering"] = "cell-list"
        f.attrs["mhd"] = any(v in hydro_fields for v in
                             ("Bx", "By", "Bz", "Bx_l", "Bx_r",
                              "By_l", "By_r", "Bz_l", "Bz_r"))
        if not cubic_pow2:
            f.attrs["warning"] = ("base_npx is not a power-of-two cube; "
                                  "levelmin/levelmax unset (not RAMSES-native).")
        for k, v in (units or {}).items():
            f.attrs[k] = v
    return path


# --------------------------------------------------------------------------- #
#  POLARIS (radiative transfer: synchrotron / dust polarization) octree export.
#
#  POLARIS reads ONE binary grid file whose first two bytes select the geometry;
#  id 20 is the cartesian AMR octree, which is a 1:1 structural match for our
#  AMRGrid. Layout, verified against utilities/POLARIS/src/GridOcTree.cpp
#  (loadGridFromBinaryFile, :458-700; writer :1270-1378):
#
#      uint16   grid_id                 = 20
#      uint16   data_offset             = quantities per cell
#      uint16   data_ids[data_offset]   = which POLARIS quantity each slot holds
#      float64  cube_length             = root cube edge (x conv_len -> metres)
#      <depth-first record stream>
#
#  Each record is `uint16 is_leaf`, `uint16 level`, followed (leaves only) by
#  `float32 data[data_offset]`. The loader walks the stream with a cursor: an
#  is_leaf=0 record DESCENDS into the next free child slot, an is_leaf=1 record
#  FILLS it, and the stream ends when the root's 8th child closes. Consequences
#  we have to honour exactly:
#
#    * The root record (is_leaf=0, level=0) must be written first, and the tree
#      must reach at least level 1 -- a single-cell grid is rejected (:1284).
#    * Every internal node needs EXACTLY 8 children (`cube_pos > 7` and
#      `parent.level + 1 != level` are both hard errors, :611-627), i.e. the mesh
#      must be space-filling. Build the AMRGrid with empty_max_level=None (the
#      default) -- dropping empty siblings produces an unreadable file.
#    * Child slots are geometric, index = ix + 2*iy + 4*iz (createBoundingCell,
#      :1566-1630) -- the same octant convention AMRGrid.build already uses.
#    * The root cube is placed at [-L/2, L/2]^3 (:548-552), so cell POSITIONS are
#      implicit in the tree: exporting a domain that is not centred on the origin
#      silently translates the galaxy. The recentre shift is returned in the
#      metadata for the sidecar rather than applied to any coordinate.
#    * Per-cell values are float32 (only ~7 significant digits), so all unit
#      scaling happens in float64 and is cast once, on write.
#
#  Units are the caller's job: POLARIS applies conv_dens/len/mag/vel from the
#  command file to whatever is in the file, and conv_dens hits gas density, n_th
#  AND n_cr with a single factor -- so the only self-consistent choice is to
#  write SI here and leave every conv_* at 1.0. `units`/`length_scale` are plain
#  multiplicative factors; this module stays physics-free.
# --------------------------------------------------------------------------- #

#: POLARIS grid geometry ids (`Typedefs.hpp:259-262`). Only the octree is written.
POLARIS_GRID_ID_OCTREE = 20

#: POLARIS per-cell quantity ids (`Typedefs.hpp:209-247`), keyed by both the
#: POLARIS name and this library's field name where they differ. The SI units
#: POLARIS expects: gas_mdens kg/m^3, gas_dens m^-3, temperatures K, B TESLA,
#: velocities m/s, n_th/n_cr m^-3, g_min/g_max/cr_p dimensionless.
#: NOTE: id 27 is the CR power-law index; it is deliberately NOT keyed as "p" to
#: avoid colliding with the pressure field name used by the RAMSES exporter.
POLARIS_DATA_IDS = {
    "gas_dens": 0,          # gas NUMBER density [m^-3]
    "n_gas": 0,
    "gas_mdens": 28,        # gas MASS density [kg/m^3]
    "rho": 28,
    "gas_temp": 3,
    "T": 3,
    "Bx": 4, "By": 5, "Bz": 6,
    "mx": 4, "my": 5, "mz": 6,
    "vx": 7, "vy": 8, "vz": 9,
    "n_th": 22,             # thermal electron number density [m^-3]
    "T_e": 23,
    "n_cr": 24,             # CR electron number density [m^-3], TOTAL over gamma
    "g_min": 25, "g_max": 26, "cr_p": 27,
    "cr_gamma_min": 25, "cr_gamma_max": 26,
}


def polaris_data_ids(fields):
    """Map field names to POLARIS quantity ids via `POLARIS_DATA_IDS`."""
    unknown = [f for f in fields if f not in POLARIS_DATA_IDS]
    if unknown:
        raise KeyError(
            f"no POLARIS quantity id known for field(s) {unknown}; pass "
            f"data_ids= explicitly or extend POLARIS_DATA_IDS "
            f"(known: {sorted(set(POLARIS_DATA_IDS))})")
    return [POLARIS_DATA_IDS[f] for f in fields]


def _polaris_octree_stream(nb, lo, box, cell_lo, cell_size, cell_level, cell_base):
    """Turn a flat AMR leaf list into POLARIS's depth-first record stream.

    Returns (is_leaf, level, leaf_order, base_level):
      is_leaf, level : (Nrec,) the two uint16 fields of every record, in order
      leaf_order     : (Nleaf,) leaf indices in depth-first order (the order
                       their float32 payloads appear in)
      base_level     : log2(base_npx) -- the synthetic levels above the base grid

    The mapping is the one thing here that is not a transcription of the format.
    `AMRGrid` is a FOREST: base_npx**3 independent octree roots. POLARIS wants a
    single root. When base_npx = 2**k on every axis, the base grid IS exactly
    level k of one octree, so the forest becomes a tree by prepending synthetic
    internal levels 0..k-1 and shifting every leaf level by k.

    Depth-first order is then a sort: give each leaf its octant PATH from the
    single root -- 3 bits per level, most significant first, zero-padded to the
    deepest level -- and lexicographic order on those paths IS pre-order DFS
    (no leaf's path can prefix another's in a full octree). The internal records
    fall out of the same array: between two consecutive leaves, exactly the
    ancestors below their common path prefix are being entered for the first
    time, so leaf i is preceded by internal records at levels
    (common_prefix + 1) ... (level_i - 1).
    """
    nb = np.asarray(nb, dtype=np.int64).reshape(3)
    if not (nb[0] == nb[1] == nb[2]):
        raise ValueError(
            f"POLARIS needs a cubic single-root octree, but base_npx={tuple(nb)} "
            "differs per axis; rebuild the AMRGrid with a scalar base_npx.")
    k0f = math.log2(int(nb[0]))
    if not float(k0f).is_integer():
        raise ValueError(
            f"POLARIS needs base_npx a power of two (the base grid becomes level "
            f"log2(base_npx) of one octree), got base_npx={int(nb[0])}. "
            "Use e.g. 8, 16 or 32.")
    k0 = int(round(k0f))

    box = np.asarray(box, dtype=np.float64).reshape(3)
    if not np.allclose(box, box[0], rtol=1e-9, atol=0.0):
        raise ValueError(
            f"POLARIS's octree domain must be a CUBE, got box={tuple(box)}; "
            "rebuild the AMRGrid with a cubic lo/hi.")

    nleaf = cell_lo.shape[0]
    if nleaf == 0:
        raise ValueError("empty AMRGrid: nothing to export")
    lmax = int(cell_level.max())
    level_glob = k0 + cell_level.astype(np.int64)
    if level_glob.max() < 1:
        raise ValueError(
            "POLARIS rejects an unrefined single-cell octree (GridOcTree.cpp:1284): "
            "the tree must reach level 1. Use base_npx >= 2 or max_level >= 1.")
    if level_glob.max() > 255:
        raise ValueError(
            f"POLARIS stores the level in a uchar; log2(base_npx) + max_level = "
            f"{int(level_glob.max())} exceeds 255.")

    ndig = k0 + lmax                      # octant digits from the single root
    if 3 * ndig > 62:
        raise ValueError(
            f"tree too deep to key as one int64 path: log2(base_npx) + max_level "
            f"= {ndig} (limit 20)")

    # Space-filling check. The leaves are disjoint octants of the root cube, so
    # they tile it iff the octree is full -- exactly POLARIS's "every internal
    # node has 8 children" requirement, testable as a volume budget.
    vol = float(np.sum(cell_size[:, 0] * cell_size[:, 1] * cell_size[:, 2]))
    vol_box = float(box[0] * box[1] * box[2])
    if abs(vol / vol_box - 1.0) > 1e-9:
        raise ValueError(
            f"the AMR mesh does not tile the domain (sum of leaf volumes / box "
            f"volume = {vol / vol_box:.12f}), so the octree is not full and "
            "POLARIS would reject the file. Rebuild the AMRGrid with "
            "empty_max_level=None (the default) so empty siblings are kept.")

    # --- octant digits, most significant (level 1) first ------------------- #
    nbx = nby = nbz = int(nb[0])
    b = cell_base.astype(np.int64)
    bix = b // (nby * nbz)
    biy = (b // nbz) % nby
    biz = b % nbz

    bpw = box / nb                                   # base-cell edge
    base_lo = (np.asarray(lo, dtype=np.float64).reshape(1, 3)
               + np.stack([bix, biy, biz], axis=1).astype(np.float64) * bpw)
    # leaf offset inside its base cell, in units of the FINEST cell. A leaf at
    # refinement L is aligned to 2**(lmax-L) of these, so its trailing digits are
    # zero -- which is precisely the padding the path sort needs.
    nfine = 1 << lmax
    off = np.rint((cell_lo - base_lo) / bpw * nfine).astype(np.int64)
    if np.any(off < 0) or np.any(off >= nfine):
        raise ValueError("leaf lower corners are not inside the base cell that "
                         "cell_base names; cell_lo/cell_base/lo disagree")
    # the leaf must also BE the octant its level claims (size = base / 2**level)
    expect = bpw / np.power(2.0, cell_level.astype(np.float64))[:, None]
    if not np.allclose(cell_size, expect, rtol=1e-9, atol=0.0):
        raise ValueError("leaf sizes disagree with their refinement levels "
                         "(cell_size != base_pw / 2**cell_level)")

    dig = np.empty((nleaf, ndig), dtype=np.uint8)
    for j in range(1, k0 + 1):                       # levels 1..k0: base index
        s = k0 - j
        dig[:, j - 1] = (((bix >> s) & 1)
                         + 2 * ((biy >> s) & 1)
                         + 4 * ((biz >> s) & 1))
    for j in range(1, lmax + 1):                     # levels k0+1..k0+lmax
        s = lmax - j
        dig[:, k0 + j - 1] = (((off[:, 0] >> s) & 1)
                              + 2 * ((off[:, 1] >> s) & 1)
                              + 4 * ((off[:, 2] >> s) & 1))

    path = np.zeros(nleaf, dtype=np.int64)
    for j in range(ndig):
        path |= dig[:, j].astype(np.int64) << (3 * (ndig - 1 - j))
    order = np.argsort(path, kind="stable")          # == pre-order DFS
    dig = dig[order]
    level_df = level_glob[order]

    # --- common path prefix with the previous leaf -------------------------- #
    #  prefix length d -> ancestors at levels 0..d are already open, so the new
    #  internal records are levels d+1 .. level-1. d = -1 for the first leaf
    #  (the root itself still has to be opened).
    d = np.empty(nleaf, dtype=np.int64)
    d[0] = -1
    if nleaf > 1:
        neq = dig[1:] != dig[:-1]
        if not neq.any(axis=1).all():
            raise ValueError("duplicate leaf paths in the AMR mesh (two leaves "
                             "with identical position and level)")
        d[1:] = np.argmax(neq, axis=1)

    n_int = level_df - 1 - d                         # internals before each leaf
    if np.any(n_int < 0):
        raise AssertionError("negative internal-record count (impossible for a "
                             "full octree; leaf list is inconsistent)")
    total_int = int(n_int.sum())
    # A full 8-ary tree satisfies leaves = 7 * internals + 1 exactly -- an
    # independent structural check on the walk above.
    if total_int * 7 + 1 != nleaf:
        raise AssertionError(
            f"octree structure check failed: {nleaf} leaves with {total_int} "
            f"internal nodes (a full octree needs leaves = 7*internals + 1)")

    nrec = nleaf + total_int
    is_leaf = np.zeros(nrec, dtype=np.int64)
    level = np.zeros(nrec, dtype=np.int64)

    end = np.cumsum(n_int + 1)                       # one block per leaf
    leaf_pos = end - 1
    is_leaf[leaf_pos] = 1
    level[leaf_pos] = level_df
    if total_int:
        grp_start = np.cumsum(n_int) - n_int
        within = np.arange(total_int) - np.repeat(grp_start, n_int)
        int_pos = np.repeat(end - n_int - 1, n_int) + within
        level[int_pos] = np.repeat(d + 1, n_int) + within

    return is_leaf, level, order, k0


def to_polaris(result, path, fields, data_ids=None, units=None,
               length_scale=1.0, validate=True):
    """Write an AMR interpolation as a POLARIS octree grid file (grid_id 20).

    result   : a `GridResult` whose target is an `AMRGrid` (the octree case) or a
               `UniformGrid` (the degenerate case: a fully-refined tree of level
               log2(npx)). The domain must be a CUBE and `base_npx`/`npx` a power
               of two on every axis; build the AMR mesh with the default
               empty_max_level=None so it stays space-filling.
    path     : output grid file (POLARIS's `<path_grid>`).
    fields   : ORDERED field names to write, one float32 slot per cell each. The
               order is the file's slot order and is what `data_ids` labels.
               `CMD_SYNCHROTRON` needs a gas density (id 0 or 28), the three B
               components, and n_th and/or n_cr -- plus g_min/g_max/cr_p whenever
               n_cr is present (missing ones are a hard error in POLARIS, while a
               missing/mis-scaled n_cr is a silently black map).
    data_ids : POLARIS quantity ids matching `fields` (default: looked up in
               `POLARIS_DATA_IDS`).
    units    : optional {field: factor} applied in float64 before the float32
               cast -- the hook for the code-unit -> SI conversion. Fields absent
               from the dict are written unscaled.
    length_scale : factor turning the grid's length unit into the one written as
               `cube_length` (metres, if every conv_* stays 1.0).
    validate : reject non-finite values (they would ray-trace to garbage rather
               than fail) -- keep on unless deliberately inspecting a bad grid.

    Returns a metadata dict for the run's sidecar (`<out>.polaris.json`):
    cell/record counts, level range, `cube_length`, the
    implied `recentre_shift` (POLARIS pins the root cube to [-L/2, L/2]^3, so a
    domain that was not origin-centred is translated by this vector), the
    field/id order and the per-field min/max actually written.
    """
    grid = result.target
    if isinstance(grid, AMRGrid):
        nb = grid.base_npx
        cell_lo, cell_size = grid.cell_lo, grid.cell_size
        cell_level, cell_base = grid.cell_level, grid.cell_base
        lo, hi = grid.lo, grid.hi
        flat = (lambda a: np.asarray(a))
    elif isinstance(grid, UniformGrid):
        nb = grid.npx
        table = to_cell_table(result, fields=[])
        cell_lo = np.stack([table["x"] - 0.5 * table["dx"],
                            table["y"] - 0.5 * table["dy"],
                            table["z"] - 0.5 * table["dz"]], axis=1)
        cell_size = np.stack([table["dx"], table["dy"], table["dz"]], axis=1)
        cell_level = table["level"]                  # all zero: the base grid
        cell_base = table["base"]
        lo, hi = grid.lo, grid.hi
        flat = (lambda a: np.asarray(a).ravel())
    else:
        raise TypeError("to_polaris expects an AMRGrid or UniformGrid GridResult "
                        f"(got target {type(grid).__name__})")

    fields = list(fields)
    if not fields:
        raise ValueError("to_polaris needs at least one field (POLARIS requires "
                         "a gas density for any RT calculation)")
    missing = [f for f in fields if f not in result.data]
    if missing:
        raise KeyError(f"result is missing field(s) {missing} "
                       f"(have {sorted(result.data)})")
    ids = polaris_data_ids(fields) if data_ids is None else list(data_ids)
    if len(ids) != len(fields):
        raise ValueError(f"data_ids has {len(ids)} entries for {len(fields)} fields")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate POLARIS quantity ids {ids}: each id may only "
                         "occupy one slot")
    if any(not (0 <= int(i) <= 0xFFFF) for i in ids):
        raise ValueError(f"POLARIS quantity ids must fit in uint16, got {ids}")

    box = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    is_leaf, level, order, k0 = _polaris_octree_stream(
        nb, lo, box, np.asarray(cell_lo, dtype=np.float64),
        np.asarray(cell_size, dtype=np.float64),
        np.asarray(cell_level, dtype=np.int64),
        np.asarray(cell_base, dtype=np.int64))

    # values: float64 scaling, ONE cast to float32, in depth-first leaf order
    units = dict(units or {})
    ndata = len(fields)
    vals = np.empty((order.size, ndata), dtype=np.float64)
    ranges = {}
    for j, f in enumerate(fields):
        col = np.asarray(flat(result.data[f]), dtype=np.float64) * float(
            units.get(f, 1.0))
        if col.shape[0] != order.size:
            raise ValueError(f"field {f!r} has {col.shape[0]} values for "
                             f"{order.size} cells")
        if validate and not np.all(np.isfinite(col)):
            bad = int(np.count_nonzero(~np.isfinite(col)))
            raise ValueError(
                f"field {f!r} has {bad} non-finite value(s); POLARIS would "
                "ray-trace these silently. Fix the deposit or pass "
                "validate=False.")
        vals[:, j] = col[order]
        ranges[f] = (float(col.min()), float(col.max()))
    vals32 = vals.astype(np.float32)

    cube_length = float(box[0]) * float(length_scale)
    if not cube_length > 0:
        raise ValueError(f"cube_length must be positive, got {cube_length}")

    # --- serialise. Every record is a whole number of 4-byte words (a 2x uint16
    #     header, plus ndata float32 for a leaf), so the stream is built as one
    #     uint32 buffer and scattered into, instead of per-record struct.pack.
    nrec = is_leaf.size
    words = 1 + is_leaf * ndata
    word_start = np.cumsum(words) - words
    buf = np.zeros(int(words.sum()), dtype="<u4")
    buf[word_start] = (is_leaf.astype("<u4")
                       | (level.astype("<u4") << 16))     # little-endian pair
    leaf_words = word_start[is_leaf == 1] + 1
    buf.view("<f4")[leaf_words[:, None] + np.arange(ndata)] = vals32

    with open(path, "wb") as fh:
        fh.write(struct.pack("<HH", POLARIS_GRID_ID_OCTREE, ndata))
        fh.write(struct.pack("<%dH" % ndata, *[int(i) for i in ids]))
        fh.write(struct.pack("<d", cube_length))
        fh.write(buf.tobytes())

    centre = 0.5 * (np.asarray(lo, dtype=np.float64)
                    + np.asarray(hi, dtype=np.float64))
    return {
        "path": str(path),
        "grid_id": POLARIS_GRID_ID_OCTREE,
        "ncell": int(order.size),
        "n_internal": int(nrec - order.size),
        "nrecord": int(nrec),
        "base_level": int(k0),
        "level_min": int(level[is_leaf == 1].min()),
        "level_max": int(level.max()),
        "cube_length": cube_length,
        "length_scale": float(length_scale),
        "recentre_shift": (-centre).tolist(),
        "fields": fields,
        "data_ids": [int(i) for i in ids],
        "units": {k: float(v) for k, v in units.items()},
        "field_range": ranges,
        "conv_note": "written in the caller's units; POLARIS conv_* must be 1.0",
    }


def read_polaris(path):
    """Read a POLARIS octree grid file back into leaf arrays (validation tool).

    A deliberately independent implementation of the grid format: a recursive
    descent that re-derives every leaf's geometry from the tree walk alone and
    re-checks POLARIS's own invariants (8 children per internal node,
    parent.level + 1 == level, the stream ending exactly when the root closes).
    Used to round-trip `to_polaris` -- the writer is vectorised and index-based,
    this is a per-record cursor, so agreement between them is a real check.

    Returns dict with `grid_id`, `data_ids`, `cube_length`, and per-leaf
    `level` (Nleaf,), `cell_lo`/`cell_size` (Nleaf,3), `data` (Nleaf, ndata) --
    leaves in the file's depth-first order.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    grid_id, ndata = struct.unpack_from("<HH", raw, 0)
    if grid_id != POLARIS_GRID_ID_OCTREE:
        raise ValueError(f"not a POLARIS octree grid (grid_id={grid_id}, "
                         f"expected {POLARIS_GRID_ID_OCTREE})")
    off = 4
    ids = list(struct.unpack_from("<%dH" % ndata, raw, off))
    off += 2 * ndata
    (cube_length,) = struct.unpack_from("<d", raw, off)
    off += 8

    lo0 = -0.5 * cube_length
    levels, cl, cs, data = [], [], [], []

    def descend(node_lo, node_len, expect_level, cursor):
        is_leaf, level = struct.unpack_from("<HH", raw, cursor)
        cursor += 4
        if level != expect_level:
            raise ValueError(f"level {level} where {expect_level} was expected "
                             f"(POLARIS checks parent.level + 1 == level)")
        if is_leaf == 1:
            levels.append(level)
            cl.append(node_lo)
            cs.append((node_len, node_len, node_len))
            data.append(struct.unpack_from("<%df" % ndata, raw, cursor))
            return cursor + 4 * ndata
        if is_leaf != 0:
            raise ValueError(f"is_leaf must be 0 or 1, got {is_leaf}")
        half = 0.5 * node_len
        for k in range(8):                    # child k = ix + 2*iy + 4*iz
            cursor = descend((node_lo[0] + (k & 1) * half,
                              node_lo[1] + ((k >> 1) & 1) * half,
                              node_lo[2] + ((k >> 2) & 1) * half),
                             half, level + 1, cursor)
        return cursor

    end = descend((lo0, lo0, lo0), cube_length, 0, off)
    if end != len(raw):
        raise ValueError(f"{len(raw) - end} trailing byte(s) after the root "
                         "closed: the record stream does not match the tree")
    return {
        "grid_id": grid_id,
        "data_ids": ids,
        "cube_length": cube_length,
        "ncell": len(levels),
        "level": np.array(levels, dtype=np.int64),
        "cell_lo": np.array(cl, dtype=np.float64),
        "cell_size": np.array(cs, dtype=np.float64),
        "data": np.array(data, dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
#  AREPO (moving-mesh) export.
#
#  AREPO is a finite-volume Voronoi code: an IC is just GENERATORS + per-cell
#  CONSERVED quantities; AREPO rebuilds its own Voronoi mesh from the generator
#  positions at run start and derives the primitives (rho = Mass/V_voronoi,
#  v = Momentum/Mass, ...). So ANY of our targets is a valid AREPO IC -- we hand
#  it the cell centres as generators plus the per-cell mass and (intensive)
#  primitive fields. Two paths:
#
#  * DEPOSIT (to_arepo on a GridResult): generators = cell centres of a
#    Uniform/AMR/Voronoi interpolation; Mass = rho_cell * V_cell (the extensive,
#    conserved quantity -> total mass preserved exactly regardless of how AREPO's
#    rebuilt cell volumes differ from ours); velocity / u / B are the intensive
#    cell means the deposit already produced. Use this when the mesh differs from
#    the particles (regridding / subsampling / structured grid).
#
#  * COPY (particles_to_arepo): generators = the SPH particles themselves, 1:1.
#    Each cell inherits its particle's mass (exact) and v/u/B verbatim -- no
#    kernel, no smoothing. The most faithful conversion; AREPO tessellates and
#    sets rho = m_i / V_voronoi_i. (Note: this swaps SPH's overlapping-kernel
#    volume for the space-filling Voronoi volume -- see particles_to_arepo doc.)
#
#  B is CELL-CENTRED in AREPO (Powell/Dedner cleaning), unlike RAMSES's
#  face-centred CT, so a single cell-centred B vector per generator is all that
#  is needed; the SPH field's div(B) is absorbed by AREPO's cleaning.
# --------------------------------------------------------------------------- #

#: Default per-particle field names mapped onto AREPO gas primitives.
AREPO_VELOCITY = ("vx", "vy", "vz")
AREPO_MAGNETIC = ("Bx", "By", "Bz")
#: tipsy stores the gas thermal state as a TEMPERATURE (tgdata col 8), so that is
#: the only thermodynamic source. AREPO wants specific internal energy u; the
#: gasoline conversion is u = dTuFac * T (read_tufac / `--tu-fac`).
AREPO_TEMPERATURE = ("temp", "T", "t")


def _internal_energy_from_temperature(get, temperature, tu_fac):
    """Specific internal energy u = (temperature field) * tu_fac for the AREPO IC.

    tipsy carries only a temperature column, so u comes from it via gasoline's
    dTuFac (`tu_fac`). `get(name)` returns the flattened cell field or None.
    Returns (u_array, description); raises if no temperature field is present."""
    for t in temperature:
        a = get(t)
        if a is not None:
            return a * tu_fac, f"{t!r} * tu_fac({tu_fac:g})  [u = dTuFac*T]"
    raise KeyError(
        f"to_arepo needs a temperature field (one of {temperature}) to set the "
        "specific internal energy u = dTuFac*T; add it to the interpolated "
        "fields (e.g. --fields rho,vx,vy,vz,temp).")


def _frame_coords(coords, box=None, periodic=False):
    """Place particle coordinates into AREPO's box frame [0, L] and pick L.

    box      : periodic period (3,) from the `.param` (authoritative). When
               given, each axis is centred on the data and coords are translated
               into [0, period_axis] (wrapped if `periodic`), so a gasoline box
               [-L/2,L/2] per axis maps to [0,L]. The returned `boxsize` is the
               X period (AREPO's scalar BoxSize = boxSize_X); a non-cubic box is
               flagged with the LONG_Y/LONG_Z ratios AREPO must be compiled with
               (boxSize_Y = BoxSize*LONG_Y, etc.).
    box=None : fall back to the particle extent (shift by min, L = max span).
    Returns (framed_coords, boxsize). boxsize is the X-axis period.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    if box is not None:
        box = np.asarray(box, dtype=np.float64).reshape(3)
        centre = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        lo = centre - 0.5 * box                         # per-axis box origin
        c = coords - lo
        if periodic:
            c = np.mod(c, box)                           # wrap strays into [0,L)
        if not np.allclose(box, box[0]):
            print(f"  non-cubic period {box}: AREPO needs LONG_Y="
                  f"{box[1] / box[0]:.8g} LONG_Z={box[2] / box[0]:.8g} "
                  f"(BoxSize={box[0]:g}).")
        return c, float(box[0])
    c = coords - coords.min(axis=0)
    return c, float(c.max() * (1.0 + 1e-6))


def _write_arepo_hdf5(path, coords, mass, vel, u, bfield, boxsize, ids=None,
                      time=0.0):
    """Write a GADGET/AREPO-format type-0 (gas) HDF5 IC snapshot.

    `coords` (N,3) must already be framed into [0, boxsize] (see `_frame_coords`);
    mass (N,), vel (N,3), u (N,) specific internal energy, bfield (N,3) or None.
    """
    import h5py
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = coords.shape[0]
    boxsize = float(boxsize)
    # keep strictly inside (0, L): AREPO rejects points on/over the box face
    eps = 1e-9 * boxsize
    coords = np.clip(coords, eps, boxsize - eps)
    if ids is None:
        ids = np.arange(1, n + 1, dtype=np.uint32)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        p0 = f.create_group("PartType0")
        npart = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        h.attrs.create("NumPart_ThisFile", npart)
        h.attrs.create("NumPart_Total", npart)
        h.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=np.int32))
        h.attrs.create("MassTable", np.zeros(6, dtype=np.float64))
        h.attrs.create("Time", float(time))
        h.attrs.create("Redshift", 0.0)
        h.attrs.create("BoxSize", np.float64(boxsize))
        h.attrs.create("NumFilesPerSnapshot", 1)
        h.attrs.create("Omega0", 0.0)
        h.attrs.create("OmegaB", 0.0)
        h.attrs.create("OmegaLambda", 0.0)
        h.attrs.create("HubbleParam", 1.0)
        for fl in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals",
                   "Flag_Feedback"):
            h.attrs.create(fl, 0)
        h.attrs.create("Flag_DoublePrecision", 1)
        p0.create_dataset("ParticleIDs", data=np.asarray(ids, dtype=np.uint32))
        p0.create_dataset("Coordinates", data=coords)
        p0.create_dataset("Masses", data=np.ascontiguousarray(mass, np.float64))
        p0.create_dataset("Velocities",
                          data=np.ascontiguousarray(vel, np.float64))
        p0.create_dataset("InternalEnergy",
                          data=np.ascontiguousarray(u, np.float64))
        if bfield is not None:
            p0.create_dataset("MagneticField",
                              data=np.ascontiguousarray(bfield, np.float64))
    return path, boxsize


def to_arepo(result, path, density="rho", velocity=AREPO_VELOCITY,
             temperature=AREPO_TEMPERATURE, tu_fac=1.0, magnetic=AREPO_MAGNETIC,
             box=None, periodic=False, time=0.0, preframed_boxsize=None):
    """Export an interpolation `GridResult` as an AREPO moving-mesh IC (HDF5).

    Works for UniformGrid / AMRGrid / VoronoiGrid results: the cell centres
    become the mesh generators; per cell we write the CONSERVED mass
    `Mass = rho * V_cell` and the intensive primitives (velocity, specific
    internal energy, cell-centred B). AREPO rebuilds the Voronoi mesh from the
    generators and recomputes density from its own cell volumes -- total mass is
    conserved exactly in the handoff.

    density       : field giving cell density rho (default 'rho'); required.
    velocity      : 3 field names for v (missing -> zero velocity).
    temperature   : candidate names for the tipsy temperature field; specific
                    internal energy is u = (temperature) * tu_fac (gasoline
                    dTuFac convention). A temperature field is REQUIRED.
    tu_fac        : dTuFac (u = dTuFac * T); read from the run .log by the CLI.
    magnetic      : 3 field names for cell-centred B (all present -> MHD IC).
    box           : periodic period (3,) for the AREPO BoxSize/frame (from the
                    `.param`); None -> use the generator extent.
    periodic      : wrap generators into [0, period) when `box` is given.
    preframed_boxsize : when set, the generators are ALREADY in AREPO's [0, L]
                    frame and `_frame_coords` is skipped; L = this value.  Used by
                    the `arepo-grid` decoupled re-deposit, whose generators come
                    from an ADDBACKGROUNDGRID pass already in [0, boxout] (box and
                    periodic are ignored).
    """
    grid = result.target
    if not isinstance(grid, (UniformGrid, AMRGrid, VoronoiGrid)):
        raise TypeError("to_arepo expects a UniformGrid/AMRGrid/VoronoiGrid "
                        f"GridResult (got {type(grid).__name__})")
    # Flatten geometry + every field to per-cell (Ncell,) arrays via the
    # canonical cell table (handles the uniform C-order ravel, AMR leaves and
    # Voronoi central cells uniformly).
    have = [k for k in result.data if not k.startswith("_")]
    table = to_cell_table(result, fields=have)
    coords = np.stack([table["x"], table["y"], table["z"]], axis=1)
    vol = table["volume"]

    def get(name):
        return table[name] if (name is not None and name in table) else None

    if get(density) is None:
        raise KeyError(f"to_arepo needs the density field {density!r}; have "
                       f"{sorted(have)} (interpolate it, e.g. --fields rho,...)")
    rho = get(density)
    mass = rho * vol

    vel = np.zeros((coords.shape[0], 3))
    vmiss = [c for c in velocity if get(c) is None]
    for k, c in enumerate(velocity):
        if get(c) is not None:
            vel[:, k] = get(c)

    u, udesc = _internal_energy_from_temperature(get, temperature, tu_fac)

    bfield = None
    if all(get(c) is not None for c in magnetic):
        bfield = np.stack([get(c) for c in magnetic], axis=1)

    if preframed_boxsize is not None:
        L = float(preframed_boxsize)          # coords already framed into [0, L]
    else:
        coords, L = _frame_coords(coords, box=box, periodic=periodic)
    path, L = _write_arepo_hdf5(path, coords, mass, vel, u, bfield, L,
                                time=time)
    print(f"  AREPO IC: {coords.shape[0]} generators ({type(grid).__name__}), "
          f"BoxSize={L:.6g}, total mass={mass.sum():.6g}")
    print(f"           u from {udesc}; velocity "
          f"{'zero' if len(vmiss) == 3 else 'from ' + ','.join(velocity)}; "
          f"{'MHD (cell-centred B)' if bfield is not None else 'hydro (no B)'}")
    return path


def particles_to_arepo(particles, path, velocity=AREPO_VELOCITY,
                       temperature=AREPO_TEMPERATURE, tu_fac=1.0,
                       magnetic=AREPO_MAGNETIC, box=None, periodic=None,
                       time=0.0):
    """COPY path: write the SPH particles directly as an AREPO IC (1 generator
    per particle, no interpolation). Each cell inherits its particle's mass
    (exact) and the intensive fields (v/u/B) verbatim from `particles.values`;
    AREPO tessellates and sets rho = m_i / V_voronoi_i.

    This is the most faithful transcription of the per-particle state, BUT note
    it swaps SPH's overlapping-kernel volume (m/rho_SPH) for the space-filling
    Voronoi cell volume, so the resulting cell DENSITY is AREPO's Voronoi
    estimate, not rho_SPH. For a kernel-consistent, mass-conserving density use
    the deposit path (`to_arepo` on a VoronoiGrid petkova result) instead.
    """
    vals = particles.values

    def get(name):
        return vals[name] if (name is not None and name in vals) else None

    n = particles.pos.shape[0]
    mass = np.ascontiguousarray(particles.mass, np.float64)
    vel = np.zeros((n, 3))
    vmiss = [c for c in velocity if get(c) is None]
    for k, c in enumerate(velocity):
        if get(c) is not None:
            vel[:, k] = get(c)
    u, udesc = _internal_energy_from_temperature(get, temperature, tu_fac)
    bfield = None
    if all(get(c) is not None for c in magnetic):
        bfield = np.stack([get(c) for c in magnetic], axis=1)

    # default the box / periodicity from the particle container (which from_tipsy
    # fills from the .param period).
    if box is None:
        box = particles.box
    if periodic is None:
        periodic = bool(np.any(np.atleast_1d(particles.periodic)))
    # Non-periodic (isolated) IC: there is no period. Frame to a single cubic
    # BoxSize = max particle extent (box=None path) rather than the per-axis span,
    # so no axis can exceed BoxSize and get clipped, and no spurious LONG_Y/Z is
    # reported. Periodic ICs keep the .param period.
    coords, L = _frame_coords(particles.pos,
                              box=(box if periodic else None),
                              periodic=periodic)
    path, L = _write_arepo_hdf5(path, coords, mass, vel, u, bfield, L,
                                time=time)
    print(f"  AREPO IC (copy): {n} particles -> generators, BoxSize={L:.6g}, "
          f"total mass={mass.sum():.6g}")
    print(f"           u from {udesc}; velocity "
          f"{'zero' if len(vmiss) == 3 else 'from ' + ','.join(velocity)}; "
          f"{'MHD (cell-centred B)' if bfield is not None else 'hydro (no B)'}")
    return path


def to_enzo(result, path, **kw):
    """Write Enzo/FLASH-style patch hierarchy (HDF5).

    Contract: these are patch-based, not pure octrees; tile each refinement
    `level` into rectangular grid patches (runs of same-level leaves sharing a
    base cell) and write the grid hierarchy + fields to HDF5.
    """
    raise NotImplementedError(
        "to_enzo: coalesce same-level leaves into rectangular patches and write "
        "the HDF5 grid hierarchy.")
