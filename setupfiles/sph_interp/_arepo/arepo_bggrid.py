"""AREPO ADDBACKGROUNDGRID regrid backend for the `arepo-grid` converter mode.

Runs AREPO (codes/arepo/Arepo_bggrid, built from Config_bggrid.sh) as a SHELL-OUT
IC pre-processor: given a particle AREPO IC (the exact 1:1 generator copy that
the `particles` mode produces), run `Arepo_bggrid param.txt 0` (RestartFlag 0 =
read the IC).  When ADDBACKGROUNDGRID is compiled in, AREPO does NOT simulate --
init.c calls add_backgroundgrid(), which snaps the gas to gravity-tree-node
centres (~1 cell per particle where particles exist), fills genuine voids with
coarse background cells, conservatively re-deposits the hydro/MHD fields, writes
the regridded IC as `<outdir>/snap_000.hdf5`, and exits.

Why this over the `amr` deposit mode: `amr` re-grids the WHOLE box onto a
structured oct-tree (base grid everywhere + 2:1 mass-refinement jumps), which
Voronoi-tessellates into sliver cells at every level boundary -- the "patchy"
mesh.  ADDBACKGROUNDGRID instead leaves the sampled region at the native particle
sampling and only oct-tree-fills the genuine voids, so the 2:1 transitions sit at
the void boundary (no data there) rather than through the resolved gas.  See
memory `arepo-mesh-backend` and the AREPO-grid design discussion.

B-field units: AREPO reads the IC MagneticField as GAUSS (io_fields.c: B /=
sqrt(4pi) on read, B *= sqrt(4pi) on write), so a Gauss-scaled IC round-trips to
a Gauss IC.  The caller must therefore hand bggrid an IC already in AREPO's Gauss
convention and NOT re-scale the output (the converter does exactly this).

This mirrors the RestartFlag-14 (`arepo_mesh.py`) shell-out pattern: a subprocess
so a mesher crash on pathological input cannot take down Python.
"""
import os
import shutil
import subprocess

import numpy as np

from .arepo_param import write_bggrid_param

_HERE = os.path.dirname(__file__)
# Default ADDBACKGROUNDGRID binary (built from Config_bggrid.sh; MHD so it regrids
# B for the MHD ICs and B=0 for hydro). runtest.sh overrides this with the
# aspect-matched Arepo_bggrid_asp<tag> for non-cubic boxes.
_BGGRID_EXE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "codes", "arepo", "Arepo_bggrid"))


def default_exe():
    return _BGGRID_EXE


def has_bggrid(exe=None):
    exe = exe or _BGGRID_EXE
    return os.path.isfile(exe) and os.access(exe, os.X_OK)


def regrid(particle_ic, output_ic, boxsize, gridsize, exe=None,
           maxmem=None, n_threads=1, keep=False):
    """Regrid a particle AREPO IC `particle_ic` (HDF5) into `output_ic` (HDF5) via
    an ADDBACKGROUNDGRID pass.

    boxsize   AREPO BoxSize (the X period; non-cubic y/z come from the binary's
              compile-time LONG_Y/LONG_Z, so `exe` must match the box aspect).
    gridsize  target finest background resolution per axis (<= 256).
    exe       Arepo_bggrid path (default: codes/arepo/Arepo_bggrid).
    maxmem    MaxMemSize MB (default scales with the IC particle count).

    Returns (output_ic, boxsize_out).  NOTE: ADDBACKGROUNDGRID enforces
    BoxSize >= 2 * gas_extent (FACTOR_MIN_BOX_SIZE) and <= 15 * gas_extent, so for
    a gas blob that fills most of `boxsize` it ENLARGES the box and pads the new
    volume with coarse background cells (its intended use: an isolated object in
    vacuum).  `boxsize_out` is the box AREPO actually wrote; the caller must use it
    for the downstream run param, not the input `boxsize`.  This is why arepo-grid
    is for isolated/collapse ICs, not periodic domain-filling boxes.

    Raises RuntimeError if the bggrid pass fails or writes no IC.  The output IC is
    in AREPO's Gauss B convention (see module docstring) -- do NOT re-scale it.
    """
    exe = exe or _BGGRID_EXE
    if not has_bggrid(exe):
        raise RuntimeError(
            "arepo-grid mode needs the ADDBACKGROUNDGRID binary at %s "
            "(build: AREPO_CONFIG=Config_bggrid.sh AREPO_EXEC=Arepo_bggrid "
            "AREPO_BUILD_DIR=build_bggrid ./build_arepo.sh)" % exe)
    particle_ic = os.path.abspath(particle_ic)
    output_ic = os.path.abspath(output_ic)
    if not os.path.isfile(particle_ic):
        raise RuntimeError("arepo-grid: particle IC not found: %s" % particle_ic)

    # MaxMemSize: bggrid holds the tree + the (up to gridsize^3) regridded cells.
    # Scale with the larger of the IC size and the target grid; generous headroom.
    if maxmem is None:
        try:
            import h5py
            with h5py.File(particle_ic, "r") as f:
                n_in = int(f["Header"].attrs["NumPart_Total"][0])
        except Exception:
            n_in = 0
        n_est = max(n_in, int(gridsize) ** 3)
        maxmem = max(8000, int(0.05 * n_est))

    import tempfile
    td = tempfile.mkdtemp(prefix="arepo_bggrid_")
    try:
        snap = _run_bggrid(particle_ic, boxsize, gridsize, exe, maxmem,
                           n_threads, td)
        boxsize_out = _read_boxsize(snap, fallback=float(boxsize))
        os.makedirs(os.path.dirname(output_ic) or ".", exist_ok=True)
        shutil.move(snap, output_ic)
    finally:
        if not keep:
            shutil.rmtree(td, ignore_errors=True)
    return output_ic, boxsize_out


def _run_bggrid(particle_ic, boxsize, gridsize, exe, maxmem, n_threads, td):
    """Run one ADDBACKGROUNDGRID pass inside work dir `td`; return the path to the
    regridded IC it wrote (`<base>-with-grid.hdf5`, next to InitCondFile -- NOT
    OutputDir/snap_000).  Raises RuntimeError on failure.  Caller owns `td`."""
    import glob
    ic_base = os.path.join(td, "ic")              # -> ic.hdf5 (ICFormat 3 = HDF5)
    shutil.copyfile(particle_ic, ic_base + ".hdf5")
    outdir = os.path.join(td, "output")
    os.makedirs(outdir, exist_ok=True)
    write_bggrid_param(os.path.join(td, "param.txt"), boxsize=boxsize,
                       gridsize=int(gridsize), maxmem=int(maxmem),
                       init_cond_file=ic_base, output_dir=outdir + os.sep)
    cmd = [exe, "param.txt", "0"]
    if n_threads and n_threads > 1:
        cmd = ["mpirun", "-np", str(int(n_threads))] + cmd
    r = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
    cand = glob.glob(ic_base + "-with-grid*")
    snap = next((c for c in cand if c.endswith(".hdf5")), None) \
        or (cand[0] if cand else None)
    if r.returncode != 0 or not snap or not os.path.isfile(snap):
        raise RuntimeError(
            "AREPO ADDBACKGROUNDGRID pass failed (rc=%d). Last output:\n%s"
            % (r.returncode, (r.stdout + r.stderr)[-2000:]))
    return snap


def regrid_generators(particle_ic, boxsize, gridsize, exe=None, maxmem=None,
                      n_threads=1):
    """Run an ADDBACKGROUNDGRID pass on `particle_ic` and return ONLY the mesh
    generator geometry: (gen (N,3) float64 in the [0, boxout] AREPO frame, boxout).

    This is the geometry half of the `arepo-grid` decoupled path: AREPO supplies
    the void-filled, regularized generator distribution (its one unique value),
    and we discard its fields entirely -- they are re-deposited with our exact
    petkova kernel in `deposit_on_generators`, sidestepping AREPO's broken MHD
    scatter (B_out ~ B * V_cell).  The frame is preserved (gas stays put, voids
    are added and the box grows), so the generators share origin 0 with a particle
    IC framed into [0, period]."""
    import h5py
    exe = exe or _BGGRID_EXE
    if not has_bggrid(exe):
        raise RuntimeError(
            "arepo-grid mode needs the ADDBACKGROUNDGRID binary at %s "
            "(build: AREPO_CONFIG=Config_bggrid.sh AREPO_EXEC=Arepo_bggrid "
            "AREPO_BUILD_DIR=build_bggrid ./build_arepo.sh)" % exe)
    if maxmem is None:
        try:
            with h5py.File(particle_ic, "r") as f:
                n_in = int(f["Header"].attrs["NumPart_Total"][0])
        except Exception:
            n_in = 0
        maxmem = max(8000, int(0.05 * max(n_in, int(gridsize) ** 3)))
    import tempfile
    td = tempfile.mkdtemp(prefix="arepo_bggrid_")
    try:
        snap = _run_bggrid(particle_ic, boxsize, gridsize, exe, maxmem,
                           n_threads, td)
        with h5py.File(snap, "r") as f:
            gen = np.ascontiguousarray(f["PartType0"]["Coordinates"][:],
                                       dtype=np.float64)
            boxout = float(f["Header"].attrs["BoxSize"])
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return gen, boxout


def bggrid_generators(particles, gridsize, boxsize=None, max_level=20,
                      periodic=False, include_occupied=True):
    """Build bggrid-equivalent AMR-octree mesh generators WITHOUT running AREPO:
    return (gen (M,3) in [0, box], box (3,)).

    Mesh: a Barnes-Hut oct-tree over the gas, one particle per leaf
    (`refine_count(1)`); GridSize == base_npx is the coarse floor, a leaf at `level`
    has size box/2**level. Generators are the leaf-cell centres.

    FRAMING depends on `periodic`:
      * periodic=False (isolated object) -- reproduce AREPO ADDBACKGROUNDGRID: the
        box is ENLARGED to >= 2*gas_extent (FACTOR_MIN_BOX_SIZE, add_bggrid.c:427),
        cubic, and the gas is framed positive (AREPO applies no half-box shift,
        :422,480). The octree void-fills the surrounding vacuum. Verified ~99.5%
        cell-for-cell vs real bggrid on a 20k blob.
      * periodic=True -- NO enlargement: the gas already fills the periodic box, so
        the octree tiles the box as-is (possibly non-cubic; base_npx per axis is
        scaled to keep cells ~cubic). The deposit is done with a periodic VoronoiGrid.

    `include_occupied`:
      * True (default) -- ALL leaves, occupied + empty: a generator at every
        particle's cell -> quasi-Lagrangian, the densest core is resolved. The
        better mesh for the exact petkova deposit (still mass-conserving: the leaves
        partition space).
      * False -- only the EMPTY (childless) leaves, reproducing AREPO bggrid exactly
        (it emits the empty oct-tree NODES, add_bggrid.c:102 sibling==nextnode,
        scatters the gas onto them and DELETES the particles). Occupied leaves are
        dropped only as an artifact of AREPO's scatter algorithm, not for correctness.
    """
    from sph_interp import Particles
    from sph_interp.targets import AMRGrid, refine_count
    from sph_interp.export import _frame_coords
    coords, _L = _frame_coords(np.ascontiguousarray(particles.pos, np.float64),
                               box=particles.box, periodic=periodic)
    if periodic:
        # gas fills the box -> tile it as-is, no void enlargement.
        if particles.box is not None:
            box = np.asarray(particles.box, np.float64).reshape(3)
        else:
            box = coords.max(axis=0) - coords.min(axis=0)
        ref = float(box.max())          # GridSize cells span the longest axis
        nb = tuple(max(1, int(round(gridsize * box[a] / ref))) for a in range(3))
    else:
        # isolated object -> enlarge to >= 2*gas_extent (cubic), bggrid-style.
        len_gas = float((coords.max(axis=0) - coords.min(axis=0)).max())
        L = max(float(boxsize) if boxsize else 0.0, 2.0 * len_gas)
        box = np.array([L, L, L], np.float64)
        nb = (int(gridsize), int(gridsize), int(gridsize))
    pf = Particles(pos=coords, mass=particles.mass, rho=particles.rho,
                   h=particles.h, box=box, periodic=periodic)
    amr = AMRGrid.build(pf, base_npx=nb, max_level=int(max_level),
                        criterion=refine_count(1), lo=[0.0, 0.0, 0.0],
                        hi=box, periodic=periodic, empty_max_level=None)
    cen = amr.centres()
    if not include_occupied:
        cen = cen[np.asarray(amr.cell_count) == 0]
    return np.ascontiguousarray(cen), box


def _resolve_tufac(tipsy_input):
    """dTuFac for the AREPO IC: from the gasoline .param units, else the run .log
    (matches IC_interpolate_to_grid's resolution)."""
    from sph_interp.particles import read_param_tufac
    tf = read_param_tufac(tipsy_input)
    if tf:
        return tf
    try:
        from IC_analysis_general import read_tufac
        return read_tufac(tipsy_input)
    except Exception:
        return 1.0


def regrid_amr_petkova(tipsy_input, output_ic, gridsize, fields, tu_fac=None,
                       periodic=None, nsmooth=64, method="petkova",
                       include_occupied=True):
    """Pure-Python `arepo-grid`: build bggrid-equivalent AMR-octree generators (no
    AREPO binary -- see `bggrid_generators`) and deposit the ORIGINAL SPH fields
    onto their Voronoi mesh with the exact petkova kernel.  Returns
    (output_ic, boxsize_x).

    This replaces AREPO's own ADDBACKGROUNDGRID field scatter (B_out ~ B*V_cell,
    wrong; see memory `mhd-energy-conservation-cleaning`) while keeping bggrid's
    mesh quality.  B is written in CODE units -> the caller must scale it to AREPO's
    Gauss convention afterwards (scale_hdf5_bfield).

    `periodic`: None (default) reads the gasoline `bPeriodic` flag from the .param.
      * periodic -> the octree tiles the actual (possibly non-cubic) box, deposit on
        a periodic VoronoiGrid; the AREPO BoxSize is the X period (y/z via LONG_*).
      * non-periodic -> bggrid isolated framing (box enlarged to >= 2*gas_extent,
        cubic, void-filled); for collapse/blob ICs."""
    from sph_interp import from_tipsy, interpolate, VoronoiGrid, to_arepo
    from sph_interp.export import _frame_coords, AREPO_TEMPERATURE
    from sph_interp.particles import read_param_periodic, _read_param_box
    fields = tuple(fields)
    # AREPO needs internal energy u = dTuFac*T -> ensure a temperature field is
    # deposited (mirrors IC_interpolate_to_grid's arepo path).
    if not any(f.lower() in AREPO_TEMPERATURE for f in fields):
        fields = fields + ("temp",)
    if tu_fac is None:
        tu_fac = _resolve_tufac(tipsy_input)
    if periodic is None:
        periodic = bool(read_param_periodic(tipsy_input))   # bPeriodic; default isolated
    # authoritative period from the .param (dx/dy/dzPeriod); falls back to .log /
    # particle extent inside from_tipsy when absent.
    box = _read_param_box(tipsy_input)
    p = from_tipsy(tipsy_input, fields=fields, nsmooth=nsmooth, periodic=periodic,
                   box=box)
    # generators: bggrid-equivalent octree (frames p.pos internally into [0, box]).
    gen, box = bggrid_generators(p, gridsize, periodic=periodic,
                                 include_occupied=include_occupied)
    print(f"  bggrid-octree generators: {gen.shape[0]} cells, "
          f"box={np.array2string(box, precision=5)}, "
          f"{'periodic' if periodic else 'isolated (>=2x gas extent)'}, "
          f"gridsize={gridsize}, include_occupied={include_occupied}")
    # frame the SAME particles into gen's frame (identical _frame_coords call).
    p.pos, _L = _frame_coords(p.pos, box=p.box, periodic=periodic)
    if periodic:
        bounds = [[0.0, float(box[a])] for a in range(3)]
        target = VoronoiGrid.from_points(gen, bounds=bounds, periodic=True)
        res = interpolate(p, target, method=method, values=list(fields))
        to_arepo(res, output_ic, tu_fac=tu_fac, box=box, periodic=True)
    else:
        L = float(box[0])
        target = VoronoiGrid.from_points(gen, bounds=[[0.0, L]] * 3, periodic=False)
        res = interpolate(p, target, method=method, values=list(fields))
        to_arepo(res, output_ic, tu_fac=tu_fac, preframed_boxsize=L)
    return output_ic, float(box[0])


def deposit_on_generators(tipsy_input, gen, boxout, output_ic, fields, tu_fac,
                          nsmooth=64, period=None, periodic=False,
                          method="petkova"):
    """Deposit the ORIGINAL SPH fields onto bggrid generators `gen` (in [0, boxout])
    with our exact petkova kernel, and write the final AREPO IC.

    The SPH particles are read from `tipsy_input` and framed into [0, period] -- the
    SAME frame the particle IC fed to bggrid was in -- so they share origin 0 with
    `gen` (whose gas region is unshifted; voids fill out to boxout).  The deposit is
    non-periodic (the gas is an isolated blob in the enlarged box; void generators
    get no SPH support -> zero mass).  Conserved mass = rho*V_cell; intensive v/u/B
    are cell-centred.  B is therefore correct (unlike a direct bggrid output)."""
    from sph_interp import from_tipsy, interpolate, VoronoiGrid, to_arepo
    from sph_interp.export import _frame_coords
    p = from_tipsy(tipsy_input, fields=tuple(fields), nsmooth=nsmooth,
                   periodic=periodic, box=(np.repeat(period, 3)
                                           if period is not None else None))
    # Frame the SPH particles exactly as the particle IC (and hence bggrid) framed
    # them: gasoline [-L/2, L/2] -> [0, period]; gas then aligns with `gen`.
    p.pos, _L = _frame_coords(p.pos, box=p.box, periodic=periodic)
    target = VoronoiGrid.from_points(np.ascontiguousarray(gen, np.float64),
                                     bounds=[[0.0, boxout]] * 3, periodic=False)
    res = interpolate(p, target, method=method, values=list(fields))
    # Generators already in [0, boxout] -> skip re-framing (preframed_boxsize).
    return to_arepo(res, output_ic, tu_fac=tu_fac, preframed_boxsize=boxout)


def _read_boxsize(ic_path, fallback=None):
    try:
        import h5py
        with h5py.File(ic_path, "r") as f:
            return float(f["Header"].attrs["BoxSize"])
    except Exception:
        return fallback
