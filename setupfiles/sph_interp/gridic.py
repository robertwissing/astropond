"""Code-agnostic grid initial conditions (GridIC) for finite-volume codes.

The grid analog of the particle-copy path: deposit the SPH field onto a fine
uniform grid with our interpolation (petkova, mass-conserving), and write ONE
portable "GridIC" file that every grid code's in-code IC reader consumes
(RAMSES `condinit`, Athena++ problem generator, ...). The file is always a fine
uniform field; "uniform vs AMR" is a per-code run decision (the reader averages
fine cells for coarser cells).

Canonical GridIC stream-binary layout (little-endian, C order; Fortran reads it
with access='stream', C++ with fread -- no HDF5 dependency):

    magic   int32  = 0x47524943  ('GRIC')
    version int32  = 1
    nx,ny,nz int32
    nvar    int32  = 5            (rho, vx, vy, vz, P)
    mhd     int32  = 0|1
    boxlen  f64
    gamma   f64
    time    f64
    box[3]  f64
    rho, vx, vy, vz, P            each nx*ny*nz f64, C order (ix slowest)
    [Bx_l,Bx_r,By_l,By_r,Bz_l,Bz_r]   (mhd only) each nx*ny*nz f64

The six magnetic components are the div-free FACE-centred fields from
`project_solenoidal` (constrained-transport representation; divB ~ 1e-14).
"""
import struct

import numpy as np

from .particles import from_tipsy, read_param_tufac
from .targets import UniformGrid
from .interpolate import interpolate
from .solenoidal import project_solenoidal

MAGIC = 0x47524943          # 'GRIC'
VERSION = 1


def _with_v2(p):
    """`p` plus a `_v2` value field holding |v|^2, without mutating the input.

    The caller may have handed us their own Particles; adding a field in place
    would be a surprising side effect, and `_v2` would then turn up in any
    later `values=None` deposit they do.
    """
    from dataclasses import replace
    missing = [k for k in ("vx", "vy", "vz") if k not in p.values]
    if missing:
        raise ValueError(
            f"the conservative GridIC deposit needs the velocity components "
            f"as value fields; missing {missing}. Read the snapshot with "
            f"fields=('rho','vx','vy','vz','T'), or pass conservative=False.")
    v2 = sum(np.asarray(p.values[k], dtype=np.float64) ** 2
             for k in ("vx", "vy", "vz"))
    return replace(p, values={**p.values, "_v2": v2})


def _report_budget(p, M, vx, vy, vz, u, tu_fac, e_unresolved):
    """Print how exactly the deposit conserved mass, momentum and energy.

    Printed rather than asserted: on a genuinely under-resolved grid a real
    deficit is information, not a failure.
    """
    m = p.mass
    vel = np.stack([p.values[k] for k in ("vx", "vy", "vz")], axis=1)
    tot_m = float(m.sum())
    tot_p = (m[:, None] * vel).sum(axis=0)
    tot_therm = float((m * p.values["T"]).sum() * tu_fac)
    tot_kin = float(0.5 * (m * np.sum(vel ** 2, axis=1)).sum())

    got_m = float(M.sum())
    got_p = np.array([float((M * c).sum()) for c in (vx, vy, vz)])
    got_kin = float(0.5 * (M * (vx * vx + vy * vy + vz * vz)).sum())
    got_int = float((M * u).sum())

    def rel(a, b):
        return (a - b) / b if b else a

    pscale = float(np.abs(tot_p).max()) or 1.0
    print(f"  GridIC conservative deposit: "
          f"mass {rel(got_m, tot_m):+.2e}, "
          f"momentum {np.max(np.abs(got_p - tot_p)) / pscale:+.2e}, "
          f"E_tot {rel(got_kin + got_int, tot_kin + tot_therm):+.2e} "
          f"(E_kin {rel(got_kin, tot_kin):+.2%} -> thermalised "
          f"{float(e_unresolved.sum()) / (tot_kin or 1.0):.2%} of it)")


def write_grid_ic(source, path, npx, method="petkova", mhd=False, box=None,
                  gamma=5.0 / 3.0, tu_fac=None, time=0.0,
                  fields_b=("Bx", "By", "Bz"), conservative=True):
    """Deposit an SPH IC onto a fine uniform grid and write a GridIC file.

    source : a tipsy snapshot path, or a `Particles` instance.
    path   : output GridIC file.
    npx    : cells per axis (int -> cubic; for RAMSES use 2**levelmax). A (3,)
             tuple is allowed but the codes' base grids are cubic.
    method : 'petkova' (exact, mass-conserving; default) or 'sph' (fast).
    mhd    : also deposit B and write the 6 div-free face components.
    box    : periodic period (3,); default = the particles' box.
    gamma  : adiabatic index (P = (gamma-1) * rho * u).
    tu_fac : dTuFac if the particle thermal field is a temperature (u = tu_fac*T).
    conservative : petkova only. Build the cell state from CONSERVED CONTENTS
             (momentum and energy) rather than from kernel-volume-weighted
             averages, and close the total-energy budget. See below.

    THE CONSERVATIVE PATH (default; `conservative=False` gives the
    volume-weighted state). Petkova's density conserves mass, but
    kernel-volume-weighted velocity and temperature only approximate MOMENTUM
    and ENERGY. Using the `_content_<name>` channels instead:

        M_cell   = sum_j m_j I_j                    (`_mass`)
        v_cell   = sum_j m_j v_j I_j / M_cell       (`_content_v*` / M)
        u_therm  = sum_j m_j u_j I_j / M_cell       (`_content_T` * tu_fac / M)

    which conserves total mass, momentum and thermal energy exactly.

    ENERGY CLOSURE. Mass-weighting the velocity is not energy-neutral: a cell
    holding particles with different velocities has kinetic energy
    `1/2 sum_j m_j |v_j|^2`, while the bulk motion it is collapsed to carries
    only `1/2 M |v_cell|^2`. The difference is the UNRESOLVED VELOCITY
    VARIANCE, non-negative by Jensen. A finite-volume cell has nowhere to put
    sub-cell motion except its internal energy, so it is added there:

        u_cell = u_therm + [1/2 sum_j m_j |v_j|^2 - 1/2 M |v_cell|^2] / M_cell

    making TOTAL energy exact rather than thermal energy exact. That is a
    physical choice, not a bookkeeping one: the kinetic energy of motion the
    grid cannot represent is thermalised, which is what the grid code would do
    to it within a step anyway. It requires depositing |v|^2 as its own field,
    since `sum m_j |v_j|^2` cannot be recovered from `sum m_j v_j`.

    B is deliberately NOT mass-weighted - the meaningful cell average of a
    field threading the cell is volume-weighted - so in MHD mode it is
    deposited in a second pass with the original weighting.
    """
    if isinstance(source, str):
        flds = ["rho", "vx", "vy", "vz", "T"]
        if mhd:
            flds += list(fields_b)
        p = from_tipsy(source, fields=tuple(flds), periodic=True, box=box)
        if tu_fac is None:
            tu_fac = read_param_tufac(source)   # dTuFac from the .param units
    else:
        p = source
    if tu_fac is None:
        tu_fac = 1.0
    if box is None:
        box = p.box
    box = np.asarray(box, dtype=np.float64).reshape(3)

    grid = UniformGrid.centered(box, npx)
    nx, ny, nz = grid.npx

    # Deposit the tipsy TEMPERATURE (gasoline stores the thermal state as T) and
    # form the specific internal energy u = dTuFac * T (tu_fac); P = (gamma-1)*rho*u.
    # This mirrors the AREPO/GIZMO converters (from_tipsy's derived 'u' is 0 when
    # dTuFac can't be resolved from a .log, so deposit T and scale here).
    use_cons = bool(conservative) and method == "petkova"

    values = ["rho", "vx", "vy", "vz", "T"]
    if use_cons:
        # |v|^2 must be deposited in its own right: sum_j m_j |v_j|^2 is not
        # recoverable from sum_j m_j v_j, and it is what the energy closure needs
        p = _with_v2(p)
        values = values + ["_v2"]
    if mhd and not use_cons:
        values += list(fields_b)
    res = interpolate(p, grid, method=method, values=values,
                      **({"conservative_fields": True} if use_cons else {}))
    d = res.data

    def f(name):
        return np.ascontiguousarray(np.asarray(d[name], dtype=np.float64))

    if use_cons:
        M = f("_mass")
        ok = M > 0.0
        inv = np.zeros_like(M)
        inv[ok] = 1.0 / M[ok]
        rho = M / grid.cell_volume
        vx = f("_content_vx") * inv
        vy = f("_content_vy") * inv
        vz = f("_content_vz") * inv
        # the closure: particle kinetic energy in the cell, minus the bulk
        # kinetic energy the cell can actually represent -> heat
        e_kin_particles = 0.5 * f("_content__v2")
        e_kin_bulk = 0.5 * M * (vx * vx + vy * vy + vz * vz)
        e_unresolved = np.maximum(e_kin_particles - e_kin_bulk, 0.0)
        u = (f("_content_T") * tu_fac + e_unresolved) * inv
        _report_budget(p, M, vx, vy, vz, u, tu_fac, e_unresolved)
    else:
        rho = f("rho")
        vx, vy, vz = f("vx"), f("vy"), f("vz")
        u = f("T") * tu_fac
    P = (gamma - 1.0) * rho * u

    faces = None
    if mhd:
        if use_cons:
            # second pass: B stays kernel-volume-weighted (see the docstring)
            res = interpolate(p, grid, method=method, values=list(fields_b))
        faces = project_solenoidal(res, fields=fields_b, periodic=True,
                                   add_to_result=False)

    boxlen = float(box[0])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<6i", MAGIC, VERSION, nx, ny, nz, 5))
        fh.write(struct.pack("<i", 1 if mhd else 0))
        fh.write(struct.pack("<6d", boxlen, float(gamma), float(time),
                             float(box[0]), float(box[1]), float(box[2])))
        for arr in (rho, vx, vy, vz, P):
            fh.write(np.ascontiguousarray(arr, dtype="<f8").tobytes(order="C"))
        if mhd:
            for k in ("Bx_l", "Bx_r", "By_l", "By_r", "Bz_l", "Bz_r"):
                fh.write(np.ascontiguousarray(faces[k], dtype="<f8").tobytes(order="C"))
    print(f"  GridIC: {nx}x{ny}x{nz} {method} deposit, boxlen={boxlen:g}, "
          f"total mass={rho.sum() * grid.cell_volume:.6g}, "
          f"{'MHD (face B, divB~%.1e)' % float(np.max(np.abs(faces['divB']))) if mhd else 'hydro'}")
    return path


def read_grid_ic(path):
    """Read a GridIC file back into a dict (for validation / cells_to_tipsy)."""
    with open(path, "rb") as fh:
        magic, version, nx, ny, nz, nvar = struct.unpack("<6i", fh.read(24))
        if magic != MAGIC:
            raise ValueError(f"not a GridIC file (magic {magic:#x})")
        (mhd,) = struct.unpack("<i", fh.read(4))
        boxlen, gamma, time, bx, by, bz = struct.unpack("<6d", fh.read(48))
        n = nx * ny * nz
        out = {"nx": nx, "ny": ny, "nz": nz, "mhd": bool(mhd), "boxlen": boxlen,
               "gamma": gamma, "time": time, "box": (bx, by, bz)}
        for name in ("rho", "vx", "vy", "vz", "P"):
            out[name] = np.frombuffer(fh.read(8 * n), dtype="<f8").reshape(nx, ny, nz)
        if mhd:
            for k in ("Bx_l", "Bx_r", "By_l", "By_r", "Bz_l", "Bz_r"):
                out[k] = np.frombuffer(fh.read(8 * n), dtype="<f8").reshape(nx, ny, nz)
    return out


def cells_to_tipsy(output, out_idx, x, y, z, dx, rho, vx, vy, vz, u,
                   bx=None, by=None, bz=None, time=0.0):
    """Write one tipsy gas snapshot from a flat list of grid cells.

    Reverse helper shared by grid codes whose output we read into a cell list
    (e.g. Athena++ .athdf; the RAMSES fallback path). Each cell -> one tipsy gas
    particle: mass = rho * dx**3, pos = centre, v, rho, T = u (caller pre-divides
    by dTuFac if needed), h = 0.5*dx; B (cell-centred) -> BFieldx/y/z aux.
    """
    import os
    import readtipsy as tip
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    dx = np.asarray(dx, float); n = x.size
    vol = dx ** 3
    tg = np.zeros((n, 12))
    tg[:, 0] = np.asarray(rho, float) * vol
    tg[:, 1] = x; tg[:, 2] = y; tg[:, 3] = z
    tg[:, 4] = vx; tg[:, 5] = vy; tg[:, 6] = vz
    tg[:, 7] = rho
    tg[:, 8] = u
    tg[:, 9] = 0.5 * dx
    os.makedirs(output, exist_ok=True)
    outname = os.path.join(output, f"{output}.{out_idx:05d}")
    tip.writetipsy(tg, [], [], outname, np.array([n, 3, n, 0, 0, 0]),
                   np.array([float(time)]))
    tip.writetipsyaux(np.ascontiguousarray(0.5 * dx), "smoothlength", outname)
    if bx is not None:
        tip.writetipsyaux(np.ascontiguousarray(bx, float), "BFieldx", outname)
        tip.writetipsyaux(np.ascontiguousarray(by, float), "BFieldy", outname)
        tip.writetipsyaux(np.ascontiguousarray(bz, float), "BFieldz", outname)
    return outname
