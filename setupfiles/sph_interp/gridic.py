"""Code-agnostic grid initial conditions (GridIC) for finite-volume codes.

The grid analog of the particle-copy path: deposit the SPH field onto a fine
uniform grid with our interpolation (petkova, mass-conserving), and write ONE
portable "GridIC" file that every grid code's in-code IC reader consumes
(RAMSES `condinit`, Athena++ problem generator, ...). The file is always a fine
uniform field; "uniform vs AMR" is a per-code run decision (the reader averages
fine cells for coarser cells). See codes/GRID_CODES_PLAN.md.

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


def write_grid_ic(source, path, npx, method="petkova", mhd=False, box=None,
                  gamma=5.0 / 3.0, tu_fac=None, time=0.0, fields_b=("Bx", "By", "Bz")):
    """Deposit an SPH IC onto a fine uniform grid and write a GridIC file.

    source : a tipsy snapshot path, or a `Particles` instance.
    path   : output GridIC file.
    npx    : cells per axis (int -> cubic; for RAMSES use 2**levelmax). A (3,)
             tuple is allowed but the codes' base grids are cubic (first cut).
    method : 'petkova' (exact, mass-conserving; default) or 'sph' (fast).
    mhd    : also deposit B and write the 6 div-free face components.
    box    : periodic period (3,); default = the particles' box.
    gamma  : adiabatic index (P = (gamma-1) * rho * u).
    tu_fac : dTuFac if the particle thermal field is a temperature (u = tu_fac*T).
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
    values = ["rho", "vx", "vy", "vz", "T"]
    if mhd:
        values += list(fields_b)
    res = interpolate(p, grid, method=method, values=values)
    d = res.data

    def f(name):
        return np.ascontiguousarray(np.asarray(d[name], dtype=np.float64))

    rho = f("rho")
    vx, vy, vz = f("vx"), f("vy"), f("vz")
    u = f("T") * tu_fac
    P = (gamma - 1.0) * rho * u

    faces = None
    if mhd:
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
