#!/usr/bin/env python
"""Convert RAMSES output back into gasoline/tipsy snapshots (code=4).

We convert the AMR output DIRECTLY to particles (one per leaf cell) rather than
resampling to a uniform cube: RAMSES ships `amr2cell`, which writes one ASCII row
per leaf cell -- `x y z dx cpu level var1..varN` -- with ALL hydro variables
(rho, vx, vy, vz, P, and B for MHD) in a single pass. This preserves the AMR
structure (a coarse cell -> a big particle, a fine cell -> a small one) and carries
B, unlike `amr2tipsy` (no aux) or `amr2cube` (resamples to uniform, one var at a
time). Each cell -> one tipsy gas particle via `sph_interp.gridic.cells_to_tipsy`,
which also writes the BFieldx/y/z + smoothlength aux the analysis reads.

    ICconvert_ramsestotipsy.py <rundir> <output> <idx_interval> <paramfile>
                               [--amr2cell PATH] [--lmax L] [--mhd]

  rundir        RAMSES run dir containing output_NNNNN/.
  output        run name / output dir for the tipsy snapshots.
  idx_interval  snapshot stride (= iOutInterval): output_K -> <output>.{K*stride:05d}.
  paramfile     original gasoline .param (boxlen + dTuFac).

amr2cell columns: x,y,z (cell centre, code units [0,1]), dx, icpu, ilevel, then the
RAMSES hydro vars IN FILE ORDER (hydro/output_hydro.f90, primitive mode; shared by
both solvers). The layout is solver-dependent and auto-detected from the column
count: hydro (5 vars) rho,vx,vy,vz,P; MHD (11 vars, NVAR=8) rho,vx,vy,vz,
B_left_xyz,B_right_xyz,P — the tipsy BField aux gets the cell-centred
B=(B_left+B_right)/2. --mhd forces the MHD layout (normally not needed).
"""
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp.gridic import cells_to_tipsy
from sph_interp.particles import read_param_tufac
from ICconvert_arepototipsy import write_log, _periods_from_param

# column offsets into the per-cell hydro vars (after x,y,z,dx,icpu,ilevel),
# from hydro/output_hydro.f90's primitive write order (shared by both solvers)
HYDRO = {"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4}
MHD = {"rho": 0, "vx": 1, "vy": 2, "vz": 3,
       "Bxl": 4, "Byl": 5, "Bzl": 6, "Bxr": 7, "Byr": 8, "Bzr": 9, "P": 10}
NVAR_MHD = 11                            # NVAR=8 build: 8 + 3 B_right columns


def _nvar_from_descriptor(indir):
    """Number of hydro vars from the output's hydro_file_descriptor.txt."""
    desc = os.path.join(indir, "hydro_file_descriptor.txt")
    if not os.path.exists(desc):
        return None
    return sum(1 for line in open(desc) if not line.lstrip().startswith("#"))


def _run_amr2cell(amr2cell, indir, outpath, lmax):
    cmd = [amr2cell, "-inp", indir, "-out", outpath]
    if lmax:
        cmd += ["-lma", str(lmax)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # x y z dx icpu ilevel var1..varN  (ASCII). Two Fortran-isms to repair:
    # (1) 3-digit exponents drop the 'E' ("-1.44830-120" = -1.4483e-120,
    #     denormal noise in the velocity columns) which float() cannot parse;
    # (2) amr2cell's fixed 999 format holds at most 10 var fields, so an MHD
    #     row (11 vars, 17 values) WRAPS onto a second line (16 + 1 tokens) —
    #     np.loadtxt chokes. Parse the token stream flat and reshape with the
    #     row width from hydro_file_descriptor.txt instead.
    import re
    with open(outpath) as fh:
        txt = re.sub(r"(\d)([+-]\d{3}\b)", r"\1E\2", fh.read())
    nvar = _nvar_from_descriptor(indir)
    flat = np.array(txt.split(), dtype=np.float64)
    if nvar is None:                     # no descriptor: assume unwrapped rows
        nlines = txt.count("\n") or 1
        ncols = flat.size // nlines
    else:
        ncols = 6 + nvar
    if flat.size % ncols:
        raise ValueError(f"amr2cell output {outpath}: {flat.size} values do "
                         f"not fill rows of {ncols}")
    return flat.reshape(-1, ncols)


def _info_time(indir):
    """Physical time of a RAMSES output from its info_NNNNN.txt (0.0 if absent)."""
    num = os.path.basename(indir.rstrip("/")).split("_")[-1]
    info = os.path.join(indir, f"info_{num}.txt")
    if os.path.exists(info):
        for line in open(info):
            if line.split("=")[0].strip() == "time":
                return float(line.split("=")[1])
    return 0.0


def convert_output(amr2cell, indir, output, out_idx, boxlen, gamma, tu_fac,
                   lmax, mhd, tmpdir):
    rows = _run_amr2cell(amr2cell, indir,
                         os.path.join(tmpdir, f"cells_{out_idx:05d}.txt"), lmax)
    # geometry: code units [0,1] -> gasoline-centred box [-boxlen/2, boxlen/2]
    x = (rows[:, 0] - 0.5) * boxlen
    y = (rows[:, 1] - 0.5) * boxlen
    z = (rows[:, 2] - 0.5) * boxlen
    dx = rows[:, 3] * boxlen
    var = rows[:, 6:]                       # the nvarh hydro variables
    mhd = mhd or var.shape[1] >= NVAR_MHD   # auto-detect the MHD layout
    vmap = MHD if mhd else HYDRO
    rho = var[:, vmap["rho"]]
    vx, vy, vz = var[:, vmap["vx"]], var[:, vmap["vy"]], var[:, vmap["vz"]]
    P = var[:, vmap["P"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(rho > 0, P / ((gamma - 1.0) * rho), 0.0)
    T = u / tu_fac
    bx = by = bz = None
    if mhd:
        bx = 0.5 * (var[:, vmap["Bxl"]] + var[:, vmap["Bxr"]])
        by = 0.5 * (var[:, vmap["Byl"]] + var[:, vmap["Byr"]])
        bz = 0.5 * (var[:, vmap["Bzl"]] + var[:, vmap["Bzr"]])
    t = _info_time(indir)
    cells_to_tipsy(output, out_idx, x, y, z, dx, rho, vx, vy, vz, T,
                   bx=bx, by=by, bz=bz, time=t)
    print(f"  output_{out_idx:05d}: {rows.shape[0]} leaf cells -> tipsy "
          f"(t={t:.6g}, {'MHD' if mhd else 'hydro'})")


def convertfile(rundir, output, idx_interval, paramfile, amr2cell=None,
                lmax=0, mhd=False):
    os.makedirs(output, exist_ok=True)
    outs = sorted(glob.glob(os.path.join(rundir, "output_[0-9]*")))
    if not outs:
        print(f"No output_NNNNN under '{rundir}'", file=sys.stderr)
        sys.exit(1)
    if amr2cell is None:
        amr2cell = os.environ.get("AMR2CELL", "amr2cell")
    gas = {}
    if paramfile and os.path.exists(paramfile):
        for line in open(paramfile):
            line = line.split("#")[0]
            if "=" in line:
                k, _, v = line.partition("=")
                gas[k.strip()] = v.strip()
    gamma = float(gas.get("dConstGamma", 5.0 / 3.0))
    boxlen = float(gas.get("dxPeriod", 1.0))
    # read_param_tufac takes a SNAPSHOT stem (it appends .param itself);
    # None (units missing) -> 1.0, matching read_tufac's default.
    tu_fac = None
    if paramfile:
        stem = paramfile[:-6] if paramfile.endswith(".param") else paramfile
        tu_fac = read_param_tufac(stem)
    if tu_fac is None:
        tu_fac = 1.0
    print(f"  dTuFac (u = dTuFac*T) = {tu_fac:g}")
    periods = _periods_from_param(paramfile) or (boxlen, boxlen, boxlen)
    # Stamp dTuFac so the analysis's read_tufac() recovers u = dTuFac*T from the
    # snapshot temperature column (T = u/tu_fac below); a header-only .log would
    # otherwise default it to 1.0 and undercount E_therm by dTuFac.
    write_log(os.path.join(output, output + ".log"), periods, energy_txt=None,
              tu_fac=tu_fac)

    tmpdir = os.path.join(output, "_amr2cell_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    print(f"Found {len(outs)} RAMSES output(s); amr2cell={amr2cell}")
    for k, indir in enumerate(outs):
        convert_output(amr2cell, indir, output, k * idx_interval, boxlen, gamma,
                       tu_fac, lmax, mhd, tmpdir)


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(prog="ICconvert_ramsestotipsy")
    pa.add_argument("rundir")
    pa.add_argument("output")
    pa.add_argument("idx_interval", type=int)
    pa.add_argument("paramfile", nargs="?", default=None)
    pa.add_argument("--amr2cell", default=None)
    pa.add_argument("--lmax", type=int, default=0)
    pa.add_argument("--mhd", action="store_true")
    a = pa.parse_args()
    convertfile(a.rundir, a.output, a.idx_interval, a.paramfile,
                amr2cell=a.amr2cell, lmax=a.lmax, mhd=a.mhd)
