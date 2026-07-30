#!/usr/bin/env python
"""Convert Athena++ .athdf output back into gasoline/tipsy snapshots (code=7).

Reads each gridic.out2.NNNNN.athdf (primitive variables) with Athena++'s own
vis/python/athena_read.py and writes one tipsy gas particle per cell via
`sph_interp.gridic.cells_to_tipsy` (rho, v, T = u/dTuFac, h = dx/2; MHD adds
the cell-centred Bcc -> BFieldx/y/z aux), plus the gasoline-style .log with
the dTuFac stamp the analyses need.

    ICconvert_athenatotipsy.py <rundir> <output> <idx_interval> <paramfile>
                               [--athena-vis PATH]

  rundir        Athena++ run dir containing gridic.out2.*.athdf.
  output        run name / output dir for the tipsy snapshots.
  idx_interval  snapshot stride (= iOutInterval): out2.K -> <output>.{K*stride:05d}.
  paramfile     original gasoline .param (boxlen + gamma + dTuFac units).

Uniform mesh only (first cut — matches the from_gridic pgen; a future
SMR/AMR mode would pass level=None and per-cell dx). Athena++ numbers
out2.00000 at t=0, so the k=0 label mapping matches RAMSES/GIZMO.
"""
import glob
import os
import sys

import numpy as np

from ICconvert_common import read_gasoline_param

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp.gridic import cells_to_tipsy
from sph_interp.particles import read_param_tufac
from ICconvert_arepototipsy import write_log, _periods_from_param

_DEF_VIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "codes", "athena++", "athena", "vis", "python")


def convert_output(athdf, fname, output, out_idx, gamma, tu_fac):
    d = athdf(fname)
    t = float(d["Time"])
    # cell-centred coordinates and per-axis uniform spacing (slab boxes have
    # slightly anisotropic cells); cells_to_tipsy takes ONE dx per cell, so
    # pass the volume-equivalent edge (mass = rho*dx_eff^3 stays exact,
    # h = dx_eff/2 is the isotropic smoothing proxy)
    x1v, x2v, x3v = d["x1v"], d["x2v"], d["x3v"]

    def spacing(xv, rg):
        # rg = RootGridX_i (min, max, ratio); single-cell axis -> full extent
        return float(xv[1] - xv[0]) if xv.size > 1 else float(rg[1] - rg[0])

    dx1 = spacing(x1v, d["RootGridX1"])
    dx2 = spacing(x2v, d["RootGridX2"])
    dx3 = spacing(x3v, d["RootGridX3"])
    dx = (dx1 * dx2 * dx3) ** (1.0 / 3.0)
    # athdf arrays are (nx3, nx2, nx1) — index [k, j, i]
    Z, Y, X = np.meshgrid(x3v, x2v, x1v, indexing="ij")
    rho = np.asarray(d["rho"], dtype=np.float64)
    P = np.asarray(d["press"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(rho > 0, P / ((gamma - 1.0) * rho), 0.0)
    flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()
    kw = {}
    mhd = "Bcc1" in d
    if mhd:
        kw = {"bx": flat(d["Bcc1"]), "by": flat(d["Bcc2"]), "bz": flat(d["Bcc3"])}
    cells_to_tipsy(output, out_idx, flat(X), flat(Y), flat(Z),
                   np.full(rho.size, dx), flat(rho),
                   flat(d["vel1"]), flat(d["vel2"]), flat(d["vel3"]),
                   flat(u) / tu_fac, time=t, **kw)
    print(f"  out2.{out_idx:05d}: {rho.size} cells -> tipsy "
          f"(t={t:.6g}, {'MHD' if mhd else 'hydro'})")


def convertfile(rundir, output, idx_interval, paramfile, athena_vis=None):
    sys.path.insert(0, os.path.abspath(athena_vis or _DEF_VIS))
    from athena_read import athdf

    outs = sorted(glob.glob(os.path.join(rundir, "*.out2.[0-9]*.athdf")))
    if not outs:
        print(f"No *.out2.NNNNN.athdf under '{rundir}'", file=sys.stderr)
        sys.exit(1)
    os.makedirs(output, exist_ok=True)
    gas = read_gasoline_param(paramfile)
    gamma = float(gas.get("dConstGamma", 5.0 / 3.0))
    boxlen = float(gas.get("dxPeriod", 1.0))
    tu_fac = None
    if paramfile:
        stem = paramfile[:-6] if paramfile.endswith(".param") else paramfile
        tu_fac = read_param_tufac(stem)
    if tu_fac is None:
        tu_fac = 1.0
    print(f"  dTuFac (u = dTuFac*T) = {tu_fac:g}")
    periods = _periods_from_param(paramfile) or (boxlen, boxlen, boxlen)
    write_log(os.path.join(output, output + ".log"), periods, energy_txt=None,
              tu_fac=tu_fac)

    print(f"Found {len(outs)} Athena++ output(s)")
    for k, fname in enumerate(outs):
        convert_output(athdf, fname, output, k * idx_interval, gamma, tu_fac)


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(prog="ICconvert_athenatotipsy")
    pa.add_argument("rundir")
    pa.add_argument("output")
    pa.add_argument("idx_interval", type=int)
    pa.add_argument("paramfile", nargs="?", default=None)
    pa.add_argument("--athena-vis", default=None,
                    help="path to Athena++'s vis/python (athena_read.py)")
    a = pa.parse_args()
    convertfile(a.rundir, a.output, a.idx_interval, a.paramfile,
                athena_vis=a.athena_vis)
