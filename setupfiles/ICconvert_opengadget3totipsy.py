#!/usr/bin/env python
"""Convert OpenGadget3 snapshot HDF5 output back into gasoline/tipsy snapshots.

OpenGadget3 writes GADGET type-0 HDF5 snapshots (`<base>_NNN.hdf5`, default base
"snap", PartType0 gas: Coordinates, Velocities, Masses, Density, InternalEnergy,
SmoothingLength, and MagneticField + DivergenceOfMagneticField when built with
MAGNETIC) -- the SAME layout as AREPO/GIZMO output, so the per-snapshot field
mapping is reused verbatim from `ICconvert_arepototipsy`. Only the snapshot file
glob (`snap_*.hdf5` in the OutputDir, here `<run>/output/`) differs from GIZMO's
`snapshot_*.hdf5`. The .log is header-only (per-snapshot sums; no AREPO energy.txt).

    ICconvert_opengadget3totipsy.py <input> <output> <idx_interval> [paramfile]

  input         OpenGadget3 run dir (containing output/snap_*.hdf5) or a single .hdf5.
  output        run name / output directory for the tipsy snapshots.
  idx_interval  snapshot stride (= iOutInterval): snap_K -> <output>.{K*stride:05d}.
  paramfile     original gasoline .param (for per-axis box centring + dTuFac).
"""
import os
import sys
import glob

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ICconvert_arepototipsy import (convert_snapshot, write_log, _get_tufac,
                                    _periods_from_param)
from sph_interp.export import MU0_CODE_TO_GAUSS


def _find_snaps(entry):
    if os.path.isdir(entry):
        snaps = sorted(glob.glob(os.path.join(entry, "snap_*.hdf5")))
        if not snaps:
            snaps = sorted(glob.glob(os.path.join(entry, "output", "snap_*.hdf5")))
        return snaps
    return [entry]


def convertfile(entry, output, idx_interval=1, paramfile=None):
    os.makedirs(output, exist_ok=True)
    snaps = _find_snaps(entry)
    if not snaps:
        print(f"No OpenGadget3 snap_*.hdf5 found under '{entry}'", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(snaps)} OpenGadget3 snapshot(s)")

    # Resolve dTuFac (T = u/dTuFac) from the gasoline .param (the same source the
    # forward converter used); the header-only .log carries no dTuFac, so _get_tufac
    # would otherwise fall back to 1.0 and undercount E_therm by dTuFac.
    tu_fac = None
    if paramfile:
        from sph_interp.particles import read_param_tufac
        stem = paramfile[:-6] if paramfile.endswith(".param") else paramfile
        tu_fac = read_param_tufac(stem)
    if tu_fac is None:
        tu_fac = _get_tufac(output)
    print(f"  dTuFac (u = dTuFac*T) = {tu_fac:g}")

    periods = _periods_from_param(paramfile)
    if periods is None:
        with h5py.File(snaps[0], "r") as f:
            box = float(np.atleast_1d(f["Header"].attrs["BoxSize"])[0])
        periods = (box, box, box)
        print(f"  (no .param) assuming cubic box {box:g}.")
    else:
        print(f"  box periods from .param: {periods}")

    write_log(os.path.join(output, output + ".log"), periods, energy_txt=None,
              tu_fac=tu_fac)

    cen = periods if paramfile and _periods_from_param(paramfile) else None

    # Name each tipsy snapshot by its PHYSICAL time (step = round(t/dt_out) *
    # idx_interval), so an end-of-run dump landing on/just past a scheduled output
    # time maps onto the SAME step and is dropped (same handling as GIZMO).
    times = []
    for snap in snaps:
        with h5py.File(snap, "r") as f:
            times.append(float(np.atleast_1d(f["Header"].attrs["Time"])[0]))
    diffs = [b - a for a, b in zip(times, times[1:]) if b - a > 1e-12]
    dt_out = float(np.median(diffs)) if diffs else 1.0
    seen = set()
    for snap, t in zip(snaps, times):
        out_idx = int(round(t / dt_out)) * idx_interval
        if out_idx in seen:
            print(f"  skipping redundant OpenGadget3 snapshot {os.path.basename(snap)} "
                  f"(t={t:g} duplicates step {out_idx:05d})")
            continue
        seen.add(out_idx)
        # OpenGadget3 (no MU0_UNITY) outputs B in Gauss -> divide by sqrt(4pi)
        # back to mu0=1 code units, consistent with GIZMO/AREPO.
        convert_snapshot(snap, output, out_idx, tu_fac, periods=cen,
                         b_scale=1.0 / MU0_CODE_TO_GAUSS)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        entry = sys.argv[1]
        output = sys.argv[2]
        idx_interval = int(sys.argv[3])
        paramfile = sys.argv[4] if len(sys.argv) >= 5 else None
    else:
        entry = input("OpenGadget3 run dir / snapshot: ")
        output = input("Output name: ")
        idx_interval = int(input("idx_interval (iOutInterval): "))
        paramfile = input("original tipsy .param (blank to skip): ").strip() or None
    convertfile(entry, output, idx_interval, paramfile)
