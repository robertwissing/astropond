#!/usr/bin/env python
"""Convert GIZMO snapshot HDF5 output back into gasoline/tipsy snapshots.

GIZMO writes GADGET type-0 HDF5 snapshots (`snapshot_NNN.hdf5`, PartType0 gas:
Coordinates, Velocities, Masses, Density, InternalEnergy, SmoothingLength, and
MagneticField when built with MAGNETIC) -- the SAME layout as AREPO's output, so
the per-snapshot field mapping is reused verbatim from `ICconvert_arepototipsy`.
Only the snapshot file glob (snapshot_*.hdf5, in <run>/output/) and the absence of
an AREPO-style energy.txt differ, so the .log is header-only and the analyses use
per-snapshot sums.

    ICconvert_gizmototipsy.py <input> <output> <idx_interval> [paramfile]

  input         GIZMO run dir (containing output/snapshot_*.hdf5) or a single .hdf5.
  output        run name / output directory for the tipsy snapshots.
  idx_interval  snapshot stride (= iOutInterval): snapshot_K -> <output>.{K*stride:05d}.
  paramfile     original gasoline .param (for per-axis box centring).
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


def convertfile(entry, output, idx_interval=1, paramfile=None):
    os.makedirs(output, exist_ok=True)
    if os.path.isdir(entry):
        snaps = sorted(glob.glob(os.path.join(entry, "snapshot_*.hdf5")))
        if not snaps:
            snaps = sorted(glob.glob(os.path.join(entry, "output",
                                                  "snapshot_*.hdf5")))
    else:
        snaps = [entry]
    if not snaps:
        print(f"No GIZMO snapshot_*.hdf5 found under '{entry}'", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(snaps)} GIZMO snapshot(s)")

    # Resolve dTuFac (T = u/dTuFac) from the SAME source the forward converter used
    # -- the gasoline .param units -- because the header-only .log we write below
    # carries no dTuFac/dGasConst, so _get_tufac's .log path would fall back to 1.0
    # and break the temperature round-trip by a factor of dTuFac (~2881 for these
    # runs). Fall back to the .log only if the .param can't be read.
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

    # GIZMO has no AREPO-style energy.txt -> header-only .log (per-snapshot sums).
    # Stamp dTuFac so the analysis's read_tufac() recovers u = dTuFac*T from the
    # snapshot temperature column (else E_therm is undercounted by dTuFac).
    write_log(os.path.join(output, output + ".log"), periods, energy_txt=None,
              tu_fac=tu_fac)

    cen = periods if paramfile and _periods_from_param(paramfile) else None

    # GIZMO writes snapshots at the regular cadence PLUS a redundant final dump at
    # end-of-run (run.c: savepositions() at TimeMax even when the scheduled output
    # at that time already fired, because the integrator advances one extra substep
    # so Ti_lastoutput != Ti_Current). Name each tipsy snapshot by its PHYSICAL
    # time (step = round(time/dt_out) * idx_interval) instead of file index, so the
    # extra dump maps onto the SAME step as the scheduled one and is dropped --
    # otherwise it becomes a spurious, mislabelled snapshot (e.g. .00400 at t=3,
    # alongside the real .00300).
    times = []
    for snap in snaps:
        with h5py.File(snap, "r") as f:
            times.append(float(np.atleast_1d(f["Header"].attrs["Time"])[0]))
    # Regular output interval: the median positive gap, robust to a final dump that
    # lands slightly past the scheduled time (which would fool a plain min()).
    diffs = [b - a for a, b in zip(times, times[1:]) if b - a > 1e-12]
    dt_out = float(np.median(diffs)) if diffs else 1.0
    seen = set()
    for snap, t in zip(snaps, times):
        out_idx = int(round(t / dt_out)) * idx_interval
        if out_idx in seen:
            print(f"  skipping redundant GIZMO snapshot {os.path.basename(snap)} "
                  f"(t={t:g} duplicates step {out_idx:05d})")
            continue
        seen.add(out_idx)
        # GIZMO outputs B in Gauss -> divide by sqrt(4pi) back to mu0=1 code units.
        convert_snapshot(snap, output, out_idx, tu_fac, periods=cen,
                         b_scale=1.0 / MU0_CODE_TO_GAUSS)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        entry = sys.argv[1]
        output = sys.argv[2]
        idx_interval = int(sys.argv[3])
        paramfile = sys.argv[4] if len(sys.argv) >= 5 else None
    else:
        entry = input("GIZMO run dir / snapshot: ")
        output = input("Output name: ")
        idx_interval = int(input("idx_interval (iOutInterval): "))
        paramfile = input("original tipsy .param (blank to skip): ").strip() or None
    convertfile(entry, output, idx_interval, paramfile)
