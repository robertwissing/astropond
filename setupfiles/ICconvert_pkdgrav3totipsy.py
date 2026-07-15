#!/usr/bin/env python
"""Finalize PKDGRAV3 output for the testsuite analysis pipeline.

PKDGRAV3 writes its snapshots in tipsy format already (`<runname>.<NNNNN>`), so
there is NO snapshot format conversion -- the analysis reads them directly. The
one thing the gasoline analysis framework also needs is a gasoline-style `.log`
carrying the periodic-box header (dxPeriod/dyPeriod/dzPeriod); pkdgrav3's own log
has a different format. This writes that `.log` (header-only -> the analyses use
per-snapshot sums for energies), reusing the AREPO/GIZMO converter's writer.

    ICconvert_pkdgrav3totipsy.py <rundir> <output> <idx_interval> [paramfile]

  rundir        pkdgrav3 run dir holding the tipsy snapshots <output>.<NNNNN>.
  output        run name (the tipsy snapshots' base name == this).
  idx_interval  snapshot stride (= iOutInterval); accepted for signature parity.
  paramfile     original gasoline .param (for the box periods in the .log).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ICconvert_arepototipsy import write_log, _periods_from_param


def finalize(rundir, output, paramfile=None):
    periods = _periods_from_param(paramfile)
    if periods is None:
        periods = (1.0, 1.0, 1.0)
        print("  (no .param) writing a unit-box .log header; set the .param for "
              "correct dxPeriod/dyPeriod/dzPeriod.")
    else:
        print(f"  box periods from .param: {periods}")
    # pkdgrav3 has no gasoline energy series -> header-only .log (per-snapshot sums).
    logpath = os.path.join(output, output + ".log")
    os.makedirs(output, exist_ok=True)
    write_log(logpath, periods, energy_txt=None)
    print(f"Wrote {logpath} (pkdgrav3 snapshots are already tipsy; no snapshot "
          "conversion needed).")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        rundir = sys.argv[1]
        output = sys.argv[2]
        # idx_interval = int(sys.argv[3])   # accepted for parity; unused
        paramfile = sys.argv[4] if len(sys.argv) >= 5 else None
    else:
        rundir = input("pkdgrav3 run dir: ")
        output = input("Output name: ")
        paramfile = input("original tipsy .param (blank to skip): ").strip() or None
    finalize(rundir, output, paramfile)
