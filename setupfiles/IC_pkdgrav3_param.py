#!/usr/bin/env python
"""Write a PKDGRAV3 `.par` for a tipsy IC (no IC conversion -- tipsy is native).

    IC_pkdgrav3_param.py <tipsy_ic> <runname> <par_out> [--boxsize L] [--no-hydro]

  tipsy_ic   the tipsy IC fed to pkdgrav3 (achInFile), e.g. datafiles/run.00000.
  runname    achOutName; snapshots land at <runname>/<runname>.<NNNNN>.
  par_out    path to write the .par to (e.g. <runname>/run.par).

Run parameters (dDelta, nSteps, iOutInterval, nSmooth, gamma) come from the
gasoline `.param` beside the tipsy IC. See sph_interp/_pkdgrav3/pkdgrav3_param.py
for the mapping and the SPH unit caveat.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp._pkdgrav3.pkdgrav3_param import write_run_param


def _find_gasoline_param(tipsy_input):
    base = tipsy_input
    if "." in os.path.basename(base):
        base = base.rsplit(".", 1)[0]
    cand = base + ".param"
    return cand if os.path.exists(cand) else None


def main(argv=None):
    pa = argparse.ArgumentParser(prog="IC_pkdgrav3_param")
    pa.add_argument("tipsy_ic")
    pa.add_argument("runname")
    pa.add_argument("par_out")
    pa.add_argument("--boxsize", type=float, default=None)
    pa.add_argument("--no-hydro", action="store_true",
                    help="gravity-only run (omit the SPH hydro_model block)")
    args = pa.parse_args(argv)

    gas_param = _find_gasoline_param(args.tipsy_ic)
    if gas_param is None:
        sys.exit("FATAL: no gasoline .param found beside %r" % args.tipsy_ic)
    # pkdgrav3 runs from the testsuite cwd, so achInFile is the IC path as given;
    # snapshots use achOutPath '<runname>/{name}.{step:05d}{type}' -> the suite's
    # <runname>/<runname>.<NNNNN> convention.
    outpath = os.path.join(args.runname, "{name}.{step:05d}{type}")
    write_run_param(gas_param, args.par_out, args.tipsy_ic, args.runname,
                    outpath, boxsize=args.boxsize, hydro=not args.no_hydro)
    print("Wrote PKDGRAV3 .par -> %s (run params from %s)"
          % (args.par_out, gas_param))


if __name__ == "__main__":
    main()
