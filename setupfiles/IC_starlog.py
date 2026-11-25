#!/usr/bin/env python3
"""Reader for Gasoline's `<runname>.starlog` star-formation event file.

The starlog is the ONLY record linking a star back to the gas particle it came
from (`iOrdGas`) and to the state that passed the formation criterion, so it is
what turns "N stars appeared" into a per-event check of the recipe.

Format (README.starlog + `pkdStarLogFlush`): an XDR int holding
`sizeof(SFEVENT)`, which doubles as a magic number, then one record per event:

    iOrdStar, iOrdGas          int32
    timeForm                   float64
    rForm[3], vForm[3]         float64
    massForm, rhoForm, TForm   float64

all big-endian, appended (serially, per rank) whenever the run writes an output
or a checkpoint.  The struct has no padding, so the magic equals the on-disk
record size and therefore names the layout: two compile-time options extend it,
`SFEVENTCRITWRITE` with (tcool, tdyn) and `COOLING_MOLECULARH` with H2fracForm,
in that order.  Records are NOT time-ordered across ranks -- sort on `timeForm`.

Caveats worth knowing before trusting a column:

  * `iOrdStar` is filled later, by `pkdNewOrder`, so events flushed before the
    global renumbering carry a placeholder.  Join on `iOrdGas`.
  * `TForm` is 0 under a NOCOOLING build (`starform.c` has no temperature to
    report there, and the criterion itself uses a hard-wired 10 K).
  * `rhoForm` is the density the criterion used, already divided by the cosmo
    factor, i.e. physical code density.

Standalone by design -- no analysis-framework imports -- so it can be used from
a plain Python session:

    python IC_starlog.py <run-dir-or-.starlog> [-n 20]
"""

import glob
import os
import sys

import numpy as np

# The base record, in the order pkdStarLogFlush writes it.
_BASE = [("iOrdStar", ">i4"), ("iOrdGas", ">i4"),
         ("timeForm", ">f8"),
         ("rFormx", ">f8"), ("rFormy", ">f8"), ("rFormz", ">f8"),
         ("vFormx", ">f8"), ("vFormy", ">f8"), ("vFormz", ">f8"),
         ("massForm", ">f8"), ("rhoForm", ">f8"), ("TForm", ">f8")]
_BASE_SIZE = 2 * 4 + 10 * 8              # 88

# Optional trailing fields, keyed by the record size the magic number reports.
_CRIT = [("tcool", ">f8"), ("tdyn", ">f8")]          # SFEVENTCRITWRITE
_H2 = [("H2fracForm", ">f8")]                        # COOLING_MOLECULARH
_LAYOUTS = {
    _BASE_SIZE: [],
    _BASE_SIZE + 8: _H2,
    _BASE_SIZE + 16: _CRIT,
    _BASE_SIZE + 24: _CRIT + _H2,
}


def record_dtype(size=_BASE_SIZE):
    """The on-disk (big-endian) record dtype for a given record-size magic.

    Anything that BUILDS starlog rows -- the smoke suite's synthetic runs, a
    one-off fixture -- should allocate with this, because numpy silently
    normalises a structured dtype to native byte order across operations like
    concatenate, and the file format is big-endian.
    """
    if size not in _LAYOUTS:
        raise ValueError(f"unknown starlog record size {size}")
    return np.dtype(_BASE + _LAYOUTS[size])


def find_starlog(run_arg):
    """The `<runname>.starlog` for a run directory, file prefix, or the file.

    Mirrors how the analysis scripts locate a run's '.log': accepts the run
    directory, a snapshot/file prefix, or the starlog path itself.
    """
    if os.path.isfile(run_arg) and run_arg.endswith(".starlog"):
        return run_arg
    if os.path.isdir(run_arg):
        hits = sorted(glob.glob(os.path.join(run_arg, "*.starlog")))
    else:
        hits = sorted(glob.glob(run_arg + "*.starlog"))
        if os.path.isfile(run_arg + ".starlog"):
            hits.append(run_arg + ".starlog")
    if not hits:
        raise FileNotFoundError(f"no .starlog found for '{run_arg}'")
    return hits[0]


def read_starlog(run_arg, sort=True):
    """Read a starlog into a structured array (one row per formation event).

    Returns a 0-length array with the base dtype when the run formed no stars:
    gasoline creates the file with its magic number at startup and only appends
    when `pkdStarLogFlush` has something to write, so "no events" is the normal,
    non-exceptional outcome for every control case in the suite.
    """
    path = find_starlog(run_arg)
    nbytes = os.path.getsize(path)
    if nbytes < 4:
        raise ValueError(f"{path}: too short to hold the record-size magic")
    size = int(np.fromfile(path, dtype=">i4", count=1)[0])
    if size not in _LAYOUTS:
        raise ValueError(
            f"{path}: record-size magic {size} is not a layout this reader "
            f"knows ({sorted(_LAYOUTS)}); a differently-compiled SFEVENT?")
    dtype = np.dtype(_BASE + _LAYOUTS[size])
    assert dtype.itemsize == size, (dtype.itemsize, size)
    nrec, spare = divmod(nbytes - 4, size)
    if spare:
        # A run killed mid-flush leaves a partial record; read what is whole
        # rather than refusing the whole file.
        print(f"{path}: {spare} trailing bytes are a partial record; ignoring",
              file=sys.stderr)
    events = np.fromfile(path, dtype=dtype, count=nrec, offset=4)
    if sort and nrec:
        events = events[np.argsort(events["timeForm"], kind="stable")]
    return events


def events_in_step(events, t0, t1):
    """Events with t0 < timeForm <= t1 -- the formation events of one step.

    Half-open on the left because `stfmFormStarParticle` stamps `timeForm` with
    the time at the END of the step it fired in, so a step's events all carry
    that step's t1 exactly.
    """
    t = np.asarray(events["timeForm"], dtype=float)
    return events[(t > t0) & (t <= t1)]


def summarize(events, limit=10):
    """A short human-readable digest (used by the CLI and the analysis print)."""
    lines = [f"{len(events)} star-formation event(s)"]
    if not len(events):
        return "\n".join(lines)
    t = np.asarray(events["timeForm"], dtype=float)
    lines.append("  timeForm  %.6g .. %.6g  (%d distinct)"
                 % (t.min(), t.max(), len(np.unique(t))))
    lines.append("  massForm  sum %.6g  mean %.6g"
                 % (events["massForm"].sum(), events["massForm"].mean()))
    lines.append("  rhoForm   %.6g .. %.6g"
                 % (events["rhoForm"].min(), events["rhoForm"].max()))
    lines.append("  TForm     %.6g .. %.6g"
                 % (events["TForm"].min(), events["TForm"].max()))
    lines.append("  first %d rows:" % min(limit, len(events)))
    for ev in events[:limit]:
        lines.append("   iOrdGas %8d  t %.6g  r (%+.4f %+.4f %+.4f)  m %.4g "
                     "rho %.4g  T %.4g"
                     % (ev["iOrdGas"], ev["timeForm"], ev["rFormx"],
                        ev["rFormy"], ev["rFormz"], ev["massForm"],
                        ev["rhoForm"], ev["TForm"]))
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="run directory, file prefix, or a .starlog")
    p.add_argument("-n", type=int, default=10, help="rows to print (default 10)")
    args = p.parse_args()
    events = read_starlog(args.run)
    print(find_starlog(args.run))
    print(summarize(events, limit=args.n))


if __name__ == "__main__":
    main()
