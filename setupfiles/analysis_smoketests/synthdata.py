#!/usr/bin/env python3
"""
Synthetic run-directory builder for the analysis smoke suite (tests/test_analyses.py).

The IC_analysis_* scripts read three things from a run folder:
  * tipsy snapshots  '<runname>.NNNNN'         (binary, big-endian; readtipsy)
  * per-particle aux '<runname>.NNNNN.<LABEL>' (BFieldx/y/z, DivB, ...)
  * the run log       '<runname>.log'          (52-col timestep series + a few
                                                '# key: value' parameter lines)

This module writes all three from in-memory numpy arrays, so each test can stand
up a tiny, fully-controlled run on disk (no gasoline, no real IC build) and drive
an analysis end-to-end. It is the captured, reusable form of the ad-hoc synthetic
runs the analyses were originally validated against.

Gas particle columns (the 12 tipsy gas floats, mirrored by loaddata):
  0 mass | 1-3 x,y,z | 4-6 vx,vy,vz | 7 rho | 8 temperature(=u/dTuFac)
  9 hsmooth | 10 metals | 11 phi
"""

import os

import numpy as np

from IC_analysis_general import LOG_COLUMNS

# tipsy on-disk dtypes (big-endian), matching readtipsy.readtipsy / try_aux.
_BE_I4 = np.dtype(">i4")
_BE_F4 = np.dtype(">f4")
_BE_F8 = np.dtype(">f8")


# --------------------------------------------------------------------------- #
#  Gas-particle table
# --------------------------------------------------------------------------- #

NGAS_COLS = 12


def gas_table(n):
    """An (n, 12) gas array zeroed out, with sensible defaults (mass=1/n, rho=1,
    h=2/n^(1/3)). Fill the columns you care about, pass to write_snapshot."""
    g = np.zeros((n, NGAS_COLS), dtype=float)
    g[:, 0] = 1.0 / n              # mass
    g[:, 7] = 1.0                  # rho
    g[:, 8] = 1.0                  # temperature
    g[:, 9] = 2.0 * n ** (-1.0 / 3.0)   # hsmooth
    return g


# --------------------------------------------------------------------------- #
#  Writers (binary, big-endian -- matching readtipsy / try_aux)
# --------------------------------------------------------------------------- #

# tipsy star record: mass | x,y,z | vx,vy,vz | metals | tform | eps | phi
NSTAR_COLS = 11


def star_table(n, mass=0.0):
    """An (n, 11) star array zeroed out, with a uniform particle mass."""
    s = np.zeros((n, NSTAR_COLS), dtype=float)
    s[:, 0] = mass
    return s


def write_snapshot(path, gas, time, stars=None):
    """Write a tipsy snapshot to `path` at `time` (big-endian binary, the layout
    readtipsy.readtipsy reads: time f8, header 6*i4, gas ngas*12 f4, then the
    star block nstar*11 f4).

    `stars` is optional: omit it for the gas-only snapshots most tests want.
    A test that converts gas into stars (starform) passes both blocks, and the
    header's nstar is what the analysis reads back."""
    gas = np.asarray(gas, dtype=float)
    ngas = len(gas)
    stars = (np.zeros((0, NSTAR_COLS)) if stars is None
             else np.asarray(stars, dtype=float))
    nstar = len(stars)
    header = np.array([ngas + nstar, 3, ngas, 0, nstar, 0])
    with open(path, "wb") as fh:
        np.array([float(time)]).astype(_BE_F8).tofile(fh)
        header.astype(_BE_I4).tofile(fh)
        gas.astype(_BE_F4).tofile(fh)
        if nstar:
            stars.astype(_BE_F4).tofile(fh)


def write_scalar_aux(path, label, arr):
    """Write a per-particle scalar aux '<path>.<label>' (npart i4, then f4 data)."""
    arr = np.asarray(arr, dtype=float)
    with open(path + "." + label, "wb") as fh:
        np.array([len(arr)]).astype(_BE_I4).tofile(fh)
        arr.astype(_BE_F4).tofile(fh)


def write_vector_aux(path, label, vec):
    """Write a per-particle vector aux as component files '<path>.<label>x/y/z'
    (the layout real gasoline runs use; read_vector_aux falls back to it)."""
    vec = np.asarray(vec, dtype=float)
    for i, comp in enumerate("xyz"):
        write_scalar_aux(path, label + comp, vec[:, i])


# --------------------------------------------------------------------------- #
#  Run log
# --------------------------------------------------------------------------- #

DEFAULT_LOG_PARAMS = dict(
    dMsolUnit=1.0, dKpcUnit=1.0, dTuFac=1.0, dConstGamma=5.0 / 3.0,
    dxPeriod=1.0, dyPeriod=1.0, dzPeriod=1.0, dPeriod=1.0,
)


def write_log(path, series=None, params=None, extra_lines=None):
    """Write a gasoline '<runname>.log'.

    series : dict {LOG_COLUMNS name -> 1-D array}; rows are written for every
             index of the longest array (missing columns 0). 'dTime' should be
             provided so the per-timestep series has an x-axis.
    params : overrides for the '# key: value' header lines (dTuFac, dxPeriod,
             dMsolUnit, ...). See DEFAULT_LOG_PARAMS.
    extra_lines : additional verbatim header lines, written after the standard
             block -- for sections this template does not model (the
             '# Preprocessor macros:' line, '# STAR FORMATION: ...'). They must
             start with '#' to stay out of the numeric series.
    """
    p = dict(DEFAULT_LOG_PARAMS)
    if params:
        p.update(params)
    lines = [
        "# SMOKE-TEST synthetic log",
        ("# UNITS:  dMsolUnit: {dMsolUnit} dKpcUnit: {dKpcUnit} "
         "dErgPerGmUnit: 1 dGmPerCcUnit (z=0): 1 dSecUnit: 1 "
         "dKmPerSecUnit (z=0): 1").format(**p),
        ("# BOX PARAMETERS:  bPeriodic: 1 bRestart: 0 bComove: 0 "
         "dPeriod: {dPeriod} dxPeriod: {dxPeriod} dyPeriod: {dyPeriod} "
         "dzPeriod: {dzPeriod}").format(**p),
        ("# GAS PHYSICS:  dConstGamma: {dConstGamma} dMeanMolWeight: 1 "
         "dGasConst: 1 dTuFac: {dTuFac}").format(**p),
    ]
    if extra_lines:
        lines.extend(extra_lines)
    if series:
        ncol = len(LOG_COLUMNS)
        nrow = max(len(v) for v in series.values())
        rows = np.zeros((nrow, ncol), dtype=float)
        for name, vals in series.items():
            j = LOG_COLUMNS[name]
            vals = np.asarray(vals, dtype=float)
            rows[:len(vals), j] = vals
        for r in rows:
            lines.append(" ".join("%.12e" % v for v in r))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
#  Whole-run orchestration
# --------------------------------------------------------------------------- #

def make_run(parent, runname, snaps, start=10, step=10,
             log_series=None, log_params=None, log_extra_lines=None,
             starlog=None):
    """Create a run directory `parent/runname/` and populate it.

    snaps : list of per-snapshot dicts, each:
            {"time": float, "gas": (n,12) array,
             "aux": {"BField": (n,3), "DivB": (n,), ...}}   (aux optional)
            written as '<runname>.NNNNN' with NNNNN = start, start+step, ...
            (5-digit zero-padded, the suffix find_files keys on).
    log_series / log_params / log_extra_lines : forwarded to write_log
            ('<runname>.log').
    starlog : optional structured array of star-formation events (the
            IC_starlog dtype) written as '<runname>.starlog'; pass a 0-length
            array to create the magic-number-only file a run with no events
            leaves behind.

    Returns the run directory path (also a valid analysis input argument).
    """
    rundir = os.path.join(parent, runname)
    os.makedirs(rundir, exist_ok=True)
    for k, snap in enumerate(snaps):
        suffix = start + k * step
        base = os.path.join(rundir, "%s.%05d" % (runname, suffix))
        write_snapshot(base, snap["gas"], snap["time"], snap.get("stars"))
        for label, data in snap.get("aux", {}).items():
            data = np.asarray(data, dtype=float)
            if data.ndim == 2:
                write_vector_aux(base, label, data)
            else:
                write_scalar_aux(base, label, data)
    write_log(os.path.join(rundir, runname + ".log"),
              series=log_series, params=log_params, extra_lines=log_extra_lines)
    if starlog is not None:
        write_starlog(os.path.join(rundir, runname + ".starlog"), starlog)
    return rundir


def write_starlog(path, events):
    """Write a gasoline '<runname>.starlog': the record-size magic, then the
    events (an IC_starlog-dtype structured array, possibly empty).

    The dtype is forced back to big-endian first: numpy normalises a structured
    dtype to NATIVE byte order across ordinary operations (concatenate, fancy
    indexing), so a caller that assembled rows row-by-row would otherwise write
    a little-endian body under a header claiming the on-disk layout."""
    events = np.asarray(events)
    be = np.dtype([(name, events.dtype.fields[name][0].str
                    .replace("<", ">").replace("=", ">"))
                   for name in (events.dtype.names or ())])
    events = events.astype(be)
    with open(path, "wb") as fh:
        np.array([be.itemsize]).astype(_BE_I4).tofile(fh)
        if len(events):
            events.tofile(fh)
