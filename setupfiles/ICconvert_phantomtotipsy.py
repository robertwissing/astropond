#!/usr/bin/env python
"""Convert Phantom dump output back into gasoline/tipsy snapshots.

Phantom writes tagged Fortran dumps (`<prefix>_NNNNN`); they are read here with
Phantom's own scripts/readPhantomDump.read_dump and mapped to tipsy. Phantom does
not store density in the dump (it is reconstructed from the smoothing length), so
rho = massoftype * (hfact/h)**3 (the SPH h-rho relation). u is mapped back to a
temperature T = u / dTuFac.

    ICconvert_phantomtotipsy.py <rundir/prefix> <output> <idx_interval> [paramfile]

  rundir/prefix  Phantom run dir, or a dump-file prefix; globs '<prefix>_[0-9]*'.
  output         run name / output directory for the tipsy snapshots.
  idx_interval   snapshot stride (= iOutInterval): dump K -> <output>.{K*stride:05d}.
  paramfile      original gasoline .param (per-axis box periods for the .log).
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import readtipsy as tip
from ICconvert_arepototipsy import (
    write_log, _get_tufac, _periods_from_param, ENERGY_HEADER)


# Phantom .ev column name -> gasoline .log canonical column name. Phantom's .ev
# (src/main/evwrite.f90) is a DENSE per-timestep global-quantity time-series -- the
# same role gasoline's .log step rows play -- so mapping it gives the analysis a
# dense energy-vs-time series rather than the sparse snapshot sums. Quantities
# Phantom reports only as a TOTAL (momentum/angular momentum, not per-axis) or does
# not have (enstrophy, the Q* slots, wall-time) have no gasoline analogue here and
# are left 0.
_EV_TO_LOG = {
    "time": "dTime", "etot": "Etot", "ekin": "Ekin", "epot": "Epot",
    "etherm": "Eth", "emag": "Emag", "totentrop": "totentrop",
    "xcom": "cmx", "ycom": "cmy", "zcom": "cmz",
    "divB max": "divBMax", "divB ave": "divBAvg",
    "hdivB/B max": "divBerrMax", "hdivB/B ave": "divBerrAvg",
    "alpha max": "alphaMax",
    "beta_P max": "betaMax", "beta_P ave": "betaAvg", "beta_P min": "betaMin",
    "rmsmach": "rmsmach", "vrms": "vrms",
    "rho ave": "rhogasAvg", "rho max": "rhogasMax",
}


def _parse_ev_header(line):
    """{phantom_ev_name: 0-based column index} from a Phantom .ev header line like
    '# [01        time]   [02        ekin]   ...', or None if `line` is not such a
    header. The bracketed names may contain spaces ('rho max', 'beta_P ave')."""
    pairs = re.findall(r"\[\s*(\d+)\s+([^\]]+?)\s*\]", line)
    if not pairs:
        return None
    return {name.strip(): int(num) - 1 for num, name in pairs}


def _ev_log_rows(ev_paths):
    """DENSE gasoline .log value-rows (ENERGY_HEADER column order) from Phantom .ev
    file(s). Each .ev row is remapped via _EV_TO_LOG; unmapped gasoline columns are
    0. ev_paths are read in order (restart-split 01.ev, 02.ev, ...) and the merged
    rows are sorted by time. Returns [] if no usable rows are found."""
    log_index = {tok.strip("[]"): i
                 for i, tok in enumerate(ENERGY_HEADER.lstrip("#").split())}
    ncol = len(log_index)
    rows = []
    for ev in ev_paths:
        colmap = None
        with open(ev) as fh:
            for line in fh:
                hdr = _parse_ev_header(line)
                if hdr is not None:
                    colmap = hdr
                    continue
                s = line.strip()
                if not s or s.startswith("#") or colmap is None:
                    continue
                parts = s.split()
                if len(parts) < len(colmap):
                    continue
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue
                out = [0.0] * ncol
                for evname, ci in colmap.items():
                    logname = _EV_TO_LOG.get(evname)
                    if logname is not None and ci < len(vals):
                        out[log_index[logname]] = vals[ci]
                rows.append(out)
    rows.sort(key=lambda r: r[0])      # ENERGY_HEADER[0] = dTime
    return rows


def _load_reader():
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.normpath(os.path.join(
        here, "..", "codes", "phantom", "phantom", "scripts"))
    sys.path.insert(0, scripts)
    import readPhantomDump as rpd
    return rpd


def _scalar(q, key, default=0.0):
    if key in q:
        return float(np.atleast_1d(q[key])[0])
    return default


def convert_dump(rpd, path, output, out_idx, tu_fac):
    d = rpd.read_dump(path)
    data = d["blocks"][0]["data"]
    # B (and divB) live in a SEPARATE block: a Phantom MHD dump has narraylengths=4,
    # so readPhantomDump returns the gas arrays in block 0 and the magnetic arrays
    # (Bx,By,Bz,divB,...) in block 3. Find whichever block carries Bx.
    bdata = next((b["data"] for b in d["blocks"] if "Bx" in b.get("data", {})), {})
    q = d["quantities"]
    n = len(data["x"])
    massoftype = _scalar(q, "massoftype", 1.0)
    hfact = _scalar(q, "hfact", 1.2)
    time = np.array([_scalar(q, "time", 0.0)])

    tgdata = np.zeros((n, 12))
    tgdata[:, 0] = massoftype
    tgdata[:, 1] = data["x"]; tgdata[:, 2] = data["y"]; tgdata[:, 3] = data["z"]
    for col, k in ((4, "vx"), (5, "vy"), (6, "vz")):
        if k in data:
            tgdata[:, col] = data[k]
    h = np.asarray(data["h"], float)
    rho = np.where(h > 0, massoftype * (hfact / np.maximum(h, 1e-30)) ** 3, 0.0)
    tgdata[:, 7] = rho
    if "u" in data:
        tgdata[:, 8] = np.asarray(data["u"], float) / tu_fac      # T = u/dTuFac
    tgdata[:, 9] = h

    outname = os.path.join(output, f"{output}.{out_idx:05d}")
    tip.writetipsy(tgdata, [], [], outname, np.array([n, 3, n, 0, 0, 0]), time)
    tip.writetipsyaux(h, "smoothlength", outname)
    # MHD: write the magnetic field aux (BFieldx/y/z, + DivB) so the analysis can
    # compute E_mag -- matching the AREPO/GIZMO reverse converters. Phantom MHD dumps
    # carry Bx,By,Bz (and divB); hydro dumps don't, so this is skipped for them.
    if all(k in bdata for k in ("Bx", "By", "Bz")):
        tip.writetipsyaux(np.ascontiguousarray(np.asarray(bdata["Bx"], float)), "BFieldx", outname)
        tip.writetipsyaux(np.ascontiguousarray(np.asarray(bdata["By"], float)), "BFieldy", outname)
        tip.writetipsyaux(np.ascontiguousarray(np.asarray(bdata["Bz"], float)), "BFieldz", outname)
        if "divB" in bdata:
            tip.writetipsyaux(np.asarray(bdata["divB"], float), "DivB", outname)
        bnote = ", BField aux"
    else:
        bnote = ""
    print(f"Wrote {outname} from {os.path.basename(path)}{bnote}")


def convertfile(entry, output, idx_interval=1, paramfile=None):
    os.makedirs(output, exist_ok=True)
    rpd = _load_reader()
    if os.path.isdir(entry):
        dumps = sorted(glob.glob(os.path.join(entry, "*_[0-9][0-9][0-9][0-9][0-9]")))
    else:
        dumps = sorted(glob.glob(entry + "_[0-9][0-9][0-9][0-9][0-9]"))
    dumps = [d for d in dumps if not d.endswith((".ascii", ".tmp"))]
    if not dumps:
        print(f"No Phantom dumps found under '{entry}'", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(dumps)} Phantom dump(s)")

    # Resolve dTuFac (T = u/dTuFac) from the SAME source the forward converter used:
    # the gasoline .param's unit system (read_param_tufac), matching the AREPO/GIZMO
    # reverse converters. The Phantom output dump carries no dTuFac/dGasConst, so
    # _get_tufac's .log path falls back to 1.0 and would break the temperature
    # round-trip by a factor of dTuFac (~2881 for these ICs).
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
        periods = (1.0, 1.0, 1.0)
    # Stamp dTuFac into the .log so the analysis's read_tufac() recovers u = dTuFac*T
    # from the snapshot temperature column (else E_therm is undercounted by dTuFac).
    # Fold Phantom's DENSE per-timestep .ev series into the .log step rows (mapped to
    # the gasoline columns) so energy-vs-time diagnostics use it instead of the sparse
    # snapshots. The .ev lives beside the run (<prefix>NN.ev), not under output/.
    ev_paths = sorted(glob.glob(os.path.join(output, "*.ev")))
    ev_rows = _ev_log_rows(ev_paths) if ev_paths else None
    if ev_paths:
        print(f"  mapping {len(ev_paths)} Phantom .ev file(s) -> .log step rows")
    write_log(os.path.join(output, output + ".log"), periods, energy_txt=None,
              tu_fac=tu_fac, rows=ev_rows)
    for idx, dump in enumerate(dumps):
        convert_dump(rpd, dump, output, idx * idx_interval, tu_fac)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        entry = sys.argv[1]; output = sys.argv[2]; idx = int(sys.argv[3])
        paramfile = sys.argv[4] if len(sys.argv) >= 5 else None
    else:
        entry = input("Phantom run dir / prefix: ")
        output = input("Output name: ")
        idx = int(input("idx_interval (iOutInterval): "))
        paramfile = input("original tipsy .param (blank to skip): ").strip() or None
    convertfile(entry, output, idx, paramfile)
