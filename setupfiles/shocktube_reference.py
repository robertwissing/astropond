#!/usr/bin/env python3
"""
Tabulate high-resolution Athena++ reference solutions for the shock-tube presets.

WHY. The MHD Riemann problem has no closed form, so IC_analysis_shocktube overlays
piecewise-constant states transcribed from the literature (SPLASH exact_mhdshock).
That table has two limits: it only exists for the seven problems SPLASH ships (Ryu
1b #3, Ryu 2b #5 have none), and it describes the CLEAN single-fan Riemann problem
while our runs are something else -- a PERIODIC box twice as long as the shock
region, with IC_setup_shocktube's exponential velocity taper near the edges:

    lim = 0.4 * dxbound            (dxbound = dxPeriod, the FULL box length)
    v  *= exp(-8 * (|x| - lim))        for |x| > lim

So the table is right only over the part of the box neither the image fan nor the
taper has reached, which for the fast-inflow presets is a small fraction of it.

WHAT. This drives Athena++ (MHD/CT) at high resolution over the SAME initial state
and the SAME periodic box as the run, and tabulates the result as a dense polyline
in exactly the shape IC_analysis_shocktube already consumes. Two modes:

  --mode suite  (default)  patches/shocktube_suite.cpp: periodic + taper, i.e. the
                run's own IC. Valid over the WHOLE box, at the run's own time; no
                mirroring assumption and no clipping. This is the one to score
                against.
  --mode clean  Athena++'s own shock_tube.cpp: the textbook two-state problem with
                outflow boundaries. Not comparable to a periodic run -- it is here
                to CHECK the transcribed SPLASH tables (--check).

The suite reference is NOT self-similar (the box and the taper set a length scale),
so unlike the SPLASH tables it cannot be rescaled in time: one file per (preset,
box, time). Files land in reference_solution_highres/shocktube/ and are picked up
automatically by IC_analysis_shocktube.

Usage:
    python shocktube_reference.py <run-dir> [<run-dir> ...]      # box+time from the run
    python shocktube_reference.py -a 3 --dxperiod 2 --time 0.06
    python shocktube_reference.py <run-dir> --mode clean --check
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IC_analysis_shocktube as st            # noqa: E402  (presets + reference dir)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BIN = {"suite": "athena_shocktube_suite", "clean": "athena_shocktube",
       "iso": "athena_shocktube_iso"}
FIELDS = ("rho", "P", "vx", "vy", "vz", "By", "Bz")
# Athena++ `variable=prim` column names -> ours (Bcc* are the cell-centred fields).
COLMAP = {"rho": "rho", "press": "P", "vel1": "vx", "vel2": "vy", "vel3": "vz",
          "Bcc2": "By", "Bcc3": "Bz"}


def athena_bin(mode):
    src = os.environ.get("ATHENA_SRC",
                         os.path.join(ROOT, "codes", "athena++", "athena"))
    path = os.path.join(src, "bin", BIN[mode])
    if not os.path.isfile(path):
        sys.exit(f"shocktube_reference: {path} missing -- run "
                 f"codes/athena++/build_athena.sh first")
    return path


# --------------------------------------------------------------------------- #
#  Run metadata: the box and the comparison time come from the RUN, not a guess
# --------------------------------------------------------------------------- #

def run_dxperiod(run):
    """dxPeriod (the full periodic box length in x) from the run's .log header."""
    for log in sorted(glob.glob(os.path.join(run, "*.log"))):
        with open(log, "r", errors="ignore") as fh:
            for line in fh:
                if "dxPeriod" in line:
                    mm = re.search(r"dxPeriod:\s*([0-9.eE+-]+)", line)
                    if mm:
                        return float(mm.group(1))
    return None


def setup_box_and_time(choice):
    """(dxPeriod, tmax) straight from IC_setup_shocktube -- no duplicated constants.

    choose_shock() sets the preset's `ti` (box multiplier; 4 for Ryu 1a, 2 elsewhere),
    deltastep and nsteps as a side effect and returns the shock half-widths, so the box
    is ti*(xright-xleft) and the run's end time is deltastep*nsteps (NOT self.tmax,
    which most presets leave at the class default 0.2; deltastep*nsteps reproduces
    every shocktube run on disk). Only a default -- a run's own .log and final snapshot
    are authoritative, and suite_reference() rejects a reference whose box or time does
    not match the run it is asked to score."""
    import IC_setup_shocktube as ss
    s = ss.setup_shocktube()
    xleft, xright = s.choose_shock(choice)
    return s.ti * (xright - xleft), s.deltastep * s.nsteps


def run_choice(run):
    """Preset number from the run name (the runtest.sh spelling)."""
    base = os.path.basename(os.path.normpath(run))
    for name, c in sorted(st.RUNNAME_CHOICES.items(), key=lambda kv: -len(kv[0])):
        if name in base:
            return c
    return None


# --------------------------------------------------------------------------- #
#  Athena++ driver
# --------------------------------------------------------------------------- #

ATHINPUT = """<comment>
problem = {name} (choice {choice}) -- {modedesc}
generated by setupfiles/shocktube_reference.py

<job>
problem_id = stref

<output1>
file_type   = tab
variable    = prim
data_format = %24.16e
dt          = {tlim:.10g}

<time>
cfl_number  = {cfl}
nlim        = -1
tlim        = {tlim:.10g}
integrator  = {integrator}
xorder      = {xorder}
ncycle_out  = 100000

<mesh>
nx1    = {nx1:d}
x1min  = {x1min:.16g}
x1max  = {x1max:.16g}
ix1_bc = {bc}
ox1_bc = {bc}

nx2    = 1
x2min  = -0.5
x2max  = 0.5
ix2_bc = periodic
ox2_bc = periodic

nx3    = 1
x3min  = -0.5
x3max  = 0.5
ix3_bc = periodic
ox3_bc = periodic

<hydro>
{eos}

<problem>
shock_dir = 1
xshock    = 0.0
{taper}
dl = {L[0]:.16g}
pl = {L[1]:.16g}
ul = {L[2]:.16g}
vl = {L[3]:.16g}
wl = {L[4]:.16g}
bxl = {L[5]:.16g}
byl = {L[6]:.16g}
bzl = {L[7]:.16g}

dr = {R[0]:.16g}
pr = {R[1]:.16g}
ur = {R[2]:.16g}
vr = {R[3]:.16g}
wr = {R[4]:.16g}
bxr = {R[5]:.16g}
byr = {R[6]:.16g}
bzr = {R[7]:.16g}
"""


def is_isothermal(p):
    return p["gamma"] <= 1.0 + 1e-6


def run_athena(p, mode, x1min, x1max, tlim, nx1, xorder, cfl, integrator, workdir):
    """Run Athena++ once; return (x, {field: values}) of the final tab dump."""
    if mode == "suite":
        taper = (f"taper_lim  = {0.4 * (x1max - x1min):.16g}\n"
                 f"taper_rate = 8.0")
        bc = "periodic"
        modedesc = "suite IC (periodic box + IC_setup_shocktube velocity taper)"
    else:
        taper = ""
        bc = "outflow"
        modedesc = "clean two-state Riemann problem (outflow boundaries)"
    # gamma=1 presets need the isothermal binary; pressure then comes from the
    # sound speed (the pgen never reads pl/pr under a barotropic EOS).
    binmode = "iso" if (is_isothermal(p) and mode == "suite") else mode
    eos = (f"iso_sound_speed = {np.sqrt(p.get('cs2', 1.0)):.16g}"
           if is_isothermal(p) else f"gamma = {p['gamma']:.16g}")
    text = ATHINPUT.format(name=p["name"], choice=p["choice"], modedesc=modedesc,
                           tlim=tlim, cfl=cfl, integrator=integrator,
                           xorder=xorder, nx1=nx1, x1min=x1min, x1max=x1max,
                           bc=bc, eos=eos, taper=taper,
                           L=list(p["L"]), R=list(p["R"]))
    inp = os.path.join(workdir, "athinput.stref")
    with open(inp, "w") as fh:
        fh.write(text)
    exe = athena_bin(binmode)
    cmd = [exe, "-i", "athinput.stref"]
    if shutil.which("mpirun"):
        cmd = ["mpirun", "-np", "1"] + cmd
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-4000:] + res.stderr[-4000:])
        sys.exit(f"shocktube_reference: Athena++ failed ({res.returncode})")
    tabs = sorted(glob.glob(os.path.join(workdir, "*.tab")))
    if not tabs:
        sys.exit("shocktube_reference: Athena++ produced no .tab output")
    return read_tab(tabs[-1])


def read_tab(path):
    """(x, {field: values}) from an Athena++ formatted-table dump."""
    names, time = None, None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "time=" in line:
                time = float(re.search(r"time=([0-9.eE+-]+)", line).group(1))
            elif "x1v" in line:
                names = line.lstrip("#").split()
    if names is None:
        sys.exit(f"shocktube_reference: no column header in {path}")
    data = np.loadtxt(path, comments="#")
    col = {n: data[:, i] for i, n in enumerate(names)}
    out = {ours: col[theirs] for theirs, ours in COLMAP.items() if theirs in col}
    for k in FIELDS:                          # hydro build: no B columns
        out.setdefault(k, np.zeros(len(data)))
    return col["x1v"], out, time


# --------------------------------------------------------------------------- #
#  Tabulation: decimate the profile to the nodes the shape actually needs
# --------------------------------------------------------------------------- #

def decimate(x, state, tol):
    """Douglas-Peucker on the multi-field polyline; returns the kept-node mask.

    Error is the max over fields of |value - linear interpolation between the two
    surviving anchors|, each field normalised by its own range, so `tol` is a
    single relative number. Flat plateaus collapse to their endpoints and only the
    waves keep nodes -- the same economy as the hand-typed SPLASH tables, just
    resolved rather than assumed piecewise constant."""
    F = np.array([state[k] for k in FIELDS], float)
    rng = F.max(axis=1) - F.min(axis=1)
    rng[rng <= 0] = 1.0
    F = F / rng[:, None]
    n = len(x)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        t = (x[i0 + 1:i1] - x[i0]) / (x[i1] - x[i0])
        approx = F[:, i0][:, None] + (F[:, i1] - F[:, i0])[:, None] * t[None, :]
        err = np.max(np.abs(F[:, i0 + 1:i1] - approx), axis=0)
        j = int(np.argmax(err))
        if err[j] > tol:
            k = i0 + 1 + j
            keep[k] = True
            stack.append((i0, k))
            stack.append((k, i1))
    return keep


def write_csv(path, p, mode, x, state, time, x1min, x1max, nx1, tol, xorder,
              integrator, cfl):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = [
        f"shocktube reference -- {p['name']} (choice {p['choice']})",
        f"mode: {mode}  "
        + ("periodic box + IC_setup_shocktube velocity taper (lim=0.4*dxbound, "
           "rate=8): comparable over the WHOLE box"
           if mode == "suite" else
           "clean two-state Riemann problem, outflow boundaries"),
        "code: Athena++ MHD/CT, pgen "
        + ("patches/shocktube_suite.cpp" if mode == "suite" else "shock_tube.cpp"),
        f"athena: nx1={nx1} integrator={integrator} xorder={xorder} cfl={cfl}",
        f"choice: {p['choice']}",
        f"gamma: {p['gamma']:.12g}",
        f"box: {x1min:.12g} {x1max:.12g}",
        f"time: {time:.12g}",
        f"nodes: {len(x)} decimated from {nx1} at tol={tol:.3g} "
        f"(max per-field error, range-normalised)",
        "NOT self-similar: valid at this box and this time only"
        if mode == "suite" else "self-similar: positions scale as x*t/time",
        "columns: x," + ",".join(FIELDS),
    ]
    with open(path, "w") as fh:
        for line in hdr:
            fh.write("# " + line + "\n")
        fh.write("x," + ",".join(FIELDS) + "\n")
        for i in range(len(x)):
            fh.write(",".join(f"{v:.9e}" for v in
                              [x[i]] + [state[k][i] for k in FIELDS]) + "\n")
    return path


def reference_name(p, mode, time):
    return f"{p['name'].replace(' ', '_')}_{mode}_t{time:.6g}.csv"


# --------------------------------------------------------------------------- #

def generate(p, mode, dxperiod, time, args):
    x1max = 0.5 * dxperiod
    x1min = -x1max
    workdir = tempfile.mkdtemp(prefix="stref_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        x, state, tout = run_athena(p, mode, x1min, x1max, time, args.nx,
                                    args.xorder, args.cfl, args.integrator,
                                    workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    keep = decimate(x, state, args.tol)
    xs = x[keep]
    ss = {k: state[k][keep] for k in FIELDS}
    out = os.path.join(st.SUITE_REF_DIR, reference_name(p, mode, tout))
    write_csv(out, p, mode, xs, ss, tout, x1min, x1max, args.nx, args.tol,
              args.xorder, args.integrator, args.cfl)
    print(f"shocktube_reference[{p['name']}] {mode}: t={tout:.6g} "
          f"box=[{x1min:g},{x1max:g}] nx1={args.nx} -> {len(xs)} nodes  {out}")
    return out, xs, ss, tout


def check_against_table(p, xs, ss, tout):
    """Compare a CLEAN reference against the transcribed SPLASH table, if any."""
    if not p["ref"]:
        print(f"  check: no SPLASH table for {p['name']}")
        return
    rx, rstate = st.mhd_reference(p["ref"], tout, xs.min(), xs.max())
    # Compare on a UNIFORM grid over the fan. Sampling at the decimated nodes
    # instead would weight the discontinuities, where a sub-cell position
    # mismatch swamps everything and says nothing about whether the two agree.
    reach = st.mhd_ref_reach(p["ref"], tout)
    lo = max(xs.min(), -1.2*reach)
    hi = min(xs.max(), 1.2*reach)
    xg = np.linspace(lo, hi, 4001)
    print(f"  check vs SPLASH table '{p['ref']}' at t={tout:.4g} "
          f"(uniform grid over [{lo:.3g},{hi:.3g}]):")
    for k in ("rho", "P", "vx", "vy", "By"):
        a = np.interp(xg, xs, ss[k])
        b = np.interp(xg, rx, rstate[k])
        rng = max(b.max() - b.min(), 1e-30)
        d = np.abs(a - b)
        print(f"    {k:<3s} L1={d.mean():.4g} ({100*d.mean()/rng:.2f}% of the "
              f"table's range)   max|d|={d.max():.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (box + time taken "
                                           "from each run's .log and snapshots)")
    ap.add_argument("-a", "--choice", type=int, default=None,
                    help="preset number (else inferred from the run name)")
    ap.add_argument("--dxperiod", type=float, default=None,
                    help="full periodic box length in x (default: the run's .log, "
                         "else IC_setup_shocktube's own ti*(xright-xleft))")
    ap.add_argument("--time", type=float, default=None,
                    help="reference time (default: the run's final snapshot, else "
                         "IC_setup_shocktube's tmax for the preset)")
    ap.add_argument("--mode", choices=("suite", "clean"), default="suite")
    ap.add_argument("--nx", type=int, default=16384, help="Athena++ x1 zones")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="decimation tolerance (range-normalised, default 1e-3)")
    ap.add_argument("--xorder", type=int, default=2)
    ap.add_argument("--integrator", default="vl2")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--check", action="store_true",
                    help="compare the result against the SPLASH table (--mode clean)")
    args = ap.parse_args()

    jobs = []
    if args.runs:
        for run in args.runs:
            c = args.choice if args.choice is not None else run_choice(run)
            if c is None:
                sys.exit(f"shocktube_reference: cannot infer the preset from "
                         f"'{run}' -- pass -a")
            dxp = args.dxperiod or run_dxperiod(run)
            if dxp is None:
                sys.exit(f"shocktube_reference: no dxPeriod in {run}'s .log "
                         f"-- pass --dxperiod")
            t = args.time
            if t is None:
                sel = st.select_at_time(run, None)
                if sel is None:
                    sys.exit(f"shocktube_reference: no snapshot in {run}")
                t = sel[1]
            jobs.append((st.shock_params(c), dxp, t))
    else:
        if args.choice is None:
            ap.error("without a run directory, -a is required")
        p = st.shock_params(args.choice)
        dxp, tmax = setup_box_and_time(args.choice)
        jobs.append((p, args.dxperiod or dxp, args.time or tmax))

    for p, dxp, t in jobs:
        if p["choice"] == 7:
            print(f"shocktube_reference[{p['name']}]: SKIPPED -- the C-shock is a "
                  f"continuous structure held together by AMBIPOLAR DIFFUSION; "
                  f"ideal MHD cannot reproduce it. It already has its exact "
                  f"solution (Mac-Low et al. 1995, cshock_reference).",
                  file=sys.stderr)
            continue
        if p["mhd"] and abs(p["L"][5] - p["R"][5]) > 1e-12:
            print(f"shocktube_reference[{p['name']}]: SKIPPED -- B_x jumps across "
                  f"the interface, which violates div.B = 0 in 1-D; this preset is "
                  f"not a well-posed Riemann problem for any solver.",
                  file=sys.stderr)
            continue
        _out, xs, ss, tout = generate(p, args.mode, dxp, t, args)
        if args.check:
            if args.mode != "clean":
                print("  (--check compares against the central-fan table: use "
                      "--mode clean for a like-for-like comparison)")
            check_against_table(p, xs, ss, tout)


if __name__ == "__main__":
    main()
