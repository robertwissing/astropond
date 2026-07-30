#!/usr/bin/env python
"""Interp perf/budget regression bench for `sph_interp` (SPH_INTERP_PLAN task 8).

The conefix 20x petkova regression shipped unnoticed because nothing timed the
deposits. This script times mesh BUILD and INTERP per (dataset x geometry x
method x backend) and verifies the conservation budgets, so a perf or accuracy
regression shows up as a diffable number instead of a slow demo.

Datasets -- ALL SELF-GENERATED, no snapshot files (changed 2026-07-28; the
canonical mhdcollapse snapshots are not on this machine and are NOT being
restored, so the bench now stands on `sph_interp.synthetic` instead):
  clustered the historical baseline set: two Gaussian clumps + sparse halo in a
            periodic box (seeded; h from cKDTree kNN, rho consistent). KEPT
            unchanged -- every stored perf number in SPH_INTERP_PLAN task 14
            is on this set, so replacing it would break comparability.
  <family>  any of `sph_interp.synthetic.FAMILIES`, calibrated against the real
            ICs in test_cases/isolated_galaxy (plan task 15):
              uniform lattice closepacked glass slab1d slab2d
              disk merger cosmo collapse
            `glass` replaces the old `early` (mild contrast, near-uniform) and
            `collapse` replaces `core` (high dynamic range). disk/merger/cosmo
            are OPEN captures and exercise the padded-domain path the snapshot
            datasets never did.

Budgets asserted by --check (timings are informational, never asserted):
  sph-scatter deposited mass  == total particle mass   (rel tol 1e-12)
  petkova     deposited mass  == total particle mass   (uniform/AMR 1e-8;
              Voronoi 1e-4 -- hull-face reconstruction floor, see plan task 6)
  Voronoi     Sum(cell volume) == box volume           (ghost path: the
              mesher noise tol from targets; arepo native 1e-5)
  sph-gather  is NOT mass-conserving by construction: reported, never checked.

Usage:
  python _bench_interp.py                       # full table, default datasets
  python _bench_interp.py --check               # exit 1 on any budget breach
  python _bench_interp.py --json out.json       # machine-readable, for diffing
  python _bench_interp.py --quick               # small sizes (smoke/CI)
  python _bench_interp.py --datasets clustered --backends multivoro,pyvoro
  python _bench_interp.py --datasets disk,merger,cosmo --check   # open domains
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp import (interpolate, UniformGrid, AMRGrid, VoronoiGrid,
                        synthetic)
from sph_interp.particles import Particles
from sph_interp.targets import _resolve_backend
try:
    from sph_interp.targets import _VOL_BUDGET_TOL
except ImportError:                      # older targets.py: pre-noise-tol check
    _VOL_BUDGET_TOL = 1e-9
# Domain bounds live in ONE place: the scaling bench owns `domain_bounds`
# (periodic -> the particle box; open -> `open_domain_bounds(rho_edge_pad)`)
# and `_npx_for` (cubic cells on a thin box). Importing them rather than
# re-deriving keeps the two benches from drifting apart -- this bench used to
# hand-roll a 2% pad on non-periodic input, which is exactly the mistake that
# made `merger` petkova report -3.1e-3.
from _bench_scaling import domain_bounds, _npx_for

_HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ("clustered",) + synthetic.FAMILIES
DEFAULT_DATASETS = "clustered,glass,collapse"

GEOMETRIES = ("uniform", "amr", "voronoi")
METHODS = ("sph-gather", "sph-scatter", "petkova")

# relative |deposited/expected - 1| ceilings; sph-gather is non-conserving and
# never checked. The Voronoi petkova floor is the mesher's reconstructed-face
# mismatch (arepo +3e-5, multivoro +4e-6 measured on the 40k merger core).
MASS_TOL = {
    ("sph-scatter", "uniform"): 1e-12,
    ("sph-scatter", "amr"): 1e-12,
    ("sph-scatter", "voronoi"): 1e-12,
    ("petkova", "uniform"): 1e-8,
    ("petkova", "amr"): 1e-8,
    ("petkova", "voronoi"): 1e-4,
}
# OPEN captures relax the petkova ceiling: `rho_edge_pad` sizes the pad from
# the FAINT-END h rather than max(h), so a little kernel support crosses the
# wall and a conserving deposit legitimately loses it. Measured on `merger`,
# -1.14e-6 at N=12.5k falling to -4.6e-8 at N=50k. This is physics leaving an
# open boundary, not a defect -- `--open-pad full_pad` in _bench_scaling.py
# restores machine precision at 1.7x the domain volume. sph-scatter keeps its
# 1e-12: it deposits to the nearest cell and never straddles the wall.
MASS_TOL_OPEN = {"petkova": 1e-5}
VOL_TOL = {"pyvoro": _VOL_BUDGET_TOL, "multivoro": _VOL_BUDGET_TOL,
           "arepo": 1e-5}


def mass_tol(method, geometry, periodic):
    if not periodic and method in MASS_TOL_OPEN:
        return max(MASS_TOL_OPEN[method], MASS_TOL.get((method, geometry), 0.0))
    return MASS_TOL.get((method, geometry))


def make_clustered(n, seed=42):
    """Merger-like particle set in a periodic unit box centred at 0: two
    Gaussian clumps (45% + 35% of the particles, sigma = box/40) and a sparse
    uniform halo (20%). Smoothing lengths from the kNN distance (h = d_k/2,
    the 2h-compact convention) and densities consistent with them:
    rho = m*k / (4pi/3 (2h)^3)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n1 = int(0.45 * n); n2 = int(0.35 * n); n3 = n - n1 - n2
    sig = 1.0 / 40.0
    pos = np.concatenate([
        rng.normal([-0.15, 0.0, 0.0], sig, (n1, 3)),
        rng.normal([+0.15, 0.0, 0.0], sig, (n2, 3)),
        rng.uniform(-0.5, 0.5, (n3, 3)),
    ])
    pos = (pos + 0.5) % 1.0 - 0.5                      # min-image wrap into box
    k = 32
    d, _ = cKDTree(pos).query(pos, k=k + 1, workers=-1)
    h = 0.5 * d[:, -1]
    mass = np.full(n, 1.0 / n)
    rho = mass * k / (4.0 * np.pi / 3.0 * (2.0 * h) ** 3)
    return Particles(pos=pos, mass=mass, rho=rho, h=h,
                     values={"rho": rho}, box=np.ones(3), periodic=True)


def load_dataset(name, args):
    n = 20000 if args.quick else args.nparticles
    if name == "clustered":
        return make_clustered(n)
    return synthetic.make(name, n, seed=42)


def subsample(p, n, seed=0):
    """A smaller Particles view for JIT warm-up (h/rho kept as-is: warm-up only
    exercises the compiled kernels, the numbers are thrown away)."""
    if p.n <= n:
        return p
    idx = np.random.default_rng(seed).choice(p.n, n, replace=False)
    return Particles(pos=p.pos[idx], mass=p.mass[idx], rho=p.rho[idx],
                     h=p.h[idx], values={k: v[idx] for k, v in p.values.items()},
                     box=p.box, periodic=p.periodic)


def build_target(geometry, p, args, backend=None, warmup=False):
    lo, hi = domain_bounds(p)
    span = hi - lo
    if geometry == "uniform":
        npx = 16 if warmup else args.npx
        return UniformGrid(lo=lo, hi=hi, npx=_npx_for(span, npx))
    if geometry == "amr":
        if warmup:
            return AMRGrid.build(p, base_npx=4, max_level=2, lo=lo, hi=hi)
        return AMRGrid.build(p, base_npx=args.base_npx,
                             max_level=args.max_level, lo=lo, hi=hi)
    if geometry == "voronoi":
        ngen = 1500 if warmup else args.voronoi_ncells
        if ngen and ngen < p.n:
            idx = np.random.default_rng(0).choice(p.n, ngen, replace=False)
            pts = p.pos[idx]
        else:
            pts = p.pos
        per = p.periodic if any(np.atleast_1d(p.periodic)) else False
        bounds = [[lo[a], hi[a]] for a in range(3)]
        return VoronoiGrid.from_points(pts, bounds=bounds, periodic=per,
                                       backend=effective_backend(backend, p))
    raise ValueError(geometry)


def effective_backend(backend, p):
    """Resolve `auto` BY DOMAIN TYPE. arepo is the only backend with native
    per-axis periodicity, so it keeps the periodic sets; but on a padded OPEN
    domain its guarded-open path fails its own volume budget (+9.3e-5 against
    1e-5 -- the float32 ConvexHull floor amplified by the cell-size spread a
    pad creates; see plan task 6), and `auto` prefers arepo and so would refuse
    outright rather than fall back. An explicit --backends choice still wins."""
    if backend in (None, "auto") and not any(np.atleast_1d(p.periodic)):
        return "multivoro"
    return backend or "auto"


def deposited_mass(res, grid):
    """Total mass the deposit put on the mesh (see plan task 8 notes).
    sph-scatter: norm IS the accumulated cell mass. petkova: the engines
    expose `_mass` (conserved per-cell mass); fall back to rho*V, which
    telescopes to the same thing. sph-gather: rho*V (NOT conserved --
    informational)."""
    if res.method == "sph-scatter":
        return float(res.norm.sum())
    if "_mass" in res.data:
        return float(np.asarray(res.data["_mass"]).sum())
    rho = np.asarray(res.data["rho"], dtype=np.float64)
    if isinstance(grid, UniformGrid):
        return float(rho.sum() * grid.cell_volume)
    return float((rho.ravel() * grid.volumes()).sum())


def volume_error(grid, p):
    """Voronoi volume budget: Sum(central cell vol) / the TARGET's own domain
    volume - 1.

    The reference is `grid.lo/hi`, not the particle box. An open mesh is still
    a closed region -- it is built on padded bounds plus a background guard
    lattice, and it tiles THOSE bounds. Comparing it to the particle box would
    report a false error, which is why this check used to bail out on anything
    non-periodic and so never ran on an open capture at all. For a periodic
    mesh grid.lo/hi IS the box, so the periodic numbers are unchanged."""
    if not isinstance(grid, VoronoiGrid):
        return None
    span = np.asarray(grid.hi, dtype=np.float64) - np.asarray(grid.lo,
                                                              dtype=np.float64)
    return float(grid.volumes().sum()) / float(np.prod(span)) - 1.0


def ncells(grid):
    if isinstance(grid, UniformGrid):
        return int(np.prod(grid.npx))
    if isinstance(grid, AMRGrid):
        return int(grid.ncell)
    return int(grid.ncentral)


def run_bench(args):
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    rows, warmed = [], set()
    for ds in args.datasets:
        p = load_dataset(ds, args)
        per = bool(all(np.atleast_1d(p.periodic)))
        mtot = float(p.mass.sum())
        pw = subsample(p, 5000)
        print(f"# dataset {ds}: N={p.n} box={np.asarray(p.box).tolist()} "
              f"periodic={p.periodic} Mtot={mtot:.6g}", file=sys.stderr)
        for geometry in args.geometries:
            geo_backends = backends if geometry == "voronoi" else [None]
            for be in geo_backends:
                try:
                    t0 = time.perf_counter()
                    grid = build_target(geometry, p, args, backend=be)
                    t_build = time.perf_counter() - t0
                except Exception as e:
                    print(f"[skip] {ds}/{geometry}"
                          f"{'/' + be if be else ''}: build failed: {e}",
                          file=sys.stderr)
                    continue
                verr = volume_error(grid, p)
                for method in args.methods:
                    # JIT warm-up: tiny mesh + 5k particles, once per
                    # (geometry, method) -- numba cache=True still pays
                    # first-load in a fresh process, which would otherwise
                    # land in the first timed call.
                    if (geometry, method) not in warmed:
                        gw = build_target(geometry, pw, args, backend=be,
                                          warmup=True)
                        interpolate(pw, gw, method=method, values=["rho"])
                        warmed.add((geometry, method))
                    t0 = time.perf_counter()
                    res = interpolate(p, grid, method=method, values=["rho"])
                    t_interp = time.perf_counter() - t0
                    merr = deposited_mass(res, grid) / mtot - 1.0
                    tol = mass_tol(method, geometry, per)
                    ebe = effective_backend(be, p) if be else None
                    vtol = (VOL_TOL[_resolve_backend(ebe or "auto")]
                            if verr is not None else None)
                    rows.append({
                        "dataset": ds, "geometry": geometry,
                        "periodic": per,
                        "backend": (_resolve_backend(ebe) if ebe else None),
                        "method": method, "ncell": ncells(grid),
                        "t_build": round(t_build, 4),
                        "t_interp": round(t_interp, 4),
                        "mass_err": merr, "mass_tol": tol,
                        "vol_err": verr, "vol_tol": vtol,
                        "mass_ok": (None if tol is None
                                    else bool(abs(merr) <= tol)),
                        "vol_ok": (None if verr is None
                                   else bool(abs(verr) <= vtol)),
                    })
                    r = rows[-1]
                    print(_fmt_row(r), flush=True)
    return rows


_HDR = (f"{'dataset':<10} {'geometry':<8} {'backend':<10} {'method':<12} "
        f"{'ncell':>8} {'build_s':>8} {'interp_s':>9} "
        f"{'mass_err':>10} {'vol_err':>10}  budget")


def _fmt_row(r):
    def fe(x):
        return "-" if x is None else f"{x:+.2e}"
    if r["mass_ok"] is None and r["vol_ok"] is None:
        budget = "info"
    else:
        checked = [ok for ok in (r["mass_ok"], r["vol_ok"]) if ok is not None]
        budget = "ok" if all(checked) else "FAIL"
    return (f"{r['dataset']:<10} {r['geometry']:<8} "
            f"{r['backend'] or '-':<10} {r['method']:<12} "
            f"{r['ncell']:>8d} {r['t_build']:>8.3f} {r['t_interp']:>9.3f} "
            f"{fe(r['mass_err']):>10} {fe(r['vol_err']):>10}  {budget}")


def main(argv=None):
    pa = argparse.ArgumentParser(
        prog="_bench_interp",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--datasets", default=DEFAULT_DATASETS,
                    help="comma list of: " + ", ".join(DATASETS))
    pa.add_argument("--geometries", default=",".join(GEOMETRIES),
                    help="comma list of: uniform, amr, voronoi")
    pa.add_argument("--methods", default=",".join(METHODS),
                    help="comma list of: sph-gather, sph-scatter, petkova")
    pa.add_argument("--backends", default="auto",
                    help="Voronoi mesher backends to bench (comma list of "
                         "auto/pyvoro/multivoro/arepo). NOTE pyvoro is "
                         "single-threaded (~minutes at 40k clustered gens)")
    pa.add_argument("--npx", type=int, default=128,
                    help="uniform grid cells per axis")
    pa.add_argument("--base-npx", type=int, default=16, help="AMR base grid")
    pa.add_argument("--max-level", type=int, default=3,
                    help="AMR max refinement level (finest = base * 2**level)")
    pa.add_argument("--voronoi-ncells", type=int, default=40000,
                    help="Voronoi generators (subsampled); 0 = all particles")
    pa.add_argument("--nparticles", "--nclustered", type=int, default=200000,
                    dest="nparticles",
                    help="particle count for every generated dataset")
    pa.add_argument("--quick", action="store_true",
                    help="small sizes for a fast smoke run (npx 32, AMR base 8 "
                         "level 2, 4000 generators, 20k clustered particles)")
    pa.add_argument("--json", metavar="PATH",
                    help="also write results as JSON (for diffing runs)")
    pa.add_argument("--check", action="store_true",
                    help="exit 1 if any conservation budget fails "
                         "(timings are informational only)")
    args = pa.parse_args(argv)

    args.datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    args.geometries = [s.strip() for s in args.geometries.split(",") if s.strip()]
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    for ds in args.datasets:
        if ds not in DATASETS:
            pa.error(f"unknown dataset {ds!r} (choose from {', '.join(DATASETS)})")
    if args.quick:
        args.npx = 32
        args.base_npx = 8
        args.max_level = 2
        args.voronoi_ncells = 4000

    print(_HDR)
    rows = run_bench(args)

    if args.json:
        payload = {
            "meta": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "host": platform.node(),
                "ncpu": os.cpu_count(),
                "quick": bool(args.quick),
                "argv": sys.argv[1:],
            },
            "rows": rows,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"# wrote {args.json}", file=sys.stderr)

    if args.check:
        bad = [r for r in rows
               if r["mass_ok"] is False or r["vol_ok"] is False]
        if bad:
            print(f"\nBUDGET FAILURES ({len(bad)}):")
            for r in bad:
                print("  " + _fmt_row(r))
            return 1
        checked = sum(1 for r in rows
                      if r["mass_ok"] is not None or r["vol_ok"] is not None)
        print(f"\nall {checked} checked budgets ok ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
