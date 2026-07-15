"""Profile Voronoi build vs deposit scaling, separately in (a) generator count
and (b) particle count, to see which bottleneck to attack for many generators."""
import numpy as np, time
from dataclasses import replace
from sph_interp import from_tipsy, VoronoiGrid, interpolate
from sph_interp.targets import VoronoiGrid as VG

COLL = ('../test_cases/mhdcollapse/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')

p_full = from_tipsy(COLL, fields=('rho',), periodic=True)
half = 0.5 * np.asarray(p_full.box)
bounds = [[-half[a], half[a]] for a in range(3)]
N = p_full.n
print("collapse (periodic), particles N=%d" % N, flush=True)


def subset(p, n):
    if n >= p.n:
        return p
    i = np.random.default_rng(2).choice(p.n, n, replace=False)
    return replace(p, pos=np.ascontiguousarray(p.pos[i]), mass=p.mass[i],
                   rho=p.rho[i], h=p.h[i],
                   values={k: v[i] for k, v in p.values.items()})


def build(ng):
    gi = np.random.default_rng(1).choice(N, min(ng, N), replace=False)
    gens = np.ascontiguousarray(p_full.pos[gi])
    t0 = time.time()
    vg = VoronoiGrid.from_points(gens, bounds=bounds, periodic=True)
    return vg, time.time() - t0


def dep(p, vg, method):
    t0 = time.time()
    interpolate(p, vg, method=method, values=["rho"])
    return time.time() - t0


# warm up numba
vg0, _ = build(5000); dep(subset(p_full, 5000), vg0, "petkova"); dep(subset(p_full, 5000), vg0, "sph")

print("\n=== (a) vary GENERATORS (deposit ALL %d particles) ===" % N, flush=True)
print("%8s | %8s | %9s | %9s | %s" % ("Ngen", "build", "petkova", "sph", "ncell"), flush=True)
for ng in [20000, 40000, 80000, 160000]:
    vg, tb = build(ng)
    tp = dep(p_full, vg, "petkova"); ts = dep(p_full, vg, "sph")
    print("%8d | %7.1fs | %8.1fs | %8.1fs | %d" % (ng, tb, tp, ts, vg.ncell), flush=True)

print("\n=== (b) vary PARTICLES (fixed Ngen=40000) ===", flush=True)
vg, tb = build(40000)
print("build(40k) = %.1fs, ncell=%d" % (tb, vg.ncell), flush=True)
print("%9s | %9s %8s | %9s %8s" % ("Npart", "petkova", "p/s", "sph", "p/s"), flush=True)
for npart in [30000, 60000, 115075]:
    ps = subset(p_full, npart)
    tp = dep(ps, vg, "petkova"); ts = dep(ps, vg, "sph")
    print("%9d | %8.1fs %7.0f | %8.1fs %7.0f" % (npart, tp, ps.n/tp, ts, ps.n/ts), flush=True)
print("DONE", flush=True)
