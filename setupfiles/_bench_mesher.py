"""Head-to-head raw build time: pyvoro (single-thread) vs multivoro (n threads).
Non-periodic, same generators, just the tessellation call."""
import numpy as np, time
import pyvoro, multivoro
from sph_interp import from_tipsy

COLL = ('../test_cases/mhdcollapse/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')
p = from_tipsy(COLL, fields=('rho',), periodic=False)
half = 0.5 * np.asarray(p.box)


def gens(ng):
    gi = np.random.default_rng(1).choice(p.n, min(ng, p.n), replace=False)
    return np.ascontiguousarray(p.pos[gi])


for ng in [40000, 115075]:
    g = gens(ng)
    lo = g.min(0) - 0.02 * (g.max(0) - g.min(0))
    hi = g.max(0) + 0.02 * (g.max(0) - g.min(0))
    print("\n=== Ngen=%d ===" % len(g), flush=True)

    disp = float((np.prod(hi - lo) / len(g)) ** (1 / 3))
    t0 = time.time()
    pc = pyvoro.compute_voronoi(g.tolist(), [[lo[a], hi[a]] for a in range(3)],
                                disp, radii=[], periodic=[False]*3)
    tp = time.time() - t0
    print("pyvoro (1 thread)      : %6.1fs  (%d cells)" % (tp, len(pc)), flush=True)

    lim = np.array([lo, hi])
    r = np.zeros(len(g))
    for nt in [1, 8, 32, 128]:
        t0 = time.time()
        mc = multivoro.compute_voronoi(g, limits=lim, radii=r,
                                       periodic_boundaries=(False, False, False),
                                       n_threads=nt)
        tm = time.time() - t0
        print("multivoro (%3d threads): %6.1fs  (%d cells)  speedup vs pyvoro %.1fx"
              % (nt, tm, len(mc), tp / tm), flush=True)
print("\nDONE", flush=True)
