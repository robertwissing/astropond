"""Verify the multivoro backend: correct (conserving) mesh + faster build."""
import numpy as np, time
from sph_interp import from_tipsy, VoronoiGrid, interpolate
import sph_interp.targets as T
print("multivoro available:", T._HAS_MULTIVORO)

COLL = ('../test_cases/mhdcollapse/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')
p = from_tipsy(COLL, fields=('rho',), periodic=True)
half = 0.5 * np.asarray(p.box); bounds = [[-half[a], half[a]] for a in range(3)]
gi = np.random.default_rng(1).choice(p.n, 40000, replace=False)
gens = np.ascontiguousarray(p.pos[gi])
Mtot = p.mass.sum(); boxvol = float(np.prod(p.box))

for backend in ("pyvoro", "multivoro"):
    t0 = time.time()
    vg = VoronoiGrid.from_points(gens, bounds=bounds, periodic=True, backend=backend)
    tb = time.time() - t0
    res = interpolate(p, vg, method="petkova", values=["rho"])
    err = res.data["_mass"].sum() / Mtot - 1.0
    vsum = vg.volumes().sum()
    print("%-10s build %5.1fs  ncell=%d  vol_sum=%.6g (box=%.6g)  petkova err %+.4f%%"
          % (backend, tb, vg.ncell, vsum, boxvol, 100 * err), flush=True)
print("DONE", flush=True)
