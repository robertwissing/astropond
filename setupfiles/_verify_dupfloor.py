"""Verify: merge_tol=0 now BUILDS on duplicate-laden data (merger) and is a
no-op where there are no duplicates (collapse)."""
import numpy as np, time
from sph_interp import from_tipsy, VoronoiGrid

COLL = ('../test_cases/mhdcollapse/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')
MRG = ('/mn/stornext/d17/extragalactic/personal/robertwi/Projects/ICGEN/sims/'
       'OPTKERNELSIMS/testing_ground/isolated_galaxy_merger/PGALMHDJ8MERGb0/'
       'PGALMHDJ8changaMERGb0.00000')


def build(path, periodic, tol):
    p = from_tipsy(path, fields=('rho',), periodic=periodic)
    gi = np.random.default_rng(1).choice(p.n, 40000, replace=False)
    gens = np.ascontiguousarray(p.pos[gi])
    if periodic:
        half = 0.5*np.asarray(p.box); bnds = [[-half[a], half[a]] for a in range(3)]
    else:
        plo = gens.min(0); phi = gens.max(0); pad = 0.05*(phi-plo)
        bnds = [[plo[a]-pad[a], phi[a]+pad[a]] for a in range(3)]
    t0 = time.time()
    try:
        vg = VoronoiGrid.from_points(gens, bounds=bnds, periodic=periodic, merge_tol=tol)
        return 'OK ncell=%d (merged %d) [%.0fs]' % (vg.ncell, 40000-vg.ncell, time.time()-t0)
    except Exception as e:
        return 'CRASH: %s [%.0fs]' % (str(e)[:55], time.time()-t0)


print('COLLAPSE merge_tol=0 :', build(COLL, True, 0.0), flush=True)
print('MERGER   merge_tol=0 :', build(MRG, False, 0.0), flush=True)
print('MERGER   default     :', build(MRG, False, None), flush=True)
