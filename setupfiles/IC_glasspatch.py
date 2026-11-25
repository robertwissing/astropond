"""Glass patch (runtest.sh distrib 4): a sharp density contrast from two uniform periodic glasses relaxed apart and
pasted, so no particle was relaxed against the other side.

  IC_glasspatch.py pre   <prefile> <glass>   from datafiles/<prefile>.{00000,param}: random uniform pre-ICs
                                             <glass>_{inner,outer}.pre and their boxes <glass>_{inner,outer}.box
  IC_glasspatch.py paste <prefile> <glass>   pastes <glass>_{inner,outer}_IC into <glass>_IC

Only the dense region's shape and the two densities come from the .param. The setup's target smoothing
(dICdensRsmooth, the tanh of profile 3) is ignored: the contrast is always sharp. The inner glass fills a periodic box
equal to the dense region; the outer glass fills the whole box and the dense region is carved out of it. The rho
column keeps each glass's own density, so a setup with u = P / ((gamma-1) rho) gets a sharp entropy step at uniform P.
"""
import sys

import numpy as np
from glassgen.params import box_from_param, read_param

import readtipsy as tip


def geometry(p):
    """(dICdensdir, R, rho_in, rho_out, box, inner box). dICdensdir 1-3: slab |x_d| < R; 0: square/cube, all |x_i| <= R."""
    prof, d = int(float(p.get('dICdensprofile', 1))), int(float(p.get('dICdensdir', 1)))
    R, inner, outer = float(p.get('dICdensR', 0.0)), float(p.get('dICdensinner', 1.0)), float(p.get('dICdensouter', 1.0))
    if prof not in (2, 3) or d not in (0, 1, 2, 3) or inner <= outer or R <= 0.0:
        raise ValueError(f'glass patch needs a dense slab or square: dICdensprofile 2 or 3, dICdensdir 0-3, inner > outer, '
                         f'R > 0; got profile {prof}, dir {d}, inner {inner:g}, outer {outer:g}, R {R:g}')
    box = box_from_param(p)
    ibox = np.minimum(box, 2.0 * R)
    if d > 0:
        ibox = box.copy()
        ibox[d - 1] = 2.0 * R
    return d, R, inner, outer, box, ibox


def inside(pos, d, R):
    """True for positions in the dense region."""
    if d == 0:
        return np.abs(pos).max(axis=1) <= R
    return np.abs(pos[:, d - 1]) < R


def make_pre(p, m, glass):
    """Write the two random uniform pre-ICs for particle mass m, and their boxes."""
    _, _, inner, outer, box, ibox = geometry(p)
    for name, rho, b, seed in (('inner', inner, ibox, 1), ('outer', outer, box, 2)):
        n = int(round(rho * np.prod(b) / m))
        g = np.zeros((n, 12))
        g[:, 0] = rho * np.prod(b) / n
        g[:, 1:4] = np.random.default_rng(seed).uniform(-0.5 * b, 0.5 * b, (n, 3))
        g[:, 7], g[:, 8] = rho, 1.0
        g[:, 9] = 0.5 * (3 * 64 * g[0, 0] / (4 * np.pi * rho)) ** (1 / 3)
        tip.writetipsy(g, np.zeros((0, 9)), np.zeros((0, 11)), f'{glass}_{name}.pre', np.array([n, 3, n, 0, 0, 0]),
                       np.array([0.0]))
        with open(f'{glass}_{name}.box', 'w') as f:
            f.write(' '.join(repr(float(x)) for x in b) + '\n')
        print(f'glass patch {name}: uniform rho {rho:g} in box {b}: N {n}, m {g[0, 0]:.7e}')


def paste(p, glass):
    """Paste <glass>_inner_IC and the carved <glass>_outer_IC into <glass>_IC."""
    d, R, inner, outer, box, ibox = geometry(p)
    gi = np.array(tip.readtipsy(f'{glass}_inner_IC')[0], dtype=np.float64)
    go = np.array(tip.readtipsy(f'{glass}_outer_IC')[0], dtype=np.float64)
    gi[:, 1:4] -= ibox * np.rint(gi[:, 1:4] / ibox)
    go[:, 1:4] -= box * np.rint(go[:, 1:4] / box)
    go = go[~inside(go[:, 1:4], d, R)]
    g = np.vstack([gi, go])
    g[:, 4:7] = 0.0
    n = len(g)
    tip.writetipsy(g, np.zeros((0, 9)), np.zeros((0, 11)), f'{glass}_IC', np.array([n, 3, n, 0, 0, 0]), np.array([0.0]))
    print(f'glass patch {glass}_IC: sharp contact (dICdensRsmooth {p.get("dICdensRsmooth", "unset")} ignored): '
          f'dense {len(gi)} (rho {np.median(gi[:, 7]):.4g}) + light {len(go)} (rho {np.median(go[:, 7]):.4g}); '
          f'masses {gi[0, 0]:.7e} / {go[0, 0]:.7e}')


def main():
    step, prefile, glass = sys.argv[1:4]
    p = read_param(f'datafiles/{prefile}.param')
    try:
        if step == 'pre':
            gas = tip.readtipsy(f'datafiles/{prefile}.00000')[0]
            make_pre(p, float(gas[:, 0].astype(np.float64).sum()) / len(gas), glass)
        else:
            paste(p, glass)
    except ValueError as err:
        sys.exit(f'IC_glasspatch: {err}')


if __name__ == '__main__':
    main()
