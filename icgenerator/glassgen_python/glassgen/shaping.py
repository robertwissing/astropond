"""Target shaping: the curvature step, the one shaped target besides the raw .param profile.

A relaxed glass is force-free under a pseudo-pressure with a bump at a density feature, and the bump grows as
the feature sharpens (Db ~ A / w^2). --target-shape-errp replaces a 1D two-plateau .param contact by the
narrowest monotone step built for err_P = max|ln pPs|, the quantity IC_glasspressure.py checks. The step is a
function of err_P and the contrast only: open loop, nothing is measured on the glass here.

Calibration (scratch/interface_contrast_20261102/shape_calibration.py, sharpness-scan slab glasses): err_P =
a(C) B2^p, rms 4.5%, worst point +5.4%. Calibrated to facQf 0, sigma 0, variable neighbours (64), boundterm 0.05,
grad-h, onesided-isph, the O64 kernel and a point read with no kernel-match; other operators deliver a different
err_P. The IC build's .pPs check reports the value actually delivered.
"""
import numpy as np

from .density import TableDensity
from .params import ParamDensity, box_from_param

CAL_C = (2.0, 10.0, 100.0)
CAL_A = (0.3190, 0.4676, 0.6004)
CAL_P = 0.818
ERRP_MARGIN = 0.9            # aim below err_P by more than the fit's worst residual
ERRP_RANGE = (0.005, 0.1)    # the calibrated glasses; the smoothed .pPs noise floor is ~1.5e-3


def beta2_for_errp(errp, contrast):
    """Curvature limit B2 whose glass holds err_P <= errp at density contrast C."""
    if not ERRP_RANGE[0] <= errp <= ERRP_RANGE[1]:
        raise ValueError(f'err_P {errp:g} is outside the calibrated range {ERRP_RANGE[0]:g} .. {ERRP_RANGE[1]:g}')
    if contrast > CAL_C[-1]:
        raise ValueError(f'contrast {contrast:g} is above the calibrated {CAL_C[-1]:g}')
    a = np.interp(np.log(contrast), np.log(CAL_C), CAL_A)
    return (ERRP_MARGIN * errp / a) ** (1.0 / CAL_P)


def curvature_step(c, rho, m, nsmooth, beta2, geometry='slab'):
    """Narrowest monotone step between the two plateaus of rho (dense at small c, light at large
    c) whose curvature obeys g2 = h^2 |d^2 ln T/dc^2| <= beta2 with the local h = h_t(T): ln T
    bends down off the dense plateau at exactly beta2/h^2, then up onto the light plateau,
    switching where the two slopes match. The contact is placed so the mass equals that of rho.
    Returns (T, bend length, mass_error)."""
    assert np.all(np.diff(rho) <= 0.0), 'curvature_step: rho must fall with c'
    dc = c[1] - c[0]
    p = {'slab': 0, 'cyl': 1, 'sph': 2}[geometry]
    vol = np.abs(c) ** p * dc if p else np.full(len(c), dc)
    hfac = 0.5 * (3.0 * nsmooth * m / (4.0 * np.pi)) ** (1.0 / 3.0)
    k = beta2 / hfac ** 2                         # |s''| = k exp(2 s / 3), s = ln T
    sd, sl = np.log(rho.max()), np.log(rho.min())
    sm = 1.5 * np.log(0.5 * (np.exp(2.0 * sd / 3.0) + np.exp(2.0 * sl / 3.0)))   # equal-slope switch
    dy = dc / 20.0

    def half(s0, sign):                            # Verlet from a plateau (zero slope) to s = sm
        s, v, us, hs = s0, 0.0, [0.0], [s0]
        acc = sign * k * np.exp(2.0 * s / 3.0)
        while (s - sm) * sign < 0.0:
            v += 0.5 * dy * acc
            s += dy * v
            acc = sign * k * np.exp(2.0 * s / 3.0)
            v += 0.5 * dy * acc
            us.append(us[-1] + dy)
            hs.append(s)
        f = (sm - hs[-2]) / (hs[-1] - hs[-2])      # land exactly on sm
        us[-1], hs[-1] = us[-2] + f * dy, sm
        return np.array(us), np.array(hs)

    ud, sdn = half(sd, -1.0)                       # dense edge -> switch, s falling with y
    ul, sln = half(sl, 1.0)                        # light edge -> switch, s rising toward -y
    ys = np.concatenate([ud, ud[-1] + ul[-1] - ul[::-1][1:]])
    ss = np.concatenate([sdn, sln[::-1][1:]])

    def profile(c1):
        return np.exp(np.interp(c - c1, ys, ss, left=sd, right=sl))

    target = (rho * vol).sum()
    a, b = c[0] - ys[-1], c[-1]
    for _ in range(100):                           # mass rises with the contact position c1
        c1 = 0.5 * (a + b)
        if (profile(c1) * vol).sum() < target:
            a = c1
        else:
            b = c1
    T = profile(0.5 * (a + b))
    return T, ys[-1], (T * vol).sum() / target - 1.0


def curvature_table(p, m, nsmooth, beta2):
    """(TableDensity, c, rho, T, mass error) of the curvature step for the .param profile p. Raises ValueError
    when p is not a 1D contact with the dense side inside, or is too narrow to keep both plateaus at beta2."""
    prof, d = int(float(p.get('dICdensprofile', 1))), int(float(p.get('dICdensdir', 1)))
    inner, outer = float(p.get('dICdensinner', 1.0)), float(p.get('dICdensouter', 1.0))
    if prof not in (2, 3) or not 1 <= d <= 5 or inner <= outer:
        raise ValueError(f'the curvature step needs a 1D two-plateau contact with the dense side inside: '
                         f'dICdensprofile 2 or 3, dICdensdir 1-5, inner > outer; got profile {prof}, dir {d}, '
                         f'inner {inner:g}, outer {outer:g}')
    box = box_from_param(p)
    h_min = 0.5 * (3.0 * nsmooth * m / (4.0 * np.pi * inner)) ** (1.0 / 3.0)
    dc = min(float(p.get('dICdensRsmooth', 1.0)) or h_min, h_min) / 10.0
    if d <= 3:                 # |x|,|y|,|z| slab: [0, half-period], mirror ends
        geom, hi, mapping = 'slab', 0.5 * box[d - 1], 'xyz'[d - 1]
    else:                      # radial: out to the box corner
        geom = 'sph' if d == 4 else 'cyl'
        hi = 0.5 * float(np.sqrt((box[:3 if d == 4 else 2] ** 2).sum())) + 0.5 * float(box.max())
        mapping = 'r_sph' if d == 4 else 'r_cyl'
    n = int(np.ceil(hi / dc))
    c = (np.arange(n) + 0.5) * hi / n
    q = np.zeros((n, 3))
    q[:, {'x': 0, 'y': 1, 'z': 2, 'r_sph': 0, 'r_cyl': 0}[mapping]] = c
    rho = ParamDensity(p).rho0(q)
    T, _, merr = curvature_step(c, rho, m, nsmooth, beta2, geometry=geom)
    if abs(T[0] / rho.max() - 1.0) > 1e-6 or abs(T[-1] / rho.min() - 1.0) > 1e-6:
        raise ValueError(f'the feature is too narrow for B2 = {beta2:.4g}: the step leaves no plateau '
                         f'(rho at the ends {T[0]:.4g}, {T[-1]:.4g} for plateaus {rho.max():.4g}, {rho.min():.4g}). '
                         f'Use a larger err_P or the raw target.')
    return TableDensity(c, T, mapping=mapping), c, rho, T, merr


def curvature_target(p, m, nsmooth, errp):
    """(TableDensity, B2, mass error) of the curvature step that holds err_P = errp for the .param profile p."""
    inner, outer = float(p.get('dICdensinner', 1.0)), float(p.get('dICdensouter', 1.0))
    beta2 = beta2_for_errp(errp, inner / outer)
    table, _, _, _, merr = curvature_table(p, m, nsmooth, beta2)
    return table, beta2, merr
