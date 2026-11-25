"""SPH kernel in the gasoline convention.

q = r/h with support q in [0, 2] (gasoline BALL2 = (2h)^2, ar2 = q^2).

The kernel is the **Wendland C2** (3D), used because it has no pairing
instability at high neighbour counts: a particle whose ball the density-based-h
`nsmoothmin` floor expands toward high neighbour numbers must not clump, or the
ball path's rho->h->rho feedback runs away (see relax.py / ball.py). It needs
~64+ neighbours for good density accuracy (more than a cubic spline would).

The kernel SHAPE carries the full normalization so the density/force loops use
the gasoline fNorm = 1/(pi h^3). With support 2h the 3D Wendland C2 is

    W(r, 2h) = (21 / (16 pi h^3)) (1 - q/2)^4 (1 + 2q)

so w_wc2(ar2) = (pi h^3) * W = (21/16) (1 - q/2)^4 (1 + 2q), whose central value
w_wc2(0) = 21/16 = W0 is used by the explicit self contribution. dw_wc2 =
(1/q) dW_shape/dq = -(105/16) (1 - q/2)^3, so the gradient term is
dw_wc2(ar2) * fNorm1 * dx with fNorm1 = fNorm/h^2.
"""
import math

import numpy as np
from numba import njit

# central value of the kernel shape, w_wc2(0); used by the density/Q0 self terms.
W0 = 21.0 / 16.0

# Wendland C2 self-density bias correction Wzero10, ported verbatim from the
# piecewise-linear ladder in gasoline master.c (the Wzero10 branch, nSmooth
# 2..512): breakpoints nSmooth -> factor, linear between. The bare central value
# W0 = 21/16 over-counts a particle's own contribution, so a relaxed UNIFORM
# glass settles at rho slightly ABOVE the per-particle mean; scaling the self
# term by this factor removes the bias (uniform glass -> rho ~ 1.0). The C
# density self term is (21/16)*Wzero10 (smoothfcn.c KERNEL10).
_WZERO10_NS = np.array([2, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512],
                       dtype=np.float64)
_WZERO10_F = np.array([0.50, 0.80507812, 0.90039062, 0.91347656, 0.93349609,
                       0.95693359, 0.97158203, 0.97675781, 0.98554688,
                       0.98847656, 0.99394531], dtype=np.float64)


def wzero_wc2(nsmooth):
    """Self-density bias correction factor for the Wendland C2 self term
    (gasoline smf->Wzero10). Piecewise-linear in nSmooth over the master.c
    table (0.95693359 at nSmooth=64); the corrected self central value is
    W0 * wzero_wc2(nSmooth), so the density self term becomes
    mass_i * W0 * factor instead of mass_i * W0. Endpoints held outside
    [2, 512] (C leaves nSmooth>512 uncorrected; nSmooth defaults to 64).
    Pure-Python (evaluated once per run, not in the njit loop)."""
    return float(np.interp(nsmooth, _WZERO10_NS, _WZERO10_F))


@njit(cache=True, inline='always')
def w_wc2(ar2):
    if ar2 >= 4.0:
        return 0.0
    q = math.sqrt(ar2)
    t = 1.0 - 0.5 * q            # (1 - q/2)
    t2 = t * t
    return (21.0 / 16.0) * t2 * t2 * (1.0 + 2.0 * q)


@njit(cache=True, inline='always')
def dw_wc2(ar2):
    # (1/q) dW_shape/dq ; finite at q->0 (-105/16), and the dx factor kills the
    # self term anyway, but keep the ar2<=0 guard against a divide-by-zero.
    if ar2 >= 4.0 or ar2 <= 0.0:
        return 0.0
    q = math.sqrt(ar2)
    t = 1.0 - 0.5 * q
    return -(105.0 / 16.0) * t * t * t


# --------------------------------------------------------------------------- #
#  KERNELOPTIMIZED (Wissing et al. 2025) -- the kernel this repo's gasoline
#  builds actually use (`CODE_DEF += -DKERNELOPTIMIZED`).
#
#  Ported from master.c (which sets norm/c0..c8/Wzero per nSmooth) and
#  smoothfcn.c (which defines KERNEL/DKERNEL).  Same conventions as the
#  Wendland C2 pair above: `ar2 = (r/h)^2` with support at ar2 = 4, the shape
#  carries the polynomial's own `norm` so the loops keep fNorm = 1/(pi h^3), and
#  `dw_*` is the object that multiplies fnorm1*dx to give grad W.
#
#  These are NumPy rather than njit on purpose: they are used by the analysis
#  references (a few thousand particles, once), not by the relax force loop.
# --------------------------------------------------------------------------- #

# (upper nSmooth bound, norm, c0..c8).  master.c writes norm as
# `value/8./M_1_PI`, i.e. value/8*pi.
_OPT_SETS = (
    (32, 3.0702399143211334/8.0*np.pi,
     (-15.178969768800927, 73.51866968356427, -257.63874599152774,
      645.4937717480914, -1078.589778178365, 1171.7510466338265,
      -797.951908267712, 310.1744769215709, -52.578562780647644)),
    (96, 3.1160348893350283/8.0*np.pi,
     (-10.794751909300937, 34.923977443679455, -85.66799299517906,
      148.65227923861258, -124.78847965696686, -11.062047962891707,
      104.26365151520089, -73.55050556719418, 17.023869894040175)),
    (np.inf, 3.3158322817560615/8.0*np.pi,
     (-6.94709887927258, 0.6116847826437315, 40.605173332308,
      -109.39383705074306, 199.68569324945807, -269.18551329586074,
      230.97653909776236, -108.50616849023058, 21.153527253935298)),
)

# Wzero ladders, one per coefficient set, as piecewise-linear tables in nSmooth.
_OPT_WZERO = (
    ((2, 16, 24, 32), (0.40, 0.7765625, 0.85302734, 0.86884766)),
    ((32, 48, 64, 96), (0.93710937, 0.94462891, 0.95146484, 0.96054687)),
    ((96, 128, 192, 256, 512),
     (0.99023438, 0.99355469, 0.99746094, 0.99814453, 0.99931641)),
)


def opt_coeffs(nsmooth):
    """(norm, coefficients) for the KERNELOPTIMIZED set master.c would pick."""
    for index, (bound, norm, coeffs) in enumerate(_OPT_SETS):
        if nsmooth <= bound:
            return norm, coeffs, index
    raise AssertionError("unreachable: the last bound is inf")


def wzero_opt(nsmooth):
    """Self-density bias correction for the KERNELOPTIMIZED self term.

    The analogue of `wzero_wc2`: the bare central value over-counts a
    particle's own contribution, so the density self term is
    `mass_i * w_opt(0) * wzero_opt(nSmooth)`.  0.95146484 at nSmooth=64.
    """
    _norm, _c, index = opt_coeffs(nsmooth)
    xs, ys = _OPT_WZERO[index]
    return float(np.interp(nsmooth, xs, ys))


def w_opt(ar2, norm, coeffs):
    """Kernel shape.  `ak = norm*(1 + q^2 c0 + q^3 c1 + ... + q^10 c8)`,
    q = r/(2h) = sqrt(ar2)/2, matching smoothfcn.c's KERNEL."""
    ar2 = np.asarray(ar2, dtype=float)
    q = np.sqrt(np.maximum(ar2, 0.0))*0.5
    inner = np.zeros_like(q)
    for c in reversed(coeffs):
        inner = c + q*inner
    return np.where(ar2 >= 4.0, 0.0, norm*(1.0 + q*q*inner))


def dw_opt(ar2, norm, coeffs):
    """`(norm/4)*(2 c0 + 3 c1 q + ... + 10 c8 q^8)`, matching DKERNEL.

    Note this is NOT d(shape)/d(ar2): the code's factor of 4 (rather than 8) is
    what makes `dw * fnorm1 * dx` come out as grad W, which is how every pair
    sum consumes it."""
    ar2 = np.asarray(ar2, dtype=float)
    q = np.sqrt(np.maximum(ar2, 0.0))*0.5
    inner = np.zeros_like(q)
    for n, c in reversed(list(enumerate(coeffs))):
        inner = (n + 2)*c + q*inner
    return np.where((ar2 >= 4.0) | (ar2 <= 0.0), 0.0, (norm/4.0)*inner)


# --------------------------------------------------------------------------- #
#  Sinc kernels S_n (Cabezon, Garcia-Senz & Relano 2008)
#
#      W(r, h) = B_n / h^3 * [ sin(pi v / 2) / (pi v / 2) ]^n ,   v = r/h in [0,2]
#
#  Written here in the same gasoline convention as the pairs above: `ar2 =
#  (r/h)^2` with support at ar2 = 4, the shape carries its own normalization so
#  the loops keep fNorm = 1/(pi h^3), and `dw_*` is `(1/q) dW_shape/dq`.
#
#  With u = q/2 = r/(2h) the sinc argument is just numpy's sinc, sinc(u) =
#  sin(pi u)/(pi u), so
#
#      shape(u) = sinc(u)^n / (32 I_n),      I_n = int_0^1 sinc(u)^n u^2 du
#
#  which makes int W d3r = 32 int_0^1 u^2 shape du = 1 by construction.  n is a
#  free parameter: larger n is a narrower, more peaked kernel needing more
#  neighbours.  n = 6 is the exponent used for the lattice-stability comparison
#  in test_cases/cosmowave/notes/.
# --------------------------------------------------------------------------- #
SINC_N = 6


def _sinc_moment(n, nq=256):
    """I_n = int_0^1 sinc(u)^n u^2 du, by Gauss-Legendre (smooth integrand)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    u = 0.5 * (x + 1.0)
    return 0.5 * float(np.sum(w * np.sinc(u)**n * u * u))


_SINC_I = {SINC_N: _sinc_moment(SINC_N)}


def sinc_shape_norm(n=SINC_N):
    """1/(32 I_n) -- the shape's leading constant, and its central value W0."""
    if n not in _SINC_I:
        _SINC_I[n] = _sinc_moment(n)
    return 1.0 / (32.0 * _SINC_I[n])


def w_sinc(ar2, n=SINC_N):
    """Kernel shape, the analogue of w_wc2/w_opt."""
    ar2 = np.asarray(ar2, dtype=float)
    u = np.sqrt(np.maximum(ar2, 0.0)) * 0.5
    return np.where(ar2 >= 4.0, 0.0, sinc_shape_norm(n) * np.sinc(u)**n)


def dw_sinc(ar2, n=SINC_N):
    """`(1/q) dW_shape/dq`, the analogue of dw_wc2/dw_opt.

    shape'(u) = n sinc^(n-1) s'(u) with s'(u) = (cos(pi u) - sinc(u))/u, and
    (1/q) d/dq = shape'(u)/(4u), so the u^2 in the denominator is cancelled by
    the O(u^2) zero of (cos - sinc).  Finite at the origin, value
    -n pi^2 /(384 I_n)."""
    ar2 = np.asarray(ar2, dtype=float)
    u = np.sqrt(np.maximum(ar2, 0.0)) * 0.5
    small = u < 1e-4
    us = np.where(small, 1.0, u)                      # keep the divide finite
    num = np.cos(np.pi * us) - np.sinc(us)            # = O(u^2), -pi^2 u^2/3
    val = sinc_shape_norm(n) * n * np.sinc(us)**(n - 1) * num / (4.0 * us * us)
    val = np.where(small, -sinc_shape_norm(n) * n * np.pi**2 / 12.0, val)
    return np.where((ar2 >= 4.0) | (ar2 <= 0.0), 0.0, val)


# --------------------------------------------------------------------------- #
#  Kernel registry for the njit loops: relax(kernel=...), density/force/ISPH.
#  w_k/dw_k reproduce gasoline_claude's KERNEL (ar2 > 0) / DKERNEL macros for
#  the kernels it builds (WENDLAND, WENDLANDC4, KERNELOPTIMIZED); kernel_spec's
#  self term is its smf->Wzero. sinc (S_n above) has no gasoline counterpart
#  and gets no self-density correction.
# --------------------------------------------------------------------------- #
KERNELS = ('wendlandc2', 'wendlandc4', 'kerneloptimized', 'sinc')
_KID = {name: i for i, name in enumerate(KERNELS)}
_NKP = 10
KP0 = np.zeros(_NKP)


def kernel_spec(kernel='wendlandc2', nsmooth=64, self_bias=True):
    """(kid, kp, w0self, w0) for the njit loops: kernel id, parameter array
    (kerneloptimized: norm, c0..c8; sinc: n, shape norm), the density self term
    (central value times the self-density bias correction when self_bias) and
    the bare central value."""
    if kernel not in _KID:
        raise ValueError(f'unknown kernel {kernel!r}, have {KERNELS}')
    kp = np.zeros(_NKP)
    fac = 1.0
    if kernel == 'wendlandc2':
        w0 = 21.0 / 16.0
        fac = 1.0 - 0.0294 * (nsmooth * 0.01) ** -0.977
    elif kernel == 'wendlandc4':
        w0 = 495.0 / 256.0
        fac = 1.0 - 0.01342 * (nsmooth * 0.01) ** -1.579
    elif kernel == 'kerneloptimized':
        norm, coeffs, _ = opt_coeffs(nsmooth)
        kp[0] = norm
        kp[1:10] = coeffs
        w0 = norm
        fac = wzero_opt(nsmooth) if nsmooth <= 512 else 1.0
    else:
        kp[0] = SINC_N
        kp[1] = sinc_shape_norm(SINC_N)
        w0 = kp[1]
    return _KID[kernel], kp, (w0 * fac if self_bias else w0), w0


@njit(cache=True, inline='always')
def w_k(kid, kp, ar2):
    """Kernel shape for ar2 = (r/h)^2 > 0, support ar2 < 4 (self term:
    kernel_spec's w0self)."""
    if ar2 >= 4.0:
        return 0.0
    if kid == 0:
        return w_wc2(ar2)
    au = 0.5 * math.sqrt(ar2)
    if kid == 1:
        t = 1.0 - au
        t3 = t * t * t
        return (495.0 / 256.0) * t3 * t3 * (1.0 + 6.0 * au + (35.0 / 3.0) * au * au)
    if kid == 2:
        inner = kp[9]
        for k in range(8, 0, -1):
            inner = kp[k] + au * inner
        return kp[0] * (1.0 + au * au * inner)
    if au <= 0.0:
        return kp[1]
    s = math.sin(math.pi * au) / (math.pi * au)
    return kp[1] * s ** kp[0]


@njit(cache=True, inline='always')
def dw_k(kid, kp, ar2):
    """(1/q) dW/dq, the factor that multiplies fnorm1 * dx to give grad W."""
    if ar2 >= 4.0 or ar2 <= 0.0:
        return 0.0
    if kid == 0:
        return dw_wc2(ar2)
    au = 0.5 * math.sqrt(ar2)
    if kid == 1:
        t = 1.0 - au
        t2 = t * t
        return -(495.0 / 32.0 * 7.0 / 3.0 / 4.0) * t2 * t2 * t * (1.0 + 5.0 * au)
    if kid == 2:
        inner = 10.0 * kp[9]
        for n in range(7, -1, -1):
            inner = (n + 2) * kp[n + 1] + au * inner
        return 0.25 * kp[0] * inner
    n = kp[0]
    if au < 1e-4:
        return -kp[1] * n * math.pi * math.pi / 12.0
    s = math.sin(math.pi * au) / (math.pi * au)
    return kp[1] * n * s ** (n - 1.0) * (math.cos(math.pi * au) - s) / (4.0 * au * au)
