"""Petkova (2018) exact interpolation: integrate the kernel over each cell.

Mass-conserving alternative to the regular SPH deposit: instead of evaluating the
kernel at the cell centre, integrate the (cubic-spline) kernel over the cell
volume analytically (per particle, per overlapping cell), so the deposited mass
equals the particle mass to high precision.

`_vertex_integral` / `_compute_logs` are cleaned ports of the math core from
`interpolate_petkova.py` (guards instead of try/except, no origin hardwiring, no
`/5.0`). `_cube_integral` applies the per-face/edge construction of
`exact_interpolate` to an axis-aligned cube with the PARTICLE at the origin,
returning the fraction of the (unit-normalized) kernel mass contained in the cube.

The per-triangle cone integral `_cone_tri` is REGIME-DISPATCHED (2026-07-02,
see its block comment + PLAN_PETKOVA_FIX.md): exact solid angle when saturated,
calibrated-order Duffy quadrature when converged, and an exact analytic-radial
wedge decomposition for scale-separated triangles; cell integrals additionally
take a closed-form regime-0 shortcut when the kernel crosses at most one face
plane (h << cell). Far cells (entirely beyond 2h) are skipped.

VERIFIED mass-conserving: a single particle deposits its full mass for any npx;
on a real orzag snapshot the summed grid mass matches the particle mass to 0.00%
(vs ~0.5% SPH diffusion). Two things are essential and easy to get wrong:
  * the cube vertex numbering + face windings MUST match pyvoro's convention
    (see `_CUBE_FACES`) — `vertex_integral`'s sign logic depends on it;
  * the per-particle vertex scratch arrays MUST be allocated inside the prange
    loop (thread-private), else parallel threads corrupt each other.
"""

import math
import os
import numpy as np
import numba
from numba import njit, prange

from .kernels import pcum, PCUM_TABLE, gcum3D, gcum3D_anti
from .targets import (UniformGrid, Projection2D, AMRGrid, VoronoiGrid,
                      generator_buckets)

# Gauss-Legendre nodes/weights on [0,1] for the per-edge angular quadrature of
# the exact 2D column integral (the integrand is smooth and bounded, so a modest
# order is plenty / effectively exact).
_GLX, _GLW = np.polynomial.legendre.leggauss(32)
_GLX = 0.5 * (_GLX + 1.0)          # map [-1,1] -> [0,1]
_GLW = 0.5 * _GLW

# Gauss-Legendre nodes/weights on [0,1] for the 3D per-triangle CONE integral
# (`_cone_tri`): Duffy tensor orders 3/6/12 (regime-2 ladder, order picked per
# triangle by the calibrated dispatch) and the 16-node angular rule of the
# regime-3 wedge path. n=2 was calibrated too but REJECTED: its ~1e-13 error
# floor accumulates same-sign over the ~(4h/pw)^3*12 triangles of a big-h
# particle (measured 4e-8 on a single-particle mass sweep at h/pw=10).
def _gauss01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w

_D3X, _D3W = _gauss01(3)
_D6X, _D6W = _gauss01(6)
_D12X, _D12W = _gauss01(12)
# 12 angular nodes: 3.7e-14 vs the 48-node reference over the calibration
# battery (16 gave 2.6e-16, 10 gave 4e-12 — too close to the 1e-12 bar).
_AGX, _AGW = _gauss01(12)


@njit(cache=True)
def _compute_logs(u, tol=1e-12):
    if u >= 1.0 - tol:
        return 1000.0
    elif u <= -1.0 + tol:
        return -1000.0
    else:
        return 2.0 * math.atanh(u)


@njit(cache=True)
def _clamp_acos(x):
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    return math.acos(x)


@njit(cache=True)
def _iblock(phi, a, a2, r0h, r0h2, r0h3, r0h_3):
    """The I-integrals I0..I_5, I1 at angle phi (shared by all regions)."""
    cosp = math.cos(phi)
    cosp2 = cosp * cosp
    mu = cosp / a / math.sqrt(1.0 + cosp2 / a2)
    tanp = math.tan(phi)
    I0 = phi
    I_2 = phi + a2 * tanp
    I_4 = phi + 2.0 * a2 * tanp + (a2 * a2 / 3.0) * tanp * (2.0 + 1.0 / cosp2)
    arg = 1.0 - mu * mu
    u = math.sin(phi) * math.sqrt(arg) if arg >= 0.0 else 0.0
    logs = _compute_logs(u)
    I1 = math.atan(u / a)
    I_1 = (a / 2.0) * logs + I1
    om = 1.0 - u * u
    if om != 0.0:
        I_3 = I_1 + a * (1.0 + a2) / 4.0 * (2.0 * u / om + logs)
        I_5 = I_3 + a * (1.0 + a2) ** 2 / 16.0 * ((10.0 * u - 6.0 * u ** 3) / (om * om) + 3.0 * logs)
    else:
        I_3 = I_1 + a * (1.0 + a2) / 4.0 * logs
        I_5 = I_3 + a * (1.0 + a2) ** 2 / 16.0 * 3.0 * logs
    return I0, I1, I_1, I_2, I_3, I_4, I_5


@njit(cache=True)
def _vertex_integral(phi, r0, R_0, h):
    """Exact kernel wall integral (Petkova 2018), cleaned/guarded. r0,R_0,h>0."""
    if r0 == 0.0 or R_0 == 0.0 or phi == 0.0:
        return 0.0

    r03 = r0 ** 3
    r0h = r0 / h
    r0h2 = r0h ** 2
    r0h3 = r0h ** 3
    r0h_2 = (h / r0) ** 2
    r0h_3 = (h / r0) ** 3

    B1, B2, B3 = 0.0, 0.0, 0.0
    if r0 >= 2.0 * h:
        B3 = (h ** 3) / 4.0
    elif r0 > h:
        B3 = (r03 / 4.0) * (-4.0/3.0 + r0h - 0.3*r0h2 + (1.0/30.0)*r0h3 - (1.0/15.0)*r0h_3 + (8.0/5.0)*r0h_2)
        B2 = (r03 / 4.0) * (-4.0/3.0 + r0h - 0.3*r0h2 + (1.0/30.0)*r0h3 - (1.0/15.0)*r0h_3)
    else:
        B3 = (r03 / 4.0) * (-2.0/3.0 + 0.3*r0h2 - 0.1*r0h3 + (7.0/5.0)*r0h_2)
        B2 = (r03 / 4.0) * (-2.0/3.0 + 0.3*r0h2 - 0.1*r0h3 - (1.0/5.0)*r0h_2)
        B1 = (r03 / 4.0) * (-2.0/3.0 + 0.3*r0h2 - 0.1*r0h3)

    a = R_0 / r0
    a2 = a * a
    linedist = math.sqrt(r0 * r0 + R_0 * R_0)
    R = R_0 / math.cos(phi)
    r = math.sqrt(r0 * r0 + R * R)

    D2 = 0.0
    D3 = 0.0

    if linedist < 1.0 * h:
        arg = h * h - r0 * r0
        phi1 = _clamp_acos(R_0 / math.sqrt(arg)) if arg > 0.0 else 0.0
        I0, I1, I_1, I_2, I_3, I_4, I_5 = _iblock(phi1, a, a2, r0h, r0h2, r0h3, r0h_3)
        D2 = (-1.0/6 * I_2 + 0.25 * r0h * I_3 - 0.15 * r0h2 * I_4 +
              (1.0/30) * r0h3 * I_5 - (1.0/60) * r0h_3 * I1 +
              (B1 - B2) / r03 * I0)
        arg = 4.0 * h * h - r0 * r0
        phi2 = _clamp_acos(R_0 / math.sqrt(arg)) if arg > 0.0 else 0.0
        I0, I1, I_1, I_2, I_3, I_4, I_5 = _iblock(phi2, a, a2, r0h, r0h2, r0h3, r0h_3)
        D3 = (1.0/3 * I_2 - 0.25 * r0h * I_3 + 0.075 * r0h2 * I_4 -
              (1.0/120)*r0h3 * I_5 + (4.0/15)*r0h_3 * I1 +
              (B2 - B3)/r03 * I0 + D2)
    elif linedist < 2.0 * h:
        arg = 4.0 * h * h - r0 * r0
        phi2 = _clamp_acos(R_0 / math.sqrt(arg)) if arg > 0.0 else 0.0
        I0, I1, I_1, I_2, I_3, I_4, I_5 = _iblock(phi2, a, a2, r0h, r0h2, r0h3, r0h_3)
        D3 = (1.0/3 * I_2 - 0.25 * r0h * I_3 + 0.075 * r0h2 * I_4 -
              (1.0/120)*r0h3 * I_5 + (4.0/15)*r0h_3 * I1 +
              (B2 - B3)/r03 * I0 + D2)

    I0, I1, I_1, I_2, I_3, I_4, I_5 = _iblock(phi, a, a2, r0h, r0h2, r0h3, r0h_3)
    if r < 1.0 * h:
        term1 = (1.0/6 * I_2 - 0.075 * r0h2 * I_4 + (1.0/40) * r0h3 * I_5 + (B1 / r03) * I0)
        return (r0h3 / math.pi) * term1
    elif r < 2.0 * h:
        term1 = 0.25 * (4.0/3 * I_2 - r0h * I_3 + 0.3 * r0h2 * I_4 -
                        (1.0/30)*r0h3 * I_5 + (1.0/15)*r0h_3 * I1) + (B2 / r03) * I0 + D2
        return (r0h3 / math.pi) * term1
    else:
        term1 = -0.25 * r0h_3 * I1 + (B3 / r03) * I0 + D3
        return (r0h3 / math.pi) * term1


# Axis-aligned cube topology, MATCHING pyvoro's convention exactly (the working
# exact_interpolate depends on this vertex numbering + face winding for its sign
# logic). Vertex index i is the binary triple (bit0=x, bit1=y, bit2=z):
#   0(-,-,-) 1(+,-,-) 2(-,+,-) 3(+,+,-) 4(-,-,+) 5(+,-,+) 6(-,+,+) 7(+,+,+)
# Face windings are pyvoro's (verified by feeding a single-generator box cell to
# exact_interpolate and getting Mtot=1).
_CUBE_FACES = np.array([
    [1, 3, 2, 0],   # z-lo
    [1, 5, 7, 3],   # x-hi
    [1, 0, 4, 5],   # y-lo
    [2, 6, 4, 0],   # x-lo
    [2, 3, 7, 6],   # y-hi
    [4, 6, 7, 5],   # z-hi
], dtype=np.int64)


# --------------------------------------------------------------------------- #
#  The per-triangle cone integral: regime-dispatched (2026-07-02 rewrite).
#
#  cone = int_{Omega in tri} g(R(Omega)) dOmega
#       = int int_tri g(|Q|) * r0/|Q|^3 dA,     g(R) = gcum3D(R/h)/pi,
#  with the particle at the origin, Q the point on the triangle and r0 the
#  perpendicular particle->plane distance.
#
#  History: the original analytic wedge (`_vertex_integral` + foot-point
#  cond/sign assembly) was exact for generic positions but its (h/r0)^3 and
#  R_0/r0 terms blew up for lattice-aligned near-face particles (striping,
#  +2% net mass). The 2026-07-01 "conefix" replaced it with a fixed 12x12
#  Duffy tensor Gauss rule — robust near faces, but ~20x slower everywhere
#  and silently UNCONVERGED whenever h or r0 << triangle size (a fixed-order
#  rule cannot resolve the kernel bump or the solid-angle peak inside a much
#  larger triangle): +1.66%/+0.81% mass on coarse grids, 2.4e-3 on collapsed
#  cores. This rewrite dispatches per triangle on cheap geometry to a rung
#  that is BOTH converged (<=~1e-12 abs, calibrated) and near-minimal cost.
#  Prototype + calibration battery: PLAN_PETKOVA_FIX.md ("Step-1 results").
# --------------------------------------------------------------------------- #

# Regime-2 ratio thresholds (squared): use Duffy order n when
#   smax <= c_n * min(dmin, h)
# (smax = longest edge, dmin = min particle->triangle-BODY distance),
# with SEPARATE ladders for triangles whose radial range straddles a kernel
# break (T = h or 2h crosses the triangle: integrand only C3 there) and for
# non-straddling ones (smooth integrand; n=12 held <=1e-11 at any measured
# ratio — the r0/|Q|^3 peak is cancelled by gcum3D ~ T^3). Calibrated at
# tol 1e-12 abs/triangle over a 1770-config randomized battery, safety
# factor ~0.9; n=2 rejected (same-sign error accumulation over the ~1e5
# triangles of a big-h particle).
_C3SQ_NS = 0.19 ** 2
_C6SQ_NS = 0.85 ** 2
_C12SQ_NS = 8.0 ** 2
_C3SQ_ST = 0.22 ** 2
_C6SQ_ST = 0.32 ** 2
_C12SQ_ST = 0.65 ** 2


@njit(cache=True)
def _halfspace_frac(s, h):
    """Kernel-mass fraction on the INTERIOR side of a single face plane.

    s : signed particle->plane distance, positive when the particle is on the
    interior (cell) side. Exact when the kernel crosses no OTHER face plane of
    a convex cell (regime 0): the closed-form full-plane limit of the cone
    radial integral, mass beyond a plane at r0 = 1/2 - 2q(0.35 - Gc(q)),
    q = r0/h.
    """
    q = abs(s) / h
    if q >= 2.0:
        return 1.0 if s >= 0.0 else 0.0
    mb = 0.5 - 2.0 * q * (0.35 - gcum3D_anti(q))
    if s >= 0.0:
        return 1.0 - mb
    return mb


@njit(cache=True)
def _solid_frac(ax, ay, az, bx, by, bz, cx, cy, cz):
    """(1/4pi) * solid angle of triangle ABC from the origin (Van Oosterom &
    Strijbos 1983). Exact saturated cone: g = 0.25/pi for the whole triangle."""
    la = math.sqrt(ax*ax + ay*ay + az*az)
    lb = math.sqrt(bx*bx + by*by + bz*bz)
    lc = math.sqrt(cx*cx + cy*cy + cz*cz)
    triple = (ax*(by*cz - bz*cy) + ay*(bz*cx - bx*cz) + az*(bx*cy - by*cx))
    dab = ax*bx + ay*by + az*bz
    dac = ax*cx + ay*cy + az*cz
    dbc = bx*cx + by*cy + bz*cz
    denom = la*lb*lc + dab*lc + dac*lb + dbc*la
    omega = 2.0 * math.atan2(abs(triple), denom)
    return (0.25 / math.pi) * omega


@njit(cache=True)
def _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h, xs, ws):
    """Duffy-mapped tensor Gauss cone integral at the order of the given
    node/weight tables (regime 2). Q = A + s(B-A) + t(1-s)(C-A),
    dA = area2 (1-s) ds dt; r0 in the NUMERATOR keeps the integrand finite
    for near-plane particles."""
    e1x = bx - ax; e1y = by - ay; e1z = bz - az
    e2x = cx - ax; e2y = cy - ay; e2z = cz - az
    nx = e1y*e2z - e1z*e2y
    ny = e1z*e2x - e1x*e2z
    nz = e1x*e2y - e1y*e2x
    area2 = math.sqrt(nx*nx + ny*ny + nz*nz)
    if area2 < 1e-300:
        return 0.0
    invn = 1.0 / area2
    cpl = ax*nx*invn + ay*ny*invn + az*nz*invn
    r0 = abs(cpl)
    if r0 < 1e-300:
        return 0.0
    invpi = 1.0 / math.pi
    total = 0.0
    ng = xs.shape[0]
    for gi in range(ng):
        s = xs[gi]
        om1s = 1.0 - s
        wsa = ws[gi] * om1s * area2 * r0
        for gj in range(ng):
            v = xs[gj] * om1s
            qx = ax + s * e1x + v * e2x
            qy = ay + s * e1y + v * e2y
            qz = az + s * e1z + v * e2z
            Q2 = qx*qx + qy*qy + qz*qz
            if Q2 <= 0.0:
                continue
            Q = math.sqrt(Q2)
            gR = gcum3D(Q / h) * invpi
            total += ws[gj] * wsa * gR / (Q2 * Q)
    return total


@njit(cache=True)
def _panel_gauss(lo, hi, top_heavy, d, T0sq, h, gc0):
    """int_lo^hi (Gc(T/h) - gc0)/cosh(s) ds, T = sqrt(T0sq + (d sinh s)^2).

    For long segments the integrand behaves like e^{+s} (T<h branch) or e^{-s}
    (h<T<2h branch), so cover the HEAVY end with width-3 panels of 16-node
    Gauss; beyond 12 panels the remainder is < e^{-36} relative and dropped.
    """
    width = hi - lo
    if width <= 0.0:
        return 0.0
    npan = int(width / 3.0) + 1
    if npan > 12:
        npan = 12
    acc = 0.0
    ng = _AGX.shape[0]
    for p in range(npan):
        if top_heavy:
            b = hi - 3.0 * p
            a = hi - 3.0 * (p + 1)
            if a < lo:
                a = lo
        else:
            a = lo + 3.0 * p
            b = lo + 3.0 * (p + 1)
            if b > hi:
                b = hi
        wid = b - a
        if wid <= 0.0:
            continue
        for k in range(ng):
            s = a + wid * _AGX[k]
            q = d * math.sinh(s)
            T = math.sqrt(T0sq + q * q)
            acc += _AGW[k] * wid * (gcum3D_anti(T / h) - gc0) / math.cosh(s)
    return acc


@njit(cache=True)
def _edge_E(d, rfar, r0, h, gc0):
    """Angular edge integral E(d, psi) = int_0^psi K(d/cos phi) dphi of the
    regime-3 wedge decomposition, cos psi = d/rfar, with the ANALYTIC radial
    integral K(P) = (r0/(pi h)) [Gc(T/h) - Gc(r0/h)], T = sqrt(r0^2 + P^2).

    Substituting P = d cosh s (phi = atan(sinh s), dphi = ds/cosh s) makes the
    integrand vary on O(1) scales in s for ANY d/h ratio (in phi it ramps like
    1/(pi/2-phi)^2 when d << h). Segments split where T crosses the kernel
    breaks h and 2h so each Gauss panel sees one polynomial branch of Gc; the
    saturated T >= 2h tail is FULLY analytic:
        int K/cosh ds = A [atan(sinh s)] - (1/4pi) [asin(r0 tanh s / T0)],
        A = (r0/(pi h)) (0.35 - gc0),  T0^2 = r0^2 + d^2.
    """
    if rfar <= d * (1.0 + 1e-15):
        return 0.0
    T0sq = r0 * r0 + d * d
    T0 = math.sqrt(T0sq)
    send = math.acosh(rfar / d)
    E = 0.0
    # segment 1: T < h (top-heavy: integrand ~ e^{+s} for long segments)
    sa = 0.0
    if T0 < h:
        sa = math.asinh(math.sqrt(h * h - T0sq) / d)
        hi = sa if sa < send else send
        E += _panel_gauss(0.0, hi, True, d, T0sq, h, gc0)
    # segment 2: h <= T < 2h (bottom-heavy: ~ e^{-s})
    sb = 0.0
    if T0 < 2.0 * h:
        sb = math.asinh(math.sqrt(4.0 * h * h - T0sq) / d)
        lo = sa
        hi = sb if sb < send else send
        if hi > lo:
            E += _panel_gauss(lo, hi, False, d, T0sq, h, gc0)
    E *= r0 / (math.pi * h)
    # segment 3: T >= 2h, analytic (endpoint values from geometry, so huge
    # send cannot overflow sinh/tanh)
    if send > sb:
        A = (r0 / (math.pi * h)) * (0.35 - gc0)
        tfar = math.sqrt(max(rfar * rfar - d * d, 0.0))
        gd_hi = math.atan2(tfar, d)          # = atan(sinh send)
        w_hi = tfar / rfar                   # = tanh send
        gd_lo = math.atan(math.sinh(sb))
        w_lo = math.tanh(sb)
        arg_hi = r0 * w_hi / T0
        arg_lo = r0 * w_lo / T0
        if arg_hi > 1.0:
            arg_hi = 1.0
        if arg_lo > 1.0:
            arg_lo = 1.0
        E += A * (gd_hi - gd_lo) - (math.asin(arg_hi)
                                    - math.asin(arg_lo)) / (4.0 * math.pi)
    return E


@njit(cache=True)
def _cone_wedge2d(a2x, a2y, b2x, b2y, c2x, c2y, r0, h):
    """Regime-3 cone integral: foot-point wedge decomposition in the triangle
    plane (in-plane coords with the particle's foot at the 2D origin), analytic
    radial integral + `_edge_E` per edge endpoint. EXACT at any h/r0/size ratio.
    Combine/sign logic identical to the validated 2D `_pixel_column_integral`
    (foot inside segment -> EA+EB, outside -> |EA-EB|; sign from the cross
    product; abs at the end so the result is winding-independent)."""
    gc0 = gcum3D_anti(r0 / h)
    total = 0.0
    for e in range(3):
        if e == 0:
            px = a2x; py = a2y; qx = b2x; qy = b2y
        elif e == 1:
            px = b2x; py = b2y; qx = c2x; qy = c2y
        else:
            px = c2x; py = c2y; qx = a2x; qy = a2y
        lx = qx - px; ly = qy - py
        ll = math.sqrt(lx * lx + ly * ly)
        if ll < 1e-300:
            continue
        cross = px * qy - py * qx
        d = abs(cross) / ll
        rP = math.sqrt(px * px + py * py)
        rQ = math.sqrt(qx * qx + qy * qy)
        rmax = rP if rP > rQ else rQ
        if d <= 1e-16 * rmax:    # foot on the edge line: zero-area wedges
            continue
        EP = _edge_E(d, rP, r0, h, gc0)
        EQ = _edge_E(d, rQ, r0, h, gc0)
        tP = math.sqrt(max(rP * rP - d * d, 0.0))
        tQ = math.sqrt(max(rQ * rQ - d * d, 0.0))
        outside = (tP + 1e-12 * ll >= ll) or (tQ + 1e-12 * ll >= ll)
        contrib = abs(EP - EQ) if outside else (EP + EQ)
        if cross > 0.0:
            total += contrib
        else:
            total -= contrib
    return abs(total)


@njit(cache=True)
def _pseg2(px, py, qx, qy):
    """Squared distance from the 2D origin to segment P-Q."""
    lx = qx - px; ly = qy - py
    l2 = lx * lx + ly * ly
    if l2 <= 0.0:
        return px * px + py * py
    t = -(px * lx + py * ly) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = px + t * lx; ey = py + t * ly
    return ex * ex + ey * ey


@njit(cache=True)
def _tri_frame(ax, ay, az, bx, by, bz, cx, cy, cz):
    """Shared triangle geometry for the dispatch:
    (area2, r0, dmin2, smax2, a2x, a2y, b2x, b2y, c2x, c2y).

    In-plane coordinates put the foot of the particle on the triangle plane at
    the 2D origin (particle at the 3D origin). dmin2 = squared min distance
    from the particle to the triangle BODY (r0^2 if the foot is inside)."""
    e1x = bx - ax; e1y = by - ay; e1z = bz - az
    e2x = cx - ax; e2y = cy - ay; e2z = cz - az
    nx = e1y*e2z - e1z*e2y
    ny = e1z*e2x - e1x*e2z
    nz = e1x*e2y - e1y*e2x
    area2 = math.sqrt(nx*nx + ny*ny + nz*nz)
    if area2 < 1e-300:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    invn = 1.0 / area2
    nhx = nx*invn; nhy = ny*invn; nhz = nz*invn
    cpl = ax*nhx + ay*nhy + az*nhz
    r0 = abs(cpl)
    l1 = math.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
    exx = e1x / l1; exy = e1y / l1; exz = e1z / l1
    eyx = nhy*exz - nhz*exy
    eyy = nhz*exx - nhx*exz
    eyz = nhx*exy - nhy*exx
    a2x = ax*exx + ay*exy + az*exz
    a2y = ax*eyx + ay*eyy + az*eyz
    b2x = bx*exx + by*exy + bz*exz
    b2y = bx*eyx + by*eyy + bz*eyz
    c2x = cx*exx + cy*exy + cz*exz
    c2y = cx*eyx + cy*eyy + cz*eyz
    d1 = e1x*e1x + e1y*e1y + e1z*e1z
    d2 = e2x*e2x + e2y*e2y + e2z*e2z
    fx = cx - bx; fy = cy - by; fz = cz - bz
    d3 = fx*fx + fy*fy + fz*fz
    smax2 = d1
    if d2 > smax2:
        smax2 = d2
    if d3 > smax2:
        smax2 = d3
    c1 = (b2x - a2x) * (-a2y) - (b2y - a2y) * (-a2x)
    c2 = (c2x - b2x) * (-b2y) - (c2y - b2y) * (-b2x)
    c3 = (a2x - c2x) * (-c2y) - (a2y - c2y) * (-c2x)
    inside = (c1 >= 0.0 and c2 >= 0.0 and c3 >= 0.0) or \
             (c1 <= 0.0 and c2 <= 0.0 and c3 <= 0.0)
    if inside:
        dmin2 = r0 * r0
    else:
        p1 = _pseg2(a2x, a2y, b2x, b2y)
        p2 = _pseg2(b2x, b2y, c2x, c2y)
        p3 = _pseg2(c2x, c2y, a2x, a2y)
        pm = p1
        if p2 < pm:
            pm = p2
        if p3 < pm:
            pm = p3
        dmin2 = r0 * r0 + pm
    return area2, r0, dmin2, smax2, a2x, a2y, b2x, b2y, c2x, c2y


@njit(cache=True)
def _cone_tri(ax, ay, az, bx, by, bz, cx, cy, cz, h):
    """Kernel integral over the cone from the PARTICLE (at the origin) through
    the triangle A,B,C (coords RELATIVE to the particle). Always >= 0.

    Building block of the signed-tetrahedralization cell integral:
     int_cell W = sum over faces (front/back sign) * sum over fan-triangles
    _cone_tri. Regime-dispatched on cheap geometry (see the block comment
    above and PLAN_PETKOVA_FIX.md):
      1. whole triangle beyond 2h -> exact solid angle (Van Oosterom-Strijbos);
      2. converged-by-calibration Duffy tensor Gauss at order 3/6/12, with a
         stricter ladder when the triangle straddles a kernel break (T=h/2h);
      3. otherwise (scale-separated: h or r0 << triangle) the foot-point wedge
         decomposition with analytic radial integral — exact at any ratio.
    """
    (area2, r0, dmin2, smax2,
     a2x, a2y, b2x, b2y, c2x, c2y) = _tri_frame(ax, ay, az, bx, by, bz,
                                                cx, cy, cz)
    if area2 < 1e-300 or r0 < 1e-300:
        return 0.0
    hsq = h * h
    if dmin2 >= 4.0 * hsq:
        return _solid_frac(ax, ay, az, bx, by, bz, cx, cy, cz)
    # straddle test: does the radial range [rmin, rmax] contain h or 2h?
    # Padded by 0.35*smax (the calibration battery's straddle offsets): a kink
    # just OUTSIDE the radial range degrades the smooth-ladder accuracy almost
    # as much as one inside it (worst case: the kink exactly at rmin, e.g. a
    # cell-centre particle with 2h touching all faces).
    ra2 = ax*ax + ay*ay + az*az
    rb2 = bx*bx + by*by + bz*bz
    rc2 = cx*cx + cy*cy + cz*cz
    rmax2 = ra2
    if rb2 > rmax2:
        rmax2 = rb2
    if rc2 > rmax2:
        rmax2 = rc2
    L2 = dmin2 if dmin2 < hsq else hsq
    pad = 0.35 * math.sqrt(smax2)
    rlo = math.sqrt(dmin2) - pad
    rhi = math.sqrt(rmax2) + pad
    if (rlo < h < rhi) or (rlo < 2.0 * h < rhi):
        if smax2 <= _C3SQ_ST * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D3X, _D3W)
        if smax2 <= _C6SQ_ST * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D6X, _D6W)
        if smax2 <= _C12SQ_ST * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D12X, _D12W)
    else:
        if smax2 <= _C3SQ_NS * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D3X, _D3W)
        if smax2 <= _C6SQ_NS * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D6X, _D6W)
        if smax2 <= _C12SQ_NS * L2:
            return _cone_duffy(ax, ay, az, bx, by, bz, cx, cy, cz, h,
                               _D12X, _D12W)
    return _cone_wedge2d(a2x, a2y, b2x, b2y, c2x, c2y, r0, h)


# --------------------------------------------------------------------------- #
#  Quad (rectangle) cone integral: cube faces are integrated as ONE planar
#  rectangle instead of two fan triangles — half the frame setups and rule
#  evaluations, no wasted shared-diagonal wedge work, identical regime
#  dispatch (smax = diagonal, the same value the two fan triangles saw).
#  PARALLELOGRAM assumption (Q = v0 + u e1 + v e3): exact for the axis-aligned
#  cube faces this is used for; Voronoi polygon fans keep `_cone_tri`.
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z, e3x, e3y, e3z,
                     area, r0, h, xs, ws):
    """Tensor Gauss cone integral over the parallelogram v0 + u e1 + v e3,
    u,v in [0,1]; area = |e1 x e3| (regime 2, quad version)."""
    invpi = 1.0 / math.pi
    total = 0.0
    ng = xs.shape[0]
    for gi in range(ng):
        u = xs[gi]
        bx = v0x + u * e1x
        by = v0y + u * e1y
        bz = v0z + u * e1z
        wu = ws[gi] * area * r0
        for gj in range(ng):
            v = xs[gj]
            qx = bx + v * e3x
            qy = by + v * e3y
            qz = bz + v * e3z
            Q2 = qx*qx + qy*qy + qz*qz
            if Q2 <= 0.0:
                continue
            Q = math.sqrt(Q2)
            gR = gcum3D(Q / h) * invpi
            total += ws[gj] * wu * gR / (Q2 * Q)
    return total


@njit(cache=True)
def _cone_wedge2d_quad(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, r0, h):
    """Regime-3 wedge decomposition over a planar quad (4 directed edges);
    same construction as `_cone_wedge2d`."""
    gc0 = gcum3D_anti(r0 / h)
    total = 0.0
    for e in range(4):
        if e == 0:
            px = p0x; py = p0y; qx = p1x; qy = p1y
        elif e == 1:
            px = p1x; py = p1y; qx = p2x; qy = p2y
        elif e == 2:
            px = p2x; py = p2y; qx = p3x; qy = p3y
        else:
            px = p3x; py = p3y; qx = p0x; qy = p0y
        lx = qx - px; ly = qy - py
        ll = math.sqrt(lx * lx + ly * ly)
        if ll < 1e-300:
            continue
        cross = px * qy - py * qx
        d = abs(cross) / ll
        rP = math.sqrt(px * px + py * py)
        rQ = math.sqrt(qx * qx + qy * qy)
        rmax = rP if rP > rQ else rQ
        if d <= 1e-16 * rmax:
            continue
        EP = _edge_E(d, rP, r0, h, gc0)
        EQ = _edge_E(d, rQ, r0, h, gc0)
        tP = math.sqrt(max(rP * rP - d * d, 0.0))
        tQ = math.sqrt(max(rQ * rQ - d * d, 0.0))
        outside = (tP + 1e-12 * ll >= ll) or (tQ + 1e-12 * ll >= ll)
        contrib = abs(EP - EQ) if outside else (EP + EQ)
        if cross > 0.0:
            total += contrib
        else:
            total -= contrib
    return abs(total)


@njit(cache=True)
def _cone_quad(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, h):
    """Kernel cone integral through a planar RECTANGLE v0..v3 (winding order,
    coords relative to the particle). Regime dispatch identical to
    `_cone_tri` (see there); always >= 0."""
    e1x = v1x - v0x; e1y = v1y - v0y; e1z = v1z - v0z
    e3x = v3x - v0x; e3y = v3y - v0y; e3z = v3z - v0z
    nx = e1y*e3z - e1z*e3y
    ny = e1z*e3x - e1x*e3z
    nz = e1x*e3y - e1y*e3x
    area = math.sqrt(nx*nx + ny*ny + nz*nz)
    if area < 1e-300:
        return 0.0
    invn = 1.0 / area
    nhx = nx*invn; nhy = ny*invn; nhz = nz*invn
    cpl = v0x*nhx + v0y*nhy + v0z*nhz
    r0 = abs(cpl)
    if r0 < 1e-300:
        return 0.0
    hsq = h * h
    # in-plane frame (foot of the particle at the 2D origin)
    l1 = math.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
    exx = e1x / l1; exy = e1y / l1; exz = e1z / l1
    eyx = nhy*exz - nhz*exy
    eyy = nhz*exx - nhx*exz
    eyz = nhx*exy - nhy*exx
    p0x = v0x*exx + v0y*exy + v0z*exz
    p0y = v0x*eyx + v0y*eyy + v0z*eyz
    p1x = v1x*exx + v1y*exy + v1z*exz
    p1y = v1x*eyx + v1y*eyy + v1z*eyz
    p2x = v2x*exx + v2y*exy + v2z*exz
    p2y = v2x*eyx + v2y*eyy + v2z*eyz
    p3x = v3x*exx + v3y*exy + v3z*exz
    p3y = v3x*eyx + v3y*eyy + v3z*eyz
    # origin-in-quad via edge cross signs (convex, consistent winding)
    c1 = (p1x - p0x) * (-p0y) - (p1y - p0y) * (-p0x)
    c2 = (p2x - p1x) * (-p1y) - (p2y - p1y) * (-p1x)
    c3 = (p3x - p2x) * (-p2y) - (p3y - p2y) * (-p2x)
    c4 = (p0x - p3x) * (-p3y) - (p0y - p3y) * (-p3x)
    inside = (c1 >= 0.0 and c2 >= 0.0 and c3 >= 0.0 and c4 >= 0.0) or \
             (c1 <= 0.0 and c2 <= 0.0 and c3 <= 0.0 and c4 <= 0.0)
    if inside:
        dmin2 = r0 * r0
    else:
        pm = _pseg2(p0x, p0y, p1x, p1y)
        p_ = _pseg2(p1x, p1y, p2x, p2y)
        if p_ < pm:
            pm = p_
        p_ = _pseg2(p2x, p2y, p3x, p3y)
        if p_ < pm:
            pm = p_
        p_ = _pseg2(p3x, p3y, p0x, p0y)
        if p_ < pm:
            pm = p_
        dmin2 = r0 * r0 + pm
    if dmin2 >= 4.0 * hsq:
        return (_solid_frac(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)
                + _solid_frac(v0x, v0y, v0z, v2x, v2y, v2z, v3x, v3y, v3z))
    # size measure: max diagonal (what the two fan triangles saw as smax)
    dgx = v2x - v0x; dgy = v2y - v0y; dgz = v2z - v0z
    s1 = dgx*dgx + dgy*dgy + dgz*dgz
    dgx = v3x - v1x; dgy = v3y - v1y; dgz = v3z - v1z
    s2 = dgx*dgx + dgy*dgy + dgz*dgz
    smax2 = s1 if s1 > s2 else s2
    r_a = v0x*v0x + v0y*v0y + v0z*v0z
    r_b = v1x*v1x + v1y*v1y + v1z*v1z
    if r_b > r_a:
        r_a = r_b
    r_b = v2x*v2x + v2y*v2y + v2z*v2z
    if r_b > r_a:
        r_a = r_b
    r_b = v3x*v3x + v3y*v3y + v3z*v3z
    if r_b > r_a:
        r_a = r_b
    rmax2 = r_a
    L2 = dmin2 if dmin2 < hsq else hsq
    pad = 0.35 * math.sqrt(smax2)
    rlo = math.sqrt(dmin2) - pad
    rhi = math.sqrt(rmax2) + pad
    if (rlo < h < rhi) or (rlo < 2.0 * h < rhi):
        if smax2 <= _C3SQ_ST * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D3X, _D3W)
        if smax2 <= _C6SQ_ST * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D6X, _D6W)
        if smax2 <= _C12SQ_ST * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D12X, _D12W)
    else:
        if smax2 <= _C3SQ_NS * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D3X, _D3W)
        if smax2 <= _C6SQ_NS * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D6X, _D6W)
        if smax2 <= _C12SQ_NS * L2:
            return _cone_duffy_quad(v0x, v0y, v0z, e1x, e1y, e1z,
                                    e3x, e3y, e3z, area, r0, h, _D12X, _D12W)
    return _cone_wedge2d_quad(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, r0, h)


@njit(cache=True)
def _cube_integral(vx, vy, vz, h, faces):
    """Integral of the unit-normalized kernel over a cube (particle at origin).

    vx,vy,vz : (8,) cube vertex coords RELATIVE to the particle.
    Returns the contained kernel-mass fraction (>= 0). Signed tetrahedralization
    from the particle: one front/back sign per face, fan-triangulate, sum
    `_cone_tri` (see its docstring). Mass-conserving for any cube position/size.

    Regime-0 pre-pass (h << cell): from the 6 signed inward face-plane
    distances (vertex 0/7 are the box corners), if the kernel sphere crosses
    NO plane the integral is exactly 1 (inside) or 0 (outside); if it crosses
    exactly ONE plane the cell sees a single infinite plane and the integral
    is the closed-form half-space split. Collapses the entire collapsed-core
    population (h down to ~0.01 cell) to O(1) flops; only particles within 2h
    of an edge/corner fall through to the per-triangle path.
    """
    th = 2.0 * h
    nclose = 0
    sclose = 0.0
    s = -vx[0]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    s = vx[7]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    s = -vy[0]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    s = vy[7]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    s = -vz[0]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    s = vz[7]
    if s <= -th:
        return 0.0
    if s < th:
        nclose += 1; sclose = s
    if nclose == 0:
        return 1.0
    if nclose == 1:
        return _halfspace_frac(sclose, h)
    # interior point = centroid of the 8 vertices (particle-relative), used to
    # orient each face's outward normal.
    cgx = 0.0; cgy = 0.0; cgz = 0.0
    for k in range(8):
        cgx += vx[k]; cgy += vy[k]; cgz += vz[k]
    cgx *= 0.125; cgy *= 0.125; cgz *= 0.125
    total = 0.0
    for f in range(faces.shape[0]):
        i0 = faces[f, 0]; i1 = faces[f, 1]; i2 = faces[f, 2]; i3 = faces[f, 3]
        v0x = vx[i0]; v0y = vy[i0]; v0z = vz[i0]
        ax = vx[i1]-v0x; ay = vy[i1]-v0y; az = vz[i1]-v0z
        bx = vx[i2]-v0x; by = vy[i2]-v0y; bz = vz[i2]-v0z
        nx = ay*bz - az*by; ny = az*bx - ax*bz; nz = ax*by - ay*bx
        # orient outward (point away from the centroid)
        if nx*(cgx-v0x) + ny*(cgy-v0y) + nz*(cgz-v0z) > 0.0:
            nx = -nx; ny = -ny; nz = -nz
        sgn = 1.0 if (v0x*nx + v0y*ny + v0z*nz) > 0.0 else -1.0
        # one planar-rectangle cone integral per face (was: 2 fan triangles)
        total += sgn * _cone_quad(
            v0x, v0y, v0z,
            vx[i1], vy[i1], vz[i1],
            vx[i2], vy[i2], vz[i2],
            vx[i3], vy[i3], vz[i3], h)
    if total < 0.0:
        total = 0.0
    return total


@njit(cache=True)
def _nearest_dist2(xi, yi, zi, x0, x1, y0, y1, z0, z1):
    """Squared distance from a point to an axis-aligned box [x0,x1]x..."""
    dx = 0.0
    if xi < x0:
        dx = x0 - xi
    elif xi > x1:
        dx = xi - x1
    dy = 0.0
    if yi < y0:
        dy = y0 - yi
    elif yi > y1:
        dy = yi - y1
    dz = 0.0
    if zi < z0:
        dz = z0 - zi
    elif zi > z1:
        dz = zi - z1
    return dx*dx + dy*dy + dz*dz


@njit(parallel=True, cache=True)
def _reduce_threads(local, out):
    """Parallel sum over the leading (thread) axis: out[...] = sum_t local[t].
    np.sum(axis=0) is single-threaded and costs seconds on the multi-GB
    per-thread accumulators."""
    nt = local.shape[0]
    flat = local.reshape(nt, -1)
    oflat = out.reshape(-1)
    for i in prange(oflat.shape[0]):
        s = 0.0
        for t in range(nt):
            s += flat[t, i]
        oflat[i] = s


@njit(parallel=True, cache=True)
def _reduce_threads_zslab(local, out, k0, nk):
    """out[:, :, k0:k0+nk] = sum_t local[t, :, :, :nk].

    `_reduce_threads` reshapes to (nt, -1), which a partial z view is not
    contiguous enough for, and the last slab is short whenever G does not
    divide nz. Indexing 4-D directly sidesteps both."""
    nt, ni, nj = local.shape[0], local.shape[1], local.shape[2]
    for i in prange(ni):
        for j in range(nj):
            for k in range(nk):
                s = 0.0
                for t in range(nt):
                    s += local[t, i, j, k]
                out[i, j, k0 + k] = s


@njit(parallel=True, cache=True)
def _reduce_threads_zslab_k(local, out, k0, nk):
    """The same, for the (nt, nx, ny, nz, K) field accumulator."""
    nt, ni, nj, K = (local.shape[0], local.shape[1], local.shape[2],
                     local.shape[4])
    for i in prange(ni):
        for j in range(nj):
            for k in range(nk):
                for p in range(K):
                    s = 0.0
                    for t in range(nt):
                        s += local[t, i, j, k, p]
                    out[i, j, k0 + k, p] = s


# Peak per-thread accumulator bytes tolerated before the uniform deposit is
# split into z slabs. The accumulators are (nthreads,) + cell_shape + (K,) --
# N_grid x nthreads, with NO N_part axis, so particle count is free and only the
# grid and the thread count drive this. Measured ceilings (K=1 / K=5 fields):
#
#     uniform 64^3    0.07 / 0.16 GB at 12 threads     0.75 /   1.75 GB at 128
#     uniform 128^3   0.56 / 1.31 GB at 12 threads     6.0  /  14.0  GB at 128
#     uniform 256^3   4.5  / 10.5 GB at 12 threads    48    / 112    GB at 128
#
# 2 GiB leaves every 64^3 and most 128^3 runs at G=1 (bit-identical to the
# unslabbed code, same summation order) and catches the cases that would
# otherwise have died. Override with SPH_INTERP_MAX_ACCUM_BYTES, or per call
# with `slabs=`.
MAX_ACCUM_BYTES = int(os.environ.get("SPH_INTERP_MAX_ACCUM_BYTES",
                                     2 * 1024**3))


def _slab_count(nthreads, nx, ny, nz, nstreams, cap=None):
    """Number of z slabs so the per-thread accumulators fit in `cap` bytes.

    `nstreams` is the number of float64 values accumulated per cell (1 for the
    mass grid, +1 for the kernel-volume denominator, +K for the fields), so
    `nthreads * nx * ny * nstreams * 8` is the cost of ONE z layer -- the
    quantum this can't subdivide. Returns 1 when a full-depth accumulator
    already fits, and never more than nz.
    """
    cap = MAX_ACCUM_BYTES if cap is None else cap
    per_layer = nthreads * nx * ny * nstreams * 8
    if cap <= 0 or per_layer <= 0:
        return 1
    layers = max(1, int(cap // per_layer))
    return int(min(nz, -(-nz // layers)))


def _slab_edges(nz, nslabs):
    """z cut points, as equal as nz allows: G+1 edges bounding G slabs."""
    return np.linspace(0, nz, int(nslabs) + 1).astype(np.int64)


def _needs_weights(names):
    """True unless every interpolated field is density-like (those use only the
    conserved mass accumulator, so the kernel-volume den/num accumulators — two
    of the three scattered write streams — can be skipped entirely)."""
    return any(n.lower() not in ("rho", "density") for n in names)


# Particles are usually stored in spatially coherent order, so equal-size
# prange chunks give some threads all the big-h (expensive) particles.
#
# MEASURED 2026-07-28: this chunk-size knob does NOT fix that — it does nothing
# at all. On numba 0.65.1 the deposit takes 21.1 s at _CHUNKSIZE = 1, 4, 16,
# 256, 4096 and 0 alike, i.e. `set_parallel_chunksize` never reaches this loop.
# The imbalance it was meant to cure is severe and was still fully present:
# on the bench's clustered 200k set the last 3 of 12 equal blocks of the
# particle array hold 99.5% of the work, and petkova scaled 1.03x on 2 threads
# and 1.04x on 4 (a pure-compute control in the same process scaled 2.01x and
# 3.94x, so threading itself was fine).
#
# `_balanced_order` below fixes it where the scheduler would not: it reorders
# the particles so that ANY contiguous chunk holds a representative mix of
# cheap and expensive ones. Kept alongside rather than deleted, since it is
# harmless and would start helping again if numba honours it in future.
_CHUNKSIZE = 16

# Below this, threading is not the bottleneck and the permutation is not worth
# the gather it costs.
_BALANCE_MIN_N = 4096


class _chunked:
    """Context manager: numba parallel chunk size around a deposit call."""
    def __enter__(self):
        self._old = numba.set_parallel_chunksize(_CHUNKSIZE)

    def __exit__(self, *exc):
        numba.set_parallel_chunksize(self._old)


def _balanced_order(n):
    """A permutation making every contiguous prange chunk equally expensive.

    Returns None when reordering cannot pay (single thread, or too few
    particles), in which case the caller leaves the arrays alone.

    A FIXED-SEED shuffle rather than a work-sorted deal: numba's actual
    partitioning of a prange is not part of its API, so the only property that
    holds for whatever chunking it picks is that a random permutation makes
    every contiguous subrange a uniform sample of the work distribution. It is
    deterministic across runs, so the deposit stays reproducible.

    Cost is one gather of the particle arrays and it does NOT hurt locality in
    practice — measured single-thread time is unchanged (51.7 s shuffled vs
    52.0 s in order), because the deposit is compute-bound in the per-cell
    integral, not bandwidth-bound.

    THE RESULT IS UNCHANGED except for floating-point summation order: the
    accumulators are indexed by CELL, not by particle, so a permutation only
    reorders the additions into each cell. Measured max relative difference
    5.2e-16.
    """
    if n < _BALANCE_MIN_N or numba.get_num_threads() < 2:
        return None
    return np.random.default_rng(0).permutation(n)


@njit(parallel=True, cache=True)
def _deposit_petkova_uniform(x, y, z, h, mass, volw, vals,
                             nx, ny, nz, lox, loy, loz, pwx, pwy, pwz,
                             perx, pery, perz, faces, need_w,
                             conservative_fields,
                             mass_local, den_local, num_local, kz0, kz1):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        mi = mass[i]
        hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        t = numba.get_thread_id()
        # per-iteration (thread-private) vertex scratch — must NOT be shared
        # across prange iterations or threads race and corrupt each other.
        vx = np.empty(8); vy = np.empty(8); vz = np.empty(8)
        wi = volw[i]
        xi = x[i]; yi = y[i]; zi = z[i]
        kr = 2.0 * hi
        kr2 = kr * kr
        ixmin = int(np.floor((xi - kr - lox) / pwx))
        ixmax = int(np.floor((xi + kr - lox) / pwx)) + 1
        jymin = int(np.floor((yi - kr - loy) / pwy))
        jymax = int(np.floor((yi + kr - loy) / pwy)) + 1
        kzmin = int(np.floor((zi - kr - loz) / pwz))
        kzmax = int(np.floor((zi + kr - loz) / pwz)) + 1
        if not perx:
            ixmin = max(ixmin, 0); ixmax = min(ixmax, nx)
        if not pery:
            jymin = max(jymin, 0); jymax = min(jymax, ny)
        if not perz:
            kzmin = max(kzmin, 0); kzmax = min(kzmax, nz)
            # This pass owns wrapped z indices [kz0, kz1). On a NON-periodic
            # axis the raw index IS the wrapped one, so the slab clips the loop
            # bounds and a particle outside the slab exits here, before any
            # geometry. (kz0=0, kz1=nz when unslabbed: both no-ops.)
            kzmin = max(kzmin, kz0); kzmax = min(kzmax, kz1)
            if kzmin >= kzmax:
                continue

        for ix in range(ixmin, ixmax):
            iw = ix % nx if perx else ix
            xe0 = lox + ix * pwx
            xe1 = xe0 + pwx
            for jy in range(jymin, jymax):
                jw = jy % ny if pery else jy
                ye0 = loy + jy * pwy
                ye1 = ye0 + pwy
                for kz in range(kzmin, kzmax):
                    if perz:
                        # A PERIODIC axis cannot clip the loop bounds above:
                        # the raw range lives in unwrapped index space, and a
                        # kernel straddling the boundary lands on BOTH ends of
                        # the grid — clamping it would silently drop the wrapped
                        # half. So the slab test is done in WRAPPED index space,
                        # per cell, and placed before the geometry so a cell
                        # outside this slab costs one modulo and a compare.
                        # (This is the piece `prof_slab.py` left unsolved: the
                        # probe slabs a non-periodic axis only.)
                        kw = kz % nz
                        if kw < kz0 or kw >= kz1:
                            continue
                    else:
                        kw = kz
                    ze0 = loz + kz * pwz
                    ze1 = ze0 + pwz
                    # skip cells entirely outside the kernel sphere
                    if _nearest_dist2(xi, yi, zi, xe0, xe1, ye0, ye1, ze0, ze1) >= kr2:
                        continue
                    # cube vertices relative to particle (min-image via raw index)
                    ax0 = xe0 - xi; ax1 = xe1 - xi
                    ay0 = ye0 - yi; ay1 = ye1 - yi
                    az0 = ze0 - zi; az1 = ze1 - zi
                    vx[0]=ax0; vy[0]=ay0; vz[0]=az0
                    vx[1]=ax1; vy[1]=ay0; vz[1]=az0
                    vx[2]=ax0; vy[2]=ay1; vz[2]=az0
                    vx[3]=ax1; vy[3]=ay1; vz[3]=az0
                    vx[4]=ax0; vy[4]=ay0; vz[4]=az1
                    vx[5]=ax1; vy[5]=ay0; vz[5]=az1
                    vx[6]=ax0; vy[6]=ay1; vz[6]=az1
                    vx[7]=ax1; vy[7]=ay1; vz[7]=az1
                    frac = _cube_integral(vx, vy, vz, hi, faces)
                    if frac <= 0.0:
                        continue
                    ks = kw - kz0                 # slab-local z index
                    mass_local[t, iw, jw, ks] += mi * frac
                    if need_w:
                        if not conservative_fields:
                            den_local[t, iw, jw, ks] += wi * frac
                        fw = mi if conservative_fields else wi
                        for p in range(K):
                            num_local[t, iw, jw, ks, p] += fw * vals[i, p] * frac


def interpolate_uniform_petkova(particles, grid, values=None,
                                 conservative_fields=False, slabs=None):
    """Petkova exact interpolation of `particles` onto a `UniformGrid`.

    Returns data: dict[name -> (nx,ny,nz)]. Density-like fields ('rho'/'density')
    are the MASS-CONSERVING estimate sum_j m_j (integral_cell W_j) / V_cell; other
    fields are the kernel-volume-weighted cell average
    sum_j (m_j/rho_j) A_j I / sum_j (m_j/rho_j) I. With
    ``conservative_fields=True``, non-density fields instead use the exact
    mass fractions: sum_j m_j A_j I / sum_j m_j I, and their raw integrated
    contents are exposed as ``_content_<name>``.

    ``slabs`` caps peak accumulator memory by depositing in G passes over z
    slabs, each pass allocating accumulators only one slab deep (see
    `MAX_ACCUM_BYTES`). ``None`` picks G from that cap; an int forces it.
    Every cell integral is still performed exactly once — the slab clip happens
    in the cheap bbox loop, BEFORE `_cube_integral` — so only the per-particle
    setup of a kernel spanning several slabs repeats, which the profile puts at
    ~0% of the deposit. Measured on `disk` N=200k at 128³ on 12 threads,
    slabbing is 7-11% FASTER than G=1 (the accumulator slab fits cache better)
    with mass conservation unchanged to the last digit.
    """
    if not isinstance(grid, UniformGrid):
        raise TypeError("interpolate_uniform_petkova requires a UniformGrid target")

    vals, names = particles.value_array(values)
    nx, ny, nz = grid.npx
    lo = grid.lo
    pw = grid.pixwidth
    perx, pery, perz = particles.periodic

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    need_w = _needs_weights(names)
    # float64s accumulated per cell: the mass grid, the kernel-volume
    # denominator (only when weights are needed and not conservative), and one
    # per field (only when weights are needed at all).
    nstreams = 1 + (1 if (need_w and not conservative_fields) else 0) \
                 + (K if need_w else 0)
    if slabs is None:
        nslabs = _slab_count(nthreads, nx, ny, nz, nstreams)
    else:
        nslabs = int(slabs)
        if nslabs < 1:
            raise ValueError(f"slabs must be >= 1, got {slabs}")
        nslabs = min(nslabs, nz)
    edges = _slab_edges(nz, nslabs)
    ksz = int(np.max(np.diff(edges)))          # deepest slab; the last is short

    dshape = ((nx, ny, ksz) if need_w and not conservative_fields
              else (1, 1, 1))
    nshape = (nx, ny, ksz) if need_w else (1, 1, 1)
    mass_local = np.zeros((nthreads, nx, ny, ksz), dtype=np.float64)
    den_local = np.zeros((nthreads,) + dshape, dtype=np.float64)
    num_local = np.zeros((nthreads,) + nshape + (K,), dtype=np.float64)

    mass = np.empty((nx, ny, nz), dtype=np.float64)
    den = np.zeros((nx, ny, nz), dtype=np.float64)
    num = np.zeros((nx, ny, nz, K), dtype=np.float64)

    pos = particles.pos
    h_p, m_p, volw_p, vals_p = (particles.h, particles.mass,
                                particles.mass / particles.rho, vals)
    order = _balanced_order(len(m_p))
    if order is not None:
        pos = pos[order]
        h_p, m_p, volw_p, vals_p = (h_p[order], m_p[order], volw_p[order],
                                    np.ascontiguousarray(vals_p[order]))
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2])

    for g in range(nslabs):
        k0, k1 = int(edges[g]), int(edges[g + 1])
        nk = k1 - k0
        if g > 0:                              # pass 0 gets the fresh zeros
            mass_local[:, :, :, :nk] = 0.0
            if need_w:
                if not conservative_fields:
                    den_local[:, :, :, :nk] = 0.0
                num_local[:, :, :, :nk, :] = 0.0
        with _chunked():
            _deposit_petkova_uniform(
                px, py, pz, h_p, m_p, volw_p, vals_p,
                nx, ny, nz, lo[0], lo[1], lo[2], pw[0], pw[1], pw[2],
                perx, pery, perz, _CUBE_FACES, need_w, conservative_fields,
                mass_local, den_local, num_local, k0, k1)
        if nslabs == 1:
            _reduce_threads(mass_local, mass)
            if need_w:
                if not conservative_fields:
                    _reduce_threads(den_local, den)
                _reduce_threads(num_local, num)
        else:
            _reduce_threads_zslab(mass_local, mass, k0, nk)
            if need_w:
                if not conservative_fields:
                    _reduce_threads_zslab(den_local, den, k0, nk)
                _reduce_threads_zslab_k(num_local, num, k0, nk)
    cv = grid.cell_volume
    mask = mass > 0.0 if conservative_fields else den > 0.0

    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = mass / cv
        else:
            g = np.zeros((nx, ny, nz), dtype=np.float64)
            divisor = mass if conservative_fields else den
            g[mask] = num[..., p][mask] / divisor[mask]
            data[name] = g
            if conservative_fields:
                data[f"_content_{name}"] = num[..., p]
    # always expose the conserved mass grid for convenience
    data.setdefault("_mass", mass)
    return data


# --------------------------------------------------------------------------- #
#  AMR target: exact kernel-volume integral over each octree leaf cube. Same
#  base-cell -> leaves traversal as the SPH-AMR deposit, but each leaf gets the
#  exact contained kernel-mass fraction via `_cube_integral` (mass-conserving on
#  the adaptive cells, like the UniformGrid Petkova path).
# --------------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _deposit_petkova_amr(x, y, z, h, mass, volw, vals, cell_lo, cell_size,
                         base_start, base_count, blo0, blo1, blo2,
                         bpw0, bpw1, bpw2, nbx, nby, nbz,
                         perx, pery, perz, boxx, boxy, boxz, faces, need_w,
                         conservative_fields,
                         mass_local, den_local, num_local):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        mi = mass[i]
        hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        t = numba.get_thread_id()
        vx = np.empty(8); vy = np.empty(8); vz = np.empty(8)
        wi = volw[i]
        xi = x[i]; yi = y[i]; zi = z[i]
        kr = 2.0 * hi
        kr2 = kr * kr
        ibx0 = int(np.floor((xi - kr - blo0) / bpw0))
        ibx1 = int(np.floor((xi + kr - blo0) / bpw0)) + 1
        iby0 = int(np.floor((yi - kr - blo1) / bpw1))
        iby1 = int(np.floor((yi + kr - blo1) / bpw1)) + 1
        ibz0 = int(np.floor((zi - kr - blo2) / bpw2))
        ibz1 = int(np.floor((zi + kr - blo2) / bpw2)) + 1
        if not perx:
            if ibx0 < 0: ibx0 = 0
            if ibx1 > nbx: ibx1 = nbx
        if not pery:
            if iby0 < 0: iby0 = 0
            if iby1 > nby: iby1 = nby
        if not perz:
            if ibz0 < 0: ibz0 = 0
            if ibz1 > nbz: ibz1 = nbz
        for ibx in range(ibx0, ibx1):
            if perx:
                kx = ibx // nbx; iwx = ibx - kx * nbx; sx = kx * boxx
            else:
                iwx = ibx; sx = 0.0
            for iby in range(iby0, iby1):
                if pery:
                    ky = iby // nby; iwy = iby - ky * nby; sy = ky * boxy
                else:
                    iwy = iby; sy = 0.0
                for ibz in range(ibz0, ibz1):
                    if perz:
                        kz = ibz // nbz; iwz = ibz - kz * nbz; sz = kz * boxz
                    else:
                        iwz = ibz; sz = 0.0
                    bflat = (iwx * nby + iwy) * nbz + iwz
                    s0 = base_start[bflat]
                    n0 = base_count[bflat]
                    for li in range(s0, s0 + n0):
                        xe0 = cell_lo[li, 0] + sx
                        xe1 = xe0 + cell_size[li, 0]
                        ye0 = cell_lo[li, 1] + sy
                        ye1 = ye0 + cell_size[li, 1]
                        ze0 = cell_lo[li, 2] + sz
                        ze1 = ze0 + cell_size[li, 2]
                        if _nearest_dist2(xi, yi, zi, xe0, xe1, ye0, ye1,
                                          ze0, ze1) >= kr2:
                            continue
                        ax0 = xe0 - xi; ax1 = xe1 - xi
                        ay0 = ye0 - yi; ay1 = ye1 - yi
                        az0 = ze0 - zi; az1 = ze1 - zi
                        vx[0]=ax0; vy[0]=ay0; vz[0]=az0
                        vx[1]=ax1; vy[1]=ay0; vz[1]=az0
                        vx[2]=ax0; vy[2]=ay1; vz[2]=az0
                        vx[3]=ax1; vy[3]=ay1; vz[3]=az0
                        vx[4]=ax0; vy[4]=ay0; vz[4]=az1
                        vx[5]=ax1; vy[5]=ay0; vz[5]=az1
                        vx[6]=ax0; vy[6]=ay1; vz[6]=az1
                        vx[7]=ax1; vy[7]=ay1; vz[7]=az1
                        frac = _cube_integral(vx, vy, vz, hi, faces)
                        if frac <= 0.0:
                            continue
                        mass_local[t, li] += mi * frac
                        if need_w:
                            if not conservative_fields:
                                den_local[t, li] += wi * frac
                            fw = mi if conservative_fields else wi
                            for p in range(K):
                                num_local[t, li, p] += fw * vals[i, p] * frac


def interpolate_amr_petkova(particles, grid, values=None,
                             conservative_fields=False):
    """Petkova exact interpolation of `particles` onto an `AMRGrid`.

    Returns data: dict[name -> (Ncell,)]. Density-like fields ('rho'/'density')
    are the MASS-CONSERVING estimate sum_j m_j (integral_cell W_j) / V_cell; other
    fields are the kernel-volume-weighted leaf average. With
    ``conservative_fields=True``, non-density fields are mass-weighted from the
    same exact fractions and raw integrated contents are exposed as
    ``_content_<name>``. The summed leaf mass equals the particle mass to high
    precision (the headline Petkova property), now on the adaptive cells.
    `_mass` (conserved per-leaf mass) is also exposed.
    """
    if not isinstance(grid, AMRGrid):
        raise TypeError("interpolate_amr_petkova requires an AMRGrid target")

    vals, names = particles.value_array(values)
    Ncell = grid.ncell
    nbx, nby, nbz = grid.base_npx
    blo = grid.lo
    bpw = grid.base_pw
    box = grid.box
    perx, pery, perz = grid.periodic

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    need_w = _needs_weights(names)
    nd = Ncell if need_w and not conservative_fields else 1
    nn = Ncell if need_w else 1
    mass_local = np.zeros((nthreads, Ncell), dtype=np.float64)
    den_local = np.zeros((nthreads, nd), dtype=np.float64)
    num_local = np.zeros((nthreads, nn, K), dtype=np.float64)

    pos = particles.pos
    h_p, m_p, volw_p, vals_p = (particles.h, particles.mass,
                                particles.mass / particles.rho, vals)
    order = _balanced_order(len(m_p))
    if order is not None:
        pos = pos[order]
        h_p, m_p, volw_p, vals_p = (h_p[order], m_p[order], volw_p[order],
                                    np.ascontiguousarray(vals_p[order]))
    with _chunked():
        _deposit_petkova_amr(
            np.ascontiguousarray(pos[:, 0]),
            np.ascontiguousarray(pos[:, 1]),
            np.ascontiguousarray(pos[:, 2]),
            h_p, m_p, volw_p, vals_p,
            grid.cell_lo, grid.cell_size, grid.base_start, grid.base_count,
            blo[0], blo[1], blo[2], bpw[0], bpw[1], bpw[2],
            nbx, nby, nbz, perx, pery, perz, box[0], box[1], box[2],
            _CUBE_FACES, need_w, conservative_fields,
            mass_local, den_local, num_local)

    mass = np.empty(Ncell, dtype=np.float64)
    _reduce_threads(mass_local, mass)
    den = np.zeros(Ncell, dtype=np.float64)
    num = np.zeros((Ncell, K), dtype=np.float64)
    if need_w:
        if not conservative_fields:
            _reduce_threads(den_local, den)
        _reduce_threads(num_local, num)
    vol = grid.volumes()
    mask = mass > 0.0 if conservative_fields else den > 0.0

    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = mass / vol
        else:
            g = np.zeros(Ncell, dtype=np.float64)
            divisor = mass if conservative_fields else den
            g[mask] = num[mask, p] / divisor[mask]
            data[name] = g
            if conservative_fields:
                data[f"_content_{name}"] = num[:, p]
    data.setdefault("_mass", mass)
    return data


# --------------------------------------------------------------------------- #
#  Voronoi target: the NATIVE Petkova case — integrate the kernel exactly over an
#  arbitrary convex polyhedron. `_voro_cell_integral` is `_cube_integral`
#  generalized from the fixed 8-vertex cube to the CSR face/vertex layout of a
#  VoronoiGrid cell (vertices shifted to the particle frame); the per-face/edge
#  divergence-theorem construction and pyvoro's face winding are identical.
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _voro_cell_integral(px, py, pz, h, gx, gy, gz, c,
                        cf_start, fv_start, fvx, fvy, fvz):
    """Integral of the unit-normalized kernel over Voronoi cell `c`, with the
    particle at (px,py,pz) and the cell's generator at (gx,gy,gz). Returns the
    contained kernel-mass fraction (>= 0).

    Signed tetrahedralization from the particle: one front/back sign per face
    (the generator gives the interior side to orient the outward normal),
    fan-triangulate each polygonal face, sum `_cone_tri`. This replaces the old
    foot-point wedge assembly (`_vertex_integral` + cond1/cond2 + sign_M), which
    failed to telescope for cells straddling the kernel break (h/2h) when the
    particle is outside the cell — the ~1-1.5% Voronoi conservation error on
    high-dynamic-range (collapsed-core) meshes. Now machine-precision conserving.

    Regime-0 pre-pass (h << cell), as in `_cube_integral` but per face plane
    of the convex polyhedron: if the kernel sphere crosses no plane the
    integral is exactly 1/0; exactly one plane -> closed-form half-space split.
    """
    th = 2.0 * h
    f0 = cf_start[c]; f1 = cf_start[c + 1]
    nclose = 0
    sclose = 0.0
    grx = gx - px; gry = gy - py; grz = gz - pz
    for f in range(f0, f1):
        i0 = fv_start[f]; i1 = fv_start[f + 1]
        if i1 - i0 < 3:
            continue
        v0x = fvx[i0] - px;   v0y = fvy[i0] - py;   v0z = fvz[i0] - pz
        ax = fvx[i0+1] - px - v0x; ay = fvy[i0+1] - py - v0y; az = fvz[i0+1] - pz - v0z
        bx = fvx[i0+2] - px - v0x; by = fvy[i0+2] - py - v0y; bz = fvz[i0+2] - pz - v0z
        nx = ay*bz - az*by; ny = az*bx - ax*bz; nz = ax*by - ay*bx
        # orient outward (away from the generator)
        if nx*(grx - v0x) + ny*(gry - v0y) + nz*(grz - v0z) > 0.0:
            nx = -nx; ny = -ny; nz = -nz
        nn = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nn < 1e-300:
            continue
        # signed inward particle->plane distance (positive on the cell side)
        s = (v0x*nx + v0y*ny + v0z*nz) / nn
        if s <= -th:
            return 0.0
        if s < th:
            nclose += 1
            sclose = s
    if nclose == 0:
        return 1.0
    if nclose == 1:
        return _halfspace_frac(sclose, h)
    total = 0.0
    for f in range(f0, f1):
        i0 = fv_start[f]; i1 = fv_start[f + 1]
        nfv = i1 - i0
        if nfv < 3:
            continue
        # first three face vertices (particle-relative) -> face normal
        v0x = fvx[i0] - px;   v0y = fvy[i0] - py;   v0z = fvz[i0] - pz
        ax = fvx[i0+1] - px - v0x; ay = fvy[i0+1] - py - v0y; az = fvz[i0+1] - pz - v0z
        bx = fvx[i0+2] - px - v0x; by = fvy[i0+2] - py - v0y; bz = fvz[i0+2] - pz - v0z
        nx = ay*bz - az*by; ny = az*bx - ax*bz; nz = ax*by - ay*bx
        # generator relative to the particle, to orient the outward normal
        grx = gx - px; gry = gy - py; grz = gz - pz
        if nx*(grx - v0x) + ny*(gry - v0y) + nz*(grz - v0z) > 0.0:
            nx = -nx; ny = -ny; nz = -nz
        sgn = 1.0 if (v0x*nx + v0y*ny + v0z*nz) > 0.0 else -1.0
        # fan-triangulate the face polygon from vertex 0
        for k in range(1, nfv - 1):
            ka = i0 + k; kb = i0 + k + 1
            total += sgn * _cone_tri(
                v0x, v0y, v0z,
                fvx[ka] - px, fvy[ka] - py, fvz[ka] - pz,
                fvx[kb] - px, fvy[kb] - py, fvz[kb] - pz, h)
    if total < 0.0:
        total = 0.0
    return total


@njit(cache=True)
def _petkova_voro_pair(xi, yi, zi, hi, mi, wi, i, c, kr,
                       genx, geny, genz, gen_rad,
                       cf_start, fv_start, fvx, fvy, fvz, need_w, vals,
                       t, mass_local, den_local, num_local):
    """One particle-cell pair: exact circumradius prune (the kernel reaches
    cell c only if |particle-gen| < 2h + rad_c), then the cell integral."""
    dx = genx[c] - xi; dy = geny[c] - yi; dz = genz[c] - zi
    reach = kr + gen_rad[c]
    if dx*dx + dy*dy + dz*dz >= reach * reach:
        return
    frac = _voro_cell_integral(xi, yi, zi, hi,
                               genx[c], geny[c], genz[c], c,
                               cf_start, fv_start, fvx, fvy, fvz)
    if frac <= 0.0:
        return
    mass_local[t, c] += mi * frac
    if need_w:
        den_local[t, c] += wi * frac
        for p in range(vals.shape[1]):
            num_local[t, c, p] += wi * vals[i, p] * frac


@njit(parallel=True, cache=True)
def _deposit_petkova_voronoi(x, y, z, h, mass, volw, vals,
                             genx, geny, genz, gen_rad,
                             nb, blo, bw, bstart, border, bradmax, rmax_small,
                             brute_cell,
                             cf_start, fv_start, fvx, fvy, fvz, need_w,
                             mass_local, den_local, num_local):
    """Candidate search FUSED into the deposit: per particle, walk the uniform
    generator-bucket grid (`generator_buckets`) over the index cube reached by
    2h + rmax_small, prune whole buckets by distance-to-bucket-box against
    2h + bradmax[b], and test each surviving generator with the exact
    |particle-gen| < 2h + rad_c condition. No per-particle candidate lists —
    materializing them (cKDTree.query_ball_point Python lists) cost minutes at
    >=1e8 pairs and dominated the whole deposit. `brute_cell` holds the few
    giant cells (rad > rmax_small) EVERY particle tests directly, so the walk
    extent stays bounded by the SMALL-cell radius, not the global max."""
    npart = x.shape[0]
    nbrute = brute_cell.shape[0]
    nb2 = nb * nb
    for i in prange(npart):
        mi = mass[i]
        hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        t = numba.get_thread_id()
        wi = volw[i]
        xi = x[i]; yi = y[i]; zi = z[i]
        kr = 2.0 * hi
        ext = kr + rmax_small
        ix0 = int((xi - ext - blo[0]) / bw[0]); ix1 = int((xi + ext - blo[0]) / bw[0])
        iy0 = int((yi - ext - blo[1]) / bw[1]); iy1 = int((yi + ext - blo[1]) / bw[1])
        iz0 = int((zi - ext - blo[2]) / bw[2]); iz1 = int((zi + ext - blo[2]) / bw[2])
        if ix0 < 0: ix0 = 0
        if iy0 < 0: iy0 = 0
        if iz0 < 0: iz0 = 0
        if ix1 > nb - 1: ix1 = nb - 1
        if iy1 > nb - 1: iy1 = nb - 1
        if iz1 > nb - 1: iz1 = nb - 1
        for ix in range(ix0, ix1 + 1):
            bx0 = blo[0] + ix * bw[0]
            dxb = 0.0
            if xi < bx0:
                dxb = bx0 - xi
            elif xi > bx0 + bw[0]:
                dxb = xi - bx0 - bw[0]
            for iy in range(iy0, iy1 + 1):
                by0 = blo[1] + iy * bw[1]
                dyb = 0.0
                if yi < by0:
                    dyb = by0 - yi
                elif yi > by0 + bw[1]:
                    dyb = yi - by0 - bw[1]
                dxy2 = dxb * dxb + dyb * dyb
                for iz in range(iz0, iz1 + 1):
                    b = (ix * nb + iy) * nb + iz
                    s0 = bstart[b]; s1 = bstart[b + 1]
                    if s1 == s0:
                        continue
                    bz0 = blo[2] + iz * bw[2]
                    dzb = 0.0
                    if zi < bz0:
                        dzb = bz0 - zi
                    elif zi > bz0 + bw[2]:
                        dzb = zi - bz0 - bw[2]
                    breach = kr + bradmax[b]
                    if dxy2 + dzb * dzb >= breach * breach:
                        continue
                    for jj in range(s0, s1):
                        _petkova_voro_pair(
                            xi, yi, zi, hi, mi, wi, i, border[jj], kr,
                            genx, geny, genz, gen_rad,
                            cf_start, fv_start, fvx, fvy, fvz, need_w, vals,
                            t, mass_local, den_local, num_local)
        for jj in range(nbrute):
            _petkova_voro_pair(
                xi, yi, zi, hi, mi, wi, i, brute_cell[jj], kr,
                genx, geny, genz, gen_rad,
                cf_start, fv_start, fvx, fvy, fvz, need_w, vals,
                t, mass_local, den_local, num_local)


def interpolate_voronoi_petkova(particles, grid, values=None):
    """Petkova exact interpolation of `particles` onto a `VoronoiGrid`.

    The native Petkova case: each particle's kernel is integrated EXACTLY over
    every overlapping Voronoi polyhedron, so the summed cell mass equals the
    particle mass to high precision (the headline mass-conserving property), now
    on an unstructured mesh. Density-like fields ('rho'/'density') are
    sum_j m_j (integral_cell W_j)/V_cell; other fields are the kernel-volume-
    weighted cell average. `_mass` (conserved per-cell mass) is also exposed.

    Returns data: dict[name -> (Ncell,)].
    """
    if not isinstance(grid, VoronoiGrid):
        raise TypeError("interpolate_voronoi_petkova requires a VoronoiGrid target")

    vals, names = particles.value_array(values)
    Ntot = grid.ntotal                # deposit over ALL cells (incl. ghosts)

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    need_w = _needs_weights(names)
    nw = Ntot if need_w else 1
    mass_local = np.zeros((nthreads, Ntot), dtype=np.float64)
    den_local = np.zeros((nthreads, nw), dtype=np.float64)
    num_local = np.zeros((nthreads, nw, K), dtype=np.float64)

    pos = particles.pos
    px = np.ascontiguousarray(pos[:, 0])
    py = np.ascontiguousarray(pos[:, 1])
    pz = np.ascontiguousarray(pos[:, 2])
    gx = np.ascontiguousarray(grid.gen[:, 0])
    gy = np.ascontiguousarray(grid.gen[:, 1])
    gz = np.ascontiguousarray(grid.gen[:, 2])
    volw = particles.mass / particles.rho

    # Cell overlap condition: |particle-gen| < 2h + cell circumradius (the
    # integral itself -> 0 for any marginal cell the kernel does not actually
    # reach). The candidate search is FUSED into the numba deposit as a
    # uniform-bucket-grid walk (`generator_buckets`): tree-querying every
    # particle with a circumradius-inflated radius and materializing the
    # Python candidate lists cost minutes at >=1e8 pairs (measured 1.6e9
    # pairs / 13 GB / ~4 min on a clustered merger mesh vs ~2e8 true pairs),
    # dominating the whole deposit. The few GIANT cells (rad > 8x median)
    # would put every particle in reach anyway, so they go to the deposit as
    # a brute list and the bucket-walk extent stays bounded by the small-cell
    # circumradius, not the global max.
    rad = np.asarray(grid.cell_rad, dtype=np.float64)
    nbig = int((rad > 8.0 * float(np.median(rad))).sum()) if Ntot else 0
    nbig = min(nbig, 2048)
    order = np.argsort(rad)
    brute = np.ascontiguousarray(order[Ntot - nbig:]) if nbig else \
        np.zeros(0, dtype=np.int64)
    small = order[:Ntot - nbig]
    rmax_s = float(rad[small].max()) if small.size else 0.0
    NB = 64
    blo, bw, bstart, border, bradmax = generator_buckets(
        grid.gen[small], rad[small], nb=NB)
    border = small[border] if border.size else border   # bucket CSR -> global ids
    # NB `order` above is the CELL radius sort; this is a PARTICLE permutation
    porder = _balanced_order(len(particles.mass))
    h_p, m_p, volw_p, vals_p = particles.h, particles.mass, volw, vals
    if porder is not None:
        px = np.ascontiguousarray(px[porder])
        py = np.ascontiguousarray(py[porder])
        pz = np.ascontiguousarray(pz[porder])
        h_p, m_p, volw_p, vals_p = (h_p[porder], m_p[porder], volw_p[porder],
                                    np.ascontiguousarray(vals_p[porder]))
    with _chunked():
        _deposit_petkova_voronoi(
            px, py, pz,
            h_p, m_p, volw_p, vals_p,
            gx, gy, gz, rad,
            NB, blo, bw, bstart,
            np.ascontiguousarray(border, dtype=np.int64), bradmax, rmax_s,
            brute,
            grid.cf_start, grid.fv_start, grid.fvx, grid.fvy, grid.fvz,
            need_w, mass_local, den_local, num_local)

    # fold periodic ghost-cell deposits back onto the real (central) cells
    msum = np.empty(Ntot, dtype=np.float64)
    _reduce_threads(mass_local, msum)
    mass = grid.fold_to_central(msum)
    if need_w:
        dsum = np.empty(Ntot, dtype=np.float64)
        _reduce_threads(den_local, dsum)
        den = grid.fold_to_central(dsum)
        nsum = np.empty((Ntot, K), dtype=np.float64)
        _reduce_threads(num_local, nsum)
        num = grid.fold_to_central(nsum)
    else:
        den = np.zeros(grid.ncell, dtype=np.float64)
        num = np.zeros((grid.ncell, K), dtype=np.float64)
    Ncell = grid.ncell
    vol = grid.volumes()
    mask = den > 0.0

    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = mass / vol
        else:
            g = np.zeros(Ncell, dtype=np.float64)
            g[mask] = num[mask, p] / den[mask]
            data[name] = g
    data.setdefault("_mass", mass)
    return data


# --------------------------------------------------------------------------- #
#  Exact 2D column (Petkova projection): integrate the projected kernel Y(R,h)
#  exactly over each pixel square, instead of point-sampling it (SPH 2D).
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _edge_angular(doh, psi, ptable, glx, glw):
    """E(d,psi) = int_0^psi M(d/cos t) dt with M(r) = (1/pi) P(r/h).

    doh = d/h (perpendicular distance / smoothing length); integrand bounded
    because the argument d/(h cos t) <= r/h <= 2 over the subtended range."""
    if psi <= 0.0:
        return 0.0
    acc = 0.0
    for k in range(glx.shape[0]):
        t = psi * glx[k]
        arg = doh / math.cos(t)
        acc += glw[k] * pcum(arg, ptable)
    return psi * acc / math.pi


@njit(cache=True)
def _pixel_column_integral(cx, cy, h, ptable, glx, glw):
    """Exact fraction of a particle's column mass falling in a pixel.

    cx,cy : (4,) pixel-corner coords RELATIVE to the particle's projected
            position (particle at origin), ordered around the pixel.
    Returns int_pixel Y(R,h) dA (in [0,1]); sum over a covering grid -> 1.
    """
    eps = 1e-12
    total = 0.0
    for e in range(4):
        ax = cx[e]; ay = cy[e]
        bx = cx[(e + 1) % 4]; by = cy[(e + 1) % 4]
        lx = bx - ax; ly = by - ay
        ll = math.sqrt(lx * lx + ly * ly)
        if ll < eps:
            continue
        d = abs(ax * ly - ay * lx) / ll          # perp distance O->line(A,B)
        r12 = math.sqrt(ax * ax + ay * ay)
        r13 = math.sqrt(bx * bx + by * by)
        psiA = math.acos(min(d / r12, 1.0)) if r12 > eps else 0.0
        psiB = math.acos(min(d / r13, 1.0)) if r13 > eps else 0.0
        EA = _edge_angular(d / h, psiA, ptable, glx, glw)
        EB = _edge_angular(d / h, psiB, ptable, glx, glw)
        # perpendicular foot inside the segment -> the two wedges add; outside ->
        # they subtract (same construction as the 3D vertex_integral combine).
        outside = (r12 * math.sin(psiA) + eps >= ll) or (r13 * math.sin(psiB) + eps >= ll)
        contrib = abs(EA - EB) if outside else (EA + EB)
        sgn = ax * by - ay * bx                   # signed triangle (O,A,B) area*2
        if sgn > 0.0:
            total += contrib
        elif sgn < 0.0:
            total -= contrib
    if total < 0.0:
        total = 0.0
    return total


@njit(parallel=True, cache=True)
def _deposit_petkova_proj(a1, a2, h, mass, volw, vals, nx, ny, lo0, lo1,
                          pw0, pw1, per0, per1, ptable, glx, glw,
                          a_vol, w_vol, a_mass, w_mass):
    """Exact 2D column deposit. Per-pixel weight = (int_pixel Y dA)/pixarea, i.e.
    the exact pixel-averaged projected kernel (cf. the point-sampled SPH version).
    Accumulators per-thread (nthreads, nx, ny[, K])."""
    npart = a1.shape[0]
    K = vals.shape[1]
    pixarea = pw0 * pw1
    for i in prange(npart):
        mi = mass[i]
        hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        t = numba.get_thread_id()
        vi = volw[i]
        p1 = a1[i]; p2 = a2[i]
        kr = 2.0 * hi
        imin = int(np.floor((p1 - kr - lo0) / pw0))
        imax = int(np.floor((p1 + kr - lo0) / pw0)) + 1
        jmin = int(np.floor((p2 - kr - lo1) / pw1))
        jmax = int(np.floor((p2 + kr - lo1) / pw1)) + 1
        if not per0:
            imin = max(imin, 0); imax = min(imax, nx)
        if not per1:
            jmin = max(jmin, 0); jmax = min(jmax, ny)
        cx = np.empty(4); cy = np.empty(4)
        for ix in range(imin, imax):
            iw = ix % nx if per0 else ix
            xe0 = lo0 + ix * pw0 - p1
            xe1 = xe0 + pw0
            for iy in range(jmin, jmax):
                ye0 = lo1 + iy * pw1 - p2
                ye1 = ye0 + pw1
                # CCW corners relative to particle
                cx[0] = xe0; cy[0] = ye0
                cx[1] = xe1; cy[1] = ye0
                cx[2] = xe1; cy[2] = ye1
                cx[3] = xe0; cy[3] = ye1
                ipix = _pixel_column_integral(cx, cy, hi, ptable, glx, glw)
                if ipix <= 0.0:
                    continue
                wgt = ipix / pixarea            # exact pixel-averaged column kernel
                jw = iy % ny if per1 else iy
                mw = mi * wgt
                vw = vi * wgt
                w_mass[t, iw, jw] += mw
                w_vol[t, iw, jw] += vw
                for p in range(K):
                    a = vals[i, p]
                    a_vol[t, iw, jw, p] += vw * a
                    a_mass[t, iw, jw, p] += mw * a


def interpolate_projection2d_petkova(particles, target, values=None):
    """Exact (Petkova) 2D column interpolation onto a Projection2D target.

    Supports the column-type modes ('column', 'average', 'rhocolumn'); 'slice' is
    not a column integral (use the SPH slice, or the 3D Petkova on a thin slab).
    Returns data: dict[name -> (nx,ny)] with the same conventions as the SPH 2D
    projection, but each particle's column is integrated EXACTLY over every pixel
    (mass-conserving per pixel, even when h < pixel).
    """
    if not isinstance(target, Projection2D):
        raise TypeError("requires a Projection2D target")
    if target.mode == "slice":
        raise NotImplementedError(
            "Petkova-2D is a column integral; use method='sph' for slice (or the "
            "3D Petkova on a thin slab).")

    vals, names = particles.value_array(values)
    nx, ny = target.npx
    lo = target.lo
    pw = target.pixwidth
    a1c = np.ascontiguousarray(particles.pos[:, target.axis1])
    a2c = np.ascontiguousarray(particles.pos[:, target.axis2])
    per0 = particles.periodic[target.axis1]
    per1 = particles.periodic[target.axis2]

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    a_vol = np.zeros((nthreads, nx, ny, K))
    w_vol = np.zeros((nthreads, nx, ny))
    a_mass = np.zeros((nthreads, nx, ny, K))
    w_mass = np.zeros((nthreads, nx, ny))

    _deposit_petkova_proj(a1c, a2c, particles.h, particles.mass,
                          particles.mass / particles.rho, vals, nx, ny,
                          lo[0], lo[1], pw[0], pw[1], per0, per1,
                          PCUM_TABLE, _GLX, _GLW, a_vol, w_vol, a_mass, w_mass)

    a_vol = a_vol.sum(axis=0); w_vol = w_vol.sum(axis=0)
    a_mass = a_mass.sum(axis=0); w_mass = w_mass.sum(axis=0)
    mv = w_vol > 0.0; mm = w_mass > 0.0

    data = {}
    for p, name in enumerate(names):
        is_dens = name.lower() in ("rho", "density")
        if target.mode == "column":
            data[name] = w_mass.copy() if is_dens else a_vol[..., p]
        elif target.mode == "average":
            g = np.zeros((nx, ny)); g[mv] = a_vol[..., p][mv] / w_vol[mv]
            data[name] = g
        else:  # rhocolumn
            g = np.zeros((nx, ny)); g[mm] = a_mass[..., p][mm] / w_mass[mm]
            data[name] = g
    return data
