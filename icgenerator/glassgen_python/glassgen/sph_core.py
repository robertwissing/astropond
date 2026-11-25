"""Numba pair loops: SPH density, ICGenerator force, quality diagnostics.

Ports the ICGEN pair physics of icgenerator/gasoline:
- density: gather sum rho_i = 1/(pi h_i^3) sum_j m_j W(q_ij) including self
  (Density/DenDVDX, smoothfcn.c).
- force (ICGenerator, smoothfcn.c:5794): in C each mutual pair is visited
  twice by the symmetric resmooth (once with each owner's h) and the kernel
  carries fNorm1 = 0.5/(pi h^5), which yields the arithmetic-mean kernel
  symmetrization. Here that is reformulated gather-only: particle i
  accumulates its own-ball terms (kernel h_i, neighbors indices[i]) plus the
  h_j-side terms, so each prange thread writes only icvel[i] - no races.
- diagnostics (CHECKQUALITY): Q0_i = sum m_j/rho_j W_ij(h_i) (partition of
  unity, -> 1; the zeroth-order interpolation consistency - see
  diagnostics.py for the full Q0/Q1/E0/E1 hierarchy) and E0_i = zero-order
  force error (the force with pP == 1, scaled by rho_i h_i), accumulated in
  the same loop.

The pair force is the gasoline CORRPARTITION parameter family: sigma_f (the
force/gradient volume exponent, dFacQf: 1 = GDSPH, 0 = TSPH) sets the pair
weights, and `gradient` (GRADIENTS) sets the window rule and separation:
'average-*' uses the arithmetic-mean kernel 0.5[gradW(h_i)+gradW(h_j)] with
both pressures, 'onesided-*' (gasoline ONESIDED) lets each window owner
contribute only its own term at full kernel weight, and '*-isph' replaces the
separation dx by the linear-exact C dx (isph_matrix, gasoline DenDVDX).
'onesided-isph' with sigma_f = 1 (default) is gasoline's default GDFORCE+ISPH
build; 'average-sph' is the C icgenerator.

A SINGLE force loop (_force_loop) serves every backend: it walks CSR
neighbour lists (indptr/indices). The kNN backend passes indptr=arange*k,
indices=idx.ravel() (a view, identical iteration order); the ball backend
passes its native CSR. The h_j (transposed) side is taken from the precomputed
transpose graph (symmetric) or evaluated inline from i's own list (one-sided,
SPH-EXA style, no transpose) per the `one_sided` flag.
"""
import math

import numpy as np
from numba import njit, prange

from .flavours import ig_pair
from .kernels import KP0, dw_k, kernel_spec, w_k

M_1_PI = 1.0 / math.pi


@njit(cache=True, parallel=True)
def _density_loop(pos, mass, h, idx, box, periodic, rho, w0self, kid, kp):
    # positions must be wrapped to [-L/2, L/2] when periodic: min-image is
    # then two predictable compares instead of a divide+rint per axis.
    # w0self is the (possibly bias-corrected) self central value W(0) and
    # kid/kp the kernel, both from kernels.kernel_spec.
    n, k = idx.shape
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        hi = h[i]
        ih2 = 1.0 / (hi * hi)
        s = mass[i] * w0self  # self contribution, W(0)
        for jj in range(k):
            j = idx[i, jj]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > hbx:
                    dx -= bx
                elif dx < -hbx:
                    dx += bx
                if dy > hby:
                    dy -= by
                elif dy < -hby:
                    dy += by
                if dz > hbz:
                    dz -= bz
                elif dz < -hbz:
                    dz += bz
            ar2 = (dx * dx + dy * dy + dz * dz) * ih2
            s += mass[j] * w_k(kid, kp, ar2)
        rho[i] = s * M_1_PI / (hi * hi * hi)


@njit(cache=True, parallel=True)
def _density_loop_csr(pos, mass, h, indptr, indices, box, periodic, rho, w0self,
                      kid, kp):
    """SPH density over a CSR (ball-gather) neighbour list: each particle i
    sums over indices[indptr[i]:indptr[i+1]] (all within ~2*h_i). w0self is the
    (possibly bias-corrected) self central value W(0) - see density_loop."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        hi = h[i]
        ih2 = 1.0 / (hi * hi)
        s = mass[i] * w0self
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > hbx:
                    dx -= bx
                elif dx < -hbx:
                    dx += bx
                if dy > hby:
                    dy -= by
                elif dy < -hby:
                    dy += by
                if dz > hbz:
                    dz -= bz
                elif dz < -hbz:
                    dz += bz
            ar2 = (dx * dx + dy * dy + dz * dz) * ih2
            if ar2 < 4.0:
                s += mass[j] * w_k(kid, kp, ar2)
        rho[i] = s * M_1_PI / (hi * hi * hi)


def density_loop(pos, mass, h, idx, box, periodic, rho, w0self, kid=0, kp=KP0):
    """SPH density over the dense kNN candidates; kid/kp from
    kernels.kernel_spec (default Wendland C2), w0self its self term."""
    _density_loop(pos, mass, h, idx, box, periodic, rho, w0self, kid, kp)


def density_loop_csr(pos, mass, h, indptr, indices, box, periodic, rho, w0self,
                     kid=0, kp=KP0):
    """density_loop over a CSR (ball-gather) neighbour list."""
    _density_loop_csr(pos, mass, h, indptr, indices, box, periodic, rho,
                      w0self, kid, kp)


# gradient -> (onesided window, isph matrix)
GRADIENTS = {'onesided-isph': (True, True), 'onesided-sph': (True, False),
             'average-isph': (False, True), 'average-sph': (False, False)}
_NOCMAT = np.zeros((1, 6))
# boundary-term threshold of the ISPH fallback (gasoline boundterm0; set by --isph-boundterm)
ISPH_BOUNDTERM = 2.0
# gasoline's grad-h owner factor (CORRPARTITION_GRADH_EXACT; set by --gradh, one-sided gradients)
GRADH = False


def _gradient_spec(gradient):
    if gradient not in GRADIENTS:
        raise ValueError(f"unknown gradient {gradient!r}, have {sorted(GRADIENTS)}")
    return GRADIENTS[gradient]


@njit(cache=True, parallel=True)
def isph_matrix(pos, mass, h, rho, indptr, indices, box, periodic, sigf,
                kid, kp, cmat, bmax=2.0):
    """ISPH/LESPH linear-exact gradient correction as gasoline DenDVDX
    (ISPH+ONESIDED, GDFORCE): C_i = T_i^-1, T_i = sum_j V_ij
    (r_j - r_i) (x) grad_i W_ij(h_i) over i's own support, with the sigma_f
    volume V_ij = m_j / (rho_i^(1-sigf) rho_j^sigf). Identity where
    gasoline falls back: < 10 neighbours, cond = |T|_F |T^-1|_F outside
    (0, 6], or boundary term |T^-1 I_1| h > bmax. cmat: xx, xy, xz, yy, yz, zz."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        hi = h[i]
        ih2 = 1.0 / (hi * hi)
        fn1 = M_1_PI * ih2 * ih2 / hi
        txx = 0.0
        txy = 0.0
        txz = 0.0
        tyy = 0.0
        tyz = 0.0
        tzz = 0.0
        ix = 0.0
        iy = 0.0
        iz = 0.0
        nn = 1  # gasoline's list counts the particle itself
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > hbx:
                    dx -= bx
                elif dx < -hbx:
                    dx += bx
                if dy > hby:
                    dy -= by
                elif dy < -hby:
                    dy += by
                if dz > hbz:
                    dz -= bz
                elif dz < -hbz:
                    dz += bz
            ar2 = (dx * dx + dy * dy + dz * dz) * ih2
            if ar2 >= 4.0:
                continue
            nn += 1
            if sigf == 1.0:
                vol = mass[j] / rho[j]
            elif sigf == 0.0:
                vol = mass[j] / rho[i]
            else:
                vol = mass[j] / (rho[i] ** (1.0 - sigf) * rho[j] ** sigf)
            rq = -dw_k(kid, kp, ar2) * fn1 * vol
            txx += dx * dx * rq
            txy += dx * dy * rq
            txz += dx * dz * rq
            tyy += dy * dy * rq
            tyz += dy * dz * rq
            tzz += dz * dz * rq
            ix += dx * rq
            iy += dy * rq
            iz += dz * rq
        cxx = 1.0
        cxy = 0.0
        cxz = 0.0
        cyy = 1.0
        cyz = 0.0
        czz = 1.0
        det = (txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz)
               + txz * (txy * tyz - tyy * txz))
        if nn >= 10 and det != 0.0:
            idet = 1.0 / det
            axx = (tyy * tzz - tyz * tyz) * idet
            axy = (txz * tyz - txy * tzz) * idet
            axz = (txy * tyz - txz * tyy) * idet
            ayy = (txx * tzz - txz * txz) * idet
            ayz = (txy * txz - txx * tyz) * idet
            azz = (txx * tyy - txy * txy) * idet
            cond = math.sqrt(txx * txx + 2 * txy * txy + 2 * txz * txz + tyy * tyy
                             + 2 * tyz * tyz + tzz * tzz) \
                * math.sqrt(axx * axx + 2 * axy * axy + 2 * axz * axz + ayy * ayy
                            + 2 * ayz * ayz + azz * azz)
            vx = ix * axx + iy * axy + iz * axz
            vy = ix * axy + iy * ayy + iz * ayz
            vz = ix * axz + iy * ayz + iz * azz
            bterm = math.sqrt(vx * vx + vy * vy + vz * vz) * hi
            if cond > 0.0 and cond <= 6.0 and bterm <= bmax:
                cxx = axx
                cxy = axy
                cxz = axz
                cyy = ayy
                cyz = ayz
                czz = azz
        cmat[i, 0] = cxx
        cmat[i, 1] = cxy
        cmat[i, 2] = cxz
        cmat[i, 3] = cyy
        cmat[i, 4] = cyz
        cmat[i, 5] = czz


@njit(cache=True, parallel=True)
def gradh_factor(pos, mass, h, rho, indptr, indices, box, periodic, sigf,
                 kid, kp, isph, cmat, fac):
    """gasoline's divv corrector (CORRPARTITION_GRADH_EXACT, fDivvCGradH). With h ~ rho^-1/3
    the kernel's Euler identity reduces it exactly to f_i = 3 / tr(C_i T_i): 1 where the
    linear-exact matrix is used, 3 / tr T_i where it falls back (or for the plain gradient).
    T_i and the sigma_f volumes as isph_matrix; clipped to [0.6, 5] as in gasoline."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        hi = h[i]
        ih2 = 1.0 / (hi * hi)
        fn1 = M_1_PI * ih2 * ih2 / hi
        if isph:
            cxx, cxy, cxz = cmat[i, 0], cmat[i, 1], cmat[i, 2]
            cyy, cyz, czz = cmat[i, 3], cmat[i, 4], cmat[i, 5]
        else:
            cxx, cxy, cxz, cyy, cyz, czz = 1.0, 0.0, 0.0, 1.0, 0.0, 1.0
        tr = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > hbx:
                    dx -= bx
                elif dx < -hbx:
                    dx += bx
                if dy > hby:
                    dy -= by
                elif dy < -hby:
                    dy += by
                if dz > hbz:
                    dz -= bz
                elif dz < -hbz:
                    dz += bz
            ar2 = (dx * dx + dy * dy + dz * dz) * ih2
            if ar2 >= 4.0:
                continue
            if sigf == 1.0:
                vol = mass[j] / rho[j]
            elif sigf == 0.0:
                vol = mass[j] / rho[i]
            else:
                vol = mass[j] / (rho[i] ** (1.0 - sigf) * rho[j] ** sigf)
            rq = -dw_k(kid, kp, ar2) * fn1 * vol
            tr += rq * (dx * (cxx * dx + cxy * dy + cxz * dz)
                        + dy * (cxy * dx + cyy * dy + cyz * dz)
                        + dz * (cxz * dx + cyz * dy + czz * dz))
        f = 3.0 / tr if tr > 0.0 else 1.0
        fac[i] = min(max(f, 0.6), 5.0)


def _gradh_pP(state, isph, cm, pos, mass, h, rho, indptr, indices, box, periodic,
              sigf, kid, kp, pP):
    """pP times the grad-h owner factor: in the one-sided operator each window term
    carries only its owner's pressure, so gasoline's per-owner factor is pP_k -> f_k pP_k."""
    fac = state.get('gradh')
    if fac is None or len(fac) != len(pos):
        fac = state['gradh'] = np.empty(len(pos))
    gradh_factor(pos, mass, h, rho, indptr, indices, box, periodic, sigf, kid, kp,
                 isph, cm, fac)
    return pP * fac


@njit(cache=True, inline='always')
def _sep(isph, cmat, k, dx, dy, dz):
    """The vector a pair term multiplies: C_k dx (isph, window owner k) or dx."""
    if isph:
        return (cmat[k, 0] * dx + cmat[k, 1] * dy + cmat[k, 2] * dz,
                cmat[k, 1] * dx + cmat[k, 3] * dy + cmat[k, 4] * dz,
                cmat[k, 2] * dx + cmat[k, 4] * dy + cmat[k, 5] * dz)
    return dx, dy, dz


@njit(cache=True, inline='always')
def _pref(own, g_own, g_other):
    """Prefactor on the half-normalised window: avg sums both members' terms
    (arithmetic-mean kernel), own (ONESIDED) keeps the window owner's term at
    full weight."""
    if own:
        return 2.0 * g_own
    return g_own + g_other


@njit(cache=True, parallel=True)
def _force_loop(sigf, own, isph, kid, kp, w0, pos, mass, h, rho, pP, cmat,
                indptr, indices,
                rev_indptr, rev_indices, one_sided,
                box, periodic, icvel, q0d, e0, diag):
    """ICGenerator pair force, gather-only, over CSR neighbour lists.

    own-ball side: for each j in indices[indptr[i]:indptr[i+1]] evaluate the
    h_i kernel (i is 'p'). The h_j side (i is 'q', owner j's kernel) is then
    supplied either
      - one_sided=False: from the transposed graph rev_indices[i] (the j's
        whose own ball contains i) - the exact arithmetic-mean symmetrization;
      - one_sided=True : inline from i's OWN list (same j's, kernel h_j), no
        transpose - SPH-EXA style; misses only pairs where i sits in j's ball
        but j is beyond i's list (h_j/h_i large; kernel-edge terms).
    own selects the window rule (_pref), isph the matrix-corrected
    separation (_sep; cmat unused when False). Density is independent of these choices (it is computed
    separately from the own list); only the FORCE differs. Each prange thread
    writes only icvel[i] (+ q0d[i]/e0[i] when diag) - no races."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        hi = h[i]
        ih2 = 1.0 / (hi * hi)
        fnorm1_i = 0.5 * M_1_PI * ih2 * ih2 / hi  # 0.5/(pi h^5)
        ax = 0.0
        ay = 0.0
        az = 0.0
        q0 = 0.0
        e0x = 0.0
        e0y = 0.0
        e0z = 0.0
        # own-ball side: i is "p", kernel h_i (+ inline h_j side if one_sided)
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > hbx:
                    dx -= bx
                elif dx < -hbx:
                    dx += bx
                if dy > hby:
                    dy -= by
                elif dy < -hby:
                    dy += by
                if dz > hbz:
                    dz -= bz
                elif dz < -hbz:
                    dz += bz
            d2 = dx * dx + dy * dy + dz * dz
            ar2 = d2 * ih2
            if ar2 < 4.0:
                rq = dw_k(kid, kp, ar2) * fnorm1_i * mass[j]
                ig_i, ig_j = ig_pair(sigf, pP[i], pP[j], rho[i], rho[j])
                gx, gy, gz = _sep(isph, cmat, i, dx, dy, dz)
                s = _pref(own, ig_i, ig_j) * rq
                ax -= s * gx
                ay -= s * gy
                az -= s * gz
                if diag:
                    q0 += mass[j] / rho[j] * w_k(kid, kp, ar2)
                    g_i, g_j = ig_pair(sigf, 1.0, 1.0, rho[i], rho[j])
                    se = _pref(own, g_i, g_j) * rq
                    e0x += se * gx
                    e0y += se * gy
                    e0z += se * gz
            if one_sided:
                # neighbor-h side from i's own list (the term the symmetric
                # walk delivers via the transposed graph)
                hj = h[j]
                ihj2 = 1.0 / (hj * hj)
                ar2j = d2 * ihj2
                if ar2j < 4.0:
                    rp = dw_k(kid, kp, ar2j) * (0.5 * M_1_PI * ihj2 * ihj2 / hj) \
                        * mass[j]
                    ig_j, ig_i = ig_pair(sigf, pP[j], pP[i], rho[j], rho[i])
                    gx, gy, gz = _sep(isph, cmat, j, dx, dy, dz)
                    s = _pref(own, ig_j, ig_i) * rp
                    ax -= s * gx
                    ay -= s * gy
                    az -= s * gz
                    if diag:
                        g_j, g_i = ig_pair(sigf, 1.0, 1.0, rho[j], rho[i])
                        se = _pref(own, g_j, g_i) * rp
                        e0x += se * gx
                        e0y += se * gy
                        e0z += se * gz
        # transposed side: i is "q", ball owner j is "p", kernel h_j
        if not one_sided:
            for ptr in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[ptr]
                hj = h[j]
                ihj2 = 1.0 / (hj * hj)
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                if periodic:
                    if dx > hbx:
                        dx -= bx
                    elif dx < -hbx:
                        dx += bx
                    if dy > hby:
                        dy -= by
                    elif dy < -hby:
                        dy += by
                    if dz > hbz:
                        dz -= bz
                    elif dz < -hbz:
                        dz += bz
                ar2 = (dx * dx + dy * dy + dz * dz) * ihj2
                if ar2 >= 4.0:
                    continue
                rp = dw_k(kid, kp, ar2) * (0.5 * M_1_PI * ihj2 * ihj2 / hj) * mass[j]
                ig_j, ig_i = ig_pair(sigf, pP[j], pP[i], rho[j], rho[i])
                gx, gy, gz = _sep(isph, cmat, j, dx, dy, dz)
                s = _pref(own, ig_j, ig_i) * rp
                # C: ICvel[q] += (...) * rp * dx_pq with dx_pq = r_p - r_q = -dx
                ax -= s * gx
                ay -= s * gy
                az -= s * gz
                if diag:
                    g_j, g_i = ig_pair(sigf, 1.0, 1.0, rho[j], rho[i])
                    se = _pref(own, g_j, g_i) * rp
                    e0x += se * gx
                    e0y += se * gy
                    e0z += se * gz
        icvel[i, 0] = ax
        icvel[i, 1] = ay
        icvel[i, 2] = az
        if diag:
            hi3 = hi * hi * hi
            q0d[i] = (q0 + mass[i] / rho[i] * w0) * M_1_PI / hi3
            e0[i, 0] = rho[i] * hi * e0x
            e0[i, 1] = rho[i] * hi * e0y
            e0[i, 2] = rho[i] * hi * e0z


def _cmat(state, isph, pos, mass, h, rho, indptr, indices, box, periodic,
          sigf, kid, kp):
    """This call's ISPH matrices (buffer reused across calls), or the unused
    dummy for the plain gradient."""
    if not isph:
        return _NOCMAT
    cm = state.get('cmat')
    if cm is None or len(cm) != len(pos):
        cm = state['cmat'] = np.empty((len(pos), 6))
    isph_matrix(pos, mass, h, rho, indptr, indices, box, periodic, sigf, kid,
                kp, cm, ISPH_BOUNDTERM)
    return cm


def get_force_loop_csr(sigma_f=1.0, gradient='onesided-isph', one_sided=False,
                       kernel='wendlandc2', nsmooth=64):
    """CSR (ball-gather) force loop bound to sigma_f, a GRADIENTS name, the
    one_sided (transpose-free walk) flag and a kernels.KERNELS name (nsmooth
    selects the kerneloptimized coefficient set)."""
    own, isph = _gradient_spec(gradient)
    kid, kp, _, w0 = kernel_spec(kernel, nsmooth)
    sigf = float(sigma_f)
    osf = one_sided
    state = {}
    gradh = GRADH
    if gradh and not own:
        raise ValueError('the grad-h factor needs a one-sided gradient')

    def force_loop(pos, mass, h, rho, pP, indptr, indices, rev_indptr,
                   rev_indices, box, periodic, icvel, q0d, e0, diag):
        cm = _cmat(state, isph, pos, mass, h, rho, indptr, indices, box,
                   periodic, sigf, kid, kp)
        if gradh:
            pP = _gradh_pP(state, isph, cm, pos, mass, h, rho, indptr, indices,
                           box, periodic, sigf, kid, kp, pP)
        return _force_loop(sigf, own, isph, kid, kp, w0, pos, mass, h, rho, pP,
                           cm, indptr, indices, rev_indptr, rev_indices, osf,
                           box, periodic, icvel, q0d, e0, diag)
    return force_loop


def get_force_loop(sigma_f=1.0, gradient='onesided-isph', one_sided=False,
                   kernel='wendlandc2', nsmooth=64):
    """kNN (dense) force loop, arguments as get_force_loop_csr. The dense
    (N, k) candidate array is presented to the shared CSR _force_loop as
    indptr=arange*k, indices=idx.ravel() (a view; identical iteration order, so
    results are bit-identical to a per-particle range(k) walk)."""
    own, isph = _gradient_spec(gradient)
    kid, kp, _, w0 = kernel_spec(kernel, nsmooth)
    sigf = float(sigma_f)
    osf = one_sided
    state = {}
    gradh = GRADH
    if gradh and not own:
        raise ValueError('the grad-h factor needs a one-sided gradient')

    def force_loop(pos, mass, h, rho, pP, idx, rev_indptr, rev_indices,
                   box, periodic, icvel, q0d, e0, diag):
        n, k = idx.shape
        if state.get('key') != (n, k):
            state['indptr'] = np.arange(0, (n + 1) * k, k, dtype=np.int64)
            state['key'] = (n, k)
        indices = idx.reshape(-1)
        cm = _cmat(state, isph, pos, mass, h, rho, state['indptr'], indices,
                   box, periodic, sigf, kid, kp)
        if gradh:
            pP = _gradh_pP(state, isph, cm, pos, mass, h, rho, state['indptr'],
                           indices, box, periodic, sigf, kid, kp, pP)
        return _force_loop(sigf, own, isph, kid, kp, w0, pos, mass, h, rho, pP,
                           cm, state['indptr'], indices, rev_indptr,
                           rev_indices, osf, box, periodic, icvel, q0d, e0,
                           diag)
    return force_loop
