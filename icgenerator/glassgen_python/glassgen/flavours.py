"""SPH flavour registry: the discrete operator FAMILY per flavour.

A flavour is an integer id dispatched inside the inlined helpers below,
mirroring ICGenerator (smoothfcn.c): GDFORCE uses pP/(rho_i rho_j) (the
geometric density average, Wadsley+ 2017 / Wissing & Shen 2020), standard
uses pP/rho^2. Integer dispatch (rather than inlining per-flavour closures
via a factory) keeps every consumer loop a top-level @njit(cache=True)
function, so it compiles once ever instead of on every process start.

A flavour defines BOTH members of the discrete operator pair:

  SYMMETRIC (conservative) gradient - the force/induction side; ig_pair
  gives its pair prefactors (g_i, g_j):

      (grad f)_i = rho_i sum_j m_j (g_i + g_j) gradWbar_ij

  DIFFERENCE (constant-exact) gradient/divergence - the estimate side;
  diff_weight gives its pair weight D_ij:

      (grad f)_i = sum_j m_j D_ij (f_j - f_i) gradWbar_ij
      (div  v)_i = sum_j m_j D_ij (v_j - v_i) . gradWbar_ij

  with gradWbar_ij = 0.5 [gradW_ij(h_i) + gradW_ij(h_j)] in both (the
  averaged kernel makes each pair term antisymmetric under i<->j, which
  the adjointness below needs; sph_core's two-window candidate/transpose
  walk delivers exactly these halves).

The two are exact NEGATIVE ADJOINTS under the volume inner product
<u, v> = sum_i (m_i/rho_i) u_i v_i for any flavour whose ig_pair has the
structure g_i = f_i F(rho_i, rho_j), g_j = f_j F(rho_j, rho_i) - then
D_ij = rho_i F(rho_i, rho_j) = rho_i * ig_pair(1, .)[0], which is what
diff_weight returns. Consumers that need an energy-conserving pair (e.g.
the mhd_divBclean psi <-> B exchange) must take BOTH operators from this
registry; a new flavour added here inherits its conjugate difference form
automatically. gdforce: D_ij = 1/rho_j (volume-weighted); rho2:
D_ij = 1/rho_i (the Tricco & Price / Phantom convention).

Future ISPH flavours additionally replace the separation vector dx by the
matrix-corrected -(C @ dx) and weight pairs by the kernel value instead of
its derivative; the force loop in sph_core reserves the cmat slot for that
(such a flavour breaks the F-structure above - it must then also override
diff_weight explicitly to keep the adjoint guarantee).

===========================================================================
THE OPERATOR FAMILY
===========================================================================

The second half of this module instantiates that algebra as ready-to-call
particle operators, so a consumer never re-derives a pair walk:

              scalar f -> vector     vector V -> scalar
              vector V -> vector     vector V -> tensor
              scalar/vector -> itself                (interp)
    ---------------------------------------------------------------
    difference   grad      div        curl      gradtensor    interp
    symmetric    grad      div        curl         -             -

selected by (form, op) through one entry point::

    from glassgen.flavours import sph_operator, operator_state

    st = operator_state(pos, mass, box=box, nsmooth=64)   # neighbours + rho,h
    B,    st = sph_operator('curl', pos, mass, A, flavour='gdforce', state=st)
    divB, st = sph_operator('div',  pos, mass, B, flavour='gdforce', state=st)

`state` caches the density estimate and the candidate list + transpose, so a
sequence of operators over the same frozen configuration costs one build.
Thin aliases (:func:`curl`, :func:`divergence`, :func:`gradient`,
:func:`grad_tensor`) wrap the same dispatch for readability.

`interp` is the ZEROTH member: the Shepard-normalised SPH interpolant
(kernel smoothing), which weights by the kernel VALUE rather than its
gradient. It is filed with the difference operators because it carries the
same pair weight D_ij and shares their constant-exactness; there is no
conservative counterpart to pair it with, so ('symmetric', 'interp') does
not exist.

Every kernel is the SAME pair walk - only the accumulation differs - so all
of them share :func:`_pair_term` (gradient window) or its value companion
:func:`_pair_value`, which own the one copy of the minimum-image separation
and the Wendland C2 window factors (glassgen convention, support 2h, no
grad-h Omega terms; the kernel shape itself lives in kernels.py). Adding a
member of the family means writing an accumulation loop, not a neighbour walk.
In every kernel the h_i halves come from i's own candidate list and the h_j
halves from the transpose (pairs where i sits inside j's support), so an
asymmetric candidate list still yields the symmetrised pair sum.

WHAT THE FAMILY GUARANTEES (and what it does not):

- The difference operators annihilate a CONSTANT field exactly, in either
  kernel mode, including against one-sided kernels at an open boundary. The
  interpolant's counterpart of that guarantee is that it REPRODUCES a
  constant exactly, which is what its Shepard normalisation buys - an
  unnormalised SPH sum does not, least of all at an open boundary where the
  neighbour weights no longer sum to one.
- A LINEAR field (e.g. the uniform-field potential A = 1/2 B0 x r) is
  reproduced only to the accuracy of the distribution's normalisation
  condition sum_j m_j D_ij gradWbar_ij (x) dx_ij = I. The DIRECTION comes
  back essentially exactly; the AMPLITUDE is systematically LOW by the
  shortfall of that sum - ~2% at nsmooth=64 and ~0.03% at nsmooth=128 even
  on a PERFECT lattice (so it is a finite-neighbour-sum property, not a
  disorder one), growing to ~5% for a strongly disordered distribution at
  nsmooth=64. A consumer that needs a prescribed field STRENGTH should
  renormalise after the curl (IC_setBField.py does); a consumer that needs
  linear exactness needs the matrix-corrected dx of the ISPH note above.
- A linear A is NOT PERIODIC: in a periodic box the minimum-image walk sees
  an O(L) jump in A across the wrap and the boundary shell is meaningless.
  Uniform components of a periodic field must be carried analytically
  (cf. sph_interp.vector_potential's Bbar split), not through a linear A.
- div(curl) is NOT identically zero for these operators (that identity needs
  a De Rham complex, which the meshfree pair is not). B = curl A therefore
  carries a small divergence residual - typically an order of magnitude below
  a directly prescribed field, and removable to any target afterwards with
  mhd_divBclean, whose cleaning uses the divergence/symmetric-gradient members
  of THIS registry.
"""
import numpy as np
from numba import njit, prange

from .kernels import dw_wc2, w_wc2

M_1_PI = 1.0 / np.pi
DIVBERR_FLOOR_FRACTION = 1.0e-12

# weight-family ids consumed by ig_pair/diff_weight (the force loop's axis)
FLAVOURS = {
    'gdforce': 0,
    'rho2': 1,
}

# kernel-window mode ids: how the two smoothing-length windows combine.
#   avg: every pair term uses gradWbar = 0.5[gradW(h_i)+gradW(h_j)] - the
#        arithmetic-mean symmetrization sph_core's force loop implements
#        (fnorm1 carries the 0.5, transpose walk supplies the h_j half).
#   own: Phantom/Tricco pairing - the symmetric gradient couples g_i to
#        gradW(h_i) and g_j to gradW(h_j); its conjugate difference
#        operator then uses the OWN (h_i, gather) window only. D_ij is
#        UNCHANGED between modes; only the window rule moves.
KMODE_AVG = 0
KMODE_OWN = 1

# full flavour registry: name -> (weight id, kernel-window mode). The force
# loop (sph_core) implements only the avg mode today, so FLAVOURS above
# stays its authoritative choice list; operator consumers that honour both
# axes (e.g. mhd_divBclean, the operator family below) select from here.
FLAVOUR_SPECS = {
    'gdforce': (FLAVOURS['gdforce'], KMODE_AVG),
    'rho2': (FLAVOURS['rho2'], KMODE_AVG),
    'gdforce-own': (FLAVOURS['gdforce'], KMODE_OWN),
    'rho2-own': (FLAVOURS['rho2'], KMODE_OWN),
}


def flavour_spec(name):
    """(weight id, kernel-mode id) for a FLAVOUR_SPECS name (with a clear
    error). The rho2-own pair is the Tricco & Price / Phantom convention;
    gdforce (avg) is the gasolinemhd/GDSPH one."""
    try:
        return FLAVOUR_SPECS[name]
    except KeyError:
        raise ValueError(f'unknown SPH flavour {name!r} '
                         f'(registry: {sorted(FLAVOUR_SPECS)})') from None


@njit(cache=True, inline='always')
def ig_pair(iflav, pP_i, pP_j, rho_i, rho_j):
    """(g_i, g_j) prefactors of the flavour's SYMMETRIC gradient pair term."""
    if iflav == 0:  # gdforce
        ig = 1.0 / (rho_i * rho_j)
        return pP_i * ig, pP_j * ig
    # rho2
    return pP_i / (rho_i * rho_i), pP_j / (rho_j * rho_j)


@njit(cache=True, inline='always')
def diff_weight(iflav, rho_i, rho_j):
    """D_ij of the flavour's energy-conjugate DIFFERENCE operator (see the
    module docstring). Derived from ig_pair so the adjoint identity holds by
    construction for every registry flavour of the standard F-structure."""
    g_i, _ = ig_pair(iflav, 1.0, 0.0, rho_i, rho_j)
    return rho_i * g_i


# ===========================================================================
# the shared pair primitive
# ===========================================================================

@njit(cache=True, inline='always')
def _pair_term(pos, i, j, hw, box, periodic):
    """Minimum-image separation dx_ij = r_i - r_j and the window factor g
    such that gradW_ij(hw) = g * dx_ij, for the Wendland C2 window hw.

    Returns (dx, dy, dz, g); g == 0.0 outside the support, which every caller
    uses to skip the accumulation. This is the ONLY place the separation
    convention and the kernel normalisation 1/(pi h^5) are written down.
    """
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dz = pos[i, 2] - pos[j, 2]
    if periodic:
        if dx > 0.5 * box[0]:
            dx -= box[0]
        elif dx < -0.5 * box[0]:
            dx += box[0]
        if dy > 0.5 * box[1]:
            dy -= box[1]
        elif dy < -0.5 * box[1]:
            dy += box[1]
        if dz > 0.5 * box[2]:
            dz -= box[2]
        elif dz < -0.5 * box[2]:
            dz += box[2]
    ih2 = 1.0 / (hw * hw)
    fn1 = M_1_PI * ih2 * ih2 / hw            # 1/(pi h^5)
    ar2 = (dx * dx + dy * dy + dz * dz) * ih2
    return dx, dy, dz, dw_wc2(ar2) * fn1


@njit(cache=True, inline='always')
def _pair_value(pos, i, j, hw, box, periodic):
    """The Wendland C2 kernel VALUE W_ij(hw), minimum-imaged; 0 outside 2hw.

    The value companion of :func:`_pair_term`: same separation convention,
    the other normalisation, 1/(pi h^3). Only the weight is returned - a
    value operator never needs the separation vector. Unlike the gradient
    window this is FINITE at zero separation, so the self pair j == i is a
    real contribution and every interpolant must add it explicitly (the
    candidate lists exclude self).
    """
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dz = pos[i, 2] - pos[j, 2]
    if periodic:
        if dx > 0.5 * box[0]:
            dx -= box[0]
        elif dx < -0.5 * box[0]:
            dx += box[0]
        if dy > 0.5 * box[1]:
            dy -= box[1]
        elif dy < -0.5 * box[1]:
            dy += box[1]
        if dz > 0.5 * box[2]:
            dz -= box[2]
        elif dz < -0.5 * box[2]:
            dz += box[2]
    ih2 = 1.0 / (hw * hw)
    ar2 = (dx * dx + dy * dy + dz * dz) * ih2
    return w_wc2(ar2) * M_1_PI * ih2 / hw    # 1/(pi h^3)


# ===========================================================================
# DIFFERENCE form: constant-exact estimate operators
# ===========================================================================

@njit(parallel=True, cache=True)
def diff_gradient(pos, mass, rho, h, f, iflav, ikmode, idx, rev_indptr,
                  rev_indices, box, periodic, out):
    """(grad f)_i = sum_j m_j D_ij (f_j - f_i) gradWbar_ij   -> out (N,3)."""
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                w = mass[j] * diff_weight(iflav, rho[i], rho[j]) * g * kfi
                df = f[j] - f[i]
                sx += w * df * dx
                sy += w * df * dy
                sz += w * df * dz
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
                if g != 0.0:
                    w = 0.5 * mass[j] * diff_weight(iflav, rho[i], rho[j]) * g
                    df = f[j] - f[i]
                    sx += w * df * dx
                    sy += w * df * dy
                    sz += w * df * dz
        out[i, 0] = sx
        out[i, 1] = sy
        out[i, 2] = sz


@njit(parallel=True, cache=True)
def diff_divergence(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
                    rev_indices, box, periodic, out):
    """(div V)_i = sum_j m_j D_ij (V_j - V_i) . gradWbar_ij   -> out (N,).

    The divergence mhd_divBclean sources psi from, and the one a cleaned IC
    is clean in. Annihilates a constant field exactly in either kernel mode.
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        s = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                s += mass[j] * diff_weight(iflav, rho[i], rho[j]) * g * kfi * (
                    (V[j, 0] - V[i, 0]) * dx
                    + (V[j, 1] - V[i, 1]) * dy
                    + (V[j, 2] - V[i, 2]) * dz)
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
                if g != 0.0:
                    s += 0.5 * mass[j] * diff_weight(
                        iflav, rho[i], rho[j]) * g * (
                        (V[j, 0] - V[i, 0]) * dx
                        + (V[j, 1] - V[i, 1]) * dy
                        + (V[j, 2] - V[i, 2]) * dz)
        out[i] = s


@njit(parallel=True, cache=True)
def diff_curl(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
              rev_indices, box, periodic, out):
    """(curl V)_i = sum_j m_j D_ij [gradWbar_ij x (V_j - V_i)]  -> out (N,3).

    The componentwise partner of diff_divergence - same weight D_ij, same
    window rule, same pair antisymmetry - with the contraction of gradWbar
    against (V_j - V_i) replaced by their cross product. This is the operator
    that turns a vector potential into B = curl A in the flavour the consumer
    code will later measure div B with.
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                w = mass[j] * diff_weight(iflav, rho[i], rho[j]) * g * kfi
                ax = V[j, 0] - V[i, 0]
                ay = V[j, 1] - V[i, 1]
                az = V[j, 2] - V[i, 2]
                cx += w * (dy * az - dz * ay)
                cy += w * (dz * ax - dx * az)
                cz += w * (dx * ay - dy * ax)
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
                if g != 0.0:
                    w = 0.5 * mass[j] * diff_weight(iflav, rho[i], rho[j]) * g
                    ax = V[j, 0] - V[i, 0]
                    ay = V[j, 1] - V[i, 1]
                    az = V[j, 2] - V[i, 2]
                    cx += w * (dy * az - dz * ay)
                    cy += w * (dz * ax - dx * az)
                    cz += w * (dx * ay - dy * ax)
        out[i, 0] = cx
        out[i, 1] = cy
        out[i, 2] = cz


@njit(parallel=True, cache=True)
def diff_grad_tensor(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
                     rev_indices, box, periodic, out):
    """T[i,c,d] = (d V_c / d x_d)_i = sum_j m_j D_ij (V_j-V_i)_c gradWbar_d.

    The PARENT of diff_divergence (trace) and diff_curl (antisymmetric part:
    curl_x = T[2,1]-T[1,2] and cyclic), kept separate because the specialised
    two carry ~1/3 of the flop count and sit in hot loops. Use this one when
    the full velocity/field gradient is wanted (shear, vorticity + divergence
    together, Balsara-style switches). -> out (N,3,3).
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        for c in range(3):
            for d in range(3):
                out[i, c, d] = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                w = mass[j] * diff_weight(iflav, rho[i], rho[j]) * g * kfi
                for c in range(3):
                    dv = w * (V[j, c] - V[i, c])
                    out[i, c, 0] += dv * dx
                    out[i, c, 1] += dv * dy
                    out[i, c, 2] += dv * dz
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
                if g != 0.0:
                    w = 0.5 * mass[j] * diff_weight(iflav, rho[i], rho[j]) * g
                    for c in range(3):
                        dv = w * (V[j, c] - V[i, c])
                        out[i, c, 0] += dv * dx
                        out[i, c, 1] += dv * dy
                        out[i, c, 2] += dv * dz


@njit(parallel=True, cache=True)
def interp_scalar(pos, mass, rho, h, f, iflav, ikmode, idx, rev_indptr,
                  rev_indices, box, periodic, out):
    """<f>_i = sum_j m_j D_ij f_j Wbar_ij / sum_j m_j D_ij Wbar_ij  -> out (N,).

    The Shepard-normalised SPH interpolant: kernel smoothing of a field that
    already lives on the particles. The pair weight is the family's D_ij, so
    with gdforce (D_ij = 1/rho_j) the numerator is the textbook interpolant
    sum_j (m_j/rho_j) f_j W_ij, and with rho2 (D_ij = 1/rho_i) it is the
    density-normalised form; the Shepard denominator makes the two agree on a
    constant field and differ only in how they weight a density contrast.

    THE SELF PAIR IS INCLUDED (i's own value, at the kernel's peak weight
    W(0) = 21/(16 pi h^3)) - it is the single largest term, and the candidate
    lists exclude it, so it is added by hand here. Dropping it does not
    cancel against the Shepard denominator: it turns the operator into "the
    mean of my neighbours, excluding me", which erases a particle's own value
    in one pass instead of smoothing it.

    Isolated particles (no neighbour inside 2h and zero self weight cannot
    both happen, so the guard only fires on a non-positive rho) keep f_i.
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        # Wbar_ii = 0.5[W(h_i) + W(h_i)] = W(h_i): full weight in both modes
        s = mass[i] * diff_weight(iflav, rho[i], rho[i]) * _pair_value(
            pos, i, i, h[i], box, periodic)
        acc = s * f[i]
        for jj in range(k):
            j = idx[i, jj]
            wv = _pair_value(pos, i, j, h[i], box, periodic)
            if wv != 0.0:
                w = mass[j] * diff_weight(iflav, rho[i], rho[j]) * wv * kfi
                s += w
                acc += w * f[j]
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                wv = _pair_value(pos, i, j, h[j], box, periodic)
                if wv != 0.0:
                    w = 0.5 * mass[j] * diff_weight(iflav, rho[i], rho[j]) * wv
                    s += w
                    acc += w * f[j]
        out[i] = acc / s if s > 0.0 else f[i]


@njit(parallel=True, cache=True)
def interp_vector(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
                  rev_indices, box, periodic, out):
    """Componentwise :func:`interp_scalar` on an (N,3) field -> out (N,3).

    One shared Shepard denominator for the three components, so the smoothed
    vector is a convex combination of the neighbours' vectors and cannot be
    rotated by the normalisation.
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        kfi = 0.5 if avg else 1.0
        s = mass[i] * diff_weight(iflav, rho[i], rho[i]) * _pair_value(
            pos, i, i, h[i], box, periodic)
        ax = s * V[i, 0]
        ay = s * V[i, 1]
        az = s * V[i, 2]
        for jj in range(k):
            j = idx[i, jj]
            wv = _pair_value(pos, i, j, h[i], box, periodic)
            if wv != 0.0:
                w = mass[j] * diff_weight(iflav, rho[i], rho[j]) * wv * kfi
                s += w
                ax += w * V[j, 0]
                ay += w * V[j, 1]
                az += w * V[j, 2]
        if avg:
            for p in range(rev_indptr[i], rev_indptr[i + 1]):
                j = rev_indices[p]
                wv = _pair_value(pos, i, j, h[j], box, periodic)
                if wv != 0.0:
                    w = 0.5 * mass[j] * diff_weight(iflav, rho[i], rho[j]) * wv
                    s += w
                    ax += w * V[j, 0]
                    ay += w * V[j, 1]
                    az += w * V[j, 2]
        if s > 0.0:
            out[i, 0] = ax / s
            out[i, 1] = ay / s
            out[i, 2] = az / s
        else:
            out[i, 0] = V[i, 0]
            out[i, 1] = V[i, 1]
            out[i, 2] = V[i, 2]


def interp_any(pos, mass, rho, h, field, iflav, ikmode, idx, rev_indptr,
               rev_indices, box, periodic, out):
    """Arity dispatch for 'interp', in the uniform kernel signature.

    The only family member whose output shape follows its input, so the
    registry entry points here and the scalar/vector kernels stay separate
    njit functions (numba specialises on the array rank anyway).
    """
    kernel = interp_scalar if field.ndim == 1 else interp_vector
    kernel(pos, mass, rho, h, field, iflav, ikmode, idx, rev_indptr,
           rev_indices, box, periodic, out)


# ===========================================================================
# SYMMETRIC form: conservative force/induction operators
# ===========================================================================

@njit(parallel=True, cache=True)
def sym_gradient(pos, mass, rho, h, f, iflav, ikmode, idx, rev_indptr,
                 rev_indices, box, periodic, out):
    """(grad f)_i = rho_i sum_j m_j PAIR gradW-terms              -> out (N,3).

      avg: PAIR = (g_i + g_j) 0.5[gradW(h_i)+gradW(h_j)]   (gasolinemhd/GDSPH)
      own: PAIR = g_i gradW(h_i) + g_j gradW(h_j)          (Phantom/Tricco)

    The exact negative adjoint of diff_divergence under the volume inner
    product - the property mhd_divBclean's energy proof rests on.
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                g_i, g_j = ig_pair(iflav, f[i], f[j], rho[i], rho[j])
                pref = 0.5 * (g_i + g_j) if avg else g_i
                w = mass[j] * pref * g
                sx += w * dx
                sy += w * dy
                sz += w * dz
        for p in range(rev_indptr[i], rev_indptr[i + 1]):
            j = rev_indices[p]
            dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
            if g != 0.0:
                g_i, g_j = ig_pair(iflav, f[i], f[j], rho[i], rho[j])
                pref = 0.5 * (g_i + g_j) if avg else g_j
                w = mass[j] * pref * g
                sx += w * dx
                sy += w * dy
                sz += w * dz
        out[i, 0] = rho[i] * sx
        out[i, 1] = rho[i] * sy
        out[i, 2] = rho[i] * sz


@njit(parallel=True, cache=True)
def sym_divergence(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
                   rev_indices, box, periodic, out):
    """Conservative divergence, the componentwise dot of sym_gradient's pair.

    (div V)_i = rho_i sum_j m_j (F_ij V_i + F_ji V_j) . gradWbar_ij, with
    (F_ij, F_ji) = ig_pair(1, 1) the flavour's density prefactors. NOT
    constant-exact (a constant field leaves a surface term) - that is the
    price of conservation; use diff_divergence for ESTIMATES (div B errors)
    and this one only where the momentum/energy budget must close. -> (N,).
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        s = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                F_ij, F_ji = ig_pair(iflav, 1.0, 1.0, rho[i], rho[j])
                if avg:
                    px = 0.5 * (F_ij * V[i, 0] + F_ji * V[j, 0])
                    py = 0.5 * (F_ij * V[i, 1] + F_ji * V[j, 1])
                    pz = 0.5 * (F_ij * V[i, 2] + F_ji * V[j, 2])
                else:
                    px = F_ij * V[i, 0]
                    py = F_ij * V[i, 1]
                    pz = F_ij * V[i, 2]
                s += mass[j] * g * (px * dx + py * dy + pz * dz)
        for p in range(rev_indptr[i], rev_indptr[i + 1]):
            j = rev_indices[p]
            dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
            if g != 0.0:
                F_ij, F_ji = ig_pair(iflav, 1.0, 1.0, rho[i], rho[j])
                if avg:
                    px = 0.5 * (F_ij * V[i, 0] + F_ji * V[j, 0])
                    py = 0.5 * (F_ij * V[i, 1] + F_ji * V[j, 1])
                    pz = 0.5 * (F_ij * V[i, 2] + F_ji * V[j, 2])
                else:
                    px = F_ji * V[j, 0]
                    py = F_ji * V[j, 1]
                    pz = F_ji * V[j, 2]
                s += mass[j] * g * (px * dx + py * dy + pz * dz)
        out[i] = rho[i] * s


@njit(parallel=True, cache=True)
def sym_curl(pos, mass, rho, h, V, iflav, ikmode, idx, rev_indptr,
             rev_indices, box, periodic, out):
    """Conservative curl, the cross-product partner of sym_divergence.

    (curl V)_i = rho_i sum_j m_j [gradWbar_ij x (F_ij V_i + F_ji V_j)].
    Same caveat as sym_divergence: conservative, not constant-exact. -> (N,3).
    """
    n, k = idx.shape
    avg = ikmode == 0
    for i in prange(n):
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for jj in range(k):
            j = idx[i, jj]
            dx, dy, dz, g = _pair_term(pos, i, j, h[i], box, periodic)
            if g != 0.0:
                F_ij, F_ji = ig_pair(iflav, 1.0, 1.0, rho[i], rho[j])
                if avg:
                    px = 0.5 * (F_ij * V[i, 0] + F_ji * V[j, 0])
                    py = 0.5 * (F_ij * V[i, 1] + F_ji * V[j, 1])
                    pz = 0.5 * (F_ij * V[i, 2] + F_ji * V[j, 2])
                else:
                    px = F_ij * V[i, 0]
                    py = F_ij * V[i, 1]
                    pz = F_ij * V[i, 2]
                w = mass[j] * g
                cx += w * (dy * pz - dz * py)
                cy += w * (dz * px - dx * pz)
                cz += w * (dx * py - dy * px)
        for p in range(rev_indptr[i], rev_indptr[i + 1]):
            j = rev_indices[p]
            dx, dy, dz, g = _pair_term(pos, i, j, h[j], box, periodic)
            if g != 0.0:
                F_ij, F_ji = ig_pair(iflav, 1.0, 1.0, rho[i], rho[j])
                if avg:
                    px = 0.5 * (F_ij * V[i, 0] + F_ji * V[j, 0])
                    py = 0.5 * (F_ij * V[i, 1] + F_ji * V[j, 1])
                    pz = 0.5 * (F_ij * V[i, 2] + F_ji * V[j, 2])
                else:
                    px = F_ji * V[j, 0]
                    py = F_ji * V[j, 1]
                    pz = F_ji * V[j, 2]
                w = mass[j] * g
                cx += w * (dy * pz - dz * py)
                cy += w * (dz * px - dx * pz)
                cz += w * (dx * py - dy * px)
        out[i, 0] = rho[i] * cx
        out[i, 1] = rho[i] * cy
        out[i, 2] = rho[i] * cz


# ===========================================================================
# dispatch: (form, op) -> kernel, with the field arity and output shape
# ===========================================================================

#   name -> (kernel, input arity 'scalar'|'vector', output shape code)
OPERATORS = {
    ('difference', 'grad'): (diff_gradient, 'scalar', 'vector'),
    ('difference', 'div'): (diff_divergence, 'vector', 'scalar'),
    ('difference', 'curl'): (diff_curl, 'vector', 'vector'),
    ('difference', 'gradtensor'): (diff_grad_tensor, 'vector', 'tensor'),
    ('difference', 'interp'): (interp_any, 'any', 'same'),
    ('symmetric', 'grad'): (sym_gradient, 'scalar', 'vector'),
    ('symmetric', 'div'): (sym_divergence, 'vector', 'scalar'),
    ('symmetric', 'curl'): (sym_curl, 'vector', 'vector'),
}

FORMS = ('difference', 'symmetric')


def operator_spec(op, form='difference'):
    """(kernel, arity, out-shape) for an OPERATORS entry, with a clear error."""
    try:
        return OPERATORS[(form, op)]
    except KeyError:
        avail = sorted(f'{f}/{o}' for f, o in OPERATORS)
        raise ValueError(
            f'unknown operator {form!r}/{op!r} (family: {avail})') from None


class OperatorState:
    """Frozen-configuration workspace shared by a run of operators.

    Holds the density estimate and the candidate list + its transpose, so a
    sequence of operator calls over the same particle configuration pays for
    one neighbour build. Cheap to pass around; treat as read-only once built.
    """

    __slots__ = ('rho', 'h', 'idx', 'rev_indptr', 'rev_indices', 'box',
                 'periodic', 'nsmooth')

    def __init__(self, rho, h, idx, rev_indptr, rev_indices, box, periodic,
                 nsmooth):
        self.rho = rho
        self.h = h
        self.idx = idx
        self.rev_indptr = rev_indptr
        self.rev_indices = rev_indices
        self.box = box
        self.periodic = periodic
        self.nsmooth = nsmooth


def operator_state(pos, mass, box=None, nsmooth=64, rho=None, h=None):
    """Build the :class:`OperatorState` for a frozen particle configuration.

    rho/h default to glassgen's own kNN density estimate (the same one the
    relax uses), so an operator applied here matches the relax's measure;
    pass them explicitly to reuse a consumer's own (rho, h).
    """
    from .neighbors import NeighborList
    from .reconstruct import particle_density

    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    periodic = box is not None
    boxarr = (np.asarray(box, dtype=np.float64).reshape(3) if periodic
              else np.ones(3))
    if rho is None or h is None:
        rho, h = particle_density(pos, mass, box=box, nsmooth=nsmooth)
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    nl = NeighborList(boxarr if periodic else None, nsmooth,
                      build_transpose=True)
    idx, _, rev_indptr, rev_indices = nl.update(pos)
    return OperatorState(rho, h, idx, rev_indptr, rev_indices, boxarr,
                         periodic, nsmooth)


def sph_operator(op, pos, mass, field, form='difference', flavour='gdforce',
                 box=None, nsmooth=64, rho=None, h=None, state=None):
    """Apply a member of the operator family to `field`. -> (result, state).

    op     : 'grad' | 'div' | 'curl' | 'gradtensor' | 'interp' (see OPERATORS)
    form   : 'difference' (constant-exact estimate) | 'symmetric'
             (conservative). See the module docstring for which to use.
    field  : (N,) for 'grad', (N,3) otherwise; 'interp' takes either and
             returns the same shape.
    state  : reuse an :class:`OperatorState` across calls (built if omitted).

    The returned state is always the one used, so the idiom

        divB, st = sph_operator('div', pos, m, B, box=box)
        B2,   st = sph_operator('curl', pos, m, A, box=box, state=st)

    builds neighbours exactly once.
    """
    kernel, arity, outshape = operator_spec(op, form)
    iflav, ikmode = flavour_spec(flavour)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    field = np.ascontiguousarray(field, dtype=np.float64)
    n = len(mass)
    if pos.shape != (n, 3):
        raise ValueError(f'pos must be (N,3) with N = len(mass) = {n}')
    if arity == 'any':
        if field.shape not in ((n,), (n, 3)):
            raise ValueError(f'{form}/{op} takes a scalar {(n,)} or vector '
                             f'{(n, 3)} field, got {field.shape}')
    else:
        want = (n,) if arity == 'scalar' else (n, 3)
        if field.shape != want:
            raise ValueError(f'{form}/{op} takes a {arity} field of shape '
                             f'{want}, got {field.shape}')
    if state is None:
        state = operator_state(pos, mass, box=box, nsmooth=nsmooth, rho=rho,
                               h=h)
    out = np.empty(field.shape if outshape == 'same'
                   else {'scalar': (n,), 'vector': (n, 3),
                         'tensor': (n, 3, 3)}[outshape])
    kernel(pos, mass, state.rho, state.h, field, iflav, ikmode, state.idx,
           state.rev_indptr, state.rev_indices, state.box, state.periodic, out)
    return out, state


# -- readability aliases over the same dispatch -----------------------------

def curl(pos, mass, V, **kw):
    """B = curl V at the particles (difference form by default)."""
    return sph_operator('curl', pos, mass, V, **kw)


def divergence(pos, mass, V, **kw):
    """div V at the particles (difference form by default)."""
    return sph_operator('div', pos, mass, V, **kw)


def gradient(pos, mass, f, **kw):
    """grad f at the particles (difference form by default)."""
    return sph_operator('grad', pos, mass, f, **kw)


def grad_tensor(pos, mass, V, **kw):
    """The full (N,3,3) gradient tensor dV_c/dx_d (difference form only)."""
    return sph_operator('gradtensor', pos, mass, V, **kw)


def interpolate(pos, mass, field, **kw):
    """Shepard-normalised SPH kernel smoothing of a scalar or vector field.

    One pass returns the interpolant itself; a consumer that wants partial
    smoothing blends toward it ((1-a) f + a <f>) and repeats, which is what
    IC_setQuantity's `--op smooth` does.
    """
    return sph_operator('interp', pos, mass, field, **kw)


def divergence_error(V, divV, h,
                     floor_fraction=DIVBERR_FLOOR_FRACTION):
    """Dimensionless error ``h|div V|/max(|V|, delta*Vrms)`` per particle.

    This is the conventional ``h|div V|/|V|`` away from true field nulls.
    The tiny scale-relative floor, ``delta=1e-12`` by default, only prevents
    division by zero. An identically zero vector field has zero error.
    """
    mag = np.linalg.norm(V, axis=1)
    rms = float(np.sqrt(np.mean(mag * mag)))
    if rms == 0.0:
        return np.zeros_like(mag)
    floor = float(floor_fraction) * rms
    return h * np.abs(divV) / np.maximum(mag, floor)
