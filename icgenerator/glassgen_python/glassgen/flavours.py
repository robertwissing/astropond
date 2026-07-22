"""SPH flavour registry: the discrete gradient-operator family per flavour.

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
"""
from numba import njit

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
# axes (e.g. mhd_divBclean) select from here.
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
