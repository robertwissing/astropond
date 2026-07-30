"""The glassgen.flavours operator family: algebra, accuracy and dispatch."""
import numpy as np
import pytest

from glassgen.flavours import (DIVBERR_FLOOR_FRACTION, FLAVOUR_SPECS,
                               OPERATORS, curl, divergence, divergence_error,
                               grad_tensor, gradient, interpolate,
                               operator_state, sph_operator)
from glassgen.kernels import w_wc2

AVG_FLAVOURS = ['gdforce', 'rho2']          # kernel mode with gradWbar
ALL_FLAVOURS = sorted(FLAVOUR_SPECS)


def _jittered(n_side=10, seed=0, box=1.0, jitter=0.2):
    """Jittered lattice in a periodic box: a stand-in for a relaxed glass."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side * box - 0.5 * box
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    pos = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    pos += rng.uniform(-jitter, jitter, pos.shape) * (box / n_side)
    pos = (pos + 0.5 * box) % box - 0.5 * box
    n = len(pos)
    return pos, np.full(n, 1.0 / n)


def test_divergence_error_uses_only_a_tiny_relative_null_floor():
    V = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    divV = np.array([3.0, -4.0])
    h = np.array([5.0, 6.0])
    vrms = np.sqrt(np.mean(np.sum(V * V, axis=1)))
    expected = h * np.abs(divV) / np.array(
        [2.0, DIVBERR_FLOOR_FRACTION * vrms])

    np.testing.assert_allclose(divergence_error(V, divV, h), expected)
    np.testing.assert_allclose(
        divergence_error(7.0 * V, 7.0 * divV, h), expected)
    np.testing.assert_array_equal(
        divergence_error(np.zeros_like(V), np.zeros_like(divV), h),
        np.zeros(len(V)))


# ---------------------------------------------------------------------------
# constant-exactness: the defining property of the DIFFERENCE form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
@pytest.mark.parametrize('op', ['div', 'curl', 'gradtensor'])
def test_difference_operators_annihilate_a_constant_vector_field(flavour, op):
    pos, mass = _jittered(n_side=8)
    V = np.tile([1.0, -2.0, 0.5], (len(pos), 1))
    out, _ = sph_operator(op, pos, mass, V, box=np.ones(3), nsmooth=32,
                          flavour=flavour)
    assert np.max(np.abs(out)) == 0.0


@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
def test_difference_gradient_annihilates_a_constant_scalar(flavour):
    pos, mass = _jittered(n_side=8)
    f = np.full(len(pos), 3.25)
    g, _ = gradient(pos, mass, f, box=np.ones(3), nsmooth=32, flavour=flavour)
    assert np.max(np.abs(g)) == 0.0


def test_constant_field_exact_at_an_open_boundary():
    # one-sided kernels must not break it (no periodic wrap, no box)
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.5, 0.5, (800, 3))
    mass = np.full(len(pos), 1.0 / len(pos))
    V = np.tile([0.3, 0.0, -1.1], (len(pos), 1))
    d, st = divergence(pos, mass, V, box=None, nsmooth=32)
    c, _ = curl(pos, mass, V, state=st)
    assert np.max(np.abs(d)) == 0.0
    assert np.max(np.abs(c)) == 0.0


# ---------------------------------------------------------------------------
# the family is internally consistent: div and curl ARE parts of gradtensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
def test_grad_tensor_trace_is_divergence_and_antisymmetry_is_curl(flavour):
    pos, mass = _jittered(n_side=8, seed=5)
    rng = np.random.default_rng(11)
    V = rng.normal(size=(len(pos), 3))
    box = np.ones(3)
    T, st = grad_tensor(pos, mass, V, box=box, nsmooth=32, flavour=flavour)
    d, _ = divergence(pos, mass, V, state=st, flavour=flavour)
    c, _ = curl(pos, mass, V, state=st, flavour=flavour)

    trace = T[:, 0, 0] + T[:, 1, 1] + T[:, 2, 2]
    np.testing.assert_allclose(trace, d, rtol=1e-12, atol=1e-12)
    # curl_x = dVz/dy - dVy/dz = T[2,1] - T[1,2], and cyclic
    curl_from_T = np.stack([T[:, 2, 1] - T[:, 1, 2],
                            T[:, 0, 2] - T[:, 2, 0],
                            T[:, 1, 0] - T[:, 0, 1]], axis=1)
    np.testing.assert_allclose(curl_from_T, c, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# accuracy: does curl A reproduce the field A was built from?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
def test_curl_of_the_uniform_field_potential_recovers_B0(flavour):
    """A = 1/2 B0 x r  =>  curl A = B0 exactly in the continuum.

    OPEN domain on purpose: this potential is linear in r and therefore NOT
    periodic, so in a periodic box the minimum-image walk sees an O(L) jump in
    A across the wrap and the boundary shell is garbage (errors ~5x B0, vs
    ~0.09 in the interior). A periodic IC wanting a uniform component must
    carry it analytically instead of through a linear A. Interior particles
    only: one-sided kernels at the open edge cannot reproduce a linear field
    either, which is a property of the estimator, not of this potential.
    """
    pos, mass = _jittered(n_side=12, seed=1, jitter=0.15)
    B0 = np.array([0.0, 0.0, 1.0])
    A = 0.5 * np.cross(np.tile(B0, (len(pos), 1)), pos)
    B, st = curl(pos, mass, A, box=None, nsmooth=64, flavour=flavour)
    interior = np.all(np.abs(pos) < 0.5 - 2.5 * st.h.mean(), axis=1)
    assert interior.sum() > 100
    Bi = B[interior]
    err = np.linalg.norm(Bi - B0, axis=1) / np.linalg.norm(B0)
    assert np.median(err) < 0.05
    # the DIRECTION is recovered essentially exactly ...
    assert np.abs(Bi.mean(axis=0)[:2]).max() < 5e-3
    # ... the AMPLITUDE carries the estimator's normalisation deficit,
    # sum_j m_j D_ij gradWbar_ij (x) dx_ij < I on a finite neighbour sample:
    # a few percent LOW at nsmooth=64 (present even on a perfect lattice, so
    # it is a kernel-sum property, not a disorder one). See the convergence
    # test below; IC_setBField.py renormalises B after the curl for this
    # reason rather than pretending the raw amplitude is the requested one.
    assert 0.93 < Bi.mean(axis=0)[2] < 1.01


def test_uniform_field_amplitude_deficit_shrinks_with_neighbour_count():
    B0 = np.array([0.0, 0.0, 1.0])
    got = []
    for nsmooth in (32, 128):
        pos, mass = _jittered(n_side=14, seed=1, jitter=0.05)
        A = 0.5 * np.cross(np.tile(B0, (len(pos), 1)), pos)
        B, st = curl(pos, mass, A, box=None, nsmooth=nsmooth)
        interior = np.all(np.abs(pos) < 0.5 - 2.5 * st.h.mean(), axis=1)
        got.append(abs(B[interior].mean(axis=0)[2] - 1.0))
    assert got[1] < 0.2 * got[0], f'no convergence: {got}'
    assert got[1] < 0.01


@pytest.mark.parametrize('flavour', AVG_FLAVOURS)
def test_curl_recovers_an_analytic_sinusoidal_field(flavour):
    # A = (0, 0, A0 cos(kx))  =>  curl A = (0, A0 k sin(kx), 0)
    pos, mass = _jittered(n_side=16, seed=2, jitter=0.15)
    k = 2.0 * np.pi
    A0 = 1.0
    A = np.zeros((len(pos), 3))
    A[:, 2] = A0 * np.cos(k * pos[:, 0])
    exact = np.zeros((len(pos), 3))
    exact[:, 1] = A0 * k * np.sin(k * pos[:, 0])
    B, _ = curl(pos, mass, A, box=np.ones(3), nsmooth=64, flavour=flavour)
    rel = np.linalg.norm(B - exact) / np.linalg.norm(exact)
    assert rel < 0.15


@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
def test_curl_output_is_far_more_solenoidal_than_a_prescribed_field(flavour):
    """div(curl) is not machine zero here, but it must be a small residual.

    Compares eps = h|div B|/|B| for B = curl A against the same measure on a
    deliberately non-solenoidal field of comparable amplitude - the claim the
    IC generator's 'potential' mode rests on.
    """
    from glassgen.flavours import divergence_error

    pos, mass = _jittered(n_side=12, seed=4, jitter=0.15)
    box = np.ones(3)
    k = 2.0 * np.pi
    A = np.zeros((len(pos), 3))
    A[:, 2] = np.cos(k * pos[:, 0]) * np.sin(k * pos[:, 1])
    B, st = curl(pos, mass, A, box=box, nsmooth=64, flavour=flavour)
    dB, _ = divergence(pos, mass, B, state=st, flavour=flavour)
    eps_curl = divergence_error(B, dB, st.h).mean()

    # a pure gradient field (100% divergence carrier) at the same amplitude
    G = np.stack([-k * np.sin(k * pos[:, 0]) * np.sin(k * pos[:, 1]),
                  k * np.cos(k * pos[:, 0]) * np.cos(k * pos[:, 1]),
                  np.zeros(len(pos))], axis=1)
    G *= np.linalg.norm(B) / np.linalg.norm(G)
    dG, _ = divergence(pos, mass, G, state=st, flavour=flavour)
    eps_grad = divergence_error(G, dG, st.h).mean()

    assert eps_curl < 0.2 * eps_grad


# ---------------------------------------------------------------------------
# the adjoint identity the cleaning energy proof rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flavour', AVG_FLAVOURS)
def test_symmetric_gradient_is_the_negative_adjoint_of_difference_divergence(
        flavour):
    """<grad f, V>_vol == -<f, div V>_vol under sum_i (m_i/rho_i) u_i . v_i.

    Exact (to roundoff) for the avg kernel mode even with an ASYMMETRIC kNN
    candidate list - which is precisely what the transpose walk buys: each
    pair contributes antisymmetrically to its two members regardless of which
    direction made it into the list.
    """
    pos, mass = _jittered(n_side=8, seed=7)
    rng = np.random.default_rng(13)
    f = rng.normal(size=len(pos))
    V = rng.normal(size=(len(pos), 3))
    box = np.ones(3)

    st = operator_state(pos, mass, box=box, nsmooth=32)
    gf, _ = sph_operator('grad', pos, mass, f, form='symmetric', state=st,
                         flavour=flavour)
    dV, _ = sph_operator('div', pos, mass, V, form='difference', state=st,
                         flavour=flavour)
    vol = mass / st.rho
    lhs = float(np.sum(vol[:, None] * gf * V))
    rhs = -float(np.sum(vol * f * dV))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


# ---------------------------------------------------------------------------
# interp: the zeroth member (Shepard-normalised SPH interpolation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flavour', ALL_FLAVOURS)
def test_interp_reproduces_a_constant_field(flavour):
    """The interpolant's counterpart of difference-form constant-exactness."""
    pos, mass = _jittered(n_side=8)
    V = np.tile([1.0, -2.0, 0.5], (len(pos), 1))
    f = np.full(len(pos), 3.25)
    st = operator_state(pos, mass, box=np.ones(3), nsmooth=32)
    outV, _ = interpolate(pos, mass, V, state=st, flavour=flavour)
    outf, _ = interpolate(pos, mass, f, state=st, flavour=flavour)
    np.testing.assert_allclose(outV, V, rtol=0, atol=1e-13)
    np.testing.assert_allclose(outf, f, rtol=0, atol=1e-13)


def test_interp_reproduces_a_constant_at_an_open_boundary():
    """Shepard normalisation is exactly what buys this: the raw neighbour
    weights do not sum to one where the kernel is one-sided."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.5, 0.5, (800, 3))
    mass = np.full(len(pos), 1.0 / len(pos))
    V = np.tile([0.3, 0.0, -1.1], (len(pos), 1))
    out, _ = interpolate(pos, mass, V, box=None, nsmooth=32)
    np.testing.assert_allclose(out, V, rtol=0, atol=1e-13)


def _direct_interp(pos, mass, field, st, box):
    """Plain-python gather interpolant, own (h_i) window, self INCLUDED."""
    out = np.zeros_like(field)
    for i in range(len(pos)):
        js = np.concatenate([[i], st.idx[i]])
        d = pos[js] - pos[i]
        if box is not None:
            d -= box * np.round(d / box)
        r2 = (d ** 2).sum(axis=1)
        w = np.array([w_wc2(v) for v in r2 / st.h[i] ** 2])
        w /= np.pi * st.h[i] ** 3
        wv = mass[js] / st.rho[js] * w            # gdforce: D_ij = 1/rho_j
        out[i] = (wv[:, None] * field[js]).sum(axis=0) / wv.sum()
    return out


def test_interp_matches_a_direct_sph_sum():
    """gdforce-own is the textbook gather interpolant sum_j (m_j/rho_j) f_j W,
    self term included, so a direct numpy sum must reproduce it exactly."""
    box = np.ones(3)
    pos, mass = _jittered(n_side=6, seed=5)
    rng = np.random.default_rng(11)
    V = np.ascontiguousarray(rng.normal(size=(len(pos), 3)))
    st = operator_state(pos, mass, box=box, nsmooth=32)
    out, _ = interpolate(pos, mass, V, state=st, flavour='gdforce-own')
    np.testing.assert_allclose(out, _direct_interp(pos, mass, V, st, box),
                               rtol=1e-12, atol=1e-14)


def test_interp_includes_the_self_term():
    """A single particle far from its neighbours must keep essentially its own
    value: the self weight W(0) dominates. Dropping the self pair turns the
    operator into 'the mean of my neighbours, excluding me' and this fails.
    """
    pos, mass = _jittered(n_side=6, seed=2)
    # one particle displaced onto its own island, with all others left alone
    pos = np.vstack([pos, [[0.0, 0.0, 0.0]]])
    pos[:-1] = pos[:-1] * 0.02 + 0.4          # everyone else in a far cluster
    mass = np.full(len(pos), 1.0 / len(pos))
    f = np.zeros(len(pos))
    f[-1] = 1.0
    out, _ = interpolate(pos, mass, f, box=None, nsmooth=8)
    assert out[-1] > 0.99


def test_interp_damps_high_wavenumbers_more_than_low():
    """It is a low-pass filter: that is the whole point of --op smooth."""
    pos, mass = _jittered(n_side=12, seed=4)
    st = operator_state(pos, mass, box=np.ones(3), nsmooth=32)
    ratios = []
    for k in (1, 4):
        f = np.sin(2 * np.pi * k * pos[:, 0])
        out, _ = interpolate(pos, mass, f, state=st)
        ratios.append(float(np.std(out) / np.std(f)))
    assert ratios[0] > 0.85           # the box-scale mode survives
    assert ratios[1] < 0.65           # the k=4 mode is visibly damped
    assert ratios[1] < ratios[0]


def test_interp_of_a_vector_shares_one_shepard_denominator():
    """Componentwise weights must be identical, so smoothing cannot rotate a
    locally-aligned field - only shorten it."""
    pos, mass = _jittered(n_side=6, seed=8)
    rng = np.random.default_rng(1)
    amp = rng.uniform(0.5, 2.0, len(pos))
    axis = np.array([0.6, -0.8, 0.0])
    V = amp[:, None] * axis                     # varying length, fixed axis
    out, _ = interpolate(pos, mass, V, box=np.ones(3), nsmooth=32)
    mag = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(out / mag[:, None], np.tile(axis, (len(pos), 1)),
                               rtol=0, atol=1e-12)


def test_interp_is_idempotent_on_an_already_smooth_field_scale():
    """Repeated passes must converge downward monotonically, never ring."""
    pos, mass = _jittered(n_side=10, seed=6)
    st = operator_state(pos, mass, box=np.ones(3), nsmooth=32)
    f = np.sin(2 * np.pi * pos[:, 0])
    amps = []
    for _ in range(4):
        f, _ = interpolate(pos, mass, f, state=st)
        amps.append(float(np.std(f)))
    assert all(b < a for a, b in zip(amps, amps[1:]))


# ---------------------------------------------------------------------------
# dispatch surface
# ---------------------------------------------------------------------------

def test_every_registry_entry_is_callable_with_the_right_arity():
    pos, mass = _jittered(n_side=6)
    n = len(pos)
    field = {'scalar': np.ones(n), 'vector': np.ones((n, 3)),
             'any': np.ones((n, 3))}
    shape = {'scalar': (n,), 'vector': (n, 3), 'tensor': (n, 3, 3),
             'same': None}
    st = operator_state(pos, mass, box=np.ones(3), nsmooth=32)
    for (form, op), (_kernel, arity, outshape) in OPERATORS.items():
        f = field[arity]
        out, _ = sph_operator(op, pos, mass, f, form=form, state=st)
        assert out.shape == (shape[outshape] or f.shape), (form, op)


def test_state_is_reused_not_rebuilt():
    pos, mass = _jittered(n_side=6)
    st = operator_state(pos, mass, box=np.ones(3), nsmooth=32)
    _, st2 = divergence(pos, mass, np.ones((len(pos), 3)), state=st)
    assert st2 is st


def test_clear_errors_on_a_bad_operator_or_arity():
    pos, mass = _jittered(n_side=6)
    with pytest.raises(ValueError, match='unknown operator'):
        sph_operator('laplacian', pos, mass, np.ones(len(pos)))
    with pytest.raises(ValueError, match='unknown SPH flavour'):
        divergence(pos, mass, np.ones((len(pos), 3)), flavour='nope')
    with pytest.raises(ValueError, match='vector field'):
        # curl needs (N,3), handed a scalar
        curl(pos, mass, np.ones(len(pos)))
