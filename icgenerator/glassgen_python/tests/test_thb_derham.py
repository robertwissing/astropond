"""Hierarchical (THB) spline De Rham readout, variant 1 (A-projection):
project an analytic A into S1, curl it, recover the analytic B; exactly
div-free on a graded mesh."""

import numpy as np
import pytest

from glassgen import reconstruct

try:
    reconstruct._import_sph_interp()
    import sph_interp.thb_derham as td
    HAVE_SPH_INTERP = True
except ImportError:
    td = None
    HAVE_SPH_INTERP = False

needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason="sph_interp not found in testsuite layout")


def _A(pos, base):
    k = 2.0 * np.pi / base
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.column_stack([np.sin(k * (y + z)),
                            np.sin(k * (z + x)),
                            np.sin(k * (x + y))])


def _B(pos, base):
    k = 2.0 * np.pi / base
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    c = lambda a: k * np.cos(k * a)                    # noqa: E731
    return np.column_stack([c(x + y) - c(z + x),
                            c(y + z) - c(x + y),
                            c(z + x) - c(y + z)])


@needs_sph_interp
def test_uniform_thb_recovers_analytic_B():
    # no refinement: the hierarchical readout reduces to the uniform complex
    base = (6, 6, 6)
    R = td.THBDeRham(base, degree=3, refined=[], nq=4)
    B = R.project_and_curl(_A(R.qpts, 6),
                           np.random.default_rng(0).uniform(0.5, 5.5, (300, 3)))
    q = np.random.default_rng(0).uniform(0.5, 5.5, (300, 3))
    B = R.project_and_curl(_A(R.qpts, 6), q)
    rel = np.linalg.norm(B - _B(q, 6)) / np.linalg.norm(_B(q, 6))
    assert rel < 0.05, rel


@needs_sph_interp
def test_graded_thb_recovers_analytic_B_and_div_free():
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    R = td.THBDeRham(base, degree=3, refined=[ref0], nq=4)
    Aq = _A(R.qpts, 4)
    q = np.random.default_rng(1).uniform(0.5, 3.5, (200, 3))
    coeffs = R.project_A(Aq)
    B = R.curl_readout(coeffs, q)
    rel = np.linalg.norm(B - _B(q, 4)) / np.linalg.norm(_B(q, 4))
    assert rel < 0.08, rel
    # divergence-free at arbitrary interior points
    h = 1e-4
    div = np.zeros(len(q))
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = h
        div += (R.curl_readout(coeffs, q + e)[:, ax]
                - R.curl_readout(coeffs, q - e)[:, ax]) / (2 * h)
    assert np.max(np.abs(div)) < 1e-3 * np.linalg.norm(B, axis=1).mean()


@needs_sph_interp
def test_bfit_recovers_analytic_B_and_div_free():
    # variant 2: fit A so curl A matches the target B directly
    base = (6, 6, 6)
    R = td.THBDeRham(base, degree=3, refined=[], nq=4)
    Bq = _B(R.qpts, 6)
    q = np.random.default_rng(2).uniform(0.6, 5.4, (300, 3))
    Bh = R.bfit(Bq, q)
    rel = np.linalg.norm(Bh - _B(q, 6)) / np.linalg.norm(_B(q, 6))
    assert rel < 0.05, rel
    # div-free at arbitrary points
    hh = 1e-4
    div = np.zeros(len(q))
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = hh
        div += (R.bfit(Bq, q + e)[:, ax] - R.bfit(Bq, q - e)[:, ax]) / (2 * hh)
    assert np.max(np.abs(div)) < 1e-3 * np.linalg.norm(Bh, axis=1).mean()


@needs_sph_interp
def test_bfit_restores_mean_field():
    # a target B with a nonzero uniform component: curl(S1) is mean-free, so
    # Bbar must be split off and restored - the readout must recover the mean
    base = (4, 4, 4)
    R = td.THBDeRham(base, degree=3, refined=[], nq=3)
    Bbar0 = np.array([0.4, -0.9, 1.3])
    Bq = _B(R.qpts, 4) + Bbar0[None, :]
    q = np.random.default_rng(3).uniform(0.6, 3.4, (150, 3))
    Bh = R.bfit(Bq, q)                              # auto mean split
    rel = np.linalg.norm(Bh - (_B(q, 4) + Bbar0)) / np.linalg.norm(_B(q, 4) + Bbar0)
    assert rel < 0.06, rel
    # and the recovered mean matches
    assert np.allclose(Bh.mean(axis=0), (_B(q, 4) + Bbar0).mean(axis=0), atol=0.05)


@needs_sph_interp
def test_bfit_is_optimal_for_B_vs_A_projection():
    # curl-fit is L2-OPTIMAL for B in the norm it minimises (the quadrature-
    # weighted domain L2), so it must beat-or-match the A-projection there.
    base = (5, 5, 5)
    R = td.THBDeRham(base, degree=3, refined=[], nq=4)
    qp = R.qpts
    Bt = _B(qp, 5)
    sw = np.sqrt(R.qw)[:, None]

    def wl2(Bh):
        return np.linalg.norm(sw * (Bh - Bt)) / np.linalg.norm(sw * Bt)

    err_aproj = wl2(R.project_and_curl(_A(qp, 5), qp))
    err_bfit = wl2(R.bfit(_B(qp, 5), qp))
    # optimal in the domain-L2 norm (small slack for the auto mean split)
    assert err_bfit <= err_aproj + 1e-4, (err_bfit, err_aproj)


@needs_sph_interp
def test_matrixfree_projector_matches_dense():
    # scalable matrix-free CG projection (no dense mass matrix) reaches the
    # same coefficients as the dense solve.  Small/no-refine to stay fast (the
    # localized primitives are unoptimised Python).
    R = td.THBDeRham((4, 4, 4), degree=3, refined=[], nq=2)
    Aq = _A(R.qpts, 4)
    c_dense = R.project_A(Aq)
    c_mf = R.project_A_matfree(Aq, rtol=1e-9)
    for comp in range(3):
        assert np.max(np.abs(c_mf[comp] - c_dense[comp])) < 1e-6


@needs_sph_interp
def test_matrixfree_bfit_matches_dense():
    # scalable curl-fit (variant 2) via matrix-free CG on the curl-curl system
    # reaches the same B as the dense pinv-based bfit
    R = td.THBDeRham((4, 4, 4), degree=3, refined=[], nq=2)
    Bq = _B(R.qpts, 4)
    q = np.random.default_rng(0).uniform(0.6, 3.4, (30, 3))
    B_mf = R.bfit_matfree(Bq, q, rtol=1e-9)
    B_d = R.bfit(Bq, q)
    assert np.max(np.abs(B_mf - B_d)) < 1e-6 * np.max(np.abs(B_d))


@needs_sph_interp
def test_sparse_matfree_bfit_recovers_B():
    # variant 2 on the fully-sparse space: curl-fit recovers analytic B,
    # div-free (small/no-refine to keep the Python primitives fast)
    R = td.THBDeRham((4, 4, 4), degree=3, refined=[], nq=2, sparse=True)
    q = np.random.default_rng(1).uniform(0.6, 3.4, (30, 3))
    Bh = R.bfit_matfree(_B(R.qpts, 4), q, rtol=1e-8)
    assert np.linalg.norm(Bh - _B(q, 4)) / np.linalg.norm(_B(q, 4)) < 0.06


@needs_sph_interp
def test_sparse_leaf_quadrature_integrates_volume():
    # per-leaf Gauss weights must sum to the domain volume (level-0 units)
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    R = td.THBDeRham(base, degree=3, refined=[ref0], nq=3, sparse=True)
    assert np.isclose(R.qw.sum(), float(np.prod(base)))


@needs_sph_interp
def test_sparse_curl_matches_dense_readout():
    # THBSparse and THB3D share the basis, so the localized curl readout is
    # identical given the same coefficients
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    Rd = td.THBDeRham(base, degree=3, refined=[ref0], nq=3)
    Rs = td.THBDeRham(base, degree=3, refined=[ref0], nq=3, sparse=True)
    coeffs = Rd.project_A(_A(Rd.qpts, 4))
    q = np.random.default_rng(0).uniform(0.5, 3.5, (40, 3))
    np.testing.assert_allclose(Rs.curl_readout_local(coeffs, q),
                               Rd.curl_readout_local(coeffs, q), atol=1e-12)


@needs_sph_interp
def test_sparse_matfree_variant1_recovers_B():
    # the whole scalable variant-1 pipeline (sparse space + per-leaf quad +
    # matrix-free projection + localized curl) recovers analytic B, div-free.
    # Small/no-refine to keep the unoptimised Python primitives fast.
    R = td.THBDeRham((4, 4, 4), degree=3, refined=[], nq=2, sparse=True)
    coeffs = R.project_A_matfree(_A(R.qpts, 4), rtol=1e-8)
    q = np.random.default_rng(0).uniform(0.5, 3.5, (30, 3))
    Bh = R.curl_readout_local(coeffs, q)
    assert np.linalg.norm(Bh - _B(q, 4)) / np.linalg.norm(_B(q, 4)) < 0.06


@needs_sph_interp
def test_matfree_projector_cg_matches_direct():
    # the Jacobi-preconditioned matrix-free mass solve (needed at galaxy scale,
    # where the spsolve fill-in is the memory wall) must reach the same
    # coefficients as the direct sparse factorisation
    R = td.THBDeRham((4, 4, 4), degree=3, refined=[], nq=2, sparse=True)
    Aq = _A(R.qpts, 4)
    c_direct = R.project_A_matfree(Aq, solver="direct")
    c_cg = R.project_A_matfree(Aq, solver="cg", rtol=1e-12)
    for comp in range(3):
        scale = max(1.0, float(np.max(np.abs(c_direct[comp]))))
        assert np.max(np.abs(c_cg[comp] - c_direct[comp])) < 1e-7 * scale
    R.free_operators()                       # cached operators are droppable
    assert not hasattr(R, "_Eop")
    c_again = R.project_A_matfree(Aq, solver="cg", rtol=1e-12)
    assert np.max(np.abs(c_again[0] - c_direct[0])) < 1e-7


@needs_sph_interp
def test_project_and_curl_matfree_graded_recovers_B():
    # VARIANT 1 end to end through the scalable entry point (sparse space +
    # per-leaf quadrature + CG projection + fast curl readout)
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    R = td.THBDeRham(base, degree=3, refined=[ref0], nq=2, sparse=True)
    q = np.random.default_rng(7).uniform(0.5, 3.5, (60, 3))
    B, coeffs = R.project_and_curl_matfree(_A(R.qpts, 4), q, return_A=True)
    assert np.linalg.norm(B - _B(q, 4)) / np.linalg.norm(_B(q, 4)) < 0.08
    # same thing composed by hand
    np.testing.assert_allclose(
        B, R.curl_readout_fast(coeffs, q), rtol=0, atol=1e-12)
    # and a uniform field can be added back (curl of the space is mean-free)
    Bbar = np.array([0.3, -0.7, 1.1])
    np.testing.assert_allclose(
        R.project_and_curl_matfree(_A(R.qpts, 4), q, Bbar=Bbar), B + Bbar,
        rtol=0, atol=1e-12)


@needs_sph_interp
def test_variant1_anisotropic_cell_units_recover_physical_B():
    # The THB space lives in level-0 CELL units, so a PHYSICAL A on a box with
    # anisotropic base cells enters as the 1-form pullback A~_a = A_a * pw_a and
    # comes back as the 2-form curl_u A~|_a = pw_b pw_c B_a (a,b,c cyclic).
    # This is exactly what the galaxy harness (recon_m8/magnetic_galaxy_amr_thb)
    # does for variant 1 fed the captured leaf A - lock it here.
    base = (4, 4, 4)
    pw = np.array([0.7, 1.3, 2.1])                 # anisotropic base cell
    k = 2.0 * np.pi / base[0]
    R = td.THBDeRham(base, degree=3, refined=[], nq=3, sparse=True)

    def B_phys(u):
        # curl_x of A(u(x)) with x_a = pw_a u_a: d/dx_a = (1/pw_a) d/du_a
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        c = lambda a: k * np.cos(k * a)             # noqa: E731
        return np.column_stack([c(x + y) / pw[1] - c(z + x) / pw[2],
                                c(y + z) / pw[2] - c(x + y) / pw[0],
                                c(z + x) / pw[0] - c(y + z) / pw[1]])

    q = np.random.default_rng(8).uniform(0.5, 3.5, (200, 3))
    curl = R.project_and_curl_matfree(_A(R.qpts, 4) * pw[None, :], q)
    B = curl / np.array([pw[1] * pw[2], pw[2] * pw[0], pw[0] * pw[1]])[None, :]
    rel = np.linalg.norm(B - B_phys(q)) / np.linalg.norm(B_phys(q))
    assert rel < 0.05, rel


@needs_sph_interp
def test_scalable_curl_readout_matches_dense():
    # the finest-grid-free curl readout reproduces the dense expand+curl exactly
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    R = td.THBDeRham(base, degree=3, refined=[ref0], nq=3)
    coeffs = R.project_A(_A(R.qpts, 4))
    q = np.random.default_rng(3).uniform(0.5, 3.5, (60, 3))
    np.testing.assert_allclose(R.curl_readout_local(coeffs, q),
                               R.curl_readout(coeffs, q), atol=1e-12)


def _cell_field(R, fn, base):
    """Analytic field sampled at the finest cell centres = a table proxy."""
    d = R.dimsL
    ax = [(np.arange(d[a]) + 0.5) * R.hf for a in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return fn(pts, base).reshape(d + (3,))


@needs_sph_interp
def test_table_cell_A_projection_recovers_B():
    # feed a piecewise-constant cell A (Petkova-leaf proxy), project, curl.
    # error is O(h^2), TABLE-resolution-limited (0.20 -> 0.089 -> 0.051 at
    # base 4->6->8; the galaxy table refines where B has structure).
    errs = []
    for base in [(4, 4, 4), (6, 6, 6)]:
        R = td.THBDeRham(base, degree=3, refined=[], nq=3)
        cellA = _cell_field(R, _A, base[0])
        q = np.random.default_rng(5).uniform(0.6, base[0] - 0.6, (300, 3))
        B = R.curl_readout(R.project_A_cells(cellA), q)
        errs.append(np.linalg.norm(B - _B(q, base[0])) / np.linalg.norm(_B(q, base[0])))
    assert errs[-1] < 0.10, errs
    assert errs[-1] < 0.6 * errs[0]             # ~O(h^2) convergence


@needs_sph_interp
def test_table_cell_B_fit_recovers_B_div_free():
    base = (6, 6, 6)
    R = td.THBDeRham(base, degree=3, refined=[], nq=3)
    cellB = _cell_field(R, _B, 6)
    q = np.random.default_rng(6).uniform(0.6, 5.4, (300, 3))
    Bh = R.bfit_cells(cellB, q)
    # table-resolution-limited at 6 cells/wavelength (O(h^2), converges)
    assert np.linalg.norm(Bh - _B(q, 6)) / np.linalg.norm(_B(q, 6)) < 0.10
    hh = 1e-4
    div = np.zeros(len(q))
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = hh
        div += (R.bfit_cells(cellB, q + e)[:, ax]
                - R.bfit_cells(cellB, q - e)[:, ax]) / (2 * hh)
    assert np.max(np.abs(div)) < 1e-3 * np.linalg.norm(Bh, axis=1).mean()


@needs_sph_interp
def test_helicity_beltrami_sign_and_magnitude():
    # A single Beltrami mode B=(0, cos kx, sin kx) has curl B = -k B, so its
    # vector potential is A = -B/k and the helicity H = int A.B dV = -(1/k) int
    # |B|^2 dV = -(1/k) * Vol  (|B|^2 == 1).  The THB curl-fit recovers A
    # explicitly, so R.helicity must land on that value AND flip sign under the
    # mirror x -> -x (opposite handedness).
    base = (8, 8, 8)
    k = 2.0 * np.pi / base[0]
    R = td.THBDeRham(base, degree=3, refined=[], nq=4, sparse=True)

    def beltrami(P, sgn):
        x = P[:, 0]
        return np.column_stack([np.zeros_like(x), np.cos(k * x),
                                sgn * np.sin(k * x)])

    _, A = R.bfit_matfree(beltrami(R.qpts, +1.0), R.qpts, rtol=1e-10,
                          return_A=True)
    H = R.helicity(A, R.qpts, R.qw)
    H_exact = -np.prod(base) / k
    assert abs(H / H_exact - 1.0) < 0.02                 # sign + magnitude
    _, Am = R.bfit_matfree(beltrami(R.qpts, -1.0), R.qpts, rtol=1e-10,
                           return_A=True)
    Hm = R.helicity(Am, R.qpts, R.qw)
    assert np.sign(Hm) != np.sign(H) and abs(Hm / H + 1.0) < 0.02
