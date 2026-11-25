"""Periodic uniform B-spline De Rham complex: structure (div curl == 0) +
analytic B recovery."""

import numpy as np
import pytest

from glassgen import reconstruct

try:
    si = reconstruct._import_sph_interp()
    HAVE_SPH_INTERP = True
except ImportError:
    si = None
    HAVE_SPH_INTERP = False

needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason="sph_interp not found in testsuite layout")

TWO_PI = 2.0 * np.pi


def _analytic_A(pos):
    """Smooth periodic vector potential on the unit box (cross-variable, so
    curl A is genuinely 3D)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.column_stack([np.sin(TWO_PI * (y + z)),
                            np.sin(TWO_PI * (z + x)),
                            np.sin(TWO_PI * (x + y))])


def _analytic_B(pos):
    """B = curl A for the A above (div B == 0 analytically)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    c = lambda a: TWO_PI * np.cos(TWO_PI * a)          # noqa: E731
    Bx = c(x + y) - c(z + x)
    By = c(y + z) - c(x + y)
    Bz = c(z + x) - c(y + z)
    return np.column_stack([Bx, By, Bz])


def _cell_A(grid):
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return _analytic_A(pts).reshape(grid.npx + (3,))


@needs_sph_interp
def test_projector_interpolates_cell_centres():
    # the FFT-deconvolution projector reproduces the samples at cell centres.
    # Odd n has no Nyquist bin, so the odd-degree half-sample null never
    # appears and the interpolation is exact.
    n = 21
    grid = si.UniformGrid.centered(1.0, n)
    cell_A = _cell_A(grid)
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    cx_c = si.spline_derham.SplineDeRham(grid, degree=3).project_A(cell_A)
    from sph_interp.spline_derham import _eval_scalar, _A_component_degrees
    degA = _A_component_degrees(3)
    u = [np.mod((centres[:, a] - grid.lo[a]) / grid.pixwidth[a], n)
         for a in range(3)]
    for comp in range(3):
        val = _eval_scalar(cx_c[comp], u, degA[comp], (n, n, n))
        np.testing.assert_allclose(val, cell_A[..., comp].ravel(), atol=1e-9)


@needs_sph_interp
@pytest.mark.parametrize("degree", [2, 3])
def test_div_curl_is_machine_zero(degree):
    # structure-preserving: div(curl A) == 0 at the coefficient level
    n = 16
    grid = si.UniformGrid.centered(1.0, n)
    cell_A = _cell_A(grid)
    cplx = si.SplineDeRham(grid, degree=degree)
    d = cplx.divergence_coeffs(cell_A)
    Bx, By, Bz = cplx.curl_from_A(cell_A)
    scale = max(np.abs(Bx).max(), np.abs(By).max(), np.abs(Bz).max())
    assert np.max(np.abs(d)) < 1e-10 * scale


@needs_sph_interp
def test_curl_div_free_pointwise_off_lattice():
    # div B == 0 at arbitrary off-grid points (finite-difference probe)
    n = 24
    grid = si.UniformGrid.centered(1.0, n)
    cell_A = _cell_A(grid)
    cplx = si.SplineDeRham(grid, degree=3)
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.4, 0.4, size=(150, 3))
    h = 1e-4
    div = np.zeros(len(q))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = h
        Bp = cplx.query(cell_A, q + e)
        Bm = cplx.query(cell_A, q - e)
        div += (Bp[:, axis] - Bm[:, axis]) / (2 * h)
    Bmag = np.linalg.norm(cplx.query(cell_A, q), axis=1).mean()
    assert np.max(np.abs(div)) < 1e-4 * Bmag


@needs_sph_interp
def test_recovers_analytic_B_and_converges():
    # <5% analytic recovery, and the error decreases with resolution
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.45, 0.45, size=(400, 3))
    B_true = _analytic_B(q)
    tnorm = np.linalg.norm(B_true)
    errs = []
    for n in (16, 32):
        grid = si.UniformGrid.centered(1.0, n)
        cell_A = _cell_A(grid)
        B = si.spline_derham_curl(cell_A, grid, q, degree=3)
        errs.append(np.linalg.norm(B - B_true) / tnorm)
    assert errs[-1] < 0.05, errs
    assert errs[-1] < errs[0]              # convergence with resolution


def _compact_A(pos, sig=0.08):
    """Compact vector potential (Gaussian bump, ~0 at the box edge)."""
    b = np.exp(-np.sum(pos ** 2, axis=1) / (2.0 * sig ** 2))
    return np.column_stack([b, b, b])


def _compact_B(pos, sig=0.08):
    """B = curl A for _compact_A (div-free, compact)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    b = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2.0 * sig ** 2))
    s2 = sig ** 2
    return np.column_stack([(z - y) / s2 * b,
                            (x - z) / s2 * b,
                            (y - x) / s2 * b])


@needs_sph_interp
def test_open_recovers_compact_B_and_converges():
    # open (zero-pad) readout of a compact source, evaluated in the interior
    rng = np.random.default_rng(3)
    q = rng.uniform(-0.22, 0.22, size=(400, 3))       # interior, off boundary
    B_true = _compact_B(q)
    tnorm = np.linalg.norm(B_true)
    errs = []
    for n in (20, 40):
        grid = si.UniformGrid.centered(1.0, n)
        cell_A = np.zeros(grid.npx + (3,))
        cx, cy, cz = grid.centres()
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        cell_A[:] = _compact_A(pts).reshape(grid.npx + (3,))
        B = si.spline_derham_curl(cell_A, grid, q, degree=3, periodic=False)
        errs.append(np.linalg.norm(B - B_true) / tnorm)
    assert errs[-1] < 0.05, errs
    assert errs[-1] < errs[0]


@needs_sph_interp
def test_open_curl_div_free_pointwise():
    # div B == 0 in the interior for the open embedding too
    n = 48
    grid = si.UniformGrid.centered(1.0, n)
    cell_A = np.zeros(grid.npx + (3,))
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    cell_A[:] = _compact_A(pts).reshape(grid.npx + (3,))
    rng = np.random.default_rng(4)
    q = rng.uniform(-0.25, 0.25, size=(120, 3))
    h = 1e-4
    div = np.zeros(len(q))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = h
        Bp = si.spline_derham_curl(cell_A, grid, q + e, degree=3, periodic=False)
        Bm = si.spline_derham_curl(cell_A, grid, q - e, degree=3, periodic=False)
        div += (Bp[:, axis] - Bm[:, axis]) / (2 * h)
    Bmag = np.linalg.norm(
        si.spline_derham_curl(cell_A, grid, q, degree=3, periodic=False),
        axis=1).mean()
    assert np.max(np.abs(div)) < 1e-3 * Bmag


@needs_sph_interp
def test_uniform_mean_added_exactly():
    # a constant Bbar is reproduced exactly via the analytic 1/2 Bbar x r term
    n = 16
    grid = si.UniformGrid.centered(1.0, n)
    cell_A = np.zeros(grid.npx + (3,))         # zero fluctuation
    Bbar = np.array([0.3, -0.7, 1.1])
    q = np.random.default_rng(2).uniform(-0.4, 0.4, size=(50, 3))
    B = si.spline_derham_curl(cell_A, grid, q, Bbar=Bbar, degree=3)
    np.testing.assert_allclose(B, np.broadcast_to(Bbar, B.shape), atol=1e-12)
