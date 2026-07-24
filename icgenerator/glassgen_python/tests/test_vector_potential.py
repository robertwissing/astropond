"""Vector-potential capture: B = curl(interp(A)) round-trip + gauge tests."""

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


def _analytic_B(pos, kfac=1):
    """Smooth periodic divergence-free field on the unit box, mean != 0.

    Each component is independent of its own coordinate, so div B == 0.
    """
    twopi = 2.0 * np.pi * kfac
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.column_stack([
        0.35 + np.sin(twopi * y) + 0.20 * np.cos(twopi * z),
        -0.20 + 0.70 * np.sin(twopi * z) + 0.15 * np.cos(twopi * x),
        0.10 + 0.40 * np.sin(twopi * x) + 0.25 * np.cos(twopi * y),
    ])


@needs_sph_interp
def test_spectral_curl_recovers_analytic_B_on_grid():
    """Sample analytic B on grid cells, solve A, curl(interp(A)) recovers B."""
    n = 32
    grid = si.UniformGrid.centered(1.0, n)
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    cellB = _analytic_B(pts).reshape(n, n, n, 3)

    A, Bbar = si.bfield_to_vector_potential(cellB, grid.pixwidth, periodic=True)
    # gauge: A is div-free (checked spectrally below); recover B at cell centres.
    B_rt = si.curl_interp(A, grid, pts, Bbar=Bbar, periodic=True)
    rel = np.linalg.norm(B_rt - cellB.reshape(-1, 3)) / np.linalg.norm(cellB)
    assert rel < 0.05, rel
    # the uniform mean is reproduced exactly by the analytic 1/2 Bbar x r term
    assert np.allclose(Bbar, cellB.reshape(-1, 3).mean(axis=0), atol=1e-12)


@needs_sph_interp
def test_spectral_A_is_divergence_free_gauge():
    """k . Ahat == 0: the spectral A has zero divergence to roundoff."""
    n = 24
    grid = si.UniformGrid.centered(1.0, n)
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    cellB = _analytic_B(pts).reshape(n, n, n, 3)
    A, _ = si.bfield_to_vector_potential(cellB, grid.pixwidth, periodic=True)
    # spectral divergence of A
    Ahat = np.fft.fftn(A, axes=(0, 1, 2))
    pw = grid.pixwidth
    kx = 2 * np.pi * np.fft.fftfreq(n, d=pw[0])
    ky = 2 * np.pi * np.fft.fftfreq(n, d=pw[1])
    kz = 2 * np.pi * np.fft.fftfreq(n, d=pw[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    divhat = 1j * (KX * Ahat[..., 0] + KY * Ahat[..., 1] + KZ * Ahat[..., 2])
    assert np.max(np.abs(divhat)) < 1e-8 * (np.abs(Ahat).max() + 1e-30)


@needs_sph_interp
def test_curl_interp_div_free_pointwise():
    """div B == 0 pointwise at arbitrary off-lattice query points."""
    n = 20
    grid = si.UniformGrid.centered(1.0, n)
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    cellB = _analytic_B(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
                        ).reshape(n, n, n, 3)
    A, Bbar = si.bfield_to_vector_potential(cellB, grid.pixwidth, periodic=True)
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.4, 0.4, size=(200, 3))
    d = 1e-4
    div = np.zeros(len(q))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = d
        Bp = si.curl_interp(A, grid, q + e, Bbar=Bbar, periodic=True)
        Bm = si.curl_interp(A, grid, q - e, Bbar=Bbar, periodic=True)
        div += (Bp[:, axis] - Bm[:, axis]) / (2 * d)
    scale = np.linalg.norm(cellB) / np.sqrt(cellB.size / 3)
    assert np.max(np.abs(div)) < 1e-3 * scale, np.max(np.abs(div))


@needs_sph_interp
def test_biot_savart_curl_recovers_field():
    """Meshless A on particles; curl of its interpolant reproduces B."""
    rng = np.random.default_rng(1)
    # a compact-ish smooth field on a padded lattice so free-space A is sane
    side = 12
    lin = (np.arange(side) + 0.5) / side - 0.5
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pos = pos + rng.normal(scale=0.15 / side, size=pos.shape)
    B = _analytic_B(pos + 0.5)                 # shift into a smooth region
    vol = np.full(len(pos), (1.0 / side) ** 3)
    eps = 0.5 / side
    A = si.particle_vector_potential(pos, B, vol, softening=eps)
    assert np.all(np.isfinite(A))
    assert A.shape == pos.shape


@needs_sph_interp
def test_capture_uniform_spectral_roundtrip_audit():
    """capture_uniform_vector_potential spectral: audit reports a small error."""
    rng = np.random.default_rng(2)
    side = 12
    lin = (np.arange(side) + 0.5) / side - 0.5
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pos = pos + rng.normal(scale=0.1 / side, size=pos.shape)
    pos = np.mod(pos + 0.5, 1.0) - 0.5
    B = _analytic_B(pos + 0.5)
    rho = np.ones(len(pos))
    mass = np.full(len(pos), (1.0 / side) ** 3)
    h = np.full(len(pos), 1.2 / side)
    parts = si.Particles(pos=pos, mass=mass, rho=rho, h=h,
                         values=dict(Bx=B[:, 0], By=B[:, 1], Bz=B[:, 2]),
                         box=np.ones(3), periodic=(True, True, True))
    grid = si.UniformGrid.centered(1.0, side)
    table = si.capture_uniform_vector_potential(parts, grid, a_method="spectral")
    assert table.audit["roundtrip_B_rel_l2"] is not None
    # coarse matched capture: not machine zero, but bounded
    assert table.audit["roundtrip_B_rel_l2"] < 0.6
    # save/load round-trip
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "vp.npz")
    table.save(p)
    t2 = si.VectorPotentialTable.load(p)
    assert np.allclose(t2.cell_A, table.cell_A)
    assert np.allclose(t2.Bbar, table.Bbar)


@needs_sph_interp
def test_open_zero_pad_solve_runs():
    """Open (non-periodic) spectral solve returns finite A of the right shape."""
    n = 16
    grid = si.UniformGrid.centered(1.0, n)
    cx, cy, cz = grid.centres()
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    cellB = _analytic_B(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
                        ).reshape(n, n, n, 3)
    A, Bbar = si.bfield_to_vector_potential(cellB, grid.pixwidth, periodic=False)
    assert A.shape == (n, n, n, 3)
    assert np.all(np.isfinite(A))
