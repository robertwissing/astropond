"""`AMRGrid.build`'s default domain on an OPEN (box=None) particle set.

The periodic branch has always been right: the box IS the domain, and a
particle's kernel wraps rather than leaves. The open branch used the bare
particle bounding box, `pos.min(axis=0) .. pos.max(axis=0)`, which is exactly
the domain that clips half the compact support of every extremal particle. It
failed quietly — mass integrated to less than the particle mass, with no error
and no budget — and it is the same mistake `_bench_interp.py` made with a
hand-rolled 2% pad, which is why the fix routes through the package's own
resolver (`kernel_support_bounds`) rather than another literal.

Every in-tree production caller (`IC_interpolate_to_grid`, `IC_export_polaris`,
the benches) passes explicit `lo`/`hi`, so this was a latent trap rather than a
live bug — these tests keep it that way.
"""
import numpy as np
import pytest

from sph_interp import AMRGrid, Particles, UniformGrid, interpolate
from sph_interp.domain import kernel_support_bounds


def _open_blob(n=3000, seed=0):
    """An isolated object: no box, and the extremal particles carry the LARGEST
    h, so a bounding-box domain clips the most support it possibly could."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 0.3, (n, 3))
    r = np.linalg.norm(pos, axis=1)
    mass = np.full(n, 1.0 / n)
    h = 0.02 + 0.08 * (r / r.max())              # h grows outward
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    return Particles(pos=pos, mass=mass, rho=rho, h=h,
                     values={'rho': rho}, box=None, periodic=False)


def test_the_default_open_domain_contains_every_kernel_support():
    p = _open_blob()
    g = AMRGrid.build(p, base_npx=8, max_level=2)
    support = 2.0 * p.h[:, None]
    assert np.all(g.lo <= (p.pos - support).min(axis=0))
    assert np.all(g.hi >= (p.pos + support).max(axis=0))


def test_it_is_strictly_larger_than_the_bare_bounding_box():
    """The regression evidence: the old default was the inner box, and on this
    blob it clips a real fraction of the domain, not a rounding hair."""
    p = _open_blob()
    g = AMRGrid.build(p, base_npx=8, max_level=2)
    assert np.all(g.lo < p.pos.min(axis=0))
    assert np.all(g.hi > p.pos.max(axis=0))
    grew = (g.hi - g.lo) / (p.pos.max(axis=0) - p.pos.min(axis=0))
    assert np.all(grew > 1.05)


def test_the_default_matches_the_package_resolver_exactly():
    """No second definition of 'how much to pad' is allowed to exist here."""
    p = _open_blob()
    g = AMRGrid.build(p, base_npx=8, max_level=2)
    lo, hi = kernel_support_bounds(p.pos, p.h)
    np.testing.assert_array_equal(g.lo, lo)
    np.testing.assert_array_equal(g.hi, hi)


def test_mass_is_conserved_on_the_default_open_domain():
    """What the clipping actually cost. petkova is exact per cell, so on a
    domain that contains the support the deposit must return the full mass."""
    p = _open_blob()
    g = AMRGrid.build(p, base_npx=8, max_level=2)
    res = interpolate(p, g, method='petkova', values=['rho'])
    got = float((np.asarray(res.data['rho']) * g.volumes()).sum())
    assert got == pytest.approx(float(p.mass.sum()), rel=1e-10)


def test_the_bare_bounding_box_really_does_lose_mass():
    """Pins the failure mode itself, so the test above cannot pass vacuously:
    ask for the old domain explicitly and the mass goes missing."""
    p = _open_blob()
    g = AMRGrid.build(p, base_npx=8, max_level=2,
                      lo=p.pos.min(axis=0), hi=p.pos.max(axis=0))
    res = interpolate(p, g, method='petkova', values=['rho'])
    got = float((np.asarray(res.data['rho']) * g.volumes()).sum())
    assert got < 0.999 * float(p.mass.sum())


def test_explicit_bounds_still_win():
    p = _open_blob()
    lo, hi = np.full(3, -2.0), np.full(3, 2.0)
    g = AMRGrid.build(p, base_npx=8, max_level=2, lo=lo, hi=hi)
    np.testing.assert_array_equal(g.lo, lo)
    np.testing.assert_array_equal(g.hi, hi)


def test_the_periodic_branch_is_untouched():
    """A boxed set still tiles its box exactly — the period is the domain, and
    padding it would break the wrap."""
    n = 2000
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.5, 0.5, (n, 3))
    mass = np.full(n, 1.0 / n)
    h = np.full(n, 0.05)
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    p = Particles(pos=pos, mass=mass, rho=rho, h=h, values={'rho': rho},
                  box=np.ones(3), periodic=True)
    g = AMRGrid.build(p, base_npx=8, max_level=1)
    np.testing.assert_allclose(g.lo, -0.5)
    np.testing.assert_allclose(g.hi, 0.5)
    assert float(g.volumes().sum()) == pytest.approx(1.0, rel=1e-12)
