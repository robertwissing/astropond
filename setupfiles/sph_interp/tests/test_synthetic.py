"""Tests for `sph_interp.synthetic` — the generated IC families.

These guard the properties the scaling bench DEPENDS on, not the cosmetics of
the distributions: a box that does not move with N, determinism, particles
inside their own domain, and the dynamic range each family exists to provide.
"""

import numpy as np
import pytest

from sph_interp import synthetic as S

SMALL = 4000


@pytest.mark.parametrize("fam", S.FAMILIES)
def test_basic_sanity(fam):
    p = S.make(fam, SMALL)
    assert p.n > 0
    for name, a in (("pos", p.pos), ("mass", p.mass), ("rho", p.rho),
                    ("h", p.h)):
        assert np.all(np.isfinite(a)), f"{fam}: non-finite {name}"
    assert np.all(p.mass > 0)
    assert np.all(p.rho > 0)
    assert np.all(p.h > 0)
    assert "rho" in p.values
    assert np.allclose(p.values["rho"], p.rho)
    assert pytest.approx(1.0, rel=1e-12) == float(p.mass.sum())


@pytest.mark.parametrize("fam", S.FAMILIES)
def test_particle_count_close_to_request(fam):
    """Lattice families round N to what their tiling admits; nothing may be
    off by more than ~15%, and the free-form families must be exact."""
    p = S.make(fam, SMALL)
    if fam in ("lattice", "closepacked", "slab1d", "slab2d"):
        assert 0.6 * SMALL <= p.n <= 1.6 * SMALL
    else:
        assert p.n == SMALL


@pytest.mark.parametrize("fam", S.FAMILIES)
def test_box_does_not_depend_on_n(fam):
    """The bench sweeps N with the box held fixed.

    If the box tracked the sampled particles, every point of the sweep would
    present the AREPO Voronoi backend with a new aspect ratio and charge a
    binary rebuild to the mesh time -- and the grid sweep's cells would change
    shape underneath the measurement.
    """
    b1 = np.asarray(S.make(fam, SMALL).box)
    b2 = np.asarray(S.make(fam, 4 * SMALL).box)
    assert np.allclose(b1, b2), f"{fam}: box moved with N ({b1} -> {b2})"


@pytest.mark.parametrize("fam", S.FAMILIES)
def test_particles_lie_inside_the_box(fam):
    p = S.make(fam, SMALL)
    half = 0.5 * np.asarray(p.box)
    # a hair of tolerance: periodic wraps land exactly on the face
    assert np.all(np.abs(p.pos) <= half * (1.0 + 1e-9)), \
        f"{fam}: particles outside the declared box"


@pytest.mark.parametrize("fam", S.FAMILIES)
def test_deterministic(fam):
    a = S.make(fam, SMALL, seed=7)
    b = S.make(fam, SMALL, seed=7)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.h, b.h)


def test_seed_changes_the_sample():
    a = S.make("uniform", SMALL, seed=1)
    b = S.make("uniform", SMALL, seed=2)
    assert not np.array_equal(a.pos, b.pos)


# --------------------------------------------------------------------------
# per-family properties: the reason each family is in the list
# --------------------------------------------------------------------------

def test_lattice_is_perfectly_regular():
    p = S.make("lattice", SMALL)
    assert np.ptp(p.h) == pytest.approx(0.0, abs=1e-12)


def test_closepacked_is_regular_but_not_cubic():
    """Same uniform h as the cubic lattice, different neighbour geometry."""
    cp = S.make("closepacked", SMALL)
    assert np.ptp(cp.h) == pytest.approx(0.0, abs=1e-9)
    lat = S.make("lattice", SMALL)
    # equal spacing would make the two identical; the ABC stacking must not be
    assert not np.allclose(np.sort(cp.pos[:, 2])[:50],
                           np.sort(lat.pos[:, 2])[:50])


def test_glass_is_more_uniform_than_poisson():
    """The point of a glass: quasi-uniform WITHOUT lattice symmetry."""
    g = S.make("glass", SMALL)
    u = S.make("uniform", SMALL)
    gd = float(g.h.max() / g.h.min())
    ud = float(u.h.max() / u.h.min())
    assert gd < ud, f"glass h-range {gd:.3f} not tighter than Poisson {ud:.3f}"
    assert np.ptp(g.h) > 0.0, "a glass must not be a perfect lattice"


@pytest.mark.parametrize("fam,expect", [("slab1d", S.SLAB1D_ASPECT),
                                        ("slab2d", S.SLAB2D_ASPECT)])
def test_slab_aspect_matches_the_real_setups(fam, expect):
    """128:12:12 (shocktube) and 128:128:12 (Orszag-Tang)."""
    p = S.make(fam, 20000)
    box = np.asarray(p.box)
    want = np.asarray(expect, dtype=float)
    assert np.allclose(box / box[0], want / want[0], rtol=1e-12)


def test_slab1d_has_a_density_jump():
    """Two states meeting at x=0, sampled at their own spacings."""
    p = S.make("slab1d", 40000)
    left = p.pos[:, 0] < 0.0
    assert left.sum() > 0 and (~left).sum() > 0
    ratio = float(np.median(p.rho[left]) / np.median(p.rho[~left]))
    assert 4.0 < ratio < 16.0, f"density jump {ratio:.2f} not ~8:1"


@pytest.mark.parametrize("fam", ("disk", "merger"))
def test_isolated_systems_are_open(fam):
    """An isolated galaxy and an isolated merger are not periodic systems."""
    p = S.make(fam, SMALL)
    assert not any(np.atleast_1d(p.periodic)), f"{fam} must be open"


@pytest.mark.parametrize("fam", ("uniform", "lattice", "closepacked", "glass",
                                 "slab1d", "slab2d", "collapse", "cosmo"))
def test_box_families_are_periodic(fam):
    p = S.make(fam, SMALL)
    assert all(np.atleast_1d(p.periodic)), f"{fam} must be periodic"


@pytest.mark.parametrize("fam,floor", [("disk", 300.0), ("merger", 300.0),
                                       ("collapse", 100.0), ("cosmo", 100.0)])
def test_structured_families_have_real_dynamic_range(fam, floor):
    """REGRESSION (merger, open-boundary change): `_sample_inside` truncated
    candidates in array order while the draw function returned disc and bulge
    as consecutive BLOCKS, so the slice dropped the entire bulge -- h_dyn fell
    from 6.4e3 to 28 and the family silently stopped testing anything.

    The floors are far below the calibrated values, so this catches a lost
    component without pinning the tuning.
    """
    p = S.make(fam, 60000)
    dyn = float(p.h.max() / p.h.min())
    assert dyn > floor, f"{fam}: h dynamic range {dyn:.3g} below {floor}"


def test_uniform_family_has_almost_none():
    """The control: it must NOT have dynamic range, or it tests nothing."""
    p = S.make("uniform", SMALL)
    assert float(p.h.max() / p.h.min()) < 5.0


def test_describe_and_occupancy():
    p = S.make("disk", 20000)
    d = S.describe(p)
    assert d["n"] == p.n
    assert d["h_dyn"] == pytest.approx(p.h.max() / p.h.min())
    assert 0.0 < d["occupancy_64"] <= 1.0
    # the disk leaves most of its box empty -- that is the point of the family
    assert d["occupancy_64"] < 0.2
    assert S.occupancy(p, npx=8) >= S.occupancy(p, npx=64)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        S.make("no_such_family", 100)
