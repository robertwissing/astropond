"""Mixed per-axis periodicity in the ghost-image Voronoi path.

`_from_points_periodic` is entered whenever ANY axis is periodic. It pads the
tessellation box only on the periodic axes (the only ones generators are
imaged across), leaving a voro++ wall at lo/hi on the others - the same
treatment the fully-open path gives them - and runs the closed-box volume
budget for mixed periodicity too. A pad on a non-periodic axis would leave its
boundary cells uncut, so the central cells would tile the PADDED box.
"""
import importlib.util

import numpy as np
import pytest

from sph_interp import Particles, VoronoiGrid, interpolate

# The ghost-image path under test is shared by pyvoro and multivoro
# (only arepo diverges to _from_arepo_periodic), so either exercises
# it -- but the installed pyvoro (pyvoro-bazan 1.0.5) returns a mesh
# whose total volume depends on the `dispersion` block-size hint and
# is wrong at many N, so a regression test resting on it would fail
# for reasons unrelated to what it checks. Use multivoro.
MESHER = 'multivoro'
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('multivoro') is None,
    reason='multivoro is required: the installed pyvoro is unreliable')

MIXED = [(True, True, False), (True, False, False), (False, True, False),
         (False, False, True)]


def _points(n=1500, seed=0, lo=None, hi=None):
    lo = np.full(3, -0.5) if lo is None else lo
    hi = np.full(3, 0.5) if hi is None else hi
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 3)), lo, hi


@pytest.mark.parametrize('per', MIXED + [(True, True, True)])
def test_central_cells_tile_exactly_one_period(per):
    """The defining property of the periodic path, which must not depend on
    HOW MANY axes are periodic."""
    pts, lo, hi = _points()
    vg = VoronoiGrid.from_points(pts, bounds=list(zip(lo, hi)), periodic=per,
                                 backend=MESHER)
    box = float(np.prod(hi - lo))
    got = float(vg.volumes()[:vg.ncentral].sum())
    assert got == pytest.approx(box, rel=1e-6), f'{got / box:.6f} x box'


@pytest.mark.parametrize('per', MIXED)
def test_no_cell_extends_beyond_a_non_periodic_wall(per):
    """The mechanism, not just the symptom: with the pad clipped there is a
    wall at lo/hi on a non-periodic axis, so no cell vertex may cross it."""
    pts, lo, hi = _points(n=800, seed=2)
    vg = VoronoiGrid.from_points(pts, bounds=list(zip(lo, hi)), periodic=per,
                                 backend=MESHER)
    verts = (vg.fvx, vg.fvy, vg.fvz)
    for a in range(3):
        if per[a]:
            continue
        v = verts[a]
        assert v.min() >= lo[a] - 1e-9, f'axis {a} below the wall'
        assert v.max() <= hi[a] + 1e-9, f'axis {a} above the wall'


def test_a_thin_slab_is_the_worst_case():
    """The aspect ratio the KH / shearing-slab geometry actually uses: the
    pad is a large fraction of the short axis, so a padded non-periodic axis
    would be off by orders of magnitude rather than tens of percent."""
    lo = np.array([-0.5, -0.5, -0.2])
    hi = np.array([0.5, 0.5, 0.2])
    pts, lo, hi = _points(n=2000, seed=5, lo=lo, hi=hi)
    vg = VoronoiGrid.from_points(pts, bounds=list(zip(lo, hi)),
                                 periodic=(True, True, False),
                                 backend=MESHER)
    box = float(np.prod(hi - lo))
    assert float(vg.volumes()[:vg.ncentral].sum()) == pytest.approx(box,
                                                                    rel=1e-6)


@pytest.mark.parametrize('per', [(True, True, False), (True, False, False)])
def test_petkova_loses_mass_only_through_the_open_walls(per):
    """A conservation statement with the open boundary accounted for.

    Mass whose kernel reaches past a WALL is genuinely outside the mesh and
    is not deposited, so a deficit is correct - but it must be confined to
    the boundary layer. Restricting to particles more than 2h from every open
    wall, the deposit conserves to the usual petkova tolerance.
    """
    pts, lo, hi = _points(n=3000, seed=7)
    n = len(pts)
    mass = np.full(n, 1.0 / n)
    h = np.full(n, 1.6 * float(np.prod(hi - lo) / n) ** (1 / 3.0))
    rho = np.ones(n)
    interior = np.ones(n, dtype=bool)
    for a in range(3):
        if not per[a]:
            interior &= (pts[:, a] - lo[a] > 2 * h) & (hi[a] - pts[:, a] > 2 * h)
    assert interior.any() and not interior.all()

    p = Particles(pos=pts[interior], mass=mass[interior], rho=rho[interior],
                  h=h[interior], box=hi - lo, periodic=per,
                  values={'rho': rho[interior]})
    vg = VoronoiGrid.from_points(pts, bounds=list(zip(lo, hi)), periodic=per,
                                 backend=MESHER)
    res = interpolate(p, vg, method='petkova', values=['rho'])
    nc = vg.ncentral
    dep = float((np.asarray(res.data['rho'])[:nc] * vg.volumes()[:nc]).sum())
    assert dep == pytest.approx(float(mass[interior].sum()), rel=2e-6)
