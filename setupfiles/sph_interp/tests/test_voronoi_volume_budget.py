"""The PERIODIC Voronoi volume budget raises instead of warning.

A Voronoi tessellation partitions its container, so the central cells of a
periodic mesh must sum to exactly one box volume. `_from_points_periodic` grows
the image shell until that closes, and its two failure exits raise, like the
non-periodic `_tessellate` (a warning would be invisible in a batch run).

The two exits are NOT the same failure and are checked separately:
  * a DEFICIT is mesher volume error (gaps between cells) -- growing the shell
    cannot fix it, so it must fail immediately rather than re-tessellate;
  * an EXCESS still present at the maximum (one-period) shell means the
    boundary cells are never getting cut by their images.

`_VOL_BUDGET_TOL` is 1e-5, four decades above the mesher float noise
(2e-16..5e-12, rare excursions to ~1e-6) and far below any real failure
(>= ~1%), so these tests perturb by 1.5x / 0.5x -- nowhere near the boundary.
"""
import importlib.util
import warnings

import numpy as np
import pytest

from sph_interp import VoronoiGrid
from sph_interp.targets import _HAS_AREPO, _VOL_BUDGET_TOL

MESHER = 'multivoro'
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('multivoro') is None,
    reason='multivoro is required: the installed pyvoro is unreliable')

BOUNDS = [[-0.5, 0.5]] * 3


def _pts(n=200, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (n, 3))


def _scale_volumes(monkeypatch, factor):
    """Make every tessellation report `factor` x its true cell volumes.

    Perturbing the RESULT rather than the input is deliberate: it reproduces
    'the mesher handed back something that does not tile' without needing a
    mesher bug, which is precisely the class of failure the budget exists for.
    """
    orig = VoronoiGrid._tessellate.__func__

    def fake(cls, *a, **k):
        vg = orig(cls, *a, **k)
        vg.volume = vg.volume * factor
        return vg

    monkeypatch.setattr(VoronoiGrid, '_tessellate', classmethod(fake))


def test_a_deficit_raises_and_does_not_retessellate(monkeypatch):
    _scale_volumes(monkeypatch, 0.5)
    with pytest.raises(RuntimeError) as e:
        VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=True,
                                backend=MESHER, periodic_background=0)
    msg = str(e.value)
    assert 'short by' in msg
    assert 'gaps' in msg                      # names the actual failure
    assert 'cannot fix it' in msg             # and why growing the shell won't


def test_an_excess_that_never_closes_raises_at_the_maximum_shell(monkeypatch):
    _scale_volumes(monkeypatch, 1.5)
    with pytest.raises(RuntimeError) as e:
        VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=True,
                                backend=MESHER, periodic_background=0)
    msg = str(e.value)
    assert 'maximum' in msg
    assert 'one-period' in msg               # names the ceiling it gave up at


def test_the_deficit_branch_does_not_fire_inside_the_noise_band(monkeypatch):
    """The budget must not cry wolf: a perturbation an order of magnitude
    INSIDE the tolerance is float noise and has to pass."""
    _scale_volumes(monkeypatch, 1.0 - 0.1 * _VOL_BUDGET_TOL)
    vg = VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=True,
                                 backend=MESHER, periodic_background=0)
    assert vg.ncentral == 200


def test_a_healthy_periodic_mesh_still_tiles_its_box():
    """Unpatched control — the budget these tests fake is really being met."""
    vg = VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=True,
                                 backend=MESHER, periodic_background=0)
    assert float(vg.volumes().sum()) == pytest.approx(1.0, abs=_VOL_BUDGET_TOL)


@pytest.mark.parametrize('box', [(1.0, 0.09375, 0.09375),     # slab1d, quasi-1D
                                 (1.0, 1.0, 0.09375)])        # slab2d, quasi-2D
def test_a_high_aspect_box_closes_the_budget(box):
    """A thin box must still tile, and the shell must be able to GROW to get
    there.

    The thin axes here hold ~5 generator spacings, so a boundary cell's Voronoi
    reach runs past a half-period and the budget closes only at a shell of ~L;
    the shell ceiling is therefore anchored on the period.
    """
    L = np.asarray(box, dtype=float)
    bounds = [(-0.5 * L[a], 0.5 * L[a]) for a in range(3)]
    pts = np.random.default_rng(1).uniform(-0.5, 0.5, (1500, 3)) * L
    vg = VoronoiGrid.from_points(pts, bounds=bounds, periodic=True,
                                 backend=MESHER, periodic_background=0)
    # (a few of 1500 random generators get de-slivered, hence not == 1500)
    assert 1400 < vg.ncentral <= 1500
    assert float(vg.volumes().sum()) == pytest.approx(
        float(np.prod(L)), rel=_VOL_BUDGET_TOL)


def test_the_non_periodic_path_is_unchanged():
    """The open budget is checked from the same entry point, so the two paths
    stay symmetric."""
    vg = VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=False,
                                 backend=MESHER, periodic_background=0)
    assert float(vg.volumes().sum()) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# refinement: a cell too big for the periodic unwrap is fixed by SPLITTING it
# ---------------------------------------------------------------------------

def _thin_case(box=(1.0, 1.0, 0.09375), n=1500, seed=1):
    L = np.asarray(box, dtype=float)
    pts = np.random.default_rng(seed).uniform(-0.5, 0.5, (n, 3)) * L
    return np.ascontiguousarray(pts), -0.5 * L, 0.5 * L, L


def test_the_unwrap_limit_is_a_quarter_period_not_a_half():
    """The bound is on the cell's DIAMETER, so cell_rad <= L/4.

    This is the number the whole refinement hangs on, and the intuitive value
    (half a period, i.e. the cell merely FITS in the box) is measurably too
    loose: refining the 1500-generator slab meshes to L/2 left them at +1.07%
    and +0.50%, while L/4 closed both in a single round.
    """
    assert VoronoiGrid._periodic_cell_rad_limit(
        np.array([1.0, 1.0, 0.09375]), (True, True, True)) == 0.25 * 0.09375
    # a non-periodic axis imposes no unwrap constraint, so it is not counted
    assert VoronoiGrid._periodic_cell_rad_limit(
        np.array([1.0, 1.0, 0.09375]), (True, True, False)) == 0.25
    assert VoronoiGrid._periodic_cell_rad_limit(
        np.array([1.0, 1.0, 1.0]), (False, False, False)) == float('inf')


def test_insertion_targets_the_big_cells_and_avoids_the_generators():
    """Points go INSIDE the offending cells only, and never on top of an
    existing generator (that would hand the mesher a sliver instead of a fix)."""
    pts, lo, hi, L = _thin_case()
    per = (True, True, True)
    lim = VoronoiGrid._periodic_cell_rad_limit(L, per)
    # one cell flagged: everything inserted must land within its circumsphere
    cell_rad = np.full(len(pts), 0.5 * lim)
    cell_rad[7] = 4.0 * lim
    mesh = {'gen': pts, 'cell_rad': cell_rad}
    out, nadd = VoronoiGrid._insert_for_half_period(
        pts, lo, hi, per, mesh, lim, lim)
    assert nadd > 0 and len(out) == len(pts) + nadd
    added = out[len(pts):]
    d = np.abs(added - pts[7])
    d = np.minimum(d, L - d)                      # periodic separation
    assert np.all(np.linalg.norm(d, axis=1) <= cell_rad[7])
    dg = np.abs(added[:, None, :] - pts[None, :, :])
    dg = np.minimum(dg, L - dg)
    assert np.linalg.norm(dg, axis=2).min() >= 0.5 * lim

    # nothing flagged -> nothing added, and the caller's set is handed back as-is
    same, nadd = VoronoiGrid._insert_for_half_period(
        pts, lo, hi, per, {'gen': pts, 'cell_rad': np.full(len(pts), 0.5 * lim)},
        lim, lim)
    assert nadd == 0 and same is pts


def test_a_sliver_falls_through_to_the_midpoint_split():
    """The lattice can come up empty on a cell that is WIDE but not roomy: its
    circumsphere is full of other generators, so every candidate is rejected for
    sitting on one. The split cannot come up empty -- it inserts between the
    generator and its own far vertices, which are inside the (convex) cell.

    Built here as a cell whose generator sits at the origin with vertices out at
    4x the limit, surrounded by generators packed at limit/4 so no lattice point
    can survive the spacing test.
    """
    per = (True, True, True)
    L = np.array([1.0, 1.0, 1.0])
    lo, hi = -0.5 * L, 0.5 * L
    lim = VoronoiGrid._periodic_cell_rad_limit(L, per)
    fine = 0.25 * lim
    g = np.mgrid[-8:9, -8:9, -8:9].reshape(3, -1).T * fine
    pts = np.ascontiguousarray(np.concatenate(([[0.0, 0.0, 0.0]], g[np.any(g, 1)])))
    cell_rad = np.full(len(pts), 0.1 * lim)
    cell_rad[0] = 4.0 * lim
    # the offending cell's vertices, as one face block of the CSR layout
    verts = np.array([[4 * lim, 0, 0], [0, 4 * lim, 0], [0, 0, 4 * lim],
                      [-4 * lim, 0, 0]])
    mesh = {'gen': pts, 'cell_rad': cell_rad,
            'cf_start': np.array([0, 1] + [1] * (len(pts) - 1), dtype=np.int64),
            'fv_start': np.array([0, len(verts)], dtype=np.int64),
            'fvx': verts[:, 0].copy(), 'fvy': verts[:, 1].copy(),
            'fvz': verts[:, 2].copy()}
    out, nadd = VoronoiGrid._insert_for_half_period(
        pts, lo, hi, per, mesh, lim, lim)
    assert nadd > 0, 'the split must rescue what the lattice cannot place'
    added = out[len(pts):]
    # every split sits at half the distance to a far vertex, i.e. inside the cell
    assert np.all(np.linalg.norm(added - pts[0], axis=1) <= 0.5 * cell_rad[0])
    assert np.all(np.linalg.norm(added - pts[0], axis=1) >= 0.5 * lim)


@pytest.mark.skipif(not _HAS_AREPO, reason='needs the built Arepo executable')
@pytest.mark.parametrize('box', [(1.0, 0.09375, 0.09375),
                                 (1.0, 1.0, 0.09375)])
def test_native_arepo_refines_a_thin_box_instead_of_falling_back(box):
    """The quasi-2D coarse mesh (+12.9% native) must close by REFINING, and stay
    on the native per-axis-period tessellation rather than handing the thin box
    to the ghost path whose cubic clip is what the native path exists to avoid.
    """
    from sph_interp._arepo import arepo_mesh
    pts, lo, hi, L = _thin_case(box=box)
    bounds = [(lo[a], hi[a]) for a in range(3)]
    m = arepo_mesh.tessellate(pts, lo, hi, n_threads=1,
                              periodic=(True, True, True))
    native = float(np.sum(m['volume'])) / float(np.prod(L)) - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')            # the fallback warns -> fails
        vg = VoronoiGrid.from_points(pts, bounds=bounds, periodic=True,
                                     backend='arepo', periodic_background=0)
    assert float(np.sum(vg.volume[:vg.ncentral])) == pytest.approx(
        float(np.prod(L)), rel=_VOL_BUDGET_TOL)
    if abs(native) > _VOL_BUDGET_TOL:
        # refined: cells were added, and every one now fits the unwrap limit
        assert vg.ncentral > len(pts)
        lim = VoronoiGrid._periodic_cell_rad_limit(L, (True, True, True))
        assert vg.cell_rad[:vg.ncentral].max() <= lim
    else:
        # already tiles -> the generator set must be handed back UNTOUCHED;
        # refinement is a repair, not something a healthy mesh pays for
        assert vg.ncentral == len(pts)
