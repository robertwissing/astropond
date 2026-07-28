"""The PERIODIC Voronoi volume budget raises instead of warning.

A Voronoi tessellation partitions its container, so the central cells of a
periodic mesh must sum to exactly one box volume. `_from_points_periodic` grows
the image shell until that closes -- but until 2026-07-28 its two failure exits
called `warnings.warn` and handed the mesh back anyway, while the non-periodic
`_tessellate` raised on the same budget. That asymmetry is what these tests
remove: warnings are invisible in a batch run, which is exactly how task 3's
+50%-per-axis error survived.

The two exits are NOT the same failure and are checked separately:
  * a DEFICIT is mesher volume error (gaps between cells) -- growing the shell
    cannot fix it, so it must fail immediately rather than re-tessellate;
  * an EXCESS still present at the maximum (min-image, half-period) shell means
    the boundary cells are never getting cut by their images.

`_VOL_BUDGET_TOL` is 1e-5, four decades above the mesher float noise
(2e-16..5e-12, rare excursions to ~1e-6) and far below any real failure
(>= ~1%), so these tests perturb by 1.5x / 0.5x -- nowhere near the boundary.
"""
import importlib.util

import numpy as np
import pytest

from sph_interp import VoronoiGrid
from sph_interp.targets import _VOL_BUDGET_TOL

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
    mesher bug, which is precisely the class of failure the budget exists for
    (task 3's pad, task 13's `radii=[]`).
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
    assert 'half-period' in msg


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


def test_the_non_periodic_path_is_unchanged():
    """The open budget already raised; this pins that it still does, from the
    same entry point, so the two paths stay symmetric."""
    vg = VoronoiGrid.from_points(_pts(), bounds=BOUNDS, periodic=False,
                                 backend=MESHER, periodic_background=0)
    assert float(vg.volumes().sum()) == pytest.approx(1.0, rel=1e-9)
