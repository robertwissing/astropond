"""Generators vs the declared domain.

A PERIODIC axis has a canonical position for every coordinate, and
`from_points` wraps onto it. A NON-PERIODIC axis does not:
a generator outside [lo, hi] there is simply outside the domain and the mesher
has no cell for it.

voro++ does fail on that - but with "number of cells found was not equal to
the number of particles", which is the SAME message it emits for coincident
generators (handled by the tol_dup floor). These tests pin the pre-check that
tells the two causes apart.
"""
import importlib.util

import numpy as np
import pytest

from sph_interp import VoronoiGrid

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


def _pts(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, (n, 3))


BOUNDS = [[-0.5, 0.5]] * 3


def test_a_generator_outside_a_non_periodic_axis_is_named():
    pts = _pts()
    pts[7, 2] = 3.0
    with pytest.raises(ValueError, match='outside the domain'):
        VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=False,
                                backend=MESHER, periodic_background=0)


def test_the_message_identifies_the_axis_and_the_excursion():
    """The point of the check is that the message is actionable, so assert on
    what it actually says rather than only that it raised."""
    pts = _pts()
    pts[3, 1] = 2.0
    with pytest.raises(ValueError) as e:
        VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=False,
                                backend=MESHER, periodic_background=0)
    msg = str(e.value)
    assert 'y axis' in msg
    assert '1 generator' in msg
    assert '1.5' in msg                      # the excursion beyond hi = 0.5


def test_only_the_non_periodic_axes_are_checked():
    """The same coordinate is fine on a periodic axis - it wraps."""
    pts = _pts()
    pts[5, 0] = 2.4
    VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=(True, False, False),
                            backend=MESHER, periodic_background=0)


def test_roundoff_scale_excursions_are_clamped_not_refused():
    """An IC writer landing a hair beyond a face must keep working, exactly as
    the periodic branch forgives the same thing."""
    pts = _pts()
    pts[1, 2] = 0.5 + 1e-13
    pts[2, 2] = -0.5 - 1e-13
    vg = VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=False,
                                 backend=MESHER, periodic_background=0)
    assert len(vg.volumes()) == len(pts)


def test_an_in_bounds_set_is_untouched():
    pts = _pts()
    before = pts.copy()
    VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=False,
                            backend=MESHER, periodic_background=0)
    np.testing.assert_array_equal(pts, before)   # the copy is the callee's
