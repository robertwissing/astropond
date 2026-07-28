"""Every installed mesher must satisfy the one invariant everything rests on.

A Voronoi tessellation of a point set PARTITIONS the container: the cell
volumes sum to the container volume, exactly. Every conservation property in
this package - petkova's mass budget, the periodic volume budget, the AMR
capture ledgers - is downstream of that. Nothing else in the suite asserted it
per backend, which is how a mesher that fails it at half the N values tried
went unnoticed for so long: the default backend (arepo) is fine, and the tests
that name a backend explicitly happened to use N values where the broken call
still worked.

The N values here include the ones that used to FAIL (400, 5000) as well as
ones that always passed (1500, 3000), so this cannot go quiet again by luck of
parameter choice.

Also asserts the meshers AGREE. The Voronoi tessellation of a point set is
unique, so two correct implementations must produce identical volumes - a
much sharper statement than each being self-consistent.
"""
import importlib.util

import numpy as np
import pytest

from sph_interp import VoronoiGrid
from sph_interp.targets import _HAS_MULTIVORO, _resolve_backend

LO, HI = -0.5, 0.5
BOUNDS = [[LO, HI]] * 3
BOX = 1.0
NS = [400, 1500, 3000, 5000]


def _available():
    out = []
    if importlib.util.find_spec('pyvoro') is not None:
        out.append('pyvoro')
    if _HAS_MULTIVORO:
        out.append('multivoro')
    if _resolve_backend('auto') == 'arepo':
        out.append('arepo')
    return out


AVAILABLE = _available()

# No backend is expected to fail. This dict existed while pyvoro was thought
# to be broken; the actual fault was our own call passing `radii=[]`, which
# made pyvoro substitute a nonzero uniform radius and run a RADICAL
# tessellation (see _tessellate). Kept as the mechanism for quarantining a
# backend if one is ever found genuinely broken.
BROKEN = {}


def _params():
    for b in AVAILABLE:
        for n in NS:
            marks = [pytest.mark.xfail(reason=BROKEN[b], strict=False)] \
                if b in BROKEN else []
            yield pytest.param(b, n, marks=marks, id=f'{b}-{n}')


@pytest.mark.parametrize('backend,n', list(_params()))
def test_cells_partition_the_container(backend, n):
    # PERIODIC: the one geometry every backend handles directly and where the
    # invariant is exact. (An OPEN arepo build requires background guard
    # generators, whose cells extend past the box, so the same one-line budget
    # would not apply there.)
    rng = np.random.default_rng(0)
    pts = rng.uniform(LO, HI, (n, 3))
    vg = VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=True,
                                 backend=backend)
    got = float(vg.volumes()[:vg.ncentral].sum())
    # 1e-5, not machine precision: arepo rebuilds each cell as a ConvexHull of
    # float32 dumped vertices and carries a ~5e-7 volume floor for it (the
    # accuracy gap of the dismissed task 6), while multivoro is exact to ~1e-12.
    # The point of this guard is catastrophic failure - pyvoro misses it by a
    # factor of 1000 - so a tolerance that accommodates arepo costs nothing.
    assert got == pytest.approx(BOX, rel=1e-5), (
        f'{backend} at N={n}: cell volumes sum to {got:.6g}, not {BOX} - '
        f'the mesh is not a partition of the container')


@pytest.mark.parametrize('n', NS)
def test_a_cell_has_a_plausible_number_of_faces(n):
    """The direct symptom of a failed neighbour search, and independent of the
    volume budget: a 3D Poisson Voronoi cell averages ~15.5 faces. Half that
    means half the neighbours were never found. Checked on the DEFAULT backend,
    which is what callers actually get."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(LO, HI, (n, 3))
    vg = VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=True)
    faces = np.diff(vg.cf_start)[:vg.ncentral]
    assert np.median(faces) > 10, (
        f'median {np.median(faces)} faces per cell at N={n} - too few for a '
        'Poisson Voronoi mesh (~15.5); the neighbour search is missing cells')


@pytest.mark.skipif(len(AVAILABLE) < 2, reason='need two backends to compare')
@pytest.mark.parametrize('n', [1500, 3000])
def test_the_working_backends_agree_exactly(n):
    """The tessellation is unique, so correct implementations must AGREE, not
    merely each be self-consistent.

    `merge_tol=0` is essential: the de-sliver merge floor is BACKEND-AWARE
    (`_DESLIVER_FRAC`: multivoro 0.20 x median-NN, arepo 0.0), so by default
    the two mesh DIFFERENT generator sets - 1496 cells vs 1500 on this input -
    and a per-cell comparison would be meaningless. With the merge off they
    agree to 2e-6 worst case, which is arepo's float32 hull-reconstruction
    floor (the dismissed task 6), not a disagreement about the geometry.
    """
    good = [b for b in AVAILABLE if b not in BROKEN]
    if len(good) < 2:
        pytest.skip('need two trusted backends')
    rng = np.random.default_rng(0)
    pts = rng.uniform(LO, HI, (n, 3))
    ref = None
    for b in good:
        g = VoronoiGrid.from_points(pts, bounds=BOUNDS, periodic=True,
                                    backend=b, merge_tol=0)
        v = np.sort(g.volumes()[:g.ncentral])
        if ref is None:
            ref, ref_name = v, b
        else:
            # 1e-4 sits ~50x above the measured 2e-6 hull floor and ~4 orders
            # below the factor-level disagreement a failed neighbour search
            # produces
            np.testing.assert_allclose(
                v, ref, rtol=1e-4, atol=1e-12,
                err_msg=f'{b} disagrees with {ref_name} at N={n}')
