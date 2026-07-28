"""The petkova deposit's particle reordering (`_balanced_order`).

Particles arrive in spatially coherent order, so the expensive (large-h) ones
sit contiguously: on the bench's clustered set the last 3 of 12 equal blocks of
the array held 99.5% of the work. numba's prange chunks the loop contiguously,
so most threads had nothing to do — petkova scaled 1.03x on 2 threads and 1.04x
on 4, while a pure-compute control in the same process scaled 2.01x and 3.94x.
The `_CHUNKSIZE` knob that was supposed to prevent this does nothing on numba
0.65.1 (identical run time at 1, 4, 16, 256, 4096 and 0).

Reordering the particles fixed it: 8.4x on 12 threads, 21.2 s -> 6.1 s.

What must be TRUE regardless of speed, and is what these tests pin: the deposit
is a sum into cell-indexed accumulators, so the answer cannot depend on the
order particles are visited — beyond floating-point associativity.
"""
import numpy as np
import pytest

import numba

from sph_interp import AMRGrid, Particles, UniformGrid, interpolate
from sph_interp.petkova_method import _BALANCE_MIN_N, _balanced_order


def _skewed(n=8000, seed=0, box=1.0):
    """Positions in spatially coherent order with h growing along the array —
    the pathological layout, made explicit."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-box / 2, box / 2, (n, 3))
    pos = pos[np.argsort(pos[:, 0])]                 # coherent order
    mass = np.full(n, 1.0 / n)
    h = np.linspace(0.004, 0.05, n)                  # cheap ... expensive
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    return Particles(pos=pos, mass=mass, rho=rho, h=h,
                     values={'rho': rho, 'q': np.sin(pos[:, 1] * 7)},
                     box=np.full(3, box), periodic=True)


def test_the_permutation_is_deterministic():
    """A reproducible deposit matters more than an optimal one: two runs of the
    same build must give bit-identical output."""
    numba.set_num_threads(4)
    a = _balanced_order(50_000)
    b = _balanced_order(50_000)
    np.testing.assert_array_equal(a, b)


def test_it_is_a_permutation_not_a_subset():
    numba.set_num_threads(4)
    o = _balanced_order(50_000)
    np.testing.assert_array_equal(np.sort(o), np.arange(50_000))


def test_it_declines_when_it_cannot_pay():
    numba.set_num_threads(4)
    assert _balanced_order(_BALANCE_MIN_N - 1) is None
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        assert _balanced_order(10 * _BALANCE_MIN_N) is None
    finally:
        numba.set_num_threads(old)


def test_every_contiguous_chunk_gets_a_fair_share_of_the_work():
    """The property the reorder exists for. Unbalanced: one twelfth of the
    array holds ~40% of the work; balanced: every twelfth holds ~1/12."""
    p = _skewed(50_000)
    work = p.h ** 3
    raw = np.array([b.sum() for b in np.array_split(work, 12)])
    numba.set_num_threads(4)
    o = _balanced_order(len(work))
    bal = np.array([b.sum() for b in np.array_split(work[o], 12)])
    assert raw.max() / raw.mean() > 2.0, 'the fixture is not skewed enough'
    assert bal.max() / bal.mean() < 1.1


@pytest.mark.parametrize('nt', [1, 4])
def test_the_deposit_is_independent_of_particle_order(nt):
    """The correctness statement. Reordering may only change floating-point
    summation order, so a shuffled input must reproduce the same grid."""
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(nt)
        p = _skewed()
        grid = UniformGrid.centered(p.box, 24)
        a = interpolate(p, grid, method='petkova', values=['rho', 'q'])
        rng = np.random.default_rng(7)
        k = rng.permutation(p.n)
        q = Particles(pos=np.ascontiguousarray(p.pos[k]), mass=p.mass[k],
                      rho=p.rho[k], h=p.h[k],
                      values={n: v[k] for n, v in p.values.items()},
                      box=p.box, periodic=p.periodic)
        b = interpolate(q, grid, method='petkova', values=['rho', 'q'])
        for name in ('rho', 'q'):
            x, y = np.asarray(a.data[name]), np.asarray(b.data[name])
            scale = np.abs(x).max() or 1.0
            assert np.abs(x - y).max() / scale < 1e-12, name
    finally:
        numba.set_num_threads(old)


def test_thread_count_does_not_change_the_result():
    """Reordering happens per call and depends on the thread count, so the
    deposit must still agree across thread counts."""
    old = numba.get_num_threads()
    try:
        p = _skewed()
        grid = UniformGrid.centered(p.box, 24)
        out = []
        for nt in (1, 4):
            numba.set_num_threads(nt)
            out.append(np.asarray(interpolate(
                p, grid, method='petkova', values=['rho']).data['rho']))
        scale = np.abs(out[0]).max()
        assert np.abs(out[0] - out[1]).max() / scale < 1e-12
    finally:
        numba.set_num_threads(old)


def test_mass_is_still_conserved_on_every_target():
    """The reorder must not touch the budget that petkova exists for."""
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(4)
        p = _skewed()
        for grid in (UniformGrid.centered(p.box, 24),
                     AMRGrid.build(p, base_npx=8, max_level=2)):
            res = interpolate(p, grid, method='petkova', values=['rho'])
            rho = np.asarray(res.data['rho'])
            vol = (grid.cell_volume if isinstance(grid, UniformGrid)
                   else grid.volumes())
            got = float((rho * vol).sum())
            assert got == pytest.approx(float(p.mass.sum()), rel=1e-10)
    finally:
        numba.set_num_threads(old)
