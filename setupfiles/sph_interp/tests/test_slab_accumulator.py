"""Slabbed accumulators.

The petkova accumulators are `(nthreads,) + cell_shape + (K,)` — N_grid x
nthreads with NO N_part axis — so a 256³ grid on a 128-thread node wants 112 GB
and simply dies. The fix is to deposit in G passes over z slabs with
accumulators only one slab deep: peak memory falls by G, and because the slab
clip happens in the cheap bbox loop BEFORE `_cube_integral`, every cell integral
is still done exactly once.

What that buys is a memory ceiling, and what it must not cost is the answer.
These tests pin the strong version of that: the slabbed result is BIT-IDENTICAL
to the unslabbed one, not merely close. It can be, because slabbing only skips
cells — for any given cell the set of contributions and the order they are
summed in are unchanged.

The subtle half is the PERIODIC axis. There the raw z range is in unwrapped
index space and a kernel
straddling the boundary lands on BOTH ends of the grid, so clipping the loop
bounds would silently drop the wrapped half. The deposit tests the slab in
WRAPPED index space instead; `test_a_kernel_wrapping_the_z_boundary_*` is the
regression test for exactly that.
"""
import numpy as np
import numba
import pytest

from sph_interp import Particles, UniformGrid, interpolate
from sph_interp import petkova_method as pm

GS = [1, 2, 3, 5, 8]


def _periodic(n=8000, seed=0, box=1.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-box / 2, box / 2, (n, 3))
    mass = np.full(n, 1.0 / n)
    h = np.full(n, 0.05 * box)
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    return Particles(pos=pos, mass=mass, rho=rho, h=h,
                     values={'rho': rho, 'q': np.sin(pos[:, 0] * 5.0)},
                     box=np.full(3, box), periodic=True)


def _open(n=8000, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 0.3, (n, 3))
    r = np.linalg.norm(pos, axis=1)
    mass = np.full(n, 1.0 / n)
    h = 0.03 + 0.06 * (r / r.max())
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    support = 2.0 * h[:, None]
    lo = (pos - support).min(axis=0)
    hi = (pos + support).max(axis=0)
    p = Particles(pos=pos, mass=mass, rho=rho, h=h,
                  values={'rho': rho, 'q': np.cos(pos[:, 2] * 4.0)},
                  box=None, periodic=False)
    return p, UniformGrid(lo=lo, hi=hi, npx=32)


@pytest.fixture(autouse=True)
def _fixed_threads():
    """Bit-identity is claimed at a FIXED thread count: the reduction sums over
    threads in index order, so changing nthreads changes summation order (as it
    already does without slabbing)."""
    old = numba.get_num_threads()
    numba.set_num_threads(4)
    yield
    numba.set_num_threads(old)


def _grids(p, names, **kw):
    g = UniformGrid.centered(p.box, 32)
    return {G: interpolate(p, g, method='petkova', values=names, slabs=G,
                           **kw) for G in GS}, g


def test_periodic_slabbing_is_bit_identical():
    res, _ = _grids(_periodic(), ['rho', 'q'])
    for name in ('rho', 'q'):
        ref = np.asarray(res[1].data[name])
        for G in GS[1:]:
            np.testing.assert_array_equal(np.asarray(res[G].data[name]), ref)


def test_open_slabbing_is_bit_identical():
    p, g = _open()
    ref = None
    for G in GS:
        r = interpolate(p, g, method='petkova', values=['rho', 'q'], slabs=G)
        got = np.asarray(r.data['q'])
        if ref is None:
            ref = got
        else:
            np.testing.assert_array_equal(got, ref)


def test_conservative_fields_are_bit_identical_too():
    """Exercises the `_content_<name>` channel and the mass-weighted divisor,
    i.e. the num accumulator without the den one."""
    p = _periodic()
    res, _ = _grids(p, ['rho', 'q'], conservative_fields=True)
    for name in ('q', '_content_q', '_mass'):
        ref = np.asarray(res[1].data[name])
        for G in GS[1:]:
            np.testing.assert_array_equal(np.asarray(res[G].data[name]), ref)


def test_mass_is_conserved_at_every_slab_count():
    p = _periodic()
    res, g = _grids(p, ['rho'])
    for G in GS:
        got = float((np.asarray(res[G].data['rho']) * g.cell_volume).sum())
        assert got == pytest.approx(float(p.mass.sum()), rel=1e-12)


def test_a_kernel_wrapping_the_z_boundary_keeps_all_its_mass():
    """The trap in the periodic path. One particle just inside the +z face with
    a kernel wide enough to wrap: its raw z indices run past nz, and clamping
    them to a slab (rather than testing the WRAPPED index) would drop the part
    that belongs at the -z end. Mass loss is the visible symptom."""
    box = 1.0
    pos = np.array([[0.0, 0.0, 0.5 - 1e-3]])
    h = np.array([0.12])                       # 2h = 0.24 >> the 1e-3 clearance
    mass = np.array([1.0])
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    p = Particles(pos=pos, mass=mass, rho=rho, h=h, values={'rho': rho},
                  box=np.full(3, box), periodic=True)
    g = UniformGrid.centered(p.box, 32)
    for G in GS:
        r = interpolate(p, g, method='petkova', values=['rho'], slabs=G)
        got = float((np.asarray(r.data['rho']) * g.cell_volume).sum())
        assert got == pytest.approx(1.0, rel=1e-10), f"mass lost at G={G}"


def test_a_kernel_wrapping_the_z_boundary_lands_on_both_ends():
    """Conservation alone could be met by dumping the wrapped mass in the wrong
    place, so check the geometry too: both z ends must light up."""
    box = 1.0
    pos = np.array([[0.0, 0.0, 0.5 - 1e-3]])
    h = np.array([0.12])
    mass = np.array([1.0])
    rho = mass * 30 / (4 * np.pi / 3 * (2 * h) ** 3)
    p = Particles(pos=pos, mass=mass, rho=rho, h=h, values={'rho': rho},
                  box=np.full(3, box), periodic=True)
    g = UniformGrid.centered(p.box, 32)
    ref = np.asarray(interpolate(p, g, method='petkova', values=['rho'],
                                 slabs=1).data['rho'])
    assert ref[:, :, 0].sum() > 0.0 and ref[:, :, -1].sum() > 0.0
    for G in GS[1:]:
        got = np.asarray(interpolate(p, g, method='petkova', values=['rho'],
                                     slabs=G).data['rho'])
        np.testing.assert_array_equal(got, ref)


def test_slab_count_returns_one_when_the_accumulator_already_fits():
    assert pm._slab_count(12, 64, 64, 64, 1, cap=2 * 1024**3) == 1


def test_slab_count_scales_with_the_cap():
    """64³, 12 threads, K=1 is 0.0503 GB — halve the cap and it must split."""
    full = 12 * 64 * 64 * 64 * 1 * 8
    assert pm._slab_count(12, 64, 64, 64, 1, cap=full) == 1
    assert pm._slab_count(12, 64, 64, 64, 1, cap=full // 2) == 2
    assert pm._slab_count(12, 64, 64, 64, 1, cap=full // 4) == 4


def test_slab_count_never_exceeds_nz():
    """One z layer is the quantum: it cannot be subdivided, so an impossible
    cap gives nz passes rather than an error or an infinite split."""
    assert pm._slab_count(128, 256, 256, 256, 7, cap=1) == 256


def test_the_resulting_accumulator_really_fits_the_cap():
    nt, nx, ny, nz, ns = 12, 128, 128, 128, 7
    cap = 64 * 1024**2
    G = pm._slab_count(nt, nx, ny, nz, ns, cap=cap)
    ksz = int(np.max(np.diff(pm._slab_edges(nz, G))))
    assert nt * nx * ny * ksz * ns * 8 <= cap


def test_slab_edges_cover_z_exactly_once():
    for nz, G in [(64, 1), (64, 7), (70, 8), (13, 13), (13, 5)]:
        e = pm._slab_edges(nz, G)
        assert e[0] == 0 and e[-1] == nz
        assert np.all(np.diff(e) > 0)


def test_an_invalid_slab_count_is_refused():
    p = _periodic(n=500)
    g = UniformGrid.centered(p.box, 8)
    with pytest.raises(ValueError, match='slabs must be >= 1'):
        interpolate(p, g, method='petkova', values=['rho'], slabs=0)


def test_more_slabs_than_z_cells_is_clamped_not_an_error():
    p = _periodic(n=500)
    g = UniformGrid.centered(p.box, 8)
    r = interpolate(p, g, method='petkova', values=['rho'], slabs=999)
    got = float((np.asarray(r.data['rho']) * g.cell_volume).sum())
    assert got == pytest.approx(float(p.mass.sum()), rel=1e-12)
