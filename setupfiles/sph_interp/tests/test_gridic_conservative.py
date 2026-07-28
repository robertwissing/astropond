"""The conservative GridIC deposit: mass, momentum and TOTAL energy.

`write_grid_ic` always conserved mass (petkova's density is a mass-conserving
estimate), but built velocity and temperature from kernel-volume-weighted
averages, so momentum and energy were only approximated. It now uses the
`_content_<name>` channels that the field-reconstruction work added, and closes
the energy budget by thermalising the velocity variance the grid cannot
represent.

The distinction that matters, and that these tests pin: E_kin is SUPPOSED to
drop (a cell has one velocity; sub-cell motion is not representable), but that
energy must reappear as internal energy rather than vanish. So E_kin is allowed
to fall while E_tot must not move.
"""
import numpy as np
import pytest

from sph_interp.gridic import read_grid_ic, write_grid_ic
from sph_interp.particles import Particles

GAMMA = 5.0 / 3.0
TU = 1.0


def _particles(n=4000, seed=0, box=1.0):
    """A periodic box with a sheared velocity field AND a density contrast.

    Both are needed to exercise anything:
      * a uniform velocity makes the energy closure trivially zero;
      * a uniform DENSITY makes the old volume-weighted average and the new
        mass-weighted one coincide exactly (the weights are m/rho and m, which
        are proportional when rho is constant), so the old path would look
        perfect and the comparison would prove nothing. Real data has a density
        contrast - orzag shows the old path off by 1.5e-4 in momentum.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-box / 2, box / 2, (n, 3))
    mass = np.full(n, 1.0 / n)
    rho = 1.0 + 0.8 * np.sin(2 * np.pi * pos[:, 0] / box)
    h = np.full(n, 1.4 * (box ** 3 / n) ** (1 / 3.0))
    vx = np.sin(2 * np.pi * pos[:, 1] / box) + 0.3 * rng.normal(size=n)
    vy = np.cos(2 * np.pi * pos[:, 2] / box) + 0.3 * rng.normal(size=n)
    vz = 0.5 * rng.normal(size=n)
    T = 1.0 + 0.5 * rng.random(n)
    return Particles(pos=pos, mass=mass, rho=rho, h=h, box=np.full(3, box),
                     periodic=True,
                     values={'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz, 'T': T})


def _totals(p):
    m = p.mass
    vel = np.stack([p.values[k] for k in ('vx', 'vy', 'vz')], axis=1)
    e_kin = float(0.5 * (m * np.sum(vel ** 2, axis=1)).sum())
    e_int = float((m * p.values['T']).sum() * TU)
    return dict(mass=float(m.sum()), mom=(m[:, None] * vel).sum(axis=0),
                e_kin=e_kin, e_int=e_int, e_tot=e_kin + e_int)


def _from_file(path):
    """Budgets recomputed from the WRITTEN file, so the test covers the whole
    deposit -> serialise -> read chain rather than in-memory arrays."""
    g = read_grid_ic(path)
    cv = float(np.prod(np.asarray(g['box'])
                       / np.array([g['nx'], g['ny'], g['nz']], float)))
    rho, P = g['rho'], g['P']
    v = np.stack([g['vx'], g['vy'], g['vz']], axis=-1)
    M = rho * cv
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(rho > 0, P / ((GAMMA - 1.0) * np.maximum(rho, 1e-300)), 0)
    e_kin = float(0.5 * (M * np.sum(v ** 2, axis=-1)).sum())
    e_int = float((M * u).sum())
    return dict(mass=float(M.sum()),
                mom=(M[..., None] * v).sum(axis=(0, 1, 2)),
                e_kin=e_kin, e_int=e_int, e_tot=e_kin + e_int)


@pytest.fixture(scope='module')
def deposited(tmp_path_factory):
    p = _particles()
    d = tmp_path_factory.mktemp('gridic')
    out = {}
    for cons in (False, True):
        path = str(d / f'ic_{cons}.bin')
        write_grid_ic(p, path, npx=32, gamma=GAMMA, tu_fac=TU,
                      conservative=cons)
        out[cons] = _from_file(path)
    return _totals(p), out


def test_mass_is_conserved_either_way(deposited):
    ref, got = deposited
    for cons in (False, True):
        assert got[cons]['mass'] == pytest.approx(ref['mass'], rel=1e-12)


def test_momentum_is_conserved_to_round_off(deposited):
    """The headline gain: momentum was approximated, now it is carried."""
    ref, got = deposited
    scale = float(np.abs(ref['mom']).max())
    err = np.max(np.abs(got[True]['mom'] - ref['mom'])) / scale
    assert err < 1e-10, f'momentum error {err:.2e}'


def test_the_conservative_path_beats_the_old_one_on_momentum(deposited):
    """Guards against a silent regression to volume-weighted averaging."""
    ref, got = deposited
    scale = float(np.abs(ref['mom']).max())
    old = np.max(np.abs(got[False]['mom'] - ref['mom'])) / scale
    new = np.max(np.abs(got[True]['mom'] - ref['mom'])) / scale
    assert new < old / 100.0, f'old {old:.2e} vs new {new:.2e}'


def test_total_energy_is_conserved(deposited):
    """The closure. E_tot must be exact even though E_kin is not."""
    ref, got = deposited
    assert got[True]['e_tot'] == pytest.approx(ref['e_tot'], rel=1e-10)


def test_kinetic_energy_falls_and_reappears_as_heat(deposited):
    """The physics of the closure, stated as a test: sub-cell motion is not
    representable on a grid, so E_kin MUST drop - and exactly that much must
    turn up in E_int. A test that demanded E_kin be conserved would be wrong.
    """
    ref, got = deposited
    lost_kin = ref['e_kin'] - got[True]['e_kin']
    gained_int = got[True]['e_int'] - ref['e_int']
    assert lost_kin > 0.0, 'the test field is too smooth to exercise the closure'
    assert gained_int == pytest.approx(lost_kin, rel=1e-8)


def test_the_old_path_loses_that_energy(deposited):
    """Why this was worth doing: the volume-weighted path drops the unresolved
    kinetic energy on the floor instead of thermalising it."""
    ref, got = deposited
    assert got[False]['e_tot'] < ref['e_tot'] * (1 - 1e-6)


def test_conservative_needs_the_velocity_fields(tmp_path):
    p = _particles(n=200)
    p.values.pop('vx')
    with pytest.raises(ValueError, match='velocity components'):
        write_grid_ic(p, str(tmp_path / 'x.bin'), npx=8, conservative=True)


def test_the_input_particles_are_not_mutated(tmp_path):
    """`_v2` is added to a COPY - it must not leak into the caller's object."""
    p = _particles(n=200)
    before = set(p.values)
    write_grid_ic(p, str(tmp_path / 'y.bin'), npx=8, gamma=GAMMA, tu_fac=TU)
    assert set(p.values) == before
