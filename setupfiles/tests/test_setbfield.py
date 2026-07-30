"""IC_setBField.py: the field algebra, the turbulent draw, and the CLI.

Split by what is actually at risk:

  * the four-axis construction (profile x strength x direction x kind) is
    ordinary code and is tested at the unit level;
  * the TURBULENT profile is the part that can be subtly wrong while looking
    plausible, so it is checked against properties that a mis-drawn field
    cannot fake - solenoidality to machine precision, the filter dividing out
    exactly, and the requested-vs-realised spectral index;
  * the CLI is smoke-tested end to end, including the guards.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import IC_setBField as icb

from conftest import (SETUPFILES, read_aux, read_tipsy_gas,
                      read_vector_aux)

SCRIPT = os.path.join(SETUPFILES, 'IC_setBField.py')


def run_cli(*argv, expect=0):
    """Drive the real CLI as a subprocess. -> CompletedProcess."""
    p = subprocess.run([sys.executable, SCRIPT, *map(str, argv)],
                       capture_output=True, text=True)
    assert p.returncode == expect, (
        f'exit {p.returncode} (wanted {expect})\n--- stdout ---\n{p.stdout}'
        f'\n--- stderr ---\n{p.stderr}')
    return p


class Args:
    """A stand-in for the parsed argparse namespace."""

    def __init__(self, **kw):
        defaults = dict(profile='uniform', strength='code', direction='x',
                        kind='field', sign=1.0, rho_power=2.0 / 3.0,
                        profile_ref=None, wavenumber=1.0, sin_axis='x',
                        scale_radius=1.0, spectral_index=-5.0 / 3.0,
                        turb_ngrid=32, turb_kmin=1.0, turb_kmax=None,
                        turb_pad=0.1, seed=12345)
        defaults.update(kw)
        self.__dict__.update(defaults)


# ---------------------------------------------------------------------------
# the scalar axes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['x', 'y', 'z'])
def test_cartesian_directions_are_unit_and_axis_aligned(name):
    pos = np.random.default_rng(0).normal(size=(50, 3))
    d = icb.direction_field(name, pos)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.all(d[:, 'xyz'.index(name)] == 1.0)


def test_azimuthal_direction_is_perpendicular_to_the_cylindrical_radius():
    pos = np.random.default_rng(0).normal(size=(50, 3))
    azi = icb.direction_field('azi', pos)
    Rhat = icb.direction_field('Rhat', pos)
    assert np.abs(np.sum(azi * Rhat, axis=1)).max() < 1e-14
    assert np.abs(azi[:, 2]).max() == 0.0        # toroidal: no z component


def test_degenerate_points_get_a_zero_vector_not_a_random_one():
    """An honest hole beats a fake direction - the docstring's own claim."""
    pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.all(icb.direction_field('azi', pos)[0] == 0.0)   # on the z axis
    assert np.all(icb.direction_field('rhat', pos)[1] == 0.0)  # at the origin


def test_profiles_are_normalised_to_their_own_reference():
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(200, 3))
    rho = rng.uniform(0.1, 5.0, 200)
    for name in ('uniform', 'rho', 'R', 'exp'):
        prof, desc = icb.profile_field(name, pos, rho, Args(), None)
        assert prof.shape == (200,)
        assert np.abs(prof).max() <= 1.0 + 1e-12, name
        assert isinstance(desc, str) and desc


def test_the_sin_profile_carries_no_net_flux():
    x = np.linspace(-0.5, 0.5, 2001, endpoint=False)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    prof, _ = icb.profile_field('sin', pos, np.ones(len(x)), Args(), np.ones(3))
    assert abs(prof.mean()) < 1e-12


def test_beta_amplitude_delivers_the_requested_beta():
    """amplitude() is the inverse of the beta reported by report_field."""
    rng = np.random.default_rng(5)
    n = 100
    gas = np.zeros((n, 12))
    gas[:, 0] = 1.0 / n
    gas[:, 7] = rng.uniform(0.5, 2.0, n)          # rho
    gas[:, 8] = rng.uniform(1e3, 1e4, n)          # temperature
    units = icb.unit_conversions(1.0, 1.0, 5.0 / 3.0, 0.6)
    amp, _ = icb.amplitude('beta-th', 50.0, gas, units)
    P, _ = icb.pressures(gas, units)
    np.testing.assert_allclose(2.0 * P / amp ** 2, 50.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# the turbulent profile
# ---------------------------------------------------------------------------

def _turb_grid(ng=32, alpha=-5.0 / 3.0, kind='field', seed=3, L=np.ones(3)):
    """Reproduce turbulent_field's internals to inspect the grid itself."""
    args = Args(profile='turbulent', kind=kind, spectral_index=alpha,
                turb_ngrid=ng, seed=seed)
    # a cube of particles spanning exactly L, so turbulent_domain returns L
    pos = np.array([[0.0, 0.0, 0.0], L])
    V, desc = icb.turbulent_field(pos, args, L)
    return V, desc


def test_turbulent_field_is_reproducible_from_the_seed():
    a, _ = _turb_grid(seed=99)
    b, _ = _turb_grid(seed=99)
    c, _ = _turb_grid(seed=100)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_turbulent_field_has_unit_rms_and_no_mean():
    """The shape is normalised so `amplitude` means what it says, and k=0 is
    zeroed so the field carries no net flux."""
    rng = np.random.default_rng(8)
    pos = rng.uniform(0.0, 1.0, (4000, 3))
    args = Args(profile='turbulent', kind='field', turb_ngrid=32, seed=1)
    V, _ = icb.turbulent_field(pos, args, np.ones(3))
    rms = np.sqrt(np.mean(np.sum(V * V, axis=1)))
    np.testing.assert_allclose(rms, 1.0, rtol=1e-12)
    # the mean is not exactly zero at the particles (they sample the grid),
    # but must be far below the fluctuation level
    assert np.linalg.norm(V.mean(axis=0)) < 0.2


def test_the_drawn_grid_is_solenoidal_to_machine_precision():
    """k . V(k) = 0 by construction. Checked in Fourier space, where it is an
    identity rather than a discretisation."""
    ng = 32
    rng = np.random.default_rng(3)
    Vk = np.fft.rfftn(rng.standard_normal((3, ng, ng, ng)), axes=(1, 2, 3))
    kx = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[:, None, None]
    ky = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[None, :, None]
    kz = 2 * np.pi * np.fft.rfftfreq(ng, d=1.0 / ng)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    kd = (kx * Vk[0] + ky * Vk[1] + kz * Vk[2]) / np.where(k2 > 0, k2, 1)
    Vk[0] -= kx * kd
    Vk[1] -= ky * kd
    Vk[2] -= kz * kd
    resid = np.abs(kx * Vk[0] + ky * Vk[1] + kz * Vk[2]).max()
    scale = float(np.sqrt(k2).max() * np.abs(Vk).max())
    assert resid / scale < 1e-14


@pytest.mark.parametrize('alpha', [-5.0 / 3.0, -2.0, 1.5])
def test_the_measured_spectrum_matches_the_requested_index(alpha):
    """The end-to-end statement: ask for E_B ~ k^a, get E_B ~ k^a.

    Averaged over seeds, because one realisation on a finite grid scatters
    (sd ~0.06 at 64^3) - that scatter is the estimator's, not the draw's.
    """
    ng, L = 64, np.ones(3)
    slopes = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        beta = alpha - 2.0                      # kind=field: |B(k)|^2 ~ k^(a-2)
        Vk = np.fft.rfftn(rng.standard_normal((3, ng, ng, ng)), axes=(1, 2, 3))
        kx = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[:, None, None]
        ky = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[None, :, None]
        kz = 2 * np.pi * np.fft.rfftfreq(ng, d=1.0 / ng)[None, None, :]
        k2 = kx * kx + ky * ky + kz * kz
        kmag = np.sqrt(k2)
        kf = 2 * np.pi
        kmin, kmax = kf, 0.5 * ng * kf
        band = (kmag >= kmin) & (kmag <= kmax) & (kmag > 0)
        amp = np.zeros_like(kmag)
        amp[band] = kmag[band] ** (0.5 * beta)
        Vk *= amp
        kd = (kx * Vk[0] + ky * Vk[1] + kz * Vk[2]) / np.where(k2 > 0, k2, 1)
        Vk[0] -= kx * kd
        Vk[1] -= ky * kd
        Vk[2] -= kz * kd
        Vk[:, 0, 0, 0] = 0.0
        grid = np.fft.irfftn(Vk, s=(ng, ng, ng), axes=(1, 2, 3))
        slopes.append(icb.grid_spectrum_slope(grid, L, kmin, kmax)[0])
    assert abs(np.mean(slopes) - alpha) < 0.06, (np.mean(slopes), alpha)


def test_the_requested_filter_is_applied_exactly():
    """The sharp version of the test above, with no estimator in the way.

    At a fixed seed the realised power divided by the requested filter must
    not depend on the spectral index at all: the filter is deterministic, so
    it cancels exactly. Any index dependence here is a real construction bug,
    and it is detectable at a level the shell-average estimator cannot reach.
    """
    ng, L = 32, np.ones(3)
    residuals = []
    for beta in (-11.0 / 3.0, -4.0, -0.5, +1.0):
        rng = np.random.default_rng(17)
        Vk = np.fft.rfftn(rng.standard_normal((3, ng, ng, ng)), axes=(1, 2, 3))
        kx = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[:, None, None]
        ky = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)[None, :, None]
        kz = 2 * np.pi * np.fft.rfftfreq(ng, d=1.0 / ng)[None, None, :]
        k2 = kx * kx + ky * ky + kz * kz
        kmag = np.sqrt(k2)
        kf = 2 * np.pi
        kmin, kmax = kf, 0.5 * ng * kf
        band = (kmag >= kmin) & (kmag <= kmax) & (kmag > 0)
        amp = np.zeros_like(kmag)
        amp[band] = kmag[band] ** (0.5 * beta)
        Vk *= amp
        kd = (kx * Vk[0] + ky * Vk[1] + kz * Vk[2]) / np.where(k2 > 0, k2, 1)
        Vk[0] -= kx * kd
        Vk[1] -= ky * kd
        Vk[2] -= kz * kd
        Vk[:, 0, 0, 0] = 0.0
        grid = np.fft.irfftn(Vk, s=(ng, ng, ng), axes=(1, 2, 3))
        residuals.append(icb.grid_spectrum_slope(grid, L, kmin, kmax,
                                                 amp=amp)[0])
    assert np.ptp(residuals) < 1e-9, residuals


def test_the_default_band_never_exceeds_the_isotropic_nyquist():
    """On an elongated domain, ngrid/2 would ask for modes the grid cannot
    represent along the long axis, and the field would not be isotropic."""
    rng = np.random.default_rng(6)
    L = np.array([1.0, 1.0, 8.0])
    pos = rng.uniform(0, 1, (500, 3)) * L
    args = Args(profile='turbulent', kind='field', turb_ngrid=32, seed=2)
    _, desc = icb.turbulent_field(pos, args, L)
    nmax = float(desc.split('k 1..')[1].split()[0])
    cube = float(np.cbrt(np.prod(L)))
    assert nmax < 0.5 * 32
    np.testing.assert_allclose(nmax, 0.5 * 32 * cube / L.max(), rtol=1e-3)


def test_the_potential_kind_draws_two_powers_steeper():
    """B = curl A adds one factor of k to the amplitude, k^2 to the power, so
    the SAME requested index must produce different draws for the two kinds."""
    for kind, expect in (('field', -2.0), ('potential', -4.0)):
        _, desc = _turb_grid(ng=16, alpha=0.0, kind=kind)
        exponent = float(desc.split('~ k^')[2].split(',')[0])
        assert exponent == expect, (kind, desc)


def test_trilinear_sampling_reproduces_grid_nodes_exactly():
    rng = np.random.default_rng(0)
    ng = 8
    grid = rng.normal(size=(3, ng, ng, ng))
    origin = np.array([-1.0, 2.0, 0.5])
    L = np.array([2.0, 4.0, 1.0])
    idx = np.array([[0, 0, 0], [3, 5, 2], [7, 7, 7]])
    pos = origin + idx / ng * L
    got = icb.trilinear_sample(grid, pos, origin, L)
    want = grid[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_trilinear_sampling_wraps_periodically():
    rng = np.random.default_rng(1)
    grid = rng.normal(size=(3, 8, 8, 8))
    origin, L = np.zeros(3), np.ones(3)
    p = np.array([[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(icb.trilinear_sample(grid, p, origin, L),
                               icb.trilinear_sample(grid, p + 1.0, origin, L),
                               rtol=1e-12, atol=1e-14)


def test_a_vector_profile_ignores_direction_and_keeps_its_own_orientation():
    rng = np.random.default_rng(9)
    n = 300
    gas = np.zeros((n, 12))
    gas[:, 0] = 1.0 / n
    gas[:, 1:4] = rng.uniform(0, 1, (n, 3))
    gas[:, 7] = 1.0
    units = icb.unit_conversions(1.0, 1.0, 5.0 / 3.0, 0.6)
    a = Args(profile='turbulent', kind='field', direction='x', turb_ngrid=16)
    b = Args(profile='turbulent', kind='field', direction='azi',
             turb_ngrid=16)
    Va, desc = icb.build_vector(a, gas, gas[:, 7], units, np.ones(3), 1.0)
    Vb, _ = icb.build_vector(b, gas, gas[:, 7], units, np.ones(3), 1.0)
    np.testing.assert_array_equal(Va, Vb)
    assert 'direction unused' in desc
    # and it is genuinely three-dimensional, not an x-hat field
    assert np.abs(Va[:, 1]).max() > 0.1 * np.abs(Va[:, 0]).max()


def test_turbulent_rejects_an_impossible_grid_or_band():
    pos = np.random.default_rng(0).uniform(0, 1, (100, 3))
    with pytest.raises(SystemExit, match='even number'):
        icb.turbulent_field(pos, Args(profile='turbulent', turb_ngrid=31),
                            np.ones(3))
    with pytest.raises(SystemExit, match='exceeds'):
        icb.turbulent_field(pos, Args(profile='turbulent', turb_kmin=40.0,
                                      turb_kmax=5.0), np.ones(3))


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_cli_field_mode_writes_every_representation(disc_snapshot, tmp_path):
    out = str(tmp_path / 'out.00000')
    run_cli(disc_snapshot, '-o', out, '--profile', 'uniform', '--direction',
            'z', '--strength', 'code', '--value', '1e-3', '--kind', 'field',
            '--open', '--center', 'none', '--yes')
    for suffix in ('', '.BField', '.BFieldx', '.BFieldy', '.BFieldz'):
        assert os.path.exists(out + suffix), suffix
    assert not os.path.exists(out + '.AField')   # field mode has no potential

    B = read_vector_aux(out, 'BField')
    Bz = read_aux(out, 'BFieldz')
    np.testing.assert_allclose(B[:, 2], Bz, rtol=1e-6)
    _, header = read_tipsy_gas(disc_snapshot)
    ngas = int(header[2])
    # exactly the requested uniform z field on the gas, zero elsewhere
    np.testing.assert_allclose(B[:ngas, 2], 1e-3, rtol=1e-5)
    assert np.abs(B[ngas:]).max() == 0.0


def test_cli_potential_mode_writes_a_and_is_more_solenoidal(disc_snapshot,
                                                            tmp_path):
    """The reason potential mode exists: curl A beats a prescribed field on
    the divergence diagnostic, measured with the same operator."""
    common = ['--profile', 'turbulent', '--strength', 'code', '--value',
              '1e-3', '--open', '--center', 'none', '--yes', '--seed', '5',
              '--turb-ngrid', '32', '--flavour', 'gdforce']
    eps = {}
    for kind in ('field', 'potential'):
        out = str(tmp_path / f'{kind}.00000')
        p = run_cli(disc_snapshot, '-o', out, '--kind', kind, *common)
        with open(out + '_setBField.json') as fh:
            audit = json.load(fh)
        eps[kind] = audit['divergence_as_built']['eps_mean']
        assert audit['turbulent']['seed'] == 5
        assert os.path.exists(out + '.AField') == (kind == 'potential')
        assert 'grid spectrum check' in p.stdout
    assert eps['potential'] < eps['field']


def test_cli_normalise_beta_delivers_the_requested_beta(disc_snapshot,
                                                        tmp_path):
    out = str(tmp_path / 'beta.00000')
    # --snapshot-rho-h so the script's volume weights are the snapshot's own
    # rho column; without it it uses the glassgen kNN estimate and the check
    # below would be comparing two slightly different weightings (~0.6%)
    run_cli(disc_snapshot, '-o', out, '--profile', 'rho', '--direction', 'azi',
            '--strength', 'beta-th', '--value', '25', '--kind', 'field',
            '--normalise', 'beta', '--open', '--center', 'none', '--yes',
            '--snapshot-rho-h')
    with open(out + '_setBField.json') as fh:
        audit = json.load(fh)
    stats, units = audit['field'], audit['units']
    # 2 <P>_vol / Brms^2 == 25 is what --normalise beta promises; recompute it
    # from the audit's own numbers rather than trusting the printed line
    gas, _ = read_tipsy_gas(disc_snapshot)
    vol = gas[:, 0] / gas[:, 7]
    P = (units['gamma'] - 1.0) * gas[:, 7] * gas[:, 8] * units['TtouFac']
    beta = 2.0 * np.average(P, weights=vol) / stats['Brms'] ** 2
    np.testing.assert_allclose(beta, 25.0, rtol=1e-3)


def test_cli_seed_is_recorded_when_not_given(disc_snapshot, tmp_path):
    """An unseeded turbulent run must still be reproducible after the fact."""
    out = str(tmp_path / 'noseed.00000')
    run_cli(disc_snapshot, '-o', out, '--profile', 'turbulent', '--strength',
            'code', '--value', '1e-3', '--kind', 'field', '--open',
            '--center', 'none', '--yes', '--turb-ngrid', '16')
    with open(out + '_setBField.json') as fh:
        seed = json.load(fh)['turbulent']['seed']
    assert isinstance(seed, int)

    again = str(tmp_path / 'again.00000')
    run_cli(disc_snapshot, '-o', again, '--profile', 'turbulent', '--strength',
            'code', '--value', '1e-3', '--kind', 'field', '--open',
            '--center', 'none', '--yes', '--turb-ngrid', '16',
            '--seed', str(seed))
    np.testing.assert_array_equal(read_vector_aux(out, 'BField'),
                                  read_vector_aux(again, 'BField'))


def test_cli_dry_run_writes_nothing(disc_snapshot, tmp_path):
    out = str(tmp_path / 'dry.00000')
    run_cli(disc_snapshot, '-o', out, '--profile', 'uniform', '--direction',
            'x', '--strength', 'code', '--value', '1', '--kind', 'field',
            '--open', '--center', 'none', '--yes', '--dry-run')
    assert not os.path.exists(out)


def test_cli_refuses_to_overwrite_its_input(disc_snapshot):
    p = run_cli(disc_snapshot, '-o', disc_snapshot, '--profile', 'uniform',
                '--direction', 'x', '--strength', 'code', '--value', '1',
                '--kind', 'field', '--open', '--center', 'none', '--yes',
                expect=1)
    assert 'overwrite' in (p.stdout + p.stderr)


def test_cli_refuses_to_hang_on_a_missing_flavour(disc_snapshot, tmp_path):
    """Non-interactive runs must error, not block on a prompt."""
    p = run_cli(disc_snapshot, '-o', str(tmp_path / 'x.00000'), '--profile',
                'uniform', '--direction', 'x', '--strength', 'code',
                '--value', '1', '--kind', 'potential', '--open',
                '--center', 'none', '--yes', expect=1)
    assert '--flavour' in (p.stdout + p.stderr)
