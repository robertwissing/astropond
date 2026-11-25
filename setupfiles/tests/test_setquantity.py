"""IC_setQuantity.py: the operation vocabulary, chaining, and the aux twins.

Weighted toward silent failure modes that produce a plausible-looking output
file: an operation that computes a result and then writes the ORIGINAL array
back, an advertised operation with no implementation, and edits that touch the
packed vector aux but not its x/y/z twins. So every test here checks that the OUTPUT changed in the way asked for, and
the vector ops check the twins agree with the packed file.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import (SETUPFILES, centre_of_mass, read_aux, read_tipsy_all,
                      read_tipsy_gas, read_vector_aux,
                      write_tipsy, write_vector_aux)

SCRIPT = os.path.join(SETUPFILES, 'IC_setQuantity.py')


def run_cli(*argv, expect=0):
    p = subprocess.run([sys.executable, SCRIPT, *map(str, argv)],
                       capture_output=True, text=True)
    assert p.returncode == expect, (
        f'exit {p.returncode} (wanted {expect})\n--- stdout ---\n{p.stdout}'
        f'\n--- stderr ---\n{p.stderr}')
    return p


def check_twins(path, label='BField'):
    """The packed aux and its x/y/z twins must agree, and be the same length.

    A gas-only twin next to a full-length packed file would give a consumer
    reading components a short, misaligned field.
    """
    packed = read_vector_aux(path, label)
    for i, comp in enumerate('xyz'):
        twin = read_aux(path, label + comp)
        assert len(twin) == len(packed), f'{label}{comp} length'
        np.testing.assert_allclose(packed[:, i], twin, rtol=1e-6, atol=1e-30)
    return packed


# ---------------------------------------------------------------------------
# scalar edits
# ---------------------------------------------------------------------------

def test_set_writes_the_value_and_leaves_others_alone(disc_snapshot, tmp_path):
    out = str(tmp_path / 'set.00000')
    run_cli(disc_snapshot, '-o', out, 'set', '--field', 'temp', '--value',
            '1234', '--type', 'gas')
    before, _ = read_tipsy_gas(disc_snapshot)
    after, _ = read_tipsy_gas(out)
    np.testing.assert_allclose(after[:, 8], 1234.0, rtol=1e-5)
    np.testing.assert_allclose(after[:, 0], before[:, 0], rtol=1e-6)  # mass
    np.testing.assert_allclose(after[:, 1:4], before[:, 1:4], rtol=1e-5)


def test_set_honours_a_where_selection(disc_snapshot, tmp_path):
    out = str(tmp_path / 'where.00000')
    run_cli(disc_snapshot, '-o', out, 'set', '--field', 'temp', '--value',
            '7', '--type', 'gas', '--where', 'r<0.5')
    before, _ = read_tipsy_gas(disc_snapshot)
    after, _ = read_tipsy_gas(out)
    r = np.linalg.norm(before[:, 1:4], axis=1)
    inside, outside = r < 0.5, r >= 0.5
    assert inside.any() and outside.any()
    np.testing.assert_allclose(after[inside, 8], 7.0, rtol=1e-5)
    np.testing.assert_allclose(after[outside, 8], before[outside, 8],
                               rtol=1e-5)


def test_set_modes_add_and_multiply(disc_snapshot, tmp_path):
    before, _ = read_tipsy_gas(disc_snapshot)
    for mode, expect in (('add', before[:, 8] + 3.0),
                         ('multiply', before[:, 8] * 3.0)):
        out = str(tmp_path / f'{mode}.00000')
        run_cli(disc_snapshot, '-o', out, 'set', '--field', 'temp', '--value',
                '3', '--mode', mode, '--type', 'gas')
        after, _ = read_tipsy_gas(out)
        np.testing.assert_allclose(after[:, 8], expect, rtol=1e-4)


def test_remove_drops_particles_and_their_aux_rows(disc_snapshot, tmp_path):
    out = str(tmp_path / 'rm.00000')
    run_cli(disc_snapshot, '-o', out, 'remove', '--type', 'gas', '--where',
            'r>0.5')
    before, hb = read_tipsy_gas(disc_snapshot)
    after, ha = read_tipsy_gas(out)
    r = np.linalg.norm(before[:, 1:4], axis=1)
    assert int(ha[2]) == int((r <= 0.5).sum()) < int(hb[2])
    # the aux must have been cut to match, or a consumer reads garbage
    assert len(check_twins(out)) == int(ha[0])


# ---------------------------------------------------------------------------
# geometry: the vector aux must travel with the particles
# ---------------------------------------------------------------------------

def test_rotation_turns_the_field_with_the_positions(disc_snapshot, tmp_path):
    out = str(tmp_path / 'rot.00000')
    run_cli(disc_snapshot, '-o', out, 'rotate', '--axis', 'z', '--angle', '90')
    before, _ = read_tipsy_gas(disc_snapshot)
    after, _ = read_tipsy_gas(out)
    # +90 deg about z: (x, y) -> (-y, x)
    np.testing.assert_allclose(after[:, 1], -before[:, 2], atol=1e-5)
    np.testing.assert_allclose(after[:, 2], before[:, 1], atol=1e-5)
    B0 = read_vector_aux(disc_snapshot, 'BField')
    B1 = check_twins(out)
    np.testing.assert_allclose(B1[:, 0], -B0[:, 1], atol=1e-8)
    np.testing.assert_allclose(B1[:, 1], B0[:, 0], atol=1e-8)


def test_rotation_preserves_the_field_magnitude(disc_snapshot, tmp_path):
    out = str(tmp_path / 'rot2.00000')
    run_cli(disc_snapshot, '-o', out, 'rotate', '--axis', 'y', '--angle', '37')
    m0 = np.linalg.norm(read_vector_aux(disc_snapshot, 'BField'), axis=1)
    m1 = np.linalg.norm(check_twins(out), axis=1)
    np.testing.assert_allclose(m1, m0, rtol=1e-5)


def test_centering_moves_the_centre_of_mass_to_the_origin(disc_snapshot,
                                                          tmp_path):
    """--on mass is the ALL-TYPE centre of mass, so the gas alone does not end
    up at the origin - the whole system does."""
    out = str(tmp_path / 'cen.00000')
    run_cli(disc_snapshot, '-o', out, 'center', '--on', 'mass')
    before = read_tipsy_all(disc_snapshot)
    after = read_tipsy_all(out)
    assert np.linalg.norm(centre_of_mass(*before[:3])) > 1e-3   # was off-centre
    np.testing.assert_allclose(centre_of_mass(*after[:3]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# the bfield op
# ---------------------------------------------------------------------------

def test_bfield_scale_actually_writes_the_scaled_field(disc_snapshot,
                                                       tmp_path):
    out = str(tmp_path / 'scale.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'scale', '--value',
            '2.5')
    B0 = read_vector_aux(disc_snapshot, 'BField')
    np.testing.assert_allclose(check_twins(out), 2.5 * B0, rtol=1e-5)


def test_bfield_zero_actually_zeroes(disc_snapshot, tmp_path):
    out = str(tmp_path / 'zero.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'zero')
    assert np.abs(check_twins(out)).max() == 0.0


def test_bfield_over_rho_is_not_a_no_op(disc_snapshot, tmp_path):
    """over-rho must write B/rho, not B."""
    out = str(tmp_path / 'overrho.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'over-rho')
    gas, header = read_tipsy_gas(disc_snapshot)
    ngas = int(header[2])
    B0 = read_vector_aux(disc_snapshot, 'BField')
    B1 = check_twins(out)
    assert not np.allclose(B1, B0)
    np.testing.assert_allclose(B1[:ngas], B0[:ngas] / gas[:, 7][:, None],
                               rtol=1e-4)


def test_bfield_smooth_is_implemented_and_actually_smooths(disc_snapshot,
                                                           tmp_path):
    """smooth must reduce the local scatter of a field that starts as pure
    particle-to-particle noise."""
    out = str(tmp_path / 'sm.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'smooth',
            '--nsmooth', '32')
    _, header = read_tipsy_gas(disc_snapshot)
    ngas = int(header[2])
    B0 = read_vector_aux(disc_snapshot, 'BField')[:ngas]
    B1 = check_twins(out)[:ngas]
    assert not np.allclose(B1, B0)
    # the input is white noise, so smoothing must cut its variance hard
    assert B1.std() < 0.5 * B0.std()


def test_bfield_smooth_preserves_a_uniform_field(tmp_path):
    """Shepard normalisation: a constant field must survive smoothing exactly,
    including at the open boundary where the kernel is one-sided."""
    from conftest import make_disc
    gas, dark = make_disc(n=400, seed=3)
    path = str(tmp_path / 'uni.00000')
    write_tipsy(path, gas, dark)
    n = len(gas) + len(dark)
    B = np.zeros((n, 3))
    B[:len(gas)] = [0.3, -0.2, 0.5]
    write_vector_aux(path, 'BField', B)

    out = str(tmp_path / 'unism.00000')
    run_cli(path, '-o', out, 'bfield', '--op', 'smooth', '--passes', '3',
            '--nsmooth', '32')
    got = check_twins(out)[:len(gas)]
    np.testing.assert_allclose(got, B[:len(gas)], rtol=1e-5, atol=1e-7)


def test_bfield_smooth_strength_zero_changes_nothing(disc_snapshot, tmp_path):
    """--strength is the blend factor; 0 must be an exact no-op, which is the
    cheapest possible check that the blend is wired the right way round."""
    out = str(tmp_path / 'nostrength.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'smooth',
            '--strength', '0', '--nsmooth', '32')
    np.testing.assert_allclose(check_twins(out),
                               read_vector_aux(disc_snapshot, 'BField'),
                               rtol=1e-6, atol=1e-30)


def test_bfield_smooth_converges_with_more_passes(disc_snapshot, tmp_path):
    """Repeated smoothing must keep reducing the scatter, never amplify it."""
    stds = []
    for passes in (1, 2, 4):
        out = str(tmp_path / f'p{passes}.00000')
        run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'smooth',
                '--passes', str(passes), '--nsmooth', '32')
        stds.append(float(check_twins(out).std()))
    assert stds[0] > stds[1] > stds[2]


@pytest.mark.parametrize('flavour', ['gdforce', 'rho2', 'gdforce-own'])
def test_bfield_smooth_accepts_every_flavour(disc_snapshot, tmp_path, flavour):
    out = str(tmp_path / f'{flavour}.00000')
    run_cli(disc_snapshot, '-o', out, 'bfield', '--op', 'smooth',
            '--flavour', flavour, '--nsmooth', '32')
    assert np.isfinite(check_twins(out)).all()


# ---------------------------------------------------------------------------
# chaining and the CLI surface
# ---------------------------------------------------------------------------

def test_operations_chain_in_order(disc_snapshot, tmp_path):
    """The whole point of the rewrite: several edits in one pass."""
    out = str(tmp_path / 'chain.00000')
    run_cli(disc_snapshot, '-o', out,
            'set', '--field', 'temp', '--value', '10', '--type', 'gas',
            'bfield', '--op', 'scale', '--value', '3',
            'time', '--value', '2.5')
    after, _ = read_tipsy_gas(out)
    np.testing.assert_allclose(after[:, 8], 10.0, rtol=1e-5)
    np.testing.assert_allclose(check_twins(out),
                               3.0 * read_vector_aux(disc_snapshot, 'BField'),
                               rtol=1e-5)


def test_list_ops_and_help_do_not_need_a_file():
    p = subprocess.run([sys.executable, SCRIPT, '--list-ops'],
                       capture_output=True, text=True)
    assert p.returncode == 0
    for name in ('set', 'bfield', 'rotate', 'spin'):
        assert name in p.stdout


def test_refuses_to_overwrite_its_input(disc_snapshot):
    p = run_cli(disc_snapshot, '-o', disc_snapshot, 'time', '--value', '1',
                expect=1)
    assert 'overwrite' in (p.stdout + p.stderr)


def test_missing_file_is_a_clean_error(tmp_path):
    p = run_cli(str(tmp_path / 'nope.00000'), 'time', '--value', '1', expect=1)
    assert 'no such snapshot' in (p.stdout + p.stderr)
