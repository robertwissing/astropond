"""IC_setup_impact.py: the orbit algebra, the overlap guard, and the CLI.

The orbit functions are pure and are tested against closed-form values, since
a merger IC that is wrong here is wrong in a way no plot reveals until the run
has burned a week. The overlap guard is tested in all three of its outcomes,
because it is the only part of the script that looks at how BIG the objects
are rather than how heavy.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import IC_setup_impact as imp

from conftest import (SETUPFILES, make_disc, read_tipsy_all, write_tipsy,
                      write_vector_aux)

SCRIPT = os.path.join(SETUPFILES, 'IC_setup_impact.py')


def run_cli(*argv, expect=0):
    p = subprocess.run([sys.executable, SCRIPT, *map(str, argv)],
                       capture_output=True, text=True)
    assert p.returncode == expect, (
        f'exit {p.returncode} (wanted {expect})\n--- stdout ---\n{p.stdout}'
        f'\n--- stderr ---\n{p.stderr}')
    return p


@pytest.fixture
def compact_snapshot(tmp_path):
    """A small, genuinely compact object: radius ~0.3, so a separation of a
    few units is comfortably clear and a separation of 0.1 is not."""
    gas, dark = make_disc(n=300, seed=11, radius=0.3, thickness=0.05)
    path = str(tmp_path / 'blob.00000')
    write_tipsy(path, gas, dark)
    n = len(gas) + len(dark)
    B = np.zeros((n, 3))
    B[:len(gas), 2] = 1e-3
    write_vector_aux(path, 'BField', B)
    return path


# ---------------------------------------------------------------------------
# orbit algebra
# ---------------------------------------------------------------------------

def test_parabolic_orbit_has_exactly_the_escape_speed():
    """e = 1 means v^2 = 2GM/r at every separation - the cleanest check that
    the vis-viva branch is the right one."""
    G, M, r, rp = 1.0, 10.0, 50.0, 5.0
    vr, vt = imp.keplerian_velocity(G, M, r, rp, 1.0)
    np.testing.assert_allclose(vr ** 2 + vt ** 2, 2 * G * M / r, rtol=1e-12)


def test_angular_momentum_is_conserved_along_the_orbit():
    """L = r v_t is the constant of the motion; it must not depend on r."""
    G, M, rp, e = 1.0, 3.0, 2.0, 1.0
    L = [r * imp.keplerian_velocity(G, M, r, rp, e)[1] for r in (10, 40, 100)]
    np.testing.assert_allclose(L, L[0], rtol=1e-12)
    np.testing.assert_allclose(L[0], np.sqrt(G * M * rp * (1 + e)), rtol=1e-12)


def test_at_pericentre_the_motion_is_purely_tangential():
    vr, vt = imp.keplerian_velocity(1.0, 5.0, 3.0, 3.0, 0.7)
    assert vr == 0.0
    assert vt > 0.0


def test_the_radial_velocity_is_negative_ie_approaching():
    vr, _ = imp.keplerian_velocity(1.0, 5.0, 20.0, 3.0, 1.0)
    assert vr < 0.0


def test_a_bound_orbit_that_cannot_reach_the_separation_is_refused():
    # e = 0.5, rp = 2 -> apocentre 6; asking for r = 50 is impossible
    with pytest.raises(SystemExit, match='never reaches'):
        imp.keplerian_velocity(1.0, 1.0, 50.0, 2.0, 0.5)


def test_a_separation_inside_pericentre_is_refused():
    with pytest.raises(SystemExit, match='inside the pericentre'):
        imp.keplerian_velocity(1.0, 1.0, 1.0, 5.0, 1.0)


def test_simple_orbit_reduces_to_the_escape_speed_at_vinf_zero():
    v, vesc = imp.simple_velocity(1.0, 8.0, 4.0, 0.0)
    np.testing.assert_allclose(v, vesc, rtol=1e-12)
    np.testing.assert_allclose(vesc, np.sqrt(2 * 1.0 * 8.0 / 4.0), rtol=1e-12)


# ---------------------------------------------------------------------------
# the overlap guard
# ---------------------------------------------------------------------------

def test_measure_radius_ignores_a_sparse_tail():
    """The percentile, not the maximum: a few strays must not set the size."""
    gas, _ = make_disc(n=500, seed=1, radius=1.0, thickness=0.05,
                       with_dark=False)
    gas[0, 1:4] = [50.0, 0.0, 0.0]              # one ejected particle
    snap = type('S', (), {'data': {'gas': gas, 'dark': np.zeros((0, 9)),
                                   'star': np.zeros((0, 11))}})()
    r99, rmax = imp.measure_radius(snap, types=('gas',))
    assert rmax > 49.0
    assert r99 < 2.0


def test_overlap_check_passes_when_the_objects_fit():
    assert imp.check_overlap(1.0, 2.0, 100.0, None, allow=False) is True


def test_overlap_check_refuses_when_they_do_not():
    with pytest.raises(SystemExit, match='OVERLAP'):
        imp.check_overlap(10.0, 20.0, 25.0, None, allow=False)


def test_overlap_check_can_be_overridden(capsys):
    assert imp.check_overlap(10.0, 20.0, 25.0, None, allow=True) is False
    assert 'WARNING' in capsys.readouterr().out


def test_overlap_check_notes_a_pericentre_inside_the_objects(capsys):
    imp.check_overlap(1.0, 1.0, 100.0, 0.5, allow=False)
    assert 'collision by construction' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_cli_clone_places_two_copies_at_the_right_separation(compact_snapshot,
                                                             tmp_path):
    out = str(tmp_path / 'pair.00000')
    run_cli(compact_snapshot, '-o', out, '--orbit', 'keplerian',
            '--separation', '10', '--pericentre', '1', '--ecc', '1.0')
    gas, dark, star, header = read_tipsy_all(out)
    src = read_tipsy_all(compact_snapshot)
    assert int(header[2]) == 2 * int(src[3][2])          # gas doubled
    # the two clumps sit either side of the origin, 10 apart along x
    x = gas[:, 1]
    left, right = x[x < 0].mean(), x[x > 0].mean()
    np.testing.assert_allclose(right - left, 10.0, atol=0.5)

    with open(out + '_impact.json') as fh:
        audit = json.load(fh)
    assert audit['separation'] == 10.0
    assert audit['mass_ratio'] == pytest.approx(1.0, rel=1e-6)
    assert audit['radius_target'] == pytest.approx(audit['radius_impactor'])


def test_cli_com_frame_has_zero_total_momentum(compact_snapshot, tmp_path):
    out = str(tmp_path / 'com.00000')
    run_cli(compact_snapshot, '-o', out, '--orbit', 'simple',
            '--separation', '10', '--vinf', '0')
    gas, dark, star, _ = read_tipsy_all(out)
    p = sum((a[:, 0:1] * a[:, 4:7]).sum(axis=0)
            for a in (gas, dark, star) if len(a))
    scale = sum(np.abs(a[:, 0:1] * a[:, 4:7]).sum()
                for a in (gas, dark, star) if len(a))
    assert np.abs(p).max() / scale < 1e-5


def test_cli_refuses_an_overlapping_pair(compact_snapshot, tmp_path):
    p = run_cli(compact_snapshot, '-o', str(tmp_path / 'x.00000'), '--orbit',
                'simple', '--separation', '0.05', '--vinf', '0', expect=1)
    assert 'OVERLAP' in (p.stdout + p.stderr)


def test_cli_allow_overlap_downgrades_it_to_a_warning(compact_snapshot,
                                                      tmp_path):
    out = str(tmp_path / 'ov.00000')
    p = run_cli(compact_snapshot, '-o', out, '--orbit', 'simple',
                '--separation', '0.05', '--vinf', '0', '--allow-overlap')
    assert 'WARNING' in p.stdout
    assert os.path.exists(out)


def test_cli_no_overlap_check_skips_it_entirely(compact_snapshot, tmp_path):
    p = run_cli(compact_snapshot, '-o', str(tmp_path / 'sk.00000'), '--orbit',
                'simple', '--separation', '0.05', '--vinf', '0',
                '--no-overlap-check')
    assert 'extent:' not in p.stdout


def test_cli_radius_overrides_are_honoured(compact_snapshot, tmp_path):
    """An explicit radius must be able to rescue an otherwise-refused run."""
    out = str(tmp_path / 'ovr.00000')
    run_cli(compact_snapshot, '-o', out, '--orbit', 'simple', '--separation',
            '0.05', '--vinf', '0', '--radius-target', '0.01',
            '--radius-impactor', '0.01')
    with open(out + '_impact.json') as fh:
        audit = json.load(fh)
    assert audit['radius_target'] == 0.01


def test_cli_dry_run_writes_nothing(compact_snapshot, tmp_path):
    out = str(tmp_path / 'dry.00000')
    run_cli(compact_snapshot, '-o', out, '--orbit', 'simple', '--separation',
            '10', '--vinf', '0', '--dry-run')
    assert not os.path.exists(out)


def test_cli_refuses_to_overwrite_its_input(compact_snapshot):
    p = run_cli(compact_snapshot, '-o', compact_snapshot, '--orbit', 'simple',
                '--separation', '10', '--vinf', '0', expect=2)
    assert 'overwrite' in (p.stdout + p.stderr)
