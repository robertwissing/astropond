"""setupfiles/IC_glasspatch.py: the sharp contact pasted from two uniform glasses."""
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'setupfiles'))
import IC_glasspatch as patch  # noqa: E402
import readtipsy as tip  # noqa: E402

M = 2.0e-4


def param(d=2, Rs=0.01, prof=3, **kw):
    p = dict(dICdensprofile=prof, dICdensdir=d, dICdensR=0.25, dICdensRsmooth=Rs, dICdensinner=10.0, dICdensouter=1.0,
             dxPeriod=1.0, dyPeriod=1.0, dzPeriod=1.0)
    p.update(kw)
    return {k: str(v) for k, v in p.items()}


def build(p, tmp_path):
    """make_pre + paste, with the random pre-ICs standing in for the relaxed glasses."""
    glass = str(tmp_path / 'g')
    patch.make_pre(p, M, glass)
    for part in ('inner', 'outer'):
        shutil.copy(f'{glass}_{part}.pre', f'{glass}_{part}_IC')
    patch.paste(p, glass)
    return np.array(tip.readtipsy(f'{glass}_IC')[0], dtype=np.float64), glass


@pytest.mark.parametrize('d,axis', [(1, 0), (2, 1), (3, 2)])
def test_slab_inner_box_is_the_dense_band_and_outer_the_whole_box(d, axis):
    _, _, _, _, box, ibox = patch.geometry(param(d=d))
    assert ibox[axis] == 0.5 and np.delete(ibox, axis).tolist() == [1.0, 1.0]
    assert box.tolist() == [1.0, 1.0, 1.0]


def test_square_inner_box_keeps_a_thin_slab_period():
    assert patch.geometry(param(d=0, prof=2))[5].tolist() == [0.5, 0.5, 0.5]
    assert patch.geometry(param(d=0, prof=2, dzPeriod=0.2))[5].tolist() == [0.5, 0.5, 0.2]


@pytest.mark.parametrize('kw', [dict(prof=1), dict(d=4), dict(d=6), dict(dICdensinner=0.5), dict(dICdensR=0.0)])
def test_anything_but_a_dense_slab_or_square_is_refused(kw):
    with pytest.raises(ValueError):
        patch.geometry(param(**kw))


def test_slab_regions_do_not_overlap_and_hold_the_nominal_mass(tmp_path):
    g, _ = build(param(), tmp_path)
    dense = g[:, 7] == 10.0
    assert np.all(np.abs(g[dense, 2]) <= 0.25)
    assert np.all(np.abs(g[~dense, 2]) >= 0.25)
    assert g[dense, 0].sum() == pytest.approx(10.0 * 0.5, rel=1e-3)
    assert g[~dense, 0].sum() == pytest.approx(1.0 * 0.5, rel=0.05)
    assert np.all(g[:, 4:7] == 0.0)


def test_square_regions_do_not_overlap_and_hold_the_nominal_mass(tmp_path):
    g, _ = build(param(d=0, prof=2), tmp_path)
    dense = g[:, 7] == 10.0
    assert np.all(np.abs(g[dense, 1:4]).max(axis=1) <= 0.25)
    assert np.all(np.abs(g[~dense, 1:4]).max(axis=1) > 0.25)
    assert g[dense, 0].sum() == pytest.approx(10.0 * 0.125, rel=1e-3)
    assert g[~dense, 0].sum() == pytest.approx(1.0 * 0.875, rel=0.05)


def test_thin_slab_square_keeps_every_inner_particle(tmp_path):
    g, glass = build(param(d=0, prof=2, dzPeriod=0.2), tmp_path)
    assert (g[:, 7] == 10.0).sum() == len(tip.readtipsy(f'{glass}_inner_IC')[0])
    assert np.abs(g[:, 3]).max() <= 0.1


def test_positions_outside_the_periodic_box_are_wrapped_before_the_carve(tmp_path):
    p = param()
    glass = str(tmp_path / 'g')
    patch.make_pre(p, M, glass)
    for part, shift in (('inner', 0.5), ('outer', 1.0)):
        g = np.array(tip.readtipsy(f'{glass}_{part}.pre')[0], dtype=np.float64)
        g[:, 2] += shift
        tip.writetipsy(g, np.zeros((0, 9)), np.zeros((0, 11)), f'{glass}_{part}_IC', np.array([len(g), 3, len(g), 0, 0, 0]),
                       np.array([0.0]))
    patch.paste(p, glass)
    g = np.array(tip.readtipsy(f'{glass}_IC')[0], dtype=np.float64)
    dense = g[:, 7] == 10.0
    assert np.all(np.abs(g[dense, 2]) <= 0.25) and np.all(np.abs(g[~dense, 2]) >= 0.25)


@pytest.mark.parametrize('kw', [dict(Rs=0.0001), dict(Rs=0.1), dict(prof=2)])
def test_the_setup_smoothing_changes_nothing(tmp_path, kw):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    ref, _ = build(param(), tmp_path / 'a')
    g, _ = build(param(**kw), tmp_path / 'b')
    assert np.array_equal(ref, g)
