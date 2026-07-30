import numpy as np
import pytest

from glassgen import resample
from glassgen.resample_cli import _default_output_prefix


def _setup(n=400, seed=0):
    rng = np.random.default_rng(seed)
    box = np.array([2.0, 2.0, 2.0])
    pos = (rng.random((n, 3)) - 0.5) * box
    mass = np.full(n, 3.0)
    aux = {'vel': rng.random((n, 3)), 'u': np.full(n, 2.0),
           'B': rng.random((n, 3)), 'momenti': np.full(n, 1.0),
           'rho': np.full(n, 1.0), 'h': np.full(n, 0.1)}
    return pos, mass, aux, box


def test_default_output_name_includes_count_factor():
    infile = '/data/galaxy_Bclean.00000'
    assert _default_output_prefix(infile, 'merge', 10) == (
        '/data/galaxy_Bclean_merge_1024x')
    assert _default_output_prefix(infile, 'split', 3) == (
        '/data/galaxy_Bclean_split_8x')
    assert _default_output_prefix(infile, 'split', 1) == (
        '/data/galaxy_Bclean_split_2x')
    assert _default_output_prefix(infile, 'reconstruct', 1) == (
        '/data/galaxy_Bclean_recon')


def test_split_count_and_mass_conservation():
    pos, mass, aux, box = _setup()
    n = len(pos)
    for factor in (2, 4, 8):
        p2, m2, a2 = resample.split(pos, mass, aux, box=box, factor=factor)
        assert len(p2) == n * factor
        assert np.isclose(m2.sum(), mass.sum())          # total mass conserved
        assert np.allclose(m2, mass[0] / factor)         # per-child = parent/factor


def test_split_reports_each_binary_pass():
    pos, mass, aux, box = _setup(n=100)
    events = []
    p2, _, _ = resample.split(
        pos, mass, aux, box=box, factor=8, nsmooth=32,
        progress=lambda p, n0, n1: events.append((p, n0, n1)))
    assert events == [(1, 100, 200), (2, 200, 400), (3, 400, 800)]
    assert len(p2) == 800


def test_split_rejects_non_power_of_two():
    pos, mass, aux, box = _setup()
    with pytest.raises(ValueError):
        resample.split(pos, mass, aux, box=box, factor=3)


def test_split_field_rules():
    pos, mass, aux, box = _setup()
    p2, m2, a2 = resample.split(pos, mass, aux, box=box, factor=8)
    # extensive fields (u, B, momenti) are halved 3x -> sum conserved
    assert np.isclose(aux['u'].sum(), a2['u'].sum())
    assert np.allclose(aux['B'].sum(0), a2['B'].sum(0))
    assert np.isclose(aux['momenti'].sum(), a2['momenti'].sum())
    # intensive fields inherited -> momentum conserved (vel mass-weighted)
    p0 = (mass[:, None] * aux['vel']).sum(0)
    p1 = (m2[:, None] * a2['vel']).sum(0)
    assert np.allclose(p0, p1)


def test_split_children_local_and_in_box():
    pos, mass, aux, box = _setup()
    p2, m2, a2 = resample.split(pos, mass, aux, box=box, factor=2,
                                nsmooth=64, dist=0.4)
    assert np.isfinite(p2).all()
    assert np.all(np.abs(p2) <= box / 2 + 1e-9)
    # each child pair sits near its parent: min-image separation small (< box/2)
    sep = p2[0::2] - p2[1::2]
    sep -= box * np.rint(sep / box)
    assert np.all(np.linalg.norm(sep, axis=1) < box.min() / 2)


def test_split_orient_sweep_variants():
    # jrN: iterative azimuth relaxation (randomized Jacobi). Same dumbbell
    # invariants as any orient (pair centroid = parent, in box); jr0 = the
    # bare k1jitter initializer. The sweeps target the global min pair
    # spacing; on this fixed seed the improvement is deterministic, so it
    # doubles as a regression guard.
    pos, mass, aux, box = _setup()
    from scipy.spatial import cKDTree

    def dmin(p):
        w = np.mod(p + 0.5 * box, box)
        d, _ = cKDTree(w, boxsize=box).query(w, k=2)
        return d[:, 1].min()

    base = None
    for orient in ('jr0', 'jr8'):
        p2, m2, _ = resample.split(pos, mass, aux, box=box, factor=2,
                                   dist=0.3, orient=orient, scale='vol')
        assert np.isfinite(p2).all()
        assert np.all(np.abs(p2) <= box / 2 + 1e-9)
        # pair centroid = parent (min-image: wrapping shifts a child by box)
        drift = p2[0::2] + p2[1::2] - 2 * pos
        drift -= box * np.rint(drift / box)
        assert np.allclose(drift, 0.0, atol=1e-12)
        if orient == 'jr0':
            base = dmin(p2)
        else:
            assert dmin(p2) >= base - 1e-12


def test_split_custom_halve_fields():
    pos, mass, aux, box = _setup()
    aux2 = {'Bx': aux['B'][:, 0].copy()}
    # by default 'Bx' is inherited (not in HALVE_FIELDS); override to halve it
    _, _, a_inh = resample.split(pos, mass, aux2, box=box, factor=2)
    _, _, a_hlv = resample.split(pos, mass, aux2, box=box, factor=2,
                                 halve_fields={'Bx'})
    assert np.isclose(a_inh['Bx'].sum(), 2 * aux2['Bx'].sum())  # inherited
    assert np.isclose(a_hlv['Bx'].sum(), aux2['Bx'].sum())      # halved


def test_merge_conserves_mass_and_momentum():
    pos, mass, aux, box = _setup(n=600)
    # exact momentum conservation holds for the raw pairwise merge (masses
    # summed, vel mass-weighted); mass renormalisation disabled
    pm, mm, am = resample.merge(pos, mass, aux, box=box, target_factor=2,
                                keep_mass_distribution=False)
    assert len(pm) < len(pos)                       # some pairs merged
    assert np.isclose(mm.sum(), mass.sum())         # mass conserved
    p0 = (mass[:, None] * aux['vel']).sum(0)
    p1 = (mm[:, None] * am['vel']).sum(0)
    assert np.allclose(p0, p1)                       # momentum conserved
    # extensive fields summed back
    assert np.isclose(am['u'].sum(), aux['u'].sum())


def test_merge_keeps_uniform_mass_uniform():
    # default: a uniform-mass input must come out uniform-mass (renormalised),
    # not a 1x/2x/3x mix from unpaired / re-merged particles
    pos, mass, aux, box = _setup(n=600)
    pm, mm, _ = resample.merge(pos, mass, aux, box=box, target_factor=2)
    assert np.isclose(mm.sum(), mass.sum())          # total mass conserved
    assert np.allclose(mm, mm[0])                    # single unique mass
    assert np.isclose(mm[0], mass[0] * len(pos) / len(pm))


def test_coarsen_preserves_mass_distribution_shape():
    # a two-mass input keeps its two-mass shape (scaled by the count ratio)
    pos, mass, aux, box = _setup(n=512)
    mass = mass.copy()
    mass[256:] *= 2.0                                # half m, half 2m
    cp, cm, _ = resample.coarsen(pos, mass, aux, box=box, factor=2)
    assert len(cp) == 256
    assert np.isclose(cm.sum(), mass.sum())
    uniq = np.unique(np.round(cm / cm.min(), 6))
    assert len(uniq) == 2                            # still exactly two masses
    assert np.isclose(uniq[1] / uniq[0], 2.0)        # same 1:2 ratio
    assert np.isclose(cm.min(), mass.min() * 2.0)    # each scaled by n_in/n_out


def test_open_domain_split():
    # box=None (open) should still split and conserve mass
    rng = np.random.default_rng(1)
    pos = rng.random((300, 3))
    mass = np.full(300, 1.0)
    p2, m2, _ = resample.split(pos, mass, {}, box=None, factor=2)
    assert len(p2) == 600
    assert np.isclose(m2.sum(), mass.sum())
    assert np.isfinite(p2).all()


def test_coarsen_exact_count_and_roundtrip():
    # coarsen() hits count//factor EXACTLY and is the inverse-by-count of split:
    # split(coarsen(x, F), F) restores the original count (1:1).
    pos, mass, aux, box = _setup(n=512)        # divisible by 8
    n = len(pos)
    for factor in (2, 4, 8):
        cp, cm, ca = resample.coarsen(pos, mass, aux, box=box, factor=factor)
        assert len(cp) == n // factor                  # EXACT target count
        assert np.isclose(cm.sum(), mass.sum())        # mass conserved
        assert np.isclose(ca['u'].sum(), aux['u'].sum())   # extensive aux summed
        sp, sm, _ = resample.split(cp, cm, ca, box=box, factor=factor)
        assert len(sp) == n                            # round-trips to input N
        assert np.isclose(sm.sum(), mass.sum())


def test_coarsen_rejects_non_power_of_two():
    pos, mass, aux, box = _setup()
    with pytest.raises(ValueError):
        resample.coarsen(pos, mass, aux, box=box, factor=3)
