import numpy as np
import pytest

from glassgen.flavours import FLAVOUR_SPECS
from mhd_divBclean import clean_divergence


def _jittered(n_side=8, seed=0, box=1.0):
    rng = np.random.default_rng(seed)
    g = (np.arange(n_side) + 0.5) / n_side * box - 0.5 * box
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    pos = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    pos += rng.uniform(-0.2, 0.2, pos.shape) * (box / n_side)
    n = len(pos)
    return pos, np.full(n, 1.0 / n)


@pytest.mark.parametrize('flavour', sorted(FLAVOUR_SPECS))
def test_constant_field_is_exact_fixed_point_periodic(flavour):
    # The difference divergence annihilates a constant field exactly in
    # EVERY flavour/kernel-mode, so psi never sources and B must come back
    # bit-identical.
    pos, mass = _jittered()
    B = np.tile([1.0, -2.0, 0.5], (len(pos), 1))
    Bc, audit = clean_divergence(pos, mass, B, box=np.ones(3), nsmooth=32,
                                 steps=50, every=10, flavour=flavour)
    np.testing.assert_array_equal(Bc, B)
    assert audit['eps_after'] == (0.0, 0.0, 0.0)


def test_constant_field_unchanged_open_domain():
    # One-sided kernels at an open boundary do not break the constant-field
    # fixed point (the operator is the difference form).
    rng = np.random.default_rng(3)
    pos = rng.uniform(-0.5, 0.5, (600, 3))
    mass = np.full(len(pos), 1.0 / len(pos))
    B = np.tile([0.3, 0.0, -1.1], (len(pos), 1))
    Bc, _ = clean_divergence(pos, mass, B, box=None, nsmooth=32, steps=30,
                             every=10)
    np.testing.assert_array_equal(Bc, B)


def test_gradient_noise_is_cleaned():
    # A pure gradient field is 100% divergence carrier: cleaning must remove
    # most of it (toward the mean) and monotonically dissipate total energy.
    pos, mass = _jittered(n_side=10, seed=1)
    k = 2.0 * np.pi
    phi_grad = np.stack([
        k * np.cos(k * pos[:, 0]) * np.sin(k * pos[:, 1]),
        k * np.sin(k * pos[:, 0]) * np.cos(k * pos[:, 1]),
        np.zeros(len(pos))], axis=1)
    B = np.tile([1.0, 0.0, 0.0], (len(pos), 1)) + 0.5 * phi_grad
    Bc, audit = clean_divergence(pos, mass, B, box=np.ones(3), nsmooth=48,
                                 steps=400, every=50, sigma=1.0, dt=0.15)
    m0 = audit['eps_before'][0]
    m1 = audit['eps_after'][0]
    assert m1 < m0 / 3.0
    total = audit['trace']['e_b'] + audit['trace']['e_psi']
    assert np.all(np.diff(total) <= 1e-10 * total[0])
    # the uniform component survives the cleaning
    assert abs(np.mean(Bc[:, 0]) - np.mean(B[:, 0])) < 0.05


@pytest.mark.parametrize('flavour', sorted(FLAVOUR_SPECS))
def test_zero_damping_conserves_exchange_energy(flavour):
    # sigma=0 removes the only dissipation channel, so the staggered leapfrog
    # must conserve E_B + E_psi to integrator accuracy - THE regression that
    # each flavour's difference-div / symmetric-grad pair is energy-conjugate
    # (a wrong conjugate D or window pairing shows up as secular drift).
    pos, mass = _jittered(n_side=8, seed=2)
    k = 2.0 * np.pi
    B = np.stack([np.sin(k * pos[:, 1]),
                  np.zeros(len(pos)),
                  0.3 * np.cos(k * pos[:, 0])], axis=1)
    B[:, 0] += 0.4 * np.sin(k * pos[:, 0])   # some divergence to exchange
    _, audit = clean_divergence(pos, mass, B, box=np.ones(3), nsmooth=32,
                                steps=200, every=10, sigma=0.0, dt=0.05,
                                flavour=flavour)
    total = audit['trace']['e_b'] + audit['trace']['e_psi']
    assert np.max(np.abs(total - total[0])) < 0.05 * total[0]
