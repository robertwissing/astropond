#!/usr/bin/env python3
"""Equation-level checks for the isolated dissipation tests (`disstest`).

The focus is the logic that is NEW in the split from `cosmowave`: the
STATIC/EXPANDING arm pairing.  A static arm has one clock, so its two fits must
coincide and announce their own tautology; an expanding arm's two clocks must
be far apart, which is the discriminating power the a-factor test depends on.

Pure/read-only: no temporary run directories, no plotting, no Telegram.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.dirname(HERE)
sys.path.insert(0, SETUP)

from IC_analysis_disstest import (  # noqa: E402
    CASES, case_params, snapshot_axis, null_laws, null_residuals,
    null_reference_table, _cosmic_time, _clock_a, _fill_clocks,
    clock_column, decay_fit, decay_residuals,
    turb_reference, ladder_exponent, _hubble,
    N_A, N_T, N_RHO, N_U, N_B, N_RHO_SPR, N_U_SPR, N_B_SPR,
    N_VMAX, N_AV, N_DIFF, N_PDV, N_DIVB, N_COLS,
    T_A, T_TIME, T_CLOCKA, T_AMP, T_COLS,
)


class TestCaseParams(unittest.TestCase):
    """The guards duplicate IC_setup_disstest's on purpose: an analysis run
    with the wrong -a/-b would otherwise fit a model the run never used."""

    def test_arms_differ_only_in_the_scale_factor(self):
        expanding = case_params(1, cosmo=1)
        static = case_params(1, cosmo=0)
        self.assertAlmostEqual(expanding['a_i'], 1./128., places=13)
        self.assertAlmostEqual(static['a_i'], 1.0, places=13)
        for key in ('gamma', 'V_S', 'uuzero', 'bzero', 'k', 'x0'):
            self.assertAlmostEqual(expanding[key], static[key], places=13,
                                   msg=key)

    def test_static_arm_has_no_hubble_rate_at_all(self):
        static = case_params(0, cosmo=0)
        np.testing.assert_array_equal(_hubble(np.array([1.0, 0.5]), static),
                                      np.zeros(2))
        self.assertGreater(float(_hubble(np.array([0.5]),
                                         case_params(0, cosmo=1))[0]), 0.)

    def test_a_init_overrides_a_i_on_the_expanding_arm_only(self):
        self.assertAlmostEqual(case_params(1, cosmo=1, a_init=1./32.)['a_i'],
                               1./32., places=13)
        with self.assertRaises(ValueError):
            case_params(1, cosmo=0, a_init=1./32.)
        with self.assertRaises(ValueError):
            case_params(1, cosmo=1, a_init=2.0)

    def test_the_case_and_variant_guards(self):
        for bad in (-1, 4, 7):
            with self.assertRaises(ValueError):
                case_params(bad)
        with self.assertRaises(ValueError):
            case_params(2, bfac=0.)          # bhelix is a magnetic mode
        with self.assertRaises(ValueError):
            case_params(1, variant=1)        # variant belongs to case 3 only
        case_params(3, variant=1)            # ... and is accepted there

    def test_vamp_defaults_to_amp_but_is_separable(self):
        self.assertAlmostEqual(case_params(3, amp=0.2)['vamp'], 0.2, places=13)
        self.assertAlmostEqual(case_params(3, amp=0.2, vamp=0.05)['vamp'],
                               0.05, places=13)

    def test_every_registered_case_builds(self):
        for case in CASES:
            for cosmo in (0, 1):
                case_params(case, cosmo=cosmo)


class TestSnapshotAxis(unittest.TestCase):
    """A comoving run stores the scale factor in the tipsy time field; a static
    one stores cosmic time.  Reading it wrong silently rescales every clock."""

    def test_comoving_time_field_is_the_scale_factor(self):
        p = case_params(1, cosmo=1)
        a, t = snapshot_axis(np.array([0.25]), p)
        self.assertAlmostEqual(a, 0.25, places=13)
        self.assertAlmostEqual(t, float(_cosmic_time(np.array([0.25]), p)[0]),
                               places=13)

    def test_static_time_field_is_cosmic_time_and_a_is_pinned_to_one(self):
        p = case_params(1, cosmo=0)
        a, t = snapshot_axis(np.array([3.75]), p)
        self.assertAlmostEqual(a, 1.0, places=13)
        self.assertAlmostEqual(t, 3.75, places=13)

    def test_a_nonpositive_comoving_time_falls_back_to_a_i(self):
        p = case_params(1, cosmo=1)
        self.assertAlmostEqual(snapshot_axis(np.array([0.0]), p)[0], p['a_i'],
                               places=13)


class TestNullLaws(unittest.TestCase):

    def _series(self, p, n=40, **override):
        """An exactly-on-law null table, before any deliberate corruption."""
        if p['cosmo'] == 1:
            a = p['a_i']*np.exp(np.linspace(0., np.log(1./p['a_i']), n))
            t = _cosmic_time(a, p)
        else:
            a = np.ones(n)
            t = np.linspace(0., 1., n)
        law = null_laws(a, p)
        d = np.zeros((n, N_COLS), dtype=float)
        d[:, N_A], d[:, N_T] = a, t
        d[:, N_RHO] = 1.0*law['rho']
        d[:, N_U] = 3.5*law['u']
        d[:, N_B] = 2.0*law['B']
        d[:, N_PDV] = -1.0 if p['cosmo'] == 1 else 0.0
        for column, value in override.items():
            d[:, globals()[column]] = value
        return d

    def test_expanding_laws_are_the_analytic_exponents(self):
        p = case_params(0, cosmo=1)
        a = np.asarray([p['a_i'], 4.*p['a_i'], 1.])
        law, r = null_laws(a, p), a/p['a_i']
        np.testing.assert_allclose(law['rho'], np.ones_like(r), rtol=0, atol=0)
        np.testing.assert_allclose(law['u'], r**-2., rtol=1e-13)  # gamma=5/3
        np.testing.assert_allclose(law['B'], r**-0.5, rtol=1e-13)

    def test_static_laws_are_all_flat(self):
        """With H = 0 every source is switched off, so the null becomes
        'nothing moves at all' -- which is what makes it the operator
        reference."""
        p = case_params(0, cosmo=0)
        law = null_laws(np.asarray([1.0, 1.0, 1.0]), p)
        for key in ('rho', 'u', 'B'):
            np.testing.assert_allclose(law[key], np.ones(3), rtol=0, atol=0)

    def test_an_exactly_on_law_series_has_zero_residuals_in_both_arms(self):
        for cosmo, pdv_key in ((1, 'rate:uDotPdV+1'), (0, 'rate:uDotPdV')):
            p = case_params(0, cosmo=cosmo)
            out = null_residuals(self._series(p), p)
            for key in ('law:rho', 'law:u', 'law:B', pdv_key):
                self.assertLess(out[key], 1e-13, f"{key} (cosmo={cosmo})")

    def test_the_static_arm_requires_a_vanishing_pdv_not_a_unit_one(self):
        """There is no expansion source in the static arm, so the rate itself
        must vanish; gating on `pdv + 1` there would demand a source that
        cannot exist."""
        p = case_params(0, cosmo=0)
        out = null_residuals(self._series(p), p)
        self.assertIn('rate:uDotPdV', out)
        self.assertNotIn('rate:uDotPdV+1', out)

    def test_a_wrong_power_of_a_is_caught(self):
        """The failure mode the whole suite exists for: one power of a."""
        p = case_params(0, cosmo=1)
        d = self._series(p)
        d[:, N_U] = d[:, N_U]*(d[:, N_A]/p['a_i'])   # a^-1 instead of a^-2
        out = null_residuals(d, p)
        self.assertGreater(out['law:u'], 1.0)
        self.assertLess(out['law:B'], 1e-13)

    def test_invariants_are_gauged_to_the_first_snapshot(self):
        """A run whose IC normalization differs from the analytic value must
        still pass: only the expansion LAW is under test."""
        p = case_params(0, cosmo=1)
        d = self._series(p)
        d[:, N_U] *= 137.0
        d[:, N_B] *= 0.011
        out = null_residuals(d, p)
        self.assertLess(out['law:u'], 1e-13)
        self.assertLess(out['law:B'], 1e-13)

    def test_uniformity_and_divB_are_gated_on_drift_not_magnitude(self):
        """A relaxed glass carries per-cent SPH density scatter and a finite
        h|divB|/|B| for a uniform field; a lattice carries neither.  A constant
        offset must not fail, a growing one must."""
        p = case_params(0, cosmo=1)
        constant = null_residuals(
            self._series(p, N_RHO_SPR=0.041, N_DIVB=3.9e-5), p)
        self.assertLess(constant['drift:spread:rho'], 1e-13)
        self.assertLess(constant['drift:divB'], 1e-13)
        d = self._series(p)
        d[:, N_RHO_SPR] = np.linspace(0.041, 0.061, d.shape[0])
        self.assertAlmostEqual(
            null_residuals(d, p)['drift:spread:rho'], 0.020, places=12)

    def test_any_dissipation_at_all_is_a_failure(self):
        p = case_params(0, cosmo=1)
        out = null_residuals(self._series(p, N_AV=1e-3, N_DIFF=2e-4), p)
        self.assertAlmostEqual(out['rate:uDotAV'], 1e-3, places=13)
        self.assertAlmostEqual(out['rate:uDotDiff'], 2e-4, places=13)

    def test_reference_table_covers_every_gated_series(self):
        p = case_params(0, cosmo=1)
        table = null_reference_table(self._series(p))
        for key in ('x', 'rho', 'u', 'B', 'rho_spread', 'u_spread', 'mach',
                    'uDotAV', 'uDotDiff'):
            self.assertIn(key, table)
            self.assertEqual(table[key].shape, table['x'].shape)


class TestDecayClocks(unittest.TestCase):
    """Cases 1 and 2 share one table layout and one pair of clocks."""

    def _table(self, p, rate, channel, n=60):
        """A mode decaying EXACTLY exponentially in `channel`'s clock."""
        if p['cosmo'] == 1:
            a = p['a_i']*np.exp(np.linspace(0., np.log(1./p['a_i']), n))
            t = _cosmic_time(a, p)
        else:
            a, t = np.ones(n), np.linspace(0., 2., n)
        d = np.zeros((n, T_COLS), dtype=float)
        d[:, T_A], d[:, T_TIME] = a, t
        d = _fill_clocks(d, p)
        clock = d[:, clock_column(channel)]
        d[:, T_AMP] = 0.75*np.exp(-rate*(clock - clock[0]))
        return d

    def test_the_two_clocks_are_distinct_when_expanding(self):
        p = case_params(1, cosmo=1)
        a = np.asarray([p['a_i'], 0.25, 1.0])
        # integral a dt grows far more slowly than t early on, because a is
        # small exactly where most of the cosmic time is spent.
        self.assertLess(float(_clock_a(a, p)[-1]),
                        float(_cosmic_time(a, p)[-1]))

    def test_the_static_arm_has_exactly_one_clock(self):
        p = case_params(1, cosmo=0)
        d = self._table(p, 1.3, 'physical')
        np.testing.assert_allclose(d[:, T_CLOCKA], d[:, T_TIME],
                                   rtol=0, atol=0)

    def test_decay_fit_recovers_a_planted_rate(self):
        for cosmo in (0, 1):
            p = case_params(1, cosmo=cosmo)
            lam, residual = decay_fit(self._table(p, 2.5, 'physical'), p,
                                      'physical')
            self.assertAlmostEqual(lam, 2.5, places=8)
            self.assertLess(residual, 1e-10)

    def test_static_arm_sensitivity_is_exactly_one(self):
        """The intended self-announcement: with one clock the two fits
        coincide, so the arm reports that it has no a-dependence to
        discriminate rather than passing as a result."""
        p = case_params(1, cosmo=0)
        out = decay_residuals(self._table(p, 2.5, 'physical'), p, 'physical')
        self.assertAlmostEqual(out['sensitivity'], 1.0, places=9)

    def test_expanding_arm_sensitivity_is_large(self):
        """The same table on the expanding arm must strongly prefer its own
        clock -- that gap is the a-factor test's entire discriminating power."""
        p = case_params(1, cosmo=1)
        out = decay_residuals(self._table(p, 2.5, 'physical'), p, 'physical')
        self.assertLess(out['residual'], 1e-10)
        self.assertGreater(out['null:wrong-a-power'], 1e-3)
        self.assertGreater(out['sensitivity'], 1e3)

    def test_clock_column_rejects_an_unknown_channel(self):
        with self.assertRaises(ValueError):
            clock_column('sideways')


class TestLadder(unittest.TestCase):

    def test_exponent_recovers_a_planted_power_law(self):
        a = np.geomspace(1./128., 1., 8)
        for power in (-1., 0., 1., 2.):
            slope, residual = ladder_exponent(a, 3.7*a**power)
            self.assertAlmostEqual(slope, power, places=9)
            self.assertLess(residual, 1e-10)

    def test_exponent_needs_at_least_three_rungs(self):
        slope, _res = ladder_exponent(np.array([0.5, 1.0]),
                                      np.array([1.0, 2.0]))
        self.assertTrue(np.isnan(slope))

    def test_turbulent_reference_is_the_shear_rate_times_h_squared(self):
        """h_com^2 |S_phys| carries no scale factor for a fixed comoving
        template: h is comoving, and the two powers of a in
        |d(a xdot)/d(a x_com)| cancel."""
        p = case_params(3, variant=1, amp=0.1)
        x = np.array([p['x0'] + 0.25])       # phase = pi/2, |sin| = 1
        h = np.array([0.05])
        reference, note = turb_reference(x, h, p)
        self.assertAlmostEqual(float(reference[0]),
                               0.05**2*p['k']*p['vamp']*p['V_S'], places=13)
        self.assertIn('transverse', note)

    def test_longitudinal_reference_is_reported_but_not_asserted(self):
        """div v != 0 there, so |S| depends on the trace-free convention."""
        _reference, note = turb_reference(np.array([0.]), np.array([1.]),
                                          case_params(3, variant=0))
        self.assertIn('not asserted', note)


if __name__ == '__main__':
    unittest.main(verbosity=2)
