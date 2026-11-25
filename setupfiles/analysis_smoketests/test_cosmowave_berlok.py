#!/usr/bin/env python3
"""Equation-level checks for the Berlok cosmowave setup and analysis.

These tests are pure/read-only: they create no temporary run directories and
exercise no Telegram or plotting paths.
"""

import os
import sys
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.dirname(HERE)
sys.path.insert(0, SETUP)

from IC_analysis_framework import (  # noqa: E402
    Param, build_parser, params_from_args,
    _case_from_input_name, _infer_case_from_inputs, _resolve_case,
)
from IC_analysis_cosmowave import (  # noqa: E402
    _berlok_figure1_components, _complex_mode,
    _cumulative_integral, _endpoint_convergence_data,
    _endpoint_phase_bin_count,
    _figure_title, _gamma_label, _physical_case_name,
    _fit_standing_alfven_eq77, _oscillator_phase_advance,
    _phase_bin_l1_error, _power_law_order, _real_mode, case_params,
    cosmowave_energy_budget,
    read_timestep_convergence_metadata,
    _spatial_mode_amplitude,
    divclean_theory_coefficients, theory_coefficients,
    N_A, N_RHO, N_U, N_B, N_RHO_SPR, N_U_SPR, N_B_SPR,
    N_VMAX, N_AV, N_DIFF, N_PDV, N_DIVB, N_COLS,
    null_laws, null_residuals, null_reference_table,
    T_A, T_AMP, T_COLS, thermal_clock, thermal_fit, thermal_residuals,
    _cosmic_time,
)
from IC_setup_cosmowave import eds_time, flat_cosmic_time  # noqa: E402
from IC_fixdens import get_mass  # noqa: E402
from IC_writeascii import convertfromcodeunits  # noqa: E402


class TestCaseInference(unittest.TestCase):
    def test_standard_run_names_supply_case_number(self):
        self.assertEqual(
            _case_from_input_name(
                "/runs/cosmowave32_N64glasscase0A1G0u0.1GASOLINE"),
            0)
        self.assertEqual(_case_from_input_name("cosmowave64_case10trial"), 10)
        self.assertIsNone(_case_from_input_name("cosmowave_without_tag"))

    def test_matching_tags_are_inferred_without_prompt(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            case = _infer_case_from_inputs(
                ["/runs/cosmowave32_case0A", "/runs/cosmowave64_case0B"],
                default_case=3,
                prompt=lambda: self.fail("matching tags must not prompt"))
        self.assertEqual(case, 0)
        self.assertIn("inferred case 0", stderr.getvalue())

    def test_conflicting_tags_warn_and_require_confirmation(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            case = _infer_case_from_inputs(
                ["/runs/cosmowave32_case0A", "/runs/cosmowave64_case3B"],
                default_case=3,
                prompt=lambda: "yes")
        warning = stderr.getvalue()
        self.assertEqual(case, 0)
        self.assertIn("encode different cases", warning)
        self.assertIn("case 0: /runs/cosmowave32_case0A", warning)
        self.assertIn("case 3: /runs/cosmowave64_case3B", warning)
        self.assertIn("[y/N]", warning)

    def test_conflicting_tags_decline_stops_analysis(self):
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaisesRegex(SystemExit, "2"):
                _infer_case_from_inputs(
                    ["cosmowave_case1", "cosmowave_case2"],
                    default_case=3,
                    prompt=lambda: "n")

    def test_explicit_case_wins_without_inspecting_conflicting_names(self):
        self.assertEqual(
            _resolve_case(4, ["cosmowave_case0", "cosmowave_case3"],
                          default_case=3,
                          prompt=lambda: self.fail("explicit case must not prompt")),
            4)

    def test_missing_tag_retains_default_with_warning(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            case = _infer_case_from_inputs(["renamed_run"], default_case=3,
                                           prompt=lambda: "")
        self.assertEqual(case, 3)
        self.assertIn("historical default case 3", stderr.getvalue())

    def test_multiple_tags_in_one_name_are_rejected_as_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "contains case0, case3"):
            _case_from_input_name("comparison_case0_against_case3")

    def test_shared_parser_infers_only_an_actual_case_parameter(self):
        case_spec = [Param("a", "case", 3, int, "mode selector")]
        parser = build_parser(case_spec)
        with patch("sys.stderr", new_callable=StringIO):
            args = parser.parse_args(["cosmowave_case0"])
            params = params_from_args(args, case_spec)
        self.assertEqual(params["case"], 0)
        self.assertEqual(args.case, 0)

        beta_spec = [Param("a", "beta", 0.1, float, "plasma beta")]
        parser = build_parser(beta_spec)
        args = parser.parse_args(["unrelated_case7_directory"])
        params = params_from_args(args, beta_spec)
        self.assertEqual(params["beta"], 0.1)

    def test_shared_parser_preserves_an_explicit_case(self):
        case_spec = [Param("a", "case", 3, int, "mode selector")]
        parser = build_parser(case_spec)
        args = parser.parse_args(
            ["cosmowave_case0", "cosmowave_case3", "-a", "4"])
        self.assertEqual(params_from_args(args, case_spec)["case"], 4)

    def test_test_owned_tags_infer_multiple_omitted_parameters(self):
        spec = [
            Param("a", "wavetype", 0, int, "wave selector", "wt"),
            Param("b", "ampl", 1.e-4, float, "amplitude", "a"),
        ]
        parser = build_parser(spec)
        with patch("sys.stderr", new_callable=StringIO):
            args = parser.parse_args(["wave64_N32glasswt3a2.5e-4GASOLINE"])
            params = params_from_args(args, spec)
        self.assertEqual(params["wavetype"], 3)
        self.assertAlmostEqual(params["ampl"], 2.5e-4)

    def test_name_map_supports_shocktube_style_presets(self):
        spec = [Param("a", "choice", 1, int, "shock preset",
                      {"Sod_shock": 1, "Brio-Wu_Ryu_5a": 6})]
        parser = build_parser(spec)
        with patch("sys.stderr", new_callable=StringIO):
            args = parser.parse_args(
                ["shocktube64_N32glassBrio-Wu_Ryu_5aGASOLINE"])
            params = params_from_args(args, spec)
        self.assertEqual(params["choice"], 6)

    def test_mixed_runname_values_prompt_once_and_use_first_input(self):
        spec = [
            Param("a", "case", 3, int, "case", "case"),
            Param("b", "factor_A", 1., float, "field factor", "A"),
        ]
        parser = build_parser(spec)
        args = parser.parse_args(
            ["cosmowave_case0A1", "cosmowave_case3A2"])
        with patch("builtins.input", return_value="yes") as ask:
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                params = params_from_args(args, spec)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(params, {"case": 0, "factor_A": 1.})
        warning = stderr.getvalue()
        self.assertIn("cosmowave_case0A1", warning)
        self.assertIn("cosmowave_case3A2", warning)
        self.assertIn("-a/--case=0", warning)
        self.assertIn("-b/--factor_A=2.0", warning)

    def test_explicit_flag_supersedes_only_its_runname_value(self):
        spec = [
            Param("a", "wavetype", 0, int, "wave selector", "wt"),
            Param("b", "ampl", 1.e-4, float, "amplitude", "a"),
        ]
        parser = build_parser(spec)
        args = parser.parse_args(
            ["wave64_N32glasswt3a2.5e-4GASOLINE", "-a", "1"])
        with patch("sys.stderr", new_callable=StringIO):
            params = params_from_args(args, spec)
        self.assertEqual(params["wavetype"], 1)
        self.assertAlmostEqual(params["ampl"], 2.5e-4)


class TestFigureHeadings(unittest.TestCase):
    def test_every_case_has_a_descriptive_gamma_heading(self):
        expected = {
            0: ("Standing Alfvén wave", "4/3"),
            1: ("Traveling Alfvén wave", "4/3"),
            2: ("Standing magnetosonic wave", "4/3"),
            3: ("Traveling magnetosonic wave", "4/3"),
            4: ("Divergence-cleaning wave", "4/3"),
            5: (r"Standing compressible wave ($\Lambda$CDM)", "5/3"),
            6: ("Standing isothermal compressible wave", "1"),
            7: ("Standing adiabatic compressible wave", "5/3"),
            8: ("Homogeneous expanding MHD state", "5/3"),
            9: ("Sinusoidal thermal-diffusion mode", "5/3"),
            10: ("Force-free helical magnetic-diffusion mode", "5/3"),
        }
        for case, (description, gamma) in expected.items():
            with self.subTest(case=case):
                p = case_params(case)
                self.assertEqual(_physical_case_name(p), description)
                self.assertEqual(_gamma_label(p), rf'$\gamma = {gamma}$')
                heading = _figure_title(p, description)
                self.assertEqual(
                    heading, f"{description} ($\\gamma = {gamma}$)")
                self.assertNotIn("CosmoWave Case", heading)


class TestBerlokCoefficients(unittest.TestCase):
    def test_shared_endpoint_phase_bin_count_is_resolution_bounded(self):
        for resolution, expected in ((4, 8), (16, 16), (256, 128),
                                     (None, 64)):
            with self.subTest(resolution=resolution):
                with patch(
                        "IC_analysis_cosmowave._cosmowave_resolution",
                        return_value=resolution):
                    self.assertEqual(
                        _endpoint_phase_bin_count("unused"), expected)

    def test_phase_bin_l1_averages_noise_before_taking_norm(self):
        centers = np.asarray([-0.375, -0.125, 0.125, 0.375])
        x = np.repeat(centers, 2)
        coherent = np.repeat([0.04, -0.03, 0.02, -0.01], 2)
        particle_noise = np.tile([1.0, -1.0], 4)
        measured = coherent+particle_noise
        expected = np.zeros_like(measured)
        weights = np.ones_like(measured)

        direct_l1 = np.average(np.abs(measured-expected), weights=weights)
        phase_l1 = _phase_bin_l1_error(
            x, measured, expected, weights, nbins=4)

        self.assertGreater(direct_l1, 0.9)
        self.assertAlmostEqual(phase_l1, 0.025, places=14)

    def test_convergence_orders_use_the_expected_axis_direction(self):
        resolution = np.asarray([8., 16., 32., 64.])
        spatial_error = 0.3*resolution**-2
        eta_c = np.asarray([0.4, 0.2, 0.1, 0.05])
        timestep_error = 0.7*eta_c**2
        self.assertAlmostEqual(
            _power_law_order(resolution, spatial_error, inverse=True),
            2.0, places=13)
        self.assertAlmostEqual(
            _power_law_order(eta_c, timestep_error), 2.0, places=13)

    def test_figure6_joint_fit_recovers_frequency_and_decay(self):
        p = case_params(0, factor_A=2., factor_G=0., A_u=0.1)
        s = np.linspace(0., np.log(1./p['a_i']), 180)
        a = p['a_i']*np.exp(s)
        frequency = 1.018*p['kappa_A']
        decay = 0.037
        envelope = np.exp(-decay*s)
        magnetic = envelope*(-0.13*np.sin(frequency*s)
                             + 0.02*np.cos(frequency*s))
        velocity = envelope*(-0.06*np.sin(frequency*s)
                             + 0.98*np.cos(frequency*s))
        data = np.zeros((len(a), 34), dtype=float)
        data[:, 0] = a
        # _fit_standing_alfven_eq77 reconstructs these two scaled amplitudes
        # from the dimensional complex Fourier columns.
        data[:, 13] = (-magnetic*p['A_u']*p['Bc0']
                       /(a/p['a_i'])**0.25)
        data[:, 10] = (velocity*p['A_u']*p['V_A']
                       /(a/p['a_i'])**0.75)
        fitted = _fit_standing_alfven_eq77(data, p)
        self.assertAlmostEqual(fitted['frequency'], frequency, places=9)
        self.assertAlmostEqual(fitted['decay'], decay, places=9)

    def test_endpoint_convergence_metric_matches_exact_case3_profile(self):
        p = case_params(3, factor_A=1., factor_G=0., A_u=0.1)
        a = 1.0
        x = np.linspace(-0.5, 0.5, 256, endpoint=False)
        theory = theory_coefficients(np.asarray([a]), p)
        rho_mode = _real_mode(x, theory['rho'][0], p)
        u_mode = _real_mode(x, theory['u'][0], p)
        b_mode = _real_mode(x, theory['B'][0], p)
        profile = {
            'a': a, 'x': x, 'volume': np.ones_like(x),
            'rho': p['rhozero']*(1.+rho_mode),
            'u': p['V_S']*u_mode,
            'p': 1.+p['gamma']*rho_mode,
            'p_background': 1.,
            'B': p['Bc0']*(1.+b_mode),
        }
        data = np.zeros((1, 34), dtype=float)
        data[0, 0] = a
        with patch('IC_analysis_cosmowave._endpoint_profile',
                   return_value=profile):
            result = _endpoint_convergence_data('unused', data, p)
        self.assertEqual(set(result['quantities']), {'rho', 'u', 'p', 'B'})
        for quantity in result['quantities'].values():
            self.assertLess(quantity['max'], 2.e-13)

        offset_profile = dict(profile)
        offset_profile['u'] = profile['u']+0.002*p['V_S']
        with patch('IC_analysis_cosmowave._endpoint_profile',
                   return_value=offset_profile):
            normalized = _endpoint_convergence_data('unused', data, p)
            paper_scaled = _endpoint_convergence_data(
                'unused', data, p, normalize_amplitude=False)
        self.assertAlmostEqual(normalized['quantities']['u']['max'],
                               paper_scaled['quantities']['u']['max']/p['A_u'],
                               places=13)

    def test_cubic_source_quadrature_is_exact_for_a_cubic(self):
        coordinate = np.linspace(0.2, 1.0, 9)
        values = coordinate**3-2.*coordinate+4.
        got = _cumulative_integral(values, coordinate, quadrature='cubic')
        primitive = coordinate**4/4.-coordinate**2+4.*coordinate
        np.testing.assert_allclose(got, primitive-primitive[0],
                                   rtol=2.e-14, atol=2.e-14)

    def test_timestep_budget_recovers_scale_factor_from_legacy_total(self):
        p = case_params(4, factor_A=1., factor_G=0., A_u=0.001)
        a = np.geomspace(p['a_i'], 1.0, 41)
        # Match the historical %.4e log formatting that motivated the
        # recovery.  Its direct redshift conversion is inaccurate at ~3e-5.
        z_rounded = np.asarray([float(f"{value:.4e}")
                                for value in 1./a-1.])
        zeros = np.zeros_like(a)
        log = {
            'dTime': np.arange(a.size, dtype=float), 'z': z_rounded,
            'Ekin': zeros, 'Epot': zeros,
            'Eth': 2.0/a, 'Emag': 0.1/a, 'EClean': zeros,
        }
        reservoir = log['Eth']+log['Emag']+log['EClean']
        log['Etot'] = log['Ekin']+a*reservoir
        budget = cosmowave_energy_budget(
            log, p, quadrature='cubic', recover_legacy_a=True)

        self.assertEqual(budget['scale_factor_source'],
                         'legacy Etot identity')
        np.testing.assert_allclose(budget['a'], a, rtol=2.e-13, atol=2.e-13)
        np.testing.assert_allclose(budget['residual'], 0., atol=2.e-13)

        # A B8/work-corrected total has a different normalization and must not
        # be used to infer a; the deliberately rounded z values remain visible.
        log['Etot'] = np.full_like(a, a[0]**2*reservoir[0])
        current = cosmowave_energy_budget(
            log, p, quadrature='cubic', recover_legacy_a=True)
        self.assertEqual(current['scale_factor_source'], 'logged redshift')
        self.assertGreater(np.max(np.abs(current['a']-a)), 1.e-8)

    def test_timestep_metadata_requires_zero_artificial_viscosity(self):
        values = {
            ('run_coarse', 'dEtaCourant'): 0.1,
            ('run_coarse', 'dConstAlpha'): 0.0,
            ('run_coarse', 'dConstBeta'): 0.0,
            ('run_fine', 'dEtaCourant'): 0.025,
            ('run_fine', 'dConstAlpha'): 0.0,
            ('run_fine', 'dConstBeta'): 0.0,
        }

        def lookup(run, key, default=None):
            return values.get((run, key), default)

        with patch('IC_analysis_cosmowave.read_log_param', side_effect=lookup):
            metadata = read_timestep_convergence_metadata(
                ['run_coarse', 'run_fine'])
        self.assertEqual([item['eta_c'] for item in metadata], [0.1, 0.025])

        values[('run_fine', 'dConstAlpha')] = 1.0
        with patch('IC_analysis_cosmowave.read_log_param', side_effect=lookup):
            with self.assertRaisesRegex(ValueError, 'requires AV=0'):
                read_timestep_convergence_metadata(
                    ['run_coarse', 'run_fine'])

    def test_spatial_amplitudes_follow_complex_projection_convention(self):
        values = np.asarray([0.37 - 0.21j, -0.12 + 0.44j])
        np.testing.assert_allclose(
            _spatial_mode_amplitude(values, 'cosine'), [0.37, -0.12])
        np.testing.assert_allclose(
            _spatial_mode_amplitude(values, 'sine'), [0.21, -0.44])
        with self.assertRaises(ValueError):
            _spatial_mode_amplitude(values, 'unknown')

    def test_figure1_uses_signed_unscaled_amplitudes(self):
        p = case_params(0, factor_A=1, factor_G=0, A_u=0.1)
        a = np.geomspace(p['a_i'], 1.0, 101)
        R = a/p['a_i']
        psi = p['kappa_A']*np.log(R)
        th = theory_coefficients(a, p)
        magnetic, velocity = _berlok_figure1_components(th['B'], th['u'], p)

        expected_magnetic = (-np.sqrt(p['a_i'])*p['omega_A']/p['kappa_A']
                             * R**-0.25*np.sin(psi))
        expected_velocity = (R**-0.75
                             * (np.cos(psi)
                                - np.sin(psi)/(4*p['kappa_A'])))
        np.testing.assert_allclose(magnetic, expected_magnetic,
                                   rtol=2.e-14, atol=2.e-14)
        np.testing.assert_allclose(velocity, expected_velocity,
                                   rtol=2.e-14, atol=2.e-14)
        self.assertAlmostEqual(velocity[0], 1.0, places=14)

    def test_initial_conditions(self):
        for case in range(4):
            p = case_params(case, factor_A=1, factor_G=0, A_u=0.1)
            th = theory_coefficients(np.array([p['a_i']]), p)
            self.assertAlmostEqual(th['u'][0].real, p['A_u'], places=14)
            self.assertAlmostEqual(th['u'][0].imag, 0.0, places=14)
            if case in (0, 2):
                self.assertAlmostEqual(abs(th['B'][0]), 0.0, places=14)
            if case == 2:
                self.assertAlmostEqual(abs(th['rho'][0]), 0.0, places=14)
            if case == 3:
                expected = (p['A_u'] * p['omega_S'] * np.sqrt(p['a_i'])
                            / (4*p['sigma2']) * (4*p['kappa'] + 1j))
                np.testing.assert_allclose(th['rho'][0], expected,
                                           rtol=1.e-14, atol=1.e-14)

    def test_traveling_scaled_magnitudes_are_constant(self):
        for case in (1, 3):
            p = case_params(case, factor_A=1, factor_G=0, A_u=0.1)
            a = np.geomspace(p['a_i'], 1.0, 101)
            R = a/p['a_i']
            th = theory_coefficients(a, p)
            np.testing.assert_allclose(np.abs(th['u'])*R**0.75, p['A_u'],
                                       rtol=2.e-14, atol=2.e-14)
            np.testing.assert_allclose(np.abs(th['B'])*R**0.25,
                                       np.abs(th['B'][0]),
                                       rtol=2.e-14, atol=2.e-14)

    def test_complex_projection_keeps_phase(self):
        p = case_params(3)
        q = np.linspace(0.0, 1.0, 4096, endpoint=False)
        x = -0.5 + q + 0.03*np.sin(2*np.pi*q)
        weights = 1.0 + 0.2*np.cos(4*np.pi*q)
        expected = 0.37 - 0.21j
        field = np.real(expected*np.exp(1j*p['k']*(x-p['x0'])))
        got = _complex_mode(weights, x, field, p)
        np.testing.assert_allclose(got, expected, rtol=1.e-13, atol=1.e-13)

    def test_real_space_and_projection_share_convention(self):
        p = case_params(3)
        x = np.linspace(-0.5, 0.5, 3072, endpoint=False)
        weights = 1.0 + 0.15*np.sin(6*np.pi*x)
        expected = -0.19 + 0.43j
        field = _real_mode(x, expected, p)
        got = _complex_mode(weights, x, field, p)
        np.testing.assert_allclose(got, expected, rtol=1.e-13, atol=1.e-13)

    def test_eds_duration_lands_at_one(self):
        p = case_params(3)
        delta = (eds_time(1.0, p['H0']) - eds_time(p['a_i'], p['H0']))/100
        afinal = ((eds_time(p['a_i'], p['H0']) + 100*delta)
                  * 3*p['H0']/2)**(2./3.)
        self.assertAlmostEqual(afinal, 1.0, places=14)

    def test_cartesian_stretch_mass_includes_left_endpoint(self):
        xgrid = np.linspace(-0.5, 0.5, 20001)
        np.testing.assert_allclose(
            get_mass(0.5, -0.5, xgrid, np.ones_like(xgrid)),
            1.0, rtol=0.0, atol=2.e-14)

        amplitude = 0.013
        rho = 1.0 + amplitude*np.cos(2*np.pi*(xgrid + 0.5))
        np.testing.assert_allclose(
            get_mass(0.5, -0.5, xgrid, rho),
            1.0, rtol=0.0, atol=2.e-12)

    def test_lcdm_numeric_reference_initial_conditions(self):
        p = case_params(5, factor_A=1, factor_G=0, A_u=0.1)
        self.assertEqual(p['gamma'], 5./3.)
        self.assertAlmostEqual(p['omega_S'], np.pi/5.)
        self.assertEqual(p['omega_m'], 0.3)
        self.assertEqual(p['omega_lambda'], 0.7)
        a = np.asarray([p['a_i'], p['a_i']*(1. + 1.e-5), 1.0])
        th = theory_coefficients(a, p)
        np.testing.assert_allclose(th['u'][0], p['A_u'],
                                   rtol=1.e-12, atol=1.e-12)
        np.testing.assert_allclose(th['rho'][0], 0.0,
                                   rtol=0.0, atol=1.e-14)
        np.testing.assert_allclose(th['B'], th['rho'],
                                   rtol=1.e-14, atol=1.e-14)
        self.assertTrue(np.all(np.isfinite(th['u'])))
        self.assertTrue(np.all(np.isfinite(th['rho'])))

    def test_figure3_and_figure4_case_constants(self):
        fig3 = case_params(6, factor_A=1, factor_G=0, A_u=0.1)
        fig4 = case_params(7, factor_A=1, factor_G=0, A_u=0.1)
        self.assertEqual(fig3['gamma'], 1.)
        self.assertAlmostEqual(fig3['omega_S'], 2.*np.pi)
        self.assertEqual(fig4['gamma'], 5./3.)
        self.assertAlmostEqual(fig4['omega_S'], np.pi/5.)
        for p in (fig3, fig4):
            self.assertEqual(p['omega_m'], 1.)
            self.assertEqual(p['omega_lambda'], 0.)
            th = theory_coefficients(np.asarray([p['a_i'], 1.0]), p)
            np.testing.assert_allclose(th['u'][0], p['A_u'],
                                       rtol=2.e-12, atol=2.e-12)
            np.testing.assert_allclose(th['rho'][0], 0., atol=2.e-13)
            np.testing.assert_allclose(th['B'], th['rho'],
                                       rtol=2.e-12, atol=2.e-12)

    def test_isothermal_temperature_conversion_uses_storage_gamma(self):
        gamma_eos = 1.0000000001
        cs2 = 8.25
        internal_energy = cs2/(gamma_eos*(gamma_eos-1.))
        state = SimpleNamespace(
            gamma=1., gamma_eos=gamma_eos, molweight=1.,
            vx=np.zeros(1), vy=np.zeros(1), vz=np.zeros(1),
            u=np.asarray([internal_energy]),
            Bx=np.zeros(1), By=np.zeros(1), Bz=np.zeros(1))
        d_kpc = 1000.
        d_msol = 277511472282.1312
        converted = convertfromcodeunits(state, d_kpc, d_msol, 1, 0, 0)

        d_gas_const = (d_kpc*3.085678e21*1.38e-16
                       / (1.67e-24*6.67e-8*d_msol*1.99e33))
        expected_temperature = (internal_energy*(gamma_eos-1.)
                                / d_gas_const)
        self.assertGreater(converted.u[0], 0.)
        np.testing.assert_allclose(converted.u[0], expected_temperature,
                                   rtol=2.e-15, atol=0.)

    def test_energy_budget_closes_expansion_and_reconstructs_gravity_work(self):
        p = case_params(2, factor_A=0., factor_G=1., A_u=0.1)
        a = np.linspace(0.1, 1.0, 1001)
        kinetic0, thermal0, magnetic0, cleaning0 = 0.4, 2.0, 0.3, 0.2
        potential_raw = -0.25
        log = {
            'dTime': a.copy(),
            'z': 1./a-1.,
            'Ekin': np.full_like(a, kinetic0),
            'Epot': a*potential_raw,
            'Eth': thermal0/a,
            'Emag': magnetic0/a,
            'EClean': cleaning0/a,
        }
        base = (log['Ekin']+log['Epot']
                + a**(3.*(p['gamma']-1.))*log['Eth']
                + a*(log['Emag']+log['EClean']))
        # Emulate the current Gasoline logging call, which reports the comoving
        # base sum without retaining a cumulative dEcosmo correction.
        log['Etot'] = base.copy()
        budget = cosmowave_energy_budget(log, p)

        np.testing.assert_allclose(budget['residual'], 0., atol=3.e-14)
        expected_gravity = potential_raw*(a-a[0])
        np.testing.assert_allclose(budget['gravity_work'], expected_gravity,
                                   atol=2.e-14)
        np.testing.assert_allclose(budget['gravity_log_error'],
                                   -expected_gravity, atol=2.e-14)

    def test_energy_budget_gamma_five_thirds_has_no_thermal_hubble_work(self):
        p = case_params(7, factor_A=1., factor_G=0., A_u=0.1)
        a = np.linspace(0.1, 1.0, 1001)
        thermal0, magnetic0 = 1.7, 0.35
        log = {
            'dTime': a.copy(),
            'z': 1./a-1.,
            'Ekin': np.full_like(a, 0.2),
            'Epot': np.zeros_like(a),
            'Eth': thermal0/a**2,
            'Emag': magnetic0/a,
            'EClean': np.zeros_like(a),
        }
        log['Etot'] = (log['Ekin']
                       + a**(3.*(p['gamma']-1.))*log['Eth'] + a*log['Emag'])
        budget = cosmowave_energy_budget(log, p)

        np.testing.assert_allclose(budget['residual'], 0., atol=2.e-14)
        # gamma=5/3 is where a^(3(gamma-1)) is exactly a^2: the comoving
        # thermal term is constant with no expansion work left to remove.
        np.testing.assert_allclose(budget['thermal'], thermal0, atol=2.e-14)
        np.testing.assert_allclose(budget['magnetic'], magnetic0, atol=2.e-14)

    def test_isothermal_energy_budget_is_explicitly_not_conservative(self):
        p = case_params(6, factor_A=0., factor_G=0., A_u=0.1)
        a = np.asarray([0.1, 0.5, 1.0])
        zeros = np.zeros_like(a)
        log = {
            'dTime': a.copy(), 'z': 1./a-1., 'Etot': np.ones_like(a),
            'Ekin': zeros, 'Epot': zeros, 'Eth': np.ones_like(a),
            'Emag': zeros, 'EClean': zeros,
        }
        self.assertFalse(cosmowave_energy_budget(log, p)['conservative'])

    def test_damped_case4_energy_budget_restores_the_known_glm_sink(self):
        p = case_params(4, factor_A=1., factor_G=0., A_u=0.1,
                        clean_fch=1., clean_damping=0.05)
        h_comoving = 0.04
        a = np.geomspace(p['a_i'], 1.0, 10001)
        theory = divclean_theory_coefficients(
            a, p, h_comoving=h_comoving)
        bmode = -theory['B'].imag
        qmode = theory['Q'].real
        zeros = np.zeros_like(a)
        log = {
            'dTime': np.arange(a.size, dtype=float), 'z': 1./a-1.,
            'Ekin': zeros, 'Epot': zeros, 'Eth': zeros,
            'Emag': bmode*bmode/a, 'EClean': qmode*qmode/a,
        }
        log['Etot'] = a*(log['Emag']+log['EClean'])
        budget = cosmowave_energy_budget(
            log, p, h_comoving=h_comoving)

        self.assertGreater(budget['damping_loss'][-1], 0.)
        # Floor is the trapezoidal quadrature of the damping sink over this
        # grid, not a modelling error: the endpoint closes to ~3e-11.
        np.testing.assert_allclose(budget['residual'], 0., atol=1.e-9)

    def test_legacy_case4_cli_modes_are_absent(self):
        analyzer = os.path.join(SETUP, 'IC_analysis_cosmowave.py')
        with open(analyzer, 'r') as stream:
            text = stream.read()
        self.assertNotIn('--clean-fch-values', text)
        self.assertNotIn('--clean-clock-values', text)
        self.assertNotIn('--divclean-energy-budget', text)

    def test_divclean_hyperbolic_mode_uses_log_a_and_conserves_energy(self):
        p = case_params(4, factor_A=1, factor_G=0, A_u=0.1,
                        clean_fch=1., clean_damping=0.)
        omega = p['omega_S']
        s = np.asarray([0., np.pi/(2.*omega), np.pi/omega, 2.*np.pi/omega])
        a = p['a_i']*np.exp(s)
        th = divclean_theory_coefficients(a, p)
        np.testing.assert_allclose(-th['B'].imag,
                                   p['A_u']*np.cos(omega*s), atol=2.e-14)
        np.testing.assert_allclose(th['Q'].real,
                                   -p['A_u']*np.sin(omega*s), atol=2.e-14)
        energy = np.abs(th['B'])**2+np.abs(th['Q'])**2
        np.testing.assert_allclose(energy, p['A_u']**2,
                                   rtol=2.e-14, atol=2.e-14)
        # Flux freezing leaves the comoving amplitude constant; in particular,
        # it is not multiplied by the obsolete (a/a_i)^2 envelope.
        self.assertAlmostEqual(abs(th['B'][-1]), p['A_u'], places=14)

    def test_divclean_damped_mode_loses_energy_monotonically(self):
        p = case_params(4, factor_A=1, factor_G=0, A_u=0.1,
                        clean_fch=1., clean_damping=1.)
        a = np.geomspace(p['a_i'], 1., 1001)
        th = divclean_theory_coefficients(a, p, h_comoving=0.05)
        energy = np.abs(th['B'])**2+np.abs(th['Q'])**2
        self.assertTrue(np.all(np.diff(energy) <= 2.e-15))
        self.assertLess(energy[-1], energy[0])
        self.assertGreater(th['decay'], 0.)

    def test_divclean_damped_phase_uses_full_oscillator(self):
        p = case_params(4, factor_A=1, factor_G=0, A_u=0.1,
                        clean_fch=1., clean_damping=0.05)
        a = np.geomspace(p['a_i'], 1., 1001)
        th = divclean_theory_coefficients(a, p, h_comoving=0.0387659)
        oscillator = -th['B'].imag+1j*th['Q'].real
        phase = _oscillator_phase_advance(oscillator)
        uniform_phase = th['omega']*np.log(a/p['a_i'])

        self.assertAlmostEqual(phase[0], 0., places=14)
        self.assertGreater(np.max(np.abs(phase-uniform_phase)), 1.e-3)
        np.testing.assert_allclose(
            _oscillator_phase_advance(oscillator), phase,
            rtol=0., atol=0.)

    def test_case4_uses_sound_speed_scale_without_a_background_field(self):
        p = case_params(4, factor_A=0.)
        self.assertEqual(p['omega_A'], 0.)
        self.assertAlmostEqual(p['Bc0'], p['V_S']*np.sqrt(p['rhozero']))
        th = divclean_theory_coefficients(
            np.asarray([p['a_i'], p['a_i']*np.e]), p)
        self.assertAlmostEqual(th['omega'], p['omega_S'])

    def test_runner_disables_hidden_preclean(self):
        runner = os.path.join(os.path.dirname(SETUP), 'runtest.sh')
        with open(runner, 'r') as stream:
            text = stream.read()
        self.assertIn('-psidecfac $clean_damping -bDivBClean 0', text)
        self.assertIn('datafiles/$ICfile.red', text)

    def test_lcdm_duration_is_finite_and_longer_than_eds(self):
        ai = 1./128.
        H0 = case_params(5)['H0']
        lcdm_duration = (flat_cosmic_time(1., H0, 0.3, 0.7)
                         - flat_cosmic_time(ai, H0, 0.3, 0.7))
        eds_duration = eds_time(1., H0) - eds_time(ai, H0)
        self.assertGreater(lcdm_duration, eds_duration)
        self.assertAlmostEqual(
            flat_cosmic_time(1., H0, 0.3, 0.7),
            2./(3.*H0*np.sqrt(0.7))*np.arcsinh(np.sqrt(0.7/0.3)),
            places=14)

    def test_nonoscillatory_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            case_params(0, factor_A=0.01)
        with self.assertRaises(ValueError):
            case_params(3, factor_A=0, factor_G=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestCase8NullTest(unittest.TestCase):
    """Case 8 (dissipation suite): the homogeneous null / invariants test.

    These check the reference laws and the residual reduction, not Gasoline.
    """

    def _series(self, p, n=40, **override):
        """An exactly-on-law case-8 table, before any deliberate corruption."""
        a = p['a_i']*np.exp(np.linspace(0., np.log(1./p['a_i']), n))
        law = null_laws(a, p)
        d = np.zeros((n, N_COLS), dtype=float)
        d[:, N_A] = a
        d[:, N_RHO] = 1.0*law['rho']
        d[:, N_U] = 3.5*law['u']
        d[:, N_B] = 2.0*law['B']
        d[:, N_PDV] = -1.0          # the local source is exactly -3H(gamma-1)u
        for column, value in override.items():
            d[:, globals()[column]] = value
        return d

    def test_case8_uses_gamma_five_thirds_and_the_family_a_i(self):
        p = case_params(8, factor_A=1., factor_G=0., A_u=0.1)
        self.assertAlmostEqual(p['gamma'], 5./3., places=13)
        self.assertAlmostEqual(p['a_i'], 1./128., places=13)

    def test_a_init_overrides_a_i_only_for_the_dissipation_cases(self):
        p = case_params(8, factor_A=1., a_init=1./32.)
        self.assertAlmostEqual(p['a_i'], 1./32., places=13)
        with self.assertRaises(ValueError):
            case_params(3, factor_A=1., a_init=1./32.)
        with self.assertRaises(ValueError):
            case_params(8, factor_A=1., a_init=2.0)

    def test_null_laws_match_the_analytic_expansion_exponents(self):
        p = case_params(8, factor_A=1.)
        a = np.asarray([p['a_i'], 4.*p['a_i'], 1.])
        law = null_laws(a, p)
        r = a/p['a_i']
        np.testing.assert_allclose(law['rho'], np.ones_like(r), rtol=0, atol=0)
        # gamma = 5/3 makes the thermal law exactly a^-2
        np.testing.assert_allclose(law['u'], r**-2., rtol=1e-13)
        np.testing.assert_allclose(law['B'], r**-0.5, rtol=1e-13)

    def test_an_exactly_on_law_series_has_zero_residuals(self):
        p = case_params(8, factor_A=1.)
        out = null_residuals(self._series(p), p)
        for key in ('law:rho', 'law:u', 'law:B', 'rate:uDotPdV+1'):
            self.assertLess(out[key], 1e-13, key)

    def test_a_wrong_power_of_a_is_caught(self):
        """The failure mode the whole suite exists for: one power of a."""
        p = case_params(8, factor_A=1.)
        d = self._series(p)
        a_ratio = d[:, N_A]/p['a_i']
        d[:, N_U] = d[:, N_U]*a_ratio        # u propto a^-1 instead of a^-2
        out = null_residuals(d, p)
        self.assertGreater(out['law:u'], 1.0)
        self.assertLess(out['law:B'], 1e-13)

    def test_the_invariants_are_gauged_to_the_first_snapshot(self):
        """A run whose IC normalization differs from the analytic value must
        still pass: only the expansion LAW is under test."""
        p = case_params(8, factor_A=1.)
        d = self._series(p)
        d[:, N_U] *= 137.0
        d[:, N_B] *= 0.011
        out = null_residuals(d, p)
        self.assertLess(out['law:u'], 1e-13)
        self.assertLess(out['law:B'], 1e-13)

    def test_uniformity_and_divB_are_gated_on_drift_not_magnitude(self):
        """A relaxed glass carries a few per cent of SPH density scatter and a
        finite h|divB|/|B| for a uniform field; a lattice carries neither.  A
        constant offset must not fail the null test, a growing one must."""
        p = case_params(8, factor_A=1.)
        constant = null_residuals(
            self._series(p, N_RHO_SPR=0.041, N_DIVB=3.9e-5), p)
        self.assertLess(constant['drift:spread:rho'], 1e-13)
        self.assertLess(constant['drift:divB'], 1e-13)
        d = self._series(p)
        d[:, N_RHO_SPR] = np.linspace(0.041, 0.061, d.shape[0])
        self.assertAlmostEqual(
            null_residuals(d, p)['drift:spread:rho'], 0.020, places=12)

    def test_any_dissipation_at_all_is_a_failure(self):
        p = case_params(8, factor_A=1.)
        out = null_residuals(self._series(p, N_AV=1e-3, N_DIFF=2e-4), p)
        self.assertAlmostEqual(out['rate:uDotAV'], 1e-3, places=13)
        self.assertAlmostEqual(out['rate:uDotDiff'], 2e-4, places=13)

    def test_reference_table_covers_every_gated_series(self):
        p = case_params(8, factor_A=1.)
        table = null_reference_table(self._series(p))
        for key in ('x', 'rho', 'u', 'B', 'rho_spread', 'u_spread', 'mach',
                    'uDotAV', 'uDotDiff'):
            self.assertIn(key, table)
            self.assertEqual(table[key].shape, table['x'].shape)


class TestCase9ThermalMode(unittest.TestCase):
    """Case 9: the sinusoidal thermal mode and its two scale-factor clocks.

    The whole test rests on the two channels' clocks differing by exactly one
    power of a, so these check the clocks and the fit that separates them.
    """

    def _run(self, p, decay, channel, n=120):
        """A series decaying exactly exponentially in `channel`'s clock."""
        a = p['a_i']*np.exp(np.linspace(0., np.log(1./p['a_i']), n))
        clock = thermal_clock(a, p, channel)
        d = np.zeros((n, T_COLS), dtype=float)
        d[:, T_A] = a
        d[:, T_AMP] = 0.1*np.exp(-decay*(clock - clock[0]))
        return d

    def test_case9_shares_the_family_a_i_and_uses_gamma_five_thirds(self):
        p = case_params(9, factor_A=0., factor_G=0., A_u=0.1)
        self.assertAlmostEqual(p['gamma'], 5./3., places=13)
        self.assertAlmostEqual(p['a_i'], 1./128., places=13)

    def test_the_comoving_clock_is_eds_cosmic_time(self):
        p = case_params(9, factor_A=0.)
        a = np.asarray([0.1, 0.5, 1.0])
        np.testing.assert_allclose(thermal_clock(a, p, 'comoving'),
                                   (2./3.)*a**1.5/p['H0'], rtol=1e-12)

    def test_the_physical_clock_is_the_integral_of_a_dt(self):
        """int a dt = int da/H(a); in EdS that is (2/5) a^{5/2}/H0."""
        p = case_params(9, factor_A=0.)
        a = np.asarray([0.1, 0.5, 1.0])
        got = thermal_clock(a, p, 'physical') - thermal_clock(0., p, 'physical')
        np.testing.assert_allclose(got, (2./5.)*a**2.5/p['H0'], rtol=1e-6)

    def test_the_two_clocks_differ_by_exactly_one_power_of_a(self):
        """Their ratio over the run is <a>, which is what the cross-arm check
        compares against with no coefficient in it."""
        p = case_params(9, factor_A=0.)
        a = p['a_i']*np.exp(np.linspace(0., np.log(1./p['a_i']), 4000))
        ca = thermal_clock(a, p, 'physical')
        ct = thermal_clock(a, p, 'comoving')
        mean_a = (ca[-1] - ca[0])/(ct[-1] - ct[0])
        # <a> = int a dt / int dt, evaluated directly on the same grid
        expected = np.trapz(a, ct)/(ct[-1] - ct[0])
        self.assertAlmostEqual(mean_a, expected, places=6)

    def test_the_fit_recovers_the_decay_rate_on_the_right_clock(self):
        p = case_params(9, factor_A=0.)
        for channel in ('physical', 'comoving'):
            lam, res = thermal_fit(self._run(p, 7.77, channel), p, channel)
            self.assertAlmostEqual(lam, 7.77, places=6, msg=channel)
            self.assertLess(res, 1e-10, channel)

    def test_the_wrong_clock_is_caught(self):
        """The failure mode the case exists for: one power of a.  A mode that
        decays in `int a dt` must NOT look exponential in t, and vice versa."""
        p = case_params(9, factor_A=0.)
        for channel, other in (('physical', 'comoving'),
                               ('comoving', 'physical')):
            d = self._run(p, 7.77, channel)
            _lam, res_wrong = thermal_fit(d, p, other)
            self.assertGreater(res_wrong, 0.05, f"{channel} vs {other}")

    def test_residuals_report_own_other_and_the_discrimination_ratio(self):
        p = case_params(9, factor_A=0.)
        out = thermal_residuals(self._run(p, 7.77, 'physical'), p, 'physical')
        self.assertLess(out['residual'], 1e-10)
        self.assertGreater(out['null:wrong-a-power'], 0.05)
        self.assertGreater(out['sensitivity'], 1e6)
        self.assertAlmostEqual(out['Lambda'], 7.77, places=6)

    def test_an_unknown_channel_is_rejected(self):
        p = case_params(9, factor_A=0.)
        with self.assertRaises(ValueError):
            thermal_clock(np.asarray([0.5]), p, 'nonsense')
