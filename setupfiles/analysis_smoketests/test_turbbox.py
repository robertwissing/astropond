#!/usr/bin/env python3
"""Focused, file-free checks for the OU turbulence-box setup and analysis."""

import contextlib
import io
import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.dirname(HERE)
sys.path.insert(0, SETUP)

import IC_analysis_turbbox as analysis  # noqa: E402
import IC_setup_turbbox as setup         # noqa: E402
import turbbox_ensemble as ensemble      # noqa: E402
import turbbox_render as field_render   # noqa: E402
from turbbox_magnetic import particle_magnetic_moments  # noqa: E402
import IC_analysis_render as render      # noqa: E402
from sph_interp import GridResult, UniformGrid  # noqa: E402


class TestTurbboxSetup(unittest.TestCase):
    @staticmethod
    def _populate_lattice(sim, *_args, **_kwargs):
        q = np.array([-0.25, 0.25])
        x, y, z = np.meshgrid(q, q, q, indexing="ij")
        sim.x = x.ravel().tolist()
        sim.y = y.ravel().tolist()
        sim.z = z.ravel().tolist()
        sim.rho = np.ones(x.size).tolist()

    def _create(self, **kwargs):
        sim = setup.setup_turbbox()
        with patch.object(setup.distribute, "setbound"), \
                patch.object(setup.distribute, "setcloseddist",
                             side_effect=self._populate_lattice), \
                patch.object(setup.smth, "getsmooth",
                             side_effect=lambda obj, _n: np.ones(obj.npart)), \
                contextlib.redirect_stdout(io.StringIO()):
            sim.create(2, **kwargs)
        return sim

    def test_reference_scaling_and_projection_controls(self):
        sol = self._create(mach=2, forcing="sol", mhd=0,
                           crossings=0.25, seed=7)
        self.assertEqual(sol.npart, 8)
        self.assertEqual(sol.turbbox_forcing, "solenoidal")
        self.assertEqual(sol.turbbox_mhd, 0)
        self.assertEqual(sol.StSolWeight, 1.0)
        self.assertAlmostEqual(sol.StDecay, 0.25)
        self.assertAlmostEqual(sol.StDtFreq, 0.025)
        self.assertAlmostEqual(sol.StEnergy, 8.0 * (2.0 / 10.0) ** 3)
        self.assertEqual(sol.StCorrectBulkMotion, 1)
        self.assertEqual(sol.nsteps, 25)
        self.assertEqual(sol.freqout, 25)
        self.assertTrue(np.all(sol.Bz == 0.0))
        self.assertTrue(np.all(sol.vx == 0.0))

        comp = self._create(mach=10, forcing="compressive", mhd=1)
        self.assertEqual(comp.StSolWeight, 0.0)
        self.assertAlmostEqual(comp.StDecay, 0.05)
        self.assertAlmostEqual(comp.StEnergy, 8.0)
        np.testing.assert_allclose(comp.Bz, math.sqrt(2.0) * 1.0e-5,
                                   rtol=0.0, atol=0.0)

    def test_invalid_inputs_are_rejected(self):
        for kwargs in (
                {"mach": 0}, {"mach": float("nan")}, {"forcing": "mixed"},
                {"mhd": 2}, {"B0": -1}, {"crossings": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._create(**kwargs)

    def test_published_driving_selection_and_duration(self):
        paper = self._create()
        self.assertEqual(paper.StForcingFile, "forcing/forcing.dat")
        self.assertEqual((paper.StKmin, paper.StKmax), (6.28, 18.90))
        self.assertEqual(self._create(mach=2).StForcingFile, "")
        self.assertEqual(self._create(driving="generated").StForcingFile, "")
        for kwargs in ({"driving": "paper", "mach": 2},
                       {"driving": "paper", "seed": 7},
                       {"driving": "paper", "forcing": "comp"},
                       {"crossings": 101}, {"driving": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._create(**kwargs)


class TestTurbboxSpectra(unittest.TestCase):
    @staticmethod
    def _mode(component, mode=3, n=24):
        x = np.arange(n, dtype=float) / n
        wave = np.sin(2.0 * np.pi * mode * x)[:, None, None]
        vector = np.zeros((3, n, n, n), dtype=float)
        vector[component] = np.broadcast_to(wave, (n, n, n))
        return vector

    def test_longitudinal_mode_is_purely_compressive_and_parseval_normalized(self):
        spec = analysis.isotropic_vector_spectrum(self._mode(0))
        peak = int(np.argmax(spec["total"]))
        self.assertEqual(spec["k"][peak], 3.0)
        self.assertAlmostEqual(float(np.sum(spec["total"])), 0.25, places=13)
        self.assertLess(float(np.sum(spec["solenoidal"])), 1.0e-28)
        np.testing.assert_allclose(spec["compressive"], spec["total"],
                                   rtol=1.0e-13, atol=1.0e-28)

    def test_transverse_mode_is_purely_solenoidal(self):
        spec = analysis.isotropic_vector_spectrum(self._mode(1))
        self.assertLess(float(np.sum(spec["compressive"])), 1.0e-28)
        np.testing.assert_allclose(spec["solenoidal"], spec["total"],
                                   rtol=1.0e-13, atol=1.0e-28)

    def test_runname_tags_decode(self):
        run = "turbbox32_N64glassM10compmhdB1e-5tc100seed17GASOLINE"
        self.assertEqual(analysis._forcing_from_runname(run), "compressive")
        self.assertEqual(analysis._mhd_from_runname(run), 1)

    def test_tricco_kinetic_weight_is_rho_to_one_half(self):
        rho = np.array([[[1.0, 4.0]]])
        velocity = np.ones((3, 1, 1, 2))
        weighted = analysis.density_weighted_velocity(rho, velocity, 0.5)
        np.testing.assert_allclose(weighted[..., 0], 1.0)
        np.testing.assert_allclose(weighted[..., 1], 2.0)
        with self.assertRaises(ValueError):
            analysis.density_weighted_velocity(-rho, velocity, 0.5)


class TestTurbboxEnsembles(unittest.TestCase):
    def test_cache_signature_ignores_cli_but_detects_numerical_change(self):
        original = 'def spectrum(x):\n return x*x\ndef main():\n pass\n'
        changed_cli = original.replace('pass', 'print(42)')
        changed_math = original.replace('x*x', 'x+x')
        expected = ensemble.numerical_signature(original, ('spectrum',))
        self.assertEqual(expected, ensemble.numerical_signature(changed_cli, ('spectrum',)))
        self.assertNotEqual(expected, ensemble.numerical_signature(changed_math, ('spectrum',)))

    def test_contract_rejects_nonfinite_or_malformed_later_occurrences(self):
        for token in ('nan','inf','invalid'):
            _values,errors=ensemble.numeric_contract(
                f'# dConstAlpha: 0.5\n# dConstAlpha: {token}', {'dConstAlpha': .5})
            self.assertTrue(errors)

    def test_fixed_bins_preserve_unresolved_tail_probability(self):
        pdf, outside = ensemble.fixed_pdf([-3, -0.5, 0.5, 3], [-1, 0, 1])
        np.testing.assert_allclose(pdf, [0.25, 0.25])
        self.assertEqual(outside, 0.5)
        weighted, outside = ensemble.fixed_pdf([0.5, 2], [0, 1], [3, 1])
        np.testing.assert_allclose(weighted, [0.75])
        self.assertEqual(outside, 0.25)

    def test_uniform_time_selection_rejects_missing_coverage_and_duplicates(self):
        rows = [{"time": t*0.05} for t in np.arange(80, 100.01, 0.25)]
        picked = ensemble.select_samples(rows, 80, 100, 11)
        np.testing.assert_allclose([r["time"]/0.05 for r in picked],
                                   np.arange(80, 100.1, 2))
        with self.assertRaises(ValueError):
            ensemble.select_samples(rows, 79, 100, 11)
        with self.assertRaises(ValueError):
            ensemble.select_samples(rows, 80, 81, 10)

    def test_realised_contract_rejects_contradictory_headers(self):
        _, errors = ensemble.numeric_contract(
            "# dConstAlpha: 0.5\n# dConstAlpha: 2", {"dConstAlpha": 0.5})
        self.assertTrue(errors)
        _, errors = ensemble.numeric_contract("# dConstAlpha: 0.5", {"dConstAlpha": 0.5})
        self.assertFalse(errors)

    def test_temporal_scatter_is_sample_standard_deviation(self):
        mean, std = ensemble.aggregate([{"x": np.array([1, 3])},
                                        {"x": np.array([3, 7])}], "x")
        np.testing.assert_allclose(mean, [2, 5])
        np.testing.assert_allclose(std, np.sqrt([2, 8]))

    def test_grid_energy_and_pdf_are_invariant_to_uniform_bulk_velocity(self):
        velocity = TestTurbboxSpectra._mode(1, mode=2, n=8)
        rho = np.ones((8, 8, 8))
        data = {"rho": rho, "_mass": rho/512,
                "vx": velocity[0], "vy": velocity[1], "vz": velocity[2],
                "Bx": rho, "By": rho*0, "Bz": rho*0}
        row = {"time": 4, "npart": 512, "mass": 1, "ekin": 0.25, "emag": 0.5,
               "bulk_vx": 0, "bulk_vy": 0, "bulk_vz": 0}
        reference = ensemble.grid_diagnostics(data, row)
        shifted = dict(data)
        for key, bulk, offset in zip(("vx", "vy", "vz"),
                                    ("bulk_vx", "bulk_vy", "bulk_vz"), (2, -3, 4)):
            shifted[key] = data[key] + offset
            row[bulk] = offset
        actual = ensemble.grid_diagnostics(shifted, row)
        np.testing.assert_allclose(actual["kinetic_total"], reference["kinetic_total"], atol=1e-28)
        self.assertAlmostEqual(actual["grid_kinetic_over_particle"], 1)
        self.assertAlmostEqual(actual["grid_magnetic_over_particle"], 1)
        self.assertAlmostEqual(actual["magnetic_zero_mode_power"], 0.5)
        self.assertAlmostEqual(np.sum(actual["rho_pdf"]*np.diff(ensemble.RHO_EDGES)), 1)


class TestTurbboxAnalysis(unittest.TestCase):
    def test_magnetic_mean_uses_signed_vectors_and_volume_weights(self):
        gas = np.zeros((2, 12))
        gas[:, 0] = [1, 3]
        gas[:, 7] = 1
        field = np.array([[-3, 2, 0], [1, 2, 0]])
        result = particle_magnetic_moments(gas, field)
        np.testing.assert_allclose(result["mean_B"], [0, 2, 0])
        np.testing.assert_allclose(result["integral_B"], [0, 8, 0])
        self.assertEqual(result["volume_sum"], 4)
        self.assertEqual(result["magnetic_energy"], 14)
        self.assertAlmostEqual(result["mean_energy_fraction"], 4/7)
        uniform = particle_magnetic_moments(gas, np.array([[0, 0, 2], [0, 0, 2]]))
        self.assertEqual(uniform["mean_energy_fraction"], 1)

    def test_magnetic_mean_rejects_invalid_quadrature_inputs(self):
        gas = np.ones((2, 12))
        for field in (np.zeros((2, 2)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                particle_magnetic_moments(gas, field)
        gas[0, 7] = 0
        with self.assertRaises(ValueError):
            particle_magnetic_moments(gas, np.ones((2, 3)))

    def test_shared_density_slice_preserves_stored_density_and_wraps_periodically(self):
        # A stored density may differ from the renderer's mass-kernel sum,
        # e.g. when the simulation uses CORRPARTITION or a different kernel.
        # A constant input field must retain that density wherever supported.
        gas = np.zeros((1,12))
        gas[0,0] = .3
        gas[0,1:4] = [.125,.125,-.5]
        gas[0,7] = 19
        spec = render.styled('rho',1,2,7,label='density')
        periods = {1:1.,2:1.,3:1.}
        with patch.object(render,'_smoothing_length',return_value=np.array([.1])):
            actual = render._grid_image('synthetic',gas,1,spec,gas[:,7],
                                       0,1,0,1,4,'slice',.5,periods)
            wrapped = gas.copy()
            wrapped[0,1] += 1
            wrapped[0,3] += 1
            other = render._grid_image('synthetic',wrapped,1,spec,wrapped[:,7],
                                      0,1,0,1,4,'slice',.5,periods)
        self.assertAlmostEqual(actual[0,0],19.,places=11)
        self.assertTrue(np.any(actual > 0.))
        np.testing.assert_allclose(actual[actual > 0.],19.,rtol=0,atol=1e-12)
        np.testing.assert_allclose(actual,other,rtol=0,atol=1e-12)

    def test_column_integrates_mass_and_magnetic_magnitude_before_projection(self):
        gas = np.zeros((2,12))
        gas[:,0] = .5
        gas[:,1:4] = [[.125,.125,-.25],[.125,.125,.25]]
        gas[:,7] = 2.
        header = np.zeros(6,dtype=int)
        header[2] = 2
        # Opposite fields cancel as vectors, but integral |B| dz must be positive.
        magnetic = np.array([[2.,0.,0.],[-2.,0.,0.]])
        with patch.object(field_render,'select_at_time',return_value=('synthetic',.5)) as select, \
                patch.object(field_render,'_cached_readtipsy',return_value=(gas,None,None,header,.5)), \
                patch.object(field_render,'read_box_periods',return_value={1:1.,2:1.,3:1.}), \
                patch.object(render,'read_vector_aux',return_value=magnetic), \
                patch.object(render,'_smoothing_length',return_value=np.full(2,.1)):
            fields, metadata = field_render._snapshot_fields(
                'run',2.,64,.5,64,'column',tcross=.25)
        select.assert_called_once_with('run',.5,test='turbbox',warn_outside=False)
        # A resolved SPH image integral approximates sum(m) and sum((m/rho)*|B|).
        for field in fields:
            self.assertAlmostEqual(float(np.mean(field)),1.,delta=1e-4)
        self.assertEqual(metadata['time_crossings'],2.)
        self.assertIsNone(metadata['zslice'])
        self.assertEqual(metadata['column_axis'],'z')
        self.assertEqual(metadata['render_backend'],'grid')
        self.assertEqual(metadata['render_method'],'shared SPH column')

    def test_render_sequence_default_times_labels_and_shared_colour_scales(self):
        import matplotlib.pyplot as plt

        def sample(inp,tau,resolution,zslice,nsmooth,project,tcross):
            value = tau+(100 if inp == 'run_b' else 0)
            return [np.full((4,4),value),np.full((4,4),value*1e-5)], {'run':inp,'time_crossings':tau}

        for project in ('slice','column'):
            with patch.object(field_render,'_snapshot_fields',side_effect=sample), \
                    contextlib.redirect_stdout(io.StringIO()):
                figure,metadata = field_render.render_sequence(
                    ['run_a','run_b'],labels=['A','B'],resolution=4,project=project)
            try:
                self.assertEqual(metadata['requested_times_crossings'],[1.,2.,4.,8.])
                image_axes = [ax for ax in figure.axes if ax.images]
                self.assertEqual(len(image_axes),16)
                expected = [1.,2.,4.,8.,101.,102.,104.,108.]
                np.testing.assert_allclose([ax.images[0].get_array()[0,0] for ax in image_axes[:8]],expected)
                for block in (image_axes[:8],image_axes[8:]):
                    self.assertTrue(all(ax.images[0].norm is block[0].images[0].norm for ax in block))
                self.assertEqual(image_axes[0].get_ylabel(),'A')
                self.assertEqual(image_axes[4].get_ylabel(),'B')
                self.assertIn('z',figure._suptitle.get_text())
            finally:
                plt.close(figure)

    def test_missing_render_time_is_distinct_from_invalid_data(self):
        for chosen in (None, ('last_snapshot',.05)):
            with patch.object(field_render,'select_at_time',return_value=chosen), \
                    patch.object(field_render,'_cached_readtipsy') as read:
                with self.assertRaises(field_render.RenderTimeUnavailable):
                    field_render._snapshot_fields('run',2.,4,.5,64,'slice')
                read.assert_not_called()

    def test_render_sequence_warns_and_keeps_only_available_time_columns(self):
        import matplotlib.pyplot as plt

        def sample(inp,tau,*args):
            if tau != 1.:
                raise field_render.RenderTimeUnavailable(f'{inp}: nearest snapshot is at 1')
            return [np.ones((4,4)),np.full((4,4),1e-5)], {'run':inp,'time_crossings':tau}

        for project in ('slice','column'):
            warnings = io.StringIO()
            with patch.object(field_render,'_snapshot_fields',side_effect=sample), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(warnings):
                figure,metadata = field_render.render_sequence(
                    ['run'],labels=['128 cubed'],resolution=4,project=project)
            try:
                self.assertEqual(warnings.getvalue().count('warning:'),3)
                self.assertEqual(metadata['requested_times_crossings'],[1.,2.,4.,8.])
                self.assertEqual(metadata['rendered_times_crossings'],[1.])
                self.assertEqual(len(metadata['skipped_panels']),3)
                self.assertEqual(len([ax for ax in figure.axes if ax.images]),2)
            finally:
                plt.close(figure)

    def test_render_sequence_labels_missing_run_panels_without_losing_other_runs(self):
        import matplotlib.pyplot as plt

        def sample(inp,tau,*args):
            if tau > 2. or (inp == 'short_run' and tau > 1.):
                raise field_render.RenderTimeUnavailable('snapshot unavailable')
            return [np.full((4,4),tau),np.full((4,4),tau*1e-5)], {'run':inp,'time_crossings':tau}

        with patch.object(field_render,'_snapshot_fields',side_effect=sample), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            figure,metadata = field_render.render_sequence(
                ['short_run','long_run'],labels=['Short','Long'],resolution=4)
        try:
            self.assertEqual(metadata['rendered_times_crossings'],[1.,2.])
            self.assertEqual(len(metadata['panels']),3)
            self.assertEqual(len(metadata['skipped_panels']),5)
            self.assertEqual(len([ax for ax in figure.axes if ax.images]),6)
            missing = [ax for ax in figure.axes
                       if any(text.get_text() == 'Snapshot\nunavailable' for text in ax.texts)]
            self.assertEqual(len(missing),2)
        finally:
            plt.close(figure)

    def test_render_sequence_with_no_available_times_returns_without_figure(self):
        warnings = io.StringIO()
        with patch.object(field_render,'_snapshot_fields',side_effect=field_render.RenderTimeUnavailable('no snapshot')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(warnings):
            figure,metadata = field_render.render_sequence(['run'],labels=['Run'],resolution=4)
        self.assertIsNone(figure)
        self.assertEqual(metadata['rendered_times_crossings'],[])
        self.assertEqual(len(metadata['skipped_panels']),4)
        self.assertEqual(warnings.getvalue().count('warning:'),4)
        # Backend/data failures are not silently classified as missing times.
        with patch.object(field_render,'_snapshot_fields',side_effect=ValueError('invalid box')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'invalid box'):
                field_render.render_sequence(['run'],labels=['Run'],resolution=4)

    def test_snapshot_diagnostics_remove_mass_weighted_bulk_flow(self):
        gas = np.zeros((3, 12), dtype=float)
        gas[:, 0] = [1.0, 2.0, 1.0]
        gas[:, 4:7] = [[2.0, 0.0, 0.0],
                       [4.0, 0.0, 0.0],
                       [6.0, 0.0, 0.0]]
        gas[:, 7] = 1.0
        header = np.zeros(6, dtype=int)
        header[2] = 3
        with patch.object(analysis.tip, "readtipsy",
                          return_value=(gas, None, None, header,
                                        np.array([0.125]))), \
                patch.object(analysis, "read_vector_aux", return_value=None):
            row = analysis.snapshot_diagnostics("synthetic.00001")
        self.assertAlmostEqual(row["bulk_vx"], 4.0)
        self.assertAlmostEqual(row["vrms"], math.sqrt(2.0))
        self.assertAlmostEqual(row["mach"], math.sqrt(2.0))
        self.assertAlmostEqual(row["ekin"], 4.0)
        self.assertTrue(math.isnan(row["emag"]))

    def test_snapshot_alfven_diagnostic_is_rms_of_local_ratio(self):
        gas = np.zeros((2, 12), dtype=float)
        gas[:, 0] = 0.5
        gas[:, 4:7] = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        gas[:, 7] = [1.0, 4.0]
        field = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        header = np.zeros(6, dtype=int)
        header[2] = 2
        with patch.object(analysis.tip, "readtipsy",
                          return_value=(gas, None, None, header,
                                        np.array([0.125]))), \
                patch.object(analysis, "read_vector_aux",
                             return_value=field):
            row = analysis.snapshot_diagnostics("synthetic.00001")
        # Both particles have v_A=1 and local M_A=1.
        self.assertAlmostEqual(row["va_rms_over_cs"], 1.0)
        self.assertAlmostEqual(row["alfven_mach_rms"], 1.0)
        self.assertAlmostEqual(row["vrms_over_va_rms"], 1.0)
        self.assertAlmostEqual(row["emag"], 0.5)

    def test_declared_stationarity_and_plateau_metrics(self):
        tau = np.linspace(0.0, 100.0, 401)
        mass = np.ones_like(tau)
        mach = 10.0 + 0.2 * np.sin(tau)
        ekin = 0.5 * mach ** 2
        emag = 1.5 * (1.0 + 0.03 * np.cos(tau))
        table = {
            "time": 0.05 * tau,
            "mass": mass,
            "mach": mach,
            "ekin": ekin,
            "emag": emag,
            "emag_over_ekin": emag / ekin,
        }
        metrics = analysis.convergence_diagnostics(
            table, tcross=0.05, target_mach=10.0,
            initial_B0=math.sqrt(2.0) * 1.0e-5)
        self.assertTrue(metrics["mach_stationary_by_declared_window_test"])
        self.assertTrue(
            metrics["kinetic_energy_stationary_by_declared_window_test"])
        self.assertTrue(
            metrics["magnetic_energy_plateau_by_declared_window_test"])
        self.assertTrue(metrics["mach_within_10_percent_of_driving_input"])

    def test_petkova_grid_calls_only_petkova_and_checks_mass(self):
        particles = SimpleNamespace(n=2, mass=np.array([0.4, 0.6]))
        data = {
            "rho": np.ones((4, 4, 4)),
            "vx": np.zeros((4, 4, 4)),
            "vy": np.zeros((4, 4, 4)),
            "vz": np.zeros((4, 4, 4)),
            "_mass": np.full((4, 4, 4), 1.0 / 64.0),
        }

        def fake_interpolate(p, target, method, values):
            self.assertIs(p, particles)
            self.assertIsInstance(target, UniformGrid)
            self.assertEqual(method, "petkova")
            self.assertEqual(values, ["rho", "vx", "vy", "vz"])
            return GridResult(data=data, norm=None, target=target,
                              method=method)

        with patch.object(analysis, "from_tipsy", return_value=particles) as load, \
                patch.object(analysis, "interpolate",
                             side_effect=fake_interpolate):
            result, returned, error = analysis.petkova_grid(
                "synthetic.00001", mhd=False, grid_res=4, nsmooth=32)
        self.assertIs(returned, particles)
        self.assertIs(result.data, data)
        self.assertAlmostEqual(error, 0.0)
        self.assertEqual(load.call_args.kwargs["periodic"], True)
        np.testing.assert_array_equal(load.call_args.kwargs["box"], np.ones(3))

    def test_all_synthetic_mhd_figure_paths_are_finite(self):
        import matplotlib.pyplot as plt

        n = 12
        target = UniformGrid.centered(np.ones(3), npx=n)
        x, y, z = np.meshgrid(*target.centres(), indexing="ij")
        rho = np.exp(0.15 * np.sin(2 * np.pi * x)
                     + 0.08 * np.cos(2 * np.pi * y))
        data = {
            "rho": rho,
            "vx": np.sin(4 * np.pi * x),
            "vy": np.cos(4 * np.pi * x),
            "vz": 0.3 * np.sin(2 * np.pi * z),
            "Bx": 1.0e-5 * (1.0 + 0.1 * np.sin(2 * np.pi * y)),
            "By": 1.0e-6 * np.cos(2 * np.pi * z),
            "Bz": 1.0e-5 * (1.0 + 0.1 * np.cos(2 * np.pi * x)),
        }
        grid = GridResult(data=data, norm=None, target=target, method="petkova")
        figures = []
        try:
            figures.append(analysis.maps_figure(grid, True, 0.25))
            spectrum, smetrics = analysis.spectra_figure(grid, True)
            figures.append(spectrum)
            statistics, pmetrics = analysis.statistics_figure(
                grid, True, "solenoidal", achieved_mach=2.0)
            figures.append(statistics)
            self.assertTrue(np.isfinite(smetrics["compressive_mode_fraction"]))
            self.assertEqual(smetrics["mhd_kinetic_spectrum_field"],
                             "rho^(1/2) v")
            self.assertTrue(np.isfinite(pmetrics["log_density_sigma"]))
            self.assertTrue(np.isfinite(pmetrics["brho_log_slope"]))
        finally:
            for figure in figures:
                plt.close(figure)

    def test_validation_page_renders_structured_evidence(self):
        import matplotlib.pyplot as plt

        payload = {
            "title": "Synthetic validation",
            "provenance": ["source fingerprint abc123"],
            "checks": [{
                "name": "projection",
                "status": "PASS",
                "evidence": ["longitudinal leakage = 0"],
            }],
            "limitations": ["synthetic fixture only"],
        }
        figure = analysis._report_validation(payload)
        try:
            rendered = "\n".join(text.get_text() for text in figure.texts)
            self.assertIn("Synthetic validation", rendered)
            self.assertIn("[PASS] projection", rendered)
            self.assertIn("synthetic fixture only", rendered)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
