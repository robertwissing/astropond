#!/usr/bin/env python3
"""Artifact-free formula checks for the external power-law test."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.dirname(HERE)
sys.path.insert(0, SETUP)

from IC_analysis_powerlawring import winding_rate  # noqa: E402
from IC_setup_powerlawring import (  # noqa: E402
    DOMAIN_ANNULUS,
    DOMAIN_DISK,
    EQUILIBRIUM_CUTOFF_DENSITY,
    EQUILIBRIUM_ENVELOPE_FLOOR,
    EQUILIBRIUM_ENVELOPE_SKIRT,
    EQUILIBRIUM_RADIAL_FLOOR,
    EQUILIBRIUM_SCALE_HEIGHT,
    slab_half_height,
    GEOMETRY_CYLINDRICAL,
    HOPKINS_PRESSURE_2D,
    HOPKINS_SURFACE_DENSITY,
    R_INNER,
    R_OUTER,
    FORCE_BOUNDED_KEPLER,
    FORCE_POWERLAW,
    TAPER_INNER_SUPPORT_RADIUS,
    circular_speed,
    equilibrium_azimuthal_speed,
    equilibrium_buffer_bounds,
    equilibrium_density,
    equilibrium_density_table,
    equilibrium_pressure,
    equilibrium_pressure_from_density,
    equilibrium_rotation_terms,
    equilibrium_surface_height,
    equilibrium_target_mass,
    equilibrium_vertical_extent,
    hopkins_scale_height,
    powerlaw_potential_difference,
    radial_density_factor,
    retained_particle_bounds,
    rotate_x,
    setup_powerlawring,
    toroidal_field_strength,
    toroidal_lorentz_v2,
    uniform_ring_bounds,
    validated_force_form,
)
from IC_createsetup import write_glass_density_table  # noqa: E402


class TestPowerLawRing(unittest.TestCase):
    def test_unsoftened_endpoints(self):
        radius = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(circular_speed(radius, beta=-0.5),
                                   radius ** -0.5)
        np.testing.assert_allclose(circular_speed(radius, beta=0.0), 1.0)
        np.testing.assert_allclose(circular_speed(radius, beta=1.0), radius)

    def test_softened_speed_matches_force_balance(self):
        radius = np.array([0.7, 1.4])
        beta, eps = 0.25, 0.2
        speed = circular_speed(radius, beta, eps)
        softened = np.sqrt(radius ** 2 + eps ** 2)
        acceleration_factor = softened ** (2.0 * beta - 2.0)
        np.testing.assert_allclose(speed ** 2 / radius,
                                   acceleration_factor * radius)

    def test_winding_endpoints(self):
        radius = np.array([1.0, 2.0])
        amplitude = 3.0e-4
        solid = winding_rate(radius, 1.0, amplitude)
        np.testing.assert_allclose(solid, 0.0)
        flat = winding_rate(radius, 0.0, amplitude)
        np.testing.assert_allclose(flat, -amplitude / radius ** 2)
        kepler = winding_rate(radius, -0.5, amplitude)
        np.testing.assert_allclose(kepler,
                                   -1.5 * amplitude * radius ** -2.5)

    def test_full_disk_toroidal_core_is_finite(self):
        radius = np.array([0.0, 0.1, 1.0])
        amplitude, eps = 2.0e-4, 0.1
        field = toroidal_field_strength(
            radius, amplitude, eps, DOMAIN_DISK)
        np.testing.assert_allclose(
            field, amplitude * radius / (radius ** 2 + eps ** 2))
        self.assertEqual(field[0], 0.0)
        self.assertTrue(np.all(np.isfinite(field)))
        np.testing.assert_allclose(
            toroidal_field_strength(radius[1:], amplitude, 0.0,
                                     DOMAIN_ANNULUS),
            amplitude / radius[1:])

    def test_full_disk_validation(self):
        with self.assertRaisesRegex(ValueError, "positive softening"):
            setup_powerlawring().create(8, softening=0.0,
                                        domain_mode=DOMAIN_DISK)
        with self.assertRaisesRegex(ValueError, "annulus-only"):
            setup_powerlawring().create(8, field_mode=2, softening=0.1,
                                        domain_mode=DOMAIN_DISK)

    def test_softened_full_disk_speed_is_regular(self):
        speed = circular_speed(np.array([0.0, 0.05, 0.5]), -0.5, 0.1)
        self.assertEqual(speed[0], 0.0)
        self.assertTrue(np.all(np.isfinite(speed)))

    def test_tilt_round_trip(self):
        vectors = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
        np.testing.assert_allclose(
            rotate_x(rotate_x(vectors, 37.0), -37.0), vectors,
            rtol=0.0, atol=2.0e-15)

    def test_equilibrium_radial_taper_is_smooth_and_positive(self):
        radius = np.array([-0.01, 0.0, 0.125, 0.25, 0.375, 0.5, 1.0,
                           2.0, 2.25, 2.5, 2.75, 3.0, 3.01])
        factor, dlog = radial_density_factor(radius, DOMAIN_ANNULUS)
        np.testing.assert_allclose(
            factor[[0, 1, 11, 12]], EQUILIBRIUM_ENVELOPE_FLOOR)
        np.testing.assert_allclose(
            factor[[3, 9]], EQUILIBRIUM_RADIAL_FLOOR)
        np.testing.assert_allclose(factor[[5, 6, 7]], 1.0)
        np.testing.assert_allclose(
            dlog[[0, 1, 3, 5, 6, 7, 9, 11, 12]], 0.0)
        self.assertGreater(dlog[2], 0.0)
        self.assertGreater(dlog[4], 0.0)
        self.assertLess(dlog[8], 0.0)
        self.assertLess(dlog[10], 0.0)
        self.assertEqual(EQUILIBRIUM_ENVELOPE_SKIRT, 0.5)

    def test_force_form_spelling(self):
        """0 and 1 are the only values, and everything else is rejected."""
        self.assertEqual(validated_force_form(0), FORCE_POWERLAW)
        self.assertEqual(validated_force_form(1), FORCE_BOUNDED_KEPLER)
        for bad in (2, 3, -1, 0.5):
            with self.assertRaisesRegex(ValueError, "force_form"):
                validated_force_form(bad)

    def test_hopkins_ring_is_the_carved_flat_ring(self):
        """The literature ring is a flat, sharp-edged R=0.5..2 annulus.

        Hopkins (2015) realizes the published surface density for the particle
        methods by DELETION -- an evenly spaced square with everything inside
        R=0.5 and outside R=2 removed -- so what the IC must reproduce is a
        uniform ring, not the profile's mesh-arm wings.  Its temperature is the
        published P=1e-6 on Sigma=1.01, and because the ring is flat the
        barotrope reproduces that as a constant pressure exactly.
        """
        hslab = slab_half_height(64)
        self.assertEqual(retained_particle_bounds(DOMAIN_ANNULUS,
                                                  GEOMETRY_CYLINDRICAL,
                                                  FORCE_BOUNDED_KEPLER),
                         uniform_ring_bounds(DOMAIN_ANNULUS))
        self.assertEqual(uniform_ring_bounds(DOMAIN_ANNULUS),
                         (R_INNER, R_OUTER))

        scale_height = hopkins_scale_height(1.0)
        radius = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
        density = equilibrium_density(
            radius, 0.0, geometry=GEOMETRY_CYLINDRICAL,
            scale_height=scale_height, force_form=FORCE_BOUNDED_KEPLER,
            slab_half=hslab)
        np.testing.assert_allclose(density, 1.0, rtol=0.0, atol=1.0e-15)

        pressure = equilibrium_pressure(
            radius, 0.0, geometry=GEOMETRY_CYLINDRICAL,
            scale_height=scale_height, force_form=FORCE_BOUNDED_KEPLER,
            slab_half=hslab)
        target = HOPKINS_PRESSURE_2D / HOPKINS_SURFACE_DENSITY
        np.testing.assert_allclose(pressure, target, rtol=1.0e-14)
        self.assertAlmostEqual(
            float(equilibrium_pressure_from_density(1.0, scale_height)),
            target, places=18)

        # The rotation law is the bare Keplerian circular speed: the flat
        # profile has no pressure gradient, and the force's modified branches
        # (R<0.35, R>2.1) lie outside everything emitted.
        terms = equilibrium_rotation_terms(
            radius, 0.0, geometry=GEOMETRY_CYLINDRICAL,
            scale_height=scale_height, force_form=FORCE_BOUNDED_KEPLER,
            slab_half=hslab)
        np.testing.assert_allclose(terms["pressure"], 0.0, atol=0.0)
        np.testing.assert_allclose(terms["magnetic"], 0.0, atol=0.0)
        np.testing.assert_allclose(terms["total"], 1.0 / radius, rtol=1.0e-15)
        np.testing.assert_allclose(
            equilibrium_azimuthal_speed(
                radius, 0.0, geometry=GEOMETRY_CYLINDRICAL,
                scale_height=scale_height, force_form=FORCE_BOUNDED_KEPLER,
                slab_half=hslab),
            circular_speed(radius, -0.5, 0.0), rtol=1.0e-15)

    def test_hopkins_ring_emits_the_published_disc(self):
        """An ordered start carves exactly the paper's annulus."""
        setup = setup_powerlawring()
        setup.create(24, distri=0, beta=-0.5,
                     geometry=GEOMETRY_CYLINDRICAL, field_mode=0,
                     softening=0.0, domain_mode=DOMAIN_ANNULUS,
                     force_form=FORCE_BOUNDED_KEPLER)
        radius = np.hypot(setup.x, setup.y)
        self.assertGreaterEqual(float(np.min(radius)), R_INNER)
        self.assertLessEqual(float(np.max(radius)), R_OUTER)
        # -h is not a knob here: create() substitutes the published value.
        self.assertAlmostEqual(setup.powerlaw_scale_height,
                               hopkins_scale_height(1.0), places=15)
        # Flat, equal mass, and the total is rho * the annulus volume.
        mass = np.asarray(setup.mass, dtype=float)
        np.testing.assert_allclose(np.asarray(setup.rho), 1.0,
                                   rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(mass, mass[0], rtol=1.0e-14)
        self.assertAlmostEqual(
            float(np.sum(mass)),
            np.pi * (R_OUTER ** 2 - R_INNER ** 2) * 2.0
            * setup.powerlaw_slab_half, places=12)
        # Exactly Keplerian, with no radial motion.
        vphi = ((np.asarray(setup.x) * np.asarray(setup.vy) -
                 np.asarray(setup.y) * np.asarray(setup.vx)) / radius)
        vr = ((np.asarray(setup.x) * np.asarray(setup.vx) +
               np.asarray(setup.y) * np.asarray(setup.vy)) / radius)
        np.testing.assert_allclose(vphi, radius ** -0.5, rtol=1.0e-14)
        np.testing.assert_allclose(vr, 0.0, atol=1.0e-14)
        # The .param carries the same spelling the CLI does.
        self.assertEqual(setup.powerlaw_force_form, FORCE_BOUNDED_KEPLER)

    def test_hopkins_mode_rejects_nonliterature_combinations(self):
        cases = (
            ({"geometry": 3}, "cylindrical"),
            ({"geometry": GEOMETRY_CYLINDRICAL,
              "domain_mode": DOMAIN_DISK, "softening": 0.1}, "domain_mode"),
            ({"geometry": GEOMETRY_CYLINDRICAL, "beta": 0.0}, "beta"),
            ({"geometry": GEOMETRY_CYLINDRICAL,
              "softening": 0.1}, "unsoftened"),
            ({"geometry": GEOMETRY_CYLINDRICAL,
              "field_mode": 1}, "hydrodynamic"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    setup_powerlawring().create(
                        8, force_form=FORCE_BOUNDED_KEPLER, **kwargs)

    def test_powerlaw_potential_difference_has_flat_limit(self):
        radius = np.array([0.75, 1.5, 2.25])
        height = np.array([0.03, 0.12, -0.2])
        eps = 0.17
        expected_flat = 0.5 * np.log(
            (radius ** 2 + height ** 2 + eps ** 2) /
            (radius ** 2 + eps ** 2))
        np.testing.assert_allclose(
            powerlaw_potential_difference(radius, height, 0.0, eps),
            expected_flat)
        np.testing.assert_allclose(
            powerlaw_potential_difference(radius, height, 1.0, eps),
            0.5 * height ** 2)
        np.testing.assert_allclose(
            powerlaw_potential_difference(radius, height, 1.0e-10, eps),
            expected_flat)

    def test_general_beta_vertical_profile_is_hydrostatic(self):
        radius = np.array([0.75, 1.5, 2.25])
        height = np.array([0.03, 0.08, -0.12])
        eps = 0.1
        delta = 1.0e-6
        for beta in (-0.5, 0.0, 0.25, 1.0):
            density = equilibrium_density(
                radius, height, beta=beta, softening=eps)
            pminus = equilibrium_pressure(
                radius, height - delta, beta=beta, softening=eps)
            pplus = equilibrium_pressure(
                radius, height + delta, beta=beta, softening=eps)
            dpressure_dz = (pplus - pminus) / (2.0 * delta)
            gravity_z = (height *
                         (radius ** 2 + height ** 2 + eps ** 2) **
                         (beta - 1.0))
            np.testing.assert_allclose(
                dpressure_dz, -density * gravity_z,
                rtol=2.0e-8, atol=2.0e-10)

    def test_beta_one_core_is_lowered_gaussian(self):
        radius = np.array([0.75, 1.5, 2.25])
        height = np.array([0.03, 0.12, 0.2])
        factor, _derivative = radial_density_factor(radius, DOMAIN_ANNULUS)
        gaussian = np.exp(-0.5 * (height / EQUILIBRIUM_SCALE_HEIGHT) ** 2)
        expected = np.maximum(
            (factor + EQUILIBRIUM_CUTOFF_DENSITY) * gaussian
            - EQUILIBRIUM_CUTOFF_DENSITY, 0.0)
        np.testing.assert_allclose(
            equilibrium_density(radius, height, beta=1.0), expected)
        self.assertGreater(float(equilibrium_pressure_from_density(1.0)), 0.0)

    def test_untapered_toroidal_lorentz_term_matches_field_derivative(self):
        amplitude = 2.0e-3
        for radius in (0.75, 1.5, 2.25):
            height = 0.04
            density = equilibrium_density(radius, height)
            delta = 1.0e-6
            bminus = toroidal_field_strength(
                radius - delta, amplitude, domain_mode=DOMAIN_ANNULUS)
            bplus = toroidal_field_strength(
                radius + delta, amplitude, domain_mode=DOMAIN_ANNULUS)
            bphi = toroidal_field_strength(
                radius, amplitude, domain_mode=DOMAIN_ANNULUS)
            derivative = (bplus ** 2 - bminus ** 2) / (2.0 * delta)
            direct = (bphi ** 2 + 0.5 * radius * derivative) / density
            analytic = toroidal_lorentz_v2(
                radius, density, amplitude, domain_mode=DOMAIN_ANNULUS)
            self.assertAlmostEqual(float(analytic), float(direct), places=10)
        self.assertAlmostEqual(
            float(toroidal_lorentz_v2(
                1.5, equilibrium_density(1.5, 0.04), amplitude)),
            0.0, places=14)

        eps = 0.1
        for radius in (0.1, 0.7, 1.7):
            density = equilibrium_density(
                radius, 0.0, domain_mode=DOMAIN_DISK, softening=eps)
            delta = 1.0e-6
            bminus = toroidal_field_strength(
                radius - delta, amplitude, eps, DOMAIN_DISK)
            bplus = toroidal_field_strength(
                radius + delta, amplitude, eps, DOMAIN_DISK)
            bphi = toroidal_field_strength(
                radius, amplitude, eps, DOMAIN_DISK)
            derivative = (bplus ** 2 - bminus ** 2) / (2.0 * delta)
            direct = (bphi ** 2 + 0.5 * radius * derivative) / density
            analytic = toroidal_lorentz_v2(
                radius, density, amplitude, eps, DOMAIN_DISK)
            self.assertAlmostEqual(float(analytic), float(direct), places=10)

    def test_equilibrium_rotation_uses_current_free_annular_field(self):
        radius = np.linspace(0.5, 2.5, 401)
        height = np.zeros_like(radius)
        terms = equilibrium_rotation_terms(
            radius, height, field_mode=1, field_amp=1.0e-4, beta=1.0)
        speed = equilibrium_azimuthal_speed(
            radius, height, field_mode=1, field_amp=1.0e-4, beta=1.0)
        np.testing.assert_allclose(speed ** 2, terms["total"])
        self.assertTrue(np.all(terms["total"] >= 0.0))
        core = (radius >= R_INNER) & (radius <= R_OUTER)
        np.testing.assert_allclose(speed[core], radius[core],
                                   rtol=0.0, atol=2.0e-15)
        self.assertGreater(np.max(np.abs(terms["pressure"])), 0.0)
        np.testing.assert_allclose(terms["magnetic"], 0.0)

    def test_general_beta_radial_balance_matches_pressure_derivative(self):
        radius = np.array([0.75, 1.5, 2.25])
        height = np.array([0.03, 0.08, 0.06])
        eps = 0.1
        delta = 1.0e-6
        for beta in (-0.5, 0.0, 0.25, 1.0):
            terms = equilibrium_rotation_terms(
                radius, height, softening=eps, beta=beta)
            density = equilibrium_density(
                radius, height, beta=beta, softening=eps)
            pminus = equilibrium_pressure(
                radius - delta, height, beta=beta, softening=eps)
            pplus = equilibrium_pressure(
                radius + delta, height, beta=beta, softening=eps)
            dpressure_dr = (pplus - pminus) / (2.0 * delta)
            gravity_at_height = (
                radius ** 2
                * (radius ** 2 + height ** 2 + eps ** 2) ** (beta - 1.0))
            direct = gravity_at_height + radius * dpressure_dr / density
            np.testing.assert_allclose(terms["total"], direct,
                                       rtol=3.0e-8, atol=3.0e-10)

    def test_general_beta_core_speed_is_softened_circular_speed(self):
        radius = np.array([1.1, 1.5, 1.9])
        height = np.array([0.02, 0.05, 0.08])
        eps = 0.1
        for beta in (-0.5, 0.0, 0.25, 1.0):
            np.testing.assert_allclose(
                equilibrium_azimuthal_speed(
                    radius, height, softening=eps, beta=beta),
                circular_speed(radius, beta, eps), rtol=2.0e-15, atol=0.0)

    def test_radial_field_taper_controls_alfven_speed(self):
        radius = np.array([0.55, 0.75, 1.5, 2.25, 2.45])
        height = np.full_like(radius, 0.1)
        amplitude = 1.0e-4
        density = equilibrium_density(radius, height)
        field = toroidal_field_strength(
            radius, amplitude, domain_mode=DOMAIN_ANNULUS,
            radial_taper=True)
        factor, _derivative = radial_density_factor(radius, DOMAIN_ANNULUS)
        expected_alfven2 = amplitude ** 2 * factor / (radius ** 2 * density)
        np.testing.assert_allclose(field ** 2 / density, expected_alfven2,
                                   rtol=2.0e-15, atol=0.0)

    def test_cylindrical_arm_is_a_sharp_edged_uniform_ring(self):
        inner, outer = uniform_ring_bounds(DOMAIN_ANNULUS)
        # both sharp surfaces ARE the science edges; only a taper keeps the
        # disposable support bands outside them.
        self.assertEqual((inner, outer), (R_INNER, R_OUTER))
        support_inner, support_outer = equilibrium_buffer_bounds(
            DOMAIN_ANNULUS)
        self.assertEqual(support_inner, TAPER_INNER_SUPPORT_RADIUS)
        self.assertGreater(support_outer, outer)
        hslab = slab_half_height(16)
        radius = np.array([inner - 0.01, inner, 1.0, outer, outer + 0.01])
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        # uniform in R inside the sharp edges, and uniform in z through the
        # whole periodic half period, with an exact top-hat surface.
        for height in (0.0, 0.1, hslab):
            np.testing.assert_allclose(
                equilibrium_density(radius, height,
                                    geometry=GEOMETRY_CYLINDRICAL,
                                    slab_half=hslab),
                expected, rtol=4.0e-16, atol=0.0)
        np.testing.assert_allclose(
            equilibrium_density(radius, hslab + 1.0e-9,
                                geometry=GEOMETRY_CYLINDRICAL,
                                slab_half=hslab), 0.0)
        np.testing.assert_allclose(
            equilibrium_surface_height(radius,
                                       geometry=GEOMETRY_CYLINDRICAL,
                                       slab_half=hslab), hslab)
        self.assertEqual(
            equilibrium_vertical_extent(DOMAIN_ANNULUS, -0.5, 0.0,
                                        GEOMETRY_CYLINDRICAL,
                                        slab_half=hslab), hslab)

        # a flat profile leaves NO pressure term, so v_phi is the exact
        # circular speed everywhere and B_phi=C/R stays current-free.
        ring = np.linspace(inner, outer, 401)
        for beta in (-0.5, 0.0, 0.25, 1.0):
            for eps in (0.0, 0.1):
                terms = equilibrium_rotation_terms(
                    ring, 0.2, softening=eps, beta=beta,
                    geometry=GEOMETRY_CYLINDRICAL)
                np.testing.assert_allclose(terms["pressure"], 0.0)
                np.testing.assert_allclose(
                    np.sqrt(terms["total"]), circular_speed(ring, beta, eps),
                    rtol=4.0e-16, atol=0.0)

        # the temperature cannot touch the cylindrical target at all
        for height in (0.0, 0.2):
            np.testing.assert_allclose(
                equilibrium_density(radius, height,
                                    geometry=GEOMETRY_CYLINDRICAL,
                                    scale_height=0.02, slab_half=hslab),
                expected, rtol=4.0e-16, atol=0.0)

        # its pre-glass is a plain uniform periodic box: no density table.
        setup = setup_powerlawring()
        setup.preglass = True
        setup.create(16, distri=1, entry="preglass", beta=-0.5,
                     geometry=GEOMETRY_CYLINDRICAL, field_mode=1)
        self.assertIsNone(setup.glass_density_table)
        # the seed box is the RING's bounding box, not the support radius:
        # a sharp ring must not pay to relax a shell it then throws away.
        self.assertAlmostEqual(float(np.sum(setup.mass)),
                               (2.0 * outer) ** 2 * 2.0 * hslab, places=12)

    def test_equilibrium_table_is_mapped_2d_and_mass_is_finite(self):
        beta, eps = -0.5, 0.05
        axes, values = equilibrium_density_table(DOMAIN_ANNULUS, beta, eps)
        self.assertEqual(values.shape, (len(axes[0]), len(axes[1])))
        self.assertEqual(axes[1][-1],
                         equilibrium_vertical_extent(DOMAIN_ANNULUS,
                                                     beta, eps))
        self.assertLessEqual(float(values[-1, 0]),
                             1.01 * EQUILIBRIUM_ENVELOPE_FLOOR)
        np.testing.assert_allclose(values[:, -1], 0.0)
        self.assertGreater(
            equilibrium_target_mass(DOMAIN_ANNULUS, beta, eps), 0.0)
        with patch("IC_createsetup.np.savez") as save:
            write_glass_density_table(
                "unused.npz", (axes, values, ('r_cyl', 'z')))
        payload = save.call_args.kwargs
        self.assertEqual(tuple(payload["mapping"]), ('r_cyl', 'z'))
        self.assertEqual(payload["values"].shape, values.shape)
        self.assertEqual(len(payload["ax0"]), len(axes[0]))
        self.assertEqual(len(payload["ax1"]), len(axes[1]))

    def test_equilibrium_only_validation_and_preglass_target(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            setup_powerlawring().create(
                8, beta=1, geometry=4)
        with self.assertRaisesRegex(ValueError, "tilt requires spherical"):
            setup_powerlawring().create(
                8, beta=1, geometry=GEOMETRY_CYLINDRICAL, tilt=30.0)
        # small integers must not be silently read as an absurdly hot disk.
        for bad in (0, 1.0, 2.0, -0.1):
            with self.assertRaisesRegex(ValueError, "scale_height"):
                setup_powerlawring().create(8, scale_height=bad)
        with self.assertRaisesRegex(ValueError, "requires a glass IC"):
            setup_powerlawring().create(
                8, beta=0, geometry=3)
        with self.assertRaisesRegex(TypeError, "equilibrium_mode"):
            setup_powerlawring().create(
                8, equilibrium_mode=0)
        with self.assertRaisesRegex(ValueError, "too shallow"):
            equilibrium_surface_height(
                np.array([2.0]), DOMAIN_ANNULUS, beta=-0.5,
                softening=10.0)

        setup = setup_powerlawring()
        setup.preglass = True
        nx = 16
        beta, eps = -0.5, 0.05
        setup.create(nx, distri=1, entry="preglass", beta=beta, geometry=3,
                     field_mode=1, softening=eps)
        self.assertEqual(setup.glass_density_table[2], ('r_cyl', 'z'))
        self.assertAlmostEqual(float(np.sum(setup.mass)),
                               equilibrium_target_mass(
                                   DOMAIN_ANNULUS, beta, eps), places=12)
        spacing = 4.0 / nx
        self.assertEqual(setup.npart,
                         round(equilibrium_target_mass(
                             DOMAIN_ANNULUS, beta, eps) / spacing ** 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
