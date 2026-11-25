#!/usr/bin/env python3
"""Pure checks for per-analysis runtest.sh runname_extra decoders."""

import importlib
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch


HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.dirname(HERE)
sys.path.insert(0, SETUP)

from IC_analysis_framework import build_parser, params_from_args  # noqa: E402


RUN_NAMES = [
    ("abcflow", "abcflow32_N64glassB0.002amp1.5GASOLINE",
     {"B0": 0.002, "amp": 1.5}),
    ("alfven", "alfven32_N64glassrot0GASOLINE", {"rotate": 0}),
    ("blob", "blob32_N64glassrhodiff12beta0.5mach3inflow1cool1wr1wd25tc5000GASOLINE",
     {"rhodiff": 12., "beta": 0.5, "mach": 3., "inflow": 1.,
      "cool": 1., "windref": 1., "winddens": 25., "tcloud": 5000.}),
    ("coldkeplerian", "coldkeplerian32_N64glasshr0.04ri0.8ro2.5q0.6GASOLINE",
     {"hr": 0.04, "rinner": 0.8, "rdisk": 2.5, "q": 0.6}),
    ("cosmowave", "cosmowave32_N64glasscase0A2G0.5u1e-3GASOLINE",
     {"case": 0, "factor_A": 2., "factor_G": 0.5, "A_u": 1.e-3}),
    ("currentsheet", "currentsheet32_N64glassbeta0.2v0.05GASOLINE",
     {"beta": 0.2, "v0": 0.05}),
    ("disstest", "disstest32_N64glasscase3c1a0.2b1v1GASOLINE",
     {"case": 3, "cosmo": 1, "amp": 0.2, "bfac": 1., "variant": 1}),
    ("gresho", "gresho32_N64glassmach0.3GASOLINE", {"mach": 0.3}),
    ("isentropicvortex", "isentropicvortex32_N64glassb4vadv2GASOLINE",
     {"beta_v": 4., "vadv": 2.}),
    ("kh", "kh32_N64glassrhodiff3mach0.6smooth0B0.25Bdir2GASOLINE",
     {"rhodiff": 3., "mach": 0.6, "smooth": 0., "B0": 0.25,
      "Bdir": 2.}),
    ("mhdcollapse", "mhdcollapse32_N64glassmu5rd400Erat0.03GASOLINE",
     {"mu": 5., "rhodiff": 400., "Erat": 0.03}),
    ("mhddivtest", "mhddivtest32_N64glassrd2s0sq1cos1GASOLINE",
     {"rhodiff": 2., "smooth": 0, "square": 1, "cosmo": 1}),
    ("mhdloop", "mhdloop32_N64glassrhodiff2GASOLINE", {"rhodiff": 2.}),
    ("mhdrotor", "mhdrotor32_N64glassrhodiff8GASOLINE", {"rhodisk": 8.}),
    ("mhdisowave", "mhdisowave32_N64glassbeta0.25GASOLINE", {"beta": 0.25}),
    ("noh", "noh32_N64glassd30p2e-6v-2GASOLINE",
     {"is3d": 0, "pini": 2.e-6, "v0": 2.}),
    ("orzag", "orzag32_N64glass_3dGASOLINE", {"dim3": 1.}),
    ("ponomarenko", "ponomarenko32_N64glassB0.2GASOLINE", {"B0": 0.2}),
    ("powerlawring",
     "powerlawring256_N84glassbeta-0.5geom2fm0pb1e6eps0.05tilt0dom0"
     "ri0.5wi0.25ro2wo2.5H0.001form0NEWTESTab0fch0a05bGASOLINE-A",
     {"beta": -0.5, "geometry": 2, "field_mode": 0,
      "plasma_beta": 1.e6, "softening": 0.05, "tilt": 0.,
      "domain_mode": 0, "scale_height": 0.001, "force_form": 0}),
    # Hopkins ring: it emits the sharp R=0.5..2 ring, so it carries the ring's
    # radial tag rather than a taper's support radii.
    ("powerlawring",
     "powerlawring256_N84glassbeta-0.5geom2fm0pb1e6eps0tilt0dom0"
     "ri0.5ro2P2d1e-6form1GASOLINE-A",
     {"beta": -0.5, "geometry": 2, "field_mode": 0,
      "plasma_beta": 1.e6, "softening": 0., "tilt": 0.,
      "domain_mode": 0, "scale_height": 0.15, "force_form": 1}),
    ("rt", "rt32_N64glassrhodiff3smooth0.1nlow4c2beta5bdir3GASOLINE",
     {"rhodiff": 3., "case": 2, "nlowdens": 4., "smooth": 0.1,
      "beta": 5., "bdir": 3}),
    ("sedov", "sedov32_N64glassselect0beta2ublast12Rin0.2Pin80Pout0.5GASOLINE",
     {"select": 0, "betain": 2., "ublast": 12., "Rin": 0.2,
      "Pin": 80., "Pout": 0.5}),
    ("shocktube", "shocktube32_N64glassBrio-Wu_Ryu_5aGASOLINE",
     {"choice": 6}),
    ("square", "square32_N64glasschi6P3g1.4d31GASOLINE",
     {"rhodiff": 6., "przero": 3., "gamma": 1.4, "is3d": 1}),
    ("taylorgreen", "taylorgreen32_N64glasseta0.03xi0.02GASOLINE",
     {"etavisc": 0.03, "xivisc": 0.02}),
    ("wave", "wave32_N64glasswt2a2e-4GASOLINE",
     {"wavetype": 2, "ampl": 2.e-4}),
    ("zeldovich", "zeldovich32_N64glassL500zi49GASOLINE",
     {"Lboxinkpc": 500., "zi": 49.}),
]


class TestRunnameParams(unittest.TestCase):
    def test_representative_runtest_names_decode(self):
        for test_name, run_name, expected in RUN_NAMES:
            with self.subTest(test=test_name):
                module = importlib.import_module(f"IC_analysis_{test_name}")
                parser = build_parser(module.PARAM_SPEC)
                with patch("sys.stderr", new_callable=StringIO):
                    args = parser.parse_args([run_name])
                    params = params_from_args(args, module.PARAM_SPEC)
                for key, expected_value in expected.items():
                    if isinstance(expected_value, float):
                        self.assertAlmostEqual(params[key], expected_value)
                    else:
                        self.assertEqual(params[key], expected_value)

    def test_cosmowave_custom_suffix_does_not_fake_case9_variant(self):
        module = importlib.import_module("IC_analysis_cosmowave")
        parser = build_parser(module.PARAM_SPEC)
        run_name = ("cosmowave32_N64glasscase4A1G0u0.1f2g0"
                    "testingv6afacGASOLINE")
        with patch("sys.stderr", new_callable=StringIO):
            args = parser.parse_args([run_name])
            params = params_from_args(args, module.PARAM_SPEC)
        self.assertEqual(params["case"], 4)
        self.assertEqual(params["clean_fch"], 2.)
        self.assertEqual(params["clean_damping"], 0.)
        self.assertEqual(params["diss_variant"], 0)

    def test_disstest_conditional_tags_decode_only_in_their_layout(self):
        module = importlib.import_module("IC_analysis_disstest")
        parser = build_parser(module.PARAM_SPEC)
        run_name = "disstest32_N64glasscase1c1a0.1b1k0.04ai0.1s0.2GASOLINE"
        with patch("sys.stderr", new_callable=StringIO):
            args = parser.parse_args([run_name])
            params = params_from_args(args, module.PARAM_SPEC)
        self.assertEqual(params["case"], 1)
        self.assertAlmostEqual(params["a_init"], 0.1)
        self.assertAlmostEqual(params["strat"], 0.2)
        self.assertEqual(params["variant"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
