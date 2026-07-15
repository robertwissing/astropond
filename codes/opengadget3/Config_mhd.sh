#!/bin/bash
# OpenGadget3 MHD (SPMHD) config for the testsuite (-> OpenGadget3_mhd binary).
# Periodic SPH + induction-equation magnetic fields. The magnetic-block flags are
# chosen to match the parameterfile our generator writes (and the project's own
# tests/MHDTest/paramfile_wc6.par), i.e. the Tricco divB cleaning + artificial
# magnetic dissipation tags:
#   ARTIFICIAL_MAGNETIC_DISSIPATION -> ArtificialMagneticDissipationConstant
#   DIVBCLEANING_TRICCO             -> DivBcleaningParabolicSigma, DivBOverCleanFac
#   ARTIFICIAL_CONDUCTIVITY         -> ArtCondConstant
# Keep this set in sync with sph_interp/_opengadget3/opengadget3_param.py.

PERIODIC
NOGRAVITY                # suite box tests are gravity-free (like GIZMO SELFGRAVITY_OFF);
                         # also REQUIRED by OpenGadget3 for stretched LONG_X/Y/Z boxes
HAVE_HDF5
INPUT_IN_DOUBLEPRECISION  # the suite's GADGET HDF5 IC is float64 (Flag_DoublePrecision=1)
OUTPUT_IN_DOUBLEPRECISION # write float64 snapshots (full precision for energy diagnostics)
EXPERT_LEVEL=1           # allow the suite's non-default parameter values
MULTIPLEDOMAINS=4

#--------------------------------------- Magnetic fields (SPMHD)
MAGNETIC                       # B field + induction equation (writes MagneticField
                               #   + DivergenceOfMagneticField into snapshots)
# NB: NO MU0_UNITY. Like GIZMO/AREPO, this code runs in Gaussian B units
# (mu0_inv=1/4pi). To stay consistent across the three Gadget-style codes, the
# tipsy->code converters instead write the IC B in GAUSS (multiply the mu0=1 code
# B by sqrt(4pi)) and the reverse converters divide by sqrt(4pi). See
# sph_interp/export.MU0_CODE_TO_GAUSS and ICconvert_tipsyto{gizmo,arepo,opengadget3}.
ARTIFICIAL_MAGNETIC_DISSIPATION
DIVBCLEANING_TRICCO            # hyperbolic/parabolic divB cleaning (Tricco & Price)
ARTIFICIAL_CONDUCTIVITY        # Price-Monaghan artificial conductivity
