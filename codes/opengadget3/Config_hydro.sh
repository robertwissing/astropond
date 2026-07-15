#!/bin/bash
# OpenGadget3 HYDRO (TreeSPH) config for the testsuite (-> OpenGadget3 binary).
# Minimal periodic SPH build with HDF5 I/O. Self-gravity is neutralised at runtime
# via GravityConstantInternal=0 in the generated parameterfile (the suite's
# hydro/MHD box tests are gravity-free), so no gravity-specific flags here.
# Copy this to Config.sh (or pass CONFIG=Config_hydro.sh) before building.

PERIODIC
NOGRAVITY                # suite box tests are gravity-free (like GIZMO SELFGRAVITY_OFF);
                         # also REQUIRED by OpenGadget3 for stretched LONG_X/Y/Z boxes
HAVE_HDF5
INPUT_IN_DOUBLEPRECISION  # the suite's GADGET HDF5 IC is float64 (Flag_DoublePrecision=1)
OUTPUT_IN_DOUBLEPRECISION # write float64 snapshots (full precision for energy diagnostics)
EXPERT_LEVEL=1           # allow the suite's non-default parameter values
MULTIPLEDOMAINS=4

# Time-dependent (Morris-Monaghan) artificial viscosity is left off -> the
# constant ArtBulkViscConst from the parameterfile is used.
ARTIFICIAL_CONDUCTIVITY   # Price-Monaghan artificial conductivity (ArtCondConstant)
