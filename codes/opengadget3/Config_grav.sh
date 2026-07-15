#!/bin/bash
# OpenGadget3 SELF-GRAVITY (TreeSPH) config for the testsuite (-> OpenGadget3_grav).
# For the isolated self-gravity collapse tests (evrard, polytrope): tree self-gravity
# ON (no NOGRAVITY), OPEN/isolated box (no PERIODIC -> no Ewald / no PMGRID), and gas
# gravitational softening that tracks the SPH smoothing length (ADAPTIVE_GRAVSOFT_*).
# G=1 in CODE units is set via GravityConstantInternal=1.0 in the generated
# parameterfile (opengadget3_param.py keys off bDoGravity), NOT physical G. This is
# the OpenGadget3 analogue of GIZMO's Config_grav.sh (SELFGRAVITY on +
# ADAPTIVE_GRAVSOFT_FORGAS). Copy to Config.sh (or pass CONFIG=Config_grav.sh).

# (no PERIODIC)                 # open/isolated box -> tree gravity, no Ewald/PMGRID
# (no NOGRAVITY)                # tree self-gravity ON
HAVE_HDF5
INPUT_IN_DOUBLEPRECISION  # the suite's GADGET HDF5 IC is float64 (Flag_DoublePrecision=1)
OUTPUT_IN_DOUBLEPRECISION # write float64 snapshots (full precision for energy diagnostics)
EXPERT_LEVEL=1           # allow the suite's non-default parameter values
MULTIPLEDOMAINS=4

ARTIFICIAL_CONDUCTIVITY   # Price-Monaghan artificial conductivity (ArtCondConstant)

ADAPTIVE_GRAVSOFT_FORGAS       # variable gravitational softening for gas particles
ADAPTIVE_GRAVSOFT_FORGAS_HSML  # gas softening = SPH smoothing length (needs ADAPTIVE_GRAVSOFT_FORGAS)
