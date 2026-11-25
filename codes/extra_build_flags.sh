#!/usr/bin/env bash
# Campaign/site build additions for managed simulation codes.
#
# This file is loaded automatically by the Gasoline and ChaNGa adapters. Keep
# the arrays empty for their baseline builds; uncomment or add entries for a
# compiler-feature campaign. Changing an effective entry changes the build key.
# The uniform naming convention is <CODE>_EXTRA_BUILD_FLAGS.

# Gasoline entries are preprocessor definitions. The adapter adds -D, so write
# RECONSTRUCTION rather than -DRECONSTRUCTION (both forms are accepted).
# A define that carries a VALUE must be ONE array element written NAME=VALUE:
# bash splits on whitespace, so "NAME VALUE" would become -DNAME -DVALUE, which
# defines NAME as 1 and silently selects the wrong branch of a #if NAME == ...
# MHD and BAROTROPIC are profile-owned and must not be listed here.
# The star-formation magnetic handoff and the evolved magnetic variable live
# here, not on runtest.sh's command line: both are build-only defines like
# every other in this array.
#
#   SFBFIELD      the handoff itself. ON by default because the handoff cases
#                 (starform 6, 7, 9, 10, 11) are about it. COMMENT IT OUT to
#                 build the A/B twin those cases are measured against: the same
#                 stars form and the same gas is deleted either way, so one
#                 subtraction isolates the deposit from the deletion.
#   MHD_B_VAR     W = B rho^-n. Unset is the default B/sqrt(rho) (=1).
#                 MHD_B_VAR=0 -> W = B;  MHD_B_VAR=2 -> W = B/rho.
GASOLINE_EXTRA_BUILD_FLAGS=(
     CHECKQUALITY
     RECONSTRUCTION
     #SFBFIELD
     #MHD_B_VAR=0
     #MHD_B_VAR=2
     #RECONSTRUCTION
     #AV_LAPLACIAN_TURB

     RECONB
     #RECON_DIVGATE
     #RECON_DIVGATE_K=1.0
     #RECON_DIVGATE_RAMP
     #RECONSTRUCTION_LIMITER=SLOPELIM_MINMOD
     MHDRESISTIVITY_ALT
     #DIFFUSION_GENERAL
     #AV_LAPLACIAN
   
     #CORRPARTITION
     #CORRPARTITION_DEBUG
     #CORRPARTITION_SURFACE
     #CORRPARTITION_GRADH_EXACT
     #CORRPARTITION_UPPER_LIMIT
     #CORRPARTITION_ENTROPY_CLEANING
)

# ChaNGa entries are configure feature switches supported by that source tree;
# configure translates them to the actual compiler definitions/options.
# MHD and barotropic are profile-owned. The private MHD-TP profile already
# enables reconstruction, so that entry would be redundant there.
CHANGA_EXTRA_BUILD_FLAGS=(
    # --enable-corrpartition
     --enable-reconstruction
)
