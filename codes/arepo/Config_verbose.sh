#!/bin/bash            # this line only there to enable syntax highlighting in this file

## MHD config for the test suite (3D, periodic). Built as a SEPARATE binary
## (Arepo_mhd, build dir build_mhd) so the hydro/mesh-dump Arepo used by the
## sph_interp Voronoi backend (RestartFlag 14, Config.sh) is left untouched:
##   make -j8 CONFIG=Config_mhd.sh BUILD_DIR=build_mhd EXEC=Arepo_mhd
## Modelled on examples/current_sheet_2d (periodic MHD) but 3D, for orzag etc.

#--------------------------------------- Basic operation mode of code
## 3D (no ONEDIMS/TWODIMS), fully PERIODIC (no REFLECTIVE_*; PeriodicBoundariesOn=1
## in the param). Matches the suite's periodic boxes and the sph_interp IC.

#--------------------------------------- Magnetohydrodynamics
MHD                                      # Master switch for magnetohydrodynamics
MHD_POWELL                               # Powell div(B) cleaning scheme
MHD_POWELL_LIMIT_TIMESTEP                # Timestep constraint due to Powell cleaning
RIEMANN_HLLD                             # HLLD approximate Riemann solver (required for MHD)

#--------------------------------------- Mesh motion and regularization
REGULARIZE_MESH_CM_DRIFT                 # Move mesh generating point towards centre of mass
REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED  # Limit regularization speed by local sound speed
REGULARIZE_MESH_FACE_ANGLE               # Use maximum face angle as roundness criterion

#--------------------------------------- Time integration options
TREE_BASED_TIMESTEPS                     # non-local timestep criterion (signal speed)

#---------------------------------------- Single/Double Precision
DOUBLEPRECISION=1                        # full double precision
INPUT_IN_DOUBLEPRECISION                 # initial conditions are in double precision
OUTPUT_VOLUME                            # output cell volumes (density cross-check)
OUTPUT_BFIELD_GRADIENT                   # gradient estimates of the B-field

#--------------------------------------- Output/Input options
HAVE_HDF5                                # HDF5 I/O (MagneticField + MagneticFieldDivergence
                                         # are auto-output under MHD)

#--------------------------------------- Testing and Debugging options
DEBUG                                    # enables core-dumps

VERBOSE                                  # print Powell source-term energy injection per step
