#!/bin/bash
# AREPO SELF-GRAVITY (moving-mesh) config for the testsuite (-> Arepo_grav).
# For the isolated self-gravity collapse tests (evrard, polytrope): tree self-gravity
# ON (SELFGRAVITY), OPEN gravity boundaries (GRAVITY_NOT_PERIODIC -> no Ewald), and
# gas gravitational softening that tracks the Voronoi CELL SIZE (ADAPTIVE_HYDRO_
# SOFTENING) -- the moving-mesh analogue of the Gadget codes' ADAPTIVE_GRAVSOFT_FORGAS.
# G=1 in CODE units is set via GravityConstantInternal=1.0 in the generated param.txt
# (arepo_param.py keys off bDoGravity). Pure hydro here (NO MHD) -- evrard is B=0.
#
# NB isolated moving-mesh ICs need mesh generators in the OPEN space around the object
# (else the outer Voronoi cells are unbounded). runtest forces AREPOICMODE=arepo-grid
# for grav so the ADDBACKGROUNDGRID-equivalent background mesh fills the box.

# --- moving mesh (same regularization as the MHD Config, minus MHD) ---
REGULARIZE_MESH_CM_DRIFT                  # move mesh point toward centre of mass
REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED   # limit regularization by local sound speed
REGULARIZE_MESH_FACE_ANGLE               # max face angle roundness criterion
TREE_BASED_TIMESTEPS                     # non-local (signal-speed) timestep criterion

# --- self-gravity (isolated) ---
SELFGRAVITY                              # gravity between cells/particles
GRAVITY_NOT_PERIODIC                     # open gravity boundaries (isolated collapse)
EVALPOTENTIAL                            # compute gravitational potential (energy diagnostics)
OUTPUTACCELERATION                       # output gravitational acceleration
NSOFTTYPES=6                             # one softening class per particle type (param writes SofteningComovingType0..5); >1 enables MULTIPLE_NODE_SOFTENING
MULTIPLE_NODE_SOFTENING                  # tree nodes softened per mass-component softening
ADAPTIVE_HYDRO_SOFTENING                 # gas softening tracks cell size (GasSoftFactor*cellrad)
NSOFTTYPES_HYDRO=64                      # discrete adaptive-softening bins for gas

# --- I/O / precision (same as the MHD Config) ---
DOUBLEPRECISION=1                        # full double precision
INPUT_IN_DOUBLEPRECISION                 # ICs are double precision
OUTPUT_IN_DOUBLEPRECISION                # snapshots double precision
OUTPUT_VOLUME                            # output cell volumes (density cross-check)
HAVE_HDF5                                # HDF5 I/O
DEBUG                                    # core-dumps
