#!/usr/bin/env bash
# Build Phantom for the testsuite (code=6 in runtest.sh).
#
# DEPENDENCIES: a Fortran compiler (gfortran), GNU make, optional MPI/OpenMP.
#   Phantom selects physics at compile time via SETUP=<name> (and SYSTEM=<host>).
#   Run get_phantom.sh first.
#
# Phantom builds `phantom` (evolve) + `phantomsetup` (IC) per SETUP. Build a hydro,
# an MHD, and a self-gravity setup so the suite can pick per test (RUNPHYSICS):
#   - phantom / phantomsetup         : SETUP=$PHANTOM_SETUP       (hydro)
#   - phantom_mhd / phantomsetup_mhd  : SETUP=$PHANTOM_SETUP_MHD  (MHD)
#   - phantom_grav / phantomsetup_grav: SETUP=$PHANTOM_SETUP_GRAV (self-gravity)
# The grav SETUP (default `evrard`) compiles GRAVITY=yes, adiabatic, NON-periodic
# (isolated tree gravity, no Ewald) -- exactly the evrard/polytrope collapse needs.
# Only the evolve binary's COMPILE flags matter (we supply our own full-dump IC via
# ICconvert_tipsytophantom.py, which sets G_code=1 units); phantomsetup is unused.
# Idempotent (skips an existing binary). Binaries are placed under bin/.
#
# UNSUPPORTED TESTS: grid/cosmology tests (zeldovich, cosmowave); confirm which of
#   the suite's setups map to a Phantom SETUP (Phantom is SPH so most hydro/MHD
#   tests are feasible, but each needs a SETUP + the tipsy->phantom converter).
#   Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${PHANTOM_SRC:-$here/phantom}"
JOBS="${JOBS:-$(nproc)}"
SYSTEM="${PHANTOM_SYSTEM:-gfortran}"
SETUP="${PHANTOM_SETUP:-sedov}"
SETUP_MHD="${PHANTOM_SETUP_MHD:-orstang}"   # Orszag-Tang setup name in Phantom
# Self-gravity build: Phantom's `evrard` SETUP = setup_star.f90 with GRAVITY=yes,
# adiabatic, non-periodic (Makefile_setups). Compile flags are all we use.
SETUP_GRAV="${PHANTOM_SETUP_GRAV:-evrard}"
# MPI flavour. The suite launches `phantom` under `mpirun` (runtest.sh), so the
# binary MUST be a real MPI build — otherwise mpirun forks N independent serial
# copies that race on the same dump files. Phantom's `MPI=yes` hardcodes OpenMPI's
# `mpif90 --showme:compile` syntax, which our Intel oneAPI `mpif90` (a gfortran
# wrapper) does not understand. `MPI=intel` sets FC=mpif90 plainly (no --showme),
# which is the correct flavour for this toolchain. Set MPI=no for a serial binary
# (then runtest.sh must launch without mpirun).
MPI="${PHANTOM_MPI:-intel}"

[ -f "$SRC/build/Makefile" ] || { echo "FATAL: $SRC/build/Makefile missing — run get_phantom.sh first"; exit 1; }

# Set PHANTOM_REBUILD=1 to force a rebuild (e.g. switching MPI flavour): removes the
# suffixed binaries so the idempotent skip below doesn't keep a stale serial build.
REBUILD="${PHANTOM_REBUILD:-0}"

build_one () {  # <setup> <suffix>
    local setup="$1" sfx="$2"
    if [ "$REBUILD" = 1 ]; then rm -f "$SRC/bin/phantom$sfx" "$SRC/bin/phantomsetup$sfx"; fi
    if [ -x "$SRC/bin/phantom$sfx" ]; then echo "Phantom already built: $SRC/bin/phantom$sfx"; return 0; fi
    echo "Building Phantom SETUP=$setup SYSTEM=$SYSTEM MPI=$MPI -> phantom$sfx"
    # Build 'phantom' (evolve) and 'setup' (phantomsetup) as SEPARATE, sequential
    # make goals. Passing them together ("make phantom setup") makes Phantom's parent
    # Makefile fire BOTH its 'phantom:' rule and its '%::' catch-all, spawning two
    # concurrent sub-makes in the same build/ dir that race on the shared .mod/.o
    # files ("Cannot open module file 'dim.mod'") and a bogus 'phantom setup' target.
    #
    # Build SERIALLY (no -j): Phantom's build/Makefile only guarantees correct module
    # ordering through the SRCFILES order — parallel (-j) builds need a generated
    # .depends and otherwise race ("Cannot open module file 'dim.mod'/'io.mod'").
    ( cd "$SRC" && make clean >/dev/null 2>&1 || true
      make SETUP="$setup" SYSTEM="$SYSTEM" MPI="$MPI" phantom
      make SETUP="$setup" SYSTEM="$SYSTEM" MPI="$MPI" setup
      mkdir -p bin
      mv bin/phantom      "bin/phantom$sfx"      2>/dev/null || true
      mv bin/phantomsetup "bin/phantomsetup$sfx" 2>/dev/null || true )
    echo "Done: $SRC/bin/phantom$sfx"
}

# Build the SUFFIXED variants (_mhd, _grav) FIRST and the un-suffixed `phantom` LAST.
# Each build_one runs `make clean` (required between different SETUPs), and Phantom's
# clean does `rm -f bin/phantom bin/phantomsetup` — the UN-suffixed names. Building a
# suffixed variant last would let its clean delete the already-built hydro `phantom`;
# this order means the final clean only targets names that don't exist yet. (The
# suffixed binaries survive each other's clean because clean only hits unsuffixed.)
build_one "$SETUP_MHD"  "_mhd"
build_one "$SETUP_GRAV" "_grav"
build_one "$SETUP"      ""
