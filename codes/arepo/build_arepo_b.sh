#!/usr/bin/env bash
# Build AREPO for the testsuite (code=3 in runtest.sh).
#
# DEPENDENCIES (SYSTYPE="stornext", see Makefile): MPI C compiler (mpicc; Intel
#   oneAPI MPI here), GSL (-lgsl -lgslcblas), GMP (-lgmp), HDF5 (-lhdf5 -lz),
#   FFTW3, perl (prepare-config.perl). On stornext these live in /usr; on other
#   systems set the *_INCL / *_LIB blocks in the Makefile and Makefile.systype.
#
# Builds the base mesh+MHD binary `Arepo` from Config.sh (MHD + MHD_POWELL +
# RIEMANN_HLLD + HDF5 + double precision). runtest.sh builds the per-box-aspect
# variants (Arepo_asp<LY>_<LZ>) on demand for non-cubic domains, so only the base
# binary is produced here.
#
# UNSUPPORTED TESTS (moving-mesh FV code, no sink/external-potential or cosmology
#   wired in this Config): accdisk, coldkeplerian, planet, planetcollision (need
#   sinks / central potential), zeldovich, cosmowave (comoving integration).
#   MHD tests ARE supported (orzag, alfven, currentsheet, mhdrotor, mhdloop,
#   mhdloopshear, mhdisowave, mhddivtest, mhdbalsaravortex, abcflow, ponomarenko,
#   mhdcollapse). Confirm the exclusion list before gating those.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

CONFIG="${AREPO_CONFIG:-Config.sh}"
EXEC="${AREPO_EXEC:-Arepo}"
BUILD_DIR="${AREPO_BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc)}"

if [ -x "$here/$EXEC" ]; then
    echo "AREPO already built: $here/$EXEC (remove it to force a rebuild)"
    exit 0
fi

echo "Building $EXEC from $CONFIG (SYSTYPE=$(sed 's/SYSTYPE=//' Makefile.systype 2>/dev/null))"
make -j"$JOBS" CONFIG="$CONFIG" BUILD_DIR="$BUILD_DIR" EXEC="$EXEC"
echo "Done: $here/$EXEC"
