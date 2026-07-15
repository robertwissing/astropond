#!/usr/bin/env bash
# Build PKDGRAV3 for the testsuite (code=8 in runtest.sh).
#
# DEPENDENCIES: CMake >= 3.14, MPI C/C++ compiler, GSL, FFTW3 (with MPI), HDF5
#   (optional), Python3 (the .par parameter files are Python). Run get_pkdgrav3.sh
#   first.
#
# CMake out-of-source build producing the `pkdgrav3` binary under build/.
# Idempotent (skips if the binary exists). PKDGRAV3 reads tipsy ICs natively, so
# no IC format conversion is needed (only a .par parameter file generated from the
# gasoline .param -- see the wiring in runtest.sh).
#
# UNSUPPORTED TESTS: PKDGRAV3 is an N-body + (optional) SPH gravity/cosmology code
#   -- no MHD. Unsupported: all MHD tests (orzag, alfven, currentsheet, mhd*,
#   abcflow, ponomarenko). Gravity/hydro + cosmology tests are the target set.
#   Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${PKDGRAV3_SRC:-$here/pkdgrav3}"
BUILD="${PKDGRAV3_BUILD:-$SRC/build}"
JOBS="${JOBS:-$(nproc)}"
BIN="$BUILD/pkdgrav3"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_pkdgrav3.sh first"; exit 1; }

if [ -x "$BIN" ]; then
    echo "PKDGRAV3 already built: $BIN"
    exit 0
fi

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "${PKDGRAV3_CMAKE_ARGS:-}"
cmake --build "$BUILD" -j"$JOBS"
echo "Done: $BIN"
