#!/usr/bin/env bash
# Build Gasoline for the testsuite (code=0 in runtest.sh).
#
# DEPENDENCIES: an MPI C compiler (mpicc) or serial gcc, and the standard math
#   libs. Gasoline's tipsy I/O needs no HDF5/FFTW for the suite's tests. Build
#   flags are selected by Makefile targets / -D defines (see the repo's Makefile).
#   Run get_gasoline.sh first.
#
# Gasoline is configured by editing the Makefile flag block then `make <target>`.
# This script builds the default target into `gasoline` unless GASOLINE_TARGET /
# GASOLINE_EXEC are overridden. Idempotent (skips if the binary exists).
#
# MHD NOTE: public Gasoline is hydro+gravity only — there is no `gasolinemhd`
#   here. The suite's MHD reference binary comes from the private gasoline_claude
#   tree; point GASOLINE_DIR (in runtest.sh) at that build for MHD tests.
#
# UNSUPPORTED TESTS (public build): all MHD tests (orzag, alfven, currentsheet,
#   mhd*, abcflow, ponomarenko). Hydro + gravity tests are supported (sedov, kh,
#   blob, evrard, polytrope, gresho, shocktube, noh, rt, accdisk, planet, …).
#   Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

SRC="${GASOLINE_SRC:-$here/gasoline-public}"
TARGET="${GASOLINE_TARGET:-}"          # e.g. a specific Makefile target; empty = default
EXEC="${GASOLINE_EXEC:-gasoline}"
JOBS="${JOBS:-$(nproc)}"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_gasoline.sh first"; exit 1; }

if [ -x "$SRC/$EXEC" ]; then
    echo "Gasoline already built: $SRC/$EXEC"
    exit 0
fi

( cd "$SRC" && make -j"$JOBS" $TARGET ) || { echo "FATAL: Gasoline build failed"; exit 1; }
echo "Done: $SRC/$EXEC"
