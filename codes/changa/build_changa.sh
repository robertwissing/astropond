#!/usr/bin/env bash
# Build ChaNGa for the testsuite (code=1 in runtest.sh).
#
# DEPENDENCIES: MPI/network layer via Charm++ (built here, netlrts-linux-x86_64
#   smp), a C/C++ compiler (gcc/g++), gfortran, and the N-BodyShop `utility` repo
#   (STRUCT_DIR). HDF5/FFTW are not required for the tipsy I/O used by the suite.
#   Run get_changa.sh first to fetch changa-public/, utility/ and charm/.
#
# Builds the SMP binary `ChaNGa.smp` and symlinks `charmrun` next to it, matching
# what runtest.sh (CHANGA_DIR) and ci/run_changa_ci_test.sh expect. The build is
# idempotent (skips once ChaNGa.smp exists). This mirrors the working build steps
# in ci/run_changa_ci_test.sh:99-120.
#
# MHD NOTE: public N-BodyShop/ChaNGa lacks the MHD-TP scheme. The MHD tests (orzag,
#   alfven, currentsheet, mhd*, abcflow, ponomarenko) therefore run field-free here
#   and need the robertwissing/changa_uw fork (branch MHD_tp_replication) until the
#   scheme is upstreamed. Override CHANGA_SRC/CHANGA_BRANCH to build the fork.
#
# UNSUPPORTED TESTS (public build): all MHD tests (see note above). Hydro + gravity
#   tests are supported (sedov, kh, blob, evrard, polytrope, gresho, sod/shocktube,
#   noh, rt, taylorgreen, accdisk, planet, …). Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

CHANGA_SRC="${CHANGA_SRC:-$here/changa-public}"
CHARM_SRC="${CHARM_SRC:-$here/charm}"
UTIL_SRC="${UTIL_SRC:-$here/utility}"
CHARM_TARGET="${CHARM_TARGET:-netlrts-linux-x86_64-smp}"
CHANGA_BRANCH="${CHANGA_BRANCH:-master}"   # fork: MHD_tp_replication
JOBS="${JOBS:-$(nproc)}"

[ -d "$CHANGA_SRC" ] || { echo "FATAL: $CHANGA_SRC missing — run get_changa.sh first"; exit 1; }
[ -d "$CHARM_SRC"  ] || { echo "FATAL: $CHARM_SRC missing — run get_changa.sh first";  exit 1; }
[ -d "$UTIL_SRC"   ] || { echo "FATAL: $UTIL_SRC missing — run get_changa.sh first";   exit 1; }

if [ -x "$CHANGA_SRC/ChaNGa.smp" ]; then
    echo "ChaNGa already built: $CHANGA_SRC/ChaNGa.smp"
    exit 0
fi

# 1) Charm++ (SMP target). Skip if already built.
( cd "$CHARM_SRC" && \
  { [ -x "$CHARM_TARGET/bin/charmrun" ] || \
    ./build ChaNGa netlrts-linux-x86_64 smp --with-production --enable-error-checking -j"$JOBS"; } ) \
  || { echo "FATAL: charm build failed"; exit 1; }

export CHARM_DIR="$CHARM_SRC" STRUCT_DIR="$UTIL_SRC/structures"

# 2) ChaNGa. Public master is hydro-only; the fork adds --enable-mhd … (see note).
( cd "$CHANGA_SRC" && git checkout "$CHANGA_BRANCH" 2>/dev/null || true )
CONFIGURE_FLAGS="${CHANGA_CONFIGURE_FLAGS:-}"   # fork: --enable-mhd --enable-mhdcleaning --enable-isph
( cd "$CHANGA_SRC" && ./configure $CONFIGURE_FLAGS && make -j"$JOBS" ) \
  || { echo "FATAL: ChaNGa build failed"; exit 1; }

ln -sf "$CHANGA_SRC/ChaNGa" "$CHANGA_SRC/ChaNGa.smp"
ln -sf "$CHARM_DIR/$CHARM_TARGET/bin/charmrun" "$CHANGA_SRC/charmrun"
echo "Done: $CHANGA_SRC/ChaNGa.smp (+ charmrun symlink)"
