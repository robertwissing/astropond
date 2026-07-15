#!/usr/bin/env bash
# Build OpenGadget3 for the testsuite (code=9 in runtest.sh).
#
# DEPENDENCIES: MPI C/C++ compiler (mpicc/mpicxx), GSL, FFTW3 (for TreePM; not used
#   by these box tests but linked), HDF5 (HAVE_HDF5). OpenGadget3 picks per-host
#   compiler/lib paths from a SYSTYPE block: export SYSTYPE=<name> matching a block
#   in the repo Makefile (or its systype include), OR run with SYSTYPE unset to use
#   the repo "Default" block. Run get_opengadget3.sh first.
#
# Built per physics config (like GIZMO): a hydro (TreeSPH) and an MHD (SPMHD) binary
# so the suite can pick per test:
#   - OpenGadget3      : hydro,      Config_hydro.sh
#   - OpenGadget3_mhd  : + MAGNETIC, Config_mhd.sh
# The Config_*.sh shipped beside this script are copied into the source tree and
# built via the Makefile's CONFIG=/EXEC= overrides, so each variant's `clean` only
# removes its OWN binary. Idempotent (skips an existing binary).
#
# UNSUPPORTED TESTS: accdisk, coldkeplerian, planet, planetcollision (sinks /
#   central potential not configured); zeldovich, cosmowave (cosmological ICs).
#   MHD tests (orzag, alfven, currentsheet, mhd*, abcflow, ponomarenko) need
#   OpenGadget3_mhd. Confirm the list with R. Wissing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${OPENGADGET3_SRC:-$here/opengadget3}"
JOBS="${JOBS:-$(nproc)}"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_opengadget3.sh first"; exit 1; }

if [ -z "${SYSTYPE:-}" ]; then
    echo "NOTE: SYSTYPE unset — OpenGadget3 will use its 'Default' block. Set"
    echo "      SYSTYPE=<host> (matching a Makefile block) if the default fails."
fi

build_one () {  # <config-in-here> <exec>
    local cfg_src="$here/$1" exec="$2" cfg_dst="$SRC/$1"
    if [ -x "$SRC/$exec" ]; then echo "OpenGadget3 already built: $SRC/$exec"; return 0; fi
    [ -f "$cfg_src" ] || { echo "SKIP $exec: $cfg_src missing"; return 0; }
    cp "$cfg_src" "$cfg_dst"
    echo "Building OpenGadget3 from $1 -> $exec"
    ( cd "$SRC" && make clean EXEC="$exec" >/dev/null 2>&1 || true
      make -j"$JOBS" CONFIG="$1" EXEC="$exec" )
    echo "Done: $SRC/$exec"
}

build_one Config_hydro.sh OpenGadget3
build_one Config_mhd.sh   OpenGadget3_mhd
build_one Config_grav.sh  OpenGadget3_grav

cat <<'EOF'

OpenGadget3 build step finished.
UNSUPPORTED suite tests (gate in runtest.sh / confirm with R. Wissing):
  - sinks/central-potential: accdisk, coldkeplerian, planet, planetcollision
  - cosmological ICs: zeldovich, cosmowave
  - MHD tests require the OpenGadget3_mhd binary.
EOF
