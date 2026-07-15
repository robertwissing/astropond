#!/usr/bin/env bash
# Build GIZMO for the testsuite (code=5 in runtest.sh).
#
# DEPENDENCIES: MPI C compiler (mpicc), GSL, FFTW3 (single + double, --enable-mpi
#   builds), HDF5, HWLOC. Configured by Config.sh (compile-time flags) + a
#   SYSTYPE block in Makefile.systype (set the include/lib paths for this host).
#   Run get_gizmo.sh first.
#
# GIZMO is compiled per physics config. Build a hydro (MFM), an MHD, and a
# self-gravity binary so the suite can pick per test:
#   - GIZMO      : hydro (MFM), Config_hydro.sh
#   - GIZMO_mhd  : + MAGNETIC (MHD), Config_mhd.sh
#   - GIZMO_grav : isolated self-gravity (evrard/collapse), Config_grav.sh
#                  (no BOX_PERIODIC, no SELFGRAVITY_OFF, + ADAPTIVE_GRAVSOFT_FORGAS)
# Each is copied to the active Config.sh before its build. Idempotent (skips an
# existing binary). Provide Config_hydro.sh / Config_mhd.sh / Config_grav.sh and a
# matching SYSTYPE before first build (see Template_Config.sh in the repo).
#
# UNSUPPORTED TESTS: accdisk, coldkeplerian, planet, planetcollision (sinks /
#   central potential not configured); zeldovich, cosmowave (cosmological ICs).
#   MHD tests need GIZMO_mhd. Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${GIZMO_SRC:-$here/gizmo}"
JOBS="${JOBS:-$(nproc)}"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_gizmo.sh first"; exit 1; }

build_one () {  # <config> <exec>
    local cfg="$1" exec="$2"
    if [ -x "$SRC/$exec" ]; then echo "GIZMO already built: $SRC/$exec"; return 0; fi
    [ -f "$SRC/$cfg" ] || { echo "SKIP $exec: $SRC/$cfg missing (copy Template-Config.sh and edit)"; return 0; }
    echo "Building GIZMO from $cfg -> $exec"
    # Build directly to $exec via the Makefile's CONFIG=/EXEC= overrides, so each
    # variant's `clean` (rm $(OBJS) $(EXEC)) only removes its OWN binary -- the
    # other variant (e.g. GIZMO when building GIZMO_mhd) survives.
    ( cd "$SRC" && make clean EXEC="$exec" >/dev/null 2>&1 || true
      make -j"$JOBS" CONFIG="$cfg" EXEC="$exec" )
    echo "Done: $SRC/$exec"
}

build_one Config_hydro.sh GIZMO
build_one Config_mhd.sh   GIZMO_mhd
build_one Config_grav.sh  GIZMO_grav
