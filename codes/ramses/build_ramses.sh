#!/usr/bin/env bash
# Build RAMSES for the testsuite (code=4 in runtest.sh).
#
# DEPENDENCIES: a Fortran compiler (gfortran) and GNU make. No HDF5 needed for
#   native RAMSES output. Run get_ramses.sh first.
#   MPI: if an mpif90 wrapper is on PATH (must wrap gfortran — COMPILER=GNU;
#   the Intel oneAPI mpif90 does), *_mpi twins of both binaries are built too
#   (MPI=1 MPIF90=mpif90). runtest.sh selects them via RAMSESNP>1. Serial
#   binaries remain the default runtime (RAMSESNP unset/1); MPI is needed for
#   the 128^3+ MHD convergence runs (~16x the 64^3 cost, see
#   MHD_OT_ENERGY_PLAN.md).
#
# RAMSES is configured per-build via make variables (NDIM, SOLVER, PATCH, EXEC).
# Three artifacts:
#   - ramses3d      : SOLVER=hydro, PATCH=patches/suite (our gridic condinit —
#                     WITHOUT the patch, condinit_kind='gridic' silently falls
#                     through to the default region IC).
#   - ramses3d_mhd  : SOLVER=mhd, PATCH=patches/suite_mhd (gridic condinit with
#                     the MHD q layout + 6 face-B slots; a separate patch dir
#                     because VPATH puts PATCH first and the hydro condinit.f90
#                     would otherwise SHADOW mhd/condinit.f90 and compile the
#                     wrong q layout).
#   - utils/f90/amr2cell : reverse-converter helper (leaf cells -> ASCII rows).
#                     NO Makefile target in utils/f90 — compiled explicitly.
# Idempotent (skips artifacts that already exist). Override via RAMSES_* env vars.
#
# UNSUPPORTED TESTS (grid AMR code, no sinks/central potential or SPH-style setups
#   wired): accdisk, coldkeplerian, planet, planetcollision; cosmology tests
#   (zeldovich, cosmowave) need RAMSES cosmo + grafic ICs, out of scope here.
#   Non-cubic (slab) boxes — kh, orzag — need a non-cubic GridIC strategy first.
#   Gravity tests (mhdcollapse, evrard, polytrope) need poisson=.true. wiring.
#   MHD: only cubic-box MHD tests (orzag -a 1); the 2D orzag slabs stay
#   unsupported (incommensurate z).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${RAMSES_SRC:-$here/ramses}"
PATCH_DIR="${RAMSES_PATCH:-$here/patches/suite}"
PATCH_DIR_MHD="${RAMSES_PATCH_MHD:-$here/patches/suite_mhd}"
# RAMSES's bin/Makefile has no .mod inter-file dependencies: parallel make races
# ("Cannot open module file amr_parameters.mod") — keep the default serial.
JOBS="${JOBS:-1}"
NDIM="${RAMSES_NDIM:-3}"

[ -d "$SRC/bin" ] || { echo "FATAL: $SRC/bin missing — run get_ramses.sh first"; exit 1; }
[ -f "$PATCH_DIR/condinit.f90" ] || { echo "FATAL: $PATCH_DIR/condinit.f90 missing"; exit 1; }
[ -f "$PATCH_DIR_MHD/condinit.f90" ] || { echo "FATAL: $PATCH_DIR_MHD/condinit.f90 missing"; exit 1; }

build_one () {  # <solver> <patch(empty=stock)> <exec_out> [mpi(0|1)]
    local solver="$1" patch="$2" out="$3" mpi="${4:-0}"
    if [ -x "$SRC/bin/$out" ]; then echo "RAMSES already built: $SRC/bin/$out"; return 0; fi
    # distinct EXEC per build so the link target never clobbers a sibling binary
    local exec_base="ramses_${solver}"
    [ "$mpi" = 1 ] && exec_base="ramses_${solver}_mpi"
    echo "Building RAMSES NDIM=$NDIM SOLVER=$solver MPI=$mpi PATCH='${patch:-<stock>}' -> $out"
    ( cd "$SRC/bin" && make clean >/dev/null 2>&1 || true
      make -j"$JOBS" NDIM="$NDIM" SOLVER="$solver" PATCH="$patch" EXEC="$exec_base" \
           MPI="$mpi" MPIF90="$MPIF90" )
    mv "$SRC/bin/${exec_base}${NDIM}d" "$SRC/bin/$out"
    [ -x "$SRC/bin/$out" ] || { echo "FATAL: build produced no $out"; exit 1; }
    echo "Done: $SRC/bin/$out"
}

MPIF90="${RAMSES_MPIF90:-mpif90}"

build_one hydro "$PATCH_DIR"     "ramses${NDIM}d"
build_one mhd   "$PATCH_DIR_MHD" "ramses${NDIM}d_mhd"

# MPI twins (same patches; skipped when no mpif90 wrapper is available)
if command -v "$MPIF90" >/dev/null 2>&1; then
    build_one hydro "$PATCH_DIR"     "ramses${NDIM}d_mpi"     1
    build_one mhd   "$PATCH_DIR_MHD" "ramses${NDIM}d_mhd_mpi" 1
else
    echo "NOTE: no '$MPIF90' on PATH — skipping the MPI builds (serial only)."
fi

# amr2cell (reverse converter): no target in utils/f90/Makefile — compile directly.
if [ -x "$SRC/utils/f90/amr2cell" ]; then
    echo "amr2cell already built: $SRC/utils/f90/amr2cell"
else
    echo "Building utils/f90/amr2cell"
    # -fallow-argument-mismatch: hilbert3d is called with a scalar order arg
    # (rank mismatch), a hard error since gfortran 10 in this old utils code.
    ( cd "$SRC/utils/f90" && gfortran -O2 -fallow-argument-mismatch -o amr2cell amr2cell.f90 )
    echo "Done: $SRC/utils/f90/amr2cell"
fi

echo ""
echo "RAMSES build complete:"
echo "  $SRC/bin/ramses${NDIM}d        (hydro, gridic condinit patch)"
echo "  $SRC/bin/ramses${NDIM}d_mhd    (MHD, gridic condinit patch suite_mhd)"
[ -x "$SRC/bin/ramses${NDIM}d_mpi" ] && \
echo "  $SRC/bin/ramses${NDIM}d_mpi / ramses${NDIM}d_mhd_mpi  (MPI twins; select via RAMSESNP>1)"
echo "  $SRC/utils/f90/amr2cell        (reverse converter helper)"
echo "UNSUPPORTED: accdisk coldkeplerian planet planetcollision zeldovich cosmowave;"
echo "  non-cubic boxes (kh, 2D orzag slabs); gravity tests (mhdcollapse, evrard,"
echo "  polytrope) until poisson wiring. MHD supported for cubic boxes (orzag -a 1)."
