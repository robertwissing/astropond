#!/usr/bin/env bash
# Build POLARIS for the synchrotron export pipeline (POLARIS_EXPORT_PLAN.md M0).
#
# DEPENDENCIES: CMake >= 3.5, a C and C++ compiler, and OpenMP. cfitsio and
#   CCfits ship as tarballs in lib/ and are built here -- nothing external is
#   needed for FITS. Run get_polaris.sh first.
#
# Idempotent (skips if the binary exists; --force to rebuild).
#
# WHY NOT compile.sh: POLARIS's own compile.sh uses bash 4 parameter expansion
#   (${OPTARG,,}) and macOS ships bash 3.2, so it dies on argument parsing before
#   it compiles anything. This script does what compile.sh does -- cfitsio, then
#   CCfits, then src -- driving cmake directly, plus two portability fixes:
#
#   1. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 on the two bundled tarballs. They carry
#      an ancient cmake_minimum_required() that CMake >= 4.0 refuses outright.
#
#   2. OpenMP on macOS. Apple clang has no built-in OpenMP runtime, so libomp
#      must be supplied by hand -- AND not just any libomp: POLARIS compiled by
#      clang >= 17 emits __kmpc_dispatch_deinit, which libomp < 17 does not
#      export, giving an undefined-symbol link failure. This script probes every
#      libomp it can find and picks one that HAS the symbol (on this machine
#      homebrew's is 16.0.2 and fails; the conda glass env's works). The chosen
#      lib's directory goes into CMAKE_INSTALL_RPATH so the binary resolves it at
#      run time without DYLD_LIBRARY_PATH. On Linux/gcc none of this applies and
#      cmake's own FindOpenMP is used unchanged.
#
# VERIFY: --check runs the two shipped reference projects; both must report
#   "SUCCESSFULLY FINISHED".
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

ROOT="${POLARIS_SRC:-$here/POLARIS}"
BIN="$ROOT/bin/polaris"
JOBS="${JOBS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4 )}"
FORCE=0
CHECK=0
for a in "$@"; do
    case "$a" in
        --force) FORCE=1 ;;
        --check) CHECK=1 ;;
        *) echo "usage: $0 [--force] [--check]"; exit 2 ;;
    esac
done

[ -d "$ROOT" ] || { echo "FATAL: $ROOT missing — run get_polaris.sh first"; exit 1; }

if [ -x "$BIN" ] && [ "$FORCE" -eq 0 ] && [ "$CHECK" -eq 0 ]; then
    echo "POLARIS already built: $BIN"
    exit 0
fi

# --------------------------------------------------------------------------- #
#  OpenMP: only macOS/AppleClang needs the manual hunt (see note 2 above)
# --------------------------------------------------------------------------- #
OMP_ARGS=()
if [ "$(uname -s)" = "Darwin" ] && [ -z "${POLARIS_NO_OMP_HUNT:-}" ]; then
    # Candidates in preference order; POLARIS_LIBOMP overrides the whole search.
    cands=()
    [ -n "${POLARIS_LIBOMP:-}" ] && cands+=("$POLARIS_LIBOMP")
    for p in "${CONDA_PREFIX:-}" "$HOME/anaconda3/envs/glass" \
             "$HOME/miniconda3/envs/glass" "$HOME/anaconda3" "$HOME/miniconda3" \
             /opt/homebrew/opt/libomp /usr/local/opt/libomp; do
        [ -n "$p" ] && cands+=("$p/lib/libomp.dylib")
    done

    omp_lib=""
    for c in "${cands[@]}"; do
        [ -f "$c" ] || continue
        # The symbol clang >= 17 needs. `nm -gU` lists exported (defined) symbols.
        if nm -gU "$c" 2>/dev/null | grep -q '__kmpc_dispatch_deinit'; then
            omp_lib="$c"; break
        fi
        echo "  libomp too old (no __kmpc_dispatch_deinit): $c"
    done

    if [ -z "$omp_lib" ]; then
        echo "FATAL: no usable libomp found. POLARIS needs one exporting"
        echo "       __kmpc_dispatch_deinit (libomp >= 17). Tried:"
        printf '         %s\n' "${cands[@]}"
        echo "       Set POLARIS_LIBOMP=/path/to/libomp.dylib, or install a"
        echo "       newer libomp (brew upgrade libomp / conda install llvm-openmp)."
        exit 1
    fi

    omp_dir="$(cd "$(dirname "$omp_lib")" && pwd)"
    omp_inc="$(dirname "$omp_dir")/include"
    echo "  OpenMP: $omp_lib"
    OMP_ARGS=(
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$omp_inc"
        -DOpenMP_CXX_LIB_NAMES="omp"
        -DOpenMP_omp_LIBRARY="$omp_lib"
        -DCMAKE_INSTALL_RPATH="$omp_dir;$ROOT/lib/lib"
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    )
fi

# Bundled tarballs predate CMake 4's minimum-policy floor.
OLDCM=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release)

if [ ! -x "$BIN" ] || [ "$FORCE" -eq 1 ]; then
    echo "=================== 1/3 cfitsio ==================="
    cd "$ROOT/lib"
    rm -rf cfitsio CCfits
    tar -xf cfitsio.tar.gz
    cmake -S cfitsio -B cfitsio/build "${OLDCM[@]}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$ROOT/lib" \
        -DCMAKE_C_FLAGS="-w -O3"
    cmake --build cfitsio/build --parallel "$JOBS"
    cmake --install cfitsio/build

    echo "=================== 2/3 CCfits ==================="
    cd "$ROOT/lib"
    tar -xf CCfits.tar.gz
    cmake -S CCfits -B CCfits/build "${OLDCM[@]}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_PREFIX_PATH="$ROOT/lib" \
        -DCMAKE_INSTALL_PREFIX="$ROOT/lib" \
        -DCMAKE_CXX_FLAGS="-w -O3"
    cmake --build CCfits/build --parallel "$JOBS"
    cmake --install CCfits/build

    echo "=================== 3/3 polaris ==================="
    cmake -S "$ROOT/src" -B "$ROOT/src/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$ROOT" \
        -DCMAKE_PREFIX_PATH="$ROOT/lib" \
        -DCMAKE_CXX_FLAGS="-w -O3" \
        "${OMP_ARGS[@]}"
    cmake --build "$ROOT/src/build" --parallel "$JOBS"
    cmake --install "$ROOT/src/build"

    [ -x "$BIN" ] || { echo "FATAL: build finished but $BIN is missing"; exit 1; }
    echo "Done: $BIN"
fi

# --------------------------------------------------------------------------- #
#  --check: the two shipped reference projects
# --------------------------------------------------------------------------- #
if [ "$CHECK" -eq 1 ]; then
    echo "=================== reference runs ==================="
    # The shipped .cmd files reference their grid/output with paths RELATIVE to
    # the POLARIS root, so run from there or every one of them fails on I/O.
    cd "$ROOT"
    fail=0
    for proj in stellar_sed reemission_sphere; do
        cmd="projects/test/$proj/POLARIS.cmd"
        [ -f "$cmd" ] || { echo "  SKIP $proj (no $ROOT/$cmd)"; continue; }
        if "$BIN" "$cmd" 2>&1 | tail -40 | grep -q "SUCCESSFULLY FINISHED"; then
            echo "  PASS $proj"
        else
            echo "  FAIL $proj"; fail=1
        fi
    done
    [ "$fail" -eq 0 ] || { echo "reference runs FAILED"; exit 1; }
    echo "reference runs OK"
fi
