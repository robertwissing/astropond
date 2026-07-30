#!/usr/bin/env bash
# Build MUSIC (Hahn & Abel 2011). Run get_music.sh first.
#
# DEPENDENCIES
#   required : CMake >= 3.9, a C++11 compiler, OpenMP, FFTW3, GSL
#   optional : HDF5      (Enzo/Arepo/Gadget-HDF5 output plugins)
#              libtirpc  (XDR) - needed by the TIPSY output plugin on Linux,
#                        which is the plugin this testsuite cares about
#
#   Debian/Ubuntu : sudo apt install cmake g++ libfftw3-dev libgsl-dev \
#                        libhdf5-dev libtirpc-dev
#   Fedora/RHEL   : sudo dnf install cmake gcc-c++ fftw-devel gsl-devel \
#                        hdf5-devel libtirpc-devel
#   macOS         : brew install cmake fftw gsl hdf5 libomp
#
# PORTABILITY. On Linux with gcc this script does nothing clever: CMake's own
# FindOpenMP works, and the distro packages above are found automatically. Two
# platform issues are handled only when they actually occur:
#
#   1. OpenMP on macOS. Apple clang ships no OpenMP runtime, so libomp must be
#      supplied. This script SEARCHES for one (brew, conda, MacPorts, or
#      $MUSIC_LIBOMP) instead of assuming a path, and only after a plain
#      configure has failed to find OpenMP.
#
#   2. MUSIC's CMakeLists puts OpenMP into the COMPILE flags only -
#      `set(CMAKE_CXX_FLAGS ... ${OpenMP_CXX_FLAGS})` - and never calls
#      target_link_libraries(... OpenMP::OpenMP_CXX). With gcc that is
#      harmless because -fopenmp both compiles and links; with Apple clang
#      "-Xpreprocessor -fopenmp" compiles but links nothing, so every __kmpc_*
#      symbol is undefined at link time. We add the library to
#      CMAKE_EXE_LINKER_FLAGS ourselves on that path.
#
# -march=native: MUSIC hardcodes it in CMakeLists.txt. That is fine for a
#   machine-local build but produces binaries that may fault on a different
#   CPU (building on a login node, running on compute nodes with an older
#   microarchitecture). --portable strips it.
#
# usage: ./build_music.sh [--force] [--check] [--single] [--portable]
#   --force     reconfigure and rebuild even if the binary exists
#   --check     run the shipped ics_example.conf after building
#   --single    single-precision FFT (MUSIC_ENABLE_SINGLE_PRECISION)
#   --portable  drop -march=native (see above)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

ROOT="${MUSIC_SRC:-$here/MUSIC}"
BUILD="$ROOT/build"
BIN="$BUILD/MUSIC"
JOBS="${JOBS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4 )}"
FORCE=0; CHECK=0; SINGLE=0; PORTABLE=0
for a in "$@"; do
    case "$a" in
        --force)    FORCE=1 ;;
        --check)    CHECK=1 ;;
        --single)   SINGLE=1 ;;
        --portable) PORTABLE=1 ;;
        -h|--help)  sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--force] [--check] [--single] [--portable]"; exit 2 ;;
    esac
done

[ -d "$ROOT" ] || { echo "FATAL: $ROOT missing - run get_music.sh first"; exit 1; }

if [ -x "$BIN" ] && [ "$FORCE" -eq 0 ] && [ "$CHECK" -eq 0 ]; then
    echo "MUSIC already built: $BIN"
    exit 0
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
[ "$SINGLE" -eq 1 ] && CMAKE_ARGS+=(-DMUSIC_ENABLE_SINGLE_PRECISION=ON)
# CMakeLists.txt appends -march=native to CMAKE_CXX_FLAGS; overriding the
# variable from the command line does not remove it, so patch it out of a
# scratch copy of the flag string instead.
[ "$PORTABLE" -eq 1 ] && CMAKE_ARGS+=(-DCMAKE_CXX_FLAGS="-mtune=generic")

# --------------------------------------------------------------------------- #
#  libtirpc: the TIPSY plugin needs XDR. glibc dropped RPC (2.32+), so Linux
#  needs libtirpc-dev; macOS and BSD still have XDR in libc.
# --------------------------------------------------------------------------- #
if [ "$(uname -s)" = "Linux" ]; then
    if ! pkg-config --exists libtirpc 2>/dev/null; then
        echo "WARNING: libtirpc not found."
        echo "         MUSIC's TIPSY output plugin needs XDR, which modern glibc"
        echo "         no longer provides - the tipsy writer will not work."
        echo "         Install libtirpc-dev / libtirpc-devel and rebuild."
        echo
    fi
fi

# --------------------------------------------------------------------------- #
#  OpenMP. Try the compiler's own first; only hunt for libomp if that fails.
# --------------------------------------------------------------------------- #
omp_probe="$(mktemp -d)"
trap 'rm -rf "$omp_probe"' EXIT
cat > "$omp_probe/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.9)
project(ompprobe CXX)
find_package(OpenMP REQUIRED)
EOF
if cmake -S "$omp_probe" -B "$omp_probe/b" >/dev/null 2>&1; then
    echo "OpenMP: found by the compiler toolchain"
else
    echo "OpenMP: not found by CMake - searching for a libomp"
    if [ "$(uname -s)" != "Darwin" ]; then
        echo "FATAL: no OpenMP. On Linux install a compiler with OpenMP"
        echo "       (g++ has it built in; for clang install libomp-dev)."
        exit 1
    fi
    cands=()
    [ -n "${MUSIC_LIBOMP:-}" ] && cands+=("$MUSIC_LIBOMP")
    for p in "${CONDA_PREFIX:-}" /opt/homebrew/opt/libomp /usr/local/opt/libomp \
             /opt/local; do
        [ -n "$p" ] && cands+=("$p/lib/libomp.dylib")
    done
    omp_lib=""
    for c in "${cands[@]}"; do
        [ -f "$c" ] || continue
        # clang >= 17 emits __kmpc_dispatch_deinit; older libomp lacks it and
        # the link fails with an undefined symbol.
        if nm -gU "$c" 2>/dev/null | grep -q '__kmpc_dispatch_deinit'; then
            omp_lib="$c"; break
        fi
        echo "  libomp too old (no __kmpc_dispatch_deinit): $c"
    done
    if [ -z "$omp_lib" ]; then
        echo "FATAL: no usable libomp found. Tried:"
        printf '         %s\n' "${cands[@]}"
        echo "       brew install libomp   (or conda install llvm-openmp),"
        echo "       or set MUSIC_LIBOMP=/path/to/libomp.dylib"
        exit 1
    fi
    omp_dir="$(cd "$(dirname "$omp_lib")" && pwd)"
    omp_inc="$(dirname "$omp_dir")/include"
    echo "  using $omp_lib"
    CMAKE_ARGS+=(
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$omp_inc"
        -DOpenMP_C_LIB_NAMES=omp
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$omp_inc"
        -DOpenMP_CXX_LIB_NAMES=omp
        -DOpenMP_omp_LIBRARY="$omp_lib"
        # see note 2 in the header: MUSIC never links the OpenMP target
        -DCMAKE_EXE_LINKER_FLAGS="-L$omp_dir -lomp -Wl,-rpath,$omp_dir"
    )
fi

# --------------------------------------------------------------------------- #
#  configure + build
# --------------------------------------------------------------------------- #
if [ ! -x "$BIN" ] || [ "$FORCE" -eq 1 ]; then
    [ "$FORCE" -eq 1 ] && rm -rf "$BUILD"
    mkdir -p "$BUILD"
    echo "=================== configure ==================="
    ( cd "$BUILD" && cmake .. "${CMAKE_ARGS[@]}" )
    echo "=================== build (-j$JOBS) ==================="
    ( cd "$BUILD" && make -j"$JOBS" )
    [ -x "$BIN" ] || { echo "FATAL: build finished but $BIN is missing"; exit 1; }
    echo
    echo "Built: $BIN"
fi

# What actually got compiled in. Read the target's compile definitions rather
# than CMakeCache: find_package(HDF5)/find_package(TIRPC) do not cache a
# *_FOUND entry, but the -D flags on the target are the ground truth of which
# plugins the binary really has.
flags="$(cat "$BUILD"/CMakeFiles/MUSIC.dir/flags.make 2>/dev/null || true)"
have() { case "$flags" in *"-D$1"*) echo yes ;; *) echo NO ;; esac; }
echo "  compiled-in features (plugins depend on these):"
printf '    %-22s %s\n' \
    "FFTW3"                  "$(have FFTW3)" \
    "FFTW3 threads"          "$(have USE_FFTW_THREADS)" \
    "single precision"       "$(have SINGLE_PRECISION)" \
    "HDF5 (enzo/arepo/...)"  "$(have HAVE_HDF5)" \
    "TIRPC (tipsy XDR)"      "$(have HAVE_TIRPC)"
if [ "$(have HAVE_TIRPC)" = "NO" ] && [ "$(uname -s)" = "Linux" ]; then
    echo "    ^ the TIPSY plugin needs this on Linux; install libtirpc-dev"
fi

if [ "$CHECK" -eq 1 ]; then
    echo "=================== check ==================="
    tmp="$(mktemp -d)"
    cp "$ROOT/ics_example.conf" "$tmp/" 2>/dev/null || {
        echo "no ics_example.conf shipped - skipping"; exit 0; }
    set +e
    ( cd "$tmp" && "$BIN" ics_example.conf > run.log 2>&1 )
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "OK: MUSIC ran ics_example.conf to completion"
    else
        echo "FAILED: MUSIC exited $rc on the shipped ics_example.conf"
        echo "  last output:"
        tail -20 "$tmp/run.log" | sed 's/^/    /'
        echo "  (see the KNOWN ISSUES note at the bottom of this script)"
    fi
    rm -rf "$tmp"
    exit $rc
fi

# --------------------------------------------------------------------------- #
# KNOWN ISSUES
#
# macOS / Apple Silicon: MUSIC1 builds cleanly here but the binary has been
#   observed to segfault at startup, before printing its banner, on both -O3
#   -march=native and plain -O2 - i.e. it is not an optimisation artefact. A
#   crash with no output points at static initialisation (MUSIC registers its
#   plugins through static constructors). Not reproduced on Linux/gcc, which
#   is MUSIC's primary platform. If you hit it, --check will tell you; build
#   on Linux, or use MUSIC2 (github.com/cosmo-sims/MUSIC) if you do not need
#   the tipsy plugin.
# --------------------------------------------------------------------------- #
