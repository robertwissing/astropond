#!/usr/bin/env bash
# Build SPH-EXA for the testsuite (code=2 in runtest.sh).
#
# DEPENDENCIES: CMake >= 3.22, a C++20 compiler (gcc >= 11), MPI (the suite uses
#   gcc/13.3.1 + openmpi/gcc13/5.0.9 via `module load`), and HDF5 (the suite's IC
#   exchange is HDF5; configure with -DSPH_EXA_WITH_H5PART=ON or system HDF5).
#   Optional CUDA for GPU builds (off here — CI/run host is CPU). Run
#   get_sphexa.sh first.
#
# Produces the CPU `sphexa` binary under build/main/src/sphexa/. runtest.sh's
# SPHEXA_DIR points at that directory. Idempotent (skips if the binary exists).
#
# UNSUPPORTED TESTS: SPH-EXA is a pure hydro/gravity SPH mini-app — no MHD, no
#   sinks/central potential, no comoving cosmology in this build. Unsupported:
#   all MHD tests (orzag, alfven, currentsheet, mhd*, abcflow, ponomarenko),
#   accdisk, coldkeplerian, planet, planetcollision, zeldovich, cosmowave.
#   Supported: sedov, kh, blob, evrard, noh, gresho, taylorgreen, rt, isentropicvortex…
#   Confirm before gating.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

SRC="${SPHEXA_SRC:-$here/sphexa}"
BUILD="${SPHEXA_BUILD:-$SRC/build}"
JOBS="${JOBS:-$(nproc)}"
BIN="$BUILD/main/src/sphexa/sphexa"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_sphexa.sh first"; exit 1; }

if [ -x "$BIN" ]; then
    echo "SPH-EXA already built: $BIN"
    exit 0
fi

# Match the runtime toolchain runtest.sh loads for code=2, when modules exist.
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load gcc/13.3.1 || true
    module load openmpi/gcc13/5.0.9 || true
fi

# First configure: FetchContent pulls the H5hut source into _deps/h5hut-src.
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release ${SPHEXA_CMAKE_ARGS:-}

# H5hut's CMakeLists links `hdf5-shared` if that CMake target exists, else falls back
# to `hdf5-static`. A system HDF5 found in FindHDF5 MODULE mode exports NEITHER target
# (it sets HDF5_*_LIBRARIES full paths), so the fallback tries to link a nonexistent
# `hdf5-static` -> `ld: cannot find -lhdf5-static`. Point the fallback at the plain
# shared lib (`-lhdf5`). Idempotent; re-processed on the second configure below.
H5HUT_CML="$BUILD/_deps/h5hut-src/CMakeLists.txt"
if [ -f "$H5HUT_CML" ] && grep -q "target_link_libraries(H5hut hdf5-static)" "$H5HUT_CML"; then
    echo "Patching fetched H5hut to link shared HDF5 (no hdf5-static on this host)"
    sed -i 's/target_link_libraries(H5hut hdf5-static)/target_link_libraries(H5hut hdf5)/' "$H5HUT_CML"
    # Re-configure so the patched H5hut link deps take effect (FetchContent won't
    # re-fetch the already-populated source).
    cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release ${SPHEXA_CMAKE_ARGS:-}
fi

cmake --build "$BUILD" --target sphexa -j"$JOBS"
echo "Done: $BIN"
