#!/usr/bin/env bash
# Build AREPO for the testsuite (code=3) and sph_interp's Voronoi backend.
#
# Source:     codes/arepo/AREPO/       (created by get_arepo.sh)
# Executable: codes/arepo/AREPO/Arepo  (where sph_interp looks for it)
#
# DEPENDENCIES: MPI C compiler, GSL, GMP, HDF5, FFTW3, HWLOC, and perl.
# Linux uses AREPO_SYSTYPE (or the checkout's Makefile.systype, else Ubuntu).
# Darwin uses Homebrew (Apple Silicon /opt/homebrew and Intel /usr/local are both
# supported). The default Config.sh is periodic hydro+MHD (Powell cleaning +
# HLLD) and also supports restart-mode-14 mesh dumps.
#
# runtest.sh builds non-cubic Arepo_asp<LY>_<LZ> variants and the isolated
# Arepo_grav variant on demand through this helper. Override AREPO_CONFIG,
# AREPO_EXEC, or AREPO_BUILD_DIR to select those variants manually.
#
# UNSUPPORTED: accdisk, coldkeplerian, planet, planetcollision (need sinks or a
# central potential), zeldovich and cosmowave (comoving integration).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${AREPO_SRC:-$here/AREPO}"
config="${AREPO_CONFIG:-Config.sh}"
exec_name="${AREPO_EXEC:-Arepo}"
jobs="${JOBS:-}"

[ -d "$src" ] || {
    echo "FATAL: $src missing — run get_arepo.sh first"
    exit 1
}

if [[ "$exec_name" = /* ]]; then
    exec_path="$exec_name"
else
    exec_path="$src/$exec_name"
fi
if [ -x "$exec_path" ] && [ "${AREPO_REBUILD:-0}" != 1 ]; then
    echo "AREPO already built: $exec_path (set AREPO_REBUILD=1 to force)"
    exit 0
fi

if [ -z "$jobs" ]; then
    if command -v nproc >/dev/null 2>&1; then
        jobs="$(nproc 2>/dev/null || true)"
    fi
    if [ -z "$jobs" ] && command -v sysctl >/dev/null 2>&1; then
        jobs="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    jobs="${jobs:-4}"
fi

if [ -f "$config" ]; then
    config_src="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
elif [ -f "$here/$config" ]; then
    config_src="$here/$config"
elif [ -f "$src/$config" ]; then
    config_src="$src/$config"
else
    echo "FATAL: AREPO config '$config' not found"
    exit 1
fi
config_name="$(basename "$config")"
config_dst="$src/$config_name"
if [ "$config_src" != "$config_dst" ]; then
    cp "$config_src" "$config_dst"
fi
# Keep the on-demand gravity config available in the upstream build directory.
if [ -f "$here/Config_grav.sh" ]; then
    cp "$here/Config_grav.sh" "$src/Config_grav.sh"
fi

make_args=(
    -j"$jobs"
    CONFIG="$config_name"
    BUILD_DIR="${AREPO_BUILD_DIR:-build}"
    EXEC="$exec_path"
)
if [ "${AREPO_REBUILD:-0}" = 1 ]; then
    # A dependency ABI may have changed while every source/object timestamp
    # stayed the same. Force compilation/relinking so the executable records
    # the current shared-library install names.
    make_args=(-B "${make_args[@]}")
fi

if [ "$(uname -s)" = Darwin ]; then
    brew_prefix="${HOMEBREW_PREFIX:-}"
    if [ -z "$brew_prefix" ] && command -v brew >/dev/null 2>&1; then
        brew_prefix="$(brew --prefix)"
    fi
    if [ -z "$brew_prefix" ]; then
        if [ -d /opt/homebrew ]; then
            brew_prefix=/opt/homebrew
        elif [ -d /usr/local ]; then
            brew_prefix=/usr/local
        else
            echo "FATAL: Homebrew not found (needed for AREPO dependencies on Darwin)"
            exit 1
        fi
    fi
    if [ ! -x "$brew_prefix/bin/mpicc" ]; then
        echo "FATAL: $brew_prefix/bin/mpicc missing — install Homebrew open-mpi"
        exit 1
    fi
    make_args+=(
        SYSTYPE=Darwin
        CC="$brew_prefix/bin/mpicc"
        OPTIMIZE="-std=c11 -ggdb -O3 -Wall -Wno-format-security -Wno-unknown-pragmas -Wno-unused-function"
        MPICH_LIB="-L$brew_prefix/lib -lmpi"
        GSL_INCL="-I$brew_prefix/include"
        GSL_LIB="-L$brew_prefix/lib -lgsl -lgslcblas"
        HWLOC_INCL="-I$brew_prefix/include"
        HWLOC_LIB="-L$brew_prefix/lib -lhwloc"
        FFTW_INCL="-I$brew_prefix/include"
        FFTW_LIBS="-L$brew_prefix/lib"
        HDF5_INCL="-I$brew_prefix/include -DH5_USE_16_API"
        HDF5_LIB="-L$brew_prefix/lib -lhdf5 -lz"
    )
else
    if [ -n "${AREPO_SYSTYPE:-}" ]; then
        make_args+=(SYSTYPE="$AREPO_SYSTYPE")
    elif [ ! -f "$src/Makefile.systype" ]; then
        make_args+=(SYSTYPE=Ubuntu)
    fi
fi

echo "Building $exec_name from $config_name -> $exec_path"
make -C "$src" "${make_args[@]}"
echo "Done: $exec_path"
