#!/usr/bin/env bash
# Small deterministic adapter configuration used by test_resolver.sh.

fixture_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
GASOLINE_REFERENCE_SRC="$fixture_config_dir/source"
GASOLINE_DEFAULT_TARGET="spx"
GASOLINE_DEFAULT_COMPILER="${GASOLINE_TEST_COMPILER:-cc}"
GASOLINE_DEFAULT_KERNEL="KERNELOPTIMIZED"
GASOLINE_SOURCE_INPUTS=(. "${GASOLINE_TEST_SECONDARY:-$fixture_config_dir/secondary}")

GASOLINE_COMMON_DEFINES=(
    GASOLINE
    DTADJUST
    DODVDS
    DIFFUSION
    VSIGVISC
    GDFORCE
)

GASOLINE_BUILD_SETS=("hydro" "mhd" "mhd eos=barotropic")

GASOLINE_DEFAULT_COOLING="metal"
GASOLINE_COOLING_OPTIONS=(none metal)

# The fake Makefile ignores these, but the adapter requires the axis to exist.
gasoline_cooling_make_vars() {
    case "$1" in
        none)  printf '%s\n' 'COOLING_OBJ=' 'COOLING_DEF=-DNOCOOLING' ;;
        metal) printf '%s\n' 'COOLING_OBJ=cooling_metal.o' 'COOLING_DEF=-DCOOLING_METAL' ;;
        *) return 1 ;;
    esac
}

gasoline_module_defines() {
    case "$1" in
        starform)         printf '%s\n' STARFORM ;;
        starform=divvoff) printf '%s\n' STARFORM DIVVOFF ;;
        *) return 1 ;;
    esac
}

# Additive, like the real config: one map from token to defines.
gasoline_physics_defines() {
    case "$1" in
        hydro|grav|eos=adiabatic) ;;
        mhd) printf '%s\n' MHD ;;
        eos=barotropic) printf '%s\n' BAROTROPIC ;;
        *) return 1 ;;
    esac
}

# The additive map IS the capability statement, so the name is only composed
# once every physics token in the set is one this config knows. That keeps
# `supports`/`resolve` refusing an unbuildable token, not just the build step.
gasoline_profile_for() {
    local token
    for token in $(build_set_physics "$1"); do
        gasoline_physics_defines "$token" >/dev/null || return 1
    done
    for token in $(build_set_modules "$1"); do
        gasoline_module_defines "$token" >/dev/null || return 1
    done
    build_set_profile_name "$1"
}

gasoline_print_capabilities() {
    printf '%s\n' 'fixture capabilities: hydro, mhd, mhd_barotropic'
}
