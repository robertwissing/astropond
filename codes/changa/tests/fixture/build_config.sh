#!/usr/bin/env bash
# Small three-profile configuration used only by test_resolver.sh.

fixture_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CHANGA_REFERENCE_SRC="$fixture_config_dir/source"
CHANGA_DEFAULT_CHARM_SOURCE="../charm"
CHANGA_DEFAULT_UTILITY_SOURCE="../utility"
CHANGA_CHARM_COMPONENTS=(bin include lib tmp)
CHANGA_COMMON_CONFIGURE_FLAGS=(--enable-fixture-base)
CHANGA_BUILD_SETS=("hydro" "mhd" "mhd eos=barotropic")

# Additive, like the real configs: one map from token to configure flags.
changa_physics_configure_flags() {
    case "$1" in
        hydro|grav|eos=adiabatic) ;;
        mhd) printf '%s\n' --enable-mhd ;;
        eos=barotropic) printf '%s\n' --enable-barotropic ;;
        *) return 1 ;;
    esac
}

# The additive map IS the capability statement, so the name is only composed
# once every physics token in the set is one this config knows. That keeps
# `supports`/`resolve` refusing an unbuildable token, not just the build step.
changa_profile_for() {
    local token
    for token in $(build_set_physics "$1"); do
        changa_physics_configure_flags "$token" >/dev/null || return 1
    done
    # ChaNGa has no module axis, so any leftover token is one nothing understands.
    [ -z "$(build_set_modules "$1")" ] || return 1
    build_set_profile_name "$1"
}

changa_print_capabilities() {
    printf '%s\n' 'ChaNGa fixture: hydro, mhd, and mhd_barotropic'
}

unset fixture_config_dir
