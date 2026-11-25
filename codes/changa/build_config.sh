#!/usr/bin/env bash
# Public ChaNGa capabilities for this testsuite.  The public tree is the
# repository default even when a private profile is selected locally.

changa_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CHANGA_REFERENCE_SRC="$changa_config_dir/changa-public"
CHANGA_DEFAULT_CHARM_SOURCE="../charm"
CHANGA_DEFAULT_UTILITY_SOURCE="../utility"
CHANGA_CHARM_COMPONENTS=(bin include lib tmp)
CHANGA_COMMON_CONFIGURE_FLAGS=()
# Compiler CPU target (-march/-mtune): "native" = the build host's CPU, "" = generic x86-64.
CHANGA_CPU_TARGET="${CHANGA_CPU_TARGET-native}"
# The builds `all` produces: each entry is a full RUNPHYSICS set.
CHANGA_BUILD_SETS=("hydro")

# Physics is ADDITIVE: each token contributes its own configure flags and a
# build is their union. A token this fork cannot do is simply absent from the
# map, which is how the public tree rejects mhd and eos=barotropic.
changa_physics_configure_flags() {  # $1 = one set token
    case "$1" in
        hydro|grav|eos=adiabatic) ;;
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
    cat <<'EOF'
ChaNGa public physics tokens (additive -- a build is the union of its tokens;
grav is a runtime parameter and contributes no flag):
  hydro                  -> (none)
  grav                   -> (none)
  eos=adiabatic          -> (none, the default)
  mhd, eos=barotropic    -> NOT available in the public tree

The public profile intentionally rejects MHD.  Select an MHD-capable fork and
matching build config, for example codes/changa/build_config_private.sh.
EOF
}

unset changa_config_dir
