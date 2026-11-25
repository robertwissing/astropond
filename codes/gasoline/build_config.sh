#!/usr/bin/env bash
# Public Gasoline capabilities for this testsuite.  The public N-BodyShop tree is
# hydro+gravity only; MHD, the barotropic EOS, star formation and turbulent
# driving need the private tree and codes/gasoline/build_config_private.sh.
#
# The build machinery below (target, MDL sibling, common defines, kernel,
# cooling) is carried over from the private config and has not yet been built
# against gasoline-public.

gasoline_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
GASOLINE_REFERENCE_SRC="$gasoline_config_dir/gasoline-public"
GASOLINE_DEFAULT_TARGET="spx"
GASOLINE_DEFAULT_COMPILER="mpicc"
# Compiler CPU target (-march/-mtune): "native" = the build host's CPU, "" = generic x86-64.
GASOLINE_CPU_TARGET="${GASOLINE_CPU_TARGET-native}"
GASOLINE_DEFAULT_KERNEL="KERNELOPTIMIZED"

# Paths are relative to GASOLINE_SRC unless absolute. The reference spx target
# compiles both the Gasoline tree and its sibling MPI MDL implementation.
GASOLINE_SOURCE_INPUTS=(
    .
    ../mdl/mpi
)

GASOLINE_COMMON_DEFINES=(
    GASOLINE
    DTADJUST
    DODVDS
    DIFFUSION
    VSIGVISC
    GDFORCE
)

# The builds `all` produces: each entry is a full RUNPHYSICS set.
GASOLINE_BUILD_SETS=("hydro")

# Cooling-module axis (see build_config_private.sh for why the default is metal).
GASOLINE_DEFAULT_COOLING="metal"
GASOLINE_COOLING_OPTIONS=(
    none
    metal
)

gasoline_cooling_make_vars() {
    case "$1" in
        none)
            printf '%s\n' 'COOLING_OBJ=' 'COOLING_DEF=-DNOCOOLING'
            ;;
        metal)
            printf '%s\n' 'COOLING_OBJ=cooling_metal.o' 'COOLING_DEF=-DCOOLING_METAL'
            ;;
        *)
            return 1
            ;;
    esac
}

# Physics is ADDITIVE: each token contributes its own defines and a build is
# their union. A token this tree cannot do is simply absent from the map, which
# is how the public tree rejects mhd and eos=barotropic.
gasoline_physics_defines() {  # $1 = one set token
    case "$1" in
        hydro|grav|eos=adiabatic) ;;
        *) return 1 ;;
    esac
}

# No physics modules (starform, turbdriving) in the public tree.
gasoline_module_defines() {
    return 1
}

# The additive map IS the capability statement, so the name is only composed
# once every token in the set is one this config knows. That keeps
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
    cat <<'EOF'
Gasoline public physics tokens (additive -- a build is the union of its tokens;
grav is a runtime parameter and contributes no define):
  hydro                  -> (none)
  grav                   -> (none)
  eos=adiabatic          -> (none, the default)
  mhd, eos=barotropic    -> NOT available in the public tree

Cooling module (cooling= in the set, or --cooling):
  none                   -> -DNOCOOLING, no cooling object
  metal                  -> -DCOOLING_METAL, cooling_metal.o   (default)

Physics modules starform, starform=divvoff, turbdriving -> NOT available.
Select an MHD-capable fork and matching build config, for example
codes/gasoline/build_config_private.sh.
EOF
}

unset gasoline_config_dir
