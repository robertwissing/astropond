# The RUNPHYSICS set vocabulary, shared by runtest.sh and the run adapters.
#
# A set is capability TOKENS separated by spaces: bare physics (hydro|mhd|grav),
# key=value axes (eos=barotropic, cooling=metal) and modules (starform=divvoff).
# These three readers are the only way a set is ever taken apart on the run side;
# the BUILD side has its own copies in codes/build_common.sh, because an adapter
# is invoked standalone and must not depend on runtest.sh.

physics_value() {  # $1 = key, $2 = set -> prints its value ("" if bare); 1 if absent
    local key="$1" t
    for t in $2; do
        [ "$t" = "$key" ] && { printf '\n'; return 0; }
        case "$t" in "$key="*) printf '%s\n' "${t#*=}"; return 0 ;; esac
    done
    return 1
}

physics_has() {  # $1 = key, $2 = set -> 1/0
    if physics_value "$1" "$2" >/dev/null; then printf 1; else printf 0; fi
}

# Codes without a build adapter keep one prebuilt binary per physics variant, so
# they need the set as a single discrete choice. Derived at the point of use,
# never carried around as a second declaration: the set stays the one truth.
#
# This reads TWO BITS and discards the rest of the set, so it is only ever safe
# behind adapter_require_tokens(), which checks every token first. `grav+mhd` is
# a label, never a capability: a code claiming grav and mhd claims both, and one
# that ships no combined binary says so in its own case arm below -- the fact
# lives in the adapter that knows it, not in a SUPPORT row that has to be
# hand-edited the day such a binary is first built.
physics_build_variant() {  # $1 = set -> hydro|mhd|grav|grav+mhd
    case "$(physics_has grav "$1")$(physics_has mhd "$1")" in
        00) printf 'hydro\n' ;;
        01) printf 'mhd\n' ;;
        10) printf 'grav\n' ;;
        11) printf 'grav+mhd\n' ;;
    esac
}
