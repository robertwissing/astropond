# Shared helpers for per-code RUN adapters.
#
# A run adapter owns everything code-specific about turning a prepared tipsy IC
# into a finished run: converting the IC into the code's own format, the module
# environment and command-line grammar its binary needs, and converting the
# code's output back to tipsy. runtest.sh knows only where the adapter lives and
# where the code's binary was built -- never a code's file formats, config names
# or flags.
#
#   adapter.sh prepare --run-desc FILE   do the side effects, then PRINT the run
#                                        line on stdout, one argument per line
#   adapter.sh collect --run-desc FILE   convert the code's output back to tipsy
#
# stdout of `prepare` is the run line and nothing else, so every diagnostic goes
# to stderr. The run description is a key=value file runtest.sh writes once; it
# carries the run, not the code, so an adapter reads what it needs and derives
# the rest from the .param itself (which is why no value flows back out).
# Paths in it are relative to runtest.sh's working directory at hand-over
# (the test case dir, which holds datafiles/), and an adapter inherits that
# directory -- so relative paths mean the same thing on both sides.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/physics_set.sh"

adapter_die() { printf '%s\n' "$*" >&2; exit 1; }

adapter_load_rundesc() {  # $1 = run-description file -> one RD_<key> per entry
    local file="$1" line key value
    [ -r "$file" ] || adapter_die "run description is not readable: $file"
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"; value="${line#*=}"
        case "$key" in
            [A-Za-z_]*) ;;
            *) adapter_die "malformed run description line: $line" ;;
        esac
        printf -v "RD_$key" '%s' "$value"
    done <"$file"
}

adapter_require() {  # $@ = run-description keys that must be present, non-empty
    local key name
    for key in "$@"; do
        name="RD_$key"
        [ -n "${!name:-}" ] || adapter_die "run description is missing '$key'"
    done
}

adapter_param() {  # $1 = key -> its value in the run's .param file
    awk -F= -v key="$1" '
        $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
            gsub(/[[:space:]]*/, "", $2)
            print $2
        }
    ' "$RD_paramfile"
}

adapter_parse_args() {  # $@ = the adapter's own "$@" -> ADAPTER_COMMAND, RD_*
    ADAPTER_COMMAND="${1:-}"
    [ -n "$ADAPTER_COMMAND" ] || adapter_die "no command given (prepare|collect)"
    shift
    local rundesc=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --run-desc) [ "$#" -ge 2 ] || adapter_die "--run-desc requires a value"
                        rundesc="$2"; shift 2 ;;
            --run-desc=*) rundesc="${1#*=}"; shift ;;
            *) adapter_die "unknown option: $1" ;;
        esac
    done
    [ -n "$rundesc" ] || adapter_die "$ADAPTER_COMMAND requires --run-desc"
    adapter_load_rundesc "$rundesc"
}

# Box geometry from the .param, byte-identical in three adapters before this.
# `tag` is computed unconditionally; it is only ever USED on the non-cubic path.
adapter_box_geometry() {  # -> dxP dyP dzP boxsize LONGY LONGZ iscubic tag
    dxP="$(adapter_param dxPeriod)"
    dyP="$(adapter_param dyPeriod)"
    dzP="$(adapter_param dzPeriod)"
    [ -n "$dxP" ] || adapter_die "Error: missing dxPeriod in $RD_paramfile"
    [ -n "$dyP" ] || dyP="$dxP"
    [ -n "$dzP" ] || dzP="$dxP"
    boxsize="$dxP"
    LONGY=$(awk -v a="$dyP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
    LONGZ=$(awk -v a="$dzP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
    iscubic=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { d=0.0005; print (y>1-d && y<1+d && z>1-d && z<1+d)?1:0 }')
    tag=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { printf "%.4f_%.4f", y, z }' | tr '.' 'p')
}

# A code WITHOUT a build adapter has nobody who can answer "is there a binary
# for this whole set": runtest.sh's gate only compares the set against a row it
# was handed, and physics_build_variant() below reads just the grav and mhd bits
# and drops every other token on the floor. So a set like "mhd starform" used to
# match the plain mhd binary and run a star-formation test in a binary with no
# star formation in it, silently. This is where that stops: the adapter names
# the tokens its builds understand, and anything else halts the run by name.
# Declare a whole axis with a trailing * (cooling=*) to accept any value on it.
adapter_require_tokens() {  # $1 = label, $2 = the set, $@ = tokens accepted
    local label="$1" set="$2" token known accepted
    shift 2
    for token in $set; do
        accepted=0
        for known in "$@"; do
            case "$known" in
                *'*') case "$token" in "${known%\*}"*) accepted=1 ;; esac ;;
                *)    [ "$token" = "$known" ] && accepted=1 ;;
            esac
            [ "$accepted" = 1 ] && break
        done
        [ "$accepted" = 1 ] || adapter_die \
            "Error: $label has no build for '$token' (RUNPHYSICS=\"$set\"); it understands: $*"
    done
}

# The ported run steps call get_param, which each per-code block used to define
# for itself -- eight identical copies. One definition, here.
get_param() { adapter_param "$1"; }
