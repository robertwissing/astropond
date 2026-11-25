# Shared helpers for the bin/ tool wrappers.  SOURCED, never executed.
#
# Why it lives here and not in bin/: bin/ holds tool wrappers and nothing else,
# and every wrapper already resolves setupfiles/ to build PYTHONPATH.
#
# CONTRACT.  The caller must set `root` (the repository root) before sourcing,
# because it needs `root` to find this file in the first place:
#
#     root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
#     . "$root/setupfiles/_bin_common.sh"
#
# Sourcing then defines `setup` and `glassgen`, plus the ic_* helpers below.
#
# PORTABILITY.  Must work on bash 3.2 (what macOS ships as /bin/bash) as well
# as modern bash on Linux.  Two rules follow:
#   * no arrays - on 3.2 under `set -u`, "${arr[@]}" on an EMPTY array is a
#     fatal "unbound variable".  Callers that do need arrays must expand them
#     as ${arr[@]+"${arr[@]}"}.
#   * POSIX test `[ ]` rather than `[[ ]]`, and no `local -n`, `${var^^}`, etc.
# Nothing here may encode this machine: no conda env names, no absolute home
# paths, no assumption about which python is on PATH.

setup=$root/setupfiles
glassgen=$root/icgenerator/glassgen_python

# ic_die <tool> <message>
ic_die() {
    echo "$1: $2" >&2
    exit 1
}

# ic_require_script <tool> <script> - the Python entry point must exist.
ic_require_script() {
    [ -f "$2" ] || ic_die "$1" "cannot find $2"
}

# ic_require_snapshot <tool> "$@" - a leading positional that is not a flag
# must be an existing snapshot.  Safe with zero arguments.
ic_require_snapshot() {
    tool=$1
    shift
    if [ $# -ge 1 ] && [ "${1#-}" = "$1" ] && [ ! -f "$1" ]; then
        ic_die "$tool" "no such snapshot: $1"
    fi
}

# ic_exec <tool-name> <script> <pythonpath> [args...]
# <tool-name> becomes IC_TOOL_NAME so --help names the command the user typed.
# Any extra environment (e.g. divbclean's SETUP) is simply exported by the
# caller beforehand - `env` inherits it, so no array of K=V pairs is needed.
ic_exec() {
    ic_tool=$1
    ic_script=$2
    ic_pypath=$3
    shift 3
    exec env IC_TOOL_NAME="$ic_tool" \
        PYTHONPATH="$ic_pypath${PYTHONPATH:+:$PYTHONPATH}" \
        python "$ic_script" "$@"
}

# ic_forward <tool> <target> [args...] - hand off to an executable that holds
# the real implementation.  Used by the glassgen wrappers, which forward rather
# than move or symlink because the target resolves its package root from $0.
ic_forward() {
    tool=$1
    target=$2
    shift 2
    [ -x "$target" ] || ic_die "$tool" "cannot find executable $target"
    exec "$target" "$@"
}
