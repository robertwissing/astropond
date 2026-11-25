#!/bin/bash
# Gasoline run adapter. Gasoline reads the tipsy IC natively and writes tipsy
# snapshots, so a run is just its command line -- there is nothing to convert on
# the way in or out. The BUILD side lives beside this file in build_gasoline.sh.
#
# No `set -e`: this body ran inside runtest.sh, which does not set it.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"
adapter_parse_args "$@"

dir="$RD_build_dir"; coderun="$RD_binary"; runname="$RD_runname"
ICfile="$RD_icfile"; mpirun="$RD_mpirun"; runlauncher="$RD_runlauncher"
Nsmooth="$RD_nsmooth"

# Glass ICs are relaxed with relax.config's kernel and pair operator (runtest
# exports them in glass mode); warn when this build differs, as the glass is
# then not force-free under the run's own operator.
gasoline_check_glass_operator() {
    [ -n "${SPH_KERNEL:-}" ] && [ -r "$dir/effective-defines.txt" ] || return 0
    local defs k g f
    defs=" $(tr '\n' ' ' < "$dir/effective-defines.txt") "
    case "$defs" in
        *" -DWENDLAND "*) k=wendlandc2 ;;
        *" -DWENDLANDC4 "*) k=wendlandc4 ;;
        *" -DKERNELOPTIMIZED "*) k=kerneloptimized ;;
        *) k=other ;;
    esac
    case "$defs" in
        *" -DISPH "*) g=onesided-isph ;;
        *" -DONESIDED "*) g=onesided-sph ;;
        *) g=average-sph ;;
    esac
    case "$defs" in *" -DGDFORCE "*) f=1 ;; *) f=0 ;; esac
    [ "$k" = "$SPH_KERNEL" ] ||
        printf 'WARNING: Gasoline build kernel %s, glass relaxed with SPH_KERNEL=%s (relax.config)\n' "$k" "$SPH_KERNEL" >&2
    [ -z "${GLASS_GRADIENT:-}" ] || [ "$g" = "$GLASS_GRADIENT" ] ||
        printf 'WARNING: Gasoline build gradient %s, glass relaxed with GLASS_GRADIENT=%s (relax.config)\n' "$g" "$GLASS_GRADIENT" >&2
    [ -z "${GLASS_SIGMA_F:-}" ] || awk -v a="$f" -v b="$GLASS_SIGMA_F" 'BEGIN { exit (a + 0 == b + 0) ? 0 : 1 }' ||
        printf 'WARNING: Gasoline build sigma_f %s (GDFORCE), glass relaxed with GLASS_SIGMA_F=%s (relax.config)\n' "$f" "$GLASS_SIGMA_F" >&2
}

case "$ADAPTER_COMMAND" in
prepare)
    gasoline_check_glass_operator
    runline=( ${runlauncher:-$mpirun} $dir/${coderun} -s $Nsmooth -o $runname/$runname -I datafiles/$ICfile.00000  $RD_extracondrun datafiles/$ICfile.param)
    printf '%s\n' "${runline[@]}"
    ;;
collect)
    # Gasoline writes tipsy snapshots directly: nothing to convert back.
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
