#!/bin/bash
# ChaNGa run adapter. ChaNGa reads the tipsy IC natively and writes tipsy
# snapshots, so a run is just its command line -- there is nothing to convert on
# the way in or out. The BUILD side lives beside this file in build_changa.sh.
#
# No `set -e`: this body ran inside runtest.sh, which does not set it.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"
adapter_parse_args "$@"

dir="$RD_build_dir"; coderun="$RD_binary"; runname="$RD_runname"
ICfile="$RD_icfile"; mpirun="$RD_mpirun"; runlauncher="$RD_runlauncher"
Nsmooth="$RD_nsmooth"

case "$ADAPTER_COMMAND" in
prepare)
    # dCloudDensity is a Gasoline-only parameter; ChaNGa rejects the whole file.
    sed -i '/^dCloudDensity/d' datafiles/$ICfile.param
    # Gasoline's bDoGravity covers ALL gravity (external potentials included)
    # and bDoSelfGravity only the tree. ChaNGa ignores bDoSelfGravity: there
    # bDoGravity is the tree alone and external forces have their own switch.
    # So "bDoSelfGravity = 0" (powerlawring, rhoit setups) becomes -g, which
    # overrides the .param without editing it -- the .param is shared with the
    # Gasoline run of the same IC.
    selfgrav=()
    grep -Eq '^[[:space:]]*bDoSelfGravity[[:space:]]*=[[:space:]]*0' datafiles/$ICfile.param && selfgrav=(-g)
    # FDM tests: <Nsmooth> is the FDM neighbour number (nSmoothFDM). The gas nSmooth then comes from the .param
    # (a test with gas writes it there) or ChaNGa's default, so gas and FDM can use different numbers.
    nsopt=(-s $Nsmooth)
    [ "$(physics_has fdm "$RD_physics")" = 1 ] && nsopt=(-nSmoothFDM $Nsmooth)
    runline=( ${runlauncher:-$mpirun} $dir/${coderun} "${nsopt[@]}" -o $runname/$runname -I datafiles/$ICfile.00000 -b 24 +useckloop +balancer MultistepLB_SFC -D 1 "${selfgrav[@]}" $RD_extracondrun datafiles/$ICfile.param)
    printf '%s\n' "${runline[@]}"
    ;;
collect)
    # ChaNGa writes tipsy snapshots directly: nothing to convert back.
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
