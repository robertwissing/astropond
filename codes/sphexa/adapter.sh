#!/bin/bash
# SPH-EXA run adapter. Every SPH-EXA-specific step of a run lives here: the hdf5
# IC conversion, the module environment the binary was built against, its
# command-line grammar, and the conversion of its hdf5 output back to tipsy.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"

# These describe SPH-EXA, so they live with SPH-EXA rather than in runtest.sh.
SPHEXA_RUN_FLAGS="${EXTRACONDSPHEXA:---prop std}"
SPHEXA_FIELDS="${SPHEXA_FIELDS:-m,x,y,z,vx,vy,vz,temp,u,rho,h,nc}"
# The binary is linked against these modules, so the RUN and the reverse
# conversion both need them. Carrying them in the emitted command scopes them to
# the step that needs them instead of mutating runtest.sh's own environment.
SPHEXA_MODULES='module purge && module load gcc/13.3.1 && module load openmpi/gcc13/5.0.9'

adapter_parse_args "$@"

case "$ADAPTER_COMMAND" in
prepare)
    adapter_require build_dir binary icfile runname paramfile analysisdir
    iout="$(adapter_param iOutInterval)"
    ddelta="$(adapter_param dDelta)"
    nsteps="$(adapter_param nSteps)"
    if [ -z "$iout" ] || [ -z "$ddelta" ] || [ -z "$nsteps" ]; then
        adapter_die "Error: missing required parameters in $RD_paramfile"
    fi
    outinterval="$(awk -v io="$iout" -v dd="$ddelta" 'BEGIN { printf "%.8f", io * dd }')"
    endtime="$(awk -v ns="$nsteps" -v dd="$ddelta" 'BEGIN { printf "%.8f", ns * dd }')"

    # convert the tipsy IC into the SPH-EXA hdf5 format
    echo "python ${RD_analysisdir}/ICconvert_tipsytosphexa.py datafiles/$RD_icfile.00000 datafiles/$RD_icfile " >&2
    python "${RD_analysisdir}/ICconvert_tipsytosphexa.py" \
        "datafiles/$RD_icfile.00000" "datafiles/$RD_icfile" >&2

    printf '%s\n' bash -c "$SPHEXA_MODULES && exec $(printf '%q' "$RD_build_dir/$RD_binary") \
--init ./datafiles/$RD_icfile.h5 -o $RD_runname/$RD_runname.h5 $SPHEXA_RUN_FLAGS \
-s $endtime -w $outinterval -f $SPHEXA_FIELDS"
    ;;
collect)
    adapter_require runname paramfile current_dir
    iout="$(adapter_param iOutInterval)"
    # convert the SPH-EXA hdf5 output back into tipsy snapshots
    bash -c "$SPHEXA_MODULES && exec python \
$(printf '%q' "$RD_current_dir/setupfiles/ICconvert_sphexatotipsy.py") \
$(printf '%q' "$RD_runname/$RD_runname.h5") $(printf '%q' "$RD_runname") $(printf '%q' "$iout")"
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
