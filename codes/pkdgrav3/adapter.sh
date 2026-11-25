#!/bin/bash
# PKDGRAV3 run adapter. PKDGRAV3 reads the tipsy IC natively and writes tipsy
# snapshots, so there is no IC conversion: preparing a run means generating its
# .par from the gasoline .param, and collecting one means writing the
# gasoline-style .log the analysis framework expects beside the snapshots.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"

adapter_parse_args "$@"

case "$ADAPTER_COMMAND" in
prepare)
    adapter_require build_dir binary icfile runname paramfile analysisdir
    dxp="$(adapter_param dxPeriod)"
    iout="$(adapter_param iOutInterval)"
    if [ -z "$dxp" ] || [ -z "$iout" ]; then
        adapter_die "Error: missing dxPeriod/iOutInterval in $RD_paramfile"
    fi
    pkdpar="$RD_runname/run.par"
    echo "python ${RD_analysisdir}/IC_pkdgrav3_param.py datafiles/$RD_icfile.00000 $RD_runname $pkdpar --boxsize $dxp" >&2
    python "${RD_analysisdir}/IC_pkdgrav3_param.py" \
        "datafiles/$RD_icfile.00000" "$RD_runname" "$pkdpar" --boxsize "$dxp" >&2 ||
        adapter_die "Error: PKDGRAV3 .par generation failed."
    # $RD_mpirun is a launcher PREFIX ("mpirun", "mpirun -np 8"): word-split it.
    printf '%s\n' ${RD_mpirun:-} "$RD_build_dir/$RD_binary" "$pkdpar"
    ;;
collect)
    adapter_require runname icfile paramfile current_dir
    iout="$(adapter_param iOutInterval)"
    # PKDGRAV3 snapshots are already tipsy (<runname>/<runname>.<NNNNN>); just
    # write the gasoline-style .log (box params) the analysis framework needs.
    python "$RD_current_dir/setupfiles/ICconvert_pkdgrav3totipsy.py" \
        "$RD_runname" "$RD_runname" "$iout" "datafiles/$RD_icfile.param"
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
