#!/bin/bash
# Athena++ run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting Athena++'s output back to tipsy.
#
# No `set -e`: this body ran inside runtest.sh, which does not set it, and the
# code relies on plain control flow plus explicit `|| { ...; exit 1; }` guards.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"
adapter_parse_args "$@"

# The names the run steps below use, taken from the run description.
analysisdir="$RD_analysisdir"; current_dir="$RD_current_dir"
codes_dir="$RD_current_dir/codes"
dir="$RD_build_dir"; coderun="$RD_binary"; runname="$RD_runname"
ICfile="$RD_icfile"; paramfile="$RD_paramfile"; mpirun="$RD_mpirun"
Nres="$RD_nres"; RUNPHYSICS="$RD_physics"
# iOutInterval was set by the prepare step and read by the collect step when
# both lived in runtest.sh. A subprocess cannot carry a value between verbs, so
# each derives it from the .param -- the same rule the whole contract rests on.
iOutInterval="$(adapter_param iOutInterval)"

case "$ADAPTER_COMMAND" in
prepare)
    # stdout carries the run line, so the steps below print to stderr.
    {
	# Athena++ (grid, UNIFORM mesh) — same GridIC layer as RAMSES:
	# tipsy -> ic_gridic.bin (petkova; 6 solenoidal face-B components when B
	# is genuinely nonzero) + athinput -> the from_gridic problem generator.
	# NON-CUBIC (slab) boxes supported: the converter scales the per-axis
	# cell counts by the box aspect (kh + the 2D orzag slabs run here even
	# though RAMSES's 2^levelmin cube can't). MPI by default (ATHENANP ranks,
	# default one per physical core; built against the local parallel HDF5 — see
	# codes/athena++/build_athena.sh), capped at the meshblock count the
	# converter prints (ranks > meshblocks aborts Athena++; meshblock sizes
	# are divisors of each axis' cell count, so no power-of-2 Nres needed).
	paramfile="datafiles/$ICfile.param"
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# npx = Nres = the SPH per-axis X resolution; other axes follow the aspect
	Nres="$RD_nres"
	# MHD binary from the TEST's declared RUNPHYSICS; the converter independently
	# auto-detects the same nonzero B for the CT face-B GridIC (they always agree).
	if [ "$(physics_has mhd "$RUNPHYSICS")" = 1 ]; then
	    coderun=athena_mhd
	    echo "MHD test -> $coderun (CT, gridic face-B)"
	fi
	ATHENANP="${ATHENANP:-$RD_ncores}"
	mkdir -p "$runname/output"
	convertcmd="python ${analysisdir}/ICconvert_tipsytoathena.py datafiles/$ICfile.00000 $runname/output --npx $Nres --np $ATHENANP"
	echo "$convertcmd"
	convout=$(eval "$convertcmd") || { echo "Error: Athena++ IC conversion failed."; exit 1; }
	echo "$convout"
	# cap ranks at the meshblock count (blocks are >= 8 cells/axis, so small
	# grids can have fewer blocks than cores; ranks > blocks aborts Athena++)
	nblocks=$(echo "$convout" | awk -F= '/^meshblocks=/{print $2}')
	if [ -n "$nblocks" ] && [ "$ATHENANP" -gt "$nblocks" ]; then
	    echo "ATHENANP=$ATHENANP > $nblocks meshblocks -> using -np $nblocks"
	    ATHENANP=$nblocks
	fi
	# Athena++ writes its outputs into cwd -> run from $runname/output/ (the
	# analysis globs the run-dir ROOT, same layout rule as RAMSES/GIZMO).
	# '[from_gridic] loaded' in the log = the suite pgen initialised the run.
	runline=( bash -c "cd $runname/output && mpirun -np $ATHENANP $dir/${coderun} -i athinput.gridic" )
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# convert Athena++ gridic.out2.NNNNN.athdf (in the run dir) back into
	# tipsy snapshots: one gas particle per cell via cells_to_tipsy, MHD Bcc
	# -> BField aux, .log with the dTuFac stamp.
	python $current_dir/setupfiles/ICconvert_athenatotipsy.py $runname/output $runname $iOutInterval datafiles/$ICfile.param
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
