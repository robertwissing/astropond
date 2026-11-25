#!/bin/bash
# Phantom run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting Phantom's output back to tipsy.
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
phantomprefix="$runname"   # collect needs it too; it is just the run name

case "$ADAPTER_COMMAND" in
prepare)
    # stdout carries the run line, so the steps below print to stderr.
    {
	# PHANTOM (SPH particle code): 1:1 copy the tipsy SPH particles into a
	# Phantom full-dump IC (no interpolation) + a .in run file. Phantom runs as
	# `phantom <prefix>.in` and writes dumps <prefix>_NNNNN into the cwd, so run
	# from inside $runname/.
	# Need iOutInterval (snapshot stride) for the reverse converter below.
	paramfile="datafiles/$ICfile.param"
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	phantomprefix="$runname"
	# Binary from the TEST's declared RUNPHYSICS (not sniffed). The hydro `phantom`
	# binary silently ignores any magnetic block and evolves pure hydro; phantom_mhd
	# is the MHD build; phantom_grav is the self-gravity build (SETUP=evrard:
	# GRAVITY=yes, adiabatic, non-periodic). The converter auto-detects MHD (real B)
	# and self-gravity (G_code=1 units when bDoGravity) from the IC, so it needs no
	# physics flag.
	# Validate the WHOLE set first: the lookup below reads only the grav and
	# mhd bits, so without this an unknown token would be dropped and a
	# nearby binary silently used in its place.
	adapter_require_tokens PHANTOM "$RUNPHYSICS" hydro mhd grav eos=adiabatic
	case "$(physics_build_variant "$RUNPHYSICS")" in
	    hydro) coderun=phantom ;;
	    mhd)   coderun=phantom_mhd ;;
	    grav)  coderun=phantom_grav ;;
	    *) echo "Error: PHANTOM has no build for RUNPHYSICS=\"$RUNPHYSICS\"."; exit 1 ;;
	esac
	# Build on demand (grav especially -- it's the newest variant). Phantom builds
	# are serial/finicky, so delegate to the idempotent build script (skips the
	# already-built phantom/phantom_mhd, builds only what's missing).
	if [ ! -x "$dir/$coderun" ]; then
	    echo "PHANTOM binary $coderun missing -> building via codes/phantom/build_phantom.sh"
	    bash "$current_dir/codes/phantom/build_phantom.sh"
	    if [ ! -x "$dir/$coderun" ]; then echo "Error: PHANTOM $coderun build failed (see codes/phantom/build_phantom.sh)."; exit 1; fi
	fi
	convertcmd="python ${analysisdir}/ICconvert_tipsytophantom.py datafiles/$ICfile.00000 $runname/$phantomprefix --in-out $runname/$phantomprefix.in"
	echo "$convertcmd"; eval "$convertcmd"
	if [ $? -ne 0 ]; then echo "Error: PHANTOM IC conversion failed."; exit 1; fi
	# phantom reads/writes relative to its cwd -> launch from $runname/.
	#
	# Two-pass .in handshake: Phantom rejects our minimal .in, REWRITES a complete
	# one for its compile options and aborts ("try again"). So: keep our values in
	# .in.user, run pass 1 (expected non-zero -> '|| true'), re-assert our values
	# onto Phantom's rewritten .in with phantom_patch_infile.py, then run pass 2 to
	# actually evolve.
	#
	# SINGLE MPI RANK + OpenMP (mpirun -np 1, parallelism via OMP_NUM_THREADS).
	# Phantom is built with MPI, but multi-rank MPI is broken on this Intel-MPI stack
	# ("Fatal error in internal_Test: Invalid MPI_Request" once the run starts) -- the
	# same upstream MPI breakage seen with AREPO/madvoro here. Single-rank + OpenMP
	# evolves correctly and still parallelises across the node, so BOTH passes run
	# -np 1. (Pass 1 must be single-rank anyway: under multi-rank MPI every rank calls
	# MPI_Abort(666) on the incomplete .in *without* the master rewriting it, so the
	# .in is never completed and pass 2 aborts on the same 4-entry file.)
	#
	# Pass 1 rewrites the incomplete .in and aborts ("try again" -> '|| true'); the
	# patch re-asserts our values onto Phantom's rewritten .in; pass 2 evolves.
	#
	# env -u GFORTRAN_CONVERT_UNIT: the global big_endian (set for SPLASH/tipsy)
	# would make Phantom misread our native-endian dumps as 'wrong endian'. Phantom
	# dumps are native-endian, so strip the override for these launches only.
	#
	# OMP_STACKSIZE: Phantom's kdtree neighbour cache lives on the OpenMP thread
	# stack; the default size overflows and SEGVs in getneigh on the first density
	# iteration. 512M is the value Phantom's own docs recommend.
	export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"
	export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$RD_ncores}"
	patchscript="$current_dir/setupfiles/phantom_patch_infile.py"
	runline=( bash -c "cd $runname \
	    && cp $phantomprefix.in $phantomprefix.in.user \
	    && { env -u GFORTRAN_CONVERT_UNIT mpirun -np 1 $dir/${coderun} $phantomprefix.in || true; } \
	    && python $patchscript $phantomprefix.in --from $phantomprefix.in.user \
	    && env -u GFORTRAN_CONVERT_UNIT mpirun -np 1 $dir/${coderun} $phantomprefix.in" )
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# Move Phantom's native dumps (<runname>_NNNNN, underscore) into <runname>/output/
	# BEFORE reverse conversion: the analysis globs the run-dir root for tipsy
	# snapshots (<runname>.NNNNN, dot) and gets confused by the underscore dumps +
	# .ev/.in sitting beside them. The reverse converter then reads from output/.
	mkdir -p $runname/output
	mv $runname/${phantomprefix}_[0-9][0-9][0-9][0-9][0-9] $runname/output/ 2>/dev/null || true
	# convert Phantom dumps (<runname>/output/<runname>_NNNNN) back into tipsy
	# snapshots (read via Phantom's own readPhantomDump; rho reconstructed from h,
	# B + dTuFac stamped). Pass the .param for box centring + dTuFac.
	python $current_dir/setupfiles/ICconvert_phantomtotipsy.py $runname/output/$runname $runname $iOutInterval datafiles/$ICfile.param
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
