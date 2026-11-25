#!/bin/bash
# AREPO run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting AREPO's output back to tipsy.
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
	# AREPO moving-mesh: convert the tipsy IC -> AREPO HDF5 IC (sph_interp),
	# then build an AREPO param.txt whose BoxSize / run times come from the
	# tipsy .param we generated for this lattice (NOT the particle extent).
	paramfile="datafiles/$ICfile.param"


	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	arepo_boxopt=""
	if [ "$(physics_has grav "$RUNPHYSICS")" = 1 ]; then
	    # Isolated self-gravity (evrard, ...): the Arepo_grav binary (Config_grav.sh:
	    # SELFGRAVITY + GRAVITY_NOT_PERIODIC + ADAPTIVE_HYDRO_SOFTENING) with the
	    # arepo-grid IC mode FORCED. An isolated cloud needs mesh generators in the
	    # OPEN space around it (else the outer Voronoi cells are unbounded), so
	    # arepo-grid builds the ADDBACKGROUNDGRID-equivalent oct-tree, enlarges the box
	    # to >=2x the gas extent (background shell), and writes THAT box into param.txt.
	    # No dxPeriod / LONG_* aspect (bPeriodic=0, the .param has none). The converter
	    # auto-detects gravity (bDoGravity) -> GravityConstantInternal=1 + adaptive
	    # softening params. Build Arepo_grav on demand.
	    coderun=Arepo_grav
	    AREPOICMODE=arepo-grid
		if [ ! -x "$dir/$coderun" ]; then
		    echo "Building self-gravity AREPO -> $coderun (Config_grav.sh)"
		    ( AREPO_CONFIG=Config_grav.sh AREPO_BUILD_DIR=build_grav \
		      AREPO_EXEC="$coderun" JOBS=8 "$codes_dir/arepo/build_arepo.sh" )
		    if [ ! -x "$dir/$coderun" ]; then echo "Error: Arepo_grav build failed (see codes/arepo/Config_grav.sh)."; exit 1; fi
		fi
	else
	    dxP=$(get_param dxPeriod)
	    dyP=$(get_param dyPeriod)
	    dzP=$(get_param dzPeriod)
	    if [ -z "$dxP" ]; then echo "Error: missing dxPeriod in $paramfile"; exit 1; fi
	    [ -z "$dyP" ] && dyP=$dxP
	    [ -z "$dzP" ] && dzP=$dxP
	    # AREPO BoxSize is the X period; the other axes are set by the COMPILE-TIME
	    # LONG_Y/LONG_Z = dyPeriod/dxPeriod, dzPeriod/dxPeriod (boxSize_Y=BoxSize*LONG_Y).
	    boxsize=$dxP
	    arepo_boxopt="--boxsize $boxsize"
	    adapter_box_geometry   # dxP dyP dzP boxsize LONGY LONGZ iscubic tag

	    # Non-cubic box -> AREPO needs LONG_Y/LONG_Z compiled in (public AREPO is
	    # cubic by default). Cubic (both ~1): use the prebuilt MHD `Arepo`. Else
	    # build (once, cached by aspect tag) Arepo_asp<LY>_<LZ> from Config.sh + the
	    # LONG_Y/LONG_Z lines.
	    if [ "$iscubic" != "1" ]; then
		coderun="Arepo_asp${tag}"
		if [ ! -x "$dir/$coderun" ]; then
		    echo "Building AREPO with LONG_Y=$LONGY LONG_Z=$LONGZ -> $coderun"
		    cfg="Config_asp${tag}.sh"
		    cp "$dir/Config.sh" "$dir/$cfg"
		    printf "\nLONG_Y=%s\nLONG_Z=%s\n" "$LONGY" "$LONGZ" >> "$dir/$cfg"
		    ( AREPO_CONFIG="$dir/$cfg" AREPO_BUILD_DIR="build_asp${tag}" \
		      AREPO_EXEC="$coderun" JOBS=8 "$codes_dir/arepo/build_arepo.sh" )
		    if [ ! -x "$dir/$coderun" ]; then echo "Error: AREPO LONG build failed."; exit 1; fi
		fi
		echo "Non-cubic box [$dxP,$dyP,$dzP] -> using $coderun (LONG_Y=$LONGY LONG_Z=$LONGZ)"
	    fi
	fi

	# tipsy IC -> AREPO IC (sph_interp). $AREPOICMODE picks the conversion
	# (default "particles": 1:1 particle->generator copy). Periodic box + dTuFac
	# come from the .param/.log inside the converter; coords are framed per-axis
	# into [0,period_axis] so they fit the LONG_Y/LONG_Z box.
	# --param-out also writes a FRESH AREPO param.txt with run parameters
	# (timestepping, box, output cadence) derived from the tipsy .param --
	# replacing the old noh-template sed (MaxSizeTimestep is now dDelta, not 0.01).
	arepoIC="datafiles/${ICfile}_arepo.hdf5"
	arepoparam="$runname/param.txt"
	# Grid resolution for the DEPOSIT modes -> tie to the SPH per-axis resolution
	# (Nres = $2) instead of the converter's npx=128 default:
	#   uniform : npx = Nres                 (Nres^3 uniform cells = the SPH lattice).
	#   amr     : quasi-Lagrangian (RAMSES-style, the converter default --refine
	#             mass). START from a COARSE base grid = Nres/2 and refine DOWN to
	#             the SPH resolution (finest npx = Nres) where a cell's mass exceeds
	#             the mean base-cell mass. The base MUST be coarser than the particle
	#             sampling: a mass/Lagrangian threshold can only tell dense from
	#             sparse when each base cell holds SEVERAL particles -- at base=Nres
	#             every cell holds <~1 particle and it refines ~everywhere.
	# Override with AREPOICNPX (finest/uniform), AREPOICBASE (amr base grid).
	# (particles mode: 1:1 copy, no grid -> no resolution flag.)
	Nres="$RD_nres"
	arepres=""
	if [ "$AREPOICMODE" == "uniform" ]; then
	    arepres="--npx ${AREPOICNPX:-$Nres}"
	elif [ "$AREPOICMODE" == "amr" ]; then
	    arepres="--base-npx ${AREPOICBASE:-$((Nres / 2))} --npx ${AREPOICNPX:-$Nres}"
	elif [ "$AREPOICMODE" == "arepo-grid" ]; then
	    # pure-Python bggrid octree: --gridsize is the COARSE void floor (== AREPO
	    # GridSize == base_npx); refines down to ~1 cell/particle where the gas is.
	    arepres="--gridsize ${AREPOICGRIDSIZE:-16}"
	    # AREPOICOCCUPIED=0 -> emit ONLY empty-leaf centres (reproduce AREPO bggrid
	    # exactly); default keeps a generator at every particle cell (finer mesh).
	    [ "${AREPOICOCCUPIED:-1}" == "0" ] && arepres="$arepres --no-include-occupied"
	fi
	convertcmd="python ${analysisdir}/ICconvert_tipsytoarepo.py datafiles/$ICfile.00000 $arepoIC $AREPOICMODE $arepres --param-out $arepoparam --output-dir $runname/ $arepo_boxopt"
	echo "$convertcmd"
	eval "$convertcmd"
	if [ $? -ne 0 ]; then echo "Error: AREPO IC conversion failed."; exit 1; fi

	# RestartFlag 0 = start from the IC (AREPO rebuilds the Voronoi mesh).
	runline=( $mpirun "$dir/${coderun}" "$arepoparam" 0)
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# convert the AREPO snap_*.hdf5 output back into tipsy snapshots so the
	# analysis pipeline (which reads tipsy) can process the AREPO run. Pass the
	# tipsy .param so the per-axis periods (non-cubic box) centre the snapshots
	# correctly and land in the .log.
	python $current_dir/setupfiles/ICconvert_arepototipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
