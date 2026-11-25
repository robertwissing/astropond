#!/bin/bash
# OpenGadget3 run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting OpenGadget3's output back to tipsy.
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
	# OpenGadget3 (Gadget-family TreeSPH/SPMHD particle code): 1:1 copy the tipsy
	# SPH particles into the GADGET type-0 HDF5 it reads (no interpolation; same
	# writer as GIZMO/AREPO), and write a GADGET-style parameterfile from the
	# gasoline .param. Snapshots land in $runname/output/ as snap_NNN.hdf5.
	paramfile="datafiles/$ICfile.param"
	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# Binary/config from the TEST's declared RUNPHYSICS (not sniffed from the IC);
	# the converter auto-detects MHD (B aux) / gravity (bDoGravity) itself (standalone).
	# Validate the WHOLE set first: the lookup below reads only the grav and
	# mhd bits, so without this an unknown token would be dropped and a
	# nearby binary silently used in its place.
	adapter_require_tokens OpenGadget3 "$RUNPHYSICS" hydro mhd grav eos=adiabatic
	case "$(physics_build_variant "$RUNPHYSICS")" in
	    hydro) og_base="OpenGadget3";      og_cfg="Config_hydro.sh" ;;
	    mhd)   og_base="OpenGadget3_mhd";  og_cfg="Config_mhd.sh" ;;
	    grav)  og_base="OpenGadget3_grav"; og_cfg="Config_grav.sh" ;;
	    *) echo "Error: OpenGadget3 has no build for RUNPHYSICS=\"$RUNPHYSICS\"."; exit 1 ;;
	esac
	coderun_run="$og_base"
	boxopt=""
	if [ "$(physics_has grav "$RUNPHYSICS")" = 1 ]; then
	    # Isolated self-gravity (evrard, ...): the .param has no dxPeriod (bPeriodic=0).
	    # Build the isolated OpenGadget3_grav (Config_grav.sh: no PERIODIC/NOGRAVITY +
	    # ADAPTIVE_GRAVSOFT_FORGAS) on demand; no BoxSize/LONG_* aspect (open box). The
	    # converter auto-detects gravity (bDoGravity) -> GravityConstantInternal=1 +
	    # PeriodicBoundariesOn=0 (isolated).
	    if [ ! -x "$dir/$coderun_run" ]; then
		echo "Building isolated self-gravity OpenGadget3 -> $coderun_run (Config_grav.sh)"
		( cd "$dir" && make clean EXEC="$coderun_run" >/dev/null 2>&1 || true
		  make -j8 CONFIG="$og_cfg" EXEC="$coderun_run" )
		if [ ! -x "$dir/$coderun_run" ]; then echo "Error: OpenGadget3_grav build failed (run codes/opengadget3/build_opengadget3.sh)."; exit 1; fi
	    fi
	else
	    # Periodic hydro/MHD: BoxSize=dxPeriod; OpenGadget3 bakes the box aspect in at
	    # COMPILE time via LONG_X/LONG_Y/LONG_Z (dimension_i = BoxSize*LONG_i), like
	    # GIZMO's BOX_LONG_*. Cubic -> prebuilt base binary; else build (once, cached
	    # by aspect tag) <base>_asp<LY>_<LZ>.
	    dxP=$(get_param dxPeriod); dyP=$(get_param dyPeriod); dzP=$(get_param dzPeriod)
	    if [ -z "$dxP" ]; then echo "Error: missing dxPeriod in $paramfile"; exit 1; fi
	    [ -z "$dyP" ] && dyP=$dxP; [ -z "$dzP" ] && dzP=$dxP
	    boxopt="--boxsize $dxP"
	    adapter_box_geometry   # dxP dyP dzP boxsize LONGY LONGZ iscubic tag
	    if [ "$iscubic" != "1" ]; then
		coderun_run="${og_base}_asp${tag}"
		if [ ! -x "$dir/$coderun_run" ]; then
		    echo "Building OpenGadget3 with LONG_Y=$LONGY LONG_Z=$LONGZ -> $coderun_run"
		    cfg="$(basename "$og_cfg" .sh)_asp${tag}.sh"
		    cp "$dir/$og_cfg" "$dir/$cfg"
		    printf "\nLONG_X=1\nLONG_Y=%s\nLONG_Z=%s\n" "$LONGY" "$LONGZ" >> "$dir/$cfg"
		    ( cd "$dir" && make clean EXEC="$coderun_run" >/dev/null 2>&1 || true
		      make -j8 CONFIG="$cfg" EXEC="$coderun_run" )
		    if [ ! -x "$dir/$coderun_run" ]; then echo "Error: OpenGadget3 LONG_* build failed."; exit 1; fi
		fi
		echo "Non-cubic box [$dxP,$dyP,$dzP] -> using $coderun_run (LONG_Y=$LONGY LONG_Z=$LONGZ)"
	    fi
	fi
	ogIC="datafiles/${ICfile}_opengadget3.hdf5"
	ogparam="$runname/opengadget3.params"
	mkdir -p "$runname/output"
	# No --mhd/--grav: the converter auto-detects MHD (B aux) + gravity (bDoGravity).
	convertcmd="python ${analysisdir}/ICconvert_tipsytoopengadget3.py datafiles/$ICfile.00000 $ogIC --param-out $ogparam --output-dir $runname/output/ $boxopt"
	echo "$convertcmd"; eval "$convertcmd"
	if [ $? -ne 0 ]; then echo "Error: OpenGadget3 IC conversion failed."; exit 1; fi
	runline=( $mpirun "$dir/${coderun_run}" "$ogparam" )
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# convert OpenGadget3 snap_*.hdf5 output back into tipsy snapshots (same GADGET
	# layout as AREPO/GIZMO; reuses that field mapping). Pass the tipsy .param for
	# box centring + dTuFac.
	python $current_dir/setupfiles/ICconvert_opengadget3totipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
