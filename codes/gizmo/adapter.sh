#!/bin/bash
# GIZMO run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting GIZMO's output back to tipsy.
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

# Every variant builds in the one source tree (shared objects and GIZMO_config.h), so two runs that need a
# binary at the same time would clean and compile over each other. Build under a lock, and re-check for the
# binary once the lock is held (another run may just have built it). With a base config and BOX_LONG values
# (args 3-5) the aspect config is written inside the lock as well.
gizmo_build_locked() {  # <config> <exec> [<base config> <BOX_LONG_Y> <BOX_LONG_Z>]
    (
        flock 9
        [ -x "$dir/$2" ] && exit 0
        if [ -n "${3:-}" ]; then
            cp "$dir/$3" "$dir/$1" && printf "\nBOX_LONG_Y=%s\nBOX_LONG_Z=%s\n" "$4" "$5" >> "$dir/$1" || exit 1
        fi
        cd "$dir" && { make clean EXEC="$2" >/dev/null 2>&1 || true; make -j8 CONFIG="$1" EXEC="$2"; }
    ) 9>"$dir/.gizmo_build.lock"
}

case "$ADAPTER_COMMAND" in
prepare)
    # stdout carries the run line, so the steps below print to stderr.
    {
	# GIZMO (particle code): 1:1 copy the tipsy SPH particles into the GADGET
	# type-0 HDF5 GIZMO reads (no interpolation), and write a GIZMO
	# parameterfile from the gasoline .param. Snapshots land in $runname/output/.
	paramfile="datafiles/$ICfile.param"
	dxP=$(get_param dxPeriod); dyP=$(get_param dyPeriod); dzP=$(get_param dzPeriod)
	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# Non-periodic IC (bPeriodic=0, e.g. isolated evrard): the .param has no
	# dxPeriod. The box handling below keys on this, INDEPENDENT of gravity.
	isperiodic=1; [ "$(get_param bPeriodic)" = "0" ] && isperiodic=0
	# Binary/config from the TEST's declared RUNPHYSICS (not sniffed from the IC).
	# The converter still auto-detects MHD (B aux) / gravity (bDoGravity) itself,
	# so it agrees with this by construction and stays usable standalone.
	# Validate the WHOLE set first: the lookup below reads only the grav and
	# mhd bits, so without this an unknown token would be dropped and a
	# nearby binary silently used in its place.
	adapter_require_tokens GIZMO "$RUNPHYSICS" hydro mhd grav eos=adiabatic
	isgrav=0
	gizmo_variant="$(physics_build_variant "$RUNPHYSICS")"
	case "$gizmo_variant" in
	    hydro) gizmo_base="GIZMO";      gizmo_cfg="Config_hydro.sh" ;;
	    mhd)   gizmo_base="GIZMO_mhd";  gizmo_cfg="Config_mhd.sh" ;;
	    grav)  gizmo_base="GIZMO_grav"; gizmo_cfg="Config_grav.sh"; isgrav=1
		   # Isolated self-gravity binary (Config_grav.sh: no BOX_PERIODIC /
		   # no SELFGRAVITY_OFF / +ADAPTIVE_GRAVSOFT_FORGAS); built on demand.
		   if [ ! -x "$dir/$gizmo_base" ]; then
			echo "Building isolated self-gravity GIZMO -> $gizmo_base (Config_grav.sh)"
			gizmo_build_locked "$gizmo_cfg" "$gizmo_base"
			if [ ! -x "$dir/$gizmo_base" ]; then echo "Error: GIZMO_grav build failed (run codes/gizmo/build_gizmo.sh)."; exit 1; fi
		   fi ;;
	    *) echo "Error: GIZMO has no build for RUNPHYSICS=\"$RUNPHYSICS\"."; exit 1 ;;
	esac
	coderun_run="$gizmo_base"
	boxopt=""
	if [ "$isperiodic" = "0" ]; then
	    # Non-periodic IC (isolated: evrard, ...): the gasoline .param has no
	    # dxPeriod. The converter frames coords to the particle extent and writes
	    # THAT BoxSize into both the IC HDF5 header and the parameterfile (read
	    # back from the header), so DON'T force --boxsize and skip the BOX_LONG
	    # aspect machinery (which needs BOX_PERIODIC). NB: the binary must also be
	    # non-periodic — GIZMO_grav is (no BOX_PERIODIC); a non-periodic non-gravity
	    # hydro test would need its own non-periodic Config (none in the suite yet).
	    echo "Non-periodic run -> BoxSize derived from the particle extent."
	else
	    if [ -z "$dxP" ]; then echo "Error: missing dxPeriod in $paramfile"; exit 1; fi
	    [ -z "$dyP" ] && dyP=$dxP; [ -z "$dzP" ] && dzP=$dxP
	    boxopt="--boxsize $dxP"
	    # GIZMO bakes the box aspect ratio in at COMPILE time via BOX_LONG_Y/BOX_LONG_Z
	    # (= dyPeriod/dxPeriod, dzPeriod/dxPeriod; BoxSize=dxPeriod is the X side), like
	    # AREPO's LONG_Y/LONG_Z. Cubic (both ~1): use the prebuilt base binary. Else
	    # build (once, cached by aspect tag) <base>_asp<LY>_<LZ> from its base Config +
	    # the BOX_LONG lines.
	    adapter_box_geometry   # dxP dyP dzP boxsize LONGY LONGZ iscubic tag
	    if [ "$iscubic" != "1" ]; then
		coderun_run="${gizmo_base}_asp${tag}"
		if [ ! -x "$dir/$coderun_run" ]; then
		    echo "Building GIZMO with BOX_LONG_Y=$LONGY BOX_LONG_Z=$LONGZ -> $coderun_run"
		    cfg="$(basename "$gizmo_cfg" .sh)_asp${tag}.sh"
		    # clean+build with EXEC override so this aspect build only removes/links
		    # its OWN binary (the cubic base binaries survive).
		    gizmo_build_locked "$cfg" "$coderun_run" "$gizmo_cfg" "$LONGY" "$LONGZ"
		    if [ ! -x "$dir/$coderun_run" ]; then echo "Error: GIZMO BOX_LONG build failed."; exit 1; fi
		fi
		echo "Non-cubic box [$dxP,$dyP,$dzP] -> using $coderun_run (BOX_LONG_Y=$LONGY BOX_LONG_Z=$LONGZ)"
	    fi
	fi
	gizmoIC="datafiles/${ICfile}_gizmo.hdf5"
	gizmoparam="$runname/gizmo.params"
	mkdir -p "$runname/output"
	# No --mhd/--grav: the converter auto-detects both from the IC (standalone).
	convertcmd="python ${analysisdir}/ICconvert_tipsytogizmo.py datafiles/$ICfile.00000 $gizmoIC --param-out $gizmoparam --output-dir $runname/output/ $boxopt"
	echo "$convertcmd"; eval "$convertcmd"
	if [ $? -ne 0 ]; then echo "Error: GIZMO IC conversion failed."; exit 1; fi
	runline=( $mpirun "$dir/${coderun_run}" "$gizmoparam" )
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# convert GIZMO snapshot_*.hdf5 output back into tipsy snapshots (same GADGET
	# layout as AREPO; reuses that field mapping). Pass the tipsy .param for box
	# centring.
	python $current_dir/setupfiles/ICconvert_gizmototipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
