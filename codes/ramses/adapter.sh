#!/bin/bash
# RAMSES run adapter. Every code-specific step of a run lives here rather
# than in runtest.sh: the IC conversion, the per-physics binary choice, the run
# line, and converting RAMSES's output back to tipsy.
#
# No `set -e`: the code relies on plain control flow plus explicit
# `|| { ...; exit 1; }` guards.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../adapter_common.sh"
adapter_parse_args "$@"

# The names the run steps below use, taken from the run description.
analysisdir="$RD_analysisdir"; current_dir="$RD_current_dir"
codes_dir="$RD_current_dir/codes"
dir="$RD_build_dir"; coderun="$RD_binary"; runname="$RD_runname"
ICfile="$RD_icfile"; paramfile="$RD_paramfile"; mpirun="$RD_mpirun"
Nres="$RD_nres"; RUNPHYSICS="$RD_physics"
# A subprocess cannot carry a value between verbs, so each derives
# iOutInterval from the .param.
iOutInterval="$(adapter_param iOutInterval)"

case "$ADAPTER_COMMAND" in
prepare)
    # stdout carries the run line, so the steps below print to stderr.
    {
	# RAMSES (grid AMR code): no file-based hydro IC — deposit the SPH IC onto
	# a fine uniform GridIC (sph_interp petkova) + generate run.nml; the PATCHED
	# condinit (codes/ramses/patches/suite, condinit_kind='gridic') reads it.
	# CUBIC boxes only (the converter fails loudly on slabs), no self-gravity.
	# MHD supported: a genuinely nonzero BField aux routes to ramses3d_mhd
	# (suite_mhd gridic condinit) and the converter writes the hlld MHD
	# namelist + face-B GridIC.
	paramfile="datafiles/$ICfile.param"
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# GridIC resolution: the BASE level matches the SPH per-axis resolution,
	# npx = 2**levelmin = Nres — so Nres must be a power of two.
	Nres="$RD_nres"
	lmax=$(awk -v n="$Nres" 'BEGIN { l=int(log(n)/log(2)+0.5); print ((2^l==n)?l:-1) }')
	if [ "$lmax" == "-1" ]; then echo "Error: RAMSES needs Nres = a power of 2 (got $Nres)."; exit 1; fi
	# MHD binary from the TEST's declared RUNPHYSICS. The converter independently
	# auto-detects the same nonzero B (_has_nonzero_b) to write the MHD namelist +
	# face-B GridIC, so binary and namelist always agree; hydro tests (sedov) stay
	# on ramses3d.
	if [ "$(physics_has mhd "$RUNPHYSICS")" = 1 ]; then
	    coderun=ramses3d_mhd
	    echo "MHD test -> $coderun (hlld, gridic face-B)"
	fi
	# RAMSESICMODE selects the grid mode; BOTH keep the base level at the SPH
	# per-axis resolution (2**levelmin = Nres) so coarse regions have the SAME
	# element count as the SPH run:
	#   amr (default): quasi-Lagrangian (mass-based m_refine/mass_sph)
	#        refinement adds RAMSESAMRLEVELS (default 2) levels ABOVE the SPH
	#        IC resolution, mimicking SPH's Lagrangian adaptivity where mass
	#        concentrates — the SPH-comparable mode.
	#   uniform: levelmin=levelmax=log2(Nres) — one matched Nres^3 grid
	#        (fixed-resolution baseline).
	if [ "${RAMSESICMODE:-amr}" == "amr" ]; then
	    ramseslevels="--levelmin $lmax --levelmax $((lmax + ${RAMSESAMRLEVELS:-2})) --grid amr"
	else
	    ramseslevels="--levelmax $lmax --grid uniform"
	fi
	# Everything RAMSES-native (ic_gridic.bin, run.nml, output_NNNNN dirs) lives
	# under $runname/output/ — the analysis globs the run-dir ROOT for anything
	# ending in 5 digits, so the output_NNNNN directories must not sit there
	# (same layout as the GIZMO/Phantom blocks).
	mkdir -p "$runname/output"
	# Parallel by default like the other codes: RAMSESNP defaults to one per physical core.
	# Resolved BEFORE the conversion so the namelist's per-rank ngridmax is
	# scaled to the rank count; RAMSESNP=1 forces the serial binary.
	RAMSESNP="${RAMSESNP:-$RD_ncores}"
	convertcmd="python ${analysisdir}/ICconvert_tipsytoramses.py datafiles/$ICfile.00000 $runname/output $ramseslevels --np $RAMSESNP"
	echo "$convertcmd"; eval "$convertcmd"
	if [ $? -ne 0 ]; then echo "Error: RAMSES IC conversion failed."; exit 1; fi
	# RAMSES writes output_NNNNN into its cwd -> run from $runname/output/. The
	# '[gridic] loaded' line in the log is the sanity check that the PATCHED
	# condinit (not the stock region IC) initialised the run.
	# env -u GFORTRAN_CONVERT_UNIT: the global big_endian (for SPLASH/tipsy)
	# byte-swaps ALL gfortran unformatted I/O — RAMSES would misread the
	# little-endian GridIC ("bad magic") and write byte-swapped outputs. Strip
	# it here AND for the reverse converter below (amr2cell is gfortran too),
	# exactly like the Phantom launch.
	# RAMSESNP (resolved above, default one per physical core) > 1 selects the MPI twin
	# (built when mpif90 exists) under mpirun; a missing twin falls back to
	# serial with a warning (serial 64^3 MHD was ~32 min, 4 ranks ~4x faster).
	ramseslaunch="$dir/${coderun}"
	if [ "$RAMSESNP" -gt 1 ]; then
	    if [ -x "$dir/${coderun}_mpi" ]; then
		ramseslaunch="mpirun -np $RAMSESNP $dir/${coderun}_mpi"
	    else
		echo "WARNING: $dir/${coderun}_mpi missing (rerun build_ramses.sh with mpif90 on PATH) — falling back to serial ${coderun}."
	    fi
	fi
	runline=( bash -c "cd $runname/output && env -u GFORTRAN_CONVERT_UNIT $ramseslaunch run.nml" )
    } >&2
    printf '%s\n' "${runline[@]}"
    ;;
collect)
	# convert RAMSES output_NNNNN (in the run dir) back into tipsy snapshots:
	# the shipped amr2cell dumps one ASCII row per LEAF cell (all hydro vars in
	# one pass) -> cells_to_tipsy (one gas particle per cell, AMR structure
	# preserved) + gasoline-style .log with the dTuFac stamp. env -u: amr2cell
	# (gfortran) must read RAMSES's native-LE outputs — see the runline note.
	env -u GFORTRAN_CONVERT_UNIT python $current_dir/setupfiles/ICconvert_ramsestotipsy.py $runname/output $runname $iOutInterval datafiles/$ICfile.param --amr2cell $RAMSES_DIR/../utils/f90/amr2cell
    ;;
*)
    adapter_die "unknown command: $ADAPTER_COMMAND (prepare|collect)"
    ;;
esac
