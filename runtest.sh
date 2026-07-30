#!/bin/bash

# Per-code binary directories. Each defaults to the reproducible local build under
# codes/<code>/ (built by codes/<code>/get_<code>.sh + build_<code>.sh) but may be
# overridden from the environment (e.g. CI, or to point at a private/MHD tree).
# See codes/CODES_BUILD_PLAN.md.
codes_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/codes"
export SPHEXA_DIR="${SPHEXA_DIR:-$codes_dir/sphexa/sphexa/build/main/src/sphexa/}"
# Gasoline: public N-BodyShop build is hydro-only (binary `gasoline`); the suite's
# MHD reference binary `gasolinemhd` comes from a private tree -- export
# GASOLINE_DIR to point at that build for MHD tests (see CODES_BUILD_PLAN.md).

export GASOLINE_DIR="${GASOLINE_DIR:-$codes_dir/gasoline/gasoline-public}"
export CHANGA_DIR="${CHANGA_DIR:-$codes_dir/changa/changa-public/}"
export AREPO_DIR="${AREPO_DIR:-$codes_dir/arepo/AREPO/}"
export RAMSES_DIR="${RAMSES_DIR:-$codes_dir/ramses/ramses/bin/}"
export GIZMO_DIR="${GIZMO_DIR:-$codes_dir/gizmo/gizmo/}"
export PHANTOM_DIR="${PHANTOM_DIR:-$codes_dir/phantom/phantom/bin/}"
export ATHENA_DIR="${ATHENA_DIR:-$codes_dir/athena++/athena/bin/}"
export PKDGRAV3_DIR="${PKDGRAV3_DIR:-$codes_dir/pkdgrav3/pkdgrav3/build/}"
export OPENGADGET3_DIR="${OPENGADGET3_DIR:-$codes_dir/opengadget3/opengadget3/}"

#Directory for the setupfiles
current_dir=$(pwd -P)
analysisdir=$current_dir/setupfiles/
#Python (numba) glass-relaxation IC generator package root (see PLAN.md there)
glassgenpydir=$current_dir/icgenerator/glassgen_python

# AREPO (code=3):
# "particles" (default) = a 1:1 copy of the SPH particles as mesh generators.
#For a regridded IC instead set "voronoi", "amr" or "uniform" (deposits the SPH fields
# onto that mesh, petkova/mass-conserving).
AREPOICMODE="particles"

#RAMSESIC uniform or amr
RAMSESICMODE="amr"

#Code is selected by the 4th positional argument:
#  0=Gasoline 1=ChaNGa 2=SPH-EXA 3=AREPO  (wired: build + IC conversion)
#  5=GIZMO 6=Phantom 8=PKDGRAV3 9=OpenGadget3  (wired: build + IC conversion)
#  4=RAMSES 7=Athena++  (build only so far; IC conversion + run/convert blocks
#  pending -- see codes/CODES_BUILD_PLAN.md)
#The executable directory and binary name per code are set in the code-selection block after arg parsing.
#Each code is obtained/built by codes/<code>/get_<code>.sh + build_<code>.sh.

# --- RUNPHYSICS support matrix -------------------------------------------------
# Which physics (hydro|mhd|grav|gravmhd) each code is WIRED to run end-to-end in
# THIS script today. Each setup_<test>() declares the test's RUNPHYSICS; the gate
# below refuses unsupported (code, physics) pairs early. Seeded to current reality
# -- edit as codes gain physics (grid self-gravity, a combined grav+mhd build,
# validated PKDGRAV3 SPH-MHD, ...). `runtest.sh support` prints this table.
declare -a CODENAMES=( Gasoline ChaNGa SPH-EXA AREPO RAMSES GIZMO Phantom Athena++ PKDGRAV3 OpenGadget3 )
declare -a SUPPORT
SUPPORT[0]="hydro mhd grav gravmhd"   # Gasoline    (gasolinemhd; verified)
SUPPORT[1]="hydro mhd grav gravmhd"   # ChaNGa      (--enable-mhd; verified)
SUPPORT[2]="hydro mhd grav"           # SPH-EXA     (converter sets gravConstant=1/open box on bDoGravity; --prop std Barnes-Hut)
SUPPORT[3]="hydro mhd grav"           # AREPO       (Arepo_grav: Config_grav.sh SELFGRAVITY+GRAVITY_NOT_PERIODIC, arepo-grid bggrid IC)
SUPPORT[4]="hydro mhd"                # RAMSES      (grav needs poisson=.true. -- GRID_CODES_PLAN)
SUPPORT[5]="hydro mhd grav"           # GIZMO       (GIZMO_grav wired; gravmhd not built)
SUPPORT[6]="hydro mhd grav"           # Phantom     (phantom_grav: SETUP=evrard, adiabatic isolated self-gravity)
SUPPORT[7]="hydro mhd"                # Athena++    (grav needs self-gravity -- GRID_CODES_PLAN)
SUPPORT[8]="hydro grav"               # PKDGRAV3    (N-body+SPH: gravity native; no public SPMHD -> no mhd/gravmhd)
SUPPORT[9]="hydro mhd grav"           # OpenGadget3 (OpenGadget3_grav: Config_grav.sh, isolated adaptive-softening self-gravity)

physics_supported() {  # $1 = code index, $2 = physics label -> 0 iff supported
    local p; for p in ${SUPPORT[$1]}; do [ "$p" = "$2" ] && return 0; done; return 1
}
print_support_matrix() {
    printf '%-5s %-12s %s\n' code code-name "supported physics (RUNPHYSICS)"
    printf '%-5s %-12s %s\n' ----- ------------ ------------------------------
    local i; for i in "${!CODENAMES[@]}"; do
        printf '%-5s %-12s %s\n' "$i" "${CODENAMES[$i]}" "${SUPPORT[$i]}"
    done
}

#If running serial/mpi or with charmrun
#mpirun="$dir/charmrun.smp +p 63"
#mpirun=""
mpirun="mpirun "

#An extra word added at the end of the simulation output files and directory
EXTRANAME="amrnew3"

#An extra word added to the glass file names (e.g. to tag the code used to relax)
GLASSNAME=""

# Format a numeric parameter for file/run names: round to 2 decimals and strip
# trailing zeros (2.0 -> 2, 0.48989 -> 0.49, 100.00 -> 100). A nonzero value
# that would round to 0.00 (e.g. ampl=1e-4, pini=1e-6) keeps its %g form so
# distinct small parameters still produce distinct names (cache keys!).
# Non-numeric arguments (preset names etc.) pass through unchanged.
fmt() {
    awk -v x="$1" 'BEGIN {
        if (x !~ /^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$/) { print x; exit }
        v = x + 0
        s = sprintf("%.2f", v)
        if ((s == "0.00" || s == "-0.00") && v != 0) { printf "%g\n", v; exit }
        sub(/0+$/, "", s); sub(/\.$/, "", s)
        print s
    }'
}

# Glass relaxation: the Python (numba) glassgen.cli relaxes the pre-IC into a
# glass (multi-resolution cascade + heavy-ball momentum + the Wendland C2
# self-density correction; writes ${glassfile}_IC directly). The target density
# is read AUTOMATICALLY from the pre-IC .param (dICdens* profiles 1/2/3 =
# uniform/step/tanh). For a disk/accdisk profile (>=4) add a density table to
# GLASSMODE (e.g. "--table densitytable.npz" - the coordinate mapping lives
# in the self-describing table header; convert a legacy densitytable_xdr
# once with `python -m glassgen.density`).
#
# Extra glassgen.cli options appended to the default run, e.g.
#   "--no-progressive"  single full-resolution relax (no cascade)
#   "--icr0-stop 1e-3"  tighter glass (default 3e-3);  "--momentum 0.3"
#   "--no-self-bias"    bare W0 density self term
# Empty = the production defaults (progressive cascade, momentum 0.5,
# icr0-stop 3e-3, self-bias on). ICCONFIGPY sets the per-stage iteration cap and
# the ICR0/ICRLOOP0/rhopow/rate knobs (ICR0 = initial average move fraction of
# the box; rhopow sharpens the density match; -n is the max relaxation steps).
GLASSMODE=""
ICCONFIGPY="-n 4800 -oi 0 --icr0 1.0 --icrloop0 0.5 --rhopow 2.0 --icr0rate 1.5"

#Extra condition during runtime
#EXTRACONDALL=" -facQ 1.0 -fch 1.0 -psidecfac 1.0 -thermaldiff 0.1 -alpha 2.0 -beta 8.0 -nInitCorrIter 50 -smin 32 -smax 40000 "
EXTRACONDALL=" -thermaldiff 0.1 -alpha 0.5 -beta 8.0 "


#Extra runtime conditions and output fields specific to SPH-EXA (code=2)
EXTRACONDSPHEXA=" --prop std "

fields="m,x,y,z,vx,vy,vz,temp,u,rho,h,nc"


usage() {
    echo "Usage: $0 [-i] [-r] [-a value1] ... [-h value8] <nameofsimulation> <Nsmooth> <Nres> <distrib> <code> [vm] [extra]"
    echo ""
    echo "To see the parameter choices (-a .. -h) for a specific simulation, run it by name with no further arguments, e.g.:"
    echo "    $0 accdisk"
    echo ""
    echo "Options:"
    echo "  -a, -b, -c, ...    These options depend on the simulation. To see the specific options, enter the nameofsimulation without arguments."
    echo "  -i                 Interactive."
    echo "  -r                 Do-reference: after the run, save the analysis result"
    echo "                     as the regression baseline '<runname>_reference.json'"
    echo "                     (requires setupfiles/IC_analysis_<simulation>.py)."
    echo ""
    echo "Arguments:"
    echo "  nameofsimulation   Name of the simulation to be run (e.g., shocktube)."
    echo "  Nsmooth            Number of neigbours within smoothing kernel"
    echo "  Nres               Resolution for the simulation (usually specified as number of elements in the x direction)."
    echo "  distrib            Initial conditions type:"
    echo "                     0: lattice"
    echo "                     1: random"
    echo "                     2: glass"
    echo "  code               Simulation code: 0: Gasoline  1: ChaNGa  2: SPH-EXA  3: AREPO  4: RAMSES  5: GIZMO  6: Phantom  7: Athena++  8: PKDGRAV3  9: OpenGadget3"
    echo "                     (Each test declares its physics hydro|mhd|grav|gravmhd; a code that"
    echo "                      does not support it is rejected. Run '$0 support' for the matrix.)"
    echo "  vm                 (optional) enable variable SPH particle masses in the IC generator (default 0 = equal-mass particles)."
    echo "  extra              (optional) extra flags appended to the runtime command."
    echo "If glass data file exist for a specific Nsmooth it will skipp the glass generation and run the simulation. If you choose an nSmooth that does not have a glass datafile it will do the whole glass generation again."
    echo ""
    echo "List of available simulations:"
    echo "  accdisk            Accretion disk simulation."
    echo "  alfven             Alfvén wave simulation."
    echo "  gresho             Gresho vortex simulation."
    echo "  mhdcollapse        MHD collapse simulation."
    echo "  kh                 Kelvin-Helmholtz instability simulation."
    echo "  orzag              Orszag-Tang vortex simulation (-a 1 for true 3D, Tu 2022 / Helzel et al. 2011)."
    echo "  abcflow            ABC-flow kinematic dynamo (3D, Teyssier et al. 2006)."
    echo "  ponomarenko        Ponomarenko screw-flow kinematic dynamo (3D, Teyssier et al. 2006)."
    echo "  mhdrotor           MHD rotor simulation."
    echo "  mhdloop            MHD loop advection simulation."
    echo "  shocktube          Different kinds of shock tubes."
    echo "                     (Enter 'shocktube' without arguments for more details on available options)"
    echo "  sedov              Sedov-Taylor blast wave simulation."
    echo "  evrard             Evrard collapse simulation."
    echo "  polytrope          Polytrope / Oscillating Polytrope simulation. (Requires additional graviational relaxation)"
    echo "  taylorgreen        Taylor-Green simulation."
    echo "  areablob           Hydrostatic blob(s) simulation."
    echo "  blob               Cloud crushing (wind-cloud) simulation."
    echo "  rt                 Rayleigh-Taylor instability simulation."
    echo "  mti                Magneto-thermal instability simulation."
    echo "  mhdisowave         Isolated MHD wave simulation."
    echo "  mhddivtest         Magnetic divergence test simulation."
    echo "  planet             Planet (single body) simulation."
    echo "  planetcollision    Planet collision simulation."
    echo "  square             Hydrostatic square (2D) / inner-cube (3D) test."
    echo "  cosmowave          Cosmological MHD wave (Berlok 2022) simulation."
    echo "  mhdbalsaravortex   Balsara MHD vortex simulation."
    echo "  zeldovich          Zeldovich pancake (cosmological) simulation."
    echo "  mri                Magnetorotational instability simulation."
    echo "  mhdloopshear       Magnetized advection loop with shear simulation."
    echo "  currentsheet       MHD current sheet (reconnection robustness) simulation."
    echo "  noh                Noh problem (converging strong shock) simulation."
    echo "  isentropicvortex   Isentropic (Yee) vortex (smooth convergence) simulation."
    echo "  wave               Linear (M)HD wave: sound/fast/slow/Alfven (convergence) simulation."
    echo "  coldkeplerian      Cold Keplerian disk/ring (AV / angular-momentum) simulation."
    echo ""
    exit 1
}


# Parse command line options
interactive=0
doreference=0
declare -a args=("" "" "" "" "" "" "" "")
while getopts ":ira:b:c:d:e:f:g:h:" opt; do
  case $opt in
    i)
      interactive=1
      ;;
    r)
      doreference=1
      ;;
    a|b|c|d|e|f|g|h)
      args[$(( $(printf "%d" "'$opt") - 97 ))]=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage $testsim
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage $testsim
      ;;
  esac
done
shift $((OPTIND -1))

# `runtest.sh support` -> print the code x physics support matrix and exit.
if [ "${1:-}" = "support" ]; then
    print_support_matrix
    exit 0
fi

# Check for the testsim argument
testsim=${1:-"default"}
if [ "$testsim" = "default" ]; then
    echo "Test simulation type not specified."
    usage
fi
shift

# RUNPHYSICS helpers (see the support matrix + gate below). Each setup_<test>()
# declares RUNPHYSICS (hydro|mhd|grav|gravmhd); it is a property of the TEST, used
# ONLY to pick the per-code binary/config and to gate unsupported (code,physics)
# pairs. It is NEVER passed to a converter -- converters auto-detect physics from
# the IC so they stay usable standalone (outside runtest.sh).
nonzero() { awk -v v="$1" 'BEGIN { print (v+0 != 0) ? 1 : 0 }'; }  # numeric != 0
combine_physics() {  # $1 mhd(0/1) $2 grav(0/1) -> hydro|mhd|grav|gravmhd
    if [ "$1" = 1 ] && [ "$2" = 1 ]; then echo gravmhd
    elif [ "$1" = 1 ]; then echo mhd
    elif [ "$2" = 1 ]; then echo grav
    else echo hydro; fi
}

# Function for mhdcollapse setup with generic parameters
setup_mhdcollapse() {
    RUNPHYSICS="gravmhd"
    # Set default values if not specified
    mu=${args[0]:-10}
    rhodiff=${args[1]:-360}
    Erat=${args[2]:-0.045}

    echo "mu is the magnetic flux ratio -a "
    echo "rhodiff is the density difference between central cloud and medium -b "
    echo "Erat is the rotational energy ratio -c "

    echo "Final settings for MHD Collapse:"
    echo "mu: $mu, rhodiff: $rhodiff, Erat: $Erat"

    # Setup based on the potentially updated values
    EXTRACONDIC="$mu $rhodiff $Erat"
    EXTRACONDRUN=" "
    prefile_extra="rd$(fmt "${rhodiff}")"
    glassfile_extra="rd$(fmt "${rhodiff}")${GLASSNAME}"
    runname_extra="mu$(fmt "${mu}")rd$(fmt "${rhodiff}")Erat$(fmt "${Erat}")"
}

# Function for accdisk setup with generic parameters
setup_accdisk() {
    RUNPHYSICS="grav"
    # Set default values if not specified
    hr=${args[0]:-0.1}
    massrat=${args[1]:-1E-3}
    onlyoneBH=${args[2]:-0}
    q=${args[3]:-0.5}
    rorb=${args[4]:-1.0}
    rdisk=${args[5]:-10.0}
    rinner=${args[6]:-0.5}

    echo "Using settings for accdisk:"
    echo "Disk height ratio (hr) -a : $hr"
    echo "Mass ratio (massrat) -b : $massrat"
    echo "Binary black hole presence (onlyoneBH) -c : $onlyoneBH"
    echo "Mass distribution parameter (q) -d : $q"
    echo "Orbital radius (rorb) -e : $rorb"
    echo "Disk radius (rdisk) -f : $rdisk"
    echo "Inner disk radius (rinner) -g : $rinner"

    # Setup based on the potentially updated values
    EXTRACONDIC="$hr $massrat $onlyoneBH $q $rorb $rdisk $rinner "
    EXTRACONDRUN=" "
    prefile_extra="hr$(fmt "${hr}")mr$(fmt "${massrat}")bh$(fmt "${onlyoneBH}")q$(fmt "${q}")ro$(fmt "${rorb}")rd$(fmt "${rdisk}")ri$(fmt "${rinner}")"
    glassfile_extra="hr$(fmt "${hr}")mr$(fmt "${massrat}")bh$(fmt "${onlyoneBH}")q$(fmt "${q}")ro$(fmt "${rorb}")rd$(fmt "${rdisk}")ri$(fmt "${rinner}")${GLASSNAME}"
    runname_extra="hr$(fmt "${hr}")mr$(fmt "${massrat}")bh$(fmt "${onlyoneBH}")q$(fmt "${q}")ro$(fmt "${rorb}")rd$(fmt "${rdisk}")ri$(fmt "${rinner}")"
}

# Function for alfven setup with generic parameters
setup_alfven() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    rotate=${args[0]:-1}

    echo "Using settings for circularized alfvenwave:"
    echo "rotate wave (rotate) -a : $rotate"

    # Setup based on the potentially updated values
    EXTRACONDIC="$rotate "
    EXTRACONDRUN=" "
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra="rot$(fmt "${rotate}")"
}

# Function for Shock-tube setup with generic parameters
setup_shocktube() {
    shock=${args[0]:-1.0}
    # Array mapping shock indices to shock names (must cover every `choice` in
    # IC_setup_shocktube.choose_shock, else the index check below rejects them)
    shocks=(
        "Sod shock"
        "Ryu 1a"
        "Ryu 1b"
        "Ryu 2a"
        "Ryu 2b"
        "Brio-Wu Ryu 5a"
        "C-shock"
        "Steady shock"
        "Toth"
        "MHD discontinuity"
        "strong blast"
        "fast slow shock"
        "isothermal B98"
        "Mach 25 DW94"
        "rarefaction"
    )

    # Get the shock name based on the shock index
    shock_index=$(echo "$shock" | awk '{printf "%.0f", $0}')  # Convert shock to integer index
    if (( shock_index < 1 || shock_index > ${#shocks[@]} )); then
        echo "Invalid shock index: $shock"
        return 1
    fi
    shockname=${shocks[$((shock_index - 1))]}

    # Physics: only Sod (1) and the strong-blast (11) presets are hydro (B=0);
    # every other choose_shock preset carries a nonzero B -> MHD.
    if [ "$shock_index" = 1 ] || [ "$shock_index" = 11 ]; then
        RUNPHYSICS="hydro"
    else
        RUNPHYSICS="mhd"
    fi

    # Display the settings being used
    echo "Using settings for Shocktube:"
    echo "Different shocks (argument -a): $shock"
    echo "
    # shocks(1) = 'Sod shock'
    # shocks(2) = 'Ryu 1a'
    # shocks(3) = 'Ryu 1b'
    # shocks(4) = 'Ryu 2a'
    # shocks(5) = 'Ryu 2b'
    # shocks(6) = 'Brio-Wu (Ryu 5a)'
    # shocks(7) = 'C-shock'
    # shocks(8) = 'Steady shock'
    "
    echo "Selected shock: $shockname"

    # Setup based on the potentially updated values
    EXTRACONDIC="$shock "
    EXTRACONDRUN=""
    prefile_extra="${shockname// /_}"
    glassfile_extra="${shockname// /_}${GLASSNAME}"
    runname_extra="${shockname// /_}"

    echo "prefile_extra: $prefile_extra"
    echo "glassfile_extra: $glassfile_extra"
    echo "runname_extra: $runname_extra"

}

# Function for Orzag-Tang Vortex setup with generic parameters
setup_orzag() {
    RUNPHYSICS="mhd"
    dim3=${args[0]:-0}
    # Set default values if not specified
    echo "Using settings for Orzag-Tang Vortex:"
    echo "3D vortex -a (0: 2D slab, 1: true 3D Tu 2022 / Helzel et al. 2011) : $dim3"

    # Setup based on the potentially updated values
    EXTRACONDIC=" $dim3 "
    EXTRACONDRUN=""
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra=""
    if [ "$dim3" != "0" ]; then
        prefile_extra="_3d"
        glassfile_extra="_3d${GLASSNAME}"
        runname_extra="_3d"
    fi
}

# Function for the ABC-flow kinematic dynamo (Teyssier et al. 2006)
setup_abcflow() {
    RUNPHYSICS="mhd"
    B0=${args[0]:-1e-3}
    amp=${args[1]:-1.0}
    echo "Using settings for the ABC-flow kinematic dynamo:"
    echo "uniform seed Bz -a : $B0"
    echo "ABC flow amplitude (A=B=C) -b : $amp"
    # Order matches IC_setup_abcflow.create(... B0, amp) signature
    EXTRACONDIC="$B0 $amp "
    EXTRACONDRUN=""
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra="B$(fmt "${B0}")amp$(fmt "${amp}")"
}

# Function for the Ponomarenko screw-flow kinematic dynamo (Teyssier et al. 2006)
setup_ponomarenko() {
    RUNPHYSICS="mhd"
    B0=${args[0]:-0.1}
    echo "Using settings for the Ponomarenko screw-flow dynamo:"
    echo "seed vector-potential amplitude -a : $B0"
    # Order matches IC_setup_ponomarenko.create(... B0) signature
    EXTRACONDIC="$B0 "
    EXTRACONDRUN=""
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra="B$(fmt "${B0}")"
}

# Function for Magnetized Rotor setup with generic parameters
setup_mhdrotor() {
    RUNPHYSICS="mhd"
    rhodisk=${args[0]:-10.0}
    # Set default values if not specified
    echo "Using settings for Magnetized Rotor:"
    echo "density disk -a : $rhodisk"

    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodisk "
    EXTRACONDRUN=""
    prefile_extra="rhodiff$(fmt "${rhodisk}")"
    glassfile_extra="rhodiff$(fmt "${rhodisk}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodisk}")"
}

# Function for Magnetized advection loop setup with generic parameters
setup_mhdloop() {
    RUNPHYSICS="mhd"
    rhodisk=${args[0]:-1.0}
    # Set default values if not specified
    echo "Using settings for Magnetized Advection Loop:"
    echo "density disk -a : $rhodisk"

    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodisk "
    EXTRACONDRUN=""
    prefile_extra="rhodiff$(fmt "${rhodisk}")"
    glassfile_extra="rhodiff$(fmt "${rhodisk}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodisk}")"
}

# Function for gresho setup with generic parameters
setup_gresho() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    result=$(echo "scale=5; sqrt(3/25)" | bc -l)
    echo "$result"
    mach=${args[0]:-$result}
    
    echo "Using settings for gresho:"
    echo "mach number of flow -a : $mach"
    
    # Setup based on the potentially updated values
    EXTRACONDIC="$mach "
    EXTRACONDRUN=""
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra="mach$(fmt "${mach}")"
}

# Function for kelvin-helmholtz setup with generic parameters
setup_kh() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    result=$(echo "scale=5; sqrt(6)/5" | bc -l)
    echo "$result"
    rhodiff=${args[0]:-2.0}
    mach=${args[1]:-$result}
    smooth=${args[2]:-1}
    B0=${args[3]:-0.0}
    Bdir=${args[4]:-1}

    echo "Using settings for kelvin-helmholtz:"
    echo "rho diff -a : $rhodiff"
    echo "mach number of flow -b : $mach"
    echo "smooth density disc -c : $smooth"
    echo "initial magnetic field strength -d : $B0"
    echo "Direction of field x=1 y=2 z=3 -e : $Bdir"
    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodiff $mach $smooth $B0 $Bdir "
    EXTRACONDRUN=""
    prefile_extra="rhodiff$(fmt "${rhodiff}")smooth$(fmt "${smooth}")"
    glassfile_extra="rhodiff$(fmt "${rhodiff}")smooth$(fmt "${smooth}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodiff}")mach$(fmt "${mach}")smooth$(fmt "${smooth}")B$(fmt "${B0}")Bdir$(fmt "${Bdir}")"
}


# Function for Sedov blast wave + mhd setup with generic parameters
setup_sedov() {
    RUNPHYSICS="hydro"
    select=${args[0]:-1.0}
    betain=${args[1]:-0.0}
    ublast=${args[2]:-10.0}
    Rin=${args[3]:-0.125}
    Pin=${args[4]:-100.0}
    Pout=${args[5]:-1.0}
    # Set default values if not specified
    echo "Using settings for Sedov blast wave + mhd:"
    echo "    # For -a Select 0 we define and inner and outer pressure and an inner radius of blast. And corresponding inner plasmabeta for mhd.
    #Select 1 and 2 we choose an blast energy and give it to a few particles in centre.
    #In select 1 we have different magnetic strength in the blast and outside blast(need divergence cleaning) and select 2 we have a uniform field."
    echo "Using ublast and setting to few particles(=1,2) or using defined inner pressure and radius(=0) -a : $select"
    echo "Plasma beta of initial blast region -b : $betain"
    echo "Energy of blast(select=1) -c : $ublast"
    echo "Radius of inner blast(select=0) -d : $Rin"
    echo "Pressure inside(select=0) -e : $Pin"
    echo "Pressure outside(select=0) -f : $Pout"

    # Setup based on the potentially updated values
    EXTRACONDIC="$select $betain $ublast $Rin $Pin $Pout"
    EXTRACONDRUN=""
    prefile_extra="select$(fmt "${select}")beta$(fmt "${betain}")ublast$(fmt "${ublast}")"
    glassfile_extra="${GLASSNAME}"
    # Rin/Pin/Pout only shape the IC for select=0 (pressure-jump sphere); for
    # select 1/2 the setup overrides Rin (2*mean(h)) and recomputes Pin/Pout
    # from ublast, so they are kept out of the name there.
    runname_extra="select$(fmt "${select}")beta$(fmt "${betain}")ublast$(fmt "${ublast}")"
    if [ "$(fmt "${select}")" == "0" ]; then
        runname_extra="${runname_extra}Rin$(fmt "${Rin}")Pin$(fmt "${Pin}")Pout$(fmt "${Pout}")"
    fi
}

# Function for alfven setup with generic parameters
setup_evrard() {
    RUNPHYSICS="grav"
    # Set default values if not specifie
    echo "Using settings for evrard collapse:"
    # Setup based on the potentially updated values
    EXTRACONDIC=" "
    EXTRACONDRUN=" "
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra=""
    # evrard's density is dICdensprofile=8 (1/r cusp), which the Python glassgen
    # backend cannot evaluate analytically -> relax against the radial density
    # table the rand pre-IC writes (test_cases/evrard/densitytable.npz, the glass
    # step's cwd). Self-describing table: the r_sph mapping is in its header.
    GLASSMODE="--table densitytable.npz"
}

# Function for alfven setup with generic parameters
setup_polytrope() {
    RUNPHYSICS="grav"
        # Set default values if not specifie
        echo "Using settings for polytrope:"
        # Setup based on the potentially updated values
        EXTRACONDIC=" "
        EXTRACONDRUN=" "
        prefile_extra=""
        glassfile_extra="${GLASSNAME}"
        runname_extra=""
}

# Function for static blobs (force test) setup with generic parameters
setup_areablob() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    rhodiff=${args[0]:-1000}
    multcloud=${args[1]:-1}

    echo "mu is the magnetic flux ratio -a "
    echo "rhodiff is the density difference between central cloud and medium -b "
    echo "Erat is the rotational energy ratio -c "

    echo "Final settings for MHD Collapse:"
    echo "mu: $multcloud, rhodiff: $rhodiff"

    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodiff $multcloud"
    EXTRACONDRUN="  "
    prefile_extra="mult$(fmt "${multcloud}")rd$(fmt "${rhodiff}")${GLASSNAME}"
    glassfile_extra="mult$(fmt "${multcloud}")rd$(fmt "${rhodiff}")${GLASSNAME}"
    runname_extra="mult$(fmt "${multcloud}")rd$(fmt "${rhodiff}")"
}

# Function for Taylor-green setup with generic parameters
setup_taylorgreen() {
    RUNPHYSICS="hydro"
    etavisc=${args[0]:-0.1}
    xivisc=${args[1]:-0.0}
    # Set default values if not specified
    echo "Using settings for Taylorgreen Vortex:"
    echo "eta visc -a : $etavisc"
    echo "xi visc -b : $xivisc"

    # Setup based on the potentially updated values
    EXTRACONDIC=" "
    EXTRACONDRUN=" -PhysViscEta $etavisc -PhysViscXi $xivisc "
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra="eta$(fmt "${etavisc}")xi$(fmt "${xivisc}")"
}

# Function for cloud crushing (wind-cloud) setup with generic parameters
setup_blob() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    rhodiff=${args[0]:-10}
    beta=${args[1]:-0}
    mach=${args[2]:-2.7}
    inflow=${args[3]:-0}
    cool=${args[4]:-0}
    windref=${args[5]:-0}
    winddens=${args[6]:-26}
    tcloud=${args[7]:-4000}
    echo "Using settings for cloud crushing:"
    echo "rho diff -a : $rhodiff"
    echo "plasma beta -b : $beta"
    echo "mach number -c : $mach"
    echo "inflow conditions -d : $inflow"
    echo "with cooling -e : $cool"
    echo "reference frame of wind  -f : $windref"
    echo "density of wind g/cm  -g : $winddens"
    echo "temperature cloud  -h : $tcloud"
    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodiff $beta $mach $inflow $cool $windref $winddens $tcloud "
    EXTRACONDRUN=" "
    prefile_extra="rhodiff$(fmt "${rhodiff}")beta$(fmt "${beta}")mach$(fmt "${mach}")inflow$(fmt "${inflow}")cool$(fmt "${cool}")wr$(fmt "${windref}")wd$(fmt "${winddens}")tc$(fmt "${tcloud}")"
    glassfile_extra="rhodiff$(fmt "${rhodiff}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodiff}")beta$(fmt "${beta}")mach$(fmt "${mach}")inflow$(fmt "${inflow}")cool$(fmt "${cool}")wr$(fmt "${windref}")wd$(fmt "${winddens}")tc$(fmt "${tcloud}")"
}

# Function for Rayleigh-Taylor setup with generic parameters
setup_rt() {
    # Set default values if not specified
    rhodiff=${args[0]:-2.0}
    case=${args[1]:-1}
    nlowdens=${args[2]:-2}
    smooth=${args[3]:-0}
    beta=${args[4]:-0.0}
    bdir=${args[5]:-1}
    # MHD only when a nonzero interface plasma-beta is requested (beta=0 -> hydro).
    RUNPHYSICS="hydro"; [ "$(nonzero "$beta")" = 1 ] && RUNPHYSICS="mhd"
    echo "Using settings for rayleigh-taylor:"
    echo "rho diff -a : $rhodiff"
    echo "Case 1 for symmetric and 2 for unsymmetric -b : $case"
    echo "Number of low density segments (same size as high density) -c : $nlowdens"
    echo "Smooth -d : $smooth"
    echo "Plasma beta at the interface (0 = hydro) -e : $beta"
    echo "B direction 1=x horizontal (default) 2=y 3=z vertical -f : $bdir"
    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodiff $case $nlowdens $smooth $beta $bdir"
    EXTRACONDRUN=" "
    prefile_extra="rhodiff$(fmt "${rhodiff}")smooth$(fmt "${smooth}")nlow$(fmt "${nlowdens}")c$(fmt "${case}")beta$(fmt "${beta}")bdir$(fmt "${bdir}")"
    glassfile_extra="rhodiff$(fmt "${rhodiff}")smooth$(fmt "${smooth}")nlow$(fmt "${nlowdens}")c$(fmt "${case}")beta$(fmt "${beta}")bdir$(fmt "${bdir}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodiff}")smooth$(fmt "${smooth}")nlow$(fmt "${nlowdens}")c$(fmt "${case}")beta$(fmt "${beta}")bdir$(fmt "${bdir}")"
}

# Function for magneto-thermal instability setup
setup_mti() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    echo "Using settings for MTI:"
    # Setup based on the potentially updated values
    EXTRACONDIC=" "
    EXTRACONDRUN=" "
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra=""
}

# Function for isolated MHD wave setup with generic parameters
setup_mhdisowave() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    beta=${args[0]:-0.1}
    echo "Using settings for isolated wave:"
    echo "Plasma beta -a : $beta"
    # Setup based on the potentially updated values
    EXTRACONDIC="$beta "
    EXTRACONDRUN=" "
    prefile_extra="$(fmt "${beta}")"
    glassfile_extra="$(fmt "${beta}")${GLASSNAME}"
    runname_extra="beta$(fmt "${beta}")"
}

# Function for magnetic divergence test setup with generic parameters
setup_mhddivtest() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    rhodiff=${args[0]:-1}
    smooth=${args[1]:-1}
    square=${args[2]:-1}
    cosmo=${args[3]:-0}
    echo "Using settings for divergence test:"
    echo "Density contrast -a : $rhodiff"
    echo "Smooth density -b : $smooth"
    echo "Square test -c : $square"
    echo "Cosmo test -d : $cosmo"
    # Setup based on the potentially updated values
    EXTRACONDIC="$rhodiff $smooth $square $cosmo  "
    EXTRACONDRUN=" "
    prefile_extra="rd$(fmt "${rhodiff}")s$(fmt "${smooth}")sq$(fmt "${square}")"
    glassfile_extra="rd$(fmt "${rhodiff}")s$(fmt "${smooth}")sq$(fmt "${square}")${GLASSNAME}"
    runname_extra="rd$(fmt "${rhodiff}")s$(fmt "${smooth}")sq$(fmt "${square}")cos$(fmt "${cosmo}")"
}

# Function for planet (single body) setup with generic parameters
setup_planet() {
    RUNPHYSICS="grav"
    # Set default values if not specified
    planet=${args[0]:-"Earth_PES095_2layer"}
    echo "Using settings for planet:"
    # Setup based on the potentially updated values
    EXTRACONDIC="$planet"
    EXTRACONDRUN=" "
    prefile_extra="$planet"
    glassfile_extra="$planet${GLASSNAME}"
    runname_extra="$planet"
}

# Function for planet collision setup with generic parameters
setup_planetcollision() {
    RUNPHYSICS="grav"
    # Set default values if not specified
    target=${args[0]:-"None"}
    impactor=${args[1]:-"None"}
    impangle=${args[2]:-0.73}
    vinf=${args[3]:-0.0}
    impgeom=${args[4]:-2}
    echo "Using settings for planet collision:"
    # Setup based on the potentially updated values
    EXTRACONDIC="$target $impactor $impangle $vinf $impgeom"
    EXTRACONDRUN=" "
    prefile_extra="t${target}i${impactor}b$(fmt "${impangle}")g$(fmt "${impgeom}")"
    glassfile_extra="t${target}i${impactor}b$(fmt "${impangle}")g$(fmt "${impgeom}")${GLASSNAME}"
    runname_extra="t${target}i${impactor}b$(fmt "${impangle}")v$(fmt "${vinf}")g$(fmt "${impgeom}")"
}

# Function for hydrostatic-square test
# is3d=0: 2D thin-slab variant (Saitoh-Makino 2013; Hopkins 2013, 2015; Hu et al. 2014)
# is3d=1: 3D inner-cube variant (Sandnes et al. 2025)
setup_square() {
    RUNPHYSICS="hydro"
    gamma_default=$(echo "scale=12; 5/3" | bc -l)
    rhodiff=${args[0]:-4.0}
    przero=${args[1]:-2.5}
    gamma=${args[2]:-$gamma_default}
    is3d=${args[3]:-0}

    echo "Using settings for hydrostatic-square test:"
    echo "density contrast chi -a : $rhodiff"
    echo "uniform pressure P -b : $przero"
    echo "gamma -c : $gamma"
    echo "3D inner-cube (Sandnes 2025) -d : $is3d"

    # Order matches IC_setup_square.create(... rhodiff, przero, gamma, is3d) signature
    EXTRACONDIC="$rhodiff $przero $gamma $is3d "
    EXTRACONDRUN=""
    prefile_extra="chi$(fmt "${rhodiff}")P$(fmt "${przero}")d3$(fmt "${is3d}")"
    glassfile_extra="chi$(fmt "${rhodiff}")d3$(fmt "${is3d}")${GLASSNAME}"
    runname_extra="chi$(fmt "${rhodiff}")P$(fmt "${przero}")g$(fmt "${gamma}")d3$(fmt "${is3d}")"
}

# Function for cosmological MHD wave (Berlok 2022) setup with generic parameters
setup_cosmowave() {
    # Set default values if not specified
    case=${args[0]:-3}
    factor_A=${args[1]:-1}
    factor_G=${args[2]:-0}
    # MHD when factor_A (omega_A) != 0; self-gravity when factor_G (omega_G) != 0.
    RUNPHYSICS="$(combine_physics "$(nonzero "$factor_A")" "$(nonzero "$factor_G")")"
    echo "Using settings for cosmowave (Berlok 2022):"
    echo "test case -a : $case  (0:standing Alfven 1:traveling Alfven 2:standing compr. 3:traveling compr. gamma=4/3 4:div-clean EdS)"
    echo "factor_A (omega_A = factor_A*pi; 0=MHD off, 1=paper default) -b : $factor_A"
    echo "factor_G (omega_G = factor_G*pi/2; 0=self-grav off, 1=paper default) -c : $factor_G"
    # Order matches IC_setup_cosmowave.create(... case, factor_A, factor_G) signature
    EXTRACONDIC="$case $factor_A $factor_G "
    EXTRACONDRUN=" "
    prefile_extra="case$(fmt "${case}")A$(fmt "${factor_A}")G$(fmt "${factor_G}")"
    glassfile_extra="${GLASSNAME}"
    runname_extra="case$(fmt "${case}")A$(fmt "${factor_A}")G$(fmt "${factor_G}")"
}

# Function for Balsara MHD vortex setup with generic parameters
setup_mhdbalsaravortex() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    u=${args[0]:-1.0}
    k=${args[1]:-1.0}
    echo "Using settings for Balsara MHD vortex:"
    echo "magnetic field amplitude -a : $u"
    echo "velocity perturbation amplitude -b : $k"
    # Order matches IC_setup_mhdbalsaravortex.create(... u, k) signature
    EXTRACONDIC="$u $k "
    EXTRACONDRUN=""
    prefile_extra="u$(fmt "${u}")k$(fmt "${k}")"
    glassfile_extra="${GLASSNAME}"
    runname_extra="u$(fmt "${u}")k$(fmt "${k}")"
}

# Function for Zeldovich pancake (cosmological) setup with generic parameters
setup_zeldovich() {
    RUNPHYSICS="grav"
    # Set default values if not specified
    Lboxinkpc=${args[0]:-1000.0}
    zi=${args[1]:-100}
    echo "Using settings for Zeldovich pancake:"
    echo "box size in kpc (length unit) -a : $Lboxinkpc"
    echo "initial redshift zi (must be > collapse redshift zc=2) -b : $zi"
    # Order matches IC_setup_zeldovich.create(... Lboxinkpc, zi) signature
    EXTRACONDIC="$Lboxinkpc $zi "
    EXTRACONDRUN=" "
    prefile_extra="L$(fmt "${Lboxinkpc}")zi$(fmt "${zi}")"
    glassfile_extra="${GLASSNAME}"
    runname_extra="L$(fmt "${Lboxinkpc}")zi$(fmt "${zi}")"
}

# Function for magnetorotational instability (MRI) setup with generic parameters
setup_mri() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    case=${args[0]:-2}
    echo "Using settings for MRI:"
    echo "configuration case -a : $case  (1:thin unstrat nonetflux 2:regular unstrat nonetflux 3:tall unstrat nonetflux 4:stratified netflux 5:unstrat netflux 6:regular unstrat nonetflux 7:- 8:unstrat netflux tall)"
    # Order matches IC_setup_mri.create(... case) signature
    EXTRACONDIC="$case "
    EXTRACONDRUN=""
    prefile_extra="case$(fmt "${case}")"
    glassfile_extra="case$(fmt "${case}")${GLASSNAME}"
    runname_extra="case$(fmt "${case}")"
}

# Function for Magnetized advection loop with shear setup with generic parameters
setup_mhdloopshear() {
    RUNPHYSICS="mhd"
    rhodiff=${args[0]:-1.0}
    # Set default values if not specified
    echo "Using settings for Magnetized Advection Loop with shear:"
    echo "density contrast -a : $rhodiff"
    # Order matches IC_setup_mhdloopshear.create(... rhodiff) signature
    EXTRACONDIC="$rhodiff "
    EXTRACONDRUN=""
    prefile_extra="rhodiff$(fmt "${rhodiff}")"
    glassfile_extra="rhodiff$(fmt "${rhodiff}")${GLASSNAME}"
    runname_extra="rhodiff$(fmt "${rhodiff}")"
}

# Function for MHD current sheet (Gardiner & Stone 2005) setup with generic parameters
setup_currentsheet() {
    RUNPHYSICS="mhd"
    # Set default values if not specified
    beta=${args[0]:-0.1}
    v0=${args[1]:-0.1}
    echo "Using settings for MHD current sheet:"
    echo "plasma beta (sets gas pressure = beta*B0^2/2) -a : $beta"
    echo "velocity perturbation amplitude v0 -b : $v0"
    # Order matches IC_setup_currentsheet.create(... beta, v0) signature
    EXTRACONDIC="$beta $v0 "
    EXTRACONDRUN=""
    prefile_extra="beta$(fmt "${beta}")v$(fmt "${v0}")"
    glassfile_extra="${GLASSNAME}"
    runname_extra="beta$(fmt "${beta}")v$(fmt "${v0}")"
}

# Function for the Noh problem (converging strong shock) setup with generic parameters
setup_noh() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    is3d=${args[0]:-1}
    pini=${args[1]:-1e-6}
    v0=${args[2]:-1.0}
    echo "Using settings for Noh problem:"
    echo "3D spherical (1) / 2D cylindrical (0) -a : $is3d"
    echo "initial (tiny) pressure -b : $pini"
    echo "radial inflow speed v0 -c : $v0"
    # Order matches IC_setup_noh.create(... is3d, pini, v0) signature
    EXTRACONDIC="$is3d $pini $v0 "
    EXTRACONDRUN=""
    prefile_extra="d3$(fmt "${is3d}")"
    glassfile_extra="d3$(fmt "${is3d}")${GLASSNAME}"
    runname_extra="d3$(fmt "${is3d}")p$(fmt "${pini}")v$(fmt "${v0}")"
}

# Function for the isentropic (Yee) vortex setup with generic parameters
setup_isentropicvortex() {
    RUNPHYSICS="hydro"
    # Set default values if not specified
    beta_v=${args[0]:-5.0}
    vadv=${args[1]:-1.0}
    echo "Using settings for isentropic (Yee) vortex:"
    echo "vortex strength beta_v -a : $beta_v"
    echo "mean-flow (advection) velocity vadv -b : $vadv"
    # Order matches IC_setup_isentropicvortex.create(... beta_v, vadv) signature
    EXTRACONDIC="$beta_v $vadv "
    EXTRACONDRUN=""
    prefile_extra="b$(fmt "${beta_v}")"
    glassfile_extra="b$(fmt "${beta_v}")${GLASSNAME}"
    runname_extra="b$(fmt "${beta_v}")vadv$(fmt "${vadv}")"
}

# Function for linear (M)HD wave setup with generic parameters
setup_wave() {
    # Set default values if not specified
    wavetype=${args[0]:-0}
    ampl=${args[1]:-1e-4}
    # wavetype 0 = sound (hydro); 1/2/3 = (fast/slow magnetosonic, Alfven) = MHD.
    RUNPHYSICS="mhd"; [ "$wavetype" = 0 ] && RUNPHYSICS="hydro"
    echo "Using settings for linear (M)HD wave:"
    echo "wave type -a : $wavetype  (0:sound 1:fast magnetosonic 2:slow magnetosonic 3:Alfven)"
    echo "eigenmode amplitude -b : $ampl"
    # Order matches IC_setup_wave.create(... wavetype, ampl) signature
    EXTRACONDIC="$wavetype $ampl "
    EXTRACONDRUN=""
    prefile_extra="wt$(fmt "${wavetype}")a$(fmt "${ampl}")"
    glassfile_extra="wt$(fmt "${wavetype}")a$(fmt "${ampl}")${GLASSNAME}"
    runname_extra="wt$(fmt "${wavetype}")a$(fmt "${ampl}")"
}

# Function for the cold Keplerian disk/ring setup with generic parameters
setup_coldkeplerian() {
    RUNPHYSICS="grav"
    # Set default values if not specified
    hr=${args[0]:-0.05}
    rinner=${args[1]:-1.0}
    rdisk=${args[2]:-2.0}
    q=${args[3]:-0.5}
    echo "Using settings for cold Keplerian disk/ring:"
    echo "aspect ratio hr (=1/Mach; small=cold) -a : $hr"
    echo "inner ring radius -b : $rinner"
    echo "outer ring radius -c : $rdisk"
    echo "rotation power-law q (v=r^-q) -d : $q"
    # Order matches IC_setup_coldkeplerian.create(... hr, rinner, rdisk, q) signature
    EXTRACONDIC="$hr $rinner $rdisk $q "
    EXTRACONDRUN=""
    prefile_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")"
    glassfile_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")${GLASSNAME}"
    runname_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")"
}


# Main execution block
case $testsim in
    accdisk)
        setup_accdisk
        ;;
    alfven)
        setup_alfven
        ;;
    gresho)
        setup_gresho
        ;;
    mhdcollapse)
        setup_mhdcollapse
        ;;
    kh)
        setup_kh
        ;;
    orzag)
        setup_orzag
        ;;
    abcflow)
        setup_abcflow
        ;;
    ponomarenko)
        setup_ponomarenko
        ;;
    mhdrotor)
        setup_mhdrotor
        ;;
    mhdloop)
        setup_mhdloop
        ;;
    shocktube)
        setup_shocktube
        ;;
    sedov)
        setup_sedov
	;;
    evrard)
        setup_evrard
        ;;
    polytrope)
        setup_polytrope
        ;;
    taylorgreen)
        setup_taylorgreen
        ;;
    areablob)
        setup_areablob
        ;;
    blob)
        setup_blob
        ;;
    rt)
        setup_rt
        ;;
    mti)
        setup_mti
        ;;
    mhdisowave)
        setup_mhdisowave
        ;;
    mhddivtest)
        setup_mhddivtest
        ;;
    planet)
        setup_planet
        ;;
    planetcollision)
        setup_planetcollision
        ;;
    square)
        setup_square
        ;;
    cosmowave)
        setup_cosmowave
        ;;
    mhdbalsaravortex)
        setup_mhdbalsaravortex
        ;;
    zeldovich)
        setup_zeldovich
        ;;
    mri)
        setup_mri
        ;;
    mhdloopshear)
        setup_mhdloopshear
        ;;
    currentsheet)
        setup_currentsheet
        ;;
    noh)
        setup_noh
        ;;
    isentropicvortex)
        setup_isentropicvortex
        ;;
    wave)
        setup_wave
        ;;
    coldkeplerian)
        setup_coldkeplerian
        ;;
    *)
        echo "Error: Unknown test simulation '$testsim'."
	usage
        exit 1
        ;;
esac

mkdir test_cases
mkdir test_cases/$testsim
cd test_cases/$testsim

CURRENT_DIR=$(pwd -P)

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <Nsmooth> <Nres> <0: lattice 1: rand 2: glass> <code 0:Gasoline 1:ChaNGa 2:SPH-EXA 3:AREPO 4:RAMSES 5:GIZMO 6:Phantom 7:Athena++ 8:PKDGRAV3 9:OpenGadget3> [vm] [extra]"
    exit 1
fi
echo "$0 $1 $2 $3 $4 $5 $6"

# Append the optional extra positional flags ($6) to the runtime conditions
echo "$EXTRACONDALL"
EXTRACONDRUN=" $EXTRACONDRUN $EXTRACONDALL $6 "

# Varied-mass argument ($5): changes the particle distribution itself, so it
# must key the pre-IC file, the glass AND the run name (empty tag when off).
if [ -z "$5" ] || [ "$5" -le 1 ]; then
            vm=0.0
    else
                vm=$5
fi
if [ "$vm" != "0.0" ]; then
    vmtag="vm$(fmt "${vm}")"
else
    vmtag=""
fi
prefile_extra="${prefile_extra}${vmtag}"
glassfile_extra="${glassfile_extra}${vmtag}"
runname_extra="${runname_extra}${vmtag}"

# Set distribution and d1 based on the third command-line argument.
# prefile = the file IC_createsetup.py writes directly into datafiles/:
#  - lattice/rand: that file IS the final IC (velocities/B field included), so it
#    must be keyed by the FULL parameter set (runname_extra) -- otherwise a rerun
#    with e.g. a different B0 silently reuses the cached IC of the old run.
#  - glass: it is only the pre-relaxation density distribution, so it is keyed by
#    the density-structure params (prefile_extra) and shared across runs that
#    differ only in velocities/B; the final IC (built from the relaxed glass) is
#    keyed by runname_extra instead.
if [ "$3" -eq 0 ]; then
echo "RUNNING LATTICE MODE"
    distri="lattice"
    distri2="lattice"
    d1=0
    simfilename="$testsim$2_$distri${runname_extra}"
    prefile="$simfilename"
elif [ "$3" -eq 1 ]; then
echo "RUNNING RANDOM MODE"
    distri="rand"
    distri2="rand"
    d1=1
    simfilename="$testsim$2_$distri${runname_extra}"
    prefile="$simfilename"
else
echo "RUNNING GLASS MODE"
    distri="rand"
    distri2="glass"
    d1=1
    simfilename="$testsim$2_glassN$1${runname_extra}-I"
    prefile="$testsim$2_${distri}${prefile_extra}"
    echo "$simfilename"
fi

# Code name and binary/launcher selection based on the 4th argument
if [ "$4" -eq 0 ]; then
echo "RUNNING GASOLINE"
    codename="GASOLINE"
    dir=$GASOLINE_DIR
    coderun=gasolinemhd
elif [ "$4" -eq 1 ]; then
echo "RUNNING CHANGA"
    codename="CHANGA"
    dir=$CHANGA_DIR
    coderun=ChaNGa.smp
    runlauncher="$CHANGA_DIR/charmrun ${CHANGA_CHARMRUN_OPTS:-+p $(nproc)}"
elif [ "$4" -eq 2 ]; then
echo "RUNNING SPH-EXA"
    codename="SPHEXA"
    dir=$SPHEXA_DIR
    coderun=sphexa
elif [ "$4" -eq 3 ]; then
echo "RUNNING AREPO"
    codename="AREPO"
    dir=$AREPO_DIR
    coderun=Arepo
elif [ "$4" -eq 4 ]; then
echo "RUNNING RAMSES"
    codename="RAMSES"
    dir=$RAMSES_DIR
    coderun=ramses3d
elif [ "$4" -eq 5 ]; then
echo "RUNNING GIZMO"
    codename="GIZMO"
    dir=$GIZMO_DIR
    coderun=GIZMO
elif [ "$4" -eq 6 ]; then
echo "RUNNING PHANTOM"
    codename="PHANTOM"
    dir=$PHANTOM_DIR
    coderun=phantom
elif [ "$4" -eq 7 ]; then
echo "RUNNING ATHENA"
    codename="ATHENA"
    dir=$ATHENA_DIR
    coderun=athena_hydro
elif [ "$4" -eq 8 ]; then
echo "RUNNING PKDGRAV3"
    codename="PKDGRAV3"
    dir=$PKDGRAV3_DIR
    coderun=pkdgrav3
elif [ "$4" -eq 9 ]; then
echo "RUNNING OPENGADGET3"
    codename="OPENGADGET3"
    dir=$OPENGADGET3_DIR
    coderun=OpenGadget3
fi

# RUNPHYSICS support gate: refuse (code, physics) pairs this script can't run,
# BEFORE building any IC. RUNPHYSICS was declared by setup_<testsim>() above; the
# code blocks below map it to the per-code binary/config.
if [ -z "${RUNPHYSICS:-}" ]; then
    echo "Internal error: setup_$testsim did not set RUNPHYSICS."; exit 1
fi
if ! physics_supported "$4" "$RUNPHYSICS"; then
    echo "Error: $codename does not support $RUNPHYSICS tests (test '$testsim')."
    echo "       $codename supports: ${SUPPORT[$4]}."
    echo "       (See 'runtest.sh support'; edit the SUPPORT matrix as codes gain physics.)"
    exit 1
fi
echo "Physics: $testsim -> RUNPHYSICS=$RUNPHYSICS ($codename)"

# All codes 0-9 are wired: 0 Gasoline, 1 ChaNGa, 2 SPH-EXA, 3 AREPO, 4 RAMSES
# (cubic first cut, hydro+MHD), 5 GIZMO, 6 Phantom, 7 Athena++ (cubic first
# cut, hydro+MHD, uniform), 8 PKDGRAV3, 9 OpenGadget3.

glassfile_prefix="$testsim$2_glassN$1${glassfile_extra}"
ICfile="${simfilename}"
runname="$testsim$2_N$1${distri2}${runname_extra}${EXTRANAME}${codename}"
directory="initruns"
glassfile="${directory}/${glassfile_prefix}"

mkdir datafiles
mkdir initruns



    # Check if the file already exists
    if [ -f "datafiles/$prefile.00000" ]; then
        echo "File $prefile.00000 already exists. Skipping creation."
    else
    # File doesn't exist, create it
    preIC="python ${analysisdir}/IC_createsetup.py $testsim $2 $vm $d1 $prefile $EXTRACONDIC"
    echo $preIC
    eval $preIC
    # Check if the command was successful
    if [ $? -ne 0 ]; then
    echo "Error: Command failed to execute successfully."
    exit 1
    fi
    
    fi

##Glass relaxation    
    if [ "$3" -ne 0 ] && [ "$3" -ne 1 ]; then
    if [ -f "${glassfile}_IC" -o -f "datafiles/$ICfile.00000" ]; then
        echo "File ${glassfile}_IC already exists. Skipping creatin.."
    else
	# Python (numba) glass relaxation - glassgen.cli writes ${glassfile}_IC
	# directly; target density is read from the .param.
	relaxglass=(env PYTHONPATH="${glassgenpydir}" python -m glassgen.cli -o "$glassfile" -I datafiles/$prefile.00000 -s $1 $ICCONFIGPY --param datafiles/$prefile.param $GLASSMODE)
	echo "${relaxglass[@]}"
	"${relaxglass[@]}"
	if [ $? -ne 0 ]; then
	    echo "Error: Command failed to execute successfully."
	    exit 1
	fi
    fi

    if [ -f "datafiles/$ICfile.00000" ]; then
        echo "File $ICfile.00000 already exists. Skipping creation."
    else
    ICfromglass="python ${analysisdir}/IC_createsetup.py $testsim $2 $vm "${glassfile}_IC" ${ICfile} $EXTRACONDIC "
    echo $ICfromglass
    eval $ICfromglass

    # Check if the command was successful
    if [ $? -ne 0 ]; then
    echo "Error: Command failed to execute successfully."
    exit 1
    fi
    
    fi
    fi
    mkdir $runname
    echo "Code selection: $4"
    if [ "$4" == 0 ] || [ "$4" == 1 ]; then
    if [ "$4" == 1 ]; then sed -i '/^dCloudDensity/d' datafiles/$ICfile.param; fi
        runline=( ${runlauncher:-$mpirun} $dir/${coderun} -s $1 -o $runname/$runname -I datafiles/$ICfile.00000 $EXTRACONDRUN datafiles/$ICfile.param)
    fi

    if [ "$4" == 2 ]; then
	# SPH-EXA
	paramfile="datafiles/$ICfile.param"

	get_param () {
	    awk -F= -v key="$1" '
		$1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
		    gsub(/[[:space:]]*/, "", $2)
		    print $2
		}
	    ' "$paramfile"
	}

	iOutInterval=$(get_param iOutInterval)
	dDelta=$(get_param dDelta)
	nSteps=$(get_param nSteps)

	# Sanity check
	if [[ -z "$iOutInterval" || -z "$dDelta" || -z "$nSteps" ]]; then
	    echo "Error: missing required parameters in $paramfile"
	    exit 1
	fi

	# Compute physical times
	outinterval=$(awk -v io="$iOutInterval" -v dd="$dDelta" 'BEGIN { printf "%.8f", io * dd }')
	endtime=$(awk -v ns="$nSteps" -v dd="$dDelta" 'BEGIN { printf "%.8f", ns * dd }')

	# load special modules for sph-exa
	module purge
	#module load nvhpc/nvhpc-hpcx-cuda12/24.7
	module load gcc/13.3.1
	module load openmpi/gcc13/5.0.9

	# convert the tipsy IC into the SPH-EXA hdf5 format
	echo "python ${analysisdir}/ICconvert_tipsytosphexa.py datafiles/$ICfile.00000 datafiles/$ICfile "
	python ${analysisdir}/ICconvert_tipsytosphexa.py datafiles/$ICfile.00000 datafiles/$ICfile
	runline=( "$dir/$coderun" --init "./datafiles/$ICfile.h5" -o "$runname/$runname.h5" $EXTRACONDSPHEXA -s "$endtime" -w "$outinterval" -f "$fields")
    fi

    if [ "$4" == 3 ]; then
	# AREPO moving-mesh: convert the tipsy IC -> AREPO HDF5 IC (sph_interp),
	# then build an AREPO param.txt whose BoxSize / run times come from the
	# tipsy .param we generated for this lattice (NOT the particle extent).
	paramfile="datafiles/$ICfile.param"

	get_param () {
	    awk -F= -v key="$1" '
		$1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
		    gsub(/[[:space:]]*/, "", $2); print $2
		}' "$paramfile"
	}

	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	arepo_boxopt=""
	if [ "$RUNPHYSICS" = grav ]; then
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
	    LONGY=$(awk -v a="$dyP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
	    LONGZ=$(awk -v a="$dzP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')

	    # Non-cubic box -> AREPO needs LONG_Y/LONG_Z compiled in (public AREPO is
	    # cubic by default). Cubic (both ~1): use the prebuilt MHD `Arepo`. Else
	    # build (once, cached by aspect tag) Arepo_asp<LY>_<LZ> from Config.sh + the
	    # LONG_Y/LONG_Z lines.
	    iscubic=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { d=0.0005; print (y>1-d && y<1+d && z>1-d && z<1+d)?1:0 }')
	    if [ "$iscubic" != "1" ]; then
		tag=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { printf "%.4f_%.4f", y, z }' | tr '.' 'p')
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
	Nres=$2
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
    fi

    if [ "$4" == 5 ]; then
	# GIZMO (particle code): 1:1 copy the tipsy SPH particles into the GADGET
	# type-0 HDF5 GIZMO reads (no interpolation), and write a GIZMO
	# parameterfile from the gasoline .param. Snapshots land in $runname/output/.
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	dxP=$(get_param dxPeriod); dyP=$(get_param dyPeriod); dzP=$(get_param dzPeriod)
	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# Non-periodic IC (bPeriodic=0, e.g. isolated evrard): the .param has no
	# dxPeriod. The box handling below keys on this, INDEPENDENT of gravity.
	isperiodic=1; [ "$(get_param bPeriodic)" = "0" ] && isperiodic=0
	# Binary/config from the TEST's declared RUNPHYSICS (not sniffed from the IC).
	# The converter still auto-detects MHD (B aux) / gravity (bDoGravity) itself,
	# so it agrees with this by construction and stays usable standalone.
	isgrav=0
	case "$RUNPHYSICS" in
	    hydro) gizmo_base="GIZMO";      gizmo_cfg="Config_hydro.sh" ;;
	    mhd)   gizmo_base="GIZMO_mhd";  gizmo_cfg="Config_mhd.sh" ;;
	    grav)  gizmo_base="GIZMO_grav"; gizmo_cfg="Config_grav.sh"; isgrav=1
		   # Isolated self-gravity binary (Config_grav.sh: no BOX_PERIODIC /
		   # no SELFGRAVITY_OFF / +ADAPTIVE_GRAVSOFT_FORGAS); built on demand.
		   if [ ! -x "$dir/$gizmo_base" ]; then
			echo "Building isolated self-gravity GIZMO -> $gizmo_base (Config_grav.sh)"
			( cd "$dir" && make clean EXEC="$gizmo_base" >/dev/null 2>&1 || true
			  make -j8 CONFIG="$gizmo_cfg" EXEC="$gizmo_base" )
			if [ ! -x "$dir/$gizmo_base" ]; then echo "Error: GIZMO_grav build failed (run codes/gizmo/build_gizmo.sh)."; exit 1; fi
		   fi ;;
	    *) echo "Error: GIZMO has no build for RUNPHYSICS=$RUNPHYSICS."; exit 1 ;;
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
	    LONGY=$(awk -v a="$dyP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
	    LONGZ=$(awk -v a="$dzP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
	    iscubic=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { d=0.0005; print (y>1-d && y<1+d && z>1-d && z<1+d)?1:0 }')
	    if [ "$iscubic" != "1" ]; then
		tag=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { printf "%.4f_%.4f", y, z }' | tr '.' 'p')
		coderun_run="${gizmo_base}_asp${tag}"
		if [ ! -x "$dir/$coderun_run" ]; then
		    echo "Building GIZMO with BOX_LONG_Y=$LONGY BOX_LONG_Z=$LONGZ -> $coderun_run"
		    cfg="$(basename "$gizmo_cfg" .sh)_asp${tag}.sh"
		    cp "$dir/$gizmo_cfg" "$dir/$cfg"
		    printf "\nBOX_LONG_Y=%s\nBOX_LONG_Z=%s\n" "$LONGY" "$LONGZ" >> "$dir/$cfg"
		    # clean+build with EXEC override so this aspect build only removes/links
		    # its OWN binary (the cubic base binaries survive).
		    ( cd "$dir" && make clean EXEC="$coderun_run" >/dev/null 2>&1 || true
		      make -j8 CONFIG="$cfg" EXEC="$coderun_run" )
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
    fi

    if [ "$4" == 9 ]; then
	# OpenGadget3 (Gadget-family TreeSPH/SPMHD particle code): 1:1 copy the tipsy
	# SPH particles into the GADGET type-0 HDF5 it reads (no interpolation; same
	# writer as GIZMO/AREPO), and write a GADGET-style parameterfile from the
	# gasoline .param. Snapshots land in $runname/output/ as snap_NNN.hdf5.
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	iOutInterval=$(get_param iOutInterval)
	if [ -z "$iOutInterval" ]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# Binary/config from the TEST's declared RUNPHYSICS (not sniffed from the IC);
	# the converter auto-detects MHD (B aux) / gravity (bDoGravity) itself (standalone).
	case "$RUNPHYSICS" in
	    hydro) og_base="OpenGadget3";      og_cfg="Config_hydro.sh" ;;
	    mhd)   og_base="OpenGadget3_mhd";  og_cfg="Config_mhd.sh" ;;
	    grav)  og_base="OpenGadget3_grav"; og_cfg="Config_grav.sh" ;;
	    *) echo "Error: OpenGadget3 has no build for RUNPHYSICS=$RUNPHYSICS."; exit 1 ;;
	esac
	coderun_run="$og_base"
	boxopt=""
	if [ "$RUNPHYSICS" = grav ]; then
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
	    LONGY=$(awk -v a="$dyP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
	    LONGZ=$(awk -v a="$dzP" -v b="$dxP" 'BEGIN { printf "%.10g", a/b }')
	    iscubic=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { d=0.0005; print (y>1-d && y<1+d && z>1-d && z<1+d)?1:0 }')
	    if [ "$iscubic" != "1" ]; then
		tag=$(awk -v y="$LONGY" -v z="$LONGZ" 'BEGIN { printf "%.4f_%.4f", y, z }' | tr '.' 'p')
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
    fi

    if [ "$4" == 8 ]; then
	# PKDGRAV3 (particle code): reads the tipsy IC natively (NO conversion) and
	# writes tipsy snapshots. Just generate a .par from the gasoline .param,
	# pointing achInFile at the tipsy IC and snapshots into $runname/.
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	dxP=$(get_param dxPeriod); iOutInterval=$(get_param iOutInterval)
	if [[ -z "$dxP" || -z "$iOutInterval" ]]; then echo "Error: missing dxPeriod/iOutInterval in $paramfile"; exit 1; fi
	pkdpar="$runname/run.par"
	gencmd="python ${analysisdir}/IC_pkdgrav3_param.py datafiles/$ICfile.00000 $runname $pkdpar --boxsize $dxP"
	echo "$gencmd"; eval "$gencmd"
	if [ $? -ne 0 ]; then echo "Error: PKDGRAV3 .par generation failed."; exit 1; fi
	runline=( $mpirun "$dir/${coderun}" "$pkdpar" )
    fi

    if [ "$4" == 6 ]; then
	# PHANTOM (SPH particle code): 1:1 copy the tipsy SPH particles into a
	# Phantom full-dump IC (no interpolation) + a .in run file. Phantom runs as
	# `phantom <prefix>.in` and writes dumps <prefix>_NNNNN into the cwd, so run
	# from inside $runname/.
	# Need iOutInterval (snapshot stride) for the reverse converter below.
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	phantomprefix="$runname"
	# Binary from the TEST's declared RUNPHYSICS (not sniffed). The hydro `phantom`
	# binary silently ignores any magnetic block and evolves pure hydro; phantom_mhd
	# is the MHD build; phantom_grav is the self-gravity build (SETUP=evrard:
	# GRAVITY=yes, adiabatic, non-periodic). The converter auto-detects MHD (real B)
	# and self-gravity (G_code=1 units when bDoGravity) from the IC, so it needs no
	# physics flag.
	case "$RUNPHYSICS" in
	    hydro) coderun=phantom ;;
	    mhd)   coderun=phantom_mhd ;;
	    grav)  coderun=phantom_grav ;;
	    *) echo "Error: PHANTOM has no build for RUNPHYSICS=$RUNPHYSICS."; exit 1 ;;
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
	export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
	patchscript="$current_dir/setupfiles/phantom_patch_infile.py"
	runline=( bash -c "cd $runname \
	    && cp $phantomprefix.in $phantomprefix.in.user \
	    && { env -u GFORTRAN_CONVERT_UNIT mpirun -np 1 $dir/${coderun} $phantomprefix.in || true; } \
	    && python $patchscript $phantomprefix.in --from $phantomprefix.in.user \
	    && env -u GFORTRAN_CONVERT_UNIT mpirun -np 1 $dir/${coderun} $phantomprefix.in" )
    fi

    if [ "$4" == 4 ]; then
	# RAMSES (grid AMR code): no file-based hydro IC — deposit the SPH IC onto
	# a fine uniform GridIC (sph_interp petkova) + generate run.nml; the PATCHED
	# condinit (codes/ramses/patches/suite, condinit_kind='gridic') reads it.
	# First cut (codes/GRID_CODES_PLAN.md): CUBIC boxes only (the converter
	# fails loudly on slabs), no self-gravity. MHD supported: a genuinely
	# nonzero BField aux routes to ramses3d_mhd (suite_mhd gridic condinit)
	# and the converter writes the hlld MHD namelist + face-B GridIC.
	# Serial binary; parallelism not needed at test sizes.
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# GridIC resolution: the BASE level matches the SPH per-axis resolution,
	# npx = 2**levelmin = Nres — so Nres must be a power of two.
	Nres=$2
	lmax=$(awk -v n="$Nres" 'BEGIN { l=int(log(n)/log(2)+0.5); print ((2^l==n)?l:-1) }')
	if [ "$lmax" == "-1" ]; then echo "Error: RAMSES needs Nres = a power of 2 (got $Nres)."; exit 1; fi
	# MHD binary from the TEST's declared RUNPHYSICS. The converter independently
	# auto-detects the same nonzero B (_has_nonzero_b) to write the MHD namelist +
	# face-B GridIC, so binary and namelist always agree; hydro tests (sedov) stay
	# on ramses3d.
	if [ "$RUNPHYSICS" = mhd ]; then
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
	# Parallel by default like the other codes: RAMSESNP defaults to $(nproc).
	# Resolved BEFORE the conversion so the namelist's per-rank ngridmax is
	# scaled to the rank count; RAMSESNP=1 forces the serial binary.
	RAMSESNP="${RAMSESNP:-$(nproc)}"
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
	# RAMSESNP (resolved above, default nproc) > 1 selects the MPI twin
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
    fi

    if [ "$4" == 7 ]; then
	# Athena++ (grid, UNIFORM mesh) — same GridIC layer as RAMSES:
	# tipsy -> ic_gridic.bin (petkova; 6 solenoidal face-B components when B
	# is genuinely nonzero) + athinput -> the from_gridic problem generator.
	# NON-CUBIC (slab) boxes supported: the converter scales the per-axis
	# cell counts by the box aspect (kh + the 2D orzag slabs run here even
	# though RAMSES's 2^levelmin cube can't). MPI by default (ATHENANP ranks,
	# default nproc; built against the local parallel HDF5 — see
	# codes/athena++/build_athena.sh), capped at the meshblock count the
	# converter prints (ranks > meshblocks aborts Athena++; meshblock sizes
	# are divisors of each axis' cell count, so no power-of-2 Nres needed).
	paramfile="datafiles/$ICfile.param"
	get_param () { awk -F= -v key="$1" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" { gsub(/[[:space:]]*/,"",$2); print $2 }' "$paramfile"; }
	iOutInterval=$(get_param iOutInterval)
	if [[ -z "$iOutInterval" ]]; then echo "Error: missing iOutInterval in $paramfile"; exit 1; fi
	# npx = Nres = the SPH per-axis X resolution; other axes follow the aspect
	Nres=$2
	# MHD binary from the TEST's declared RUNPHYSICS; the converter independently
	# auto-detects the same nonzero B for the CT face-B GridIC (they always agree).
	if [ "$RUNPHYSICS" = mhd ]; then
	    coderun=athena_mhd
	    echo "MHD test -> $coderun (CT, gridic face-B)"
	fi
	ATHENANP="${ATHENANP:-$(nproc)}"
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
    fi
    echo "${runline[@]}"
    "${runline[@]}"

    if [ "$4" == 2 ]; then
	# convert the SPH-EXA hdf5 output back into tipsy snapshots
	python $current_dir/setupfiles/ICconvert_sphexatotipsy.py $runname/$runname.h5 $runname $iOutInterval
    fi

    if [ "$4" == 3 ]; then
	# convert the AREPO snap_*.hdf5 output back into tipsy snapshots so the
	# analysis pipeline (which reads tipsy) can process the AREPO run. Pass the
	# tipsy .param so the per-axis periods (non-cubic box) centre the snapshots
	# correctly and land in the .log.
	python $current_dir/setupfiles/ICconvert_arepototipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    fi

    if [ "$4" == 4 ]; then
	# convert RAMSES output_NNNNN (in the run dir) back into tipsy snapshots:
	# the shipped amr2cell dumps one ASCII row per LEAF cell (all hydro vars in
	# one pass) -> cells_to_tipsy (one gas particle per cell, AMR structure
	# preserved) + gasoline-style .log with the dTuFac stamp. env -u: amr2cell
	# (gfortran) must read RAMSES's native-LE outputs — see the runline note.
	env -u GFORTRAN_CONVERT_UNIT python $current_dir/setupfiles/ICconvert_ramsestotipsy.py $runname/output $runname $iOutInterval datafiles/$ICfile.param --amr2cell $RAMSES_DIR/../utils/f90/amr2cell
    fi

    if [ "$4" == 7 ]; then
	# convert Athena++ gridic.out2.NNNNN.athdf (in the run dir) back into
	# tipsy snapshots: one gas particle per cell via cells_to_tipsy, MHD Bcc
	# -> BField aux, .log with the dTuFac stamp.
	python $current_dir/setupfiles/ICconvert_athenatotipsy.py $runname/output $runname $iOutInterval datafiles/$ICfile.param
    fi

    if [ "$4" == 5 ]; then
	# convert GIZMO snapshot_*.hdf5 output back into tipsy snapshots (same GADGET
	# layout as AREPO; reuses that field mapping). Pass the tipsy .param for box
	# centring.
	python $current_dir/setupfiles/ICconvert_gizmototipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    fi

    if [ "$4" == 9 ]; then
	# convert OpenGadget3 snap_*.hdf5 output back into tipsy snapshots (same GADGET
	# layout as AREPO/GIZMO; reuses that field mapping). Pass the tipsy .param for
	# box centring + dTuFac.
	python $current_dir/setupfiles/ICconvert_opengadget3totipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    fi

    if [ "$4" == 8 ]; then
	# PKDGRAV3 snapshots are already tipsy (<runname>/<runname>.<NNNNN>); just
	# write the gasoline-style .log (box params) the analysis framework needs.
	python $current_dir/setupfiles/ICconvert_pkdgrav3totipsy.py $runname $runname $iOutInterval datafiles/$ICfile.param
    fi

    if [ "$4" == 6 ]; then
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
    fi

## Reference (regression baseline): -r saves the analysis result of this run as
## '<runname>_reference.json' so later runs can be compared against it. We forward
## the same -a..-h values used for the IC (only those the user set; unset ones
## fall back to defaults that the analysis script shares with this setup), so the
## analysis parameters always match the IC parameters.
if [ "$doreference" -eq 1 ]; then
    analysisscript="${analysisdir}/IC_analysis_${testsim}.py"
    if [ -f "$analysisscript" ]; then
        refflags=""
        letters=(a b c d e f g h)
        for i in 0 1 2 3 4 5 6 7; do
            if [ -n "${args[$i]}" ]; then
                refflags="$refflags -${letters[$i]} ${args[$i]}"
            fi
        done
        echo "Saving analysis reference: python $analysisscript $runname $refflags --save-reference"
        python "$analysisscript" "$runname" $refflags --save-reference
    else
        echo "No analysis script $analysisscript for '$testsim'; skipping -r reference save."
    fi
fi
