#!/bin/bash

# Per-code directories; change a line (or set it in the environment) to use
# another tree.
codes_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/codes"
# Gasoline and ChaNGa are compiled per test from a SOURCE tree by their build
# adapter. Empty = the public tree codes/<code>/<code>-public, the default set in
# codes/<code>/build_config.sh. Fill in a path to build another tree, e.g. a
# private branch; that also selects codes/<code>/build_config_private.sh unless
# <CODE>_BUILD_CONFIG names a config. <CODE>_BINARY pins a binary instead.
export GASOLINE_DIR="${GASOLINE_DIR:-/mn/stornext/d17/extragalactic/personal/robertwi/Projects/ICGEN/sims/OPTKERNELSIMS/newtests/gasoline_claude/gasoline/}"
export CHANGA_DIR="${CHANGA_DIR:-/mn/stornext/d17/extragalactic/personal/robertwi/Projects/ICGEN/sims/OPTKERNELSIMS/newtests/ChaNGA_claude/hpc_setup/CHANGA/changaMHDTP/}"
export GASOLINE_BUILD_CONFIG="${GASOLINE_BUILD_CONFIG:-$codes_dir/gasoline/build_config${GASOLINE_DIR:+_private}.sh}"
export CHANGA_BUILD_CONFIG="${CHANGA_BUILD_CONFIG:-$codes_dir/changa/build_config${CHANGA_DIR:+_private}.sh}"
# The other codes: the directory their binary was built into.
export SPHEXA_DIR="${SPHEXA_DIR:-$codes_dir/sphexa/sphexa/build/main/src/sphexa/}"
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
#Python (numba) glass-relaxation IC generator package root
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
#  4=RAMSES 7=Athena++  (wired: build + IC conversion; cubic boxes only)
#The executable directory and binary name per code are set in the code-selection block after arg parsing.
#Each code is obtained/built by codes/<code>/get_<code>.sh + build_<code>.sh.

# --- RUNPHYSICS capability matrix ----------------------------------------------
# RUNPHYSICS is a SET of capability tokens, not one enumerated label. A BARE
# token (hydro|mhd|grav|starform|turbdriving) is a boolean capability; a key=value
# token (eos=barotropic, cooling=metal, starform=divvoff) picks one value on a
# named axis. Each setup_<test>() declares the set its test needs and the gate
# below refuses, before any IC is built, whatever the chosen code is not WIRED
# for in THIS script today.
#
# A row says what a code can do: a bare token, or key=v1|v2 for the values it
# accepts on that axis. Two rules keep the gate honest while the matrix fills in:
#   * a capability a row does not list is REFUSED, valued or not -- absent means
#     the code cannot do it.  `starform` and `starform=divvoff` are both
#     capabilities, so a code without them is refused either way;
#   * the two axes in PHYSICS_UNGATED_AXES (eos, cooling) are the exception:
#     a row that does not mention one is "not yet characterised" and ungated.
#     Mention the key in any form and every value of it is enforced from then on.
# A row states CAPABILITY, not what has been exercised: it answers "can this code
# do this at all", and the question "is there a build for the whole set" is the
# BUILD side's to answer, because only it knows. There is deliberately no
# `grav+mhd` token: a code that lists grav and mhd is claiming both, and if it
# ships no combined binary its own adapter says so by name -- one statement of
# the fact, in the layer that owns it, instead of a second copy here that has to
# be hand-edited whenever a combination is first built.
# Seeded to current reality -- edit as codes gain physics (grid self-gravity,
# validated PKDGRAV3 SPH-MHD, a second code with star formation or OU driving,
# ...). `runtest.sh support` prints this table.
declare -a SUPPORT
SUPPORT[0]="hydro mhd grav starform starform=divvoff turbdriving cooling=none|metal eos=adiabatic|barotropic|isor"
                                      # Gasoline    (gasolinemhd; verified)
SUPPORT[1]="hydro mhd grav turbdriving fdm"   # ChaNGa      (--enable-mhd; turbdriving: --enable-turbdriving, turbulence_driving branch only; fdm: --enable-fdm, private MHD-TP profile only)
SUPPORT[2]="hydro mhd grav"           # SPH-EXA     (converter sets gravConstant=1/open box on bDoGravity; --prop std Barnes-Hut)
SUPPORT[3]="hydro mhd grav"           # AREPO       (Arepo_grav: Config_grav.sh SELFGRAVITY+GRAVITY_NOT_PERIODIC, arepo-grid bggrid IC)
SUPPORT[4]="hydro mhd"                # RAMSES      (grav needs poisson=.true.)
SUPPORT[5]="hydro mhd grav"           # GIZMO       (GIZMO_grav wired; no combined binary yet -- its adapter says so)
SUPPORT[6]="hydro mhd grav"           # Phantom     (phantom_grav: SETUP=evrard, adiabatic isolated self-gravity)
SUPPORT[7]="hydro mhd"                # Athena++    (grav needs self-gravity)
SUPPORT[8]="hydro grav"               # PKDGRAV3    (N-body+SPH: gravity native; no public SPMHD -> no mhd)
SUPPORT[9]="hydro mhd grav"           # OpenGadget3 (OpenGadget3_grav: Config_grav.sh, isolated adaptive-softening self-gravity)

# Everything runtest.sh knows about a code, in ONE table. Per code:
#   NAME/LABEL  what to call it in messages and in `runtest.sh support`
#   SLUG        its directory under codes/, and the name its artefacts carry
#   DIR         that directory: the suite's scripts for this code live there
#   BUILD_DIR   the directory its binary was built into (<CODE>_DIR overrides it)
#   BINARY      the default binary in there; physics variants are picked per code
# Plus the two adapters a code may have: BUILD_ADAPTER resolves a binary from a
# RUNPHYSICS set, RUN_ADAPTER owns every code-specific step of a run. For a code
# with a build adapter BUILD_DIR is the source tree handed to it (empty: the
# adapter's default) and BINARY is empty -- the adapter resolves both; a code
# with no run adapter keeps its inline block below.
declare -a CODE_NAME CODE_LABEL CODE_SLUG CODE_DIR CODE_BUILD_DIR CODE_BINARY \
           CODE_BUILD_ADAPTER CODE_RUN_ADAPTER
CODE_NAME=(  GASOLINE CHANGA SPH-EXA  AREPO RAMSES GIZMO PHANTOM ATHENA       PKDGRAV3 OPENGADGET3 )
CODE_LABEL=( Gasoline ChaNGa SPH-EXA  AREPO RAMSES GIZMO Phantom Athena++     PKDGRAV3 OpenGadget3 )
CODE_SLUG=(  gasoline changa sphexa   arepo ramses gizmo phantom athena++     pkdgrav3 opengadget3 )
CODE_BINARY=( ""      ""     sphexa   Arepo ramses3d GIZMO phantom athena_hydro pkdgrav3 OpenGadget3 )
CODE_BUILD_DIR=( "$GASOLINE_DIR" "$CHANGA_DIR" "$SPHEXA_DIR" "$AREPO_DIR" "$RAMSES_DIR" "$GIZMO_DIR" \
                 "$PHANTOM_DIR" "$ATHENA_DIR" "$PKDGRAV3_DIR" "$OPENGADGET3_DIR" )
for code_index in "${!CODE_SLUG[@]}"; do
    CODE_DIR[$code_index]="$codes_dir/${CODE_SLUG[$code_index]}"
done
unset code_index
CODE_BUILD_ADAPTER[0]="${CODE_DIR[0]}/build_gasoline.sh"
CODE_BUILD_ADAPTER[1]="${CODE_DIR[1]}/build_changa.sh"
CODE_RUN_ADAPTER[0]="${CODE_DIR[0]}/adapter.sh"
CODE_RUN_ADAPTER[1]="${CODE_DIR[1]}/adapter.sh"
CODE_RUN_ADAPTER[2]="${CODE_DIR[2]}/adapter.sh"
CODE_RUN_ADAPTER[8]="${CODE_DIR[8]}/adapter.sh"
CODE_RUN_ADAPTER[3]="${CODE_DIR[3]}/adapter.sh"
CODE_RUN_ADAPTER[5]="${CODE_DIR[5]}/adapter.sh"
CODE_RUN_ADAPTER[6]="${CODE_DIR[6]}/adapter.sh"
CODE_RUN_ADAPTER[9]="${CODE_DIR[9]}/adapter.sh"
CODE_RUN_ADAPTER[4]="${CODE_DIR[4]}/adapter.sh"
CODE_RUN_ADAPTER[7]="${CODE_DIR[7]}/adapter.sh"

source "$codes_dir/physics_set.sh"

# Axes a row may leave unmentioned without being gated; see the matrix header.
PHYSICS_UNGATED_AXES="eos cooling"
token_supported() {  # $1 = requested token, remaining args = the code's row
    local want="$1" key value entry alt axis
    key="${want%%=*}"
    value=""
    [ "$want" = "$key" ] || value="${want#*=}"
    shift
    for entry in "$@"; do
        [ "$entry" = "$want" ] && return 0
        case "$entry" in
            "$key="*)
                [ -n "$value" ] || continue
                for alt in $(printf '%s' "${entry#*=}" | tr '|' ' '); do
                    [ "$alt" = "$value" ] && return 0
                done
                ;;
        esac
    done
    for axis in $PHYSICS_UNGATED_AXES; do
        [ "$axis" = "$key" ] || continue
        for entry in "$@"; do             # mentioned in any form -> enforced ...
            case "$entry" in "$key"|"$key="*) return 1 ;; esac
        done
        return 0                          # ... not characterised -> not gated
    done
    return 1                              # a capability the row does not list
}
physics_unsupported() {  # $1 = code index, $2 = set -> prints the first missing capability
    local t
    for t in $2; do
        token_supported "$t" ${SUPPORT[$1]} || { printf '%s\n' "$t"; return 0; }
    done
    return 1
}
code_allowed() {  # $1 = code index, remaining words = per-test allow-list
    local wanted="$1" candidate; shift
    for candidate in "$@"; do [ "$candidate" = "$wanted" ] && return 0; done
    return 1
}
print_support_matrix() {
    printf '%-5s %-12s %s\n' code code-name "capabilities it is wired for (the RUNPHYSICS set)"
    printf '%-5s %-12s %s\n' ----- ------------ ------------------------------------------------
    local i; for i in "${!CODE_LABEL[@]}"; do
        printf '%-5s %-12s %s\n' "$i" "${CODE_LABEL[$i]}" "${SUPPORT[$i]}"
    done
    local adapter
    for i in "${!CODE_BUILD_ADAPTER[@]}"; do
        adapter="${CODE_BUILD_ADAPTER[$i]}"
        if [ -r "$adapter" ]; then
            printf '\nBuild capabilities supplied by the %s adapter:\n' "${CODE_LABEL[$i]}"
            bash "$adapter" capabilities
        fi
    done
}

#If running serial/mpi or with charmrun
#mpirun="$dir/charmrun.smp +p 63"
#mpirun=""
# MPIRUN overrides the launch prefix of the MPI codes, e.g. MPIRUN="mpirun -np 16 " for a small test.
mpirun="${MPIRUN:-mpirun }"

# One process/thread per PHYSICAL core that this process may use. nproc counts
# SMT siblings (128 on the 2x32-core EPYC nodes), which would put two workers on
# every core. A bare mpirun already does this (Intel MPI's default); codes that
# pick their own count take this one.
ncores="$(python3 -c '
import os
t = "/sys/devices/system/cpu/cpu%d/topology/"
print(len({(open(t % c + "physical_package_id").read(), open(t % c + "core_id").read())
           for c in os.sched_getaffinity(0)}))' 2>/dev/null)" || ncores=""
[[ "$ncores" =~ ^[1-9][0-9]*$ ]] || ncores="$(nproc)"

export I_MPI_OFI_PROVIDER="${I_MPI_OFI_PROVIDER:-shm}"

#An extra word added at the end of the simulation output files and directory
EXTRANAME="${EXTRANAME:-gradh}"

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

# Preserve an exact decimal parameter while choosing the shorter of ordinary
# and normalized scientific notation. This is for wide-range identity values
# such as plasma beta: 1000000 -> 1e6, while 0.05 stays 0.05. Unlike fmt(), it
# deliberately does not round to two decimal places.
fmt_compact() {
    python3 - "$1" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

raw = sys.argv[1]
try:
    value = Decimal(raw)
except InvalidOperation:
    print(raw)
    raise SystemExit
if not value.is_finite():
    print(raw)
    raise SystemExit
if value == 0:
    print("0")
    raise SystemExit

value = value.normalize()
plain = format(value, "f")
absolute = value.copy_abs().as_tuple()
digits = "".join(str(digit) for digit in absolute.digits)
exponent = len(digits) + absolute.exponent - 1
coefficient = digits[0]
if len(digits) > 1:
    coefficient += "." + digits[1:]
scientific = ("-" if value.is_signed() else "") + coefficient + "e" + str(exponent)
print(scientific if len(scientific) < len(plain) else plain)
PY
}

# Glass relaxation: the Python (numba) glassgen.cli relaxes the pre-IC into a
# glass (multi-resolution cascade + heavy-ball momentum + the Wendland C2
# self-density correction; kernel + operator from relax.config; writes
# ${glassfile}_IC directly). The target density
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
# icr0-stop 3e-3, self-bias on). The iteration cap, the ICR0/ICRLOOP0/rate step
# knobs and the FFT coarse-grid correction are set in relax.config
# (GLASS_MAXITER, GLASS_ICR0, GLASS_ICRLOOP0, GLASS_ICR0RATE, GLASS_ICR0_STOP,
# GLASS_POLISH_ITER, GLASS_INTER_ITER, GLASS_PROGRESSIVE, GLASS_COARSE_FFT, GLASS_COARSE_FFT_EVERY).
GLASSMODE=""
# Force table for GLASS_FORCEMATCH=1 (relax.config), set by setups whose pre-IC writes one
# (glassgen --force-table: P, g = grad P_t/rho_t, rho, closure); empty = the test has none.
# A per-pre-IC datafiles/$prefile.force.npz (accdisk) is picked up at the glass step instead.
FORCETABLE=""
# The relaxer's boundary is a property of the density target, not necessarily
# of the runnable IC.  Most tests deliberately relax a filled periodic seed
# and crop/transform it afterwards.  Free-surface targets opt into "open" in
# their setup function; glassgen then builds and later culls its envelope.
GLASSBOUNDARY="param"
# Kernel + operator the glass is relaxed with (relax.config at the testsuite
# root); exported in glass mode so a code adapter can warn when its build differs.
source "$current_dir/relax.config"
GLASSKNOBS="-n $GLASS_MAXITER -oi 0 --icr0 $GLASS_ICR0 --icrloop0 $GLASS_ICRLOOP0 --icr0rate $GLASS_ICR0RATE --icr0-stop $GLASS_ICR0_STOP --polish-iter $GLASS_POLISH_ITER --inter-iter $GLASS_INTER_ITER --kernel $SPH_KERNEL --sigma $GLASS_SIGMA --sigma-f $GLASS_SIGMA_F --gradient $GLASS_GRADIENT --neighbours $GLASS_NEIGHBOURS --rhopow $GLASS_RHOPOW"
[ "$GLASS_KERNEL_MATCH" = 1 ] && GLASSKNOBS="$GLASSKNOBS --kernel-match"
[ "$GLASS_PLOTS" = 1 ] && GLASSKNOBS="$GLASSKNOBS --plot --history-plot"
[ "$(fmt "$GLASS_ISPH_BOUNDTERM")" = 2 ] || GLASSKNOBS="$GLASSKNOBS --isph-boundterm $GLASS_ISPH_BOUNDTERM"
[ "$GLASS_GRADH" = 1 ] && GLASSKNOBS="$GLASSKNOBS --gradh"
[ "$GLASS_TARGET_SHAPE" = 1 ] && GLASSKNOBS="$GLASSKNOBS --target-shape-errp $GLASS_SHAPE_ERRP"
[ "$GLASS_PROGRESSIVE" = 1 ] || GLASSKNOBS="$GLASSKNOBS --no-progressive"
if [ "$(fmt "$GLASS_COARSE_FFT")" != 0 ]; then
    if [ "$GLASS_PROGRESSIVE" = 1 ]; then
        echo "Error: GLASS_COARSE_FFT=$GLASS_COARSE_FFT needs GLASS_PROGRESSIVE=0 in relax.config (the cascade has no FFT correction)."
        exit 1
    fi
    GLASSKNOBS="$GLASSKNOBS --coarse-fft $GLASS_COARSE_FFT"
    [ "$(fmt "$GLASS_COARSE_FFT_EVERY")" = 0 ] || GLASSKNOBS="$GLASSKNOBS --coarse-fft-every $GLASS_COARSE_FFT_EVERY"
elif [ "$(fmt "$GLASS_COARSE_FFT_EVERY")" != 0 ]; then
    echo "Error: GLASS_COARSE_FFT_EVERY=$GLASS_COARSE_FFT_EVERY needs GLASS_COARSE_FFT > 0 in relax.config."
    exit 1
fi
# Glass/IC cache-key tag of those knobs (run names are left as they are):
# kernel c2|c4|ko|sn, gradient oi|os|ai|as, f<sigma_f>, then v (variable
# neighbours), km (kernel-match), p<rhopow> when rhopow != 2, b<boundterm>
# when the ISPH threshold != 2, gh (grad-h factor), fm (force-matched relax), cv<err_P> (curvature step),
# then the schedule knobs that differ from their defaults: n<maxiter>, i<icr0>, l<icrloop0>, r<rate>, ii<inter-iter>, st<icr0-stop>, pi<polish-iter>, np (no cascade), ft<TOP>, fe<N>.
case "$SPH_KERNEL" in wendlandc2) GLASSTAG=c2 ;; wendlandc4) GLASSTAG=c4 ;;
    kerneloptimized) GLASSTAG=ko ;; sinc) GLASSTAG=sn ;; *) GLASSTAG="$SPH_KERNEL" ;; esac
case "$GLASS_GRADIENT" in onesided-isph) GLASSTAG+=oi ;; onesided-sph) GLASSTAG+=os ;;
    average-isph) GLASSTAG+=ai ;; average-sph) GLASSTAG+=as ;; *) GLASSTAG+="$GLASS_GRADIENT" ;; esac
GLASSTAG="G${GLASSTAG}f$(fmt "$GLASS_SIGMA_F")"
[ "$GLASS_NEIGHBOURS" = variable ] && GLASSTAG+=v
[ "$GLASS_KERNEL_MATCH" = 1 ] && GLASSTAG+=km
[ "$(fmt "$GLASS_RHOPOW")" = 2 ] || GLASSTAG+="p$(fmt "$GLASS_RHOPOW")"
[ "$(fmt "$GLASS_ISPH_BOUNDTERM")" = 2 ] || GLASSTAG+="b$(fmt "$GLASS_ISPH_BOUNDTERM")"
[ "$GLASS_GRADH" = 1 ] && GLASSTAG+=gh
[ "$GLASS_FORCEMATCH" = 1 ] && GLASSTAG+=fm
[ "$GLASS_TARGET_SHAPE" = 1 ] && GLASSTAG+="cv$(fmt "$GLASS_SHAPE_ERRP")"
[ "$(fmt "$GLASS_MAXITER")" = 4800 ] || GLASSTAG+="n$(fmt "$GLASS_MAXITER")"
[ "$(fmt "$GLASS_ICR0")" = 1 ] || GLASSTAG+="i$(fmt "$GLASS_ICR0")"
[ "$(fmt "$GLASS_ICRLOOP0")" = 0.5 ] || GLASSTAG+="l$(fmt "$GLASS_ICRLOOP0")"
[ "$(fmt "$GLASS_ICR0RATE")" = 1.5 ] || GLASSTAG+="r$(fmt "$GLASS_ICR0RATE")"
[ "$(fmt "$GLASS_INTER_ITER")" = 10 ] || GLASSTAG+="ii$(fmt "$GLASS_INTER_ITER")"
[ "$(fmt "$GLASS_ICR0_STOP")" = 0.003 ] || GLASSTAG+="st$(fmt "$GLASS_ICR0_STOP")"
[ "$(fmt "$GLASS_POLISH_ITER")" = 0 ] || GLASSTAG+="pi$(fmt "$GLASS_POLISH_ITER")"
[ "$GLASS_PROGRESSIVE" = 1 ] || GLASSTAG+=np
[ "$(fmt "$GLASS_COARSE_FFT")" = 0 ] || GLASSTAG+="ft$(fmt "$GLASS_COARSE_FFT")"
[ "$(fmt "$GLASS_COARSE_FFT_EVERY")" = 0 ] || GLASSTAG+="fe$(fmt "$GLASS_COARSE_FFT_EVERY")"
# Start pressure on the glass (relax.config GLASS_START_P): tags the IC name, not the glass.
case "$GLASS_START_P" in uniform) ICSTARTTAG="" ;; pseudo) ICSTARTTAG=pP ;;
    *) echo "Error: GLASS_START_P='$GLASS_START_P' in relax.config (uniform | pseudo)"; exit 1 ;; esac

#Extra condition during runtime
#EXTRACONDALL=" -facQ 1.0 -fch 1.0 -psidecfac 1.0 -thermaldiff 0.1 -alpha 2.0 -beta 8.0 -nInitCorrIter 50 -smin 32 -smax 40000 "
#EXTRACONDALL=" -bDivBClean 1 -thermaldiff 0.1 -alpha 2.0 -beta 8.0 -fch 1.0 -nInitCorrIter 50 -smin 32 -smax 40000 "
#EXTRACONDALL=" -thermaldiff 0.1 -alpha 0.5 -beta 8.0 "
EXTRACONDALL=""
#Extra runtime conditions and output fields specific to SPH-EXA (code=2)



usage() {
    echo "Usage: $0 [-r] [-a value1] ... [-k value11] <nameofsimulation> <Nsmooth> <Nres> <distrib> <code> [vm] [extra]"
    echo ""
    echo "To see the parameter choices (-a .. -k) for a specific simulation, run it by name with no further arguments, e.g.:"
    echo "    $0 accdisk"
    echo ""
    echo "Options:"
    echo "  -a, -b, -c, ...    These options depend on the simulation. To see the specific options, enter the nameofsimulation without arguments."
    echo "  -r                 Do-reference: after the run, save the analysis result"
    echo "                     as the regression baseline '<runname>_reference.json'"
    echo "                     (requires setupfiles/IC_analysis_<simulation>.py)."
    echo ""
    echo "Arguments:"
    echo "  nameofsimulation   Name of the simulation to be run (e.g., shocktube)."
    echo "  Nsmooth            Number of neigbours within smoothing kernel"
    echo "  Nres               Resolution for the simulation (usually specified as number of elements in the x direction)."
    echo "  distrib            Initial conditions type:"
    echo "                     0: close-packed (FCC) lattice"
    echo "                     1: random"
    echo "                     2: glass (relaxed from a random pre-IC)"
    echo "                     3: simple cubic lattice"
    echo "                     4: glass patch (two uniform glasses relaxed apart and pasted, always a sharp contrast; kh, square)"
    echo "  code               Simulation code: 0: Gasoline  1: ChaNGa  2: SPH-EXA  3: AREPO  4: RAMSES  5: GIZMO  6: Phantom  7: Athena++  8: PKDGRAV3  9: OpenGadget3"
    echo "                     (Each test declares the capability SET it needs -- physics modules"
    echo "                      plus eos=/cooling= axes; a code not wired for one is rejected."
    echo "                      Run '$0 support' for the matrix.)"
    echo "  vm                 (optional) enable variable SPH particle masses in the IC generator (default 0 = equal-mass particles)."
    echo "  extra              (optional) extra flags appended to the runtime command."
    echo ""
    echo "Glass relaxation (distrib 2): relax.config at the testsuite root sets the SPH_KERNEL and pair"
    echo "  operator (sigma, sigma_f, gradient, neighbours, kernel-match, rhopow, ISPH boundterm, grad-h) glassgen relaxes with;"
    echo "  GLASS_START_P (uniform | pseudo) sets the start pressure on the glass (IC name tag pP), and the IC build warns"
    echo "  when the glass holds a pressure bump above GLASS_PBUMP_WARN (setupfiles/IC_glasspressure.py)."
    echo "  GLASS_TARGET_SHAPE (0 raw | 1 curvature step, kh only) with GLASS_SHAPE_ERRP shapes a 1D contact to hold that pressure error."
    echo "  Schedule: GLASS_MAXITER, GLASS_ICR0, GLASS_ICRLOOP0, GLASS_ICR0RATE; finish: GLASS_ICR0_STOP (step-size stop, 0 off),"
    echo "  GLASS_POLISH_ITER (fixed polish budget, 0 off); GLASS_INTER_ITER (per cascade level); GLASS_PROGRESSIVE (1 cascade | 0 one full-resolution relax);"
    echo "  GLASS_COARSE_FFT (0 off | TOP, needs GLASS_PROGRESSIVE=0): FFT correction of the long-wavelength density modes;"
    echo "  GLASS_COARSE_FFT_EVERY (0 off | N): also every N iterations (transport mode, smooth profiles far from the pre-IC)."
    echo "  the glass/IC names carry a tag of them, and a run code whose build differs warns."
    echo ""
    echo "Managed build selection (Gasoline code 0 / ChaNGa code 1):"
    echo "  <CODE>_DIR=/path/to/source               select that code's source tree"
    echo "                                           (empty = the public codes/<code>/<code>-public)"
    echo "  <CODE>_BUILD_CONFIG=/path/to/config      select fork-specific capabilities"
    echo "                                           (default: build_config.sh, or build_config_private.sh"
    echo "                                            when <CODE>_DIR is set)"
    echo "  <CODE>_BUILD_MODE=auto|reuse-only        build a miss or require a cached build"
    echo "  <CODE>_BINARY=/absolute/path              explicitly pin a managed artifact"
    echo "  codes/extra_build_flags.sh               persistent extra-build configuration"
    echo "    GASOLINE_EXTRA_BUILD_FLAGS=(...)       additional Gasoline -D definitions"
    echo "    CHANGA_EXTRA_BUILD_FLAGS=(...)         additional ChaNGa configure switches"
    echo "  CODE_EXTRA_BUILD_FLAGS_FILE=/path/file   select another campaign flags file"
    echo ""
    echo "Gasoline-specific build options:"
    echo "  GASOLINE_DIR=/path/to/source             select a Gasoline fork"
    echo "  GASOLINE_KERNEL=KERNELOPTIMIZED           select the kernel macro"
    echo "  GASOLINE_COOLING=none|metal              select the cooling module compiled in"
    echo "                                           (a test's cooling= token overrides this)"
    echo "  GASOLINE_DEFINES='...'                   legacy/temporary extra override"
    echo ""
    echo "ChaNGa-specific build options:"
    echo "  CHANGA_DIR=/path/to/changaMHDTP          the private MHD-TP tree (picks build_config_private.sh)"
    echo "  CHANGA_CONFIGURE_FLAGS='--enable-...'     legacy/temporary extra override"
    echo "  CHANGA_CHARM_SRC=/path/to/charm           override the selected Charm++ build"
    echo "  CHANGA_UTILITY_SRC=/path/to/utility       override N-BodyShop utility"
    echo "  RUNTEST_RESOLVE_ONLY=1                    print selection/run name, then stop"
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
    echo "  turbbox            OU-driven periodic turbulence box."
    echo "                     (-a Mach, -b solenoidal weight zeta 0..1, -c case 0 hydro"
    echo "                      1 MHD uniform Bz 2 MHD zero-net-flux sin Bz, -d crossings,"
    echo "                      -e forcing realisation: 0 published file, N>=1 generated seed N)"
    echo "  starform           Star-formation box: analytic SF rate + SFBFIELD magnetic handoff."
    echo "                     (-a selects the case; the table lives in setupfiles/IC_setup_starform.py)"
    echo "                     (-b dCStar, -c per-step probability, -d donors per"
    echo "                      site. SFBFIELD and MHD_B_VAR go in the build flags file.)"
    echo "  disstest           Isolated dissipation operator/switch tests"
    echo "                     (static bComove=0 and expanding bComove=1 arm per case)."
    echo "  mhdbalsaravortex   Balsara MHD vortex simulation."
    echo "  zeldovich          Zeldovich pancake (cosmological) simulation."
    echo "  mri                Magnetorotational instability simulation."
    echo "  mhdloopshear       Magnetized advection loop with shear simulation."
    echo "  currentsheet       MHD current sheet (reconnection robustness) simulation."
    echo "  noh                Noh problem (converging strong shock) simulation."
    echo "  isentropicvortex   Isentropic (Yee) vortex (smooth convergence) simulation."
    echo "  wave               Linear (M)HD wave: sound/fast/slow/Alfven (convergence) simulation."
    echo "  coldkeplerian      Cold Keplerian disk/ring (AV / angular-momentum) simulation."
    echo "  powerlawring       External power-law disk/annulus (orbits / MHD winding)."
    echo "  soliton            FDM soliton: -a 0 static Gaussian core, quantum potential and force"
    echo "                     (Nori & Baldi 2018); -a 1 the same Gaussian as a free packet (gravity off);"
    echo "                     -a 2 Schroedinger-Poisson ground state in equilibrium;"
    echo "                     -a 3 head-on collision, -b separation / r_c, -c relative speed / v_c"
    echo "                     (fast -b 10 -c 6, slow -b 8 -c 1); -a 4 halo from -b N solitons (default 4)"
    echo "                     in a sphere of -c R r_c (default 6); -a 5 the -a 2 soliton with tracer stars and"
    echo "                     hydrostatic gas, each -b x its mass (default 0.001), gas c_s -c x v_c (default 0.5),"
    echo "                     -d gas EOS (0 isothermal, 1 adiabatic); <Nsmooth> is the FDM number, gas uses 64."
    echo "  gaussOverdens      FDM static 3D Gaussian overdensity on a background."
    echo "  1DdensityFront     FDM static 1D tanh density front."
    echo "  fdmwave            FDM waves (-a case: 0 free dispersion in a slab; self-gravitating"
    echo "                     cube: 1 Jeans, 2 comoving Jeans crossing, 3 null test; exact, in the"
    echo "                     slab: 4 two interfering waves with nodes, 5 two-stream intersection,"
    echo "                     6 three interfering waves (discontinuous nodes); 7 comoving growing mode"
    echo "                     (case 2 started on Schive 2025 Fig. 11's f, a = 0.06 -> 2); -b mode number,"
    echo "                     -c amplitude (case 4: contrast, default 1; case 6: must be 1; case 7: default 0.01), -d slab"
    echo "                     thickness factor for cases 0, 4, 5, 6 (default 1; 2 for 400 neighbours); case 1:"
    echo "                     -d Jeans mode number N_J, k_J = 2 pi N_J / L (default 1.5))."
    echo "                     FDM tests: ChaNGa (code 1) with --enable-fdm (V5); FDM run-time options"
    echo "                     go through [extra], e.g. '-iFDMHMode 2 +bFDMSoftH'."
    echo ""
    exit 1
}


# Parse command line options
doreference=0
declare -a args=("" "" "" "" "" "" "" "" "" "" "")
while getopts ":ra:b:c:d:e:f:g:h:i:j:k:" opt; do
  case $opt in
    r)
      doreference=1
      ;;
    a|b|c|d|e|f|g|h|i|j|k)
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
# Preserve the post-test-name positional selection for setup functions. Bash
# gives a called function its own empty "$@", so those functions cannot read
# the script's third positional argument directly.
requested_distribution=${3:-}

# RUNPHYSICS helpers (see the capability matrix + gate). Each setup_<test>()
# declares RUNPHYSICS as ONE SET holding everything the build must satisfy:
# physics modules as bare tokens, single-valued axes as key=value. EOS and the
# cooling module are axes of the set because an axis kept outside it is ungated.
#   eos=      adiabatic (default) | barotropic | isor  -- WHICH EOS IS COMPILED IN, not
#             the runtime one: turbbox is isothermal at runtime via
#             bGasIsothermal and is still eos=adiabatic here.
#   cooling=  none | metal                      -- WHICH COOLING MODULE is
#             compiled in, not whether the run cools (that is bGasCooling).
#             Absent means "the adapter's own default", so adapters that do not
#             implement the flag stay untouched.
# The set is a property of the TEST, used ONLY to pick the per-code binary/config
# and to gate unsupported (code, capability) pairs. It is NEVER passed to a
# converter -- converters auto-detect physics from the IC so they stay usable
# standalone (outside runtest.sh). The set travels WHOLE to a build adapter as
# its --physics: it IS the build requirement, so no flattened view is derived.
RUNCODES=""
nonzero() { awk -v v="$1" 'BEGIN { print (v+0 != 0) ? 1 : 0 }'; }  # numeric != 0
combine_physics() {  # $1 mhd(0/1) $2 grav(0/1) -> the physics tokens of the set
    local out=""
    [ "$2" = 1 ] && out="grav"
    [ "$1" = 1 ] && out="${out:+$out }mhd"
    printf '%s\n' "${out:-hydro}"
}

# Function for mhdcollapse setup with generic parameters
setup_mhdcollapse() {
    RUNPHYSICS="grav mhd eos=barotropic"
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
    EXTRACONDRUN=" -thermaldiff 0.0 " #barotropic so no thermal diffusion, kills timestep otherwise
    prefile_extra="rd$(fmt "${rhodiff}")"
    glassfile_extra="rd$(fmt "${rhodiff}")${GLASSNAME}"
    runname_extra="mu$(fmt "${mu}")rd$(fmt "${rhodiff}")Erat$(fmt "${Erat}")"
}

# Function for accdisk setup with generic parameters
setup_accdisk() {
    # eos=isor: the setup's u is not the disk temperature (gamma 5/3 vs dConstGamma 1+1e-10, so
    # without ISOR P/rho ~ 1e-11); ISOR sets cs = dDiskHr R^-dDiskq, both written to the .param.
    RUNPHYSICS="grav eos=isor"
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
    echo "Sound-speed exponent, cs = hr R^-q (q) -d : $q"
    echo "Orbital radius (rorb) -e : $rorb"
    echo "Disk radius (rdisk) -f : $rdisk"
    echo "Inner disk radius (rinner) -g : $rinner"

    # Setup based on the potentially updated values
    EXTRACONDIC="$hr $massrat $onlyoneBH $q $rorb $rdisk $rinner "
    # The pre-IC writes the glass target (dICdensprofile 7) and the force table on an (R, |z|)
    # grid as datafiles/$prefile.{density,force}.npz; the glass step picks both up.
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
        "weak shock p1.2"
        "weak shock p2"
        "weak shock p5"
        "weak field Brio-Wu"
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
    if [ "$shock_index" = 1 ] || [ "$shock_index" = 11 ] || \
       [ "$shock_index" = 16 ] || [ "$shock_index" = 17 ] || [ "$shock_index" = 18 ]; then
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

# Periodic OU-driven turbulence box (Price & Federrath 2010; Tricco et al. 2016).
setup_turbbox() {
    mach=${args[0]:-10}
    forcing=${args[1]:-1}
    case_id=${args[2]:-1}
    crossings=${args[3]:-100}
    seeddrv=${args[4]:-0}
    # The forcing realisation is ONE choice: 0 takes the published sequence
    # (which is seed 1's realisation), any positive integer generates its own.
    case "$seeddrv" in
        0|0.0)      driving="paper";     seed=1 ;;
        *[!0-9]*)   echo "Error: turbbox -e must be 0 (published forcing file)"
                    echo "       or a positive integer OU seed."; exit 1 ;;
        *)          driving="generated"; seed="$seeddrv" ;;
    esac
    # Reject the obsolete -f/-g flags loudly rather than ignore them.
    if [ -n "${args[5]}" ] || [ -n "${args[6]}" ]; then
        echo "Error: turbbox -f/-g were retired. Letters are now"
        echo "       -a Mach, -b solenoidal weight, -c case, -d crossings, -e forcing realisation."
        exit 1
    fi
    # One resolve for the projection weight and the case; the setup module is
    # the single source of truth for both (no duplicate bash whitelist).
    turbcfg=$(PYTHONPATH="$analysisdir" python3 -c \
'import sys
from IC_setup_turbbox import forcing_weight, forcing_tag, case_physics, case_tag
zeta = forcing_weight(sys.argv[1])[1]
mhd, field = case_physics(sys.argv[2])
physics, _seed_tag = case_tag(sys.argv[2])
print(repr(zeta), forcing_tag(sys.argv[1]), mhd, field, physics)' \
        "$forcing" "$case_id") || exit 1
    read -r forcing forcing_tag mhd field physics_tag <<<"$turbcfg"
    # `turbdriving` is the OU forcing term: the box stirs itself, so a build
    # without it does not run this test at all. It is therefore a capability of
    # the TEST and belongs in the set like any other -- each code's build config
    # owns the spelling (Gasoline: -DOUTURBDRIVER), and the gate refuses a code
    # whose SUPPORT row does not list it. The numerical SCHEME a driven run uses
    # -- reconstruction, its limiter, the resistivity operator -- is not part of
    # the requirement and lives in codes/extra_build_flags.sh with every other
    # scheme flag, so a suite can A/B it without this test forcing its hand.
    RUNPHYSICS="$physics_tag turbdriving"
    driving=$(PYTHONPATH="$analysisdir" python3 -c \
        'import sys; from IC_setup_turbbox import driving_source; print(driving_source(*sys.argv[1:]))' \
        "$driving" "$mach" "$forcing" "$seed" "$crossings") || exit 1
    if [ "$driving" = paper ] && [ ! -f "$current_dir/test_cases/turbbox/forcing/forcing.dat" ]; then
        echo "Error: paper forcing requires test_cases/turbbox/forcing/forcing.dat (Zenodo 13162225)."
        exit 1
    fi
    # EOS: isothermal at RUNTIME via bGasIsothermal, so the set stays at the
    # default eos=adiabatic -- BAROTROPIC in this fork is dimensional.
    echo "Using settings for the periodic OU turbulence box:"
    echo "target/nominal rms Mach -a : $mach"
    echo "forcing -b (solenoidal weight zeta, 1=sol 0=comp) : $forcing"
    echo "case -c (0 hydro, 1 MHD uniform Bz, 2 MHD zero-net-flux sin Bz) : $case_id -> $physics_tag, seed $field"
    echo "duration in crossing times -d : $crossings"
    echo "forcing realisation -e (0 = published file, N>=1 = generated with OU seed N) : $seeddrv -> $driving, StSeed $seed"
    EXTRACONDIC="$mach $forcing $mhd $field $crossings $seed $driving "
    if [ "$mhd" = 1 ]; then
        EXTRACONDRUN="$EXTRACONDRUN "
    fi
    if [ "$driving" = paper ]; then
        EXTRACONDRUN="$EXTRACONDRUN -StForcingFile forcing/forcing.dat "
    fi
    prefile_extra="uniform"
    glassfile_extra="uniform${GLASSNAME}"
    runname_extra="M$(fmt_compact "${mach}")${forcing_tag}${physics_tag}B${field}tc$(fmt_compact "${crossings}")seed${seed}${driving}"
}

# Periodic star-formation box: the analytic RATE arm and the controlled magnetic
# HANDOFF arm, sharing one IC and one build profile.
#
# Two COMPILE-TIME couplings are derived from the IC parameters, so a single
# command line selects both the physics and the binary:
#   * DIVVOFF is compiled in when vmode = 0.  starform.c rejects divv >= 0 and a
#     resting box gives exactly zero, so the rate arm cannot form a single star
#     without it.  The divv-criterion arm uses vmode = 1 -- with ampl = 0 for the
#     static control, which is the same IC with the criterion left live.
#   * cooling= is "none" unless the case needs the EVALUATED gas temperature.
#     Under NOCOOLING starform.c pins T = 10 K, which reaches the same verdict as
#     the evaluated temperature only for a uniform box whose T0 and 10 K fall on
#     the same side of dTempMax.
setup_starform() {
    case_id=${args[0]:-1}
    cstar=${args[1]:-0}
    pstep=${args[2]:-0}
    clump=${args[3]:-0}
    sf_req=$(PYTHONPATH="$analysisdir" python3 -c \
        'import sys
from IC_setup_starform import (case_build_requirements, case_name,
                               case_runname_tag, case_glass_tag, case_dials)
print(case_build_requirements(sys.argv[1]))
print(case_name(sys.argv[1]))
print(case_runname_tag())
print(case_glass_tag(sys.argv[1]))
print("%r %r %d" % case_dials(*sys.argv[1:5]))' \
        "$case_id" "$cstar" "$pstep" "$clump") || exit 1
    sf_line=$(printf '%s\n' "$sf_req" | sed -n 1p)
    sf_name=$(printf '%s\n' "$sf_req" | sed -n 2p)
    sf_tag=$(printf '%s\n' "$sf_req" | sed -n 3p)
    sf_glass=$(printf '%s\n' "$sf_req" | sed -n 4p)
    # -b/-c/-d are OVERRIDES where 0 means "the case's value", so the numbers
    # the run actually uses come back from case_dials -- the same call create()
    # makes, so the run name cannot advertise a dial the IC did not use.
    read -r sf_cstar sf_pstep sf_clump <<<"$(printf '%s\n' "$sf_req" | sed -n 5p)"
    sf_tokens=$(printf '%s' "$sf_line" | cut -d'|' -f1)
    sf_extra=$(printf '%s' "$sf_line" | cut -d'|' -f2)
    # One MHD build serves every case; the hydro ones simply set B0 = 0.  The
    # case table supplies the rest of the set -- the starform variant and the
    # cooling module -- as capability tokens, so the gate refuses any code not
    # wired for star formation.
    RUNPHYSICS="mhd $sf_tokens"
    echo "Using settings for the star-formation box:"
    echo "case -a : $case_id ($sf_name)"
    echo "dCStar -b : $sf_cstar"
    echo "per-step probability -c : $sf_pstep"
    echo "donors per site -d : $sf_clump"
    echo "  -> capabilities: $RUNPHYSICS ; run flags: ${sf_extra:-none}"
    EXTRACONDRUN=" $sf_extra "
    # ANY bStarForm run calls msrInitCooling, and CoolInitRatesTable ->
    # clReadMetalTable does an unconditional fopen+assert on CoolInFile even
    # when cooling is off at runtime.  The suite ships its own copy of the table
    # beside the setup files, so the path is fixed and the same table is used on
    # every machine.  It goes on the COMMAND LINE (prm short name "coolin"; the
    # .param spelling is CoolInFile) because the .param is a cached artefact.
    if [ "$(physics_value cooling "$RUNPHYSICS")" = "metal" ]; then
        cooltable="${analysisdir}cooltable_xdr"
        [ -r "$cooltable" ] || {
            echo "Error: case $case_id needs a COOLING_METAL build, whose"
            echo "       CoolInitRatesTable asserts on CoolInFile, but"
            echo "       $cooltable is missing.  Copy it from a Gasoline tree."
            exit 1
        }
        EXTRACONDRUN="$EXTRACONDRUN -coolin $cooltable "
    fi
    # Order matches IC_setup_starform.create(... case cstar pstep)
    EXTRACONDIC="$case_id $cstar $pstep $clump "
    # One relaxed glass per DENSITY TARGET, not per case: every uniform case
    # shares a single glass, while a two-density case (case 10's slab) keys on
    # its own contrast so it never gets served the uniform one.
    prefile_extra="$sf_glass"
    glassfile_extra="${sf_glass}${GLASSNAME}"
    # Every dial in the name, always, like every other setup here: the folder
    # states the run outright instead of implying it through the case number.
    runname_extra="case$(fmt "${case_id}")cst$(fmt_compact "${sf_cstar}")pst$(fmt_compact "${sf_pstep}")c$(fmt "${sf_clump}")"
    icname_extra="$runname_extra"
    runname_extra="${runname_extra}${sf_tag}"
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
    # Set default values if not specified
    result=$(echo "scale=5; sqrt(6)/5" | bc -l)
    echo "$result"
    rhodiff=${args[0]:-2.0}
    mach=${args[1]:-$result}
    smooth=${args[2]:-1}
    B0=${args[3]:-0.0}
    Bdir=${args[4]:-1}
    # MHD only when a nonzero field amplitude is requested (B0=0 -> hydro).
    RUNPHYSICS="hydro"; [ "$(nonzero "$B0")" = 1 ] && RUNPHYSICS="mhd"

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
    select=${args[0]:-1.0}
    betain=${args[1]:-0.0}
    ublast=${args[2]:-10.0}
    Rin=${args[3]:-0.125}
    Pin=${args[4]:-100.0}
    Pout=${args[5]:-1.0}
    # MHD only when a nonzero blast plasma-beta is requested (beta=0 -> B=0 -> hydro).
    RUNPHYSICS="hydro"; [ "$(nonzero "$betain")" = 1 ] && RUNPHYSICS="mhd"
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
    # GLASS_FORCEMATCH=1: P = (gamma-1) u0 rho_t, g = grad P_t/rho_t = -(gamma-1) u0/(r+eps)
    # (not gravity: the collapse IC is not in equilibrium; same pre-IC step)
    FORCETABLE="forcetable.npz"
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
        # dICdensprofile=8 (Lane-Emden), which the Python glassgen backend
        # cannot evaluate analytically -> relax against the radial density
        # table the rand pre-IC writes (test_cases/polytrope/densitytable.npz,
        # the glass step's cwd). Self-describing: r_sph mapping in its header.
        GLASSMODE="--table densitytable.npz"
        # GLASS_FORCEMATCH=1: P = K rho^gamma, g = grad P_t/rho_t = -G M(r)/r^2 inside R
        # (hydrostatic target; same pre-IC step)
        FORCETABLE="forcetable.npz"
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
    # Set default values if not specified
    rhodiff=${args[0]:-10}
    beta=${args[1]:-0}
    mach=${args[2]:-2.7}
    inflow=${args[3]:-0}
    cool=${args[4]:-0}
    windref=${args[5]:-0}
    winddens=${args[6]:-26}
    tcloud=${args[7]:-4000}
    # MHD only when a nonzero ambient plasma-beta is requested (beta=0 -> hydro).
    RUNPHYSICS="hydro"; [ "$(nonzero "$beta")" = 1 ] && RUNPHYSICS="mhd"
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
    # dICdensprofile=8 (stratified atmosphere) - not analytic in the Python
    # glassgen backend; relax against the table the rand pre-IC writes
    # (mapping='z', carried in the table header).
    GLASSMODE="--table densitytable.npz"
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
    # dICdensprofile=8 (layered planet profile) - not analytic in the Python
    # glassgen backend; relax against the radial table the rand pre-IC writes.
    GLASSMODE="--table densitytable.npz"
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
    A_u=${args[3]:-0.1}
    clean_fch=${args[4]:-1.0}
    clean_damping=${args[5]:-0.0}
    # MHD when factor_A (omega_A) != 0; self-gravity when factor_G (omega_G) != 0.
    # Case 4 is always magnetic, so IC_setup_cosmowave rejects factor_A = 0 there.
    RUNPHYSICS="$(combine_physics "$(nonzero "$factor_A")" "$(nonzero "$factor_G")")"
    echo "Using settings for cosmowave (/Figx shows which case correlates with figure in Berlok 2022) :"
    echo "test case -a : $case  (0:standing Alfven/Fig1 1:traveling Alfven/Fig7 2:standing compr. gamma=4/3 EdS/Fig2 3:traveling compr. gamma=4/3 EdS/Fig9 4:div-clean EdS/this work 5:standing compr. gamma=5/3 LambdaCDM/Fig5 6:isothermal EdS/Fig3 7:gamma=5/3 EdS/Fig4)"
    echo "factor_A (omega_A = factor_A*pi; 0=MHD off 1=default ) -b : $factor_A"
    echo "factor_G (omega_G = factor_G*pi/2; 0=self-grav off, 0=default) -c : $factor_G"
    echo "linear-mode amplitude A_u (Berlok2022: 1e-6; SPH default: 0.1) -d : $A_u"
    echo "div-clean wave-speed multiplier fch (case 4 only; 0=off) -e : $clean_fch"
    echo "div-clean damping psidecfac (case 4 only; 0=pure hyperbolic) -f : $clean_damping"
    # Order matches IC_setup_cosmowave.create(... case, factor_A, factor_G, A_u) signature
    EXTRACONDIC="$case $factor_A $factor_G $A_u "
    # thermaldiff/alphaB would damp the same Fourier mode this test measures, and
    # bDivBClean is a frozen-particle preprocessing pass that would start the
    # evolution from an already-cleaned field instead of the analytic IC.
    #EXTRACONDRUN=" -thermaldiff 0 -bDivBClean 0 "
    # Special arguments for the cleaning test: fch and psidecfac, and no static
    # pre-cleaning so the evolution starts from the analytic IC's divergence
    if [ "$case" = "4" ]; then
        EXTRACONDRUN="$EXTRACONDRUN -fch $clean_fch -psidecfac $clean_damping -bDivBClean 0 "
    fi
    if [ "$case" = "3" ]; then
        prefile_extra="case3A$(fmt "${factor_A}")G$(fmt "${factor_G}")u$(fmt "${A_u}")"
    else
        prefile_extra="uniform"
    fi
    glassfile_extra="${prefile_extra}${GLASSNAME}"
    runname_extra="case$(fmt "${case}")A$(fmt "${factor_A}")G$(fmt "${factor_G}")u$(fmt "${A_u}")"
    if [ "$case" = "4" ]; then
        runname_extra="${runname_extra}f$(fmt "${clean_fch}")g$(fmt "${clean_damping}")"
    fi
}

# Function for the isolated dissipation tests (operators and switches).
#
# Every case runs in a STATIC (-b 0, bComove=0) and an EXPANDING (-b 1,
# bComove=1) arm.  The static arm measures the discrete operator with the scale
# factor frozen out of the problem; the expanding arm is compared against it,
# so what is under test is the scale-factor bookkeeping alone rather than the
# operator and its a-factors tangled together.
setup_disstest() {
    # Set default values if not specified
    case=${args[0]:-0}
    cosmo=${args[1]:-1}
    amp=${args[2]:-0.1}
    bfac=${args[3]:-1.0}
    variant=${args[4]:-0}
    a_init=${args[5]:-0}
    vamp=${args[6]:-0}
    diss_coeff=${args[9]:-0}
    strat=${args[10]:-0}
    # Every case except a deliberately hydro-only run carries a magnetic field:
    # IC_setup_disstest writes a uniform Bx for cases 0/1/3 and the helical
    # mode for case 2, all scaled by bfac.
    RUNPHYSICS="$(combine_physics "$(nonzero "$bfac")" 0)"
    echo "Using settings for disstest:"
    echo "test case -a : $case  (0:null 1:uwave/conduction 2:bhelix/resistivity 3:ladder/flow-driven)"
    echo "arm -b : $cosmo  (1 = expanding bComove=1, 0 = static bComove=0)"
    echo "field perturbation amplitude -c : $amp"
    echo "magnetic amplitude factor (0 = hydro only) -d : $bfac"
    echo "case 3 flow -e : $variant  (0 longitudinal/compressive, 1 transverse/shear)"
    echo "expanding arm initial scale factor a_i (0 = default 1/128) -f : $a_init"
    echo "case 3 velocity amplitude (0 = use -c) -g : $vamp"
    echo "fixed dissipation coefficient (cases 1,2) -j : $diss_coeff"
    echo "case 1 background density stratification (0 = uniform) -k : $strat"
    # Order matches IC_setup_disstest.create(... case, cosmo, amp, bfac,
    # variant, a_init, vamp) signature
    EXTRACONDIC="$case $cosmo $amp $bfac $variant $a_init $vamp $strat "
    # Case 0 is the homogeneous null: it must run the PRODUCTION dissipation
    # configuration, because the point is that every channel is a pair
    # DIFFERENCE that vanishes on a uniform state.  Zeroing the coefficients
    # would make it pass trivially.  The only addition is +hydroout, which
    # writes the per-particle uDotPdV/uDotAV/uDotDiff arrays the analysis reads.
    if [ "$case" = "0" ]; then
        EXTRACONDRUN="$EXTRACONDRUN +hydroout "
    fi
    # Cases 1 and 2 isolate ONE fixed-coefficient parabolic channel and turn
    # every other one off by coefficient, so the analysis can identify the
    # active channel from the run's own log rather than being told.
    if [ "$case" = "1" ]; then
        if [ "$(nonzero "$diss_coeff")" != "1" ]; then
            echo "disstest case 1 (uwave) needs a conduction coefficient (-j)" >&2
            exit 1
        fi
        EXTRACONDRUN="$EXTRACONDRUN +hydroout -thermaldiff 0 +thermalcondlinearcodeunits -thermalcond $diss_coeff "
    fi
    if [ "$case" = "2" ]; then
        if [ "$(nonzero "$diss_coeff")" != "1" ]; then
            echo "disstest case 2 (bhelix) needs a resistivity coefficient (-j)" >&2
            exit 1
        fi
        EXTRACONDRUN="$EXTRACONDRUN +hydroout +constdiff -magdiffturb $diss_coeff -magdifflin 0 -magdiffshock 0 -thermaldiff 0 -metaldiff 0 -alphaB 0 -fch 0 -psidecfac 0 "
    fi
    # Case 3 reads the RHS at the template, so it wants ONE tiny step rather
    # than an evolution; the per-arm coefficient switches (-alpha/-beta,
    # -thermaldiff) select which channel is being laddered and are passed
    # through `extra`.
    if [ "$case" = "3" ]; then
        EXTRACONDRUN="$EXTRACONDRUN +hydroout -dt 1e-8 -n 1 -oi 1 "
    fi
    # A stratified run is RELAXED IN its profile, so it must not share the
    # uniform glass cache -- the prefile name is what keys that cache.
    if [ "$(nonzero "$strat")" = "1" ]; then
        prefile_extra="strat$(fmt "${strat}")"
    else
        prefile_extra="uniform"
    fi
    glassfile_extra="${prefile_extra}${GLASSNAME}"
    runname_extra="case$(fmt "${case}")c$(fmt "${cosmo}")a$(fmt "${amp}")b$(fmt "${bfac}")"
    if [ "$case" = "3" ]; then
        runname_extra="${runname_extra}v$(fmt "${variant}")"
    fi
    if [ "$case" = "1" ] || [ "$case" = "2" ]; then
        runname_extra="${runname_extra}k$(fmt "${diss_coeff}")"
    fi
    if [ "$(nonzero "$a_init")" = "1" ]; then
        runname_extra="${runname_extra}ai$(fmt "${a_init}")"
    fi
    if [ "$(nonzero "$strat")" = "1" ]; then
        runname_extra="${runname_extra}s$(fmt "${strat}")"
    fi
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
    ampl=${args[1]:-3e-2}
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
    hr=${args[0]:-9.5e-7}
    rinner=${args[1]:-1.0}
    rdisk=${args[2]:-2.0}
    q=${args[3]:-0.5}
    echo "Using settings for cold Keplerian disk/ring:"
    echo "thermal coefficient hr=P/rho at R=1 (small=cold; nominal Mach~1/sqrt(gamma*hr)) -a : $hr"
    echo "inner ring radius -b : $rinner"
    echo "outer ring radius -c : $rdisk"
    echo "thermal exponent q in P/rho~r^-q (Sigma and v_phi remain r^-1/2) -d : $q"
    # Order matches IC_setup_coldkeplerian.create(... hr, rinner, rdisk, q) signature
    EXTRACONDIC="$hr $rinner $rdisk $q "
    EXTRACONDRUN=""
    prefile_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")"
    glassfile_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")${GLASSNAME}"
    runname_extra="hr$(fmt "${hr}")ri$(fmt "${rinner}")ro$(fmt "${rdisk}")q$(fmt "${q}")"
}

# Function for the generalized external power-law disk or annulus.
setup_powerlawring() {
    beta=${args[0]:--0.5}
    geometry=${args[1]:-3}
    field_mode=${args[2]:-0}
    plasma_beta=${args[3]:-1e6}
    # Default softening 0.05, matching coldkeplerian's central sink softening so
    # the two Keplerian tests share one central regularization.
    # It changes v_c by under 0.2% across the science annulus and 0.74% at the
    # ring's inner edge, is exactly zero for beta=1, and bounds max|a| at 154
    # (dt ~ 0.023) instead of letting it diverge at the axis.
    softening=${args[4]:-0.05}
    tilt=${args[5]:-0.0}
    domain_mode=${args[6]:-0}
    scale_height=${args[7]:-0.15}
    force_form=${args[8]:-0}
    # The Hopkins literature ring (-i 1) is a PRESET, not a point in this
    # argument space: it is cylindrical, unsoftened, untilted, hydrodynamic,
    # Keplerian and annular by definition, and its temperature is the paper's
    # P=1e-6 on Sigma=1.01 rather than a free knob.  Fill those in when the
    # argument was omitted, and leave an explicit contradiction to
    # IC_setup_powerlawring.create(), which is this suite's single validation
    # choke point and rejects every one of them by name before any glass is
    # relaxed.  Duplicating those checks here only let the two drift.
    if [ "$force_form" = 1 ]; then
        if [ -z "${args[1]}" ]; then geometry=2; fi
        if [ -z "${args[4]}" ]; then softening=0; fi
        if [ -z "${args[5]}" ]; then tilt=0.0; fi
    fi
    case "$field_mode" in
        0) RUNPHYSICS="grav" ;;
        1|2|3|4) RUNPHYSICS="grav mhd" ;;
        *) echo "Error: powerlawring field mode must be 0, 1, 2, 3, or 4."; exit 1 ;;
    esac
    case "$domain_mode" in
        0|1) ;;
        *) echo "Error: powerlawring domain mode must be 0 (annulus) or 1 (full disk)."; exit 1 ;;
    esac
    case "$force_form" in
        0|1) ;;
        *) echo "Error: powerlawring radial mode must be 0 (power law) or 1 (bounded Kepler)."; exit 1 ;;
    esac
    case "$geometry" in
        2|3) ;;
        *) echo "Error: powerlawring geometry must be 2 (cylindrical) or 3 (spherical)."; exit 1 ;;
    esac
    # Only gate an explicitly requested distribution: running the test by name
    # with no positional arguments must still print its parameter list.
    # The Hopkins ring also accepts an ORDERED start, because the paper
    # initialises its disc on an evenly spaced grid and then removes the
    # particles outside R=0.5..2: distribution 0 (FCC) and 3 (cubic) lay the
    # lattice and let the carve do the rest -- the ring is flat, so there is
    # no stretch map.  Every other powerlawring target is relaxed, so glass only.
    if [ -n "$requested_distribution" ] && [ "$requested_distribution" != 2 ]; then
        if [ "$force_form" = 1 ] && { [ "$requested_distribution" = 0 ] ||
                                     [ "$requested_distribution" = 3 ]; }; then
            :
        else
            echo "Error: powerlawring requires glass distribution 2 (the Hopkins"
            echo "       ring -i 1 also accepts 0 FCC or 3 cubic)."
            exit 1
        fi
    fi
    # The external-force parameters exist only in the selected Gasoline tree
    # and in ChaNGa MHDTP (externalForce.cpp); reject other codes. The ChaNGa
    # adapter turns the .param's bDoSelfGravity = 0 into -g.
    RUNCODES="0 1"
    echo "Using settings for external power-law disk/annulus:"
    echo "circular-speed slope beta (-0.5 Kepler, 0 flat, 1 solid body) -a : $beta"
    echo "geometry (2 cylindrical slab, 3 spherical equilibrium) -b : $geometry"
    echo "field mode (0 hydro, 1 toroidal C/R force-free, 2 radial winding, 3 vertical invariant, 4 constant toroidal center-regular) -c : $field_mode"
    echo "plasma beta 2P/B^2 at R=1 (sets the magnetic amplitude) -d : $plasma_beta"
    echo "potential softening -e : $softening"
    echo "orbital-plane tilt in degrees (requires geometry 3) -f : $tilt"
    echo "radial domain (0 annulus R=0.5..2, 1 full disk R=0..2) -g : $domain_mode"
    if [ "$force_form" = 1 ]; then
        echo "temperature: Hopkins P=1e-6 on Sigma=1.01, i.e. H=1.0188e-3, Mach(R=1)=778 (-h is ignored)"
    else
        echo "barotrope scale height H (temperature; Mach(R=1) ~ 1/(1.261 H)) -h : $scale_height"
    fi
    echo "force law (0 power law, free edges; 1 bounded Kepler, Hopkins ring) -i : $force_form"
    # Order matches IC_setup_powerlawring.create(... beta, geometry,
    # field_mode, plasma_beta, softening, tilt, domain_mode, scale_height,
    # force_form).
    # V0=R0=1 are fixed test units.
    EXTRACONDIC="$beta $geometry $field_mode $plasma_beta $softening $tilt $domain_mode $scale_height $force_form "
    EXTRACONDRUN=""
    # Each glass identity carries exactly what moves its own target.  The
    # spherical arm's lowered-isothermal atmosphere depends on H, the cutoff
    # and floor values, beta and softening; geometry 2 has no vertical gravity,
    # so both of its targets are z-uniform slabs that depend only on the radial
    # domain.  The point target is retained exactly inside the cull contour,
    # and the mapped-target adapter supplies a q128 Cartesian volume measure to
    # the proven M20/table mass closure wherever an envelope runs.
    # "slab2r6dx" is the z period rule -- half period = 2*sqrt(6) particle
    # spacings, the suite's thin-box convention (orzag, gresho, currentsheet,
    # khslab, taylorgreen, noh, square, mhddivtest) -- and must track
    # SLAB_HALF_HEIGHT_SPACINGS in IC_setup_powerlawring.py.  The VALUE is not
    # part of the tag because it is a function of nx, which the
    # "<testsim><nx>_" filename prefix already carries: the same rule at a
    # different nx is a different slab and therefore a different glass.
    # The radial tag records the EMITTED radial construction, so the two are
    # never confused in a cache. The spherical atmosphere carries density to
    # zero across disposable support bands and keeps them, hence its science
    # edge plus support edges. A cylindrical ring has no taper and IS the
    # science domain, so it names only its two surfaces. The "wi/wo" support
    # radii in the tapered spelling keep existing spherical glasses addressable.
    radial_identity=""
    if [ "$geometry" = 2 ]; then
        if [ "$domain_mode" = 0 ]; then
            radial_identity="ri0.5ro2"
        else
            radial_identity="ro2"
        fi
    elif [ "$domain_mode" = 0 ]; then
        radial_identity="ri0.5wi0.25ro2wo2.5"
    fi
    if [ "$geometry" = 2 ]; then
        # Cylindrical sharp ring -- the suite's free-edge arm and the Hopkins
        # literature ring alike.  Relax a plain UNIFORM glass in a periodic
        # box whose z period is the run's z period, then carve the ring out of
        # it.  No envelope/rho_edge/cull and a seam-free wrap, so nothing but
        # the source box keys that glass -- and that box is now the ring's own
        # bounding box, so the radial identity keys it.
        #
        # The two force laws SHARE this glass on purpose: the relaxed object
        # is the same uniform box at the same z period over the same bounds,
        # and they differ only in temperature and runtime force, neither of
        # which the relaxer sees.  The thermal and form tags below keep the RUN
        # names
        # apart.
        GLASSBOUNDARY="param"
        GLASSMODE=""
        profile_key="ringuniformslab2r6dx${radial_identity}dom$(fmt "${domain_mode}")"
        glass_envelope_key=""
    else
        GLASSBOUNDARY="open"
        GLASSMODE="--rho-edge-max 0.05"
        profile_key="eqbaroH$(fmt "${scale_height}")rc0.01rf0.05sk0.5ef1e-08beta$(fmt "${beta}")geom$(fmt "${geometry}")eps$(fmt "${softening}")dom$(fmt "${domain_mode}")${radial_identity}"
        glass_envelope_key="ptm20nf16nl1p35q128"
    fi
    prefile_extra="$profile_key"
    glassfile_extra="${profile_key}${glass_envelope_key}${GLASSNAME}"
    if [ "$force_form" = 1 ]; then
        # create() overrides -h with hopkins_scale_height(), so name the
        # published pressure rather than a scale height the run ignores.
        thermal_identity="P2d1e-6"
    else
        thermal_identity="H$(fmt "${scale_height}")"
    fi
    runname_extra="beta$(fmt "${beta}")geom$(fmt "${geometry}")fm$(fmt "${field_mode}")pb$(fmt_compact "${plasma_beta}")eps$(fmt "${softening}")tilt$(fmt "${tilt}")dom$(fmt "${domain_mode}")${radial_identity}${thermal_identity}form$(fmt "${force_form}")"
}

# Fuzzy-DM quantum-pressure tests. The
# particles are dark matter: the pre-IC and glass are relaxed as gas, and
# IC_createsetup writes the final IC as dark particles (the setups' dark_only
# flag). -a..-h only change the IC: the boson mass is a setup constant
# (fdm_mass_eV, written to the .param). FDM run-time options (ChaNGa's
# estimator is V5) go through [extra], e.g. '-iFDMHMode 2 +bFDMSoftH',
# labelled by EXTRANAME.
# Static operator tests (Nori & Baldi 2018): no flags.
setup_fdm_static() {  # $1 = test description
    RUNPHYSICS="grav fdm"
    echo "Using settings for $1 (no -a..-h; the FDM variant goes through [extra] + EXTRANAME)."
    EXTRACONDIC=" "
    EXTRACONDRUN=""
    prefile_extra=""
    glassfile_extra="${GLASSNAME}"
    runname_extra=""
    # The rand pre-IC writes the target table into the glass step's cwd.
    GLASSMODE="--table densitytable.npz"
}
# soliton (IC_setup_soliton.py), -a case: 0 (default) the static Gaussian core;
# 1 the same Gaussian as a free wave packet (gravity off, periodic, to t = tau,
# same glass and IC as case 0);
# 2 the self-gravitating Schroedinger-Poisson ground state (r_c = 1 kpc), which
# must stay in equilibrium; 3 head-on collision of two case-2 solitons, -b
# separation / r_c (default 10), -c relative speed / v_c(r_c) (default 6).
# Presets: fast -b 10 -c 6 (passes through), slow -b 8 -c 1 (bound). 4: halo
# from -b N (default 4) case-2 solitons at rest, centres drawn (seeded) in a
# sphere of -c R r_c (default 6). 5: the case-2 soliton with baryons, each -b
# (default 0.001) x its mass: tracer stars on circular orbits and hydrostatic
# gas with c_s = -c (default 0.5) x v_c(r_c), -d gas EOS (0 isothermal, 1
# adiabatic); the ChaNGa adapter passes <Nsmooth> as nSmoothFDM, the gas keeps
# the .param nSmooth. Cases 2-5 relax a uniform glass cube and map a
# sphere of it onto the soliton; the setup sets the run length (629 Myr
# unbound, 1573 Myr bound). Case 3/4 references:
# setupfiles/soliton_reference.py -> reference_solution_highres/soliton/.
setup_soliton() {
    sol_case=${args[0]:-0}
    if [ "$sol_case" = 0 ]; then
        setup_fdm_static "the FDM Gaussian core (soliton) test"
        return
    fi
    if [ "$sol_case" = 1 ]; then
        setup_fdm_static "the FDM free wave packet (soliton case 1)"
        EXTRACONDIC="1 "
        runname_extra="case1"
        return
    fi
    RUNPHYSICS="grav fdm"
    sep=${args[1]:-10}
    vrel=${args[2]:-6}
    echo "Using settings for the FDM soliton test (the FDM variant goes through [extra] + EXTRANAME):"
    echo "case -a : $sol_case  (0 static core, 1 free packet, 2 ground state, 3 collision, 4 halo, 5 with baryons)"
    EXTRACONDRUN=""
    # Cases 2 and 3 share the uniform glass; the sphere is mapped in IC_setup_soliton.
    prefile_extra="uniform"
    glassfile_extra="uniform${GLASSNAME}"
    GLASSMODE=""
    if [ "$sol_case" = 4 ]; then
        nsol=${args[1]:-4}
        rhalo=${args[2]:-6}
        echo "number of solitons -b : $nsol"
        echo "radius of the sphere holding the centres / r_c -c : $rhalo"
        # Order matches IC_setup_soliton.create(... case, D, v): case 4 reads D = N, v = R
        EXTRACONDIC="$sol_case $nsol $rhalo "
        runname_extra="case4N$(fmt "${nsol}")R$(fmt "${rhalo}")"
    elif [ "$sol_case" = 5 ]; then
        RUNPHYSICS="hydro grav fdm"
        bfrac=${args[1]:-0.001}
        csvc=${args[2]:-0.5}
        geos=${args[3]:-0}
        echo "baryon (star and gas) mass fraction -b : $bfrac"
        echo "gas sound speed / v_c(r_c) -c : $csvc"
        echo "gas EOS -d : $geos  (0 isothermal, 1 adiabatic)"
        # Order matches IC_setup_soliton.create(... case, D, v, eos): case 5 reads D = b, v = c_s / v_c
        EXTRACONDIC="$sol_case $bfrac $csvc $geos "
        runname_extra="case5b$(fmt "${bfrac}")c$(fmt "${csvc}")eos$(fmt "${geos}")"
    elif [ "$sol_case" = 3 ]; then
        echo "separation / r_c -b : $sep"
        echo "relative speed / v_c(r_c) -c : $vrel"
        # Order matches IC_setup_soliton.create(... case, D, v)
        EXTRACONDIC="$sol_case $sep $vrel "
        runname_extra="case3D$(fmt "${sep}")v$(fmt "${vrel}")"
    else
        EXTRACONDIC="$sol_case "
        runname_extra="case$(fmt "${sol_case}")"
    fi
}
setup_gaussOverdens() { setup_fdm_static "the FDM 3D Gaussian overdensity test"; }
setup_1DdensityFront() { setup_fdm_static "the FDM 1D density front test"; }

# FDM linear waves (IC_setup_fdmwave.py). Case 0: free quantum dispersion in a
# periodic slab, gravity off: rho0 (1 + A cos kx) at rest oscillates as
# A cos(omega t), omega = (hbar/2m) k^2. Cases 1-3 are self-gravitating in a
# periodic cube cut from one shared uniform glass, the mode imprinted as a
# Zel'dovich displacement: 1 Jeans test (k_J = 2 pi N_J / L, N_J = 1.5 or -d:
# n < N_J grows, n > N_J oscillates at the reduced frequency), 2 comoving Jeans crossing (EdS, a = 0.1
# -> 1; n = 1 crosses k_J at a = 0.2), 3 null test (uniform glass, box < Jeans
# length; -b/-c ignored). Cases 4 and 5 are exact free solutions in the slab of
# case 0 (Hopkins 2019): 4 two interfering plane waves (-c contrast, 1 = nodes),
# 5 two-stream intersection (a converging Gaussian that bounces; -b/-c ignored),
# 6 three interfering plane waves with discontinuous nodes (-c must be 1).
# Case 7: the box of case 2, started on the growing solution f(xi) of Schive's
# review (Fig. 11), a = 0.06 -> 2; modes n = 1..3 (4) together cover its xi range.
setup_fdmwave() {
    # Same physics set as the static FDM tests, so they share one build; the
    # setup sets gravity (case 0 off) and the cosmology (case 2) in the .param.
    RUNPHYSICS="grav fdm"
    fdm_case=${args[0]:-0}
    mode=${args[1]:-1}
    ampl=${args[2]:-3e-2}
    if [ "$fdm_case" = 4 ] || [ "$fdm_case" = 6 ]; then ampl=${args[2]:-1}; fi
    if [ "$fdm_case" = 7 ]; then ampl=${args[2]:-1e-2}; fi
    echo "Using settings for the FDM linear wave test (the FDM variant goes through [extra] + EXTRANAME):"
    echo "case -a : $fdm_case  (0 free dispersion, 1 Jeans, 2 comoving Jeans crossing, 3 null test, 4 two-wave nodes, 5 two-stream, 6 three-wave nodes, 7 comoving growing mode)"
    echo "mode number n, k = 2 pi n / L -b : $mode"
    echo "density amplitude A -c : $ampl"
    if [ "$fdm_case" = 1 ]; then
        thick=${args[3]:-1.5}
        echo "Jeans mode number N_J, k_J = 2 pi N_J / L -d : $thick"
    else
        thick=${args[3]:-1}
        echo "slab thickness / 24 rows (cases 0, 4, 5, 6; 2 for 400 neighbours) -d : $thick"
    fi
    # Order matches IC_setup_fdmwave.create(... case, mode, ampl, thick); case 1 reads thick as N_J
    EXTRACONDIC="$fdm_case $mode $ampl $thick "
    wtag=""; if [ "$thick" != 1 ]; then wtag="w$(fmt "${thick}")"; fi
    EXTRACONDRUN=""
    if [ "$fdm_case" = 5 ]; then
        prefile_extra="case5${wtag}"
        glassfile_extra="${prefile_extra}${GLASSNAME}"
        runname_extra="${prefile_extra}"
    elif [ "$fdm_case" = 0 ] || [ "$fdm_case" = 4 ] || [ "$fdm_case" = 6 ]; then
        prefile_extra="case$(fmt "${fdm_case}")n$(fmt "${mode}")a$(fmt "${ampl}")${wtag}"
        glassfile_extra="${prefile_extra}${GLASSNAME}"
        runname_extra="${prefile_extra}"
    else
        # Cases 1-3 and 7 share the uniform cubic glass; the mode is applied in IC_setup_fdmwave.
        prefile_extra="cube"
        glassfile_extra="cube${GLASSNAME}"
        if [ "$fdm_case" = 3 ]; then
            runname_extra="case3"
        else
            runname_extra="case$(fmt "${fdm_case}")n$(fmt "${mode}")a$(fmt "${ampl}")"
            if [ "$fdm_case" = 1 ] && [ "$thick" != 1.5 ]; then runname_extra="${runname_extra}J$(fmt "${thick}")"; fi
        fi
    fi
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
    turbbox)
        setup_turbbox
        ;;
    starform)
        setup_starform
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
    disstest)
        setup_disstest
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
    powerlawring)
        setup_powerlawring
        ;;
    soliton)
        setup_soliton
        ;;
    gaussOverdens)
        setup_gaussOverdens
        ;;
    1DdensityFront)
        setup_1DdensityFront
        ;;
    fdmwave)
        setup_fdmwave
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
    echo "Usage: $0 <Nsmooth> <Nres> <0: FCC lattice 1: rand 2: glass 3: cubic lattice> <code 0:Gasoline 1:ChaNGa 2:SPH-EXA 3:AREPO 4:RAMSES 5:GIZMO 6:Phantom 7:Athena++ 8:PKDGRAV3 9:OpenGadget3> [vm] [extra]"
    exit 1
fi
echo "$0 $1 $2 $3 $4 $5 $6"

# Append the optional extra positional flags ($6) to the runtime conditions.
# Order = last wins in both Gasoline's and ChaNGa's prmArgProc: the suite-wide
# EXTRACONDALL first, then $6 (a campaign's TEST_VARIABLES), then the test's own
# EXTRACONDRUN last, so a flag the test's physics requires beats the suite-wide
# defaults (e.g. barotropic mhdcollapse switching thermaldiff off).
echo "$EXTRACONDALL"
EXTRACONDRUN=" $EXTRACONDALL $6 $EXTRACONDRUN "

# Varied-mass argument ($5): changes the particle distribution itself, so it
# must key the pre-IC file, the glass AND the run name (empty tag when off).
# Guard the slot: omitting vm shifts [extra] into it, and since fmt() passes
# non-numeric values through, the runtime flags would end up in the run NAME
# (blowing the Gasoline filename limit) while never reaching the run command.
if [ -n "$5" ] && ! [[ "$5" =~ ^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo "Error: vm (5th positional after the test name) must be numeric, got '$5'."
    echo "       Usage: $0 ... <distrib> <code> [vm] [extra] -- pass vm (0) explicitly"
    echo "       before any [extra] flag string."
    exit 1
fi
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
# d1 is IC_createsetup.py's `entry`: 0 close-packed (FCC) lattice, 3 simple cubic
# lattice, 1 random, "preglass" the random PRE-IC of a glass run (same particles
# as 1, but its .param is a glassgen input -- forced periodic box + the dICdens*
# target profile -- rather than a runnable simulation param).  IC_createsetup
# selects the lattice for every IC_distribute lattice call (see
# IC_distribute.set_lattice).
# prefile = the file IC_createsetup.py writes directly into datafiles/:
#  - lattice/rand: that file IS the final IC (velocities/B field included), so it
#    must be keyed by the FULL parameter set (runname_extra) -- otherwise a rerun
#    with e.g. a different B0 silently reuses the cached IC of the old run.
#  - glass: it is only the pre-relaxation density distribution, so it is keyed by
#    the density-structure params (prefile_extra) and shared across runs that
#    differ only in velocities/B; the final IC (built from the relaxed glass) is
#    keyed by runname_extra instead.  It is named _preglass, NOT _rand: a rand
#    run's IC is a different artefact (its own .param) and the two must not share
#    a cache entry.
#
# icname_extra is the escape hatch for a setup whose run name carries something
# the IC does NOT depend on -- a build arm, say.  Such runs must share ONE IC
# file, so that an A/B comparison between them subtracts by construction rather
# than by an argument about IC_createsetup being deterministic.  Defaults to
# runname_extra.
icname_extra="${icname_extra-$runname_extra}"
if [ "$3" -eq 0 ]; then
echo "RUNNING FCC LATTICE MODE"
    distri="fcc"
    distri2="fcc"
    d1=0
    simfilename="$testsim$2_$distri${icname_extra}"
    prefile="$simfilename"
elif [ "$3" -eq 1 ]; then
echo "RUNNING RANDOM MODE"
    distri="rand"
    distri2="rand"
    d1=1
    simfilename="$testsim$2_$distri${icname_extra}"
    prefile="$simfilename"
elif [ "$3" -eq 3 ]; then
echo "RUNNING CUBIC LATTICE MODE"
    distri="cubic"
    distri2="cubic"
    d1=3
    simfilename="$testsim$2_$distri${icname_extra}"
    prefile="$simfilename"
elif [ "$3" -eq 2 ]; then
echo "RUNNING GLASS MODE"
    export SPH_KERNEL GLASS_GRADIENT GLASS_SIGMA_F
    distri="preglass"
    distri2="glass"
    d1=preglass
    simfilename="$testsim$2_glassN$1${icname_extra}${GLASSTAG}${ICSTARTTAG}-I"
    prefile="$testsim$2_${distri}${prefile_extra}"
    echo "$simfilename"
elif [ "$3" -eq 4 ]; then
echo "RUNNING GLASS PATCH MODE"
    # Two uniform glasses relaxed apart, then pasted (setupfiles/IC_glasspatch.py). The contrast is always
    # sharp: the setup's target smoothing is ignored.
    case "$testsim" in
        kh|square) ;;
        *) echo "Error: distrib 4 (glass patch) is only set up for kh and square, not $testsim."; exit 1 ;;
    esac
    if [ "$GLASS_TARGET_SHAPE" = 1 ] || [ "$GLASS_FORCEMATCH" = 1 ] || [ "$GLASS_START_P" = pseudo ]; then
        echo "Error: distrib 4 (glass patch) has no relaxed contact, so it takes neither GLASS_TARGET_SHAPE=1, GLASS_FORCEMATCH=1 nor GLASS_START_P=pseudo."
        exit 1
    fi
    export SPH_KERNEL GLASS_GRADIENT GLASS_SIGMA_F
    distri="preglass"
    distri2="glasspatch"
    d1=preglass
    simfilename="$testsim$2_patchN$1${icname_extra}${GLASSTAG}-I"
    prefile="$testsim$2_${distri}${prefile_extra}"
    echo "$simfilename"
else
    echo "Error: unknown distribution '$3' (0: FCC lattice, 1: rand, 2: glass, 3: cubic lattice, 4: glass patch)."
    exit 1
fi

# Code name and binary/launcher selection based on the 4th argument
# Code selection is a table lookup: name, build directory, default binary. A
# managed code's build adapter replaces both below -- and a code with physics
# variants overrides $coderun in its own block.
[ -n "${CODE_NAME[$4]:-}" ] || { echo "Error: unknown code $4 (see 'runtest.sh support')."; exit 1; }
codename="${CODE_NAME[$4]}"
dir="${CODE_BUILD_DIR[$4]}"
coderun="${CODE_BINARY[$4]}"
echo "RUNNING $codename"

# RUNPHYSICS support gate: refuse (code, physics) pairs this script can't run,
# BEFORE building any IC. RUNPHYSICS was declared by setup_<testsim>() above; the
# code blocks below map it to the per-code binary/config.
if [ -z "${RUNPHYSICS:-}" ]; then
    echo "Internal error: setup_$testsim did not set RUNPHYSICS."; exit 1
fi
if missing_capability="$(physics_unsupported "$4" "$RUNPHYSICS")"; then
    echo "Error: $codename is not wired for '$missing_capability'"
    echo "       (test '$testsim' declares RUNPHYSICS=\"$RUNPHYSICS\")."
    echo "       $codename supports: ${SUPPORT[$4]}."
    echo "       (See 'runtest.sh support'; edit the SUPPORT matrix as codes gain physics.)"
    exit 1
fi
if [ -n "${RUNCODES:-}" ] && ! code_allowed "$4" $RUNCODES; then
    echo "Error: test '$testsim' is implemented only for code(s): $RUNCODES."
    echo "       Selected code $4 ($codename) does not expose its runtime parameters."
    exit 1
fi
# The set defines the build. It goes to an adapter whole and the code blocks
# below read it with physics_has/physics_build_variant, so its vocabulary is
# validated here, at the one choke point, before anything can consume it.
case "$(physics_value eos "$RUNPHYSICS")" in
    ''|adiabatic|barotropic|isor) ;;
    *) echo "Internal error: setup_$testsim asked for an unknown eos in RUNPHYSICS=\"$RUNPHYSICS\"."; exit 1 ;;
esac
echo "Requirements: $testsim -> RUNPHYSICS=\"$RUNPHYSICS\" ($codename)"

# Managed compilation is selected only after the end-to-end wiring gate has
# passed, but before ICs or run directories are created. Each adapter owns the
# translation from semantic requirements to native build flags.
build_run_tag=""
managed_adapter=""
managed_label=""
managed_prefix=""
managed_builder=""
managed_build_manifest=""
managed_build_key=""
managed_build_alias=""
managed_alias_mapping=""
managed_binary=""
managed_binary_sha=""
managed_launcher=""
if [ -n "${CODE_BUILD_ADAPTER[$4]:-}" ]; then
    managed_builder="${CODE_BUILD_ADAPTER[$4]}"
    managed_prefix="${CODE_NAME[$4]}"
    managed_label="${CODE_LABEL[$4]}"
    managed_adapter="${CODE_SLUG[$4]}"
    [ -r "$managed_builder" ] || {
        echo "Error: $managed_label build adapter is missing: $managed_builder"; exit 1;
    }
    managed_pin_name="${managed_prefix}_BINARY"
    managed_allow_name="${managed_prefix}_ALLOW_UNMANAGED_BINARY"
    managed_pinned_binary="${!managed_pin_name}"
    managed_allow_unmanaged="${!managed_allow_name}"
    if [ -n "$managed_pinned_binary" ]; then
        if command -v realpath >/dev/null 2>&1; then
            managed_binary="$(realpath -m "$managed_pinned_binary")"
        else
            managed_binary="$managed_pinned_binary"
        fi
        [ -x "$managed_binary" ] || {
            echo "Error: $managed_pin_name is not executable: $managed_binary"; exit 1;
        }
        managed_artifact_dir="$(dirname "$managed_binary")"
        candidate_manifest="$managed_artifact_dir/manifest.json"
        candidate_key="$managed_artifact_dir/build-key"
        if [ -f "$candidate_manifest" ] && [ -f "$candidate_key" ]; then
            managed_build_manifest="$candidate_manifest"
            echo "Pinned managed $managed_label build: $managed_artifact_dir"
        elif [ "${managed_allow_unmanaged:-0}" = 1 ]; then
            if command -v sha256sum >/dev/null 2>&1; then
                managed_binary_sha="$(sha256sum "$managed_binary" | awk '{print $1}')"
            else
                managed_binary_sha="$(shasum -a 256 "$managed_binary" | awk '{print $1}')"
            fi
            managed_build_key="$managed_binary_sha"
            echo "WARNING: using unmanaged $managed_pin_name; build flags are not verifiable."
        else
            echo "Error: pinned $managed_pin_name has no sibling manifest.json and build-key."
            echo "       Set $managed_allow_name=1 only for an intentional legacy escape hatch."
            exit 1
        fi
    else
        managed_resolve=(bash "$managed_builder" resolve --physics "$RUNPHYSICS")
        [ -z "${CODE_BUILD_DIR[$4]}" ] || managed_resolve+=(--source "${CODE_BUILD_DIR[$4]}")
        if ! managed_binary="$("${managed_resolve[@]}")"; then
            echo "Error: $managed_label could not satisfy RUNPHYSICS=\"$RUNPHYSICS\"."
            exit 1
        fi
        if [[ "$managed_binary" == *$'\n'* ]] || [ ! -x "$managed_binary" ]; then
            echo "Error: $managed_label resolver did not return one executable path: $managed_binary"
            exit 1
        fi
        managed_artifact_dir="$(dirname "$managed_binary")"
        managed_build_manifest="$managed_artifact_dir/manifest.json"
        [ -f "$managed_build_manifest" ] && [ -f "$managed_artifact_dir/build-key" ] || {
            echo "Error: resolved $managed_label artifact lacks provenance: $managed_artifact_dir"; exit 1;
        }
    fi

    if [ -n "$managed_build_manifest" ]; then
        manifest_info="$(python3 - "$managed_build_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
print(
    manifest.get("build_key", ""),
    manifest.get("binary", {}).get("sha256", ""),
    manifest.get("profile", ""),
    manifest.get("adapter", ""),
    manifest.get("launcher", {}).get("sha256", ""),
    sep="\t",
)
PY
)" || { echo "Error: could not read $managed_label manifest $managed_build_manifest"; exit 1; }
        IFS=$'\t' read -r manifest_key manifest_sha manifest_profile manifest_adapter manifest_launcher_sha <<<"$manifest_info"
        managed_build_key="$(<"$managed_artifact_dir/build-key")"
        [ -n "$manifest_key" ] && [ "$manifest_key" = "$managed_build_key" ] || {
            echo "Error: $managed_label manifest/build-key mismatch in $managed_artifact_dir"; exit 1;
        }
        [ "$manifest_adapter" = "$managed_adapter" ] || {
            echo "Error: artifact adapter '$manifest_adapter' is not '$managed_adapter': $managed_artifact_dir"; exit 1;
        }
        if command -v sha256sum >/dev/null 2>&1; then
            managed_binary_sha="$(sha256sum "$managed_binary" | awk '{print $1}')"
        else
            managed_binary_sha="$(shasum -a 256 "$managed_binary" | awk '{print $1}')"
        fi
        [ -n "$manifest_sha" ] && [ "$manifest_sha" = "$managed_binary_sha" ] || {
            echo "Error: $managed_label binary checksum does not match $managed_build_manifest"; exit 1;
        }
        managed_supports=(bash "$managed_builder" supports --physics "$RUNPHYSICS")
        if ! expected_profile="$("${managed_supports[@]}")"; then
            echo "Error: $managed_label adapter does not support the pinned run requirements."; exit 1
        fi
        [ -n "$manifest_profile" ] && [ "$manifest_profile" = "$expected_profile" ] || {
            echo "Error: pinned $managed_label profile '$manifest_profile' cannot satisfy required profile '$expected_profile'."
            exit 1
        }
    fi

    if [ "$4" -eq 1 ]; then
        if [ -n "$managed_build_manifest" ]; then
            managed_launcher="$managed_artifact_dir/charmrun"
        else
            managed_launcher="${CHANGA_LAUNCHER:-$managed_artifact_dir/charmrun}"
        fi
        [ -x "$managed_launcher" ] || {
            echo "Error: ChaNGa launcher is not executable: $managed_launcher"; exit 1;
        }
        if [ -n "$managed_build_manifest" ]; then
            if command -v sha256sum >/dev/null 2>&1; then
                actual_launcher_sha="$(sha256sum "$managed_launcher" | awk '{print $1}')"
            else
                actual_launcher_sha="$(shasum -a 256 "$managed_launcher" | awk '{print $1}')"
            fi
            [ -n "$manifest_launcher_sha" ] && [ "$manifest_launcher_sha" = "$actual_launcher_sha" ] || {
                echo "Error: ChaNGa launcher checksum does not match $managed_build_manifest"; exit 1;
            }
        fi
        runlauncher="$managed_launcher ${CHANGA_CHARMRUN_OPTS:-+p $ncores}"
    fi
    dir="$(dirname "$managed_binary")"
    coderun="$(basename "$managed_binary")"
    managed_alias_helper="$codes_dir/build_alias.sh"
    if [ -n "$managed_build_manifest" ]; then
        [ -r "$managed_alias_helper" ] || {
            echo "Error: $managed_label build-alias helper is missing: $managed_alias_helper"; exit 1;
        }
        if ! managed_build_alias="$(bash "$managed_alias_helper" register "$managed_artifact_dir")"; then
            echo "Error: could not register the $managed_label build alias for $managed_artifact_dir."
            exit 1
        fi
        [[ "$managed_build_alias" =~ ^[A-Z]+$ ]] || {
            echo "Error: invalid $managed_label build alias: $managed_build_alias"; exit 1;
        }
        if ! managed_alias_mapping="$(bash "$managed_alias_helper" path "$(dirname "$managed_artifact_dir")")"; then
            echo "Error: could not resolve the $managed_label build-alias mapping path."
            exit 1
        fi
        [ -f "$managed_alias_mapping" ] || {
            echo "Error: $managed_label alias mapping was not created: $managed_alias_mapping"; exit 1;
        }
        build_run_tag="-$managed_build_alias"
    else
        # Unmanaged compatibility pins have no artifact to register, so they
        # retain the self-describing short content hash.
        build_run_tag="cfg${managed_build_key:0:12}"
    fi
    echo "$managed_label executable: $managed_binary"
    echo "$managed_label build key: $managed_build_key"
    if [ -n "$managed_build_alias" ]; then
        echo "$managed_label build alias: $managed_build_alias ($managed_alias_mapping)"
    fi
fi

# All codes 0-9 are wired: 0 Gasoline, 1 ChaNGa, 2 SPH-EXA, 3 AREPO, 4 RAMSES
# (cubic, hydro+MHD), 5 GIZMO, 6 Phantom, 7 Athena++ (cubic, hydro+MHD,
# uniform), 8 PKDGRAV3, 9 OpenGadget3.

glassfile_prefix="$testsim$2_glassN$1${glassfile_extra}${GLASSTAG}"
[ "$3" -eq 4 ] && glassfile_prefix="$testsim$2_patchN$1${glassfile_extra}${GLASSTAG}"
ICfile="${simfilename}"
runname="$testsim$2_N$1${distri2}${runname_extra}${EXTRANAME}${codename}${build_run_tag}"
directory="initruns"
glassfile="${directory}/${glassfile_prefix}"

# Legacy Gasoline parses -I/-o into 256-byte strings and main.c still formats
# output names in 256-byte local buffers. The -o value repeats the run identity
# as <directory>/<prefix>, so catch an oversized identity here rather than
# allowing prmArgProc to abort on its opaque iSize assertion. Reserve eight
# characters for standard output suffixes such as .sinklog/.starlog.
if [ "$4" -eq 0 ]; then
    gasoline_output_prefix="$runname/$runname"
    gasoline_input_name="datafiles/$ICfile.00000"
    gasoline_output_limit=247
    gasoline_string_limit=255
    if [ "${#gasoline_output_prefix}" -gt "$gasoline_output_limit" ]; then
        echo "Error: Gasoline output prefix is ${#gasoline_output_prefix} characters; maximum safe length is $gasoline_output_limit."
        echo "       Gasoline receives -o '$gasoline_output_prefix' and uses fixed 256-byte filename buffers."
        exit 1
    fi
    if [ "${#gasoline_input_name}" -gt "$gasoline_string_limit" ]; then
        echo "Error: Gasoline input name is ${#gasoline_input_name} characters; parser maximum is $gasoline_string_limit."
        echo "       Gasoline receives -I '$gasoline_input_name'."
        exit 1
    fi
    echo "Gasoline output-prefix length: ${#gasoline_output_prefix}/$gasoline_output_limit"
fi

if [ -n "$managed_adapter" ] && [ "${RUNTEST_RESOLVE_ONLY:-0}" = 1 ]; then
    echo "$managed_label resolve-only run name: $runname"
    exit 0
fi

# Campaign restart: a case is finished only when the LAST snapshot (step nSteps,
# which every writer / reverse-converter emits) is on disk -- a directory that
# merely exists may hold a killed run. A partial run is NOT resumable here
# (iCheckInterval == nSteps, so it left no checkpoint) and is started over.
# Off by default so a deliberate rerun still overwrites; testing.sh turns it on.
if [ "${RUNTEST_SKIP_COMPLETED:-0}" = 1 ]; then
    skip_nsteps=$(awk -F= '$1 ~ /^[ \t]*nSteps[ \t]*$/ {gsub(/[^0-9]/, "", $2); print $2; exit}' "datafiles/$ICfile.param" 2>/dev/null)
    if [ -n "$skip_nsteps" ] && [ -f "$(printf '%s/%s.%05d' "$runname" "$runname" "$skip_nsteps")" ]; then
        echo "Run $runname already finished at step $skip_nsteps. Skipping."
        exit 0
    fi
fi

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
    if [ "$3" -eq 2 ]; then
    if [ -f "${glassfile}_IC" -o -f "datafiles/$ICfile.00000" ]; then
        echo "File ${glassfile}_IC already exists. Skipping creatin.."
    else
	# Python (numba) glass relaxation - glassgen.cli writes ${glassfile}_IC
	# directly; target density is read from the .param.
	generated_density_args=()
	if [ -f "datafiles/$prefile.density.npz" ]; then
	    generated_density_args=(--table "datafiles/$prefile.density.npz")
	fi
	glass_boundary_args=(--param "datafiles/$prefile.param")
	if [ "$GLASSBOUNDARY" = "open" ]; then
	    glass_boundary_args=(--open)
	fi
	force_args=()
	if [ -f "datafiles/$prefile.force.npz" ]; then
	    FORCETABLE="datafiles/$prefile.force.npz"
	fi
	if [ "$GLASS_TARGET_SHAPE" = 1 ]; then
	    # The setup must take its pressure from the glass density, or a shaped glass starts with a pressure error.
	    case "$testsim" in
	        kh) ;;
	        *) echo "Error: GLASS_TARGET_SHAPE=1 is only set up for kh (a 1D contact whose setup takes P from the glass density), not $testsim."; exit 1 ;;
	    esac
	fi
	if [ "$GLASS_FORCEMATCH" = 1 ]; then
	    if [ "$GLASS_START_P" = pseudo ]; then
	        echo "Error: GLASS_FORCEMATCH=1 with GLASS_START_P=pseudo: the force-matched glass is relaxed under the real pressure; use GLASS_START_P=uniform."
	        exit 1
	    fi
	    if [ -z "$FORCETABLE" ] || [ ! -f "$FORCETABLE" ]; then
	        echo "Error: GLASS_FORCEMATCH=1 but $testsim has no force table (FORCETABLE='$FORCETABLE'). Setups that support it write it with their pre-IC: delete datafiles/$prefile.00000 to regenerate."
	        exit 1
	    fi
	    force_args=(--force-table "$FORCETABLE")
	fi
	relaxglass=(env PYTHONPATH="${glassgenpydir}" python -m glassgen.cli -o "$glassfile" -I datafiles/$prefile.00000 -s $1 $GLASSKNOBS "${glass_boundary_args[@]}" $GLASSMODE "${generated_density_args[@]}" "${force_args[@]}")
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
    # Pressure-bump check and the start pressure (relax.config GLASS_START_P / GLASS_PBUMP_WARN).
    pbump_warn=$GLASS_PBUMP_WARN
    [ "$GLASS_TARGET_SHAPE" = 1 ] && pbump_warn=$GLASS_SHAPE_ERRP
    if ! python ${analysisdir}/IC_glasspressure.py "datafiles/$ICfile.00000" "${glassfile}_IC" "$GLASS_START_P" "$pbump_warn" "$GLASS_FORCEMATCH"; then
    mv "datafiles/$ICfile.00000" "datafiles/$ICfile.00000.failed"
    echo "Error: IC_glasspressure.py failed; the IC was moved to datafiles/$ICfile.00000.failed."
    exit 1
    fi
    
    fi
    fi
##Glass patch: two uniform glasses relaxed apart, then pasted
    if [ "$3" -eq 4 ]; then
    if [ -f "${glassfile}_IC" -o -f "datafiles/$ICfile.00000" ]; then
        echo "File ${glassfile}_IC already exists. Skipping creation."
    else
	env PYTHONPATH="${glassgenpydir}" python ${analysisdir}/IC_glasspatch.py pre "$prefile" "$glassfile" || exit 1
	for part in inner outer; do
	    env PYTHONPATH="${glassgenpydir}" python -m glassgen.cli -o "${glassfile}_$part" -I "${glassfile}_$part.pre" -s $1 $GLASSKNOBS --box $(<"${glassfile}_$part.box") || exit 1
	done
	env PYTHONPATH="${glassgenpydir}" python ${analysisdir}/IC_glasspatch.py paste "$prefile" "$glassfile" || exit 1
    fi
    if [ -f "datafiles/$ICfile.00000" ]; then
        echo "File $ICfile.00000 already exists. Skipping creation."
    else
    python ${analysisdir}/IC_createsetup.py $testsim $2 $vm "${glassfile}_IC" ${ICfile} $EXTRACONDIC || exit 1
    fi
    fi
    mkdir $runname
    if [ -n "$managed_adapter" ]; then
        selection_text="physics=$RUNPHYSICS
build_key=$managed_build_key"
        if [ -n "$managed_build_alias" ]; then
            selection_text="$selection_text
build_alias=$managed_build_alias"
        fi
        selection_text="$selection_text
binary=$managed_binary"
        selection_target="$runname/$managed_adapter-build-selection.txt"
        if [ -f "$selection_target" ]; then
            [ "$(<"$selection_target")" = "$selection_text" ] || {
                echo "Error: existing $managed_label build selection differs: $selection_target"; exit 1;
            }
        else
            (set -o noclobber; printf '%s\n' "$selection_text" >"$selection_target") || exit 1
        fi
        if [ -n "$managed_build_manifest" ]; then
            manifest_target="$runname/$managed_adapter-build-manifest.json"
            if [ -f "$manifest_target" ]; then
                cmp -s "$managed_build_manifest" "$manifest_target" || {
                    echo "Error: existing $managed_label manifest differs: $manifest_target"; exit 1;
                }
            else
                cp "$managed_build_manifest" "$manifest_target"
            fi
        fi
    fi
    if [ -f "datafiles/$ICfile.red" ]; then
        # Any cosmological setup (cosmo=1) writes a redshift-output schedule
        # beside its IC, but Gasoline looks for <achOutName>.red beside the
        # output prefix.  Preserve an existing identical schedule; refuse a
        # mismatch instead of overwriting user data.
        red_target="$runname/$runname.red"
        if [ -f "$red_target" ]; then
            if ! cmp -s "datafiles/$ICfile.red" "$red_target"; then
                echo "Error: existing redshift schedule differs: $red_target"
                exit 1
            fi
        else
            cp "datafiles/$ICfile.red" "$red_target"
        fi
    fi
    # The run description: what runtest.sh knows about THIS run, handed to a
    # code's adapter as one file. It carries the run, never the code -- an
    # adapter derives anything code-specific from the .param itself.
    #
    # extracondrun is the one value that is deliberately NOT code-neutral: it is
    # the runtime flag string (EXTRACONDALL + the [extra] positional + the
    # test's own EXTRACONDRUN) verbatim, in the grammar of whichever code the caller
    # meant. It travels HERE rather than in the environment because an adapter
    # runs in a CHILD bash process, which does not inherit an unexported shell
    # variable. Only an adapter whose binary speaks that grammar reads it
    # (gasoline, changa); the others ignore it.
    rundesc="$runname/run-description.txt"
    {
        printf 'code_index=%s\n'   "$4"
        printf 'code_slug=%s\n'    "${CODE_SLUG[$4]:-}"
        printf 'code_dir=%s\n'     "${CODE_DIR[$4]:-}"
        printf 'build_dir=%s\n'    "$dir"
        printf 'binary=%s\n'       "$coderun"
        printf 'runname=%s\n'      "$runname"
        printf 'icfile=%s\n'       "$ICfile"
        printf 'paramfile=%s\n'    "datafiles/$ICfile.param"
        printf 'physics=%s\n'      "$RUNPHYSICS"
        printf 'analysisdir=%s\n'  "$analysisdir"
        printf 'current_dir=%s\n'  "$current_dir"
        printf 'mpirun=%s\n'       "$mpirun"
        printf 'runlauncher=%s\n'  "${runlauncher:-}"
        printf 'ncores=%s\n'       "$ncores"
        printf 'nsmooth=%s\n'      "$1"
        printf 'nres=%s\n'         "$2"
        printf 'extracondrun=%s\n' "$EXTRACONDRUN"
    } >"$rundesc"

    echo "Code selection: $4"
    # Codes with a run adapter: runtest.sh hands over the run description and
    # takes back the run line. Everything code-specific -- IC conversion, module
    # environment, command-line grammar -- stays inside codes/<code>/adapter.sh.
    if [ -n "${CODE_RUN_ADAPTER[$4]:-}" ]; then
        if ! runline_text="$(bash "${CODE_RUN_ADAPTER[$4]}" prepare --run-desc "$rundesc")"; then
            echo "Error: ${CODE_SLUG[$4]} adapter failed to prepare the run."; exit 1
        fi
        mapfile -t runline <<<"$runline_text"
        [ "${#runline[@]}" -gt 0 ] || { echo "Error: ${CODE_SLUG[$4]} adapter produced no run line."; exit 1; }
    fi


    echo "${runline[@]}"
    "${runline[@]}"

    if [ -n "${CODE_RUN_ADAPTER[$4]:-}" ]; then
        bash "${CODE_RUN_ADAPTER[$4]}" collect --run-desc "$rundesc" ||
            { echo "Error: ${CODE_SLUG[$4]} adapter failed to collect the output."; exit 1; }
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
        # a..k, not a..h: a test may declare more than eight parameters
        # (starform does), and a reference blessed without them would carry the
        # analysis DEFAULTS instead of the values the IC was built with.
        letters=(a b c d e f g h i j k)
        for i in 0 1 2 3 4 5 6 7 8 9 10; do
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
