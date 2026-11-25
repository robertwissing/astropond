#!/usr/bin/env python3
"""Periodic star-formation box for Gasoline's STARFORM / SFBFIELD path.

A uniform periodic cube in a PHYSICAL unit system (1 code length = 1 pc,
1 code mass = 1000 Msol), which star formation requires: `dDeltaStarForm` is in
years, `dPhysDenMin` in amu/cc and `dTempMax` in K, and `bStarForm` asserts that
`dMsolUnit`/`dKpcUnit` were specified.  At rho = 1 the box is a 1 pc cube
holding 1000 Msol at n ~ 4e4 amu/cc with c_s ~ 0.34 km/s at 10 K -- the
molecular-cloud-core regime the recipe is written for.

Two modes share one IC:

  * the RATE box (coldfrac = 0): the whole box is eligible, every particle has
    the same formation probability, and the surviving gas fraction follows the
    closed recursion

        f_{k+1} = f_k (1 - p_k),   p_k = 1 - exp(-dCStar dt sqrt(4 pi rho0 f_k))

    because the star-formation density smooth is gas-only, so removing gas
    genuinely lowers rho.  IC_analysis_starform compares against that recursion.

  * the HANDOFF box (coldfrac > 0): a warm box with a sparse COLD subset.
    Eligibility is then set by temperature -- a per-particle IC quantity -- so
    the converted set is identical at any MPI decomposition, which the random
    draw is not (`starform.c` seeds with `srand(pkd->idSelf)`).  Drive it with
    dCStar dt / tdyn >> 1 (large `cstar`, `pstep` -> 1) so p -> 1 and exactly
    the cold particles convert.

`SFBFIELD` only acts when an ENTIRE gas particle becomes a star, so a partial
conversion never reaches the handoff.  That whole-particle conversion comes from
the `dInitStarMass` clamp, NOT from `dStarEff` -- which this suite sets to 0.5
and which is inert.  See `formation_probability` for the mechanism.

Temperature here is the PHYSICAL temperature the star-formation criterion sees,
i.e. `CoolCodeEnergyToTemperature` for neutral, metal-free gas -- not the tipsy
"temperature" column, which is u/dTuFac.  The two differ by the mean molecular
weight convention; see `u_from_temperature`.  Under a NOCOOLING build that
evaluation is replaced by a hard T = 10 K (`starform.c`), so any case that
exercises the temperature gate (coldfrac > 0, or T0 above `tempmax`) must be
built with a real cooling module -- runtest.sh's setup_starform picks RUNCOOLING
accordingly.

Relax this IC as a GLASS.  On an FCC lattice the default nSmooth = 64 sits in
the checkerboard-unstable band for h/dx, and the `divv` criterion of a resting
lattice is exactly zero anyway.
"""

import math

import numpy as np

import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip


def _kernel_module():
    """glassgen.kernels -- the COMPILED KERNELOPTIMIZED kernel, already in Python.

    The winding predictions are kernel integrals, so they must use the kernel
    the run was built with rather than a convenient stand-in.  glassgen is not
    on the path of a bare setupfiles import, so resolve it relative to here.
    """
    import os
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "icgenerator", "glassgen_python")
    root = os.path.normpath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from glassgen import kernels
    return kernels


def _opt_coeffs(nsmooth):
    return _kernel_module().opt_coeffs(nsmooth)


def _w_opt(ar2, norm, coeffs):
    return _kernel_module().w_opt(ar2, norm, coeffs)

# --------------------------------------------------------------------------- #
#  Gasoline's own physical constants (master.c) -- NOT the astronomical CODATA
#  values.  The temperature the star-formation criterion evaluates has to match
#  the run's to better than the gate margin, so reproduce the code's constants
#  exactly rather than something more accurate.
# --------------------------------------------------------------------------- #
MSOLG = 1.99e33                  # g
KBOLTZ = 1.38e-16                # erg/K
KPCCM = 3.085678e21              # cm
GCGS = 6.67e-8                   # cgs
MHYDR = 1.67e-24                 # g
SECONDSPERYEAR = 31557600.0

# cooling_metal.c: CL_Rgascode, and Y_H + Y_He for neutral metal-free gas
# (f_HI = f_HeI = 1, f_HeII = 0, Z = 0 -> Y_He = 0.236/4, Y_H = 1 - 4 Y_He).
CL_RGASCODE = 8.2494e7
Y_HE_NEUTRAL = 0.236 / 4.0
Y_H_NEUTRAL = 1.0 - 4.0 * Y_HE_NEUTRAL
Y_TOT_NEUTRAL = Y_H_NEUTRAL + Y_HE_NEUTRAL          # = 0.823

# The box's unit system. Kept as module constants because the analysis needs the
# same numbers and must not re-derive them from a different constant set.
DKPCUNIT = 0.001                 # 1 code length = 1 pc
DMSOLUNIT = 1000.0               # 1 code mass = 1000 Msol

# The mean molecular weight that makes gasoline's TWO temperature conventions
# agree, so the tipsy thermal column reads directly in Kelvin.
#
# The column is u/dTuFac with dTuFac = dGasConst/((gamma-1) mu), i.e. an ideal
# gas of mu proton masses per particle; the star-formation criterion instead
# asks the cooling module, which counts the ACTUAL particles of neutral
# metal-free gas -- Y_tot = Y_H + Y_He = 0.823 per proton mass, because a helium
# atom is four times heavier but still only one particle. With the suite's usual
# mu = 1 the two disagree by exactly that factor, so an IC asking for 200 K is
# read as 164 K by anything that treats the column as a temperature (which
# `pkdInitEnergy` does under bGasCooling). Setting mu to the physically correct
# value for this gas removes the discrepancy at the source rather than
# compensating for it downstream.
#
#   mu = (k_B/m_H) / ((gamma-1) Y_tot 1.5 CL_Rgascode) = 1.2171
#
# a pure number, independent of dKpcUnit/dMsolUnit. It is ~1/Y_tot = 1.2151; the
# 0.2 per cent is master.c and cooling_metal.c disagreeing about k_B/m_H
# (8.2635e7 vs 8.2494e7), which no parameter can reconcile.
# mu enters ONLY the u <-> column conversion and the (feedback-only) supernova
# blast-radius formulae, never the hydrodynamics: P = (gamma-1) rho u has no mu.

# Fixed test constants (not -a..-k parameters).
PHYSDENMIN_AMUCC = 1.0e3         # dPhysDenMin; rho0 = 1 sits a factor ~40 above
WARM_FACTOR = 2.0                # T_warm = WARM_FACTOR * dTempMax
SHEET_WIDTH_DX = 1.0             # tanh reversal width, in particle spacings
SLAB_SMOOTH_DX = 2.0             # case 10 slab interface width, in DENSE spacings
SHEET_X = -0.25                  # the reversal the cold subset is placed on
MINGAS_FRAC = 0.5                # dMinGasMass = MINGAS_FRAC * particle mass
DELTASF_MARGIN = 1.5             # dDeltaStarForm = MARGIN * dDelta (see below)
BOXLEN = 1.0


# --------------------------------------------------------------------------- #
#  CASES -- the test matrix, as data.
#
#  Following the suite's established idiom (shocktube, cosmowave, disstest,
#  mri): `-a` selects a numbered case and the physics of that case lives HERE,
#  not on the command line. Only the two dials a rate test actually wants to
#  sweep stay adjustable (`cstar`, `pstep`), and they default to "use the
#  case's own value".
#
#  This is the single source of truth for the matrix.
# --------------------------------------------------------------------------- #
CASE_DEFAULTS = dict(
    rho0=1.0,          # uniform density, code units
    T0=10.0,           # temperature of the eligible (cold) gas, K
    tempmax=1.0e4,     # dTempMax, the SF temperature gate, K
    B0=0.0,            # field amplitude, code units
    field=0,           # 0 uniform Bz, 1 tanh reversal, 2 force-free winding
    turns=0,           # field 2 winding number (0 = lock to the donor lattice)
    vmode=0,           # 0 static (DIVVOFF compiled in), 1 live divv criterion
    coldfrac=0.0,      # 0 = uniform T0; else the eligible fraction
    clump=1,           # donors per lattice SITE (1 = one each, disjoint kernels)
    ampl=0.1,          # vx amplitude when vmode = 1
    cstar=0.05,        # dCStar
    pstep=0.01,        # target per-step formation probability -> sets dDelta
    nstep=0,           # nSteps (0 = derive from pstep)
    metals=0.0,        # metallicity mass fraction
    checkevery=0,      # force iCheckInterval (0 = the shared default)
    cooling=None,      # force a cooling module; None = derive from the verdict
    # --- the multi-rung dials (case 10). 0 = off.
    rholo=0.0,         # >0: density of the DIFFUSE slab; rho0 is then the dense one
    slabhalf=0.25,     # half-width in x of the dense slab (periodic box)
    denmin=0.0,        # >0: dPhysDenMin override in amu/cc (else PHYSDENMIN_AMUCC)
    sfrung=0,          # >0: ask dDeltaStarForm = dDelta/2^sfrung -> iStarFormRung
)

CASES = {
    # Rate cases -- the rate law and the three eligibility gates.
    1:  dict(name="R1 rate baseline"),
    2:  dict(name="R3 density gate", rho0=0.01),
    # The temperature gate is tested in the regime where metal-free gas has NO
    # coolants: Lyman-alpha needs 10.2 eV, so below ~8000 K it is suppressed by
    # exp(-118000/T) and there is nothing else at Z = 0. Both cases therefore
    # hold their temperature with cooling live (bGasCooling = 1, bUV = 0).
    # Do NOT put these above ~10^4 K: at this density t_cool is ~0.3 yr against
    # a 2.7e4 yr step, so the gas would crash-cool through the gate inside the
    # first step and the "too hot" case would form stars for the wrong reason.
    3:  dict(name="R4 temperature gate", T0=5.0e3, tempmax=1.0e3),
    # Needs cooling EXPLICITLY: the verdict (eligible) is the same either way,
    # but its job is to check the TForm the starlog records against the
    # temperature the IC asked for, and a NOCOOLING build reports 0.
    4:  dict(name="R4b temperature gate, eligible twin", T0=5.0e2,
             tempmax=1.0e3, cooling="metal"),
    # The convergence gate (starform.c: `if (p->divv >= 0.0) return 0;`) is
    # tested by its SIGN: with vx = -ampl sin(2 pi x) only the middle half
    # compresses, and 100.0 per cent of first-step events land there. The `=`
    # branch (a uniform velocity field gives divv identically zero) is not
    # tested: it is reachable only from the contrived configuration that tests it.
    5:  dict(name="R7 divv sinusoid", vmode=1, ampl=0.05),
    # Handoff cases -- the SFBFIELD magnetic handoff. H1 forms 125 stars, H2 16,
    # H3 exactly 0.
    # coldfrac is calibrated at Nres=32 to keep donors >~ 4h apart; the 3-D
    # subset (field 0) and the 2-D one on the sheet (field 1) need different
    # fractions for the same spacing, so re-check on a resolution change.
    6:  dict(name="H1 uniform field handoff", T0=10.0,
             tempmax=20.0, B0=1.0, field=0, coldfrac=0.005, cstar=3000.0,
             pstep=0.999999, nstep=8),
    7:  dict(name="H2 sharp reversal handoff", T0=10.0,
             tempmax=20.0, B0=1.0, field=1, coldfrac=0.0005, cstar=3000.0,
             pstep=0.999999, nstep=8),
    8:  dict(name="H3 no recipients", T0=10.0, tempmax=20.0,
             B0=1.0, field=0, coldfrac=1.0, cstar=3000.0, pstep=0.999999,
             nstep=8),
    # Deliberately ABSENT, and not to be re-added without a reason:
    #
    #   H3b (H3 with B0 = 0, expected to convert the whole box). H1 is already
    #   the positive control H3 needs -- same field, same p -> 1, differing
    #   only in whether recipients exist -- so H3's zero stands on its own.
    #   H3b's only unique coverage was `smoothfcn.c:7552`, the escape letting
    #   an UNMAGNETISED donor finish when no recipient accepts. That line is
    #   reachable only with SFBFIELD compiled AND B identically zero AND a
    #   whole neighbourhood converting at once -- a combination that exists in
    #   a test harness and not in a real magnetised run.
    #
    #   H5 (restart invariance). Worth checking, but it is a PROCEDURE -- run
    #   with checkpoints, restart mid-way, diff the trajectories -- not a
    #   configuration of the box, and it does not belong in the case matrix.
    #
    # H6 -- the case that can tell the two conservation laws apart.
    # `sfBFieldBlend` is a VOLUME-WEIGHTED AVERAGE of the physical field, so it
    # conserves sum V B identically and debits the magnetic energy that
    # averaging destroys as heat (both verified to machine precision against a
    # direct port of smoothfcn.c:7492-7512). Under a UNIFORM field neither
    # property can be seen: B_q == B_d makes the blend the identity map, Q_mix
    # is identically zero, and sum V B and sum (1/2) B^2 V are the same
    # statement. The winding fixes |B| and rotates its direction instead, so
    # the blend does real work with an exactly predictable result.
    9:  dict(name="H6 force-free winding handoff", T0=10.0, tempmax=20.0,
            B0=1.0, field=2, coldfrac=0.005, cstar=3000.0, pstep=0.999999,
            nstep=8),
    # A LONG-RELAXATION case is deliberately absent. Asking where the donated
    # volume ends up needs many more steps than the matrix's 8, but
    # nStep is not a property of the BOX -- it is a property of one measurement.
    # Raise nSteps in the IC .param and re-run the binary on the existing IC:
    # no case entry, no IC regeneration, and no stale-IC cache trap, since the
    # IC filename does not encode nstep. A `bStarForm = 0` copy of the same
    # .param gives the frozen pre-SF reference such a measurement needs.
    # M1 -- the first case in this suite with MORE THAN ONE RUNG, and the only
    # one that can exercise the handoff's interaction with multistepping.
    #
    # Why it is needed. `postDistSFBField` rescales the recipient's stored
    # variable in anticipation of the density drop that the NEXT density
    # evaluation will produce. That evaluation is rung-gated: STAGE 5 restores
    # ACTIVE to rungs >= iKickRung, so a recipient on a COARSER rung than the
    # star-formation substep receives the rescale and does not receive the
    # density change it compensates for, until its own rung comes round. Its
    # physical B = W rho^n is then wrong by fac for that whole interval, and
    # that wrong field enters the pressure and tension it exerts.
    #
    # Cases 1-9 cannot see this. They are uniform boxes: every particle lands on
    # rung 0 and iStarFormRung is 0, so star formation only ever fires at the
    # coarsest step, where msrActiveMaskRung makes EVERYTHING active and every
    # recipient is updated immediately.
    #
    # Reaching it needs two things at once, and neither alone is enough:
    #   * iStarFormRung >= 1, so SF fires at a substep where a coarser rung is
    #     inactive -- `sfrung`, which sets dDeltaStarForm = dDelta/2^sfrung;
    #   * recipients on a coarser rung than the donors. With uniform spacing
    #     dt ~ h/c_s, so coarse means large h (diffuse) or small c_s (cold).
    #     Donors must be cold AND dense, so the only way to have coarser
    #     recipients is a DENSITY contrast -- hence rholo.
    #
    # Geometry: a dense cold slab |x| < slabhalf in a diffuse cold periodic box,
    # relaxed by glassgen against dICdensprofile=3 (tanh in x). Both slabs are
    # at T0, so both pass the temperature gate; dPhysDenMin sits BETWEEN the two
    # densities, so only the dense slab forms stars and the diffuse slab is the
    # coarse-rung recipient population sitting against the interface.
    #
    # The contrast is 8 on purpose, not more. h ~ rho^-1/3, so 8 gives exactly
    # one rung of separation and 2h_dense ~= h_diffuse -- a donor at the
    # interface still reaches real diffuse neighbours. A larger contrast buys
    # more rungs but starves the donor's kernel of the very particles the test
    # is about, because a dense donor's kernel is SMALL.
    # cstar is the dial that sets dDelta here, NOT the star-formation rate:
    # dDelta = -ln(1-pstep) tdyn / cstar, and p = 1 - exp(-cstar dt / tdyn) is
    # unchanged when the two move together, so lowering cstar buys a bigger
    # dDelta at the same p -> 1. That is what makes the rungs split: at
    # cstar=3000 (dDelta 1.3e-3) every particle's Courant limit ~4e-3 is
    # comfortably coarser and the whole box sits on rung 0; cstar=650 puts
    # dDelta between the dense and diffuse limits. MEASURED at 32^3:
    #   cstar 3000 -> (32768, 0, ...)          one rung, useless
    #   cstar  650 -> (22720, 10048, 0, ...)   TWO RUNGS
    #   cstar   30 -> (0,0,0,0,0, 32768, ...)  overshoots to rung 5
    10: dict(name="M1 multi-rung handoff", T0=10.0, tempmax=20.0,
             B0=1.0, field=0, coldfrac=0.02, cstar=650.0, pstep=0.999999,
             nstep=8, rho0=1.0, rholo=0.125, slabhalf=0.25,
             denmin=1.4e4, sfrung=2),
    # H7 -- the MULTI-DONOR case, and the only one in which a recipient is
    # blended into more than once.
    #
    # Cases 6-10 all place donors on a lattice >~ 4 h apart so their kernels are
    # disjoint. Measured on case 6's own glass: 125 donors, 7708 recipients,
    # donor multiplicity exactly 1 for every one of them. `sfBFieldBlend` is
    # therefore only ever ENTERED ONCE per recipient anywhere in this matrix,
    # and its repeated entry -- where it re-reads a field it has already
    # re-encoded at a different density -- has never been exercised.
    #
    # `clump` puts several donors on ONE site. The site count falls by the same
    # factor, so the groups end up FURTHER apart than before (8.6 h at clump=4)
    # and each stays an independent event, while a group's own members sit
    # ~1.1 h apart and share nearly all their recipients.
    #
    # The field is UNIFORM on purpose, and that is what makes this case need no
    # model at all. B_q == B_d makes the blend the identity map, so whatever the
    # density bookkeeping does, every recipient must end holding |B| = B0 and
    # the mixing heat must be IDENTICALLY ZERO -- for every MHD_B_VAR, with or
    # without CORRPARTITION. There is nothing to calibrate and no tolerance to
    # pick. Replayed through the production blend on case 6's glass at clump=4
    # (108 donors, 2795 recipients, multiplicity 1/2/3/4):
    #
    #   MHD_B_VAR=1   max |B|/B0 - 1 = 0.273   mean over recipients 0.026
    #   MHD_B_VAR=2   max 0.623                mean 0.056
    #   MHD_B_VAR=0   0 exactly -- its conversions carry no density, so this is
    #                 the immune control, not a second detector
    #
    # and the error is a monotone ladder in the donor count, zero at N=1:
    #   N=1  0        N=2  1.5e-3      N=3  9.2e-3      N=4  1.0e-1
    # (mean |B|/B0 - 1 per multiplicity class, MHD_B_VAR=1). The N=1 recipients
    # in the SAME run are the internal control, so this needs no twin. A blend
    # that keeps one density convention gives 1e-16 everywhere and heat 1e-35.
    #
    # The floor it has to beat is the volume model's own bias, ~2e-3
    # (+2234 ppm): the signal is two orders above it.
    11: dict(name="H7 clumped donors, multi-donor handoff", T0=10.0,
             tempmax=20.0, B0=1.0, field=0, coldfrac=0.005, clump=4,
             cstar=3000.0, pstep=0.999999, nstep=8),
}


def case_spec(case):
    """The full parameter set for a case index, defaults filled in."""
    key = int(float(case))
    if key not in CASES:
        raise ValueError("unknown starform case %r; choose from %s"
                         % (case, sorted(CASES)))
    spec = dict(CASE_DEFAULTS)
    spec.update(CASES[key])
    spec["case"] = key
    return spec


def case_name(case):
    return case_spec(case)["name"]


# The evolved magnetic variable W = B rho^-n (MHDVariable.h). The suite default
# is B/sqrt(rho) and is passed as NO define, so the default arm keeps the same
# build key every other test in the suite uses.
#
# These are ARMS, not physics of the box, which is why they are command-line
# knobs rather than case-table entries -- the same reasoning that keeps `cstar`
# and `pstep` on the command line. A sweep is a sweep.
BVARS = {
    "sqrtrho": (None, "", 0.5),   # W = B/sqrt(rho), the default (MHD_B_VAR=1)
    "b":       (0, "bvB", 0.0),   # W = B
    "rho":     (2, "bvRho", 1.0),  # W = B/rho
}


def bvar_spec(bvar):
    """(define or None, run-name tag, the power n in W = B rho^-n)."""
    key = str(bvar).strip().lower()
    if key in ("", "0", "none", "default"):
        key = "sqrtrho"
    if key not in BVARS:
        raise ValueError("unknown bvar %r; choose from %s"
                         % (bvar, sorted(BVARS)))
    return BVARS[key]


def handoff_ratios(n, x):
    """(flux, energy) delivered by the handoff, relative to what it owes.

    The blend hands over sum V B exactly, over a MODEL volume V + dVolSF.
    postDistSFBField then re-expresses that field at the actual SPH volume V by
    the energy-preserving factor sqrt(1 + x), x = dVolSF/V, storing it in the
    EVOLVED variable at the old density. At the next density evaluation the
    recipient's volume really has grown to V(1 + x), so the physical field
    picks up a further (1 + x)^-n, and

        flux delivered / owed   = (1 + x)^(1/2 - n)
        energy delivered / owed = (1 + x)^(1 - 2n)

    Both are 1 for n = 1/2 and ONLY for n = 1/2 -- which is the quantitative
    content of smoothfcn.c:7594's "exact sum(V*B) conservation requires the SPH
    volume to match the donated-volume model". n = 0 over-delivers, n = 1
    under-delivers by the reciprocal factors.

    CAUTION -- these are PER-RECIPIENT factors on a recipient's whole field, and
    they look reassuringly close to 1 (x ~ 1/nSmooth). They are NOT the size of
    the error on the handed-over flux. The global statement is ADDITIVE, and is
    `flux_error_fraction` below: the error is a fraction of the DONATED flux, of
    order (1/2 - n), i.e. tens of per cent -- not of order x.
    """
    n = float(n)
    x = np.asarray(x, dtype=float)
    return (1.0 + x) ** (0.5 - n), (1.0 + x) ** (1.0 - 2.0 * n)


def flux_error_fraction(n, what):
    """The handoff's flux error as a fraction of the DONATED flux: (1/2 - n) W.

    This is the number that matters, and it is not `handoff_ratios`. Expanding
    a recipient's final `V B` to first order in x = dVolSF/V,

        d(sum V B) = (1/2 - n) * What(k) * V_d B_d

    summed over donors. The x's cancel: the error does NOT shrink with the
    neighbour count, it is a fixed fraction of what was handed over.

    Why `n = 1` is the worst case, and it is worth being explicit because the
    intuition runs the other way: at n = 1 the physical flux is
    `V B = (m/rho)(W rho) = m W`, carried by MASS alone, so no density change
    can touch it. Deleting a donor therefore removes exactly its own `m_d W_d`
    and the survivors' expansion compensates for NONE of it (measured ratio
    1.0000). At n = 0 the survivors' physical B is untouched while their volume
    grows, which recovers about two thirds for free. So `n = 1` starts with the
    largest hole, and the handoff -- a VOLUME-weighted average, while the
    donated volume carries no mass -- puts back the same ~(1 - What/2) of it at
    every n. Largest hole, same patch, worst result.
    """
    return (0.5 - float(n)) * float(what)


def restored_fraction(what):
    """Fraction of the donated flux the handoff puts back: 1 - What(k)/2.

    The blend alone restores `1 - What` (it moves each recipient toward B_d by
    the donated volume share); the sqrt(1 + x) rescale of postDistSFBField adds
    a further `What/2`. Nearly independent of n -- measured 0.606 / 0.620 /
    0.635 at n = 1 / 0.5 / 0, against 0.595 predicted.
    """
    return 1.0 - 0.5 * float(what)


def case_cooling(case):
    """Which COOLING MODULE the case needs compiled in: 'none' or 'metal'.

    `none` only when the NOCOOLING hard-wired T = 10 K reaches the same verdict
    the evaluated temperature would. A case that needs the evaluated temperature
    for any OTHER reason (reporting TForm, a temperature-selected donor subset)
    states it outright in its table entry.
    """
    c = case_spec(case)
    if c["cooling"] is not None:
        return c["cooling"]
    return ("metal" if (c["coldfrac"] > 0.0
                        or not (c["T0"] <= c["tempmax"]
                                and c["tempmax"] >= 10.0))
            else "none")


def case_build_requirements(case):
    """'<RUNPHYSICS capability tokens>|<extra runtime flags>' for runtest.sh.

    The build is a PROPERTY OF THE CASE, so it is derived here beside the
    physics rather than re-deduced in shell. It is emitted as capability tokens
    for the RUNPHYSICS set, not as raw -D defines: runtest.sh gates the tokens
    per code and each code's build config owns the translation, so the day a
    second code grows star formation nothing here has to change.

      * `starform=divvoff` for a static box -- starform.c rejects divv >= 0 and
        a uniform velocity field gives exactly zero, so the rate arm cannot form
        a single star without it. The case that TESTS the criterion (vmode = 1)
        asks for plain `starform` and keeps it live.
      * SFBFIELD and MHD_B_VAR stay BUILD-ONLY defines from the build flags file
        (CODE_EXTRA_BUILD_FLAGS_FILE): they are twins of one case, not
        capabilities a code either has or lacks. The SFBFIELD-off twin the
        handoff is measured against -- same stars, same deleted gas, one
        subtraction -- is a flags-file variant.
    """
    c = case_spec(case)
    tokens = ["starform" if c["vmode"] else "starform=divvoff",
              "cooling=%s" % case_cooling(case)]
    extra = []
    if c["checkevery"]:
        extra.append("-oc %d" % int(c["checkevery"]))
    return "%s|%s" % (" ".join(tokens), " ".join(extra))


def case_glass_tag(case):
    """The glass cache key for a case: one relaxed glass per DENSITY TARGET.

    Every uniform case shares a single glass, which is why `setup_starform`
    hard-coded "uniform". A case with `rholo > 0` relaxes against a slab
    (dICdensprofile=3) and must NOT be served the uniform one, so it keys on its
    own contrast and slab width.
    """
    c = case_spec(case)
    if float(c["rholo"]) <= 0.0:
        return "uniform"
    return "slab%gr%g" % (float(c["rho0"]) / float(c["rholo"]),
                          float(c["slabhalf"]))


def case_dials(case, cstar=0.0, pstep=0.0, clump=0):
    """The EFFECTIVE (cstar, pstep, clump) a command line asks for.

    0 is a SENTINEL meaning "use the case's own value", not a value. The run
    name has to apply exactly the rule create() applies, or a folder would
    advertise a number the run did not use, so both go through here.

    clump is donors per lattice site. Orthogonal to the field geometry, so any
    handoff case can be driven into repeated blends into one recipient, not
    just the one that happens to carry clump in its table entry.
    """
    spec = case_spec(case)
    if float(cstar) > 0.0:
        spec["cstar"] = float(cstar)
    if float(pstep) > 0.0:
        spec["pstep"] = float(pstep)
    if int(float(clump)) > 0:
        spec["clump"] = int(float(clump))
    return (float(spec["cstar"]), float(spec["pstep"]),
            max(1, int(float(spec["clump"]))))


def case_runname_tag():
    """Run-name suffix for the arm. Empty: the arm is a BUILD choice, and
    the build key already distinguishes the run directories."""
    return ""


def mean_molecular_weight(gamma=5.0 / 3.0):
    """dMeanMolWeight making the tipsy thermal column read directly in Kelvin."""
    return ((KBOLTZ / MHYDR)
            / ((gamma - 1.0) * Y_TOT_NEUTRAL * 1.5 * CL_RGASCODE))


def erg_per_gm_unit(dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """Gasoline's dErgPerGmUnit = G M / L for this unit system."""
    return GCGS * dmsolunit * MSOLG / (dkpcunit * KPCCM)


def gm_per_cc_unit(dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """Gasoline's dGmPerCcUnit = M / L^3."""
    return (dmsolunit * MSOLG) / (dkpcunit * KPCCM) ** 3


def sec_unit(dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """Gasoline's dSecUnit = 1/sqrt(G rho_unit)."""
    return math.sqrt(1.0 / (gm_per_cc_unit(dkpcunit, dmsolunit) * GCGS))


def u_from_temperature(temp, dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """Code specific internal energy giving PHYSICAL temperature `temp` (K).

    Inverts `CoolCodeEnergyToTemperature` for neutral metal-free gas:
    T = u dErgPerGmUnit / (Y_tot * 1.5 * CL_Rgascode).  This is the temperature
    the star-formation criterion compares with dTempMax; it is NOT the tipsy
    temperature column (u/dTuFac), which carries dMeanMolWeight instead.
    """
    return (np.asarray(temp, dtype=float)
            * (Y_TOT_NEUTRAL * 1.5 * CL_RGASCODE)
            / erg_per_gm_unit(dkpcunit, dmsolunit))


def temperature_from_u(u, dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """Inverse of `u_from_temperature` (the run's T for a code-unit u)."""
    return (np.asarray(u, dtype=float)
            * erg_per_gm_unit(dkpcunit, dmsolunit)
            / (Y_TOT_NEUTRAL * 1.5 * CL_RGASCODE))


def physdenmin_code(amu_per_cc=PHYSDENMIN_AMUCC,
                    dkpcunit=DKPCUNIT, dmsolunit=DMSOLUNIT):
    """dPhysDenMin converted to code density exactly as master.c does."""
    return amu_per_cc * MHYDR / gm_per_cc_unit(dkpcunit, dmsolunit)


def tdyn(rho):
    """The star-formation dynamical time, `starform.c`: 1/sqrt(4 pi rho)."""
    return 1.0 / np.sqrt(4.0 * np.pi * np.asarray(rho, dtype=float))


def formation_probability(cstar, dt, rho):
    """Per-step formation probability p = 1 - exp(-dCStar dt / tdyn(rho)).

    This is the per-particle BERNOULLI probability, not an expectation over a
    partial mass conversion: `starform.c` computes exactly this as `dMProb` and
    the acceptance test is `dMProb > rand()`, independent of how much mass the
    event then converts.

    HOW MUCH is a separate question, and it is NOT dStarEff. The suite forces
    whole-particle conversion through `dInitStarMass`, matching production:

        if (stfm->dInitStarMass > 0) dDeltaM = stfm->dInitStarMass;
        else                         dDeltaM = p->fMass*stfm->dStarEff;
        double fMassMax = p->fMass;   /* then clamped */

    `dInitStarMass` is set to 1000x the gas particle mass, so the first branch
    is taken, the clamp reduces it to the whole particle, and `dStarEff` is
    INERT -- it is never read. The runs carry the production default
    `dStarEff = 0.5`, which is why reading it as 1 gets the mechanism wrong.
    """
    return 1.0 - np.exp(-float(cstar) * float(dt) / tdyn(rho))


def sheet_by(x, b0, width):
    """Periodic double tanh reversal: By = -b0 for |x| < 0.25, +b0 outside.

    Divergence-free by construction (By depends only on x) and periodic on
    [-0.5, 0.5] to O(exp(-0.5/width)).  A single smooth sine would make
    |B_q - B_d| between neighbours O(h dB/dx) and the released mixing heat
    unmeasurably small; this is sharp on the kernel scale instead.
    """
    x = np.asarray(x, dtype=float)
    return b0 * (1.0 - np.tanh((x + 0.25) / width) + np.tanh((x - 0.25) / width))


def winding_b(x, b0, turns, boxlen=BOXLEN):
    """Force-free rotating winding: B = b0 (0, cos kx, sin kx), k = 2 pi n / L.

    The field this test exists for.  Four exact properties, all of them load
    bearing:

      * div B = 0 identically -- Bx = 0 and By, Bz depend only on x.
      * periodic for integer `turns`, so the glass needs no special boundary.
      * |B| = b0 EXACTLY everywhere.  Magnetic pressure is therefore uniform,
        and curl B = -k B makes the Lorentz force (curl B) x B vanish
        identically: the configuration is an exact force-free equilibrium and
        the box does not move on its own.
      * the direction rotates on the kernel scale while the magnitude does not.

    That last pair is what a uniform field (`field = 0`) cannot do and why this
    exists.  Under a uniform field sum V B and sum (1/2) B^2 V are the SAME
    statement, so no measurement on it can tell a flux-conserving handoff from
    an energy-conserving one, and `sfBFieldBlend` is additionally the identity
    map (B_q == B_d), so the blend is never exercised at all -- only the
    sqrt(1 + dV/V) rescale is.  Here <B^2>_W = b0^2 exactly while <B>_W is
    strictly shorter than B_d, so the blend does real, predictable work and the
    two conservation laws separate.

    See `kernel_ft` for the closed-form predictions this geometry buys.
    """
    x = np.asarray(x, dtype=float)
    k = 2.0 * np.pi * float(turns) / float(boxlen)
    zero = np.zeros_like(x)
    return zero, b0 * np.cos(k * x), b0 * np.sin(k * x)


def winding_turns(ncold, turns=0, field=2):
    """Winding number, defaulting to the donor sub-lattice size `n`.

    `turns = 0` means "lock the pitch to the donor lattice".  The donors sit at
    x_i = (i + 1/2)/n - 1/2, so with `turns = n` every donor's phase is
    k x_i = 2 pi (i + 1/2) - pi n = pi (1 - n)  (mod 2 pi) -- the SAME for all
    of them, whatever n.  Every donor then carries an identical B_d (along
    +/- y), which buys two things:

      * one prediction covers the whole ensemble rather than one per donor, and
      * the donated flux sum_d V_d B_d adds COHERENTLY instead of cancelling.
        The box's own sum V B is ~0 by symmetry here, but its CHANGE across the
        SF step is that coherent sum -- first order, sign-definite, and
        measurable globally.  A pitch incommensurate with the donor lattice
        throws that away.
    """
    if int(turns) > 0:
        return int(turns)
    if not field == 2:
        return 0
    return _lattice_n(ncold, field)


def kernel_ft(k, h, nsmooth=64):
    """What(k) = <cos(k.dr)>_W over the 2h ball: the normalised kernel FT.

    The single number every winding prediction is built from.  For a uniform
    glass the SFBFIELD weights rs/rstot are the kernel evaluated at r/h, so the
    kernel-weighted neighbour mean of the winding field is

        <B>_W = What(k) B_d        (exactly -- the odd part integrates out)
        <B^2>_W = b0^2             (exactly -- |B| is constant)

    Hence, per donor:

      * the mixing heat sfBFieldBlend releases,
            Q_mix = V_d b0^2 (1 - What(k)) = 2 (1 - What) x (donor E_mag),
        since |B_q - B_d|^2 = 2 b0^2 (1 - cos k dx).
      * the deletion's own first-order flux deficit under MHD_B_VAR = B,
            dFlux = V_d (What(k) - 1) B_d,
        which is what the handoff has to make up and a uniform field sets to
        zero identically.

    Imported by IC_analysis_starform rather than re-derived there.
    """
    norm, coeffs, _ = _opt_coeffs(nsmooth)
    w_opt = _w_opt
    r = np.linspace(1.0e-12, 2.0 * h, 20001)
    w = w_opt((r / h) ** 2, norm, coeffs)
    kr = k * r
    # sin(kr)/(kr): the angular integral of cos(k.r) over a sphere of radius r.
    sinc = np.where(kr > 1.0e-12, np.sin(kr) / np.where(kr > 0, kr, 1.0), 1.0)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(w * r * r * sinc, r) / trapz(w * r * r, r))


def smoothing_length(npart, nsmooth=64, boxlen=BOXLEN):
    """h from the neighbour count on a uniform box: nSmooth = (4/3) pi (2h)^3 n.

    A starting estimate for the predictions above; the run writes the real one
    with `bDoSphhOutput`, which is what the analysis should prefer.
    """
    ndens = float(npart) / (float(boxlen) ** 3)
    return 0.5 * (3.0 * float(nsmooth) / (4.0 * np.pi * ndens)) ** (1.0 / 3.0)


def _nearest_unique(points, nodes):
    """Index of a distinct nearest `points` row for each `nodes` row.

    Used to pick a well-separated donor subset: the nodes are a coarse lattice,
    so the selected particles inherit its spacing whatever the glass looks like.
    Greedy and order-dependent by construction, but deterministic, which is all
    the cross-rank comparison needs.
    """
    points = np.asarray(points, dtype=float)
    taken = np.zeros(len(points), dtype=bool)
    out = []
    for node in np.asarray(nodes, dtype=float):
        d2 = np.sum((points - node) ** 2, axis=1)
        d2[taken] = np.inf
        i = int(np.argmin(d2))
        if not np.isfinite(d2[i]):
            break
        taken[i] = True
        out.append(i)
    return np.array(out, dtype=int)


def _lattice_n(ncold, field):
    """Donor sites per axis: sqrt for the 2-D sheet subset, cbrt otherwise."""
    if field == 1:
        return max(1, int(round(math.sqrt(ncold))))
    return max(1, int(round(ncold ** (1.0 / 3.0))))


def _sublattice_nodes(ncold, field, half=0.5 * BOXLEN):
    """Coarse lattice of donor sites: 3-D through the box, or 2-D on the sheet.

    `field` = 1 puts the cold subset ON the field reversal at x = SHEET_X, which
    is where |B_q - B_d| -- and so the predicted mixing heat -- is largest.
    `field` = 2 keeps the 3-D lattice: the winding has no special plane (every
    phase is equivalent by rotation), and `winding_turns` locks the pitch to
    this lattice so all donors share one phase anyway.
    """
    if field == 1:
        n = _lattice_n(ncold, field)
        edges = (np.arange(n) + 0.5) * (2.0 * half) / n - half
        Y, Z = np.meshgrid(edges, edges, indexing="ij")
        X = np.full(Y.size, SHEET_X)
        return np.column_stack([X, Y.ravel(), Z.ravel()])
    n = _lattice_n(ncold, field)
    edges = (np.arange(n) + 0.5) * (2.0 * half) / n - half
    X, Y, Z = np.meshgrid(edges, edges, edges, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


class setup_starform(object):
    # Uniform glass target metadata used by IC_createsetup/glassgen.
    dICdensRsmooth = 0.1
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0
    dICdensinner = 1.0
    dICdensouter = 1.0

    inflow = 0
    rhoit = 0
    npart = 0
    ngas = 0
    ndark = 0
    nstar = 0
    shape = 0
    ndim = 3
    time = 0
    cosmo = 0
    ns = 64
    periodic = 1
    grav = 0
    # Star formation needs a real unit system (master.c asserts dMsolUnit and
    # dKpcUnit were specified when bStarForm is set).
    dmsolunit = DMSOLUNIT
    dkpcunit = DKPCUNIT
    adi = 1
    gamma = 5.0 / 3.0
    molweight = mean_molecular_weight(gamma)

    rcylmin2 = 0
    rmin2 = 0
    rcylmax2 = 10 ** 22
    rmax2 = 10 ** 22
    rhozero = 1.0

    def __init__(self):
        # Instance-owned arrays: setup objects can safely be constructed more
        # than once in an interactive/test process.
        for name in ("mass", "x", "y", "z", "vx", "vy", "vz", "rho",
                     "u", "h", "P", "Bx", "By", "Bz", "Bpsi", "soft",
                     "tform", "metals", "pot", "totvol"):
            setattr(self, name, [])
        self.dxbound = self.dybound = self.dzbound = []

    def create(self, nx, distri=0, vm=0,
               entry="datafiles/starform32_preglass.00000",
               case=1, cstar=0.0, pstep=0.0, clump=0):
        # -a picks the case; -b/-c/-d optionally override the dials a
        # measurement wants to sweep. 0 means "use the case's value".
        spec = case_spec(case)
        spec["cstar"], spec["pstep"], spec["clump"] = case_dials(
            case, cstar, pstep, clump)
        self.case = spec["case"]
        self.case_name = spec["name"]
        # A case whose verdict depends on the EVALUATED temperature runs with
        # runtime cooling (adi 3 -> bGasCooling) rather than adiabatic; see the
        # B3 note in IC_createparamfile. Everything else stays adiabatic.
        self.adi = 3 if case_cooling(spec["case"]) == "metal" else 1
        rho0, T0, tempmax = spec["rho0"], spec["T0"], spec["tempmax"]
        B0, field, vmode = spec["B0"], spec["field"], spec["vmode"]
        turns = spec["turns"]
        coldfrac, ampl = spec["coldfrac"], spec["ampl"]
        clump = max(1, int(float(spec["clump"])))
        rholo = float(spec["rholo"])
        slabhalf = float(spec["slabhalf"])
        denmin = float(spec["denmin"])
        sfrung = int(spec["sfrung"])
        cstar, pstep, nstep = spec["cstar"], spec["pstep"], spec["nstep"]
        metals = spec["metals"]

        nx = int(nx)
        rho0 = float(rho0)
        T0 = float(T0)
        tempmax = float(tempmax)
        B0 = float(B0)
        field = int(float(field))
        turns = int(float(turns))
        vmode = int(float(vmode))
        coldfrac = float(coldfrac)
        ampl = float(ampl)
        cstar = float(cstar)
        pstep = float(pstep)
        nstep = int(float(nstep))
        metals = float(metals)
        if nx < 2:
            raise ValueError("nx must be at least 2")
        if not math.isfinite(rho0) or rho0 <= 0.0:
            raise ValueError("rho0 must be finite and positive")
        if not math.isfinite(T0) or T0 <= 0.0:
            raise ValueError("T0 must be finite and positive")
        if not math.isfinite(tempmax) or tempmax <= 0.0:
            raise ValueError("tempmax (dTempMax) must be finite and positive")
        if not math.isfinite(B0) or B0 < 0.0:
            raise ValueError("B0 must be finite and non-negative")
        if field not in (0, 1, 2):
            raise ValueError("field must be 0 (uniform Bz), 1 (tanh reversal) "
                             "or 2 (force-free winding)")
        if turns < 0:
            raise ValueError("turns must be non-negative (0 = lock to the "
                             "donor lattice)")
        if vmode not in (0, 1):
            raise ValueError("vmode must be 0 (static) or 1 (sinusoidal vx)")
        if not (0.0 <= coldfrac <= 1.0):
            raise ValueError("coldfrac must lie in [0, 1]")
        if not math.isfinite(cstar) or cstar <= 0.0:
            raise ValueError("cstar (dCStar) must be finite and positive")
        if not (0.0 < pstep < 1.0):
            raise ValueError("pstep must lie strictly in (0, 1)")
        if nstep < 0:
            raise ValueError("nstep must be non-negative (0 = auto)")
        if not (0.0 <= metals < 1.0):
            raise ValueError("metals must be a mass fraction in [0, 1)")

        half = 0.5 * BOXLEN
        deltax = BOXLEN / nx
        self.rhozero = rho0
        distribute.setbound(self, -half, half, -half, half, -half, half)

        if distri == 0:
            distribute.setcloseddist(
                self, -half, half, -half, half, -half, half, deltax)
        elif distri == 1:
            distribute.setrandomdist(
                self, -half, half, -half, half, -half, half, deltax)
        else:
            tgdata, _td, _ts, _header, _time = tip.readtipsy(entry)
            self.x = tgdata[:, 1]
            self.y = tgdata[:, 2]
            self.z = tgdata[:, 3]
            self.rho = tgdata[:, 7]

        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.npart = len(self.x)
        self.ngas = self.npart
        if self.npart == 0:
            raise ValueError("particle distribution is empty")

        if rholo > 0.0:
            # A dense cold slab |x| < slabhalf in a diffuse cold periodic box.
            # glassgen reads the target straight out of the pre-IC .param
            # (dICdensprofile 1/2/3 = uniform/step/tanh), so this needs no
            # density table: profile 3 in direction 1 (|x|) IS the slab, and
            # dICdensRsmooth keeps the interface resolvable rather than a step
            # the relaxer has to fight.
            self.dICdensprofile = 3
            self.dICdensdir = 1
            self.dICdensR = slabhalf
            self.dICdensinner = rho0
            self.dICdensouter = rholo
            self.dICdensRsmooth = SLAB_SMOOTH_DX * (BOXLEN / nx)
            # Equal particle masses: the contrast lives in the SPACING, which is
            # what sets h and therefore the rung. Total mass is the integral of
            # the target, not rho0*V, or the box comes out at the wrong mean.
            vol_in = 2.0 * slabhalf * BOXLEN ** 2
            mtot = rho0 * vol_in + rholo * (BOXLEN ** 3 - vol_in)
        else:
            mtot = rho0 * BOXLEN ** 3
        mpart = mtot / self.npart
        self.mass = np.full(self.npart, mpart)
        if float(vm) != 0.0:
            if self.npart % 2:
                raise ValueError("variable-mass pairing needs an even particle count")
            rng = np.random.default_rng(1)
            for i in range(0, self.npart, 2):
                dm = 0.25 * self.mass[i] * rng.random()
                self.mass[i] += dm
                self.mass[i + 1] -= dm

        # Metallicity, set here the way IC_setup_blob and IC_setup_mti set
        # theirs. Filling the array per particle is what tells IC_createsetup
        # this setup OWNS its metals rather than getting the zeroed default.
        #
        # It does NOT change the star-formation rate: `tcool` is computed from
        # it, but `starform.c:421` takes `tform = tdyn` whenever
        # `T < dTempMax`, which every eligible particle satisfies, so the tcool
        # branch is dead for this test. What it IS good for: the metal-free
        # default is what keeps a cooling-enabled run thermally static (no
        # coolants below 10^4 K), so writing it explicitly turns that silent
        # precondition into a stated one; and a non-zero value turns the
        # metal-mass conservation check from 0 == 0 into a measurement.
        self.metals = np.full(self.npart, metals)
        self.rho = np.full(self.npart, rho0)
        # 200, the suite-wide seed: the run recomputes h at init, so this only
        # has to start the neighbour search in the right place.
        self.h = smth.getsmooth(self, 200)

        # ---- temperature: uniform, or a sparse cold subset in a warm box ----
        twarm = WARM_FACTOR * tempmax
        temp = np.full(self.npart, T0)
        cold = np.ones(self.npart, dtype=bool)
        nlat = 0
        if coldfrac > 0.0:
            temp[:] = twarm
            if coldfrac >= 1.0:
                cold[:] = True
            else:
                ncold = max(1, int(round(coldfrac * self.npart)))
                # `clump` donors per SITE instead of one. The site count falls
                # by the same factor, so the groups end up further apart than
                # before while their own members overlap. _nearest_unique is
                # greedy over rows, so repeating a node k times hands back its
                # k nearest distinct particles -- a tight clump on any glass.
                nsite = max(1, ncold // clump)
                nlat = _lattice_n(nsite, field)
                nodes = np.repeat(_sublattice_nodes(nsite, field, half),
                                  clump, axis=0)
                idx = _nearest_unique(
                    np.column_stack([self.x, self.y, self.z]), nodes)
                cold[:] = False
                cold[idx] = True
            temp[cold] = T0
        self.u = u_from_temperature(temp)

        # ---- velocities ----
        self.vy = np.zeros(self.npart)
        self.vz = np.zeros(self.npart)
        if vmode == 1:
            # Converging (divv < 0) for |x| < L/4, diverging outside: the sign of
            # the `divv >= 0` rejection in starform.c is then spatial, and every
            # formation event must land in the compressing half.
            self.vx = -ampl * np.sin(2.0 * np.pi * self.x / BOXLEN)
        else:
            self.vx = np.zeros(self.npart)

        # ---- magnetic field ----
        sheet_width = SHEET_WIDTH_DX * deltax
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        self.Bz = np.zeros(self.npart)
        if field == 1:
            self.By = sheet_by(self.x, B0, sheet_width)
        elif field == 2:
            # Locking the pitch to the donor lattice puts EVERY donor at the
            # same winding phase, so one prediction covers the ensemble and the
            # donated flux adds coherently -- see winding_turns.
            nturns = winding_turns(nlat ** 3 if nlat else self.npart, turns,
                                   field)
            self.Bx, self.By, self.Bz = winding_b(self.x, B0, nturns)
            self.starform_turns = nturns
        else:
            self.Bz = np.full(self.npart, B0)

        # ---- cadence ------------------------------------------------------ #
        # dDelta is chosen so the per-step probability at rho0 is `pstep`:
        #   p = 1 - exp(-dCStar dt / tdyn)  ->  dt = -ln(1-p) tdyn / dCStar.
        tdyn0 = float(tdyn(rho0))
        self.deltastep = -math.log(1.0 - pstep) * tdyn0 / cstar
        if nstep > 0:
            self.nsteps = nstep
        else:
            # Enough steps for the rate arm to deplete to ~40 per cent of the
            # gas; the handoff arm's p -> 1 collapses that to one step, so keep
            # a floor that still shows the post-event evolution.
            self.nsteps = int(min(400, max(
                8, math.ceil(math.log(0.4) / math.log(1.0 - pstep)))))
        self.freqout = 1

        # ---- star-formation parameters (IC_createparamfile reads these) ---- #
        self.sf_bStarForm = 1
        self.sf_bFeedBack = 0
        self.sf_dCStar = cstar
        # Whole-particle conversion -- the only kind that reaches the SFBFIELD
        # handoff (`starform.c`: `if (dDeltaM >= p->fMass)`) -- is requested
        # the way production runs request it: a FIXED star mass above the
        # particle mass, which `stfmFormStars` clamps down to the whole
        # particle. BESPOKE uses dInitStarMass 1e6 against a 2e3 gas mass.
        # dStarEff is then inert (line 682 takes the dInitStarMass branch) and
        # stays a strict fraction, which is what master.c asserts.
        #
        # CAVEAT: this guarantees a whole-particle conversion only while
        # dInitStarMass exceeds EVERY gas particle's mass. This box is
        # equal-mass so the margin below is ample; on a variable-mass IC a
        # heavier particle would convert a fraction and silently bypass the
        # handoff, which dStarEff = 1 could never do.
        self.sf_dStarEff = 0.5
        self.sf_dInitStarMass = 1.0e3 * mpart
        # Asserted > 0 when dInitStarMass == 0. dStarEff = 1 leaves exactly zero
        # gas mass behind, so any positive value deletes the spent donor.
        self.sf_dMinGasMass = MINGAS_FRAC * mpart
        self.sf_dMinMassFrac = 0.1
        # A two-density box needs the gate BETWEEN the slabs: the dense one
        # forms stars, the diffuse one is the cold, ineligible, coarse-rung
        # recipient population the test is built around.
        self.sf_dPhysDenMin = denmin if denmin > 0.0 else PHYSDENMIN_AMUCC
        self.sf_dTempMax = tempmax
        # master.c snaps dDeltaStarForm down to the largest dDelta*2^-k STRICTLY
        # below the request, incrementing iStarFormRung per halving. Asking for
        # strictly more than dDelta therefore leaves iStarFormRung = 0 and star
        # formation firing exactly once per base step with dt = dDelta -- which
        # is what the analytic recursion assumes. Equality would halve it.
        # sfrung > 0 asks for star formation every dDelta/2^sfrung, which
        # master.c turns into iStarFormRung = sfrung (it counts the halvings
        # from dDelta down to the request). That is what puts the SF substep on
        # a rung where coarser particles are INACTIVE; without it every case
        # here fires SF at the coarsest step, where msrActiveMaskRung makes
        # EVERYTHING active and the multistep interaction cannot appear.
        # Same year units as the default branch -- the .param carries years.
        if sfrung > 0:
            self.sf_dDeltaStarForm = (self.deltastep / (2.0 ** sfrung)
                                      * sec_unit() / SECONDSPERYEAR)
        else:
            self.sf_dDeltaStarForm = (DELTASF_MARGIN * self.deltastep
                                      * sec_unit() / SECONDSPERYEAR)
        self.sf_iRandomSeed = 1
        self.sf_bBHForm = 0
        self.sf_bDoSphhOutput = 1

        # ---- diagnostics carried for the analysis / the operator ---------- #
        self.starform_rho0 = rho0
        self.starform_T0 = T0
        self.starform_Twarm = twarm if coldfrac > 0.0 else T0
        self.starform_tempmax = tempmax
        self.starform_B0 = B0
        self.starform_field = field
        if field != 2:
            self.starform_turns = 0
        self.starform_vmode = vmode
        self.starform_coldfrac = coldfrac
        self.starform_ampl = ampl
        self.starform_cstar = cstar
        self.starform_pstep = pstep
        self.starform_metals = metals
        self.starform_ncold = int(np.count_nonzero(cold))
        self.starform_nlat = nlat
        self.starform_clump = clump
        self.starform_mpart = mpart
        self.starform_tdyn = tdyn0

        # The separation that matters is measured in the h the RUN will have,
        # i.e. the one nSmooth implies -- not the N=200 seed stored in the IC.
        hmean = float(np.mean(smth.getsmooth2(self.mass, self.rho, self.ns)))
        pcheck = float(formation_probability(cstar, self.deltastep, rho0))
        print("Star-formation box (STARFORM / SFBFIELD)")
        print(" case %d: %s" % (self.case, self.case_name))
        print(" nx=%d N=%d rho0=%g (%.4g amu/cc) m_part=%.4g" %
              (nx, self.npart, rho0,
               rho0 * gm_per_cc_unit() / MHYDR, mpart))
        print(" T_cold=%g K (u=%.5g) T_warm=%g K dTempMax=%g K n_cold=%d" %
              (T0, float(u_from_temperature(T0)), self.starform_Twarm,
               tempmax, self.starform_ncold))
        print(" dPhysDenMin=%g amu/cc (= %.5g code; rho0/threshold = %.3g)" %
              (PHYSDENMIN_AMUCC, physdenmin_code(),
               rho0 / physdenmin_code()))
        print(" metals=%g (mass fraction)  dMeanMolWeight=%.5f (column reads "
              "in K)" % (metals, self.molweight))
        print(" dCStar=%g tdyn=%.5g dDelta=%.5g p_step=%.5g nSteps=%d" %
              (cstar, tdyn0, self.deltastep, pcheck, self.nsteps))
        print(" dDeltaStarForm=%.5g yr (> dDelta=%.5g yr, so iStarFormRung=0)" %
              (self.sf_dDeltaStarForm,
               self.deltastep * sec_unit() / SECONDSPERYEAR))
        if field == 1:
            fielddesc = "tanh reversal at x=%g, width %g" % (SHEET_X,
                                                             sheet_width)
        elif field == 2:
            fielddesc = ("force-free winding, %d turns (lambda = %.4g = %.2f dx)"
                         % (self.starform_turns,
                            BOXLEN / self.starform_turns,
                            BOXLEN / self.starform_turns / deltax))
        else:
            fielddesc = "uniform Bz"
        print(" B0=%g field=%d (%s) vmode=%d ampl=%g" %
              (B0, field, fielddesc, vmode, ampl))
        if field == 2 and B0 > 0.0:
            # Everything the handoff prediction rests on, printed from the IC
            # so a run can be checked against it without re-deriving anything.
            k = 2.0 * math.pi * self.starform_turns / BOXLEN
            what = kernel_ft(k, hmean, self.ns)
            print(" winding k=%.4g k*h=%.4g What(k)=%.5f  1-What=%.5f" %
                  (k, k * hmean, what, 1.0 - what))
            print("   -> Q_mix/donor = %.5g = %.3f x the donor's own E_mag" %
                  ((mpart / rho0) * B0 * B0 * (1.0 - what), 2.0 * (1.0 - what)))
        if 0.0 < coldfrac < 1.0:
            sep = (2.0 * half) / max(1, self.starform_nlat)
            print(" donor site spacing %.4g = %.2f h (want >~ 4 h for "
                  "independent handoffs)" % (sep, sep / hmean))
            if self.starform_clump > 1:
                print("   clump %d: %d donors per site, so every recipient in a"
                      " site's kernel is blended into up to %d times"
                      % (self.starform_clump, self.starform_clump,
                         self.starform_clump))

    # Uniform target: the glass relaxer and IC_fixdens both read a scalar.
    def getrho(self, _x):
        return self.rhozero

    def getrhoi(self, _i):
        return self.rhozero

    def getrho_vec(self, x, _y=None, _z=None):
        return np.full(np.shape(x), self.rhozero)
