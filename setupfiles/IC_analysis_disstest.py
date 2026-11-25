#!/usr/bin/env python3
"""
Analysis for the isolated dissipation tests (`disstest`).

These grew inside `cosmowave` as cases 8--11 and were split out once it was
clear they share none of Berlok's mode machinery: every one is a frozen or
single-step MEASUREMENT rather than a wave, so there is no linear solution to
compare against and nothing for `case_params` to predict.

The suite answers one question -- is the dissipation behaving properly, with no
scale factor suppressing or amplifying it -- and it answers it in two different
ways, because the code contains two different kinds of thing:

  OPERATORS have a continuum limit, so they are tested by CONVERGENCE to an
  analytic solution (``delta ~ exp(-kappa k^2 t)``).  Cases 1 and 2.

  SWITCHES have no continuum limit -- they are numerical and vanish with
  resolution by design -- so instead, for a KNOWN field each has a determined
  value, and that value is what gets checked.  Case 3.

Every case runs in a STATIC (``-b 0``, ``bComove=0``) and an EXPANDING
(``-b 1``, ``bComove=1``) arm.  That pairing is the point: the static arm
measures the discrete operator with the scale factor frozen out of the problem
entirely, and the expanding arm is compared against it, so what is under test
is the scale-factor bookkeeping alone rather than the operator and its
a-factors tangled together.  It also makes a tautological configuration
obvious -- if an arm's two clocks coincide, `sensitivity` comes back at exactly
1 and that channel carries no a-dependence to test.

Cases (-a, must match IC_setup_disstest.create's `case`):
    0 null    uniform rho, u, B at v=0.  Every dissipative channel is built on
              a pair DIFFERENCE, so all must stay at roundoff while the
              homogeneous expansion laws hold.
    1 uwave   u = u0(1 + A cos kx).  Fixed-coefficient parabolic conduction;
              the mode's decay SHAPE carries the coefficient's a-dependence.
    2 bhelix  B = (0, A cos kx, A sin kx), a Beltrami field (no Lorentz force,
              div B = 0 exactly).  Fixed-coefficient resistivity.
    3 ladder  A velocity field switches on the FLOW-DRIVEN channels (AV, the
              default resistivity, p->diff) that are identically zero at v=0
              and unreachable by cases 0--2.  One fixed comoving template is
              stamped at a ladder of scale factors and run for a single tiny
              step, so d log(rate)/d log(a) is each channel's scale-factor
              exponent, read directly with no time integration in between.

Standalone like the other IC_analysis_* modules: reuses the framework's
Param/build_parser/params_from_args/find_files/loaddata.  Several inputs
overlay as separate series; for case 3 the inputs are the rungs of ONE ladder.

Usage:
    python IC_analysis_disstest.py <dir> -a 0 -b 1 --save out
    python IC_analysis_disstest.py <static> <expanding> -a 1 --save out
    python IC_analysis_disstest.py <a1> <a2> ... -a 3 --save out
"""

import os
import re
import sys

import numpy as np

from IC_analysis_framework import (
    Param, build_parser, params_from_args,
    reference_path_for, save_reference, find_reference, residuals,
)
from IC_analysis_general import (
    loaddata, find_files, try_aux, read_bfield,
    read_tufac, read_log_param,
    setup_rcparams, finish_figure_with_legend,
)

# Fixed IC constants -- these MUST match IC_setup_disstest.create: Hubble,
# wavenumber k = 2pi (wavelength = 2*dx = 1) and phase origin xdot0 = -dx.
H0 = 2.894405
KWAVE = 2.0*np.pi
X0 = -0.5

CASES = {0: 'null', 1: 'uwave', 2: 'bhelix', 3: 'ladder'}

_RUN_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CASE3_CORE = rf"case3c{_RUN_NUMBER}a{_RUN_NUMBER}b{_RUN_NUMBER}"
_CASE1_CORE = rf"case1c{_RUN_NUMBER}a{_RUN_NUMBER}b{_RUN_NUMBER}"

# runtest.sh setup_disstest() parameter order (runtest.sh:setup_disstest).
PARAM_SPEC = [
    Param("a", "case", 0, int,
          "0 null, 1 uwave (conduction), 2 bhelix (resistivity), "
          "3 ladder (flow-driven channels)", "case"),
    Param("b", "cosmo", 1, int,
          "1 expanding arm (bComove=1), 0 static arm (bComove=0)", "c"),
    Param("c", "amp", 1e-1, float,
          "field-perturbation amplitude (u or B, per case)", "a"),
    Param("d", "bfac", 1.0, float,
          "magnetic amplitude factor (0 = hydro only; case 2 needs nonzero)",
          "b"),
    Param("e", "variant", 0, int,
          "case 3 only: 0 longitudinal (compressive), 1 transverse (shear)",
          re.compile(_CASE3_CORE + rf"v({_RUN_NUMBER})")),
    Param("f", "a_init", 0.0, float,
          "expanding arm only: initial scale factor (0 = default 1/128)",
          "ai"),
    Param("g", "vamp", 0.0, float,
          "case 3 only: velocity amplitude (0 = use amp)"),
    Param("k", "strat", 0.0, float,
          "case 1 only: background density stratification amplitude "
          "(0 = uniform)",
          re.compile(_CASE1_CORE
                     + rf"k{_RUN_NUMBER}(?:ai{_RUN_NUMBER})?s({_RUN_NUMBER})")),
]


# --------------------------------------------------------------------------- #
#  Setup constants, mirroring IC_setup_disstest.create
# --------------------------------------------------------------------------- #

def case_params(case, cosmo=1, amp=1e-1, bfac=1.0, variant=0, a_init=0.0,
                vamp=0.0, strat=0.0):
    """The per-case setup constants computed exactly as IC_setup_disstest.

    The guards duplicate the setup's, deliberately: an analysis invoked with
    the wrong `-a`/`-b` would otherwise fit a model the run never used and
    report a residual for it.
    """
    p = dict(H0=H0, k=KWAVE, x0=X0, gamma=5./3.,
             omega_m=1.0, omega_lambda=0.0)
    p['case'] = int(case)
    p['cosmo'] = int(cosmo)
    p['variant'] = int(variant)
    p['amp'] = float(amp)
    p['bfac'] = float(bfac)
    p['a_init'] = float(a_init)
    p['strat'] = float(strat)
    p['vamp'] = float(vamp) if float(vamp) > 0. else float(amp)

    if p['case'] not in CASES:
        raise ValueError("disstest case must be one of %s"
                         % ", ".join("%d (%s)" % (k, v)
                                     for k, v in sorted(CASES.items())))
    if p['cosmo'] not in (0, 1):
        raise ValueError("disstest cosmo must be 0 (static) or 1 (expanding)")
    if p['amp'] <= 0.:
        raise ValueError("amp must be positive")
    if p['case'] == 2 and p['bfac'] == 0.:
        raise ValueError("disstest case 2 (bhelix) is a magnetic mode: bfac "
                         "sets the helical amplitude and must be nonzero")
    if p['variant'] not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    if p['variant'] != 0 and p['case'] != 3:
        raise ValueError("variant is only used by case 3 (ladder)")
    if p['a_init'] != 0. and not 0. < p['a_init'] <= 1.:
        raise ValueError("a_init must lie in (0, 1]")
    if p['a_init'] != 0. and p['cosmo'] == 0:
        raise ValueError("a_init is meaningless for the static arm "
                         "(bComove=0 has no scale factor)")
    if p['strat'] != 0. and p['case'] != 1:
        raise ValueError("strat stratifies the case 1 (uwave) background; the "
                         "other cases assume a uniform one")
    if not -1. < p['strat'] < 1.:
        raise ValueError("strat must lie in (-1, 1)")

    p['a_i'] = (1.0 if p['cosmo'] == 0
                else (p['a_init'] if p['a_init'] != 0. else 1./128.))
    p['rhozero'] = 1.0
    p['V_S'] = (p['H0']/p['k'])*np.pi
    p['uuzero'] = p['V_S']**2/(p['gamma']*(p['gamma'] - 1.))
    p['bzero'] = p['bfac']*p['V_S']*np.sqrt(p['rhozero'])
    # The stratified background, mirroring IC_setup_disstest.getrho.  A cosine
    # integrates to zero over the box, so the mean density -- and with it every
    # uniform-case normalization above -- is unchanged by `strat`.  It is IN
    # PHASE with the thermal mode so the pair-averaging residual and the signal
    # share a spatial dependence (see the setup for why).
    p['drho'] = p['rhozero']*p['strat']
    # Box length, for the static arm's dimensionless rate normalization.
    p['boxlen'] = 2.*abs(p['x0'])
    return p


def background_rho(x, p):
    """The analytic background density at x, uniform unless `strat` is set."""
    x = np.asarray(x, dtype=float)
    if p['drho'] == 0.:
        return np.full_like(x, p['rhozero'])
    return p['rhozero'] + p['drho']*np.cos(p['k']*(x - p['x0']))


def _hubble(a, p):
    """H(a) in code units, or exactly 0 in the static arm."""
    if p['cosmo'] == 0:
        return np.zeros_like(np.asarray(a, dtype=float))
    a = np.asarray(a, dtype=float)
    return p['H0']*np.sqrt(p['omega_m']/a**3 + p['omega_lambda'])


def _cosmic_time(a, p):
    """Cosmic time for the flat background the IC selected (code units)."""
    a = np.asarray(a, dtype=float)
    om, ol = p['omega_m'], p['omega_lambda']
    if ol == 0.:
        return 2.*a**1.5/(3.*p['H0']*np.sqrt(om))
    return (2./(3.*p['H0']*np.sqrt(ol))
            * np.arcsinh(np.sqrt(ol/om)*a**1.5))


def _clock_a(a, p):
    """The ``integral a dt = integral da/H(a)`` clock, on a dense grid.

    A frozen mode obeys d(delta)/dt = -Lambda f(a) delta, so it is exponential
    in ``integral f(a) dt``, not in t.  For a constant PHYSICAL coefficient
    f = a and this is the clock; for a constant COMOVING one f = 1 and cosmic
    time is.  The integral is done on a dense grid rather than on the snapshot
    grid because the residual it feeds is ~1e-4 and trapezoid error over ~100
    snapshots would be a visible part of it.
    """
    a = np.asarray(a, dtype=float)
    lo = max(min(float(a.min()), 1.e-8), 1.e-12)
    grid = np.linspace(lo, max(float(a.max()), lo*2.), 20001)
    hubble = p['H0']*np.sqrt(p['omega_m']/grid**3 + p['omega_lambda'])
    cumulative = np.concatenate(
        ([0.], np.cumsum(np.diff(grid)*0.5*(1./hubble[1:] + 1./hubble[:-1]))))
    return np.interp(a, grid, cumulative)


def snapshot_axis(time, p):
    """(a, t) for one snapshot's tipsy time.

    A comoving run stores the SCALE FACTOR in the tipsy time field; a static
    one stores cosmic time.  Everything downstream reads the pair, so the two
    arms share one code path instead of branching per diagnostic.
    """
    value = float(np.asarray(time).ravel()[0])
    if p['cosmo'] == 1:
        a = value if value > 0. else p['a_i']
        return a, float(_cosmic_time(a, p))
    return 1.0, value


def _reference_rate(a, u_mean, p):
    """The dimensionless-rate denominator: u per unit time, set by the arm.

    Expanding: the homogeneous ``3 H (gamma-1) u`` PdV source, the only
    legitimate rate on a uniform state.  Static: there is no expansion source
    at all (H = 0 exactly), so the box sound-crossing rate is used instead --
    it is the only timescale the static arm has, and it keeps the two arms'
    rate columns comparable in magnitude.
    """
    if p['cosmo'] == 1:
        return 3.*float(_hubble(a, p))*(p['gamma'] - 1.)*u_mean
    cs = (np.sqrt(p['gamma']*(p['gamma'] - 1.)*u_mean) if p['gamma'] > 1.
          else p['V_S'])
    return float(cs*u_mean/p['boxlen'])


def _complex_mode(weights, x, field, p):
    """Weighted particle fit Z = Ac - i*As for f = c0 + Re[Z exp(i k(x-x0))].

    Solving the 3-column weighted least-squares problem retains phase without
    interpolating to a grid and stays unbiased when a finite glass sampling
    does not make 1, sin and cos exactly orthogonal.
    """
    weights = np.asarray(weights, dtype=float)
    x = np.asarray(x, dtype=float)
    field = np.asarray(field, dtype=float)
    good = (np.isfinite(weights) & (weights > 0) & np.isfinite(x)
            & np.isfinite(field))
    if np.count_nonzero(good) < 3:
        return complex(np.nan, np.nan)
    phase = p['k']*(x[good] - p['x0'])
    design = np.column_stack((np.ones(phase.size), np.cos(phase),
                              np.sin(phase)))
    root_weight = np.sqrt(weights[good])
    coeff, _resid, rank, _sing = np.linalg.lstsq(
        design*root_weight[:, None], field[good]*root_weight, rcond=None)
    if rank < 3:
        return complex(np.nan, np.nan)
    return complex(coeff[1], -coeff[2])


def _cell(value):
    """Compact table text: exact zeros stay '0' rather than 0.00e+00."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return "0" if value == 0.0 else f"{value:.2e}"


def _headers(results):
    if len(results) == 1:
        return ["value"]            # the run is named once, in the legend
    return [lab if len(lab) <= 16 else lab[:7] + ".." + lab[-7:]
            for lab, _d in results]


def _draw_table(axis, rows, cells, results, title):
    axis.axis('off')
    table = axis.table(cellText=cells, rowLabels=list(rows),
                       colLabels=_headers(results), cellLoc='right',
                       rowLoc='left', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for column in range(len(_headers(results))):
        table[0, column].set_text_props(color=f"C{column % 10}",
                                        fontweight='bold')
    axis.set_title(title)


# --------------------------------------------------------------------------- #
#  Case 0: the homogeneous null test
# --------------------------------------------------------------------------- #

# Column layout of scan_null()'s table.
N_A, N_T, N_RHO, N_U, N_B = 0, 1, 2, 3, 4
N_RHO_SPR, N_U_SPR, N_B_SPR = 5, 6, 7
N_VMAX, N_AV, N_DIFF, N_PDV, N_DIVB = 8, 9, 10, 11, 12   # N_VMAX is a Mach no.
N_COLS = 13


def null_laws(a, p):
    """The homogeneous invariants case 0 measures, as ratios to their a_i value.

    These are the code's OWN cosmological source terms:

    ``rho``  comoving density is not expanded, so it must be constant;
    ``u``    ``propto a^{-3(gamma-1)}`` -- the local ``-3 H P/rho`` source
             (gasoline ``smoothfcn.c``, ``SphPressureTermsSym``), which replaced
             the pairwise Hubble-PdV term and has no other end-to-end test;
    ``B``    ``propto a^{-1/2}`` -- the ``-H B/2`` induction source acting on
             the stored field ``B_s = a^{3/2} B_phys``.

    In the STATIC arm every source is switched off with H, so all three laws
    are flat and the test becomes "nothing moves at all" -- which is the right
    null for that arm and is what makes it the operator reference.

    The ``B`` law is only clean while ``rho`` is: the ``BField`` aux is
    ``mhdBFromEvolved(B, rho)``, so a drift in the comoving density would move
    it too.  The case measures both, so that coupling is observable rather than
    hidden.
    """
    r = np.asarray(a, dtype=float)/p['a_i']
    if p['cosmo'] == 0:
        return {'rho': np.ones_like(r), 'u': np.ones_like(r),
                'B': np.ones_like(r)}
    return {'rho': np.ones_like(r),
            'u': r**(-3.*(p['gamma'] - 1.)),
            'B': r**(-0.5)}


def scan_null(input_arg, p):
    """Homogeneous-state table for one case-0 run, sorted by time.

    Columns (see the ``N_*`` constants): a, cosmic time, volume-weighted
    comoving density, mass-weighted specific internal energy, mean stored field
    magnitude, the three field spreads, the peculiar Mach number, and the AV /
    diffusion / PdV rates -- each normalized by ``_reference_rate`` so they are
    dimensionless -- plus ``h|div B|/|B|``.
    """
    files = find_files(input_arg)
    print(f"[scan-null] {len(files)} snapshot(s) in {input_arg}")
    tufac = read_tufac(input_arg)
    rows = []
    for fn in files:
        try:
            tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, h = loaddata(fn)
        except Exception as e:
            print(f"  skip {fn}: {e}", file=sys.stderr)
            continue
        a, t = snapshot_axis(time, p)
        m = np.asarray(tgdata[:, 0], dtype=float)
        v = np.asarray(tgdata[:, 4:7], dtype=float)
        rho = np.asarray(tgdata[:, 7], dtype=float)
        if np.any(rho <= 0):
            print(f"  skip {fn}: non-positive density", file=sys.stderr)
            continue
        volume = m/rho
        u = tufac*np.asarray(tgdata[:, 8], dtype=float)
        rho_mean = float(np.average(rho, weights=volume))
        u_mean = float(np.average(u, weights=m))
        rho_spread = float(np.max(rho) - np.min(rho))/rho_mean
        u_spread = (float(np.max(u) - np.min(u))/u_mean if u_mean > 0
                    else np.nan)
        # Tipsy stores the coordinate velocity xdot; the physical peculiar
        # speed is a*xdot expanding and xdot static, which `a` already carries
        # (it is exactly 1 in the static arm).  Normalize by the LOCAL sound
        # speed taken from the snapshot's own u, not by the family's V_S: V_S
        # is the a=1 scale while c_s propto a^{-3(gamma-1)/2} is 128x larger at
        # a_i, so dividing by V_S would overstate early motion by that factor.
        csound = (np.sqrt(p['gamma']*(p['gamma'] - 1.)*u_mean)
                  if p['gamma'] > 1. else p['V_S'])
        vmax = (float(np.max(np.abs(a*v)))/csound if csound > 0 else np.nan)

        B = read_bfield(fn, ngas)
        if B is not None and np.any(B != 0.):
            B_mean_vec = np.average(B, axis=0, weights=volume)
            B_mean = float(np.linalg.norm(B_mean_vec))
            B_spread = (float(np.max(np.linalg.norm(B - B_mean_vec, axis=1)))
                        / B_mean) if B_mean > 0 else np.nan
        else:
            B_mean = B_spread = np.nan

        rate = _reference_rate(a, u_mean, p)

        def _peak(label):
            arr = try_aux(fn, label, ngas)
            if arr is None or not rate > 0:
                return np.nan
            return float(np.max(np.abs(np.asarray(arr, dtype=float))))/rate

        av = _peak("uDotAV")
        diff = _peak("uDotDiff")
        pdv_arr = try_aux(fn, "uDotPdV", ngas)
        if pdv_arr is not None and rate > 0:
            # Sign convention: the expanding arm's local source cools, so
            # uDotPdV = -rate there; the static arm has no source and it is 0.
            pdv = float(np.average(np.asarray(pdv_arr, dtype=float),
                                   weights=m))/rate
        else:
            pdv = np.nan

        divB = try_aux(fn, "DivB", ngas)
        if divB is not None and np.isfinite(B_mean) and B_mean > 0:
            divb = float(np.max(np.abs(np.asarray(h, dtype=float)
                                       * np.asarray(divB, dtype=float))))/B_mean
        else:
            divb = np.nan

        rows.append((a, t, rho_mean, u_mean, B_mean,
                     rho_spread, u_spread, B_spread,
                     vmax, av, diff, pdv, divb))
        axis_text = (f"a={a:.5g}" if p['cosmo'] == 1 else f"t={t:.5g}")
        message = (f"  {axis_text}  rho={rho_mean:.10g}  u={u_mean:.10g}"
                   f"  spread(rho,u)=({rho_spread:.2e},{u_spread:.2e})"
                   f"  Mach={vmax:.2e}")
        if np.isfinite(B_mean):
            message += f"  |B|={B_mean:.10g}  spread={B_spread:.2e}"
        if np.isfinite(av) or np.isfinite(diff):
            message += f"  AV/ref={av:.2e}  diff/ref={diff:.2e}"
        print(message)

    if not rows:
        print(f"disstest: no usable snapshots in {input_arg}", file=sys.stderr)
        return None
    out = np.array(rows, dtype=float)
    return out[np.argsort(out[:, N_T])]


def null_reference_table(d):
    """CI regression table for a case-0 run: the raw homogeneous series."""
    return {"x": d[:, N_T], "rho": d[:, N_RHO], "u": d[:, N_U],
            "B": d[:, N_B], "rho_spread": d[:, N_RHO_SPR],
            "u_spread": d[:, N_U_SPR], "mach": d[:, N_VMAX],
            "uDotAV": d[:, N_AV], "uDotDiff": d[:, N_DIFF]}


def null_residuals(d, p):
    """Named dimensionless residuals for one case-0 run; all must be ~0.

    The invariants are compared as ratios to their own FIRST-SNAPSHOT value, so
    the test is on the code's expansion law rather than on the IC
    normalization.
    """
    law = null_laws(d[:, N_A], p)
    out = {}
    for key, col in (('rho', N_RHO), ('u', N_U), ('B', N_B)):
        series = d[:, col]
        if not np.all(np.isfinite(series)) or not series[0] > 0:
            continue
        # d is sorted by time, so row 0 is the earliest snapshot; renormalizing
        # the law to it rather than to the analytic a_i value removes any
        # offset between the IC file and the first dump.
        predicted = law[key]*(series[0]/law[key][0])
        out[f"law:{key}"] = float(np.max(np.abs(series/predicted - 1.)))
    for key, col in (('peculiar:mach', N_VMAX), ('rate:uDotAV', N_AV),
                     ('rate:uDotDiff', N_DIFF)):
        series = d[:, col]
        if np.all(np.isfinite(series)):
            out[key] = float(np.max(np.abs(series)))
    # The state's non-uniformity and its div-B are LOAD properties, not errors:
    # a relaxed glass has a few per cent of SPH density scatter and a finite
    # h|div B|/|B| for a perfectly uniform field, both of which a lattice does
    # not.  What the null test can require of ANY load is that they do not
    # DRIFT, so gate on the change from the first snapshot.
    for key, col in (('drift:spread:rho', N_RHO_SPR),
                     ('drift:spread:u', N_U_SPR),
                     ('drift:spread:B', N_B_SPR),
                     ('drift:divB', N_DIVB)):
        series = d[:, col]
        if np.all(np.isfinite(series)):
            out[key] = float(np.max(np.abs(series - series[0])))
    pdv = d[:, N_PDV]
    if np.all(np.isfinite(pdv)):
        if p['cosmo'] == 1:
            # The local source must be exactly -3 H (gamma-1) u.
            out['rate:uDotPdV+1'] = float(np.max(np.abs(pdv + 1.)))
        else:
            # No expansion source exists, so the rate itself must vanish.
            out['rate:uDotPdV'] = float(np.max(np.abs(pdv)))
    return out


# Display order for the residual table: grouped by what they test (expansion
# laws, source/dissipation rates, motion, then the drifts), not alphabetically,
# which interleaves them uselessly.
NULL_RESIDUAL_ORDER = (
    'law:rho', 'law:u', 'law:B',
    'rate:uDotPdV+1', 'rate:uDotPdV', 'rate:uDotAV', 'rate:uDotDiff',
    'peculiar:mach',
    'drift:spread:rho', 'drift:spread:u', 'drift:spread:B', 'drift:divB',
)


def plot_null(results, p, save=None):
    """Case-0 figure: compensated invariants, with the residual table beside it.

    Left panel: the law is DIVIDED OUT rather than overlaid, so what is plotted
    is rho, u*a^{3(gamma-1)} and B*a^{1/2}, each normalized to its own first
    snapshot.  A correct power law is flat at zero and the residual keeps its
    sign and its shape.  Plotting the raw series instead would waste four
    decades showing that a^-2 is a straight line on a log axis.

    Right panel: every entry `null_residuals` computes, for every run.  On a
    passing run the dissipative channels and the uniformity/div-B diagnostics
    are flat floors, so the numbers say everything a panel would, and the
    figure keeps the one thing a number cannot -- how the residual develops.
    """
    import matplotlib.pyplot as plt
    setup_rcparams()
    marker = {'rho': 'o', 'u': 's', 'B': '^'}
    colours = [f"C{i % 10}" for i in range(len(results))]
    fig, (ax, tax) = plt.subplots(
        1, 2, figsize=(6.6 + 2.0*len(results), 5.4),
        gridspec_kw={'width_ratios': [1.25, 1.0]})

    expanding = p['cosmo'] == 1
    seen_keys = []
    drawn = False
    for (lab, d), colour in zip(results, colours):
        xaxis = (np.log(d[:, N_A]/p['a_i']) if expanding
                 else d[:, N_T] - d[0, N_T])
        law = null_laws(d[:, N_A], p)
        for key, col in (('rho', N_RHO), ('u', N_U), ('B', N_B)):
            series = d[:, col]
            if not np.all(np.isfinite(series)) or not series[0] > 0:
                continue
            predicted = law[key]*(series[0]/law[key][0])
            ax.plot(xaxis, series/predicted - 1., marker=marker[key],
                    ls='none', mfc='none', mew=0.9, color=colour)
            if key not in seen_keys:
                seen_keys.append(key)
            drawn = True
        ax.plot([], [], marker='s', ls='none', color=colour, label=lab)
    for key in seen_keys:
        ax.plot([], [], marker=marker[key], ls='none', mfc='none', mew=0.9,
                color='0.35', label=key)

    ax.axhline(0.0, color='k', lw=1.0)
    ax.set_xlabel(r"$\ln(a/a_i)$" if expanding else "elapsed time")
    if expanding:
        ax.set_ylabel(r"$\rho,\ u\,a^{3(\gamma-1)},\ B\,a^{1/2}$"
                      "\n(/ first snapshot $-\\,1$)")
    else:
        ax.set_ylabel(r"$\rho,\ u,\ B$" "\n(/ first snapshot $-\\,1$)")
    ax.set_title("compensated invariants" if expanding
                 else "invariants (static arm: nothing expands)")
    if not drawn:
        ax.text(0.5, 0.5, "no usable invariants", ha='center', va='center',
                transform=ax.transAxes)

    residuals_by_run = [null_residuals(d, p) for _lab, d in results]
    rows = [key for key in NULL_RESIDUAL_ORDER
            if any(key in r for r in residuals_by_run)]
    cells = [[_cell(r.get(key)) for r in residuals_by_run] for key in rows]
    _draw_table(tax, rows, cells, results, "residuals")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    finish_figure_with_legend(
        fig, ax, save=(f"{save}_null.png" if save else None),
        footnote=("colour = run, marker = quantity; flat at 0 means the law "
                  "holds; all residuals are dimensionless, 0 = exact"))


def analyze_null(args, params, p, labels, parser):
    """Case-0 driver: a homogeneous state has no mode to project onto, so this
    path uses none of the Fourier machinery the other cases share."""
    results = []
    for inp, lab in zip(args.inputs, labels):
        d = scan_null(inp, p)
        if d is not None:
            results.append((lab, d))
    if not results:
        print("disstest: no data to analyze; exiting.", file=sys.stderr)
        sys.exit(1)

    if args.save_reference:
        save_reference(null_reference_table(results[0][1]),
                       reference_path_for(args.inputs[0]), params, "disstest")
        return

    plot_null(results, p, save=args.save)

    worst = 0.0
    endpoint_failed = False
    for lab, d in results:
        if p['cosmo'] == 1:
            afinal = float(d[-1, N_A])
            enderr = abs(afinal - args.final_a)
            print(f"disstest[endpoint] {lab}: final a={afinal:.8g}, "
                  f"target={args.final_a:.8g}, |delta|={enderr:.4g}")
            if enderr > args.end_tol:
                endpoint_failed = True
        for key, value in sorted(null_residuals(d, p).items()):
            worst = max(worst, value)
            print(f"disstest[null] {lab}: {key:<18s} {value:.4g}")
    print(f"disstest[null] worst dimensionless residual: {worst:.4g}")

    reg_worst = None
    for i, inp in enumerate(args.inputs):
        ref_table, _path = find_reference(inp, args.reference)
        if ref_table is None:
            continue
        table = null_reference_table(results[i][1])
        for key in ("rho", "u", "B", "rho_spread", "u_spread", "mach",
                    "uDotAV", "uDotDiff"):
            res = residuals(table, ref_table, key)
            if res is None:
                continue
            mx, rms = res
            reg_worst = rms if reg_worst is None else max(reg_worst, rms)
            print(f"disstest[regression] {key:<10s} vs baseline: "
                  f"max|d|={mx:.4g}  rms={rms:.4g}")

    if args.reg_tol is not None and reg_worst is not None \
            and reg_worst > args.reg_tol:
        print(f"disstest: REGRESSION FAILED -- worst rms {reg_worst:.4g} > tol "
              f"{args.reg_tol:.4g}", file=sys.stderr)
        sys.exit(2)
    if args.theory_tol is not None and worst > args.theory_tol:
        print(f"disstest: NULL FAILED -- worst residual {worst:.4g} > tol "
              f"{args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(3)
    if endpoint_failed and not args.allow_incomplete:
        print("disstest: ENDPOINT FAILED -- use --allow-incomplete only for "
              "deliberate partial-run diagnostics", file=sys.stderr)
        sys.exit(4)


# --------------------------------------------------------------------------- #
#  Cases 1 and 2: the fixed-coefficient OPERATOR tests
# --------------------------------------------------------------------------- #

# Column layout of scan_uwave()/scan_bhelix()'s shared table.
T_A, T_TIME, T_CLOCKA, T_UMEAN, T_AMP, T_RE, T_IM = 0, 1, 2, 3, 4, 5, 6
T_COLS = 7

# The scale-factor clock each fixed coefficient is INTENDED to run on.  The two
# differ by exactly one power of a, which is the whole point: that is the error
# class this suite exists to catch.
CHANNELS = {
    'physical': "constant PHYSICAL coefficient (kappa_code = kappa*a), "
                "clock = integral a dt",
    'comoving': "constant COMOVING coefficient, clock = cosmic time",
}


def clock_column(channel):
    """Which table column holds `channel`'s clock.

    In the STATIC arm the two columns are filled with the same numbers, so the
    two fits coincide and `sensitivity` comes back at exactly 1 -- the intended
    signal that the arm carries no a-dependence to discriminate.
    """
    if channel == 'physical':
        return T_CLOCKA
    if channel == 'comoving':
        return T_TIME
    raise ValueError(f"unknown channel {channel!r}")


def _fill_clocks(out, p):
    """Fill the two clock columns of a decay table sorted by scale factor."""
    if p['cosmo'] == 1:
        out[:, T_TIME] = _cosmic_time(out[:, T_A], p)
        out[:, T_CLOCKA] = _clock_a(out[:, T_A], p)
    else:
        # T_TIME is already the snapshot time; a static run has one clock.
        out[:, T_CLOCKA] = out[:, T_TIME]
    return out


def uwave_channel(input_arg):
    """Which fixed-coefficient conduction channel a run used, and its value.

    Read from the run's own `.log` rather than passed in, so the reference can
    never disagree with what was actually run.  Returns (channel, coefficient),
    or (None, nan) if neither channel is configured.
    """
    kappa = read_log_param(input_arg, "dThermalCondCoeff", 0.) or 0.
    linear = read_log_param(input_arg, "bThermalCondLinearCodeUnits", 0.) or 0.
    diff = read_log_param(input_arg, "dThermalDiffusionCoeff", 0.) or 0.
    constant = read_log_param(input_arg, "bConstantDiffusion", 0.) or 0.
    if kappa > 0. and linear:
        return 'physical', float(kappa)
    if diff > 0. and constant:
        return 'comoving', float(diff)
    return None, float('nan')


def bhelix_channel(input_arg):
    """The fixed-coefficient magnetic channel, read from the run's own log.

    Only the TURBULENT channel of MHDRESISTIVITY_ALT is coefficient-clean:
    `dMagDiffLin` carries havrg*ALPHA*vSig*invAFac and `dMagDiffShock` carries
    |mu|, both flow-dependent.  With bConstantDiffusion the turbulent kappa is
    a constant COMOVING coefficient (p->diff = 1, no explicit a), so the mode
    decays on the comoving clock.
    """
    turb = read_log_param(input_arg, "dMagDiffTurb", 0.) or 0.
    lin = read_log_param(input_arg, "dMagDiffLin", 0.) or 0.
    shock = read_log_param(input_arg, "dMagDiffShock", 0.) or 0.
    constant = read_log_param(input_arg, "bConstantDiffusion", 0.) or 0.
    if not (turb > 0. and constant):
        return None, float('nan')
    if lin > 0. or shock > 0.:
        print(f"disstest[bhelix] {input_arg}: dMagDiffLin={lin:g} "
              f"dMagDiffShock={shock:g} are nonzero; those channels are "
              f"flow-dependent and spoil the fixed-coefficient test",
              file=sys.stderr)
        return None, float('nan')
    return 'comoving', float(turb)


def scan_uwave(input_arg, p):
    """Fourier amplitude of the thermal mode for one case-1 run.

    Columns: a, cosmic time, the ``integral a dt`` clock, the mass-weighted
    mean u, and the complex amplitude of the RELATIVE perturbation
    ``delta = u/<u> - 1`` projected on the mode.  Relative, because the local
    ``-3H(gamma-1)u`` source is linear in u and therefore cancels exactly out
    of delta while leaving the mean on its own ``a^{-3(gamma-1)}`` law.
    """
    files = find_files(input_arg)
    print(f"[scan-uwave] {len(files)} snapshot(s) in {input_arg}")
    tufac = read_tufac(input_arg)
    rows = []
    for fn in files:
        try:
            tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, _h = loaddata(fn)
        except Exception as e:
            print(f"  skip {fn}: {e}", file=sys.stderr)
            continue
        a, t = snapshot_axis(time, p)
        m = np.asarray(tgdata[:, 0], dtype=float)
        x = np.asarray(tgdata[:, 1], dtype=float)
        rho = np.asarray(tgdata[:, 7], dtype=float)
        if np.any(rho <= 0):
            print(f"  skip {fn}: non-positive density", file=sys.stderr)
            continue
        u = tufac*np.asarray(tgdata[:, 8], dtype=float)
        u_mean = float(np.average(u, weights=m))
        if not u_mean > 0:
            continue
        z = _complex_mode(m/rho, x, u/u_mean - 1., p)
        rows.append((a, t, 0., u_mean, abs(z), z.real, z.imag))
    if not rows:
        print(f"disstest: no usable snapshots in {input_arg}", file=sys.stderr)
        return None
    out = np.array(rows, dtype=float)
    return _fill_clocks(out[np.argsort(out[:, T_TIME])], p)


def scan_bhelix(input_arg, p):
    """Helical-mode amplitude for one case-2 run, in scan_uwave's layout.

    The mode is B = (0, A cos(theta), A sin(theta)) with theta = k(x-x0), so
    W = B_y + i B_z = A exp(i theta) and the amplitude is |<W exp(-i theta)>|
    exactly -- no fitting, and it is sensitive to the mode's SHAPE, where a
    plain RMS of |B| would not be.

    T_AMP holds the COMPENSATED amplitude ``A*a^{1/2}``: the ideal (eta = 0)
    behaviour of the stored field is ``B_code propto a^{-1/2}`` (the -H B/2
    induction source), so dividing that out leaves the resistive decay alone.
    In the static arm a is exactly 1 and the compensation is a no-op, which is
    correct -- there is no induction source to remove.
    """
    files = find_files(input_arg)
    print(f"[scan-bhelix] {len(files)} snapshot(s) in {input_arg}")
    rows = []
    for fn in files:
        try:
            tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, _h = loaddata(fn)
        except Exception as e:
            print(f"  skip {fn}: {e}", file=sys.stderr)
            continue
        a, t = snapshot_axis(time, p)
        B = read_bfield(fn, ngas)
        if B is None:
            print(f"  skip {fn}: no BField aux", file=sys.stderr)
            continue
        m = np.asarray(tgdata[:, 0], dtype=float)
        x = np.asarray(tgdata[:, 1], dtype=float)
        rho = np.asarray(tgdata[:, 7], dtype=float)
        if np.any(rho <= 0):
            continue
        volume = m/rho
        theta = p['k']*(x - p['x0'])
        w = (np.asarray(B[:, 1], dtype=float)
             + 1j*np.asarray(B[:, 2], dtype=float))
        amplitude = abs(np.average(w*np.exp(-1j*theta), weights=volume))
        u_mean = float(np.average(np.asarray(tgdata[:, 8], dtype=float),
                                  weights=m))
        rows.append((a, t, 0., u_mean, amplitude*np.sqrt(a), amplitude, 0.))
    if not rows:
        print(f"disstest: no usable snapshots in {input_arg}", file=sys.stderr)
        return None
    out = np.array(rows, dtype=float)
    return _fill_clocks(out[np.argsort(out[:, T_TIME])], p)


def decay_fit(d, p, channel):
    """Fit ln|delta| against `channel`'s clock; return (Lambda, max residual).

    Lambda is the discrete decay eigenvalue.  The residual is what the test
    actually gates on: it says whether the decay really is exponential in THAT
    clock, independently of the coefficient's size, and is therefore immune to
    the SPH truncation error that sets Lambda's absolute accuracy.
    """
    amp = d[:, T_AMP]
    good = np.isfinite(amp) & (amp > 0)
    if np.count_nonzero(good) < 3:
        return float('nan'), float('nan')
    clock = d[good, clock_column(channel)]
    clock = clock - clock[0]
    coefficients = np.polyfit(clock, np.log(amp[good]), 1)
    residual = np.log(amp[good]) - np.polyval(coefficients, clock)
    return -float(coefficients[0]), float(np.max(np.abs(residual)))


def decay_residuals(d, p, channel):
    """Named residuals for one run.  `channel` is the clock the coefficient the
    run was given is SUPPOSED to decay on.

    `residual` is the test: is the decay exponential in that clock.

    `null:wrong-a-power` refits the same data against the other power of `a`
    and is NOT a competing physical model -- it is the null hypothesis, showing
    what a missing or spurious factor of `a` would look like.  Reporting it is
    what demonstrates that a passing run could have failed; `sensitivity` is
    their ratio.  In the static arm the two clocks are identical, so
    `sensitivity` is exactly 1 and correctly announces that this arm has no
    a-dependence to discriminate.
    """
    other = 'comoving' if channel == 'physical' else 'physical'
    lam_own, res_own = decay_fit(d, p, channel)
    _lam_other, res_other = decay_fit(d, p, other)
    return {
        'Lambda': lam_own,
        'e-foldings': float(np.log(d[0, T_AMP]/d[-1, T_AMP])),
        'residual': res_own,
        'null:wrong-a-power': res_other,
        'sensitivity': (res_other/res_own if res_own > 0 else float('nan')),
    }


def plot_decay(results, channels, p, save=None, tag="decay"):
    """Cases 1/2 figure: normalized decay against normalized clock.

    Both axes are scaled to [0, 1] over the run, so a mode that really is
    exponential in that clock lies on the DIAGONAL whatever its coefficient.
    Filled markers use each run's own channel clock; open markers use the other
    channel's.  The gap between them is the test's discriminating power, and
    the normalization is what lets runs with different coefficients and
    different decay depths share one pair of axes.
    """
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, (ax, tax) = plt.subplots(
        1, 2, figsize=(6.6 + 2.4*len(results), 5.4),
        gridspec_kw={'width_ratios': [1.25, 1.0]})

    rows_by_run = []
    for index, (lab, d) in enumerate(results):
        colour = f"C{index % 10}"
        channel, _coefficient = channels[index]
        amp = d[:, T_AMP]
        good = np.isfinite(amp) & (amp > 0)
        decay = np.log(amp[good][0]/amp[good])
        if decay[-1] <= 0:
            continue
        decay = decay/decay[-1]
        other_channel = 'comoving' if channel == 'physical' else 'physical'
        for which, fill in ((channel, True), (other_channel, False)):
            clock = d[good, clock_column(which)]
            span = clock[-1] - clock[0]
            if not span > 0:
                continue
            ax.plot((clock - clock[0])/span, decay, marker='o', ls='none',
                    ms=4.5, mfc=(colour if fill else 'none'), mew=0.9,
                    color=colour)
        ax.plot([], [], marker='o', ls='none', ms=4.5, mfc=colour,
                color=colour, label=f"{lab} [{channel}]")
        rows_by_run.append(decay_residuals(d, p, channel))
    ax.plot([0, 1], [0, 1], ls='-', lw=1.0, color='k', zorder=0)
    ax.plot([], [], marker='o', ls='none', ms=4.5, mfc='0.35', color='0.35',
            label="own clock")
    ax.plot([], [], marker='o', ls='none', ms=4.5, mfc='none', color='0.35',
            label="null: one power of $a$ removed")
    ax.set_xlabel("elapsed clock / total")
    ax.set_ylabel(r"$\ln(\delta_0/\delta)\ /$ total")
    ax.set_title("decay against its intended clock (line = exact)")

    keys = ('Lambda', 'e-foldings', 'residual', 'null:wrong-a-power',
            'sensitivity')
    if rows_by_run:
        cells = [[_cell(r.get(key)) for r in rows_by_run] for key in keys]
        _draw_table(tax, keys, cells, results, "fit residuals")
    else:
        tax.axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    finish_figure_with_legend(
        fig, ax, save=(f"{save}_{tag}.png" if save else None),
        footnote=("filled: |residual| of ln|delta| about a straight line in "
                  "the clock the coefficient implies. open: the null, refitted "
                  "with one power of a removed -- not a competing model"))


def _analyze_decay(args, p, labels, which_channel, scan, name):
    """Shared driver for the two operator cases; they differ only in how the
    mode amplitude is measured and where the coefficient is read from."""
    results, channels = [], []
    for inp, lab in zip(args.inputs, labels):
        channel, coefficient = which_channel(inp)
        if channel is None:
            print(f"disstest[{name}] {lab}: no fixed-coefficient channel is "
                  f"configured in the run log; skipping", file=sys.stderr)
            continue
        d = scan(inp, p)
        if d is None:
            continue
        results.append((lab, d))
        channels.append((channel, coefficient))
        print(f"disstest[{name}] {lab}: channel={channel} "
              f"coefficient={coefficient:g}  ({CHANNELS[channel]})")
    if not results:
        print(f"disstest: no case-{p['case']} data to analyze; exiting.",
              file=sys.stderr)
        sys.exit(1)

    plot_decay(results, channels, p, save=args.save, tag=name)

    worst = 0.0
    for (lab, d), (channel, _c) in zip(results, channels):
        out = decay_residuals(d, p, channel)
        for key, value in out.items():
            print(f"disstest[{name}] {lab}: {key:<20s} {value:.5g}")
        worst = max(worst, out['residual'])
        if p['cosmo'] == 0:
            print(f"disstest[{name}] {lab}: static arm -- the two clocks "
                  f"coincide, so `sensitivity` is 1 by construction and this "
                  f"arm measures the OPERATOR, not the a-factor")
    print(f"disstest[{name}] worst dimensionless residual: {worst:.4g}")
    if args.theory_tol is not None and worst > args.theory_tol:
        print(f"disstest: {name.upper()} FAILED -- worst {worst:.4g} > tol "
              f"{args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(3)


def analyze_uwave(args, params, p, labels, parser):
    """Case-1 driver: parabolic conduction on a sinusoidal thermal mode."""
    _analyze_decay(args, p, labels, uwave_channel, scan_uwave, "uwave")


def analyze_bhelix(args, params, p, labels, parser):
    """Case-2 driver: same shape test on the force-free helical magnetic mode.

    Note the scope honestly: with `bConstantDiffusion` the coefficient contains
    no scale factor at all, so this case is an OPERATOR test.  What it
    establishes is the Brookshaw form and normalization, linearity in the
    coefficient, and any reconstruction-induced reduction -- not an a-factor.
    """
    _analyze_decay(args, p, labels, bhelix_channel, scan_bhelix, "bhelix")


# --------------------------------------------------------------------------- #
#  Case 3: the flow-driven channels -- scale-factor ladder and switch values
# --------------------------------------------------------------------------- #

# Column layout of scan_ladder()'s table.
L_A, L_AV, L_DIFF, L_TURB = 0, 1, 2, 3
L_COLS = 4

# Which aux array each flow-driven channel deposits its rate in.
LADDER_CHANNELS = (('uDotAV', L_AV), ('uDotDiff', L_DIFF))


def turb_reference(x, h, p):
    """The analytic subgrid turbulent coefficient for the imposed flow.

    `p->diff` is ``h_com^2 |S_phys|``.  For a fixed comoving template both
    factors are a-INDEPENDENT: h is comoving by definition, and the physical
    shear rate is
        |dv_phys/dx_phys| = |d(a*xdot)/d(a*x_com)| = |dxdot/dx_com|,
    the two powers of a cancelling exactly.  That is why the ladder predicts
    exponent 0 for this channel, and it is what makes the value check below a
    statement about the SWITCH rather than about the expansion.

    Returns (reference, note).  Only the transverse variant is asserted: it is
    a pure shear, so every trace-free convention agrees on |S| = |dv_y/dx| and
    the ratio is a clean test.  The longitudinal variant has div v != 0, where
    the value depends on the trace-free convention, so it is reported without
    being claimed as exact.
    """
    phase = p['k']*(np.asarray(x, dtype=float) - p['x0'])
    rate = p['k']*p['vamp']*p['V_S']
    h = np.asarray(h, dtype=float)
    if p['variant'] == 1:
        return h*h*rate*np.abs(np.sin(phase)), "transverse: |S| = |dv_y/dx|"
    return (h*h*rate*np.abs(np.cos(phase)),
            "longitudinal: compressive, trace-free convention not asserted")


def _rms(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights*values**2)/np.sum(weights)))


def scan_ladder(input_arg, p):
    """One ladder RUNG: the channel rates at the template, after a single step.

    The rung is run with ``-dt 1e-8 -n 1`` so `a` does not advance and the
    right-hand side is read AT the template, with no time integration and no
    evolved state in between.  What is returned is therefore the instantaneous
    rate each channel would apply, which is exactly what a scale-factor
    exponent should be measured on.

    Returns (a, rate_AV, rate_diff, turbulent-switch ratio) for the LAST
    snapshot in the directory.
    """
    files = find_files(input_arg)
    if not files:
        print(f"disstest: no snapshots in {input_arg}", file=sys.stderr)
        return None
    fn = files[-1]
    try:
        tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, h = loaddata(fn)
    except Exception as e:
        print(f"  skip {fn}: {e}", file=sys.stderr)
        return None
    a, _t = snapshot_axis(time, p)
    m = np.asarray(tgdata[:, 0], dtype=float)
    x = np.asarray(tgdata[:, 1], dtype=float)

    row = [a, np.nan, np.nan, np.nan]
    for label, col in LADDER_CHANNELS:
        arr = try_aux(fn, label, ngas)
        if arr is not None:
            row[col] = _rms(np.asarray(arr, dtype=float), m)

    # The turbulent SWITCH itself, when the build exports it.  Compared where
    # the analytic shear is well away from its zeros: at a node the reference
    # vanishes and the ratio is 0/0, which would dominate any spread estimate.
    diff = try_aux(fn, "diff", ngas)
    if diff is not None:
        reference, _note = turb_reference(x, h, p)
        good = reference > 0.25*np.max(reference)
        if np.count_nonzero(good) > 0:
            row[L_TURB] = float(np.median(
                np.asarray(diff, dtype=float)[good]/reference[good]))
    return np.array(row, dtype=float)


def ladder_exponent(a, rate):
    """d log(rate) / d log(a) and the fit's worst residual, in dex.

    A missing or spurious power of `a` is an INTEGER shift in this slope, which
    is why the test needs no reference implementation and no absolute
    normalization: only the slope carries the answer.
    """
    a = np.asarray(a, dtype=float)
    rate = np.asarray(rate, dtype=float)
    good = np.isfinite(a) & np.isfinite(rate) & (a > 0) & (rate > 0)
    if np.count_nonzero(good) < 3:
        return float('nan'), float('nan')
    lx, ly = np.log(a[good]), np.log(rate[good])
    coefficients = np.polyfit(lx, ly, 1)
    residual = ly - np.polyval(coefficients, lx)
    return float(coefficients[0]), float(np.max(np.abs(residual)))


def scan_resolution(input_arg, p):
    """One RESOLUTION rung: the channel rates and the load's mean `h`.

    The scale-factor ladder holds `nx` fixed and steps `a`; this holds `a`
    fixed and steps `nx`.  It is the test for a SWITCH rather than an operator:
    `absmu = -dvdotdrAV*a` has no continuum limit and vanishes as the
    resolution rises, so what is checked is the RATE at which it vanishes.

    Returns `(h, rate_AV, rate_diff, turbulent ratio)` in scan_ladder's layout,
    with `h` -- the mean smoothing length -- standing in for the scale factor.
    """
    files = find_files(input_arg)
    if not files:
        print(f"disstest: no snapshots in {input_arg}", file=sys.stderr)
        return None
    fn = files[-1]
    try:
        tgdata, _td, _ts, _hdr, _time, _N, ngas, _nd, _ns, h = loaddata(fn)
    except Exception as e:
        print(f"  skip {fn}: {e}", file=sys.stderr)
        return None
    m = np.asarray(tgdata[:, 0], dtype=float)
    row = [float(np.mean(np.asarray(h, dtype=float))), np.nan, np.nan, np.nan]
    for label, col in LADDER_CHANNELS:
        arr = try_aux(fn, label, ngas)
        if arr is not None:
            row[col] = _rms(np.asarray(arr, dtype=float), m)
    return np.array(row, dtype=float)


def analyze_resolution(args, params, p, labels, parser):
    """AV SWITCH driver: the inputs are the rungs of one RESOLUTION ladder.

    Constant `dConstAlpha`/`dConstBeta` do not make AV switch-free -- the pair
    strength is `(ALPHA*(vSig_p+vSig_q) + 1.5*BETA*absmu)*absmu/rhoavg` and
    `absmu` is pure flow.  Both terms must vanish as `h -> 0`.

    The derivation-free check is their RATIO, not their difference: the alpha
    term is LINEAR in `absmu` and the beta term QUADRATIC, so if
    `absmu propto x^p` then alpha goes as `x^p` and beta as `x^2p`, and

        beta exponent / alpha exponent = 2

    whatever `p` is.  (On the SCALE-FACTOR axis `p = 1`, so there the ratio 2
    and a difference of 1 coincide -- which is why the a-ladder could be read
    either way.  On the resolution axis `p != 1` in general and only the ratio
    is invariant.)
    """
    rows = []
    for inp, lab in zip(args.inputs, labels):
        row = scan_resolution(inp, p)
        if row is None:
            continue
        rows.append(row)
        print(f"disstest[avswitch] {lab}: h={row[L_A]:.6g}  "
              f"uDotAV={row[L_AV]:.6g}  uDotDiff={row[L_DIFF]:.6g}")
    if len(rows) < 3:
        print("disstest: need at least 3 resolution rungs; pass each run "
              "directory as a separate input.", file=sys.stderr)
        sys.exit(1)
    table = np.array(rows, dtype=float)
    table = table[np.argsort(table[:, L_A])]

    for name, col in LADDER_CHANNELS:
        slope, res = ladder_exponent(table[:, L_A], table[:, col])
        if not np.isfinite(slope):
            continue
        print(f"disstest[avswitch] {name:<10s} d log(rate)/d log(h) = "
              f"{slope:+.4f}   (fit resid {res:.2e} dex)")
        # h -> 0 as the resolution rises, so a POSITIVE exponent in h is what
        # vanishing looks like.  A non-positive one means the switch survives
        # refinement, which for AV would be a defect.
        if col == L_AV and slope <= 0.:
            print(f"disstest[avswitch] WARNING: {name} does NOT vanish with "
                  f"resolution (exponent {slope:+.4f} <= 0)", file=sys.stderr)
    return table


def plot_ladder(table, p, save=None, label="ladder"):
    """Case-3 figure: each channel's rate up the ladder, and its exponent.

    Left: log rate against log a, one series per channel, with the fitted power
    law drawn through it.  A channel whose a-bookkeeping is right lies on an
    INTEGER slope, and the figure shows the slope rather than asserting it.

    Right: the fitted exponents beside the predictions, which come from the
    code's own definitions rather than from theory:
    ``absmu = -dvdotdrAV*a`` is propto a for a fixed comoving template, so the
    alpha term goes as a and the beta term (one more power of mu) as a^2; the
    diffusive channels need a comoving coefficient ``eta_phys/a^2`` and both
    ``|dxdot x dx|`` and ``h_com^2 |S_phys|`` are fixed, hence exponent 0.
    """
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, (ax, tax) = plt.subplots(1, 2, figsize=(12.4, 5.4),
                                  gridspec_kw={'width_ratios': [1.25, 1.0]})

    rows, cells = [], []
    for index, (name, col) in enumerate(LADDER_CHANNELS):
        colour = f"C{index % 10}"
        a, rate = table[:, L_A], table[:, col]
        good = np.isfinite(rate) & (rate > 0) & (a > 0)
        if np.count_nonzero(good) < 2:
            continue
        ax.plot(a[good], rate[good], 'o', ms=6, mfc='none', mew=1.4,
                color=colour, label=name)
        slope, res = ladder_exponent(a, rate)
        if np.isfinite(slope):
            grid = np.array([a[good].min(), a[good].max()])
            norm = rate[good][0]/a[good][0]**slope
            ax.plot(grid, norm*grid**slope, '-', lw=1.1, color=colour)
            rows.append(name)
            cells.append([f"{slope:+.4f}", f"{res:.1e}"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("$a$")
    ax.set_ylabel("channel rate (mass-weighted rms)")
    ax.set_title("rate up the scale-factor ladder")

    tax.axis('off')
    if rows:
        table_artist = tax.table(cellText=cells, rowLabels=rows,
                                 colLabels=["exponent", "max resid [dex]"],
                                 cellLoc='right', rowLoc='left',
                                 loc='upper center')
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(9)
        table_artist.scale(1.0, 1.4)
    turb = table[:, L_TURB]
    if np.any(np.isfinite(turb)):
        _ref, note = turb_reference(np.array([0.]), np.array([1.]), p)
        tax.text(0.5, 0.28,
                 "turbulent switch p->diff / $h^2|S|$\n"
                 f"median {np.nanmedian(turb):.5f}   ({note})",
                 ha='center', va='top', transform=tax.transAxes, fontsize=9)
    tax.set_title("measured exponents")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    finish_figure_with_legend(
        fig, ax, save=(f"{save}_{label}.png" if save else None),
        footnote=("one fixed comoving template stamped at each a and run for "
                  "a single tiny step, so the rate is the RHS at the template; "
                  "a missing power of a is an integer shift in the slope"))


def analyze_ladder(args, params, p, labels, parser):
    """Case-3 driver: the inputs are the RUNGS of one ladder, not separate runs.

    The frozen v = 0 cases cannot reach any production dissipation channel --
    AV needs `dvdotdrAV`, the default resistivity needs `|dv x dx|` and
    `p->diff` needs `|S|`, all identically zero at v = 0 -- so this is the case
    that tests them, and it is the one whose channels the a-factor audit
    actually touched.
    """
    rows = []
    for inp, lab in zip(args.inputs, labels):
        row = scan_ladder(inp, p)
        if row is None:
            continue
        rows.append(row)
        print(f"disstest[ladder] {lab}: a={row[L_A]:.6g}  "
              f"uDotAV={row[L_AV]:.6g}  uDotDiff={row[L_DIFF]:.6g}"
              + (f"  p->diff/h^2|S|={row[L_TURB]:.6g}"
                 if np.isfinite(row[L_TURB]) else ""))
    if not rows:
        print("disstest: no case-3 data to analyze; exiting.", file=sys.stderr)
        sys.exit(1)
    table = np.array(rows, dtype=float)
    table = table[np.argsort(table[:, L_A])]

    turb = table[:, L_TURB]
    if np.any(np.isfinite(turb)):
        _ref, note = turb_reference(np.array([0.]), np.array([1.]), p)
        print(f"disstest[ladder] turbulent switch p->diff / h^2|S|: "
              f"median {np.nanmedian(turb):.6f}  spread "
              f"{np.nanmax(turb) - np.nanmin(turb):.2e}   ({note})")

    if p['cosmo'] == 0:
        print("disstest[ladder] static arm: there is no scale factor to "
              "ladder over, so this arm reports the SWITCH VALUES only. Run "
              "the same template with -b 1 for the exponents.")
        return
    if table.shape[0] < 3:
        print("disstest[ladder] fewer than 3 rungs: pass each rung directory "
              "as a separate input to fit an exponent.", file=sys.stderr)
        return

    plot_ladder(table, p, save=args.save)

    worst = 0.0
    for name, col in LADDER_CHANNELS:
        slope, res = ladder_exponent(table[:, L_A], table[:, col])
        if not np.isfinite(slope):
            continue
        nearest = round(slope)
        offset = abs(slope - nearest)
        worst = max(worst, offset)
        print(f"disstest[ladder] {name:<10s} exponent {slope:+.4f}  "
              f"(nearest integer {nearest:+d}, |offset| {offset:.4g}, "
              f"fit resid {res:.2e} dex)")
    print(f"disstest[ladder] worst distance from an integer power of a: "
          f"{worst:.4g}")
    if args.theory_tol is not None and worst > args.theory_tol:
        print(f"disstest: LADDER FAILED -- worst {worst:.4g} > tol "
              f"{args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(3)


# --------------------------------------------------------------------------- #

ANALYSES = {0: analyze_null, 1: analyze_uwave,
            2: analyze_bhelix, 3: analyze_ladder}


def main():
    parser = build_parser(PARAM_SPEC,
                          description="Isolated dissipation-operator and "
                                      "switch tests (disstest).")
    parser.add_argument("--reg-tol", type=float, default=None,
                        help="CI gate: exit non-zero if the worst RMS residual "
                             "vs the regression baseline exceeds this "
                             "tolerance")
    parser.add_argument("--theory-tol", type=float, default=None,
                        help="exit non-zero if the case's worst dimensionless "
                             "residual exceeds this value")
    parser.add_argument("--final-a", type=float, default=1.0,
                        help="expanding arm: intended final expansion factor")
    parser.add_argument("--end-tol", type=float, default=5.e-3,
                        help="absolute tolerance for the final expansion "
                             "factor")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="analyze a partial run without failing the "
                             "endpoint check")
    parser.add_argument("--resolution", action="store_true",
                        help="case 3 only: treat the inputs as a RESOLUTION "
                             "ladder (the AV switch test) rather than a "
                             "scale-factor one, and fit d log(rate)/d log(h)")
    args = parser.parse_args()
    params = params_from_args(args, PARAM_SPEC)

    labels = args.labels if args.labels else [
        os.path.basename(os.path.normpath(i)) for i in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error("--labels must match the number of inputs")

    p = case_params(params["case"], cosmo=params["cosmo"],
                    amp=params["amp"], bfac=params["bfac"],
                    variant=params["variant"], a_init=params["a_init"],
                    vamp=params["vamp"], strat=params["strat"])
    arm = "expanding (bComove=1)" if p['cosmo'] == 1 else "static (bComove=0)"
    print(f"[disstest case {p['case']} ({CASES[p['case']]}), {arm}] "
          f"a_i={p['a_i']:.6g} V_S={p['V_S']:.4g} u0={p['uuzero']:.4g} "
          f"B0={p['bzero']:.4g} amp={p['amp']:.4g} vamp={p['vamp']:.4g}")

    if args.resolution:
        if p['case'] != 3:
            parser.error("--resolution is the case 3 (ladder) switch test")
        analyze_resolution(args, params, p, labels, parser)
        return
    ANALYSES[p['case']](args, params, p, labels, parser)


if __name__ == "__main__":
    main()
