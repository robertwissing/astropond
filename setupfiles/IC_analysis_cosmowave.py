#!/usr/bin/env python3
"""
Analysis for the cosmowave test (Berlok 2022, MNRAS 515 3492 + this work's
divB-cleaning extension): linear (M)HD waves in an expanding Einstein-de Sitter
universe. The "time" axis is the SCALE FACTOR a (the Tipsy snapshot time in a
comoving run).  The default diagnostic scales out those power laws and plots
the perturbation magnitude versus ln(a/a_i), making it a useful precursor to
the paper's Figure 6 convergence study.  ``--berlok-figure 1`` instead uses the
paper's literal Figure 1 convention: signed, unscaled amplitudes versus linear
scale factor a.

Per snapshot, Gasoline's coordinate velocity and magnetic variable are first
converted to Berlok's physical peculiar velocity and comoving magnetic field,

    u = a*xdot,        B_c = sqrt(a)*B_Gasoline.

The complex Fourier coefficient (cosine and sine components) is retained so the
diagnostic tests propagation phase as well as amplitude.  Eulerian projections
use particle-volume weights m/rho.

Two opt-in multi-run modes are available:

``--convergence``
    Treat the inputs as a fixed-physics spatial-resolution sequence and plot
    error versus ``n_x``.  Case 0 uses the frequency/decay fit of Berlok's
    equation (77), reproducing the Figure-6 diagnostic.  The other cases use
    the volume-weighted L1 error of the endpoint phase-bin means, so disordered
    particle sampling does not let a single noisy particle set the metric.

``--convergence-t``
    Treat the inputs as a fixed-resolution Courant sequence, reading
    ``dEtaCourant`` from each Gasoline log.  It plots endpoint wave error and
    source-corrected total-energy closure against an ``eta_C^2`` guide.  This
    ideal-energy test requires ``dConstAlpha=dConstBeta=0`` in every log.

Every regular and paper-layout analysis also audits the dense run-log energy
history.  It reconstructs gravitational and gas/MHD cosmological work and plots
the source-corrected total-energy residual; the isothermal case is explicitly
reported as an open thermostat budget rather than as a conservation law.

Cases (-a, must match IC_setup_cosmowave.create's `case`):
    0 - standing Alfven                (paper Sec. 5.1.1, eqs. 75-76)
    1 - traveling Alfven, n=2          (paper Sec. 5.1.2, eqs. 79-80)
    2 - standing compressible          (paper Sec. 5.2.1/2/3, eqs. 81-82)
    3 - traveling compressible g=4/3   (paper Sec. 5.2.1, eqs. 83-84; n=3)
    4 - divergence-cleaning test       (this work)
    5 - standing compressible g=5/3 in LambdaCDM (paper Fig. 5, eqs. 73-74)
    6 - standing isothermal compressible in EdS (paper Fig. 3)
    7 - standing adiabatic g=5/3 compressible in EdS (paper Fig. 4)

Standalone like alfven/gresho/sedov/noh/wave: reuses the framework's
Param/build_parser/params_from_args/find_files/loaddata, not run_cli (the x-axis
is the scale factor, not the per-snapshot time, and there is no single metric).
Several inputs overlay as separate marker series (one shared analytic curve).

Usage:
    python IC_analysis_cosmowave.py <dir> [-a 3] [-b 1] [-c 0] [--save out]
    python IC_analysis_cosmowave.py <dir32> <dir64> -a 4 -e 1 -f 0 --save out
    python IC_analysis_cosmowave.py <dir8> <dir16> <dir32> -a 1 --convergence
    python IC_analysis_cosmowave.py <eta04> <eta01> <eta0025> -a 4 -d 0.1 \
        --convergence-t --save out
"""

import os
import re
import sys

import numpy as np

from IC_analysis_framework import (
    Param, build_parser, params_from_args,
    reference_path_for, save_reference, find_reference, residuals,
    parse_resolution,
)
from IC_analysis_general import (
    loaddata, find_files, try_aux, read_bfield,
    read_tufac, read_log_param, read_log_series,
    setup_rcparams, finish_figure, finish_figure_with_legend, _find_logs,
)

# Fixed IC constants (match IC_setup_cosmowave.create): Hubble, wavenumber
# k=2pi (wavelength = 2*dx = 1) and phase origin xdot0 = -dx.
H0 = 2.894405
KWAVE = 2.0 * np.pi
X0 = -0.5

_RUN_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CASE4_CORE = rf"case4A{_RUN_NUMBER}G{_RUN_NUMBER}u{_RUN_NUMBER}"

# runtest.sh setup_cosmowave() parameter order (runtest.sh:setup_cosmowave).
PARAM_SPEC = [
    Param("a", "case", 3, int,
          "0 standing Alfven, 1 traveling Alfven, 2 standing compr., "
          "3 traveling compr. g=4/3, 4 div-clean EdS, "
          "5 standing compr. g=5/3 LambdaCDM, 6 Fig.3 isothermal EdS, "
          "7 Fig.4 g=5/3 EdS", "case"),
    Param("b", "factor_A", 1.0, float,
          "omega_A = factor_A*pi (0 = MHD off except case 4; "
          "case 4 always uses MHD with zero background)", "A"),
    Param("c", "factor_G", 0.0, float,
          "omega_G = factor_G*pi/2 (0 = self-gravity off, 1 = paper default)",
          "G"),
    Param("d", "A_u", 1e-1, float,
          "linear-mode amplitude (paper 1e-6; practical SPH default 0.1)",
          "u"),
    Param("e", "clean_fch", 1.0, float,
          "case 4 only: hyperbolic cleaning-speed multiplier fch "
          "(0 disables cleaning)",
          re.compile(_CASE4_CORE + rf"f({_RUN_NUMBER})g")),
    Param("f", "clean_damping", 0.0, float,
          "case 4 only: parabolic cleaning coefficient psidecfac",
          re.compile(_CASE4_CORE + rf"f{_RUN_NUMBER}g({_RUN_NUMBER})")),
]


# The shared framework now performs case-name inference for the live CLI.
# This pre-existing local copy remains only for compatibility with callers
# that may import these private helpers; main() no longer calls it.
_CASE_NAME_RE = re.compile(r"case(\d+)(?!\d)", re.IGNORECASE)


def _case_from_input_name(input_arg):
    """Return the unique ``caseNR`` tag in an input basename, if present."""
    run_name = os.path.basename(os.path.normpath(str(input_arg)))
    cases = sorted({int(match.group(1))
                    for match in _CASE_NAME_RE.finditer(run_name)})
    if len(cases) > 1:
        rendered = ", ".join(f"case{case}" for case in cases)
        raise ValueError(
            f"cannot infer one Cosmowave case from {run_name!r}: its name "
            f"contains {rendered}; pass -a/--case explicitly")
    return cases[0] if cases else None


def _infer_case_from_inputs(inputs, default_case=3, prompt=None):
    """Infer one case from input basenames, confirming incomplete conflicts."""
    records = [(str(input_arg), _case_from_input_name(input_arg))
               for input_arg in inputs]
    tagged = [(path, case) for path, case in records if case is not None]
    missing = [path for path, case in records if case is None]

    if not tagged:
        print("WARNING: no -a/--case was given and no input directory name "
              f"contains a caseNR tag; using the historical default case "
              f"{int(default_case)}. Pass -a/--case to choose explicitly.",
              file=sys.stderr)
        return int(default_case)

    cases = sorted({case for _, case in tagged})
    if len(cases) == 1 and not missing:
        plural = "name" if len(records) == 1 else "names"
        print("cosmowave: no -a/--case given; inferred case "
              f"{cases[0]} from {len(records)} input directory {plural}.",
              file=sys.stderr)
        return cases[0]

    if len(cases) > 1:
        reason = "input directory names encode different cases"
    else:
        reason = "not every input directory name contains a caseNR tag"
    print(f"WARNING: no -a/--case was given and {reason}:",
          file=sys.stderr)
    for path, case in records:
        label = f"case {case}" if case is not None else "no caseNR tag"
        print(f"  {label}: {path}", file=sys.stderr)
    print("Cosmowave can project only one case for all inputs.",
          file=sys.stderr)

    chosen_path, chosen_case = tagged[0]
    print(f"Continue using case {chosen_case} from the first tagged input "
          f"({chosen_path})? [y/N] ", end="", file=sys.stderr, flush=True)
    ask = input if prompt is None else prompt
    try:
        answer = ask()
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("cosmowave: stopped; rerun with an explicit -a/--case.",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"cosmowave: continuing with inferred case {chosen_case}.",
          file=sys.stderr)
    return chosen_case


def _resolve_case(explicit_case, inputs, default_case=3, prompt=None):
    """Honor an explicit case, otherwise infer it from the input names."""
    if explicit_case is not None:
        return int(explicit_case)
    return _infer_case_from_inputs(inputs, default_case=default_case,
                                   prompt=prompt)


# --------------------------------------------------------------------------- #
#  Per-case linear-theory constants (mirrors IC_setup_cosmowave.create)
# --------------------------------------------------------------------------- #

def case_params(case, factor_A=1.0, factor_G=0.0, A_u=1e-1, theta=1e-1,
                clean_fch=1.0, clean_damping=0.0):
    """The per-case setup constants computed exactly as IC_setup_cosmowave."""
    p = dict(A_u=float(A_u), H0=H0, k=KWAVE, x0=X0,
             theta=float(theta), clean_fch=float(clean_fch),
             clean_damping=float(clean_damping))
    p['case'] = int(case)
    if p['case'] not in (0, 1, 2, 3, 4, 5, 6, 7):
        raise ValueError("cosmowave case must be one of 0, 1, 2, 3, 4, 5, 6, "
                         "7")
    if p['clean_fch'] < 0:
        raise ValueError("clean_fch must be non-negative")
    if p['clean_damping'] < 0:
        raise ValueError("clean_damping must be non-negative")
    p['omega_S'] = (2.*np.pi if p['case'] == 6 else
                    np.pi/5. if p['case'] in (5, 7) else np.pi)
    p['omega_A'] = factor_A * np.pi
    p['omega_G'] = factor_G * np.pi / 2.0
    p['gamma'] = (1. if p['case'] == 6 else
                  5./3. if p['case'] in (5, 7, 8, 9, 10) else 4./3.)
    p['omega_m'] = 0.3 if p['case'] == 5 else 1.0
    p['omega_lambda'] = 0.7 if p['case'] == 5 else 0.0
    p['alfven'] = int(case) in (0, 1)
    sigma2 = p['omega_A']**2 + p['omega_S']**2 - p['omega_G']**2
    p['sigma2'] = sigma2
    p['sigma'] = np.sqrt(sigma2) if sigma2 >= 0 else np.nan
    kappa2 = sigma2 - 1./16.
    kappa_A2 = p['omega_A']**2 - 1./16.
    if p['case'] in (2, 3) and kappa2 <= 0:
        raise ValueError("compressible Berlok cases require sigma^2 > 1/16")
    if p['case'] in (0, 1) and kappa_A2 <= 0:
        raise ValueError("Alfven Berlok cases require omega_A^2 > 1/16")
    p['kappa'] = np.sqrt(max(kappa2, 0.0))
    p['kappa_A'] = np.sqrt(max(kappa_A2, 0.0))
    if int(case) in (0, 2, 4, 5, 6, 7):
        p['a_i'] = 1. / 128.
    elif int(case) == 1:
        p['a_i'] = np.exp(-2 * 2 * np.pi / p['kappa_A'])   # n=2
    else:                                                  # case 3
        p['a_i'] = np.exp(-2 * 3 * np.pi / p['kappa'])     # n=3
    p['V_A'] = (p['H0'] / p['k']) * p['omega_A']
    p['V_S'] = (p['H0'] / p['k']) * p['omega_S']
    p['V_G'] = (p['H0'] / p['k']) * p['omega_G']
    p['rhozero'] = (((p['k'] * p['V_G'])**2) / (4 * np.pi)
                    if p['omega_G'] != 0 else 1.0)
    # Case 4 deliberately has no uniform magnetic background.  Bc0 is its
    # injected-field normalization, based on V_S so the linear cleaning speed
    # is known independently of the divergence perturbation.  The paper cases
    # retain the usual background-field definition based on V_A.
    p['Bc0'] = ((p['V_S'] if p['case'] == 4 else p['V_A'])
                * np.sqrt(p['rhozero']))
    p['Bgaszero_i'] = p['Bc0'] * p['a_i']**(-0.5)
    return p


# --------------------------------------------------------------------------- #
#  Linear-theory amplitude predictions (Berlok 2022), all comoving
# --------------------------------------------------------------------------- #

def theory_coefficients(a, p):
    """Complex coefficients Z for f(x)=Re[Z exp(i*k*(x-x0))].

    Returns dimensionless Berlok variables: delta-u/V, delta-Bc/Bc0, and
    delta-rho_c/rho_c0.  These are equations 75-84 with the right-traveling
    convention used by IC_setup_cosmowave.py.
    """
    a = np.asarray(a, dtype=float)
    ai, A_u = p['a_i'], p['A_u']
    R = a / ai
    zero = np.zeros_like(a, dtype=complex)

    if p['case'] == 0:                         # standing Alfven, eqs. 75-76
        kap = p['kappa_A']
        psi = kap * np.log(R)
        bamp = A_u * np.sqrt(ai) * p['omega_A'] / kap
        zB = 1j * bamp * R**(-0.25) * np.sin(psi)
        zu = A_u * R**(-0.75) * (
            np.cos(psi) - np.sin(psi) / (4.0 * kap))
        return {'u': zu.astype(complex), 'B': zB, 'rho': zero}

    if p['case'] == 1:                         # traveling Alfven, eqs. 79-80
        kap = p['kappa_A']
        psi = kap * np.log(R)
        phase = np.exp(-1j * psi)
        pref = A_u * np.sqrt(ai) / (4.0 * p['omega_A'])
        zB = pref * R**(-0.25) * (-4.0 * kap - 1j) * phase
        zu = A_u * R**(-0.75) * phase
        return {'u': zu, 'B': zB, 'rho': zero}

    if p['case'] == 2:                         # standing compressible, eqs. 81-82
        kap = p['kappa']
        psi = kap * np.log(R)
        ramp = A_u * p['omega_S'] * np.sqrt(ai) / kap
        zr = -1j * ramp * R**(-0.25) * np.sin(psi)
        zu = A_u * R**(-0.75) * (
            np.cos(psi) - np.sin(psi) / (4.0 * kap))
        return {'u': zu.astype(complex), 'B': zr, 'rho': zr}

    if p['case'] == 3:                         # traveling compressible, eqs. 83-84
        kap = p['kappa']
        psi = kap * np.log(R)
        phase = np.exp(-1j * psi)
        pref = A_u * p['omega_S'] * np.sqrt(ai) / (4.0 * p['sigma2'])
        zr = pref * R**(-0.25) * (4.0 * kap + 1j) * phase
        zu = A_u * R**(-0.75) * phase
        return {'u': zu, 'B': zr, 'rho': zr}

    if p['case'] in (5, 6, 7):                 # general standing mode, eqs. 73-74
        return scipy_theory_coefficients(a, p)

    return {'u': zero, 'B': zero, 'rho': zero}


def scipy_theory_coefficients(a, p):
    """Berlok's numeric reference for compressible waves (paper eqs. 73-74).

    The integration is cached on ``p`` so every plot and metric samples the
    same high-accuracy dense solution.  ``delta_w = a*delta_u`` is the second
    evolved variable, exactly as in Berlok's public reference implementation.
    """
    from scipy.integrate import solve_ivp

    if '_scipy_solution' not in p:
        def rhs(ascale, state):
            drho_frac, delta_w = state
            omega_k = 1.0 - p['omega_m'] - p['omega_lambda']
            adot = (ascale*p['H0']
                    * np.sqrt(p['omega_m']/ascale**3
                              + omega_k/ascale**2
                              + p['omega_lambda']))
            return np.asarray([
                -1j*p['k']*delta_w/(ascale**2*adot),
                -1j*p['k']/adot
                * (p['V_S']**2/ascale**(3.*(p['gamma'] - 1.))
                   + (p['V_A']**2 - p['V_G']**2)/ascale)
                * drho_frac,
            ], dtype=complex)

        initial = np.asarray([0.+0.j,
                              p['a_i']*p['A_u']*p['V_S']+0.j])
        solution = solve_ivp(rhs, (p['a_i'], 1.0), initial, method='BDF',
                             dense_output=True, atol=1.e-10, rtol=1.e-8)
        if not solution.success:
            raise RuntimeError("Berlok scipy reference integration failed: "
                               + solution.message)
        p['_scipy_solution'] = solution

    a = np.asarray(a, dtype=float)
    state = p['_scipy_solution'].sol(a)
    zr = np.asarray(state[0], dtype=complex)
    zu = np.asarray(state[1]/(a*p['V_S']), dtype=complex)
    return {'u': zu, 'B': zr, 'rho': zr}


def _alfven_envelope(a, p):
    th = theory_coefficients(a, p)
    return np.abs(th['B']), np.abs(th['u'])


def _alfven_traveling(a, p):
    th = theory_coefficients(a, p)
    return np.abs(th['B']), np.abs(th['u'])


def _compressible_envelope(a, p, traveling=None):
    th = theory_coefficients(a, p)
    return np.abs(th['rho']), np.abs(th['u']), np.abs(th['B'])


def divclean_theory_coefficients(a, p, h_comoving=None):
    """Intended comoving GLM mode for case 4.

    In an EdS gamma=4/3 background, the physical fast/cleaning speed scales as
    ``a^-1/2``.  Both the comoving spatial derivative and a physical damping
    length contribute ``1/a``, so the B/Q system has constant coefficients in
    ``s=ln(a/a_i)``::

        db/ds = Omega*q,       dq/ds = -Omega*b - Lambda*q.

    ``b`` is the sine amplitude of ``delta Bc/Bscale`` and ``q`` is the cosine
    amplitude of ``Qc/Bc0``, where ``Qc=sqrt(a*rho_c)*BClean`` for Gasoline's
    evolved cleaning variable.  A pure hyperbolic run has Lambda=0 and exactly
    conserves ``b**2+q**2``.  With damping, ``h_comoving`` is required because
    ``Lambda=psidecfac*fch*V_S/(H0*h_comoving)``.  The magnetic background is
    zero, so the linear cleaning speed is the sound speed and
    ``Omega=fch*omega_S``; magnetic-force contamination is only second order in
    the injected amplitude.
    """
    if int(p['case']) != 4:
        raise ValueError("divclean theory is defined only for cosmowave case 4")
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0):
        raise ValueError("scale factors must be positive")
    s = np.log(a/p['a_i'])
    omega = p['clean_fch']*p['omega_S']
    damping = p['clean_damping']
    if damping == 0:
        decay = 0.0
    else:
        if h_comoving is None or not np.isfinite(h_comoving) or h_comoving <= 0:
            raise ValueError("positive h_comoving is required for damped cleaning")
        decay = damping*p['clean_fch']*p['V_S']/(p['H0']*h_comoving)

    half = 0.5*decay
    discriminant = omega**2 - half**2
    prefactor = p['A_u']*np.exp(-half*s)
    tolerance = 1.e-14*max(1.0, omega**2, half**2)
    if discriminant > tolerance:
        frequency = np.sqrt(discriminant)
        bamp = prefactor*(np.cos(frequency*s)
                          + half/frequency*np.sin(frequency*s))
        qamp = -prefactor*omega/frequency*np.sin(frequency*s)
    elif discriminant < -tolerance:
        rate = np.sqrt(-discriminant)
        bamp = prefactor*(np.cosh(rate*s)
                          + half/rate*np.sinh(rate*s))
        qamp = -prefactor*omega/rate*np.sinh(rate*s)
    else:
        bamp = prefactor*(1.0 + half*s)
        qamp = -prefactor*omega*s

    return {'B': -1j*bamp, 'Q': qamp.astype(complex),
            'omega': float(omega), 'decay': float(decay)}


# --------------------------------------------------------------------------- #
#  Per-run amplitude scan vs scale factor
# --------------------------------------------------------------------------- #

# Numeric columns returned by scan_input().  Magnitude columns are retained for
# backward-compatible regression files; the real/imaginary columns carry the
# phase-sensitive diagnostic.
(C_A, C_DU, C_DB, C_DRHO, C_DBX, C_DBY, C_DBZ, C_DIV_RAW_RMS,
 C_DIV_RAW_MAX, C_PSI_MAX, C_DU_RE, C_DU_IM, C_DB_RE, C_DB_IM,
 C_DRHO_RE, C_DRHO_IM, C_BC_BG, C_DIV_DIM_RMS, C_DIV_DIM_MAX,
 C_RHO_BG, C_DP_RE, C_DP_IM, C_P_BG, C_Q_RE, C_Q_IM, C_H_MEAN,
 C_DIVBC_RMS, C_DIVBC_MAX, C_DUX_RE, C_DUX_IM, C_DUY_RE, C_DUY_IM,
 C_DUZ_RE, C_DUZ_IM) = range(34)


def _complex_mode(weights, x, field, p):
    """Weighted particle fit Z=Ac-i*As for f=c0+Re[Z exp(i*k(x-x0))].

    Solving the 3-column weighted least-squares problem retains phase without
    interpolating to a grid and remains unbiased when the finite glass sampling
    does not make 1, sin(theta), and cos(theta) exactly orthogonal.
    """
    weights = np.asarray(weights, dtype=float)
    x = np.asarray(x, dtype=float)
    field = np.asarray(field, dtype=float)
    good = (np.isfinite(weights) & (weights > 0) & np.isfinite(x)
            & np.isfinite(field))
    if np.count_nonzero(good) < 3:
        return complex(np.nan, np.nan)
    phase = p['k'] * (x[good] - p['x0'])
    design = np.column_stack((np.ones(phase.size), np.cos(phase), np.sin(phase)))
    root_weight = np.sqrt(weights[good])
    coeff, _resid, rank, _singular = np.linalg.lstsq(
        design * root_weight[:, None], field[good] * root_weight, rcond=None)
    if rank < 3:
        return complex(np.nan, np.nan)
    return complex(coeff[1], -coeff[2])


def _weighted_rms(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def _oscillator_phase_advance(oscillator):
    """Return the continuously unwrapped phase advance of ``b+i*q``.

    The minus sign makes an undamped Case-4 mode advance positively.  For a
    damped mode this phase is not linear in ``ln(a/a_i)``, so callers must
    compare it with the phase of the full analytic oscillator rather than with
    ``Omega*ln(a/a_i)``.
    """
    oscillator = np.asarray(oscillator, dtype=complex)
    if oscillator.size == 0:
        return np.asarray([], dtype=float)
    phase = -np.unwrap(np.angle(oscillator))
    return phase-phase[0]


def scan_input(input_arg, p):
    """Amplitude table for one run, sorted by scale factor a.

    Columns: a, dv_amp, dB_amp, drho_amp, dBx, dBy, dBz, divB_l2, divB_max,
    psi_max. dv/dB are the projections onto the mode each case excites."""
    files = find_files(input_arg)
    print(f"[scan] {len(files)} snapshot(s) in {input_arg}")
    rows = []
    tufac = read_tufac(input_arg)
    for fn in files:
        try:
            tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, h = loaddata(fn)
        except Exception as e:
            print(f"  skip {fn}: {e}", file=sys.stderr)
            continue
        a = float(np.asarray(time).ravel()[0])
        if a <= 0:
            a = p['a_i']
        m, x = tgdata[:, 0], tgdata[:, 1]
        vx, vy, vz = tgdata[:, 4], tgdata[:, 5], tgdata[:, 6]
        rho = tgdata[:, 7]
        if np.any(rho <= 0):
            print(f"  skip {fn}: non-positive density", file=sys.stderr)
            continue
        volume = m / rho
        rho_mean = float(np.average(rho, weights=volume))
        if p['gamma'] == 1.:
            # Isothermal p=c_s^2*rho, so its fractional mode is exactly the
            # density mode.  The arbitrary constant c_s^2 cancels here.
            pressure = rho.copy()
        else:
            pressure = ((p['gamma'] - 1.) * rho
                        * tufac * np.asarray(tgdata[:, 8], dtype=float))
        pressure_mean = float(np.average(pressure, weights=volume))
        zp = (_complex_mode(volume, x, pressure - pressure_mean, p)
              / pressure_mean)

        # Tipsy snapshots store coordinate velocity xdot.  Berlok's delta-u is
        # the physical peculiar velocity u=a*xdot.
        zUx = _complex_mode(volume, x, a*vx, p)
        zUy = _complex_mode(volume, x, a*vy, p)
        zUz = _complex_mode(volume, x, a*vz, p)
        zu = zUy if p['alfven'] else zUx
        zr = _complex_mode(volume, x, rho - rho_mean, p)

        B = read_bfield(fn, ngas)
        Bc = None
        if B is not None:
            # Gasoline's B_s=a^(3/2)B_phys; Berlok uses B_c=a^2 B_phys.
            Bc = np.sqrt(a) * B
            Bc_mean = np.average(Bc, axis=0, weights=volume)
            zBx = _complex_mode(volume, x, Bc[:, 0] - Bc_mean[0], p)
            zBy = _complex_mode(volume, x, Bc[:, 1] - Bc_mean[1], p)
            zBz = _complex_mode(volume, x, Bc[:, 2] - Bc_mean[2], p)
            zB = {0: zBy, 1: zBy, 2: zBz, 3: zBz, 4: zBx,
                  5: zBz, 6: zBz, 7: zBz, 8: zBx, 9: zBx,
                  10: zBy}[p['case']]
            if p['case'] == 4:
                # Case 4 has zero background; retain the magnitude of any mean
                # field generated by discretization as a leakage diagnostic.
                Bc_background = float(np.linalg.norm(Bc_mean))
            else:
                bg_component = {0: 0, 1: 0, 2: 2, 3: 2,
                                5: 2, 6: 2, 7: 2, 8: 0, 9: 0,
                                10: 0}[p['case']]
                Bc_background = float(Bc_mean[bg_component])
        else:
            zBx = zBy = zBz = zB = complex(np.nan, np.nan)
            Bc_background = np.nan

        divB = try_aux(fn, "DivB", ngas)           # gasoline aux is 'DivB'
        if divB is not None:
            divB_l2 = _weighted_rms(divB, volume)
            divB_max = float(np.max(np.abs(divB)))
            # The auxiliary is the divergence of Gasoline's
            # B_s=a^(3/2)B_phys.  Berlok's B_c=sqrt(a)B_s, and a is spatially
            # uniform, hence div_x(B_c)=sqrt(a)*div_x(B_s).
            divBc_l2 = np.sqrt(a)*divB_l2
            divBc_max = np.sqrt(a)*divB_max
        else:
            divB_l2 = divB_max = np.nan
            divBc_l2 = divBc_max = np.nan

        if divB is not None and B is not None:
            Bmag = np.linalg.norm(B, axis=1)
            positive = Bmag[Bmag > 0]
            Bfloor = ((float(np.median(positive)) * 1.e-30)
                      if positive.size else np.nan)
            if np.isfinite(Bfloor):
                if p['case'] == 4:
                    # The longitudinal test field has physical zero crossings,
                    # so a pointwise h|divB|/|B| is singular.  Use the global
                    # norm ratio instead.
                    Brms = _weighted_rms(Bmag, volume)
                    hdiv = np.abs(np.asarray(h)*divB)
                    div_dim_l2 = _weighted_rms(hdiv, volume)/Brms
                    div_dim_max = float(np.max(hdiv)/Brms)
                else:
                    div_dimensionless = (np.abs(np.asarray(h)*divB)
                                         / np.maximum(Bmag, Bfloor))
                    div_dim_l2 = _weighted_rms(div_dimensionless, volume)
                    div_dim_max = float(np.max(div_dimensionless))
            else:
                div_dim_l2 = div_dim_max = np.nan
        else:
            div_dim_l2 = div_dim_max = np.nan

        psi = try_aux(fn, "BClean", ngas)
        psi_max = float(np.max(np.abs(psi))) if psi is not None else np.nan
        if psi is not None:
            Qc = np.sqrt(a*rho)*np.asarray(psi, dtype=float)
            Qc_mean = float(np.average(Qc, weights=volume))
            zQ = _complex_mode(volume, x, Qc-Qc_mean, p)
        else:
            zQ = complex(np.nan, np.nan)
            Qc = None
        h_mean = float(np.average(np.asarray(h, dtype=float), weights=volume))

        rows.append((a, abs(zu), abs(zB), abs(zr), abs(zBx), abs(zBy), abs(zBz),
                     divB_l2, divB_max, psi_max,
                     zu.real, zu.imag, zB.real, zB.imag, zr.real, zr.imag,
                     Bc_background, div_dim_l2, div_dim_max, rho_mean,
                     zp.real, zp.imag, pressure_mean, zQ.real, zQ.imag,
                     h_mean, divBc_l2, divBc_max,
                     zUx.real, zUx.imag, zUy.real, zUy.imag,
                     zUz.real, zUz.imag))
        vnorm = p['V_A'] if p['alfven'] else p['V_S']
        message = (f"  a={a:.5g}  |du|/V={abs(zu)/vnorm:.3e}  "
                   f"|drho|/rho0={abs(zr)/p['rhozero']:.3e}")
        if p['Bc0'] != 0:
            message += f"  |dBc|/Bc0={abs(zB)/p['Bc0']:.3e}"
        if np.isfinite(div_dim_l2):
            message += f"  hdivB/B={div_dim_l2:.3e}"
        if p['case'] == 4 and p['Bc0'] != 0 and np.isfinite(zQ.real):
            message += f"  |Qc|/Bc0={abs(zQ)/p['Bc0']:.3e}"
            umax = max(abs(zUx), abs(zUy), abs(zUz))/p['V_S']
            message += f"  max|du_xyz|/Vs={umax:.3e}"
        print(message)

    if not rows:
        print(f"cosmowave: no usable snapshots in {input_arg}", file=sys.stderr)
        return None
    out = np.array(rows, dtype=float)
    return out[np.argsort(out[:, 0])]


# --------------------------------------------------------------------------- #
#  Plots (amplitude vs ln(a/a_i)), legend in its own figure
# --------------------------------------------------------------------------- #

def _a_curve(results, p):
    a_all = np.concatenate([d[:, 0] for _, d in results])
    return np.geomspace(p['a_i'], a_all.max(), 400)


def _gamma_label(p):
    """Exact equation-of-state label used in every figure heading."""
    gamma = float(p['gamma'])
    if np.isclose(gamma, 1.0, rtol=0.0, atol=1.e-12):
        value = '1'
    elif np.isclose(gamma, 4.0/3.0, rtol=0.0, atol=1.e-12):
        value = '4/3'
    elif np.isclose(gamma, 5.0/3.0, rtol=0.0, atol=1.e-12):
        value = '5/3'
    else:
        value = f'{gamma:g}'
    return rf'$\gamma = {value}$'


def _physical_case_name(p):
    """Describe the physical problem without exposing its numeric case ID."""
    return {
        0: "Standing Alfvén wave",
        1: "Traveling Alfvén wave",
        2: "Standing magnetosonic wave",
        3: "Traveling magnetosonic wave",
        4: "Divergence-cleaning wave",
        5: r"Standing compressible wave ($\Lambda$CDM)",
        6: "Standing isothermal compressible wave",
        7: "Standing adiabatic compressible wave",
        8: "Homogeneous expanding MHD state",
        9: "Sinusoidal thermal-diffusion mode",
        10: "Force-free helical magnetic-diffusion mode",
    }.get(int(p['case']), "Cosmological wave")


def _figure_title(p, description):
    """Append the equation of state to an already descriptive heading."""
    return f"{description} ({_gamma_label(p)})"


def plot_alfven(results, p, save=None, ref_table=None):
    import matplotlib.pyplot as plt
    setup_rcparams()
    ac = _a_curve(results, p)
    lac = np.log(ac / p['a_i'])
    if p['case'] == 0:
        dBth, duth = _alfven_envelope(ac, p)
        title = "Standing Alfvén scaled magnitudes"
    else:
        dBth, duth = _alfven_traveling(ac, p)
        title = (r"Traveling Alfvén wave: expansion-scaled $k=1$ "
                 "Fourier-mode history")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Reference overlay: the blessed amplitudes rescaled to each panel's axes.
    if ref_table is not None and "dB" in ref_table:
        ar = ref_table["x"]; lnar = np.log(ar / p['a_i'])
        ax[0].plot(lnar, (ref_table["dB"] / p['Bc0']) * (ar / p['a_i'])**0.25,
                   "--", color="0.5", lw=1.4, zorder=5, label="reference")
        if "dv" in ref_table:
            ax[1].plot(lnar, (ref_table["dv"] / p['V_A']) * (ar / p['a_i'])**0.75,
                       "--", color="0.5", lw=1.4, zorder=5, label="reference")
    for lab, d in results:
        a, dv, dB = d[:, 0], d[:, 1], d[:, 2]
        lna = np.log(a / p['a_i'])
        ax[0].plot(lna, (dB / p['Bc0']) * (a / p['a_i'])**0.25,
                   marker='o', ls='none', label=lab)
        ax[1].plot(lna, (dv / p['V_A']) * (a / p['a_i'])**0.75,
                   marker='o', ls='none', label=lab)
    ax[0].plot(lac, dBth * (ac / p['a_i'])**0.25, 'k-', label='linear theory')
    ax[1].plot(lac, duth * (ac / p['a_i'])**0.75, 'k-', label='linear theory')
    ax[0].set_xlabel(r"$\ln(a/a_i)$")
    ax[0].set_ylabel(r"$(\delta B_c / B_{c,0}) \, (a/a_i)^{1/4}$")
    ax[0].set_title("transverse magnetic mode")
    ax[1].set_xlabel(r"$\ln(a/a_i)$")
    ax[1].set_ylabel(r"$(\delta u / V_A) \, (a/a_i)^{3/4}$")
    ax[1].set_title("transverse velocity mode")
    fig.suptitle(_figure_title(p, title))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish_figure_with_legend(fig, (ax[0], ax[1]),
                              save=(f"{save}_alfven.png" if save else None))


def _spatial_mode_amplitude(values, basis):
    """Convert Z=Ac-i*As to its physical cosine or sine coefficient."""
    values = np.asarray(values, dtype=complex)
    if basis == 'cosine':
        return values.real
    if basis == 'sine':
        return -values.imag
    raise ValueError(f"unknown spatial basis: {basis}")


def _berlok_figure1_components(zB, zu, p):
    """Return the signed, unscaled amplitudes shown in Berlok Figure 1.

    ``theory_coefficients`` and the particle projection use
    ``f=Re[Z exp(i k x)]``.  The standing-wave magnetic perturbation in paper
    equation (75) is the coefficient of ``sin(k x)``, hence ``-Im(Z_B)``;
    the velocity in equation (76) is the coefficient of ``cos(k x)``, hence
    ``Re(Z_u)``.  Berlok divides simulation amplitudes by ``A_u``.
    """
    return (_spatial_mode_amplitude(zB, 'sine')/p['A_u'],
            _spatial_mode_amplitude(zu, 'cosine')/p['A_u'])


def plot_berlok_figure1(results, p, save=None):
    """Paper-style standing-Alfven history: signed amplitudes versus linear a."""
    import matplotlib.pyplot as plt
    setup_rcparams()

    amax = max(1.0, max(float(d[:, C_A].max()) for _label, d in results))
    acurve = np.linspace(p['a_i'], amax, 1600)
    theory = theory_coefficients(acurve, p)
    theory_B, theory_u = _berlok_figure1_components(
        theory['B'], theory['u'], p)

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    for label, d in results:
        measured_B = _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0'])
        measured_u = _series_coeff(d, C_DU_RE, C_DU_IM, p['V_A'])
        fig1_B, fig1_u = _berlok_figure1_components(measured_B, measured_u, p)
        points, = ax[0].plot(d[:, C_A], fig1_B, marker='o', ms=3.5,
                             ls='none', alpha=0.85, label=label)
        ax[1].plot(d[:, C_A], fig1_u, marker='o', ms=3.5, ls='none',
                   alpha=0.85, color=points.get_color(), label=label)

    ax[0].plot(acurve, theory_B, 'k-', lw=2.3, label='analytic solution')
    ax[1].plot(acurve, theory_u, 'k-', lw=2.3, label='analytic solution')
    ax[0].set_ylabel(r'$\delta B_c/B_c$')
    ax[1].set_ylabel(r'$\delta u/\mathcal{V}_A$')
    ax[1].set_xlabel(r'$a$')
    ax[1].set_xlim(0.0, amax)
    fig.suptitle(_figure_title(p, "Comoving, standing Alfvén wave"))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish_figure_with_legend(
        fig, ax,
        save=(f"{save}_standing_alfven_signed_amplitudes.png"
              if save else None),
        footnote=(r"Signed $kL=2\pi$ Fourier amplitudes divided by $A_u$, "
                  r"unscaled, versus linear $a$."))


def plot_compressible(results, p, save=None, ref_table=None):
    import matplotlib.pyplot as plt
    setup_rcparams()
    ac = _a_curve(results, p)
    lac = np.log(ac / p['a_i'])
    traveling = (p['case'] == 3)
    lcdm = (p['case'] == 5)
    paper_raw = p['case'] in (5, 6, 7)
    drth, duth, dBth = _compressible_envelope(ac, p, traveling=traveling)
    if traveling:
        title = "Traveling compressible Fourier-envelope history"
    elif lcdm:
        title = (r"Standing compressible Fourier-amplitude history "
                 r"($\Lambda$CDM; numerical reference)")
    elif p['case'] == 6:
        title = "Standing isothermal compressible history"
    elif p['case'] == 7:
        title = "Standing adiabatic compressible history"
    else:
        title = "Standing compressible Fourier-amplitude history"

    def axes_for(values, power):
        values = np.asarray(values, dtype=float)
        if paper_raw:
            return values, np.full_like(values, 1.0/p['A_u'])
        return np.log(values/p['a_i']), (values/p['a_i'])**power

    have_B = p['omega_A'] != 0 and p['Bc0'] != 0
    n_panels = 3 if have_B else 2
    fig, ax = plt.subplots(1, n_panels, figsize=(5.4*n_panels, 5), squeeze=False)
    ax = ax[0]
    # Reference overlay: the blessed amplitudes rescaled to each panel's axes.
    if ref_table is not None and "drho" in ref_table:
        ar = ref_table["x"]
        xr, sr = axes_for(ar, 0.25)
        ax[0].plot(xr, (ref_table["drho"] / p['rhozero']) * sr,
                   "--", color="0.5", lw=1.4, zorder=5, label="reference")
        if "dv" in ref_table:
            xr, sr = axes_for(ar, 0.75)
            ax[1].plot(xr, (ref_table["dv"] / p['V_S']) * sr,
                       "--", color="0.5", lw=1.4, zorder=5, label="reference")
        if have_B and "dB" in ref_table:
            xr, sr = axes_for(ar, 0.25)
            ax[2].plot(xr, (ref_table["dB"] / p['Bc0']) * sr,
                       "--", color="0.5", lw=1.4, zorder=5, label="reference")
    for lab, d in results:
        a, dv, dB, drho = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        xp, sp = axes_for(a, 0.25)
        ax[0].plot(xp, (drho / p['rhozero']) * sp,
                   marker='o', ls='none', label=lab)
        xp, sp = axes_for(a, 0.75)
        ax[1].plot(xp, (dv / p['V_S']) * sp,
                   marker='o', ls='none', label=lab)
        if have_B:
            xp, sp = axes_for(a, 0.25)
            ax[2].plot(xp, (dB / p['Bc0']) * sp,
                       marker='o', ls='none', label=lab)
    xc, sc = axes_for(ac, 0.25)
    ax[0].plot(xc, drth * sc, 'k-', lw=2.4,
               label='analytic solution')
    xc, sc = axes_for(ac, 0.75)
    ax[1].plot(xc, duth * sc, 'k-', lw=2.4,
               label='analytic solution')
    if have_B:
        xc, sc = axes_for(ac, 0.25)
        ax[2].plot(xc, dBth * sc, 'k-', lw=2.4,
                   label='analytic solution')
    xlabel = r"$a$" if paper_raw else r"$\ln(a/a_i)$"
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel((r"$|\delta\rho/\rho_0|/A_u$" if paper_raw else
                      r"$(\delta\rho/\rho_0) \, (a/a_i)^{1/4}$"))
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel((r"$|\delta u/V_S|/A_u$" if paper_raw else
                      r"$(\delta u / V_S) \, (a/a_i)^{3/4}$"))
    if have_B:
        ax[2].set_xlabel(xlabel)
        ax[2].set_ylabel((r"$|\delta B_c/B_{c,0}|/A_u$" if paper_raw else
                          r"$(\delta B_c / B_{c,0}) \, (a/a_i)^{1/4}$"))
    if paper_raw:
        for axis in ax:
            axis.set_xscale('log')
    fig.suptitle(_figure_title(p, title))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish_figure_with_legend(fig, ax,
                              save=(f"{save}_compressible.png" if save else None))


def _real_mode(x, coefficient, p):
    """Real-space mode for the analyzer's Z convention.

    If f(x)=Ac*cos(theta)+As*sin(theta), _complex_mode returns Z=Ac-i*As,
    hence f=Re[Z*exp(i*theta)].
    """
    x = np.asarray(x, dtype=float)
    return np.real(coefficient * np.exp(1j*p['k']*(x-p['x0'])))


def _endpoint_profile(input_arg, p):
    """Load the latest snapshot as raw particle values (no grid interpolation)."""
    files = find_files(input_arg)
    if not files:
        return None
    fn = files[-1]
    tgdata, _td, _ts, _hdr, time, _N, ngas, _nd, _ns, _h = loaddata(fn)
    a = float(np.asarray(time).ravel()[0])
    if a <= 0:
        a = p['a_i']
    rho = np.asarray(tgdata[:, 7], dtype=float)
    if np.any(rho <= 0):
        return None
    mass = np.asarray(tgdata[:, 0], dtype=float)
    x = np.asarray(tgdata[:, 1], dtype=float)
    volume = mass/rho
    v_coord = tgdata[:, 5] if p['alfven'] else tgdata[:, 4]
    profile = {
        'file': fn,
        'a': a,
        'x': x,
        'volume': volume,
        'u': a*np.asarray(v_coord, dtype=float)
    }
    if not p['alfven']:
        profile['rho'] = rho
        if p['gamma'] == 1.0:
            pressure = rho.copy()
        else:
            tufac = read_tufac(input_arg)
            pressure = ((p['gamma']-1.0)*rho*tufac
                        * np.asarray(tgdata[:, 8], dtype=float))
        profile['p'] = pressure
        profile['p_background'] = float(np.average(pressure, weights=volume))
    B = read_bfield(fn, ngas)
    if B is not None:
        Bc = np.sqrt(a)*B
        # Case 4 perturbs the longitudinal component; the paper cases perturb
        # the component transverse to their homogeneous field.
        component = {0: 1, 1: 1, 2: 2, 3: 2, 4: 0, 5: 2, 6: 2, 7: 2}[p['case']]
        profile['B'] = np.asarray(Bc[:, component], dtype=float)
    if p['case'] == 4:
        # Same per-particle cleaning variable the scan projects onto the mode:
        # Qc = sqrt(a*rho_c)*BClean, mean removed.
        psi = try_aux(fn, "BClean", ngas)
        if psi is not None:
            Qc = np.sqrt(a*rho)*np.asarray(psi, dtype=float)
            profile['Q'] = Qc - float(np.average(Qc, weights=volume))
    return profile


def _profile_error(measured, expected, weights):
    """RMS/max profile error normalized by the corresponding analytic norm."""
    measured = np.asarray(measured, dtype=float)
    expected = np.asarray(expected, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rms_den = _weighted_rms(expected, weights)
    max_den = float(np.max(np.abs(expected)))
    if rms_den == 0 or max_den == 0:
        return np.nan, np.nan
    return (_weighted_rms(measured-expected, weights)/rms_den,
            float(np.max(np.abs(measured-expected)))/max_den)


def _phase_bin_means(x, values, weights, nbins=64):
    """Volume-weighted phase means used for display and spatial L1 errors."""
    xphase = np.mod(np.asarray(x, dtype=float) + 0.5, 1.0)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ibin = np.minimum((xphase*nbins).astype(int), nbins - 1)
    sum_w = np.bincount(ibin, weights=weights, minlength=nbins)
    sum_y = np.bincount(ibin, weights=weights*values, minlength=nbins)
    good = sum_w > 0
    centers = (np.arange(nbins, dtype=float) + 0.5)/nbins
    return centers[good], sum_y[good]/sum_w[good]


def _phase_bin_l1_error(x, measured, expected, weights, nbins):
    """Volume-weighted L1 norm after projecting residuals into phase bins.

    Glass realizations carry irreducible particle-to-particle sampling scatter.
    Taking a maximum, or even L1, before the Eulerian phase projection measures
    that scatter instead of convergence of the resolved wave.  Average the
    residual in one bin per nominal x resolution first, then take its L1 norm.
    Coherent profile error is retained while zero-mean particle noise cancels.
    """
    xphase = np.mod(np.asarray(x, dtype=float) + 0.5, 1.0)
    residual = (np.asarray(measured, dtype=float)
                - np.asarray(expected, dtype=float))
    weights = np.asarray(weights, dtype=float)
    nbins = int(nbins)
    if nbins < 1:
        raise ValueError("phase-bin L1 requires at least one bin")
    ibin = np.minimum((xphase*nbins).astype(int), nbins - 1)
    sum_w = np.bincount(ibin, weights=weights, minlength=nbins)
    sum_residual = np.bincount(
        ibin, weights=weights*residual, minlength=nbins)
    good = sum_w > 0.0
    if not np.any(good):
        return np.nan
    bin_residual = sum_residual[good]/sum_w[good]
    return float(np.average(np.abs(bin_residual), weights=sum_w[good]))


def _endpoint_phase_bin_count(input_arg):
    """Use one phase bin per nominal x element, within a readable bound."""
    nx = _cosmowave_resolution(input_arg)
    return max(8, min(int(nx), 128)) if nx is not None else 64


def _plot_particle_phase_rows(series, quantities, analytic_curves, p,
                              description, save=None):
    """Render the shared endpoint particle/phase-mean two-row layout.

    ``series`` entries contain ``label``, ``input_arg``, ``x``, ``volume`` and
    a ``values`` mapping.  ``quantities`` entries are ``(key, ylabel, title)``.
    ``analytic_curves[key]`` is a sequence of ``(x, y, plot_kwargs)`` tuples;
    every curve is repeated identically in the particle and phase-mean rows.
    The calling case remains responsible only for preparing physical values and
    theory curves.
    """
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, axes = plt.subplots(
        2, len(quantities), figsize=(5.4*len(quantities), 8.2),
        squeeze=False, sharex='col')
    run_colours = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', [f'C{i}' for i in range(10)])

    for column, (key, ylabel, title) in enumerate(quantities):
        particle_axis = axes[0, column]
        mean_axis = axes[1, column]
        for run_index, item in enumerate(series):
            values = item['values'].get(key)
            if values is None:
                continue
            colour = run_colours[run_index % len(run_colours)]
            x = np.asarray(item['x'], dtype=float)
            volume = np.asarray(item['volume'], dtype=float)
            values = np.asarray(values, dtype=float)
            particle_axis.scatter(
                x+0.5, values, s=2.0, alpha=0.18, color=colour,
                rasterized=True, label=item['label'])
            xbin, ybin = _phase_bin_means(
                x, values, volume,
                nbins=_endpoint_phase_bin_count(item['input_arg']))
            mean_axis.plot(
                xbin, ybin, marker='o', ms=3.0, lw=1.2, color=colour,
                zorder=4, label=item['label'])

        for curve_x, curve_y, plot_kwargs in analytic_curves.get(key, ()):
            for axis in (particle_axis, mean_axis):
                axis.plot(curve_x, curve_y, **plot_kwargs)
        for axis in (particle_axis, mean_axis):
            axis.set_xlim(0, 1)
            axis.set_ylabel(ylabel)
        particle_axis.set_title(title)
        particle_axis.text(
            0.02, 0.96, 'particle samples', transform=particle_axis.transAxes,
            ha='left', va='top', fontsize=11, color='0.25')
        mean_axis.text(
            0.02, 0.96, 'volume-weighted phase means',
            transform=mean_axis.transAxes,
            ha='left', va='top', fontsize=11, color='0.25')
        mean_axis.set_xlabel(r'$x/L$')

    fig.suptitle(_figure_title(p, description))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8)
    finish_figure_with_legend(fig, tuple(axes.ravel()), save=save)


def plot_berlok_endpoint(inputs, labels, p, save=None):
    """Endpoint particles and volume-weighted phase means in separate rows."""
    if p['case'] not in (0, 1, 2, 3, 5, 6, 7):
        return
    profiles = []
    for inp, label in zip(inputs, labels):
        prof = _endpoint_profile(inp, p)
        if prof is not None:
            profiles.append((label, inp, prof))
    if not profiles:
        return

    quantities = []
    vnorm = p['V_A'] if p['alfven'] else p['V_S']
    if not p['alfven']:
        quantities.append(('rho', r'$(a/a_i)^{1/4}\,\delta\rho_c/\rho_{c,0}$',
                           0.25))
    quantities.append(('u', (r'$(a/a_i)^{3/4}\,\delta u/V_A$' if p['alfven']
                            else r'$(a/a_i)^{3/4}\,\delta u/V_S$'), 0.75))
    if p['omega_A'] != 0:
        quantities.append(('B', r'$(a/a_i)^{1/4}\,\delta B_c/B_{c,0}$', 0.25))

    xcurve = np.linspace(-0.5, 0.5, 800)
    a_curve = max(prof['a'] for _label, _inp, prof in profiles)
    theory_end = theory_coefficients(np.array([a_curve]), p)
    quantity_titles = {
        'rho': 'density',
        'u': 'transverse velocity' if p['alfven'] else 'longitudinal velocity',
        'B': 'transverse magnetic field' if p['alfven'] else 'magnetic field',
    }
    norms = {'rho': p['rhozero'], 'u': vnorm, 'B': p['Bc0']}
    backgrounds = {
        'rho': p['rhozero'],
        'u': 0.0,
        # The Alfven perturbation is transverse to its homogeneous field;
        # compressible cases perturb the component containing Bc0.
        'B': 0.0 if p['alfven'] else p['Bc0'],
    }
    paper_figure = {1: 7, 3: 9}.get(int(p['case']))
    profile_name = (f'Fig{paper_figure} profile'
                    if paper_figure is not None else 'profile')
    prepared_series = []
    for label, input_arg, prof in profiles:
        a = prof['a']
        theory_at_a = theory_coefficients(np.array([a]), p)
        values = {}
        for key, _ylabel, power in quantities:
            if key not in prof:
                continue
            scaled = ((prof[key]-backgrounds[key])/norms[key]
                      * (a/p['a_i'])**power)
            expected = (_real_mode(prof['x'], theory_at_a[key][0], p)
                        * (a/p['a_i'])**power)
            rms, mx = _profile_error(scaled, expected, prof['volume'])
            print(f"cosmowave[{profile_name}] {label} {key:<3s}: "
                  f"normalized rms={rms:.4g} max={mx:.4g}  a={a:.8g}")
            values[key] = scaled
        prepared_series.append({
            'label': label, 'input_arg': input_arg,
            'x': prof['x'], 'volume': prof['volume'], 'values': values,
        })

    analytic_curves = {}
    quantity_specs = []
    for key, ylabel, power in quantities:
        endpoint_curve = (_real_mode(xcurve, theory_end[key][0], p)
                          * (a_curve/p['a_i'])**power)
        analytic_curves[key] = ((
            xcurve+0.5, endpoint_curve,
            {'color': 'k', 'ls': '-', 'lw': 2.5,
             'label': f'analytic solution, a={a_curve:.4g}'}),)
        quantity_specs.append((key, ylabel, quantity_titles[key]))
    case_name = {
        0: 'Standing Alfvén',
        1: 'Traveling Alfvén',
        2: 'Standing magnetosonic',
        3: 'Traveling magnetosonic',
        5: 'Standing compressible',
        6: 'Standing isothermal compressible',
        7: 'Standing adiabatic compressible',
    }.get(int(p['case']), "Cosmological wave")
    profile_suffix = {
        0: 'standing_alfven_endpoint_profile',
        1: 'traveling_alfven_endpoint_profile',
        2: 'standing_magnetosonic_endpoint_profile',
        3: 'traveling_magnetosonic_endpoint_profile',
        5: 'standing_compressible_endpoint_profile',
        6: 'standing_isothermal_endpoint_profile',
        7: 'standing_adiabatic_endpoint_profile',
    }.get(int(p['case']), 'endpoint_profile')
    _plot_particle_phase_rows(
        prepared_series, tuple(quantity_specs), analytic_curves, p,
        f"{case_name} endpoint profiles at a={a_curve:.4g}",
        save=(f"{save}_{profile_suffix}.png" if save else None))


# --------------------------------------------------------------------------- #
#  Multi-run convergence diagnostics
# --------------------------------------------------------------------------- #

def _cosmowave_resolution(input_arg):
    """Return n_x without mistaking a renamed ``case4...`` directory for 4.

    Normal suite runs start with ``cosmowave<n_x>``.  Targeted investigations
    sometimes rename the containing directory while retaining the standard
    snapshot prefix, so inspect that prefix as a fallback.
    """
    candidates = [os.path.basename(os.path.normpath(input_arg))]
    try:
        files = find_files(input_arg)
    except (FileNotFoundError, OSError):
        files = []
    if files:
        candidates.append(os.path.basename(files[0]))
    for candidate in candidates:
        if candidate.lower().startswith('cosmowave'):
            resolution = parse_resolution(candidate)
            if resolution is not None:
                return resolution
    # Investigation directories are often renamed to describe the experiment,
    # while the launch line in their log still names the standard
    # ``datafiles/cosmowave<n_x>...`` IC.  Reading only that first command line
    # makes the fixed-resolution timestep check work for those runs too.
    for log_path in _find_logs(input_arg):
        try:
            with open(log_path) as stream:
                launch_line = stream.readline()
        except OSError:
            continue
        match = re.search(r'(?i)(?:^|[/\s])cosmowave(\d+)', launch_line)
        if match:
            return int(match.group(1))
    return None


def _power_law_order(x, error, inverse=False):
    """Best-fit convergence order for ``error ~ x^p`` or ``x^-p``."""
    x = np.asarray(x, dtype=float)
    error = np.asarray(error, dtype=float)
    good = np.isfinite(x) & (x > 0.0) & np.isfinite(error) & (error > 0.0)
    if np.count_nonzero(good) < 2:
        return np.nan
    slope = float(np.polyfit(np.log(x[good]), np.log(error[good]), 1)[0])
    return -slope if inverse else slope


def _endpoint_convergence_data(input_arg, d, p, normalize_amplitude=True):
    """Scaled endpoint particle profiles and direct-particle error norms.

    Berlok Figures 7 and 9 use ``max(abs(sim-theory))`` after dividing by the
    injected amplitude and applying the analytic EdS power-law scaling.  This
    helper returns that direct-particle maximum and L1 for diagnostics.  The
    spatial plot subsequently measures L1 after volume-weighted phase binning;
    the timestep diagnostic retains the particle maximum and divides by
    ``A_u`` to compare cases with different injected amplitudes.
    """
    profile = _endpoint_profile(input_arg, p)
    if profile is None:
        return None
    a = float(profile['a'])
    if int(p['case']) == 4:
        theory = _divclean_run_theory(np.asarray([a]), d, p)
        specs = (
            ('B', r'$B_c$', p['Bc0'], 0.0, 0.0, theory['B'][0]),
            ('Q', r'$Q_c$', p['Bc0'], 0.0, 0.0, theory['Q'][0]),
        )
    else:
        theory = theory_coefficients(np.asarray([a]), p)
        power_rho = 0.25 if int(p['case']) in (0, 1, 2, 3) else 0.0
        power_u = 0.75 if int(p['case']) in (0, 1, 2, 3) else 0.0
        specs = []
        if not p['alfven']:
            specs.append(('rho', r'$\delta\rho_c/\rho_{c,0}$',
                          p['rhozero'], p['rhozero'], power_rho,
                          theory['rho'][0]))
        specs.append(('u', (r'$\delta u/\mathcal{V}_A$' if p['alfven']
                            else r'$\delta u/\mathcal{V}_s$'),
                      p['V_A'] if p['alfven'] else p['V_S'], 0.0,
                      power_u, theory['u'][0]))
        if not p['alfven'] and p['gamma'] != 1.0:
            pzero = profile.get('p_background', np.nan)
            specs.append(('p', r'$\delta p_c/p_{c,0}$', pzero, pzero,
                          power_rho, p['gamma']*theory['rho'][0]))
        if p['omega_A'] != 0.0 and p['Bc0'] != 0.0:
            specs.append(('B', r'$\delta B_c/B_{c,0}$', p['Bc0'],
                          0.0 if p['alfven'] else p['Bc0'], power_rho,
                          theory['B'][0]))

    quantities = {}
    for key, display, norm, background, power, coefficient in specs:
        if key not in profile or not np.isfinite(norm) or norm == 0.0:
            continue
        amplitude_scale = p['A_u'] if normalize_amplitude else 1.0
        scale = (a/p['a_i'])**power/amplitude_scale
        measured = ((np.asarray(profile[key], dtype=float)-background)/norm
                    * scale)
        expected = _real_mode(profile['x'], coefficient, p)*scale
        difference = measured-expected
        quantities[key] = {
            'display': display,
            'power': power,
            'measured': measured,
            'expected': expected,
            'max': float(np.max(np.abs(difference))),
            'l1': float(np.average(np.abs(difference),
                                   weights=profile['volume'])),
        }
    return {'a': a, 'profile': profile, 'quantities': quantities}


def _fit_standing_alfven_eq77(d, p):
    """Joint B/u fit to Berlok equation (77).

    The magnetic and velocity amplitudes have independent sine/cosine
    coefficients but share the numerical frequency and exponential decay
    rate.  ``decay`` returned here is Gamma_num/2, the quantity plotted in the
    lower-right panel of Berlok Figure 6.
    """
    from scipy.optimize import least_squares

    a = np.asarray(d[:, C_A], dtype=float)
    s = np.log(a/p['a_i'])
    R = a/p['a_i']
    zB = _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0'])
    zu = _series_coeff(d, C_DU_RE, C_DU_IM, p['V_A'])
    magnetic = -zB.imag*R**0.25/p['A_u']
    velocity = zu.real*R**0.75/p['A_u']
    finite = (np.isfinite(s) & np.isfinite(magnetic)
              & np.isfinite(velocity))
    s, magnetic, velocity = s[finite], magnetic[finite], velocity[finite]
    if s.size < 6 or np.ptp(s) <= 0.0:
        raise ValueError("equation-77 fit needs at least six distinct snapshots")

    magnetic_scale = max(float(np.max(np.abs(magnetic))), 1.e-12)
    velocity_scale = max(float(np.max(np.abs(velocity))), 1.e-12)

    def curves(parameters, coordinate=s):
        bs, bc, us, uc, frequency, decay = parameters
        envelope = np.exp(-decay*coordinate)
        return (
            envelope*(bs*np.sin(frequency*coordinate)
                      + bc*np.cos(frequency*coordinate)),
            envelope*(us*np.sin(frequency*coordinate)
                      + uc*np.cos(frequency*coordinate)),
        )

    def residual(parameters):
        fit_B, fit_u = curves(parameters)
        return np.concatenate(((fit_B-magnetic)/magnetic_scale,
                               (fit_u-velocity)/velocity_scale))

    kappa = p['kappa_A']
    initial = np.asarray([
        -np.sqrt(p['a_i'])*p['omega_A']/kappa, 0.0,
        -1.0/(4.0*kappa), 1.0, kappa, 0.0,
    ])
    lower = np.asarray([-np.inf, -np.inf, -np.inf, -np.inf,
                        0.5*kappa, -5.0])
    upper = np.asarray([np.inf, np.inf, np.inf, np.inf,
                        1.5*kappa, 5.0])
    fit = least_squares(residual, initial, bounds=(lower, upper),
                        max_nfev=20000, xtol=1.e-12, ftol=1.e-12,
                        gtol=1.e-12)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("equation-77 nonlinear fit failed")
    fit_B, fit_u = curves(fit.x)
    return {
        's': s, 'B': magnetic, 'u': velocity,
        'B_fit': fit_B, 'u_fit': fit_u,
        'frequency': float(fit.x[4]), 'decay': float(fit.x[5]),
    }


def _plot_power_guide(axis, x, y, power, label, inverse=False):
    """Draw a slope guide through a representative finite data point."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.flatnonzero(np.isfinite(x) & (x > 0.0)
                          & np.isfinite(y) & (y > 0.0))
    if not good.size:
        return
    anchor = good[len(good)//2]
    curve_x = np.geomspace(float(np.min(x[good])), float(np.max(x[good])), 100)
    exponent = -power if inverse else power
    curve_y = y[anchor]*(curve_x/x[anchor])**exponent
    axis.plot(curve_x, curve_y, 'k--', lw=1.3, label=label)


def plot_standing_alfven_convergence(run_specs, p, save=None):
    """Berlok Figure-6 frequency and numerical-decay convergence layout."""
    entries = []
    for label, input_arg, _params, d in run_specs:
        nx = _cosmowave_resolution(input_arg)
        if nx is None:
            raise ValueError(
                f"cannot read cosmowave<n_x> resolution from '{input_arg}'")
        fitted = _fit_standing_alfven_eq77(d, p)
        entries.append((nx, label, fitted))
    entries.sort(key=lambda item: item[0])
    nxs = np.asarray([item[0] for item in entries], dtype=float)
    if len(np.unique(nxs)) != len(nxs):
        raise ValueError("--convergence requires one run per distinct n_x")

    import matplotlib.pyplot as plt
    setup_rcparams()
    fig = plt.figure(figsize=(12.6, 7.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.75, 1.0), wspace=0.30,
                            hspace=0.16)
    ax_B = fig.add_subplot(grid[0, 0])
    ax_u = fig.add_subplot(grid[1, 0], sharex=ax_B)
    ax_frequency = fig.add_subplot(grid[0, 1])
    ax_decay = fig.add_subplot(grid[1, 1], sharex=ax_frequency)

    smax = max(float(item[2]['s'].max()) for item in entries)
    scurve = np.linspace(0.0, smax, 1600)
    acurve = p['a_i']*np.exp(scurve)
    theory = theory_coefficients(acurve, p)
    theory_B = -theory['B'].imag*(acurve/p['a_i'])**0.25
    theory_u = theory['u'].real*(acurve/p['a_i'])**0.75
    ax_B.plot(scurve, theory_B, 'k-', lw=2.2, label='analytic solution')
    ax_u.plot(scurve, theory_u, 'k-', lw=2.2, label='analytic solution')

    frequencies, decays = [], []
    for nx, _label, fitted in entries:
        points, = ax_B.plot(fitted['s'], p['A_u']*fitted['B'],
                            marker='o', ms=3.2,
                            ls='none', alpha=0.75, label=rf'$N_x={nx}$')
        colour = points.get_color()
        ax_u.plot(fitted['s'], p['A_u']*fitted['u'], marker='o', ms=3.2,
                  ls='none', alpha=0.75, color=colour,
                  label=rf'$N_x={nx}$')
        order = np.argsort(fitted['s'])
        ax_B.plot(fitted['s'][order], p['A_u']*fitted['B_fit'][order], ':',
                  lw=1.7, color=colour,
                  label=rf'joint damped-mode fit, $N_x={nx}$')
        ax_u.plot(fitted['s'][order], p['A_u']*fitted['u_fit'][order], ':',
                  lw=1.7, color=colour)
        frequencies.append(fitted['frequency'])
        decays.append(fitted['decay'])
        print(f"cosmowave[convergence] n_x={nx:<5d} "
              f"kappa_num={fitted['frequency']:.9g} "
              f"frequency_error={abs(fitted['frequency']-p['kappa_A'])/p['kappa_A']:.6g} "
              f"Gamma_num/2={fitted['decay']:.6g}")

    frequencies = np.asarray(frequencies)
    decays = np.asarray(decays)
    frequency_error = np.abs(frequencies-p['kappa_A'])/p['kappa_A']
    decay_magnitude = np.abs(decays)
    ax_frequency.loglog(nxs, frequency_error, 'o-', label='joint B/u fit')
    ax_decay.loglog(nxs, decay_magnitude, 'o-', label='joint B/u fit')
    negative_decay = decays < 0.0
    if np.any(negative_decay):
        ax_decay.plot(nxs[negative_decay], decay_magnitude[negative_decay],
                      'o', mfc='white', color='C0', ls='none',
                      label='negative fitted decay')
    _plot_power_guide(ax_frequency, nxs, frequency_error, 2.0,
                      r'$\propto N_x^{-2}$', inverse=True)
    _plot_power_guide(ax_decay, nxs, decay_magnitude, 3.0,
                      r'$\propto N_x^{-3}$', inverse=True)

    ax_B.set_ylabel(r'$(a/a_i)^{1/4}\,\delta B_c/B_{c,0}$')
    ax_u.set_ylabel(r'$(a/a_i)^{3/4}\,\delta u/\mathcal{V}_A$')
    ax_u.set_xlabel(r'$\ln(a/a_i)$')
    ax_frequency.set_ylabel(r'$|\kappa_{\rm theory}-\kappa_{\rm num}|/'
                            r'\kappa_{\rm theory}$')
    ax_decay.set_ylabel(r'$|\Gamma_{\rm num}/2|$')
    ax_decay.set_xlabel(r'$N_x$')
    ax_frequency.set_title('frequency convergence')
    ax_decay.set_title('numerical decay convergence')
    ax_frequency.grid(True, which='both', alpha=0.25)
    ax_decay.grid(True, which='both', alpha=0.25)
    fig.suptitle(_figure_title(
        p, 'Comoving standing Alfvén wave: frequency and decay convergence'))
    fig.subplots_adjust(top=0.91)
    finish_figure_with_legend(
        fig, (ax_B, ax_u, ax_frequency, ax_decay),
        save=(f"{save}_convergence.png" if save else None),
        footnote=(r'Joint fit of scaled $B_c$ and $u$ to a shared damped mode; '
                  r'dashed guides show second-order frequency and third-order '
                  r'numerical-decay convergence.'))


def plot_endpoint_convergence(run_specs, p, save=None):
    """Endpoint profiles plus phase-mean L1 convergence for Cases 1--7."""
    entries = []
    for label, input_arg, _params, d in run_specs:
        nx = _cosmowave_resolution(input_arg)
        if nx is None:
            raise ValueError(
                f"cannot read cosmowave<n_x> resolution from '{input_arg}'")
        endpoint = _endpoint_convergence_data(
            input_arg, d, p, normalize_amplitude=False)
        if endpoint is None or not endpoint['quantities']:
            raise ValueError(f"no usable endpoint profile in '{input_arg}'")
        entries.append((nx, label, endpoint))
    entries.sort(key=lambda item: item[0])
    nxs = np.asarray([entry[0] for entry in entries], dtype=float)
    if len(np.unique(nxs)) != len(nxs):
        raise ValueError("--convergence requires one run per distinct n_x")

    key_order = ('rho', 'u', 'p', 'B', 'Q')
    quantity_keys = [key for key in key_order
                     if any(key in entry[2]['quantities'] for entry in entries)]
    if not quantity_keys:
        raise ValueError("no common endpoint quantities for convergence")

    import matplotlib.pyplot as plt
    setup_rcparams()
    nrows = len(quantity_keys)
    fig = plt.figure(figsize=(12.7, max(5.2, 2.55*nrows)))
    grid = fig.add_gridspec(nrows, 2, width_ratios=(1.75, 1.0),
                            wspace=0.30, hspace=0.12)
    profile_axes = []
    for row in range(nrows):
        profile_axes.append(fig.add_subplot(
            grid[row, 0], sharex=profile_axes[0] if profile_axes else None))
    error_axis = fig.add_subplot(grid[:, 1])
    xcurve = np.linspace(-0.5, 0.5, 800)

    all_errors = []
    for row, key in enumerate(quantity_keys):
        axis = profile_axes[row]
        errors = []
        valid_nx = []
        display = None
        for nx, _label, endpoint in entries:
            quantity = endpoint['quantities'].get(key)
            if quantity is None:
                continue
            display = quantity['display']
            profile = endpoint['profile']
            # One phase bin per nominal x element makes the error norm measure
            # exactly the smooth endpoint profiles displayed in this column.
            nbins = max(8, int(nx))
            xbin, ybin = _phase_bin_means(
                profile['x'], quantity['measured'], profile['volume'],
                nbins=nbins)
            axis.plot(xbin, ybin, marker='o', ms=3.0, lw=1.1,
                      label=rf'$N_x={int(nx)}$')
            phase_l1 = _phase_bin_l1_error(
                profile['x'], quantity['measured'], quantity['expected'],
                profile['volume'], nbins=nbins)
            errors.append(phase_l1)
            valid_nx.append(nx)
            print(f"cosmowave[convergence] n_x={int(nx):<5d} {key:<3s}: "
                  f"max={quantity['max']:.7g} "
                  f"particle_L1={quantity['l1']:.7g} "
                  f"phase_L1={phase_l1:.7g} "
                  f"a={endpoint['a']:.8g}")

        # The finest run supplies the displayed endpoint theory; final-a
        # validation below ensures all members target the same epoch.
        finest = next((entry[2] for entry in reversed(entries)
                       if key in entry[2]['quantities']), None)
        if finest is not None:
            quantity = finest['quantities'][key]
            profile = finest['profile']
            # Re-evaluate the already-scaled analytic profile on a dense grid
            # by fitting its exact single Fourier coefficient.
            z_expected = _complex_mode(profile['volume'], profile['x'],
                                       quantity['expected'], p)
            axis.plot(xcurve+0.5, _real_mode(xcurve, z_expected, p),
                      'k-', lw=2.1, label='analytic')
        axis.axhline(0.0, color='0.75', lw=0.7, zorder=0)
        axis.set_ylabel(display)
        if row == nrows-1:
            axis.set_xlabel(r'$x/L$')
        else:
            axis.tick_params(labelbottom=False)   # sharex: one labelled row
        valid_nx = np.asarray(valid_nx, dtype=float)
        errors = np.asarray(errors, dtype=float)
        order = _power_law_order(valid_nx, errors, inverse=True)
        order_text = f", p={order:.2f}" if np.isfinite(order) else ""
        error_axis.loglog(valid_nx, errors, 'o-',
                          label=f"{display} phase $L_1${order_text}")
        all_errors.extend(errors[np.isfinite(errors) & (errors > 0.0)])

    if all_errors:
        anchor = np.full_like(nxs, max(all_errors), dtype=float)
        _plot_power_guide(error_axis, nxs, anchor, 2.0,
                          r'$\propto N_x^{-2}$', inverse=True)
    error_axis.set_xlabel(r'$N_x$')
    error_axis.set_ylabel(r'volume-weighted phase-mean $L_1$ error')
    error_axis.set_title(r'endpoint $L_1$ error')
    error_axis.grid(True, which='both', alpha=0.25)
    title = f"{_physical_case_name(p)}: endpoint profiles and spatial convergence"
    fig.suptitle(_figure_title(p, title), y=0.985)
    # Keep the endpoint-profile keys under the profile column and the error
    # metrics under the convergence column.  Pooling every labelled artist in
    # one figure-wide legend obscures which key describes which half.
    profile_handles, profile_labels, seen = [], [], set()
    for axis in profile_axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label and label not in seen:
                seen.add(label)
                profile_handles.append(handle)
                profile_labels.append(label)
    error_handles, error_labels = error_axis.get_legend_handles_labels()

    fig.subplots_adjust(top=0.90, bottom=0.25)
    profile_box = profile_axes[-1].get_position()
    error_box = error_axis.get_position()
    fig.legend(
        profile_handles, profile_labels,
        loc='lower center',
        bbox_to_anchor=((profile_box.x0+profile_box.x1)/2.0, 0.015),
        ncol=min(4, len(profile_labels)), frameon=False,
        fontsize=10.5,
        handlelength=2.5, handletextpad=0.8, columnspacing=1.3)
    fig.legend(
        error_handles, error_labels,
        loc='lower center',
        bbox_to_anchor=((error_box.x0+error_box.x1)/2.0, 0.015),
        ncol=min(2, len(error_labels)), frameon=False,
        fontsize=10.5,
        handlelength=2.5, handletextpad=0.8, columnspacing=1.3)
    finish_figure(fig, save=(f"{save}_convergence.png" if save else None))


# Amplitude histories the temporal-convergence figure refines, as
# (key, column, background normalisation, Hubble decay power, axis label).  The
# powers divide out the analytic a^-1/4 (density/field) and a^-3/4 (velocity)
# so every curve is O(1) across the whole run.
_HISTORY_QUANTITIES = (
    ('rho', C_DRHO, 'rhozero', 0.25, r'$\delta\rho/\rho_0$'),
    ('u', C_DU, None, 0.75, r'$\delta u/V$'),
    ('B', C_DB, 'Bc0', 0.25, r'$\delta B_c/B_{c,0}$'),
)


def _history_convergence_series(d, p):
    """De-scaled amplitude histories and their analytic counterparts.

    The same quantities the case's own amplitude-history figure shows -- the
    kL=2pi Fourier magnitude of each evolved field over the whole run -- so the
    temporal error norm below measures exactly what that figure plots.
    """
    a = d[:, C_A]
    theory = theory_coefficients(a, p)
    vnorm = p['V_A'] if p['alfven'] else p['V_S']
    series = []
    for key, column, norm_name, power, ylabel in _HISTORY_QUANTITIES:
        norm = vnorm if norm_name is None else p[norm_name]
        if not norm:
            continue
        descale = (a/p['a_i'])**power/p['A_u']
        analytic = np.abs(theory[key])*descale
        if not np.any(np.isfinite(analytic) & (analytic > 0.0)):
            continue          # a quantity this case does not excite
        series.append((key, ylabel, d[:, column]/norm*descale, analytic))
    return a, series


def _history_l1(measured, analytic):
    """Normalised temporal L1 of one amplitude history against theory."""
    finite = np.isfinite(measured) & np.isfinite(analytic)
    if not np.any(finite):
        return np.nan
    denom = float(np.sqrt(np.mean(analytic[finite]**2)))
    if denom == 0.0:
        return np.nan
    return float(np.mean(np.abs(measured[finite]-analytic[finite]))/denom)


def plot_history_convergence(run_specs, p, save=None):
    """Amplitude-vs-a histories and their temporal L1 error against theory.

    The companion to the spatial figure: that one refines the ENDPOINT profile
    in x, this one refines the whole time history -- the series the case's first
    figure plots -- so a resolution sequence can be judged in a as well as in x.
    """
    entries = []
    for _label, input_arg, _params, d in run_specs:
        nx = _cosmowave_resolution(input_arg)
        if nx is None:
            raise ValueError(
                f"cannot read cosmowave<n_x> resolution from '{input_arg}'")
        a, series = _history_convergence_series(d, p)
        if not series:
            raise ValueError(f"no usable amplitude history in '{input_arg}'")
        entries.append((nx, a, series))
    entries.sort(key=lambda item: item[0])
    nxs = np.asarray([entry[0] for entry in entries], dtype=float)
    if len(np.unique(nxs)) != len(nxs):
        raise ValueError("--convergence requires one run per distinct n_x")

    keys = [key for key, _lab, _m, _t in entries[0][2]]
    # The analytic reference is evaluated on a DENSE scale-factor grid rather
    # than the snapshot cadence: at the fine oscillation rate late in the run a
    # per-snapshot curve would render as visibly piecewise-linear.
    # Clipped to the sampled range: the reference is there to be compared
    # against the points, and the L1 below is measured over exactly this span.
    smin = min(float(np.log(entry[1]/p['a_i']).min()) for entry in entries)
    smax = max(float(np.log(entry[1]/p['a_i']).max()) for entry in entries)
    scurve = np.linspace(smin, smax, 1600)
    acurve = p['a_i']*np.exp(scurve)
    curve_theory = theory_coefficients(acurve, p)

    import matplotlib.pyplot as plt
    setup_rcparams()
    nrows = len(keys)
    fig = plt.figure(figsize=(12.7, max(5.2, 2.55*nrows)))
    grid = fig.add_gridspec(nrows, 2, width_ratios=(1.75, 1.0),
                            wspace=0.30, hspace=0.12)
    history_axes = []
    for row in range(nrows):
        history_axes.append(fig.add_subplot(
            grid[row, 0], sharex=history_axes[0] if history_axes else None))
    error_axis = fig.add_subplot(grid[:, 1])

    all_errors = []
    for row, key in enumerate(keys):
        axis = history_axes[row]
        errors, valid_nx, ylabel = [], [], ''
        finest_s = None
        for nx, a, series in entries:
            entry = next((item for item in series if item[0] == key), None)
            if entry is None:
                continue
            _key, ylabel, measured, analytic = entry
            s = np.log(a/p['a_i'])
            axis.plot(s, measured, marker='o', ms=3.0, ls='none',
                      label=rf'$N_x={int(nx)}$')
            errors.append(_history_l1(measured, analytic))
            valid_nx.append(nx)
            finest_s = s
        if finest_s is not None:
            power = next(item[3] for item in _HISTORY_QUANTITIES
                         if item[0] == key)
            axis.plot(scurve,
                      np.abs(curve_theory[key])
                      * (acurve/p['a_i'])**power/p['A_u'],
                      'k-', lw=2.1, label='analytic')
        axis.axhline(0.0, color='0.75', lw=0.7, zorder=0)
        axis.set_ylabel(ylabel)
        if row == nrows-1:
            axis.set_xlabel(r'$s=\ln(a/a_i)$')
        else:
            axis.tick_params(labelbottom=False)   # sharex: one labelled row
        valid_nx = np.asarray(valid_nx, dtype=float)
        errors = np.asarray(errors, dtype=float)
        fitted = _power_law_order(valid_nx, errors, inverse=True)
        suffix = f", p={fitted:.2f}" if np.isfinite(fitted) else ""
        error_axis.loglog(valid_nx, errors, 'o-',
                          label=rf'$\delta${key} history $L_1${suffix}')
        all_errors.extend(errors[np.isfinite(errors) & (errors > 0.0)])
        print(f"cosmowave[convergence-history] {key:<3s}: "
              + " ".join(f"n_x={int(n)} L1={e:.6g}"
                         for n, e in zip(valid_nx, errors)))

    if all_errors:
        anchor = np.full_like(nxs, max(all_errors), dtype=float)
        _plot_power_guide(error_axis, nxs, anchor, 2.0,
                          r'$\propto N_x^{-2}$', inverse=True)
    error_axis.set_xlabel(r'$N_x$')
    error_axis.set_ylabel(r'normalised temporal $L_1$ error')
    error_axis.set_title(r'amplitude-history $L_1$ error')
    error_axis.grid(True, which='both', alpha=0.25)
    title = (f"{_physical_case_name(p)}: amplitude histories and temporal "
             "convergence")
    fig.suptitle(_figure_title(p, title), y=0.985)
    # Two legends, one under each column -- as plot_endpoint_convergence does.
    # Pooling them would leave the resolution keys and the error metrics in one
    # block, hiding which key describes which half of the figure.
    history_handles, history_labels, seen = [], [], set()
    for axis in history_axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label and label not in seen:
                seen.add(label)
                history_handles.append(handle)
                history_labels.append(label)
    error_handles, error_labels = error_axis.get_legend_handles_labels()

    fig.subplots_adjust(top=0.90, bottom=0.25)
    history_box = history_axes[-1].get_position()
    error_box = error_axis.get_position()
    fig.legend(
        history_handles, history_labels, loc='lower center',
        bbox_to_anchor=((history_box.x0+history_box.x1)/2.0, 0.015),
        ncol=min(4, len(history_labels)), frameon=False, fontsize=10.5,
        handlelength=2.5, handletextpad=0.8, columnspacing=1.3)
    fig.legend(
        error_handles, error_labels, loc='lower center',
        bbox_to_anchor=((error_box.x0+error_box.x1)/2.0, 0.015),
        ncol=min(2, len(error_labels)), frameon=False, fontsize=10.5,
        handlelength=2.5, handletextpad=0.8, columnspacing=1.3)
    fig.text(0.5, 0.0,
             r'Amplitudes are $kL=2\pi$ Fourier magnitudes divided by $A_u$, '
             r'with the analytic Hubble decay removed '
             r'($\times(a/a_i)^{1/4}$ for $\delta\rho,\delta B_c$; '
             r'$\times(a/a_i)^{3/4}$ for $\delta u$). The $L_1$ error is the '
             r'mean $|$measured$-$analytic$|$ over the history, normalised by '
             r'the analytic RMS.',
             ha='center', va='bottom', fontsize=9.0,
             bbox=dict(boxstyle='round', fc='white', ec='0.7', lw=0.8))
    finish_figure(fig,
                  save=(f"{save}_convergence_history.png" if save else None))


def plot_spatial_convergence(run_specs, p, save=None):
    """Dispatch the paper-specific Case-0 fit or generic endpoint metric.

    Always emits TWO figures: the case's spatial/paper convergence layout, and
    the temporal amplitude-history L1 that refines the run in a rather than x.
    """
    if len(run_specs) < 2:
        raise ValueError("--convergence needs at least two usable inputs")
    if int(p['case']) == 0:
        plot_standing_alfven_convergence(run_specs, p, save=save)
    else:
        plot_endpoint_convergence(run_specs, p, save=save)
    plot_history_convergence(run_specs, p, save=save)


def plot_fourier_components(results, p, save=None):
    """Scaled complex Fourier components with Berlok's evolving phase visible."""
    if p['case'] not in (1, 3):
        return
    import matplotlib.pyplot as plt
    setup_rcparams()
    series = [('u', C_DU_RE, C_DU_IM,
               p['V_A'] if p['alfven'] else p['V_S'], 0.75,
               r'$\delta u$')]
    if not p['alfven']:
        series.append(('rho', C_DRHO_RE, C_DRHO_IM, p['rhozero'], 0.25,
                       r'$\delta\rho_c$'))
    if p['omega_A'] != 0:
        series.append(('B', C_DB_RE, C_DB_IM, p['Bc0'], 0.25,
                       r'$\delta B_c$'))

    fig, axes = plt.subplots(1, len(series), figsize=(5.4*len(series), 5),
                             squeeze=False)
    axes = axes[0]
    ac = _a_curve(results, p)
    lac = np.log(ac/p['a_i'])
    theory = theory_coefficients(ac, p)
    for axis, (key, re_col, im_col, norm, power, title) in zip(axes, series):
        scale_curve = (ac/p['a_i'])**power
        axis.plot(lac, theory[key].real*scale_curve, 'k-', lw=2.4,
                  label='analytic solution: real')
        axis.plot(lac, theory[key].imag*scale_curve, 'k--', lw=2.4,
                  label='analytic solution: imaginary')
        for label, d in results:
            a = d[:, C_A]
            scale = (a/p['a_i'])**power/norm
            points, = axis.plot(np.log(a/p['a_i']), d[:, re_col]*scale,
                                marker='o', ls='none', label=f'{label}: real')
            axis.plot(np.log(a/p['a_i']), d[:, im_col]*scale,
                      marker='x', ls='none', color=points.get_color(),
                      label=f'{label}: imaginary')
        axis.axhline(0, color='0.75', lw=0.8, zorder=0)
        axis.set_xlabel(r'$\ln(a/a_i)$')
        axis.set_ylabel(f"scaled {title} coefficient")
        axis.set_title(title)
    fig.suptitle(_figure_title(
        p, f"{_physical_case_name(p)}: direct-particle Fourier modes and "
        "analytic phase"))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish_figure_with_legend(
        fig, axes,
        save=(f"{save}_fourier_components.png" if save else None),
        footnote=(r"$Z=A_c-iA_s$ from a volume-weighted particle least-squares "
                  r"fit to $[1,\cos kx,\sin kx]$; no grid interpolation."))


def _series_coeff(d, real_col, imag_col, norm):
    return (d[:, real_col] + 1j*d[:, imag_col]) / norm


def plot_phase(results, p, save=None):
    """Optional targeted traveling-mode phase-error diagnostic."""
    if p['case'] not in (1, 3):
        return
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, d in results:
        a = d[:, C_A]
        th = theory_coefficients(a, p)
        vnorm = p['V_A'] if p['alfven'] else p['V_S']
        series = [
            (r'$\delta u$', _series_coeff(d, C_DU_RE, C_DU_IM, vnorm), th['u']),
            (r'$\delta B_c$', _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0']), th['B']),
        ]
        if not p['alfven']:
            series.append((r'$\delta\rho_c$',
                           _series_coeff(d, C_DRHO_RE, C_DRHO_IM, p['rhozero']),
                           th['rho']))
        for name, measured, expected in series:
            phase_error = np.angle(measured * np.conj(expected))
            ax.plot(np.log(a/p['a_i']), phase_error, marker='o', ls='none',
                    label=f"{name} {lab}")
    ax.axhline(0.0, color='k', lw=1.2)
    ax.set_xlabel(r"$\ln(a/a_i)$")
    ax.set_ylabel("phase error [rad]")
    ax.set_ylim(-np.pi, np.pi)
    fig.suptitle(_figure_title(
        p, f"{_physical_case_name(p)}: traveling-mode phase error"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    finish_figure_with_legend(fig, ax,
                              save=(f"{save}_phase.png" if save else None))


def plot_diagnostics(results, p, save=None):
    """Background magnetic scaling and dimensionless divergence diagnostics."""
    if p['Bc0'] == 0:
        return
    # Case 4 has nothing left to show here: its divergence panel duplicates the
    # third panel of the divclean figure (which also shades the RMS-to-max
    # scatter), and the cleaning-mode isolation panel that used to occupy the
    # other half is reported numerically by compare_divclean_to_theory instead.
    if p['case'] == 4:
        return
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax = np.atleast_1d(ax)
    have_div = False
    run_colours = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', [f'C{i}' for i in range(10)])
    for run_index, (lab, d) in enumerate(results):
        plot_label = "Gasoline" if len(results) == 1 else lab
        colour = run_colours[run_index % len(run_colours)]
        a = d[:, C_A]
        lna = np.log(a/p['a_i'])
        ax[0].plot(lna, d[:, C_BC_BG]/p['Bc0'] - 1.0,
                   marker='o', ls='none', color=colour, label=plot_label)
        if np.any(np.isfinite(d[:, C_DIV_DIM_RMS])):
            have_div = True
            ax[1].plot(lna, d[:, C_DIV_DIM_RMS], color=colour, ls='-',
                       lw=1.8, label=plot_label)
            ax[1].plot(lna, d[:, C_DIV_DIM_MAX], color=colour, ls='--',
                       lw=1.8, label='_nolegend_')
    ax[0].set_xlabel(r"$\ln(a/a_i)$")
    ax[0].axhline(0.0, color='k', lw=1.2)
    ax[0].set_ylabel(r"$\langle B_c\rangle/B_{c,0}-1$")
    ax[0].set_title("homogeneous magnetic expansion")
    ax[1].set_xlabel(r"$\ln(a/a_i)$")
    ax[1].set_ylabel(r"$h|\nabla\!\cdot B|/|B|$")
    ax[1].set_title("dimensionless divergence")
    if have_div:
        ax[1].set_yscale('log')
        ax[1].plot([], [], color='0.2', ls='-', lw=1.8, label='RMS')
        ax[1].plot([], [], color='0.2', ls='--', lw=1.8,
                   label='maximum')
    fig.suptitle(_figure_title(
        p, f"{_physical_case_name(p)}: magnetic-background and divergence "
        "diagnostics"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    finish_figure_with_legend(fig, tuple(ax),
                              save=(f"{save}_diagnostics.png" if save else None))


def _paper_matrix_series(d, p):
    """Signed standing-wave coefficients used in Berlok Figs. 2--5."""
    rho = _series_coeff(d, C_DRHO_RE, C_DRHO_IM, p['rhozero'])
    velocity = _series_coeff(d, C_DU_RE, C_DU_IM, p['V_S'])
    pressure = d[:, C_DP_RE] + 1j*d[:, C_DP_IM]  # already fractional
    magnetic = (None if p['omega_A'] == 0 or p['Bc0'] == 0 else
                _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0']))
    return {'rho': rho, 'u': velocity, 'pressure': pressure, 'B': magnetic}


def plot_berlok_matrix(matrix_results, figure_number, save=None):
    """Reproduce the column layouts of Berlok Figs. 2--5.

    Inputs are ordered hydro, hydro+self-gravity, MHD, MHD+self-gravity.
    Signed coefficients come directly from the volume-weighted particle fit;
    no particle-to-grid interpolation is used.
    """
    import matplotlib.pyplot as plt
    setup_rcparams()
    ncols = len(matrix_results)
    all_rows = (
        ('rho', r'$\delta\rho_c/\rho_c$', 'sine', 0.25),
        ('u', r'$\delta u/\mathcal{V}_s$', 'cosine', 0.75),
        ('pressure', r'$\delta p_c/p_c$', 'sine', 0.25),
        ('B', r'$\delta B_c/B_c$', 'sine', 0.25),
    )
    # Figure 3 is isothermal, so pressure is redundant with density and is not
    # a separate row in the paper.  Figures 2, 4, and 5 contain all four rows.
    rows = tuple(row for row in all_rows
                 if not (int(figure_number) == 3 and row[0] == 'pressure'))
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(4.0*ncols, 2.85*len(rows)),
                             sharex='col', sharey='row', squeeze=False)
    scaled_eds = int(figure_number) == 2

    for col, (label, data, p) in enumerate(matrix_results):
        measured = _paper_matrix_series(data, p)
        a = data[:, C_A]
        acurve = np.geomspace(p['a_i'], max(1.0, float(a.max())), 700)
        theory = theory_coefficients(acurve, p)
        theory_series = {
            'rho': theory['rho'],
            'u': theory['u'],
            'pressure': p['gamma']*theory['rho'],
            'B': theory['B'] if p['omega_A'] != 0 else None,
        }
        axes[0, col].set_title(label)
        for row, (key, ylabel, component, power) in enumerate(rows):
            axis = axes[row, col]
            if key == 'B' and measured[key] is None:
                axis.set_axis_off()
                continue
            if scaled_eds:
                xpoints = np.log(a/p['a_i'])
                xcurve = np.log(acurve/p['a_i'])
                point_scale = (a/p['a_i'])**power/p['A_u']
                curve_scale = (acurve/p['a_i'])**power/p['A_u']
                xlabel = r'$\ln(a/a_i)$'
            else:
                xpoints = a
                xcurve = acurve
                point_scale = np.full_like(a, 1.0/p['A_u'])
                curve_scale = np.full_like(acurve, 1.0/p['A_u'])
                xlabel = r'$a$'
                axis.set_xscale('log')
            axis.plot(xcurve,
                      _spatial_mode_amplitude(theory_series[key], component)
                      * curve_scale,
                      'k-', lw=2.3, label='linear theory')
            axis.plot(xpoints,
                      _spatial_mode_amplitude(measured[key], component)
                      * point_scale,
                      marker='o', ms=3.2, ls='none', alpha=0.8,
                      label='Gasoline')
            axis.axhline(0.0, color='0.72', lw=0.7, zorder=0)
            axis.set_xlabel(xlabel)
            if col == 0:
                prefix = ((r'$(a/a_i)^{%s}\,$' % ('3/4' if key == 'u' else '1/4'))
                          if scaled_eds else '')
                axis.set_ylabel(prefix + ylabel)

    cosmology = {
        2: r'EdS, $\Omega_s=\pi$',
        3: r'EdS, $\Omega_s=2\pi$',
        4: r'EdS, $\Omega_s=\pi/5$',
        5: (r'$\Lambda$CDM $(\Omega_m,\Omega_\Lambda)=(0.3,0.7)$, '
            r'$\Omega_s=\pi/5$'),
    }[int(figure_number)]
    subset = ("; non-self-gravity columns" if ncols == 2 else "")
    fig.suptitle(_figure_title(
        matrix_results[0][2],
        f"Standing compressible waves ({cosmology}{subset})"), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    finish_figure_with_legend(
        fig, axes.ravel(),
        save=(f"{save}_compressible_matrix_fig{figure_number}.png"
              if save else None),
        footnote=(r"Physical cosine/sine amplitudes from signed $kL=2\pi$ "
                  r"Fourier coefficients fitted directly to particles; no grid "
                  r"interpolation. Curves and points are divided by $A_u$."))


def analyze_berlok_matrix(args, params, parser):
    """Analyze the ordered physics branches for paper Figures 2--5."""
    if len(args.inputs) not in (2, 4):
        parser.error("--berlok-figure requires either two non-self-gravity "
                     "inputs ordered hydro, MHD, or all four inputs ordered "
                     "hydro, hydro+self-gravity, MHD, MHD+self-gravity")
    figure_number = int(args.berlok_figure)
    case = {2: 2, 3: 6, 4: 7, 5: 5}[figure_number]
    if len(args.inputs) == 2:
        configs = ((0., 0., 'hydro'), (1., 0., 'MHD'))
    else:
        configs = ((0., 0., 'hydro'), (0., 1., 'hydro + self-gravity'),
                   (1., 0., 'MHD'), (1., 1., 'MHD + self-gravity'))
    matrix_results = []
    endpoint_failed = False
    theory_worst = 0.0
    for inp, (factor_A, factor_G, label) in zip(args.inputs, configs):
        p = case_params(case, factor_A=factor_A, factor_G=factor_G,
                        A_u=params['A_u'])
        data = scan_input(inp, p)
        if data is None:
            parser.error(f"no usable snapshots for {label}: {inp}")
        matrix_results.append((label, data, p))
        afinal = float(data[-1, C_A])
        enderr = abs(afinal - args.final_a)
        print(f"cosmowave[endpoint] Figure {figure_number} {label}: "
              f"final a={afinal:.8g}, target={args.final_a:.8g}, "
              f"|delta|={enderr:.4g}")
        endpoint_failed = endpoint_failed or enderr > args.end_tol
        theory_worst = max(theory_worst, compare_to_theory(data, p, label))
        print_mhd_diagnostics(data, p, label)

    plot_berlok_matrix(matrix_results, figure_number, save=args.save)
    plot_energy_diagnostics([
        (label, inp, p, data)
        for inp, (label, data, p) in zip(args.inputs, matrix_results)
    ], save=args.save)
    if endpoint_failed and not args.allow_incomplete:
        print("cosmowave: ENDPOINT FAILED for Berlok matrix", file=sys.stderr)
        sys.exit(3)
    if args.theory_tol is not None and theory_worst > args.theory_tol:
        print(f"cosmowave: THEORY FAILED -- worst normalized rms "
              f"{theory_worst:.4g} > tol {args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(4)


def analyze_berlok_figure1(args, params, labels, parser):
    """Analyze case 0 using the literal plotting convention of paper Fig. 1."""
    p = case_params(0, factor_A=params['factor_A'], factor_G=0.0,
                    A_u=params['A_u'])
    results = []
    endpoint_failed = False
    theory_worst = 0.0
    for inp, label in zip(args.inputs, labels):
        data = scan_input(inp, p)
        if data is None:
            parser.error(f"no usable snapshots for {label}: {inp}")
        results.append((label, data))
        afinal = float(data[-1, C_A])
        enderr = abs(afinal - args.final_a)
        print(f"cosmowave[endpoint] Figure 1 {label}: final a={afinal:.8g}, "
              f"target={args.final_a:.8g}, |delta|={enderr:.4g}")
        endpoint_failed = endpoint_failed or enderr > args.end_tol
        theory_worst = max(theory_worst, compare_to_theory(data, p, label))
        print_mhd_diagnostics(data, p, label)

    plot_berlok_figure1(results, p, save=args.save)
    plot_energy_diagnostics([
        (label, inp, p, data)
        for inp, (label, data) in zip(args.inputs, results)
    ], save=args.save)
    if endpoint_failed and not args.allow_incomplete:
        print("cosmowave: ENDPOINT FAILED for Berlok Figure 1", file=sys.stderr)
        sys.exit(3)
    if args.theory_tol is not None and theory_worst > args.theory_tol:
        print(f"cosmowave: THEORY FAILED -- worst normalized rms "
              f"{theory_worst:.4g} > tol {args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(4)


def _divclean_run_theory(a, d, p):
    """Evaluate the intended case-4 operator model for one run."""
    h_comoving = None
    if p['clean_damping'] != 0:
        finite_h = d[:, C_H_MEAN][np.isfinite(d[:, C_H_MEAN])]
        if not finite_h.size:
            raise ValueError("damped div-clean analysis requires snapshot h")
        h_comoving = float(np.median(finite_h))
    return divclean_theory_coefficients(a, p, h_comoving=h_comoving)


def plot_divclean(results, p, save=None):
    """Phase-, energy-, and divergence-level validation of comoving GLM."""
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 5))
    for run_index, (lab, d) in enumerate(results):
        first = run_index == 0
        plot_label = "Gasoline" if len(results) == 1 else lab
        a = d[:, C_A]
        lna = np.log(a / p['a_i'])
        zB = _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0'])
        zQ = _series_coeff(d, C_Q_RE, C_Q_IM, p['Bc0'])
        bmode = -zB.imag
        qmode = zQ.real
        points, = ax[0].plot(lna, bmode, marker='o', ms=3.8, ls='none',
                             mfc='none', mew=0.9,
                             label=rf'$B_c$ {plot_label}')
        color = points.get_color()
        ax[0].plot(lna, qmode, marker='x', ms=4.0, ls='none', color=color,
                   label=rf'$Q_c$ {plot_label}')

        acurve = np.geomspace(p['a_i'], max(1.0, float(a.max())), 1200)
        theory = _divclean_run_theory(acurve, d, p)
        stheory = np.log(acurve/p['a_i'])
        # Theory is always BLACK -- a colour no measured marker uses, so it
        # cannot hide underneath them -- and is labelled on the first run only:
        # it is the same operator model for every input, so one legend entry per
        # curve type is all the pooled legend needs.
        tcolor = 'k'
        ax[0].plot(stheory, -theory['B'].imag, '-', color=tcolor, lw=1.8,
                   label=r'$B_c$ theory' if first else '_nolegend_')
        ax[0].plot(stheory, theory['Q'].real, '--', color=tcolor, lw=1.8,
                   label=r'$Q_c$ theory' if first else '_nolegend_')

        measured_energy = (np.abs(zB)**2 + np.abs(zQ)**2)/p['A_u']**2
        theory_energy = ((np.abs(theory['B'])**2 + np.abs(theory['Q'])**2)
                         /p['A_u']**2)
        # Panels 2/3 reuse panel 1's marker/curve convention and its per-run
        # colour, so labelling them again only adds identical-looking entries to
        # the pooled figure legend.
        ax[1].plot(lna, measured_energy, marker='o', ms=3.8, ls='none',
                   mfc='none', mew=0.9, color=color)
        ax[1].plot(stheory, theory_energy, '-', color=tcolor, lw=1.8)

        if np.any(np.isfinite(d[:, C_DIVBC_RMS])):
            divnorm = p['k']*p['Bc0']*p['A_u']/np.sqrt(2.0)
            measured_div = d[:, C_DIVBC_RMS]/divnorm
            expected_div = np.abs(theory['B'])/p['A_u']
            # Shade RMS -> per-particle max. Both normalisations send the
            # continuum sinusoid to 1, so the band width is the particle-level
            # scatter of the divergence operator: it is widest at the nodes,
            # where the continuum vanishes and only discretisation noise is left.
            max_div = d[:, C_DIVBC_MAX]/(p['k']*p['Bc0']*p['A_u'])
            if np.any(np.isfinite(max_div)):
                ax[2].fill_between(lna, measured_div, max_div, color=color,
                                   alpha=0.15, lw=0,
                                   label=('RMS to per-particle max' if first
                                          else '_nolegend_'))
            ax[2].plot(lna, measured_div, marker='o', ms=3.8, ls='none',
                       mfc='none', mew=0.9, color=color)
            ax[2].plot(stheory, expected_div, '-', color=tcolor, lw=1.8)

    for axis in ax:
        axis.axhline(0.0, color='0.75', lw=0.7, zorder=0)
        axis.set_xlabel(r"$s=\ln(a/a_i)$")
    ax[0].set_ylabel(r"mode coefficient / $B_{c,0}$")
    ax[0].set_title(r"signed $B_c$ sine and $Q_c$ cosine modes")
    ax[1].set_ylabel(r"$(|Z_{B_c}|^2+|Z_{Q_c}|^2)/A_u^2$")
    ax[1].set_title("cleaning-mode energy")
    ax[2].set_ylabel(r"$\|\nabla_x\!\cdot B_c\|_2/(kB_{c,0}A_u/\sqrt{2})$")
    ax[2].set_title("SPH divergence operator")
    fig.suptitle(_figure_title(
        p, "Divergence-cleaning wave: mode evolution and SPH divergence"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    finish_figure_with_legend(
        fig, ax,
        save=(f"{save}_divclean.png" if save else None))


def plot_divclean_profile(inputs, labels, results, p, save=None):
    """Case-4 endpoint particles and phase means in separate rows.

    The Fourier panels fit one coefficient to all particles, so they average
    the scatter away and cannot show how noisy an individual particle is.  The
    upper row therefore retains the direct particle values, while the lower row
    shows only their volume-weighted phase means.  Both rows repeat the same
    analytic and phase-rotated analytic curves.  The other wave cases get the
    equivalent two-row picture from ``plot_berlok_endpoint``.
    """
    if int(p['case']) != 4:
        return
    entries = []
    for inp, label, (_lab, d) in zip(inputs, labels, results):
        prof = _endpoint_profile(inp, p)
        if prof is not None and 'B' in prof and 'Q' in prof:
            entries.append((label, inp, prof, d))
    if not entries:
        return

    xcurve = np.linspace(-0.5, 0.5, 800)

    # (b, q) is one rotating oscillator, so the measured state differs from
    # theory by a rotation (phase lag) plus a uniform amplitude factor. Because
    # the two components are 90 degrees apart, that single rotation moves
    # amplitude OUT of one and INTO the other -- which is why B_c can undershoot
    # while Q_c overshoots. Rotating theory to the measured phase separates the
    # two effects: whatever gap is left is pure amplitude.
    _label0, _input0, prof0, d0 = entries[0]
    a_end = float(prof0['a'])
    th_end = _divclean_run_theory(np.array([a_end]), d0, p)
    b_th, q_th = -th_end['B'].imag[0], th_end['Q'].real[0]
    b_me = -_series_coeff(d0, C_DB_RE, C_DB_IM, p['Bc0']).imag[-1]
    q_me = _series_coeff(d0, C_Q_RE, C_Q_IM, p['Bc0']).real[-1]
    amp_th, amp_me = float(np.hypot(b_th, q_th)), float(np.hypot(b_me, q_me))
    phase_th, phase_me = np.arctan2(q_th, b_th), np.arctan2(q_me, b_me)
    # The oscillator angle decreases with time, so a POSITIVE lag means the run
    # has advanced less than theory.
    lag = float((phase_me - phase_th + np.pi) % (2*np.pi) - np.pi)
    amp_ratio = amp_me/amp_th if amp_th > 0 else np.nan
    shifted = {'B': complex(0.0, -amp_th*np.cos(phase_me)),
               'Q': complex(amp_th*np.sin(phase_me), 0.0)}
    print(f"cosmowave[divclean profile] endpoint phase lag vs theory="
          f"{lag:.6g} rad ({np.degrees(lag):.4g} deg); amplitude="
          f"{amp_ratio:.6g} x theory (energy {amp_ratio**2:.6g})")
    quantities = (
        ('B', r'$\delta B_c/B_{c,0}$',
         r'longitudinal $B_c$ (sine mode)'),
        ('Q', r'$Q_c/B_{c,0}$', r'cleaning field $Q_c$ (cosine mode)'),
    )
    prepared_series = []
    for label, input_arg, prof, d in entries:
        plot_label = "Gasoline" if len(entries) == 1 else label
        a = prof['a']
        theory_at_a = _divclean_run_theory(np.array([a]), d, p)
        prepared_values = {}
        for key, _ylabel, _title in quantities:
            values = np.asarray(prof[key], dtype=float)
            # Remove the (small, spurious) zero mode so this matches the signed
            # coefficient the Fourier panels fit.
            values = values - float(np.average(values, weights=prof['volume']))
            scaled = values/p['Bc0']
            expected = _real_mode(prof['x'], theory_at_a[key][0], p)
            rms, mx = _profile_error(scaled, expected, prof['volume'])
            print(f"cosmowave[divclean profile] {plot_label} {key}: "
                  f"normalized rms={rms:.4g} max={mx:.4g}  a={a:.8g}")
            prepared_values[key] = scaled
        prepared_series.append({
            'label': plot_label, 'input_arg': input_arg,
            'x': prof['x'], 'volume': prof['volume'],
            'values': prepared_values,
        })

    analytic_curves = {}
    for key, _ylabel, _title in quantities:
        analytic_curves[key] = (
            (xcurve+0.5, _real_mode(xcurve, th_end[key][0], p),
             {'color': 'k', 'ls': '-', 'lw': 2.5,
              'label': f'analytic at a={a_end:.4g}'}),
            # Both analytic curves are black -- a colour no measured series
            # uses -- and are told apart by linestyle: solid = analytic at the
            # endpoint, dashed = the same curve rotated to the measured phase.
            (xcurve+0.5, _real_mode(xcurve, shifted[key], p),
             {'color': 'k', 'ls': '--', 'lw': 2.2, 'zorder': 6,
              'label': (r'analytic rotated to the measured phase '
                        rf'(lag {np.degrees(lag):+.2f}$^\circ$)')}),
        )
    _plot_particle_phase_rows(
        prepared_series, quantities, analytic_curves, p,
        f"Divergence-cleaning wave: endpoint profiles at a={a_end:.4g}",
        save=(f"{save}_divclean_profile.png" if save else None))


# --------------------------------------------------------------------------- #
#  Source-aware total-energy audit (all cases)
# --------------------------------------------------------------------------- #

ENERGY_LOG_COLUMNS = (
    'dTime', 'z', 'Etot', 'Ekin', 'Epot', 'Eth', 'Emag', 'EClean')


def _cumulative_trapezoid(values, coordinate):
    """Cumulative trapezoidal integral with a zero at the first sample."""
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    if values.shape != coordinate.shape or values.ndim != 1:
        raise ValueError("values and coordinate must be matching 1-D arrays")
    out = np.zeros_like(values)
    if values.size > 1:
        out[1:] = np.cumsum(
            0.5*(values[1:]+values[:-1])*np.diff(coordinate))
    return out


def _cumulative_integral(values, coordinate, quadrature='trapezoid'):
    """Cumulative source integral using the requested output quadrature.

    The regular per-run audit keeps the local trapezoid used historically.
    Timestep convergence uses a cubic spline because its members normally have
    the same root-output cadence: a fixed trapezoid error would otherwise look
    like a false zero-timestep floor.  Degenerate/short grids safely fall back
    to trapezoids.
    """
    if quadrature == 'trapezoid':
        return _cumulative_trapezoid(values, coordinate)
    if quadrature != 'cubic':
        raise ValueError("quadrature must be 'trapezoid' or 'cubic'")
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    if (values.shape != coordinate.shape or values.ndim != 1
            or values.size < 4 or np.any(np.diff(coordinate) <= 0.0)):
        return _cumulative_trapezoid(values, coordinate)
    from scipy.interpolate import CubicSpline
    antiderivative = CubicSpline(coordinate, values).antiderivative()
    return np.asarray(antiderivative(coordinate)-antiderivative(coordinate[0]),
                      dtype=float)


def _energy_scale_factor(arrays, p, recover_legacy=False):
    """Return the scale factor used by the source-work quadrature.

    Modern logs should always use ``a = 1/(1+z)``.  Some pre-B8 Gasoline
    runs, however, printed ``z`` with only five significant digits while
    reporting the old gravity-free gamma=4/3 component sum

        Etot = Ekin + a (Eth + Emag + EClean)

    to full precision.  That rounding is harmless for ordinary plots but is
    comparable to the finest timestep-convergence residual.  In the narrowly
    identifiable old format, recover ``a`` from that algebraic identity.

    The fallback is deliberately opt-in and rejects gravity, other gamma, or
    a candidate differing from logged redshift by more than its old rounding
    envelope.  In particular, B8/work-accumulator totals yield a candidate
    smaller by roughly ``a_i`` and cannot pass this test.
    """
    a_redshift = 1.0/(1.0+arrays['z'])
    if not recover_legacy or not np.isclose(p['gamma'], 4.0/3.0,
                                            rtol=0.0, atol=1.e-13):
        return a_redshift, 'logged redshift'

    energy_scale = max(1.0, float(np.nanmax(np.abs(arrays['Etot']))))
    if np.nanmax(np.abs(arrays['Epot'])) > 1.e-12*energy_scale:
        return a_redshift, 'logged redshift'
    reservoir = arrays['Eth']+arrays['Emag']+arrays['EClean']
    if np.any(~np.isfinite(reservoir)) or np.any(reservoir <= 0.0):
        return a_redshift, 'logged redshift'

    candidate = (arrays['Etot']-arrays['Ekin']-arrays['Epot'])/reservoir
    relative_offset = np.abs(candidate-a_redshift)/np.maximum(
        a_redshift, np.finfo(float).tiny)
    if (np.any(~np.isfinite(candidate)) or np.any(candidate <= 0.0)
            or np.max(relative_offset) <= 5.e-12
            or np.max(relative_offset) > 1.e-4
            or np.any(np.diff(candidate) < -1.e-12)):
        return a_redshift, 'logged redshift'
    return candidate, 'legacy Etot identity'


def cosmowave_energy_budget(log_data, p, h_comoving=None,
                            quadrature='trapezoid', recover_legacy_a=False):
    """Reconstruct the cosmological total-energy balance from a run log.

    The conserved variable is the canonical ("supercomoving") total of Berlok
    (2022) appendix B8 / Pakmor & Springel (2013),

        a^2 E_c = Ekin + a^2 (Eth + Emag + EClean) ,

    with ``E_c`` the total PHYSICAL energy per comoving volume.  ``Ekin`` needs
    no factor: Gasoline stores canonical velocities ``v = a vpec``, so the
    logged kinetic energy is already ``a^2 Tpec``.  ``Eth``, ``Emag`` and
    ``EClean`` are logged as unscaled physical energies and ``Epot`` as
    ``a Uraw``.

    Every component carries the SAME power of ``a``, which is what makes
    exchange between the thermal, kinetic and magnetic reservoirs cancel
    identically.  In return the total is not conserved on its own and the
    expansion work must be subtracted,

        W = integral a[(5-3 gamma) Eth + Emag + EClean] da ,

    whose ``(5-3 gamma)`` coefficient vanishes at gamma = 5/3.  Gravity keeps
    the separate Layzer-Irvine term ``integral(Uraw da)``, reconstructed here
    rather than trusted from ``Etot`` since some builds reset that accumulator
    at every log write.  Damped Case 4 adds back the GLM sink
    ``2 Lambda integral(a EClean da)``.  Case 6 is flagged non-conservative
    because its isothermal thermostat work is not logged.

    The residual is ``delta(a^2 E_c) - Wgrav - Wexp + Wdamp``.  Components are
    additionally returned in per-component invariant scaling for plotting only.
    ``quadrature='cubic'`` is intended for multi-run timestep convergence, where
    all members can share a fixed output cadence even though their force steps
    differ.  ``recover_legacy_a`` is used only for that precision-sensitive
    comparison and recognizes the old gamma=4/3, gravity-free ``Etot`` identity
    described in :func:`_energy_scale_factor`.
    """
    missing = [name for name in ENERGY_LOG_COLUMNS if name not in log_data]
    if missing:
        raise ValueError("energy log is missing columns: " + ", ".join(missing))
    arrays = {name: np.asarray(log_data[name], dtype=float)
              for name in ENERGY_LOG_COLUMNS}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or arrays['dTime'].ndim != 1:
        raise ValueError("energy-log columns must be matching 1-D arrays")

    a = 1.0/(1.0+arrays['z'])
    finite = np.isfinite(a) & (a > 0.0)
    for value in arrays.values():
        finite &= np.isfinite(value)
    if not np.any(finite):
        raise ValueError("energy log contains no complete finite rows")
    arrays = {name: value[finite] for name, value in arrays.items()}
    a = a[finite]
    order = np.argsort(arrays['dTime'], kind='stable')
    arrays = {name: value[order] for name, value in arrays.items()}
    a, scale_factor_source = _energy_scale_factor(
        arrays, p, recover_legacy=recover_legacy_a)

    # Components as PLOTTED: each carries the power of `a` that makes it
    # constant under pure expansion, so panel 1 shows a horizontal line for
    # pure Hubble flow and only genuine exchange moves.  These are for display
    # only -- the conservation residual below is NOT built from them.
    kinetic = arrays['Ekin']
    potential = arrays['Epot']
    thermal = a**(3.0*(p['gamma']-1.0))*arrays['Eth']
    magnetic = a*arrays['Emag']
    cleaning = a*arrays['EClean']

    # Conserved variable: Berlok (2022) appendix B8 / Pakmor & Springel (2013),
    # the canonical ("supercomoving") total a^2 E_c, where E_c is the total
    # PHYSICAL energy per comoving volume,
    #
    #     a^2 E_c = Ekin + a^2 (Eth + Emag + EClean) ,
    #
    # Ekin needing no factor because Gasoline stores canonical velocities
    # v = a vpec, so the logged kinetic energy is already a^2 Tpec.
    #
    # The uniform a^2 weight is the whole point: energy EXCHANGED between the
    # thermal, kinetic and magnetic reservoirs then cancels identically.
    # Weighting each component by the power that makes it individually
    # invariant (as the display components above do) needs no work integral and
    # looks tidier, but is valid only while nothing exchanges energy -- kinetic
    # would carry weight 1 against a^3(gamma-1) for thermal, so heat converted
    # into motion at epoch a shifts the sum by (a^2-a) dE despite physical
    # energy being conserved exactly.  On a settling glass that spurious drift
    # reaches 4e-4 of the total and does NOT converge with timestep.  This was
    # actually shipped for a day; do not "simplify" the residual that way.
    base = (kinetic+potential
            + a*a*(arrays['Eth']+arrays['Emag']+arrays['EClean']))

    # a^2 E_c is not conserved on its own -- expansion does real work on the
    # gas -- so the sources below are subtracted from its increment.
    gravity_integrand = potential/a
    gravity_work = _cumulative_integral(
        gravity_integrand, a, quadrature=quadrature)

    # W = integral a[(5-3 gamma) Eth + Emag + EClean] da.  The (5-3 gamma)
    # factor vanishes at gamma = 5/3, which is why the literature can drop the
    # thermal source; it is 1 for the gamma = 4/3 wave tests.  The integrand is
    # nearly constant in a, so this is accurate to ~1e-6 of the total despite
    # the near-cancellation against delta_base.
    # Split per reservoir: the total integrand is just the sum of the
    # adiabatic drift of each a^2-weighted term, d(a^2 E_i)/da under pure
    # expansion.  Removing each piece from its own component turns the
    # component histories into net NON-adiabatic exchange (see below); their
    # sum is unchanged, so the residual matches the single-integral form this
    # replaces to round-off (measured 4e-16 of full scale).
    thermal_work = _cumulative_integral(
        (5.0-3.0*p['gamma'])*a*arrays['Eth'], a, quadrature=quadrature)
    magnetic_work = _cumulative_integral(
        a*arrays['Emag'], a, quadrature=quadrature)
    cleaning_work = _cumulative_integral(
        a*arrays['EClean'], a, quadrature=quadrature)
    expansion_work = thermal_work+magnetic_work+cleaning_work

    damping_loss = np.zeros_like(a)
    damping_rate = 0.0
    if int(p['case']) == 4 and p['clean_damping'] != 0.0:
        if h_comoving is None or not np.isfinite(h_comoving) or h_comoving <= 0:
            raise ValueError("damped Case-4 energy audit requires h_comoving")
        damping_rate = (p['clean_damping']*p['clean_fch']*p['V_S']
                        /(p['H0']*h_comoving))
        # The GLM sink removes dQ = -2 Lambda Q ds from the comoving invariant
        # Q = a EClean, with s = ln a.  In the a^2 base the cleaning term is
        # a*Q, so the amount removed is 2 Lambda integral a EClean da.  The
        # integrand a*EClean is exactly Q -- smooth and bounded -- so this is
        # well conditioned integrated against `a`.
        damping_loss = _cumulative_integral(
            2.0*damping_rate*a*arrays['EClean'], a,
            quadrature=quadrature)

    source_work = gravity_work+expansion_work-damping_loss
    delta_base = base-base[0]
    residual = delta_base-source_work

    # ``base-Etot`` is the cosmological correction Gasoline actually applied.
    # Builds carrying the in-code accumulator fold BOTH the Layzer-Irvine
    # gravity term and the expansion work into dEcosmo, so the increment should
    # follow gravity_work+expansion_work; older builds reset the accumulator at
    # every log write and leave it at zero.  Either way this stays an
    # independent diagnostic and never feeds the residual.
    logged_gravity = base-arrays['Etot']
    logged_gravity -= logged_gravity[0]
    gravity_log_error = logged_gravity-(gravity_work+expansion_work)

    initial_components = np.asarray([
        kinetic[0], potential[0], thermal[0], magnetic[0], cleaning[0]])
    if int(p['case']) == 6:
        # Isothermal storage artefact: Gasoline represents gamma = 1 with
        # dConstGamma = 1.0000000001, so u = cs^2/(gamma(gamma-1)) is inflated
        # by 1/(gamma_eos-1) = 1e10 and the logged Eth is TEN ORDERS above every
        # physical reservoir (it is 100% of the sum below).  Normalising by it
        # pushed every real exchange curve under _plot_energy_budget's roundoff
        # gate, so the magnetic and thermal histories vanished from the figure.
        # An isothermal bath is a thermostat, not an energy reservoir -- which
        # is why this case is flagged non-conservative -- so scale by the
        # reservoirs that do carry energy.
        initial_components = np.asarray([
            kinetic[0], potential[0], magnetic[0], cleaning[0]])
    full_scale = float(np.sum(np.abs(initial_components)))
    if not np.isfinite(full_scale) or full_scale <= np.finfo(float).tiny:
        full_scale = float(max(np.nanmax(np.abs(base)), 1.0))

    glm_scale = np.nan
    if int(p['case']) == 4:
        # The undamped comoving GLM reservoir is a*(Emag+EClean).  It is the
        # physically sensitive normalization used in the original Case-4
        # diagnosis, whereas ``full_scale`` is dominated by background gas.
        glm_scale = float(a[0]*(arrays['Emag'][0]+arrays['EClean'][0]))
        if not np.isfinite(glm_scale) or glm_scale <= 0.0:
            glm_scale = np.nan

    # Expansion-corrected exchange, in the SAME uniform a^2 weighting as the
    # residual: each reservoir's canonical increment less the reversible work
    # expansion did on it.  What remains is the energy that actually moved
    # between reservoirs -- AV and resistive heating land in `thermal`, the
    # damped wave leaves `kinetic`+`magnetic` -- and by construction the five
    # add up to `residual`.  Only `kinetic` needs no correction: canonical
    # K = a^2 T_pec is already invariant under pure Hubble flow.
    def _increment(values):
        return values-values[0]

    exchange = {
        'kinetic': _increment(kinetic),
        'potential': _increment(potential)-gravity_work,
        'thermal': _increment(a*a*arrays['Eth'])-thermal_work,
        'magnetic': _increment(a*a*arrays['Emag'])-magnetic_work,
        'cleaning': (_increment(a*a*arrays['EClean'])-cleaning_work
                     + damping_loss),
    }
    if abs((5.0-3.0*p['gamma']) - 2.0) < 1.e-12:
        # gamma = 1 (case 6).  There the (5-3*gamma) = 2 expansion work cancels
        # the a^2 drift term for term, and the per-step difference reduces
        # ALGEBRAICALLY to
        #     [a^2 E]_{j+1}-[a^2 E]_j - (a_{j+1}E_{j+1}+a_j E_j)(a_{j+1}-a_j)
        #       = a_j a_{j+1} (E_{j+1}-E_j) .
        # Evaluating that closed form differences CONSECUTIVE Eth instead of two
        # ~1e11 partial sums, so the isothermal thermostat (Eth bit-identical in
        # every log row) gives identically zero rather than the ~2.5 ulp of
        # cancellation noise that would otherwise plot as an oscillating
        # reservoir.  Algebraically equivalent when Eth does vary.
        exchange['thermal'] = np.concatenate(
            ([0.0], np.cumsum(a[:-1]*a[1:]*np.diff(arrays['Eth']))))

    # --- floating-point floor -------------------------------------------------
    # Every curve here is a difference (or a cumulative difference) of terms set
    # by its OWN reservoir, so it cannot resolve anything below the spacing of
    # doubles at that magnitude -- however small the plot's normalisation later
    # makes it look.  A curve built from ~1e11 operands has a resolution of
    # ~1e-5 no matter what.  Snap that band to zero so the figures never show
    # cancellation noise as though it were physics.
    #
    # The floor is 4*N*eps*span: N*eps*span bounds the roundoff a cumulative sum
    # over N log rows can accumulate, and 4 is margin.  Measured against every
    # cosmowave run in test_cases: real exchange curves sit at 1e10-1e15 ulp of
    # their operands and real residuals at 4e11-3e12, while pure cancellation
    # (case 6's isothermal thermal term) sits at 2.5 ulp -- so a floor of a few
    # hundred ulp has seven orders of clearance on both sides.
    def _snap(values, *operands):
        span = max([0.0] + [float(np.nanmax(np.abs(np.asarray(o))))
                            for o in operands if np.size(o)])
        floor = 4.0*len(a)*np.finfo(float).eps*span
        out = np.array(values, dtype=float, copy=True)
        out[np.abs(out) < floor] = 0.0
        return out

    exchange['kinetic'] = _snap(exchange['kinetic'], kinetic)
    exchange['potential'] = _snap(exchange['potential'], potential, gravity_work)
    exchange['thermal'] = _snap(exchange['thermal'],
                                a*a*arrays['Eth'], thermal_work)
    exchange['magnetic'] = _snap(exchange['magnetic'],
                                 a*a*arrays['Emag'], magnetic_work)
    exchange['cleaning'] = _snap(exchange['cleaning'],
                                 a*a*arrays['EClean'], cleaning_work,
                                 damping_loss)
    residual = _snap(residual, base)

    return {
        'time': arrays['dTime'], 'a': a, 'base': base,
        'logged_total': arrays['Etot'], 'delta_base': delta_base,
        'kinetic': kinetic, 'potential': potential, 'thermal': thermal,
        'magnetic': magnetic, 'cleaning': cleaning,
        'exchange': exchange,
        'thermal_work': thermal_work, 'magnetic_work': magnetic_work,
        'cleaning_work': cleaning_work,
        'gravity_work': gravity_work, 'expansion_work': expansion_work,
        'damping_loss': damping_loss, 'source_work': source_work,
        'residual': residual, 'logged_gravity': logged_gravity,
        'gravity_log_error': gravity_log_error,
        'full_scale': full_scale, 'glm_scale': glm_scale,
        'damping_rate': float(damping_rate),
        'scale_factor_source': scale_factor_source,
        'conservative': int(p['case']) != 6,
    }


def _energy_save_path(save, label, index, count):
    if not save:
        return None
    if count == 1:
        return f"{save}_energy.png"
    slug = ''.join(ch if ch.isalnum() else '_' for ch in str(label)).strip('_')
    return f"{save}_energy_{index+1}_{slug or 'run'}.png"


def _plot_energy_budget(budget, p, label, save=None):
    """Plot one run's reservoir changes and invariant-energy residual."""
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 5.0))
    ax_exchange, ax_residual = ax

    a = budget['a']
    if np.ptp(a) > 1.e-10*max(1.0, float(np.max(np.abs(a)))):
        x = np.log(a/a[0])
        xlabel = r'$\ln(a/a_0)$'
    else:
        x = budget['time']-budget['time'][0]
        xlabel = r'$t-t_0$'
    scale = budget['full_scale']

    # Expansion work is removed from each reservoir in the same uniform a^2
    # weighting used by the total-energy residual.  The curves are therefore
    # commensurate: they show actual inter-reservoir exchange and add directly
    # to the closure residual in the second panel.
    component_specs = (
        ('kinetic', 'kinetic', 'C0', '-'),
        ('potential', 'potential', 'C1', (0, (5, 2))),
        ('thermal', 'thermal', 'C2', '--'),
        ('magnetic', 'magnetic', 'C3', '-.'),
        ('cleaning', 'cleaning', 'C4', ':'),
    )

    tiny = 10*np.finfo(float).eps
    exchange = budget['exchange']
    exchange_peaks = []
    for legend_label, key, colour, style in component_specs:
        values = exchange[key]/scale
        if np.any(np.abs(values) > tiny):
            ax_exchange.plot(x, values, color=colour, ls=style,
                             label=legend_label)
            exchange_peaks.append(float(np.max(np.abs(values))))
    ax_exchange.axhline(0.0, color='0.25', lw=1.0, ls='--')
    if len(exchange_peaks) > 1:
        lo, hi = min(exchange_peaks), max(exchange_peaks)
        if lo > 0.0 and hi > 100.0*lo:
            ax_exchange.set_yscale('symlog', linthresh=lo)

    residual = budget['residual']/scale
    ax_residual.plot(x, residual, color='tab:red', lw=1.8,
                     label='_nolegend_')
    ax_residual.axhline(0.0, color='0.25', lw=1.0, ls='--')

    for axis in ax:
        axis.grid(alpha=0.18)
        axis.set_xlabel(xlabel)
    ax_exchange.set_ylabel(r'$(\Delta E_i-W_{{\rm cosmo},i})/E_{scale,0}$')
    ax_exchange.set_title('reservoir contributions to invariant-energy balance')
    ax_residual.set_ylabel(
        r'$(\Delta E_{\rm can}-W_{\rm resolved})/E_{scale,0}$')
    ax_residual.set_title(
        'invariant total-energy error' if budget['conservative']
        else 'total-energy residual (thermostat work unlogged)')
    fig.suptitle(_figure_title(
        p, f"{_physical_case_name(p)}: energy audit"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    finish_figure_with_legend(fig, ax, save=save)


def plot_energy_diagnostics(run_specs, save=None):
    """Run and plot the automatic energy audit for each input.

    ``run_specs`` entries are ``(label, input_arg, params, snapshot_table)``.
    The snapshot table is used only to obtain the Case-4 damping length; all
    energy histories come from the denser per-step log.
    """
    budgets = []
    for index, (label, input_arg, p, snapshots) in enumerate(run_specs):
        log_data = read_log_series(input_arg, ENERGY_LOG_COLUMNS)
        if log_data is None:
            print(f"cosmowave[energy] {label}: no complete energy log; skipped",
                  file=sys.stderr)
            continue
        h_comoving = None
        if int(p['case']) == 4 and p['clean_damping'] != 0.0:
            finite_h = snapshots[:, C_H_MEAN][
                np.isfinite(snapshots[:, C_H_MEAN])]
            if not finite_h.size:
                print(f"cosmowave[energy] {label}: damping length unavailable; "
                      "skipped", file=sys.stderr)
                continue
            h_comoving = float(np.median(finite_h))
        try:
            budget = cosmowave_energy_budget(
                log_data, p, h_comoving=h_comoving)
        except ValueError as error:
            print(f"cosmowave[energy] {label}: {error}; skipped", file=sys.stderr)
            continue
        budgets.append((label, budget))

        normalized = budget['residual']/budget['full_scale']
        status = ('source-closed' if budget['conservative']
                  else 'unclosed isothermal')
        print(f"cosmowave[energy] {label}: {status} residual/full-scale "
              f"final={normalized[-1]:.8g}, "
              f"rms={np.sqrt(np.mean(normalized**2)):.4g}, "
              f"max={np.max(np.abs(normalized)):.4g}; "
              f"Wgrav={budget['gravity_work'][-1]:.8g}, "
              f"Wexp={budget['expansion_work'][-1]:.8g}, "
              f"Wdamp={budget['damping_loss'][-1]:.8g}")
        # Endpoint non-adiabatic exchange per reservoir, same a^2 weighting and
        # normalization as the residual above, so these are directly
        # comparable with each other and with the closure error.
        exchange = budget['exchange']
        moved = {key: values[-1]/budget['full_scale']
                 for key, values in exchange.items()
                 if np.any(values != 0.0)}
        if moved:
            print(f"cosmowave[energy] {label}: expansion-corrected exchange/"
                  "full-scale "
                  + ", ".join(f"{key}={value:.6g}"
                              for key, value in moved.items()))
        if np.isfinite(budget['glm_scale']):
            glm_residual = budget['residual']/budget['glm_scale']
            print(f"cosmowave[energy] {label}: residual/comoving-GLM "
                  f"final={glm_residual[-1]:.8g}, "
                  f"rms={np.sqrt(np.mean(glm_residual**2)):.4g}, "
                  f"max={np.max(np.abs(glm_residual)):.4g}")
        if np.any(np.abs(budget['gravity_work'])
                  > 1.e-14*budget['full_scale']):
            gravity_error = budget['gravity_log_error']/budget['full_scale']
            print(f"cosmowave[energy] {label}: logged dEcosmo tracking "
                  f"final error/full-scale={gravity_error[-1]:.8g}, "
                  f"max={np.max(np.abs(gravity_error)):.4g}")

        _plot_energy_budget(
            budget, p, label,
            save=_energy_save_path(save, label, index, len(run_specs)))
    return budgets


def read_timestep_convergence_metadata(inputs):
    """Read and validate a fixed-resolution, AV-free Courant sequence.

    Artificial viscosity exchanges energy with the externally prescribed
    Hubble background when its full physical pair contraction is used.  That
    source is intentionally outside the ideal Berlok budget, so a clean
    timestep-order test must disable both Gasoline AV coefficients.
    """
    metadata = []
    for input_arg in inputs:
        eta_c = read_log_param(input_arg, 'dEtaCourant', None)
        alpha = read_log_param(input_arg, 'dConstAlpha', None)
        beta = read_log_param(input_arg, 'dConstBeta', None)
        missing = [name for name, value in (
            ('dEtaCourant', eta_c), ('dConstAlpha', alpha),
            ('dConstBeta', beta)) if value is None]
        if missing:
            raise ValueError(
                f"'{input_arg}' is missing logged " + ', '.join(missing))
        if not np.isfinite(eta_c) or eta_c <= 0.0:
            raise ValueError(
                f"'{input_arg}' has invalid dEtaCourant={eta_c!r}")
        if (not np.isfinite(alpha) or not np.isfinite(beta)
                or abs(alpha) > 1.e-14 or abs(beta) > 1.e-14):
            raise ValueError(
                "--convergence-t is an ideal-energy test and requires AV=0: "
                f"'{input_arg}' logs dConstAlpha={alpha:g}, "
                f"dConstBeta={beta:g}; rerun with -alpha 0 -beta 0")
        metadata.append({
            'input': input_arg, 'eta_c': float(eta_c),
            'alpha': float(alpha), 'beta': float(beta),
            'n_x': _cosmowave_resolution(input_arg),
        })

    eta_values = np.asarray([item['eta_c'] for item in metadata], dtype=float)
    if len(np.unique(eta_values)) != len(eta_values):
        raise ValueError(
            "--convergence-t requires one run per distinct dEtaCourant")
    known_resolutions = {item['n_x'] for item in metadata
                         if item['n_x'] is not None}
    if len(known_resolutions) > 1:
        raise ValueError(
            "--convergence-t inputs must have the same n_x; found "
            + ', '.join(str(value) for value in sorted(known_resolutions)))
    return metadata


def _timestep_energy_budget(input_arg, d, p):
    """Cubic-output-quadrature budget for one timestep-convergence member."""
    log_data = read_log_series(input_arg, ENERGY_LOG_COLUMNS)
    if log_data is None:
        return None
    h_comoving = None
    if int(p['case']) == 4 and p['clean_damping'] != 0.0:
        finite_h = d[:, C_H_MEAN][np.isfinite(d[:, C_H_MEAN])]
        if not finite_h.size:
            raise ValueError(
                f"'{input_arg}' has no smoothing length for damped Case 4")
        h_comoving = float(np.median(finite_h))
    return cosmowave_energy_budget(
        log_data, p, h_comoving=h_comoving, quadrature='cubic',
        recover_legacy_a=True)


def _pairwise_order_text(x, y, inverse=False):
    """Compact pairwise convergence orders for console diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fields = []
    for xa, xb, ya, yb in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if min(xa, xb, abs(ya), abs(yb)) <= 0.0:
            continue
        value = np.log(abs(yb/ya))/np.log(xb/xa)
        if inverse:
            value = -value
        fields.append(f"{xa:g}->{xb:g}:{value:.3f}")
    return ', '.join(fields) if fields else 'unavailable'


def plot_timestep_convergence(run_specs, p, save=None):
    """Wave and total-energy error versus Gasoline's dEtaCourant.

    This is the reusable form of the Case-4 ``eta_C^2`` audit.  The endpoint
    particle metric checks the physical solution, while the cubic-quadrature
    Berlok-B8 residual checks energy closure independently.
    """
    if len(run_specs) < 2:
        raise ValueError("--convergence-t needs at least two usable inputs")
    metadata = read_timestep_convergence_metadata(
        [input_arg for _label, input_arg, _params, _d in run_specs])
    entries = []
    for (label, input_arg, _params, d), controls in zip(run_specs, metadata):
        endpoint = _endpoint_convergence_data(input_arg, d, p)
        budget = _timestep_energy_budget(input_arg, d, p)
        entries.append({
            'label': label, 'input': input_arg, 'd': d,
            'eta_c': controls['eta_c'], 'endpoint': endpoint,
            'budget': budget,
        })
    entries.sort(key=lambda item: item['eta_c'])
    eta = np.asarray([item['eta_c'] for item in entries], dtype=float)

    key_order = ('rho', 'u', 'p', 'B', 'Q')
    quantity_keys = [
        key for key in key_order
        if any(item['endpoint'] is not None
               and key in item['endpoint']['quantities'] for item in entries)
    ]
    energy_entries = [item for item in entries
                      if item['budget'] is not None
                      and item['budget']['conservative']]
    have_solution = bool(quantity_keys)
    have_energy = bool(energy_entries)
    if not have_solution and not have_energy:
        raise ValueError("no endpoint profiles or conservative energy logs found")

    import matplotlib.pyplot as plt
    setup_rcparams()
    panel_count = (1 if have_solution else 0)+(2 if have_energy else 0)
    fig, axes = plt.subplots(1, panel_count,
                             figsize=(5.25*panel_count, 5.5), squeeze=False)
    axes = list(axes[0])
    panel = 0

    if have_solution:
        solution_axis = axes[panel]
        panel += 1
        solution_values = []
        for key in quantity_keys:
            xs, errors = [], []
            display = key
            for item in entries:
                endpoint = item['endpoint']
                quantity = (None if endpoint is None else
                            endpoint['quantities'].get(key))
                if quantity is None:
                    continue
                xs.append(item['eta_c'])
                errors.append(quantity['max'])
                display = quantity['display']
                print(f"cosmowave[convergence-t] etaC={item['eta_c']:<9g} "
                      f"{key:<3s}: max={quantity['max']:.7g} "
                      f"L1={quantity['l1']:.7g}")
            xs = np.asarray(xs, dtype=float)
            errors = np.asarray(errors, dtype=float)
            order = _power_law_order(xs, errors)
            order_text = f", p={order:.2f}" if np.isfinite(order) else ""
            solution_axis.loglog(xs, errors, 'o-',
                                 label=f"{display} max{order_text}")
            solution_values.extend(errors[
                np.isfinite(errors) & (errors > 0.0)])
            print(f"cosmowave[convergence-t] {key} pairwise orders: "
                  f"{_pairwise_order_text(xs, errors)}")
        if solution_values:
            guide = np.full_like(eta, max(solution_values), dtype=float)
            _plot_power_guide(solution_axis, eta, guide, 2.0,
                              r'$\propto\eta_C^2$')
        solution_axis.set_ylabel(
            r'$\max_x|q_{\rm sim}-q_{\rm theory}|/A_u$')
        solution_axis.set_title('endpoint wave error')

    if have_energy:
        magnitude_axis = axes[panel]
        signed_axis = axes[panel+1]
        energy_eta = np.asarray([item['eta_c'] for item in energy_entries])
        energy_residual = np.asarray([
            item['budget']['residual'][-1]/item['budget']['full_scale']
            for item in energy_entries
        ])
        magnitude = np.abs(energy_residual)
        magnitude_axis.loglog(energy_eta, magnitude, 'o-', color='tab:red',
                              label='source-corrected total energy')
        negative = energy_residual < 0.0
        if np.any(negative):
            magnitude_axis.plot(energy_eta[negative], magnitude[negative], 'o',
                                mfc='white', color='tab:red', ls='none',
                                label='negative residual')
        signed_axis.plot(energy_eta, energy_residual, 'o-', color='tab:red')
        signed_axis.axhline(0.0, color='0.35', lw=0.9)
        positive_magnitude = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
        if positive_magnitude.size:
            guide = np.full_like(energy_eta, max(positive_magnitude))
            _plot_power_guide(magnitude_axis, energy_eta, guide, 2.0,
                              r'$\propto\eta_C^2$')
            _plot_power_guide(signed_axis, energy_eta, guide, 2.0,
                              r'$\propto\eta_C^2$')
        signed_axis.set_yscale('symlog', linthresh=max(
            1.e-16, float(np.nanmax(magnitude))*1.e-3))
        magnitude_axis.set_ylabel(
            r'$|\Delta(a^2E_c-W)|/E_{\rm scale,0}$')
        signed_axis.set_ylabel(
            r'$\Delta(a^2E_c-W)/E_{\rm scale,0}$')
        magnitude_axis.set_title('energy-residual magnitude\n(hollow = negative)')
        signed_axis.set_title('signed energy residual')
        energy_order = _power_law_order(energy_eta, magnitude)
        print("cosmowave[convergence-t] energy residuals: "
              + ', '.join(f"etaC={x:g}:{y:+.9g}"
                          for x, y in zip(energy_eta, energy_residual)))
        print("cosmowave[convergence-t] energy scale factors: "
              + ', '.join(
                  f"etaC={item['eta_c']:g}:{item['budget']['scale_factor_source']}"
                  for item in energy_entries))
        print(f"cosmowave[convergence-t] energy fitted order={energy_order:.6g}; "
              f"pairwise: {_pairwise_order_text(energy_eta, magnitude)}")
    elif int(p['case']) == 6:
        print("cosmowave[convergence-t] Case 6 has an open isothermal "
              "thermostat budget; energy convergence is omitted.")

    for axis in axes:
        axis.set_xscale('log')
        axis.invert_xaxis()
        axis.set_xlabel(r'dEtaCourant, $\eta_C$  ($\rightarrow$ finer step)')
        axis.grid(True, which='both', alpha=0.25)
    fig.suptitle(_figure_title(
        p, f"{_physical_case_name(p)}: timestep convergence (AV=0)"),
        y=0.98)
    # The magnitude panel has a two-line title; reserve a real title band so it
    # cannot collide with the figure-level heading under tight-bbox saving.
    fig.subplots_adjust(top=0.76, wspace=0.30)
    footnote = (r'Every input is verified to have '
                r'$dConstAlpha=dConstBeta=0$. Source work is cubic-spline '
                r'integrated so fixed root-output quadrature cannot masquerade '
                r'as a timestep floor. $\eta_C^2$ represents second-order '
                r'convergence while the Courant criterion controls the step.')
    if int(p['case']) == 6:
        footnote += (r' Case 6 has an unlogged isothermal thermostat source, '
                     r'so only solution convergence is meaningful.')
    elif (have_energy and any(
            item['budget']['scale_factor_source'] == 'legacy Etot identity'
            for item in energy_entries)):
        footnote += (r' For legacy gravity-free $gamma=4/3$ logs whose '
                     r'redshift was rounded to five significant digits, '
                     r'$a$ is recovered from the old exact $E_{\rm tot}$ '
                     r'component identity.')
    finish_figure_with_legend(
        fig, tuple(axes),
        save=(f"{save}_convergence_t.png" if save else None),
        footnote=footnote)


# --------------------------------------------------------------------------- #
#  Regression baseline (bless / compare the amplitude-vs-a series)
# --------------------------------------------------------------------------- #

def reference_table(inp, p):
    """Regression table {x: a, dv, drho, [dB]} for run `inp`.

    The mode amplitudes (dv, drho, and dB when magnetic) vs the scale factor a
    are the headline Berlok diagnostics, so they make the CI baseline. dB is
    included only when finite for every snapshot (a magnetic run). None if no
    usable snapshots."""
    d = scan_input(inp, p)
    if d is None:
        return None
    table = {"x": d[:, C_A], "dv": d[:, C_DU], "drho": d[:, C_DRHO],
             "du_re": d[:, C_DU_RE], "du_im": d[:, C_DU_IM],
             "dux_re": d[:, C_DUX_RE], "dux_im": d[:, C_DUX_IM],
             "duy_re": d[:, C_DUY_RE], "duy_im": d[:, C_DUY_IM],
             "duz_re": d[:, C_DUZ_RE], "duz_im": d[:, C_DUZ_IM],
             "drho_re": d[:, C_DRHO_RE], "drho_im": d[:, C_DRHO_IM],
             "rho_bg": d[:, C_RHO_BG]}
    if np.all(np.isfinite(d[:, 2])):
        table.update({"dB": d[:, C_DB],
                      "dB_re": d[:, C_DB_RE], "dB_im": d[:, C_DB_IM],
                      "Bc_bg": d[:, C_BC_BG],
                      "divB_dim_rms": d[:, C_DIV_DIM_RMS],
                      "divB_dim_max": d[:, C_DIV_DIM_MAX],
                      "divBc_rms": d[:, C_DIVBC_RMS],
                      "divBc_max": d[:, C_DIVBC_MAX]})
    if np.all(np.isfinite(d[:, C_Q_RE:C_Q_IM+1])):
        table.update({"Q_re": d[:, C_Q_RE], "Q_im": d[:, C_Q_IM],
                      "h_mean": d[:, C_H_MEAN]})
    return table


def compare_to_baseline(inp, p, explicit=None, ref_table=None):
    """Print residuals of the amplitude series vs the sibling baseline; return
    worst RMS (None if no baseline / no series). `ref_table` may be passed
    pre-discovered (the overlay already found the first input's baseline)."""
    if ref_table is None:
        ref_table, _path = find_reference(inp, explicit)
    if ref_table is None:
        return None
    table = reference_table(inp, p)
    if table is None:
        return None
    worst = 0.0
    for key in ("dv", "drho", "dB", "du_re", "du_im",
                "dux_re", "dux_im", "duy_re", "duy_im", "duz_re", "duz_im",
                "drho_re", "drho_im", "dB_re", "dB_im",
                "Bc_bg", "divB_dim_rms", "divB_dim_max",
                "divBc_rms", "divBc_max", "Q_re", "Q_im", "h_mean"):
        res = residuals(table, ref_table, key)
        if res is None:
            continue
        mx, rms = res
        worst = max(worst, rms)
        print(f"cosmowave[regression] {key:<4s} vs baseline: "
              f"max|d|={mx:.4g}  rms={rms:.4g}")
    return worst


def compare_to_theory(d, p, label="run"):
    """Print phase-sensitive normalized residuals versus Berlok theory.

    Normalization is the RMS (or peak fallback) analytic coefficient over the
    sampled scale factors, avoiding singular pointwise relative errors at the
    nodes of standing waves. Returns the worst normalized RMS.
    """
    a = d[:, C_A]
    th = theory_coefficients(a, p)
    vnorm = p['V_A'] if p['alfven'] else p['V_S']
    series = [('u', _series_coeff(d, C_DU_RE, C_DU_IM, vnorm), th['u'])]
    if p['omega_A'] != 0 and p['Bc0'] != 0:
        series.append(('B', _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0']), th['B']))
    if not p['alfven']:
        series.append(('rho', _series_coeff(d, C_DRHO_RE, C_DRHO_IM,
                                            p['rhozero']), th['rho']))
        series.append(('p', d[:, C_DP_RE] + 1j*d[:, C_DP_IM],
                       p['gamma']*th['rho']))

    worst = 0.0
    for name, measured, expected in series:
        finite = (np.isfinite(measured.real) & np.isfinite(measured.imag)
                  & np.isfinite(expected.real) & np.isfinite(expected.imag))
        if not np.any(finite):
            continue
        err = measured[finite] - expected[finite]
        denom_rms = float(np.sqrt(np.mean(np.abs(expected[finite])**2)))
        denom_max = float(np.max(np.abs(expected[finite])))
        if denom_rms == 0 or denom_max == 0:
            continue
        rms = float(np.sqrt(np.mean(np.abs(err)**2)) / denom_rms)
        mx = float(np.max(np.abs(err)) / denom_max)
        worst = max(worst, rms)
        message = (f"cosmowave[theory] {label} {name:<3s}: "
                   f"normalized max={mx:.4g} rms={rms:.4g}")
        if p['case'] in (1, 3):
            good_phase = finite & (np.abs(expected) > 0) & (np.abs(measured) > 0)
            phase = np.angle(measured[good_phase] * np.conj(expected[good_phase]))
            if phase.size:
                message += (f"  phase_rms={np.sqrt(np.mean(phase**2)):.4g} rad"
                            f" phase_max={np.max(np.abs(phase)):.4g} rad")
        print(message)
    return worst


def compare_divclean_to_theory(d, p, label="run"):
    """Validate both halves of the cosmological GLM oscillator.

    Unlike the old diagnostic, this does not accept simple decay of divergence
    as success.  It checks the signed complex B and Q Fourier coefficients, so
    a wrong cosmological propagation factor, phase, variable scaling, or hidden
    pre-cleaning step produces a finite residual.  Residuals are normalized by
    the deliberately injected amplitude A_u rather than by values at oscillator
    nodes.
    """
    a = d[:, C_A]
    theory = _divclean_run_theory(a, d, p)
    measured_B = _series_coeff(d, C_DB_RE, C_DB_IM, p['Bc0'])
    measured_Q = _series_coeff(d, C_Q_RE, C_Q_IM, p['Bc0'])
    worst = 0.0
    for name, measured, expected in (
            ('B_c', measured_B, theory['B']),
            ('Q_c', measured_Q, theory['Q'])):
        finite = (np.isfinite(measured.real) & np.isfinite(measured.imag)
                  & np.isfinite(expected.real) & np.isfinite(expected.imag))
        if not np.any(finite):
            print(f"cosmowave[divclean] {label} {name}: unavailable",
                  file=sys.stderr)
            return np.inf
        error = measured[finite]-expected[finite]
        rms = float(np.sqrt(np.mean(np.abs(error)**2))/p['A_u'])
        mx = float(np.max(np.abs(error))/p['A_u'])
        worst = max(worst, rms)
        print(f"cosmowave[divclean] {label} {name:<3s}: "
              f"max/A_u={mx:.4g} rms/A_u={rms:.4g}")

    measured_energy = (np.abs(measured_B)**2+np.abs(measured_Q)**2)/p['A_u']**2
    expected_energy = ((np.abs(theory['B'])**2+np.abs(theory['Q'])**2)
                       /p['A_u']**2)
    finite_energy = np.isfinite(measured_energy) & np.isfinite(expected_energy)
    if np.any(finite_energy):
        energy_rms = float(np.sqrt(np.mean(
            (measured_energy[finite_energy]-expected_energy[finite_energy])**2)))
        energy_max = float(np.max(np.abs(
            measured_energy[finite_energy]-expected_energy[finite_energy])))
        print(f"cosmowave[divclean] {label} mode energy: "
              f"max={energy_max:.4g} rms={energy_rms:.4g}; "
              f"clock=ln(a) "
              f"theory Omega={theory['omega']:.6g} "
              f"Lambda={theory['decay']:.6g}")

    # The B/Q pair is a phase-space clock.  Compare phase advances from the
    # first available snapshot, avoiding any 2*pi ambiguity caused by an output
    # cadence that need not include a_i.  With damping, b+i*q is not a uniform
    # rotation: compare with the phase of the full damped analytic oscillator.
    bmode = -measured_B.imag
    qmode = measured_Q.real
    oscillator = bmode+1j*qmode
    theory_oscillator = -theory['B'].imag+1j*theory['Q'].real
    finite_phase = (np.isfinite(oscillator.real)
                    & np.isfinite(oscillator.imag)
                    & np.isfinite(theory_oscillator.real)
                    & np.isfinite(theory_oscillator.imag)
                    & (np.abs(oscillator) > 1.e-12*p['A_u'])
                    & (np.abs(theory_oscillator) > 1.e-12*p['A_u']))
    if np.count_nonzero(finite_phase) >= 2:
        phase = _oscillator_phase_advance(oscillator[finite_phase])
        expected_phase = _oscillator_phase_advance(
            theory_oscillator[finite_phase])
        phase_error = phase-expected_phase
        a_phase = a[finite_phase]
        delta_s = np.log(a_phase/a_phase[0])
        delta_a = a_phase-a_phase[0]
        print(f"cosmowave[divclean] {label} phase clock: "
              f"expected=full analytic oscillator, "
              f"rms={np.sqrt(np.mean(phase_error**2)):.6g} rad, "
              f"max={np.max(np.abs(phase_error)):.6g} rad; "
              f"Delta_theta={phase[-1]:.6g} rad, "
              f"theory={expected_phase[-1]:.6g} rad")
        if p['clean_damping'] == 0:
            omega = theory['omega']
            rms_log_clock = float(np.sqrt(np.mean((phase-omega*delta_s)**2)))
            rms_a_clock = float(np.sqrt(np.mean((phase-omega*delta_a)**2)))
            rate_s = float(np.dot(delta_s, phase)/np.dot(delta_s, delta_s))
            rate_a = float(np.dot(delta_a, phase)/np.dot(delta_a, delta_a))
            print(f"cosmowave[divclean] {label} undamped clock audit: "
                  f"rms[Omega*Delta_ln(a)]={rms_log_clock:.6g} rad, "
                  f"rms[Omega*Delta_a]={rms_a_clock:.6g} rad; "
                  f"fit dtheta/dln(a)={rate_s:.6g}, "
                  f"fit dtheta/da={rate_a:.6g}")

    # These should remain zero for the continuum cleaning mode.  Report them as
    # isolation diagnostics without mixing SPH force noise into the GLM gate.
    velocity_modes = np.column_stack((
        _series_coeff(d, C_DUX_RE, C_DUX_IM, p['V_S']),
        _series_coeff(d, C_DUY_RE, C_DUY_IM, p['V_S']),
        _series_coeff(d, C_DUZ_RE, C_DUZ_IM, p['V_S'])))
    velocity_leak = np.max(np.abs(velocity_modes), axis=1)/p['A_u']
    density_leak = d[:, C_DRHO]/(p['rhozero']*p['A_u'])
    print(f"cosmowave[divclean] {label} leakage: "
          f"max_xyz |du_k|/(V_S*A_u)={np.nanmax(velocity_leak):.4g}, "
          f"max |drho|/(rho0*A_u)={np.nanmax(density_leak):.4g}")

    finite_h = d[:, C_H_MEAN][np.isfinite(d[:, C_H_MEAN])]
    if finite_h.size:
        hrel = ((np.max(finite_h)-np.min(finite_h))/np.median(finite_h))
        print(f"cosmowave[divclean] {label} comoving <h>: "
              f"median={np.median(finite_h):.6g}, range/median={hrel:.4g}")
    return worst


def print_mhd_diagnostics(d, p, label="run"):
    if p['Bc0'] == 0:
        return
    bg = d[:, C_BC_BG] / p['Bc0'] - 1.0
    if p['case'] == 4:
        mean_field_leak = d[:, C_BC_BG]/(p['Bc0']*p['A_u'])
        finite_bg = mean_field_leak[np.isfinite(mean_field_leak)]
        if finite_bg.size:
            print(f"cosmowave[mhd] {label} zero-mode magnetic leakage "
                  f"|<Bc>|/(Bc0*A_u): rms={np.sqrt(np.mean(finite_bg**2)):.4g} "
                  f"max={np.max(finite_bg):.4g}")
    else:
        finite_bg = bg[np.isfinite(bg)]
        if finite_bg.size:
            print(f"cosmowave[mhd] {label} <Bc>/Bc0-1: "
                  f"rms={np.sqrt(np.mean(finite_bg**2)):.4g} "
                  f"max={np.max(np.abs(finite_bg)):.4g}")
    if p['case'] == 4:
        rms_norm = p['k']*p['Bc0']*p['A_u']/np.sqrt(2.)
        max_norm = p['k']*p['Bc0']*p['A_u']
        divrms = d[:, C_DIVBC_RMS]/rms_norm
        divmax = d[:, C_DIVBC_MAX]/max_norm
        if np.any(np.isfinite(divrms)):
            print(f"cosmowave[mhd] {label} resolved divergence / injected "
                  f"mode: last_rms={divrms[np.flatnonzero(np.isfinite(divrms))[-1]]:.4g} "
                  f"max_all={np.nanmax(divmax):.4g}")
    else:
        divrms = d[:, C_DIV_DIM_RMS]
        divmax = d[:, C_DIV_DIM_MAX]
        if np.any(np.isfinite(divrms)):
            print(f"cosmowave[mhd] {label} h|divB|/|B|: "
                  f"last_rms={divrms[np.flatnonzero(np.isfinite(divrms))[-1]]:.4g} "
                  f"max_all={np.nanmax(divmax):.4g}")


def main():
    parser = build_parser(PARAM_SPEC,
                          description="Cosmological (M)HD wave analysis "
                                      "(Berlok 2022).")
    parser.add_argument("--berlok-figure", type=int, choices=(1, 2, 3, 4, 5),
                        default=None,
                        help="paper-layout mode: Figure 1 accepts one or more "
                             "case-0 inputs; Figures 2--5 accept either two "
                             "ordered non-self-gravity inputs (hydro, MHD), or "
                             "all four (hydro, hydro+self-gravity, MHD, "
                             "MHD+self-gravity)")
    parser.add_argument("--reg-tol", type=float, default=None,
                        help="CI gate: exit non-zero if the worst RMS residual "
                             "vs the regression baseline exceeds this tolerance")
    parser.add_argument("--theory-tol", type=float, default=None,
                        help="exit non-zero if the worst phase-sensitive normalized "
                             "RMS residual versus Berlok theory exceeds this value")
    parser.add_argument("--final-a", type=float, default=1.0,
                        help="intended final expansion factor (default 1)")
    parser.add_argument("--end-tol", type=float, default=5.e-3,
                        help="absolute tolerance for the final expansion factor")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="analyze a partial run without failing the endpoint check")
    convergence_group = parser.add_mutually_exclusive_group()
    convergence_group.add_argument(
        "--convergence", action="store_true",
        help="treat inputs as a spatial-resolution sequence; Case 0 uses "
             "Berlok Eq. (77) frequency/decay fits, other cases plot the "
             "phase-mean L1 endpoint error versus n_x")
    convergence_group.add_argument(
        "--convergence-t", action="store_true",
        help="treat inputs as a fixed-resolution dEtaCourant sequence and "
             "plot wave/energy error against eta_C^2; requires every "
             "Gasoline log to have dConstAlpha=dConstBeta=0")
    args = parser.parse_args()
    params = params_from_args(args, PARAM_SPEC)
    case = int(params["case"])

    labels = args.labels if args.labels else [
        os.path.basename(os.path.normpath(i)) for i in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error("--labels must match the number of inputs")

    if (args.convergence or args.convergence_t) and len(args.inputs) < 2:
        parser.error("convergence modes require at least two input directories")
    if ((args.convergence or args.convergence_t)
            and args.berlok_figure is not None):
        parser.error("--berlok-figure cannot be combined with a convergence mode")
    if ((args.convergence or args.convergence_t) and args.save_reference):
        parser.error("--save-reference cannot be combined with a convergence mode")
    if args.berlok_figure == 1:
        analyze_berlok_figure1(args, params, labels, parser)
        return
    if args.berlok_figure is not None:
        analyze_berlok_matrix(args, params, parser)
        return

    p = case_params(case, factor_A=params["factor_A"], factor_G=params["factor_G"],
                    A_u=params["A_u"],
                    clean_fch=params["clean_fch"],
                    clean_damping=params["clean_damping"])
    print(f"[cosmowave case {case}] kappa_A={p['kappa_A']:.4g} kappa={p['kappa']:.4g} "
          f"a_i={p['a_i']:.4g} V_A={p['V_A']:.4g} V_S={p['V_S']:.4g} "
          f"Bc0={p['Bc0']:.4g} rhozero={p['rhozero']:.4g} A_u={p['A_u']:.4g}")

    if args.convergence_t:
        try:
            read_timestep_convergence_metadata(args.inputs)
        except ValueError as error:
            parser.error(str(error))

    # --save-reference: bless the FIRST input's amplitude series as the baseline.
    if args.save_reference:
        table = reference_table(args.inputs[0], p)
        if table is None:
            print("cosmowave: no amplitude series to save as reference",
                  file=sys.stderr)
            sys.exit(1)
        save_reference(table, reference_path_for(args.inputs[0]), params,
                       "cosmowave")
        return

    results = []
    energy_specs = []
    for inp, lab in zip(args.inputs, labels):
        d = scan_input(inp, p)
        if d is not None:
            results.append((lab, d))
            energy_specs.append((lab, inp, p, d))
    if not results:
        print("cosmowave: no data to plot; exiting.", file=sys.stderr)
        sys.exit(1)

    # Reference comparison is opt-in (only when --reference is given). Discover
    # the FIRST input's baseline once -- overlaid on the plots and reused for the
    # first input's regression gate.
    ref_table = None
    if args.reference is not None:
        ref_table, _rp = find_reference(args.inputs[0], args.reference)

    try:
        if args.convergence:
            plot_spatial_convergence(energy_specs, p, save=args.save)
        elif args.convergence_t:
            plot_timestep_convergence(energy_specs, p, save=args.save)
        else:
            if case in (0, 1):
                plot_alfven(results, p, save=args.save, ref_table=ref_table)
            elif case in (2, 3, 5, 6, 7):
                plot_compressible(results, p, save=args.save,
                                  ref_table=ref_table)
            else:
                plot_divclean(results, p, save=args.save)
                plot_divclean_profile(args.inputs, labels, results, p,
                                      save=args.save)
            plot_berlok_endpoint(args.inputs, labels, p, save=args.save)
            plot_fourier_components(results, p, save=args.save)
            plot_diagnostics(results, p, save=args.save)
            plot_energy_diagnostics(energy_specs, save=args.save)
    except ValueError as error:
        parser.error(str(error))

    endpoint_failed = False
    theory_worst = 0.0
    for lab, d in results:
        afinal = float(d[-1, C_A])
        enderr = abs(afinal - args.final_a)
        print(f"cosmowave[endpoint] {lab}: final a={afinal:.8g}, "
              f"target={args.final_a:.8g}, |delta|={enderr:.4g}")
        if enderr > args.end_tol:
            endpoint_failed = True
        if case == 4:
            theory_worst = max(theory_worst,
                               compare_divclean_to_theory(d, p, lab))
        else:
            theory_worst = max(theory_worst, compare_to_theory(d, p, lab))
        print_mhd_diagnostics(d, p, lab)

    worst = None
    for i, inp in enumerate(args.inputs):
        w = compare_to_baseline(inp, p, args.reference,
                                ref_table=(ref_table if i == 0 else None))
        if w is not None:
            worst = w if worst is None else max(worst, w)
    if args.reg_tol is not None and worst is not None and worst > args.reg_tol:
        print(f"cosmowave: REGRESSION FAILED -- worst rms {worst:.4g} > tol "
              f"{args.reg_tol:.4g}", file=sys.stderr)
        sys.exit(2)
    if args.theory_tol is not None and theory_worst > args.theory_tol:
        print(f"cosmowave: THEORY FAILED -- worst normalized rms "
              f"{theory_worst:.4g} > tol {args.theory_tol:.4g}", file=sys.stderr)
        sys.exit(3)
    if endpoint_failed and not args.allow_incomplete:
        print("cosmowave: ENDPOINT FAILED -- use --allow-incomplete only for "
              "deliberate partial-run diagnostics", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
