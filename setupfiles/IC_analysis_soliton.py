#!/usr/bin/env python3
"""FDM soliton test (IC_setup_soliton.py); the case (-a) is read from the run name.
  0  static Gaussian core: Q and a_QP vs analytic (IC_analysis_fdm.py).
  1  free packet (the case-0 Gaussian, gravity off) against the free solution, s(t) = sqrt(1 + (t/tau)^2): the radii
     of the spheres holding the Gaussian's mass within sigma and 2 sigma at t = 0 (background mass rho0 c 4/3 pi r^3
     subtracted) / (sigma s, 2 sigma s); the central density (median SPH density, r < 0.4 sigma) / rho0 (s^-3 + c);
     E_kin / E_Q,0 against (t/tau)^2 / (1 + (t/tau)^2), E_Q,0 = (hbar/m)^2/8 int |grad rho|^2 / rho dV of the IC
     profile; and the energy drift (E_kin + E_Q - E_1) / E_Q,0 from the first output.
  2  Schroedinger-Poisson ground state, which must stay in equilibrium. Per snapshot: central density (mean SPH
     density of the 64 densest particles) / rho_c, half-mass radius / that of the truncated profile, core (r < r_c)
     velocity dispersion / v_c(r_c), energy drift (K + W + E_Q + sum m e - E_1) / |W_1| from the first output (a run
     writes no t = 0 snapshot), E_Q = sum m (hbar/m)^2/8 |grad ln rho|^2 (.fdmgrad), e from .fdme (bFDMDiss).
     The central-density oscillation period (least-squares sinusoid on a linear trend over the first 3 periods, runs
     >= 1.5 periods) against the ground state's quasi-normal mode, 92.1 (rho_c / Msun pc^-3)^-1/2 Myr (Schive 2025
     Eq. 81; Veltmaat, Niemeyer & Schwabe 2018 Eq. 17).
  3  head-on collision against the Schroedinger-Poisson reference (soliton_reference.py; phases 0, pi (, pi/2)).
     Per half (x < 0, x >= 0): the density peak (mean position of the 64 densest particles), the centre of mass x
     and the mass m within 1.5 r_c of it (that half only, / one soliton's mass), the peak density (mean SPH density
     of the 64 densest / rho_c); and M10, the mass within 10 r_c of the origin / both solitons' mass. A particle
     code carries no phase, so the right-hand soliton is scored against the phase band (min..max over the
     reference phases and both halves, widened by TOL): the fraction of outputs inside it. An unbound collision is
     phase-independent once the solitons separate (narrow band); a bound one is not. Reference self-check (unbound
     only): the fringe spacing at maximum overlap, from the zero crossings of the phase 0 - phase pi projected
     density (the envelope cancels), against 2 pi hbar / (m v_rel) (Schwabe, Niemeyer & Engels 2016 Eq. 17) at the
     rigid two-soliton overlap speed.
  4  multi-soliton halo against the Schroedinger-Poisson reference (soliton_reference.py --halo; phase realisations
     0, 1, 2). Per snapshot: the peak (mean position of the 64 densest particles), the peak density (their mean SPH
     density / rho_c), m1, m2, m5, m10 (mass within 1, 2, 5, 10 kpc of the peak / the total); the band is min..max
     over the realisations widened by TOL; and the spherically averaged density around the peak over the last
     quarter of the run against the reference band. The core-mass relation: the soliton mass M_sol(r_c=1) (rho_max /
     rho_c)^1/4 (late mean; a tick marks the first-collapse maximum, a line the 1/N no-growth floor) / M against
     Xi = |E| / M^3 / (G m / hbar)^2 of the initial state (runs offset +6 % each in Xi), with the relations of
     Mocz et al. 2017 (Eq. 32, M_c = soliton mass) and Schwabe, Niemeyer & Engels 2016 (Eqs. 18, 19, M_c = mass
     inside r_c, converted to the soliton mass).
  5  the case-2 soliton with tracer stars and hydrostatic gas (-b mass fraction, -c c_s / v_c(r_c), -d gas EOS):
     baryons() -- the case-2 measures of the dark particles; per star shell the radius spread, the phase error against
     the circular orbit and the spread of the specific energy; the gas mass inside 0.5, 1, 2 r_c / analytic and its rms
     velocity / c_s; total momentum; the energy drift over all types (a conservation test only for adiabatic gas: an
     isothermal gas trades PdV work with a heat bath); the rungs per type. Give the DM-only case-2 run of the same
     flags as a second run directory: it fills the dark and total panels.
Cases 3 and 4 also draw the projected mass of the reference and the runs at 4 (case 3) or 5 (case 4) times
(_collision_maps.png, _halo_maps.png); --movie draws it at every reference map time.
Usage: python IC_analysis_soliton.py <run-dir> [<run-dir> ...] [-a 3 -b 10 -c 6] [--labels ...] [--save out] [--movie]
"""
import glob
import os
import re

import numpy as np
import readtipsy as tip

import IC_analysis_fdm as fdm
import soliton_reference as sref
from IC_analysis_framework import Param, standalone_cli
from IC_analysis_general import find_files, setup_rcparams, finish_figure_with_legend
from IC_setup_soliton import (RC, RT, sp_groundstate, myr_per_code, packet_tau, setup_soliton, collision_energy,
                              gas_profile, tracer_stars, STAR_R, NSTAR_SHELL)

def _case3_tag(patterns, default):
    """Run-name rule for -b/-c: the case-3 (or case-4) tag, else the default (cases 0-2 carry no D, v)."""
    rxs = [re.compile(p) for p in patterns]

    def rule(name):
        for rx in rxs:
            m = rx.search(name)
            if m:
                return m.group(1)
        return default
    return rule


PARAM_SPEC = [
    Param("a", "case", 0, int, "test case (0 static core, 1 free packet, 2 ground state, 3 collision, 4 halo, "
                               "5 with baryons)"),
    Param("b", "D", 10.0, float, "case 3: separation / r_c; case 4: number of solitons; case 5: baryon mass fraction",
          _case3_tag([r"case3D(\d+(?:\.\d+)?)v", r"case4N(\d+(?:\.\d+)?)R", r"case5b([\d.]+(?:e[+-]?\d+)?)c"], 10.0)),
    Param("c", "v", 6.0, float, "case 3: relative speed / v_c(r_c); case 4: radius of the centres' sphere / r_c; "
                                "case 5: gas c_s / v_c(r_c)",
          _case3_tag([r"case3D[\d.]+v(\d+(?:\.\d+)?)", r"case4N[\d.]+R(\d+(?:\.\d+)?)",
                      r"case5b[\d.e+-]+c(\d+(?:\.\d+)?)"], 6.0)),
    Param("d", "eos", 0, int, "case 5: gas EOS (0 isothermal, 1 adiabatic; the energy drift is a conservation test "
                              "only for 1)", _case3_tag([r"case5b[\d.e+-]+c[\d.]+eos(\d)"], 0)),
]
NDENSE = 64
TOL = {'x': 0.25, 'm': 0.03, 'rho': 0.10, 'M10': 0.02}   # band widening: kpc, M_sol, relative, 2 M_sol
QUANT = [('x', 'xR', 'x of the right soliton [kpc]'),
         ('m', 'mR', r'its M(<1.5 r$_c$) / M$_{sol}$'),
         ('rho', 'rhoR', r'its peak density / $\rho_c$'),
         ('M10', 'M10', r'M(<10 r$_c$ of origin) / 2 M$_{sol}$')]
PHASE_LS = {0.0: '-', 1.0: '--', 0.5: ':'}


def snapshots(run):
    files = find_files(run)
    if not files:
        raise SystemExit(f'no snapshots found for {run}')
    for fn in files:
        _, d, _, _, t = tip.readtipsy(fn)
        yield fn, float(np.atleast_1d(t)[0]) * myr_per_code(), d.astype(float), \
            tip.readtipsyaux(fn, 'fdmrho')[0].astype(float)


# ---------------------------------------------------------------- case 1
def packet(run):
    from scipy.special import gammainc
    S0 = setup_soliton()
    sig, rho0, cbg, nu = S0.sigma, S0.rho0, S0.c, sp_groundstate()['nu']
    tau = packet_tau(sig, S0.fdm_mass_eV)
    r = np.linspace(0.0, 10 * sig, 200001)
    e = np.exp(-r**2 / (2 * sig**2))
    EQ0 = np.trapz(4 * np.pi * r**2 * nu**2 / 8 * (rho0 * r / sig**2 * e)**2 / (rho0 * (cbg + e)), r)
    Mg = rho0 * (2 * np.pi)**1.5 * sig**3
    fr = [gammainc(1.5, 0.5), gammainc(1.5, 2.0)]          # Gaussian mass fractions within sigma, 2 sigma
    rows = []
    for fn in find_files(run):
        _, d, _, _, t = tip.readtipsy(fn)
        d = d.astype(float)
        n = len(d)
        m, x, v = d[:, 0], d[:, 1:4], d[:, 4:7]
        rr = np.linalg.norm(x, axis=1)
        o = np.argsort(rr)
        Mgr = np.cumsum(m[o]) - rho0 * cbg * 4 / 3 * np.pi * rr[o]**3
        rad = [rr[o][np.searchsorted(np.maximum.accumulate(Mgr), f * Mg)] for f in fr]
        rho = tip.readtipsyaux(fn, 'fdmrho')[0].astype(float)
        g = (tip.readtipsyVecaux(fn, 'fdmgrad')[0].reshape(n, 3).astype(float) if os.path.exists(fn + '.fdmgrad')
             else np.full((n, 3), np.nan))   # codes without .fdmgrad (GIZMO): no E_Q
        tt = float(np.atleast_1d(t)[0]) / tau
        sx = np.sqrt(1 + tt**2)
        K = 0.5 * (m * (v**2).sum(1)).sum()
        EQ = (m * nu**2 / 8 * (g**2).sum(1)).sum()
        rows.append([tt, rad[0] / (sig * sx), rad[1] / (2 * sig * sx),
                     np.median(rho[rr < 0.4 * sig]) / (rho0 * (sx**-3 + cbg)), K / EQ0, (K + EQ) / EQ0])
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    return dict(x=a[:, 0], r1=a[:, 1], r2=a[:, 2], rho_c=a[:, 3], Ekin=a[:, 4], dE=a[:, 5] - a[0, 5])


def plot_packet(inputs, labels, args):
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.0))
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        s = packet(inp)
        c = f'C{j}'
        ax[0].plot(s['x'], s['r1'], 'o-', ms=4, color=c, label=f'{lab}: within $\sigma$')
        ax[0].plot(s['x'], s['r2'], 's--', ms=4, color=c, mfc='none', label=f'{lab}: within 2$\sigma$')
        ax[1].plot(s['x'], s['rho_c'], 'o-', ms=4, color=c, label=lab)
        ax[2].plot(s['x'], s['Ekin'], 'o', ms=4, color=c, label=lab)
        ax[3].plot(s['x'], s['dE'], 'o-', ms=4, color=c, label=lab)
        print(f'{os.path.basename(os.path.normpath(inp))}: at t = {s["x"][-1]:.2f} tau: r(sigma)/an {s["r1"][-1]:.4f}  '
              f'r(2 sigma)/an {s["r2"][-1]:.4f}  rho_c/an {s["rho_c"][-1]:.4f}  E_kin/E_Q0 {s["Ekin"][-1]:.4f} '
              f'(an {s["x"][-1]**2 / (1 + s["x"][-1]**2):.4f})  dE/E_Q0 {s["dE"][-1]:+.4f}')
    tf = np.linspace(0, 1.0, 200)
    for a_ in (ax[0], ax[1]):
        a_.axhline(1.0, color='k', lw=2, label='free packet')
    ax[2].plot(tf, tf**2 / (1 + tf**2), '-', color='k', lw=2, label='free packet')
    ax[3].axhline(0.0, color='k', lw=2, label='conserved')
    for a_, yl in zip(ax, (r'mass-shell radius / ($\sigma\,s(t)$)', r'central density / $\rho_0(s^{-3} + c)$',
                           r'$E_{kin}\,/\,E_{Q,0}$', r'$(E_{kin} + E_Q - E_1)\,/\,E_{Q,0}$')):
        a_.set_xlabel(r'$t/\tau$,  $\tau = 2\sigma^2 m/\hbar$')
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    fig.suptitle(r'FDM free Gaussian packet (soliton case 1, gravity off): $s(t) = \sqrt{1 + (t/\tau)^2}$')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_packet.png' if args.save else None))


# ---------------------------------------------------------------- case 2
def energy(fn, d, nu):
    """(E, W) of a snapshot: K + E_Q + W + sum m e, E_Q = sum m (hbar/m)^2/8 |grad ln rho|^2 (.fdmgrad), e from .fdme."""
    m, v = d[:, 0], d[:, 4:7]
    g = (tip.readtipsyVecaux(fn, 'fdmgrad')[0].reshape(len(d), 3).astype(float) if os.path.exists(fn + '.fdmgrad')
         else np.full((len(d), 3), np.nan))   # codes without .fdmgrad (GIZMO): no E_Q
    e = (m * tip.readtipsyaux(fn, 'fdme')[0]).sum() if os.path.exists(fn + '.fdme') else 0.0
    W = 0.5 * (m * d[:, 8]).sum()
    return 0.5 * (m * (v**2).sum(1)).sum() + (m * nu**2 / 8 * (g**2).sum(1)).sum() + W + e, W


def osc_period(t, y, T0):
    """(period, amplitude) of the dominant oscillation of y(t) over the first 3 T0 (a heating soliton's period drifts
    with rho_c^-1/2): least-squares sinusoid on a linear trend over periods 0.4..2.5 T0; (nan, nan) if the run spans
    less than 1.5 T0."""
    if t[-1] - t[0] < 1.5 * T0:
        return np.nan, np.nan
    w = t <= t[0] + 3 * T0
    t, y = t[w], y[w]
    best = (np.inf, np.nan, np.nan)
    for T in np.linspace(0.4 * T0, min(2.5 * T0, t[-1] - t[0]), 600):
        X = np.c_[np.ones_like(t), t, np.cos(2 * np.pi * t / T), np.sin(2 * np.pi * t / T)]
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        best = min(best, (((X @ c - y)**2).sum(), T, np.hypot(c[2], c[3])))
    return best[1], best[2]


def equilibrium(run):
    S = sp_groundstate()
    Mt = np.interp(RT * RC, S['r'], S['M'])
    r50a = np.interp(0.5 * Mt, S['M'], S['r'])
    rows = []
    for fn, t, d, rho in snapshots(run):
        m, x, v = d[:, 0], d[:, 1:4], d[:, 4:7]
        o = np.argsort(rho)[-NDENSE:]
        r = np.linalg.norm(x - x[o].mean(0), axis=1)
        oo = np.argsort(r)
        r50 = r[oo][np.searchsorted(np.cumsum(m[oo]) / m.sum(), 0.5)]
        core = r < RC
        vc = v[core] - (m[core, None] * v[core]).sum(0) / m[core].sum()
        E, W = energy(fn, d, S['nu'])
        rows.append([t, rho[o].mean() / S['rhoc'], r50 / r50a, np.sqrt((vc**2).sum(1).mean()) / S['vc'], E, W])
    a = np.array(rows)
    return dict(t=a[:, 0], rho_c=a[:, 1], r50=a[:, 2], sigma=a[:, 3], dE=(a[:, 4] - a[0, 4]) / abs(a[0, 5]))


def plot_equilibrium(inputs, labels, args):
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for j, (y0, yl) in enumerate([(1, r'central density / $\rho_c$'), (1, r'r$_{50}$ / r$_{50}$(truncated profile)'),
                                  (0, r'core $\sigma_v$ / v$_c$(r$_c$)'),
                                  (0, r'energy drift (E - E$_1$)/|W$_1$|')]):
        ax[j].axhline(y0, color='k', lw=1.5, label='equilibrium' if j == 0 else None)
        ax[j].set_ylabel(yl)
    T0 = 92.1 * (sp_groundstate()['rhoc'] / 1e9)**-0.5          # Myr, Schive 2025 Eq. 81 (rho_c in Msun/pc^3)
    for k, (inp, lab) in enumerate(zip(inputs, labels)):
        s = equilibrium(inp)
        T, A = osc_period(s['t'], s['rho_c'], T0)
        for j, q in enumerate(('rho_c', 'r50', 'sigma', 'dE')):
            ax[j].plot(s['t'], s[q], 'o-', ms=3, color=f'C{k}',
                       label=(lab + (f': period {T:.0f} Myr' if np.isfinite(T) else '')) if j == 0 else None)
        print(f'{os.path.basename(os.path.normpath(inp))}: t_end {s["t"][-1]:.0f} Myr  rho_c/rho_c,an end '
              f'{s["rho_c"][-1]:.3f} (range {s["rho_c"].min():.3f}..{s["rho_c"].max():.3f})  r50 end {s["r50"][-1]:.3f}  '
              f'sigma_v/v_c end {s["sigma"][-1]:.4f}  dE/|W| end {s["dE"][-1]:+.4f}  rho_c period ' +
              (f'{T:.0f} Myr ({T / T0 - 1:+.1%}, amplitude {A:.3f})' if np.isfinite(T) else 'n/a (run < 1.5 periods)') +
              f' vs {T0:.0f} Myr (Schive 2025 Eq. 81)')
    for a_ in ax:
        a_.set_xlabel('t [Myr]')
        a_.grid(alpha=0.25)
    fig.suptitle('FDM soliton (Schroedinger-Poisson ground state, r$_c$ = 1 kpc): equilibrium; density oscillation '
                 f'period {T0:.0f} Myr (quasi-normal mode, Schive 2025 Eq. 81)')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_equilibrium.png' if args.save else None))


# ---------------------------------------------------------------- case 3
def collision(run):
    S = sp_groundstate()
    Ms = S['M'][-1]
    rows = []
    for fn, t, d, rho in snapshots(run):
        m, x = d[:, 0], d[:, 1:4]
        row = [t, m[np.linalg.norm(x, axis=1) < 10 * RC].sum() / (2 * Ms)]
        for side in (x[:, 0] < 0, x[:, 0] >= 0):
            idx = np.where(side)[0]
            if len(idx) < NDENSE:
                row += [np.nan] * 3
                continue
            o = idx[np.argsort(rho[idx])[-NDENSE:]]
            sel = side & (np.linalg.norm(x - x[o].mean(0), axis=1) < 1.5 * RC)
            ms = m[sel].sum()
            row += [(m[sel] * x[sel, 0]).sum() / ms, ms / Ms, rho[o].mean() / S['rhoc']]
        rows.append(row)
    return dict(zip(['t', 'M10', 'xL', 'mL', 'rhoL', 'xR', 'mR', 'rhoR'], np.array(rows).T))


def load_refs(D, v, refdir):
    """[(phase, table)] of the reference CSVs for this D, v (phase 0 first)."""
    out = []
    for p in glob.glob(os.path.join(refdir, f'collision_D{sref.fmt(D)}_v{sref.fmt(v)}_phase*.csv')):
        r = sref.load(p)
        if r['meta'].get('complete') != '1':
            print(f'soliton: reference {os.path.basename(p)} is incomplete (still running?)')
        out.append((float(r['meta']['phase']), r))
    return sorted(out, key=lambda pr: (pr[0] != 0.0, pr[0]))


def band(refs, q):
    """(t, lo, hi): min..max over the phases and both halves (x mirrored) on the first reference's times."""
    t = refs[0][1]['t_Myr']
    ys = []
    for _, r in refs:
        tt = r['t_Myr']
        if q == 'M10':
            ys.append(np.interp(t, tt, r['M10'], right=np.nan))
            continue
        ys.append(np.interp(t, tt, r[q + 'R'], right=np.nan))
        ys.append(np.interp(t, tt, -r['xL'] if q == 'x' else r[q + 'L'], right=np.nan))
    ys = np.array(ys)
    return t, np.nanmin(ys, 0), np.nanmax(ys, 0)


def score(s, refs):
    """Fraction of the run's outputs (within the reference time span) inside the TOL-widened phase band."""
    out = {}
    for q, key, _ in QUANT:
        tb, lo, hi = band(refs, q)
        ok = s['t'] <= tb[-1]
        lo, hi, y = np.interp(s['t'][ok], tb, lo), np.interp(s['t'][ok], tb, hi), s[key][ok]
        if q == 'rho':
            inside = (y >= lo / (1 + TOL[q])) & (y <= hi * (1 + TOL[q]))
        else:
            inside = (y >= lo - TOL[q]) & (y <= hi + TOL[q])
        out[q] = float(inside.mean()) if inside.size else np.nan
    return out


def plot_collision(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    setup_rcparams()
    D, v = params['D'], params['v']
    refs = load_refs(D, v, args.refdir)
    if not refs:
        print(f'soliton: no reference for D {sref.fmt(D)} v {sref.fmt(v)} in {args.refdir}; generate it with '
              f'python soliton_reference.py -b {sref.fmt(D)} -c {sref.fmt(v)}')
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for j, (q, key, yl) in enumerate(QUANT):
        if refs:
            tb, lo, hi = band(refs, q)
            ax[j].fill_between(tb, lo, hi, color='0.82', lw=0, label='SP phase band')
            for ph, r in refs:
                ax[j].plot(r['t_Myr'], r[key], PHASE_LS.get(ph, '-.'), color='k', lw=1.6,
                           label=f'Schroedinger-Poisson, phase {ph:g}' + r' $\pi$')
        ax[j].set_ylabel(yl)
    for k, (inp, lab) in enumerate(zip(inputs, labels)):
        s = collision(inp)
        for j, (q, key, _) in enumerate(QUANT):
            ax[j].plot(s['t'], s[key], 'o-', ms=3, color=f'C{k}', label=lab)
        sc = score(s, refs) if refs else {}
        print(f'{os.path.basename(os.path.normpath(inp))}: t_end {s["t"][-1]:.0f} Myr  x_R {s["xR"][-1]:+.2f}  '
              f'm_R {s["mR"][-1]:.3f}  rho_R {s["rhoR"][-1]:.3f} (max {np.nanmax(s["rhoR"]):.2f} at '
              f'{s["t"][np.nanargmax(s["rhoR"])]:.0f} Myr)  M10 {s["M10"][-1]:.3f}' +
              ('  in phase band: ' + '  '.join(f'{q} {f:.0%}' for q, f in sc.items()) if sc else ''))
    for ph, r in refs:
        print(f'  reference phase {ph:g} pi: t_end {r["t_Myr"][-1]:.0f} Myr  x_R {r["xR"][-1]:+.2f}  '
              f'm_R {r["mR"][-1]:.3f}  rho_R {r["rhoR"][-1]:.3f} (max {r["rhoR"].max():.2f} at '
              f'{r["t_Myr"][np.argmax(r["rhoR"])]:.0f} Myr)  M10 {r["M10"][-1]:.3f}  M_box {r["M_box"][-1]:.3f}')
    if refs:
        fringe_check(D, v, refs, args)
    ax[2].set_yscale('log')
    for a_ in ax:
        a_.set_xlabel('t [Myr]')
        a_.grid(alpha=0.25)
    fig.suptitle(f'FDM soliton collision, D = {sref.fmt(D)} r$_c$, v$_{{rel}}$ = {sref.fmt(v)} v$_c$: right-hand '
                 'soliton vs the Schroedinger-Poisson reference')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_collision.png' if args.save else None))
    if refs:
        mapfiles = [(f'SP phase {ph:g}' + r'$\pi$', sref.reference_path(D, v, ph, args.refdir)[:-4] + '_maps.npz')
                    for ph, _ in refs]
        plot_collision_maps(inputs, labels, mapfiles, args)
        if args.movie:
            plot_map_movie(inputs, labels, mapfiles, args, 'collision',
                           f'FDM soliton collision, D = {sref.fmt(D)} r$_c$, v$_{{rel}}$ = {sref.fmt(v)} v$_c$')
    elif args.movie:
        print('soliton: --movie draws on the reference _maps.npz grid and times; there is none for this D, v')


def fringe_check(D, v, refs, args):
    """Reference self-check of an unbound collision: the fringe spacing at maximum overlap (zero crossings of the
    phase 0 - phase pi projected density within |y| < r_c, |x| < 3 r_c, where the envelope cancels) against
    2 pi hbar / (m v_rel) (Schwabe et al. 2016 Eq. 17) at the overlap speed of two rigid solitons."""
    S = sp_groundstate()
    if collision_energy(D, v, S) <= 0 or not {0.0, 1.0} <= {ph for ph, _ in refs}:
        return
    nu, M = S['nu'], S['M'][-1]
    W = -np.trapz(S['M'] * S['rho'] * 4 * np.pi * S['r'], S['r'])     # one soliton's self energy
    v0 = v * S['vc']
    vov = np.sqrt(v0**2 - 4 / M * (M**2 / (D * RC) + 2 * W))        # mutual energy -M^2/D -> 2 W at full overlap
    z0, z1 = (np.load(sref.reference_path(D, v, ph, args.refdir)[:-4] + '_maps.npz') for ph in (0.0, 1.0))
    xe, ye = z0['xedges'], z0['yedges']
    xc, band = 0.5 * (xe[1:] + xe[:-1]), np.abs(0.5 * (ye[1:] + ye[:-1])) < RC
    nf = min(len(z0['maps']), len(z1['maps']))
    a, b = z0['maps'][:nf][:, :, band].sum(2), z1['maps'][:nf][:, :, band].sum(2)
    j = int(np.argmax(np.abs(a - b).max(1) / (a + b).max(1)))       # strongest interference
    sel = np.abs(xc) < 3 * RC
    d, x = (a[j] - b[j])[sel], xc[sel]
    i = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    xz = x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])
    lp = 2 * np.pi * nu / vov
    if len(xz) < 3:
        print(f'  reference check: fringes not resolved at t = {z0["t_Myr"][j]:.0f} Myr (2 pi hbar/(m v_rel) = {lp:.2f} kpc)')
        return
    lam = 2 * np.median(np.diff(xz))
    print(f'  reference check: fringe spacing at t = {z0["t_Myr"][j]:.0f} Myr {lam:.3f} kpc vs 2 pi hbar/(m v_rel) '
          f'{lp:.3f} kpc at the overlap speed ({lam / lp - 1:+.1%}; Schwabe et al. 2016 Eq. 17; {2 * np.pi * nu / v0:.3f} '
          'at the entry speed)')


def map_rows(inputs, labels, mapfiles):
    """Projected-mass (x-y) rows on the reference maps' grid: [(label, t_Myr, frame(j) -> map)], the reference
    _maps.npz files [(label, path)] first, then the runs; and the grid edges (xe, ye)."""
    refs = [(lab, np.load(p)) for lab, p in mapfiles]
    xe, ye = refs[0][1]['xedges'], refs[0][1]['yedges']
    rows = [(lab, z['t_Myr'], lambda j, z=z: z['maps'][j]) for lab, z in refs]
    for lab, inp in zip(labels, inputs):
        files = find_files(inp)
        ts = np.array([float(np.atleast_1d(tip.readtipsy(f)[4])[0]) * myr_per_code() for f in files])

        def frame(j, files=files):
            d = tip.readtipsy(files[j])[1]
            return np.histogram2d(d[:, 1], d[:, 2], bins=[xe, ye], weights=d[:, 0])[0]
        rows.append((lab, ts, frame))
    return rows, xe, ye


def nearest(ts, t):
    """Index of the output nearest t, or None if none lies within 0.6 output spacings of it."""
    j = int(np.argmin(np.abs(ts - t)))
    return j if abs(ts[j] - t) <= 0.6 * np.median(np.diff(ts)) + 1e-6 else None


def plot_collision_maps(inputs, labels, mapfiles, args, tag='collision', fracs=(0.18, 0.35, 0.6, 1.0)):
    """Projected mass (x-y, on the reference maps' grid) of the reference phases and the runs at the times fracs x
    the reference's end time (or --map-times); saved as <save>_<tag>_maps.png."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from IC_analysis_general import finish_figure
    rows, xe, ye = map_rows(inputs, labels, mapfiles)
    tm0 = rows[0][1]
    times = args.map_times or [tm0[np.argmin(np.abs(tm0 - f * tm0[-1]))] for f in fracs]
    asp = (ye[-1] - ye[0]) / (xe[-1] - xe[0])
    fig, ax = plt.subplots(len(rows), len(times), figsize=(4.2 * len(times), len(rows) * (4.2 * asp + 0.45) + 0.6),
                           squeeze=False)
    for i, (lab, ts, frame) in enumerate(rows):
        idx = [nearest(ts, t) for t in times]
        Hs = [None if jj is None else frame(jj) for jj in idx]
        vmax = max((H.max() for H in Hs if H is not None), default=1.0)
        for j, (t, jj, H) in enumerate(zip(times, idx, Hs)):
            a = ax[i, j]
            if jj is None:
                a.axis('off')
                a.text(0.5, 0.5, f'{lab}: no output near {t:.0f} Myr', ha='center', va='center', transform=a.transAxes)
                continue
            a.pcolormesh(xe, ye, np.maximum(H.T, 1e-30), cmap='magma', norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
            a.set_aspect('equal')
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f'{lab}, t = {ts[jj]:.0f} Myr', fontsize=9, loc='left')
    fig.suptitle(f'projected mass (x-y, {xe[-1] - xe[0]:.0f} x {ye[-1] - ye[0]:.0f} kpc, {xe[1] - xe[0]:g} kpc pixels, '
                 'log, 3 dex below each row\'s maximum)')
    fig.tight_layout()
    finish_figure(fig, save=(f'{args.save}_{tag}_maps.png' if args.save else None), dpi=110)


def plot_map_movie(inputs, labels, mapfiles, args, tag, title):
    """--movie: the projected mass of the reference realisations and the runs at every reference map time, each
    panel on a fixed log scale (3 dex below its maximum over the movie)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    from IC_analysis_framework import _resolve_movie_writer
    writer, ext = _resolve_movie_writer(args.movie_fps, args.movie_format)
    if writer is None:
        print('soliton: no animation writer (ffmpeg/Pillow) available; no movie')
        return
    rows, xe, ye = map_rows(inputs, labels, mapfiles)
    times = rows[0][1]
    frames = []
    for _, ts, frame in rows:
        frames.append([None if j is None else np.maximum(frame(j).T, 1e-30) for j in (nearest(ts, t) for t in times)])
    keep = [k for k in range(len(times)) if any(fr[k] is not None for fr in frames[len(mapfiles):])]  # a run has it
    times, frames = times[keep], [[fr[k] for k in keep] for fr in frames]
    asp = (ye[-1] - ye[0]) / (xe[-1] - xe[0])
    ncol = 1 if asp < 0.5 else min(len(rows), 4)
    nrow = -(-len(rows) // ncol)
    pw = 12.0 if ncol == 1 else 4.2
    fig, ax = plt.subplots(nrow, ncol, figsize=(pw * ncol, nrow * (pw * asp + 0.45) + 0.7), squeeze=False)
    ax = ax.ravel()
    blank = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    ims, gaps = [], []
    for a, (lab, _, _), fr in zip(ax, rows, frames):
        vmax = max((H.max() for H in fr if H is not None), default=1.0)
        ims.append(a.imshow(blank, extent=(xe[0], xe[-1], ye[0], ye[-1]), origin='lower', cmap='magma',
                            norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), interpolation='nearest'))
        gaps.append(a.text(0.5, 0.5, 'no output', ha='center', va='center', transform=a.transAxes))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(lab, fontsize=9, loc='left')
    for a in ax[len(rows):]:
        a.axis('off')
    sttl = fig.suptitle('')

    def update(k):
        for im, gap, fr in zip(ims, gaps, frames):
            im.set_data(blank if fr[k] is None else fr[k])
            gap.set_visible(fr[k] is None)
        sttl.set_text(f'{title},  t = {times[k]:.0f} Myr\nprojected mass (x-y, {xe[1] - xe[0]:g} kpc pixels, log, '
                      f'3 dex below each panel\'s maximum over the movie)')
        return ims + gaps + [sttl]
    update(0)
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    if args.save:
        out = f'{args.save}_{tag}_movie.{ext}'
        ani.save(out, writer=writer, dpi=110)
        print(f'saved movie -> {out}  ({len(times)} frames @ {args.movie_fps:g} fps)')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------- case 4
HTOL = {'rho_max': 0.10, 'm1': 0.03, 'm2': 0.03, 'm5': 0.03}
HQUANT = [('rho_max', r'peak density / $\rho_c$'), ('m1', r'M(<1 kpc of the peak) / M$_{tot}$'),
          ('m5', r'M(<5 kpc of the peak) / M$_{tot}$')]


def halo(run):
    vol = 4.0 / 3.0 * np.pi * np.diff(sref.HALO_REDGES**3)
    S = sp_groundstate()
    rows, profs, M0, E0 = [], [], None, None
    for fn, t, d, rho in snapshots(run):
        m, x = d[:, 0], d[:, 1:4]
        if M0 is None:
            M0, E0 = m.sum(), energy(fn, d, S['nu'])[0]
        o = np.argsort(rho)[-NDENSE:]
        c = x[o].mean(0)
        r = np.linalg.norm(x - c, axis=1)
        rows.append([t, rho[o].mean() / S['rhoc'], *c] + [m[r < rr].sum() / M0 for rr in sref.HALO_R])
        profs.append(np.histogram(r, sref.HALO_REDGES, weights=m)[0] / vol)
    a = np.array(rows)
    return dict(t=a[:, 0], rho_max=a[:, 1], c=a[:, 2:5], m1=a[:, 5], m2=a[:, 6], m5=a[:, 7], m10=a[:, 8],
                prof=np.array(profs), M0=M0, xi=abs(E0) * S['nu']**2 / M0**3)


def load_halo_refs(N, R, refdir):
    refs = {}
    for p in (0, 1, 2):
        path = sref.halo_path(N, R, p, refdir)
        if os.path.exists(path):
            refs[p] = sref.load(path)
            pz = path[:-4] + '_prof.npz'
            if os.path.exists(pz):
                refs[p]['prof'] = np.load(pz)['rho']
    return refs


def late_profile(t, prof):
    """Mean profile over the last quarter of the run."""
    return prof[t >= t[0] + 0.75 * (t[-1] - t[0])].mean(0)


def soliton_mass(t, rho_max):
    """Core soliton mass / M_sol(r_c = 1 kpc) from the peak density / rho_c (M_sol ~ rho_c^1/4): (late = last-quarter
    mean, first collapse = maximum over the first quarter)."""
    return late_profile(t, rho_max)**0.25, rho_max[t <= t[0] + 0.25 * (t[-1] - t[0])].max()**0.25


def plot_core_relation(a_):
    """M_sol/M against Xi = |E|/M^3/(G m/hbar)^2: Mocz et al. 2017 Eq. 32 (M_c = soliton mass) and Schwabe et al. 2016
    Eqs. 18, 19 (M_c = mass inside r_c -> / M(<r_c)/M_sol; |E|/M^3 in Msun^-2 km^2 s^-2 at their m = 2.5e-22 eV, which
    is Xi (G m/hbar)^2), and the single soliton (E = W/2)."""
    S = sp_groundstate()
    Ms = S['M'][-1]
    xi1 = abs(np.trapz(S['M'] * S['rho'] * 4 * np.pi * S['r'], S['r'])) / 2 * S['nu']**2 / Ms**3
    fc = np.interp(RC, S['r'], S['M']) / Ms
    g2 = (2.5 * 3.085678e16 / (myr_per_code() * 3.15576e13) / S['nu'])**2   # (G m/hbar)^2, m = 2.5e-22 eV, km/s/Msun
    xi = np.geomspace(4e-3, 0.08, 100)
    a_.set_xlim(xi[0], xi[-1])
    a_.plot(xi, 2.6 * xi**(1 / 3), '-', color='k', lw=1.5, label=r'Mocz+2017 Eq. 32: 2.6 $\Xi^{1/3}$ (periodic box)')
    a_.plot(xi, 656 * (xi * g2)**0.25 / fc, '--', color='k', lw=1.2, label='Schwabe+2016 Eq. 18 (binaries, mass ratios)')
    a_.plot(xi, 46.7 * (xi * g2)**(1 / 6) / fc, ':', color='k', lw=1.5, label='Schwabe+2016 Eq. 19 (equal-mass binaries)')
    a_.plot(xi1, 1.0, '*', ms=13, color='k', label='single soliton')
    a_.set_xscale('log')
    a_.set_yscale('log')
    a_.set_xlabel(r'$\Xi = |E|\,/\,M^3\,/\,(Gm/\hbar)^2$ (initial)')
    a_.set_ylabel(r'M$_{sol}$ / M (late peak density)')
    a_.grid(alpha=0.25, which='both')


def plot_halo(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    setup_rcparams()
    N, R = params['D'], params['v']
    refs = load_halo_refs(N, R, args.refdir)
    rc = 0.5 * (sref.HALO_REDGES[1:] + sref.HALO_REDGES[:-1])
    fig, ax = plt.subplots(1, 5, figsize=(26, 5.2))
    plot_core_relation(ax[4])
    ax[4].axhline(1 / N, color='0.45', ls='-.', lw=1.2, label='1/N: no core growth (one progenitor)')
    ax[4].plot([], [], '_', color='k', ms=14, mew=2, label='tick: core at first collapse (max, first quarter)')
    S = sp_groundstate()
    Ms = S['M'][-1]
    refpts = []
    if refs:
        # band only where every realisation has data (an unfinished reference is shorter)
        tr = max((refs[p]['t_Myr'] for p in refs), key=len)
        for a_, (q, _) in zip(ax, HQUANT):
            arr = np.array([np.interp(tr, refs[p]['t_Myr'], refs[p][q], right=np.nan) for p in refs])
            a_.fill_between(tr, arr.min(0), arr.max(0), color='0.8', label='SP realisation band')
            for p in refs:
                a_.plot(refs[p]['t_Myr'], refs[p][q], '-', color='k', lw=1.2, alpha=0.7,
                        label=f'Schroedinger-Poisson, phase set {p}' if q == 'rho_max' else None)
        pr = [late_profile(refs[p]['t_Myr'], refs[p]['prof']) for p in refs if 'prof' in refs[p]]
        if pr:
            pr = np.array(pr)
            ax[3].fill_between(rc, pr.min(0), pr.max(0), color='0.8')
            for q in pr:
                ax[3].plot(rc, q, '-', color='k', lw=1.2, alpha=0.7)
        for p in refs:
            r_ = refs[p]
            xi = abs(r_['K'][0] + r_['W'][0]) * S['nu']**2 / (N * Ms)**3
            ms, m0 = (m / N for m in soliton_mass(r_['t_Myr'], r_['rho_max']))
            refpts.append((p, xi, ms))
            ax[4].plot([xi, xi], [ms, m0], '-', color='k', lw=1, alpha=0.6)
            ax[4].plot(xi, m0, '_', color='k', ms=14, mew=2)
            ax[4].plot(xi, ms, 'osv'[p % 3], ms=8, mfc='none', mec='k', mew=1.5, label=f'SP phase set {p}: core mass')
            print(f'  reference realisation {p}: t_end {r_["t_Myr"][-1]:.0f} Myr  rho_max {r_["rho_max"][-1]:.2f} '
                  f'(late mean {np.mean(r_["rho_max"][r_["t_Myr"] > 0.75 * r_["t_Myr"][-1]]):.2f})  m1 {r_["m1"][-1]:.3f}  '
                  f'm5 {r_["m5"][-1]:.3f}  M_box {r_["M_box"][-1]:.3f}  complete {r_["meta"].get("complete")}  '
                  f'M_sol/M {ms:.3f} (first collapse {m0:.3f}) at Xi {xi:.4f} (Mocz+2017 {2.6 * xi**(1 / 3):.3f})')
    else:
        print(f'no halo reference for N {N:g}, R {R:g} in {args.refdir}')
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        s = halo(inp)
        c = f'C{j}'
        for a_, (q, _) in zip(ax, HQUANT):
            a_.plot(s['t'], s[q], 'o-', ms=3, color=c, label=lab if q == 'rho_max' else None)
        ax[3].plot(rc, late_profile(s['t'], s['prof']), 'o-', ms=3, color=c)
        ms, m0 = (m * Ms / s['M0'] for m in soliton_mass(s['t'], s['rho_max']))
        xd = s['xi'] * 1.06**(j + 1)                      # runs share the IC's Xi: offset for legibility
        ax[4].plot([xd, xd], [ms, m0], '-', color=c, lw=1, alpha=0.6)
        ax[4].plot(xd, m0, '_', color=c, ms=14, mew=2)
        ax[4].plot(xd, ms, 'D', ms=8, color=c)
        if j == 0:
            ax[4].text(0.03, 0.03, r'runs offset +6 % each in $\Xi$', transform=ax[4].transAxes, fontsize=8)
        growth = ''
        if refpts:                                        # core growth above 1/N vs the realisation nearest in Xi
            p, _, mr = min(refpts, key=lambda q: abs(np.log(q[1] / s['xi'])))
            growth = f', core growth {(ms - Ms / s["M0"]) / (mr - 1 / N):.0%} of SP set {p}'
        frac = ''
        if refs:
            for q, _ in HQUANT:
                arr = np.array([np.interp(s['t'], refs[p]['t_Myr'], refs[p][q], right=np.nan) for p in refs])
                cov = np.isfinite(arr).all(0)
                if not cov.any():
                    continue
                arr = arr[:, cov]
                tol = HTOL[q] * (arr.max(0) if q == 'rho_max' else 1.0)
                ok = (s[q][cov] >= arr.min(0) - tol) & (s[q][cov] <= arr.max(0) + tol)
                frac += f'  {q} {100 * ok.mean():.0f}%'
            if refs and cov.sum() < len(cov):
                frac += f'  (scored to {s["t"][cov][-1]:.0f} Myr: references unfinished)' if cov.any() else ''
        late = s['t'] >= 0.75 * s['t'][-1]
        print(f'{os.path.basename(os.path.normpath(inp))}: t_end {s["t"][-1]:.0f} Myr  rho_max {s["rho_max"][-1]:.2f} '
              f'(late mean {s["rho_max"][late].mean():.2f})  m1 {s["m1"][-1]:.3f}  m5 {s["m5"][-1]:.3f}  '
              f'm10 {s["m10"][-1]:.3f}' + (f'  in realisation band:{frac}' if frac else '') +
              f'  M_sol/M {ms:.3f} (first collapse {m0:.3f}) at Xi {s["xi"]:.4f} (Mocz+2017 '
              f'{2.6 * s["xi"]**(1 / 3):.3f}{growth})')
    for a_, (q, yl) in zip(ax, HQUANT):
        a_.set_xlabel('t [Myr]')
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    ax[0].set_yscale('log')
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('r from the peak [kpc]')
    ax[3].set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$], last quarter of the run')
    ax[3].grid(alpha=0.25)
    fig.suptitle(f'FDM multi-soliton halo: {N:g} solitons (r$_c$ = 1 kpc) in a sphere of {R:g} r$_c$, against the '
                 'Schroedinger-Poisson reference')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_halo.png' if args.save else None))
    mapfiles = [(f'SP phase set {p}', sref.halo_path(N, R, p, args.refdir)[:-4] + '_maps.npz') for p in refs]
    mapfiles = [m for m in mapfiles if os.path.exists(m[1])]
    if mapfiles:
        # 0.02: the first output (ChaNGa writes no t = 0 snapshot)
        plot_collision_maps(inputs, labels, mapfiles, args, 'halo', fracs=(0.02, 0.15, 0.3, 0.6, 1.0))
        if args.movie:
            plot_map_movie(inputs, labels, mapfiles, args, 'halo',
                           f'FDM multi-soliton halo, {N:g} solitons in {R:g} r$_c$')
    else:
        print(f'soliton: the maps and --movie draw on the reference _maps.npz grid and times; there is none for '
              f'N {N:g}, R {R:g}')


# ---------------------------------------------------------------- case 5
GAS_R = (0.5, 1.0, 2.0)                                  # rc: radii of the gas mass check


def baryons(run, b, cv):
    """Case 5 per snapshot; a DM-only run (the case-2 reference) gives the dark and total columns only. Dark: the
    case-2 measures. Stars, per shell of STAR_R: std(r / r0), the circular-mean phase error against the circular orbit
    (fraction of an orbit), std(v^2/2 + phi) / v_c^2. Gas: its mass inside GAS_R of its centre of mass / analytic, and
    the rms velocity inside 2 r_c / c_s. All types: |sum m v| / (M v_c) and the energy drift
    (K + W + E_Q + sum m e + E_th - E_1) / |W_1| (E_th = sum m u from the tipsy temperature; constant if isothermal).
    Also the rung counts per type of the last output."""
    S = sp_groundstate()
    Mt = np.interp(RT * RC, S['r'], S['M'])
    r50a = np.interp(0.5 * Mt, S['M'], S['r'])
    cs = cv * S['vc']
    rt, _, Mgt = gas_profile(S, cs, b * Mt)
    xs0, vs0 = tracer_stars(S)
    r0 = np.linalg.norm(xs0, axis=1)
    e1, e2 = xs0 / r0[:, None], vs0 / np.linalg.norm(vs0, axis=1)[:, None]
    om = np.linalg.norm(vs0, axis=1) / r0
    shell = np.repeat(np.arange(len(STAR_R)), NSTAR_SHELL)
    uT = 3.085678e21 * 1.38e-16 / (1.67e-24 * 6.67e-8 * 1.99e33) / (setup_soliton.gamma - 1)  # u per K, as IC_writeascii
    rows, rungs = [], {}
    for fn in find_files(run):
        g, d, s, _, tc = tip.readtipsy(fn)
        g, d, s = (np.asarray(a, float).reshape(-1, w) for a, w in ((g, 12), (d, 9), (s, 11)))
        tc = float(np.atleast_1d(tc)[0])
        ng, nd = len(g), len(d)
        sl = slice(ng, ng + nd)
        rho = tip.readtipsyaux(fn, 'fdmrho')[0].astype(float)[sl]
        grad = tip.readtipsyVecaux(fn, 'fdmgrad')[0].reshape(-1, 3).astype(float)[sl]
        e = tip.readtipsyaux(fn, 'fdme')[0].astype(float)[sl] if os.path.exists(fn + '.fdme') else np.zeros(nd)
        md, xd, vd = d[:, 0], d[:, 1:4], d[:, 4:7]
        o = np.argsort(rho)[-NDENSE:]
        r = np.linalg.norm(xd - xd[o].mean(0), axis=1)
        oo = np.argsort(r)
        core = r < RC
        vcore = vd[core] - (md[core, None] * vd[core]).sum(0) / md[core].sum()
        m = np.concatenate((g[:, 0], md, s[:, 0]))
        v = np.concatenate((g[:, 4:7], vd, s[:, 4:7]))
        W = 0.5 * (m * np.concatenate((g[:, 11], d[:, 8], s[:, 10]))).sum()
        E = (0.5 * (m * (v**2).sum(1)).sum() + W + (md * S['nu']**2 / 8 * (grad**2).sum(1)).sum() + (md * e).sum()
             + uT * (g[:, 0] * g[:, 8]).sum())
        row = [tc * myr_per_code(), rho[o].mean() / S['rhoc'],
               r[oo][np.searchsorted(np.cumsum(md[oo]) / md.sum(), 0.5)] / r50a,
               np.sqrt((vcore**2).sum(1).mean()) / S['vc'], np.linalg.norm((m[:, None] * v).sum(0)) / (m.sum() * S['vc']),
               E, W]
        if len(s) == len(xs0):
            xs, vs = s[:, 1:4], s[:, 4:7]
            rr = np.linalg.norm(xs, axis=1) / r0
            ph = np.exp(1j * (np.arctan2((xs * e2).sum(1), (xs * e1).sum(1)) - om * tc))
            Es = (0.5 * (vs**2).sum(1) + s[:, 10]) / S['vc']**2
            for k in range(len(STAR_R)):
                w = shell == k
                row += [rr[w].std(), np.angle(ph[w].mean()) / (2 * np.pi), Es[w].std()]
        else:
            row += [np.nan] * 3 * len(STAR_R)
        if ng:
            rg = np.linalg.norm(g[:, 1:4] - (g[:, 0, None] * g[:, 1:4]).sum(0) / g[:, 0].sum(), axis=1)
            row += [g[rg < q * RC, 0].sum() / np.interp(q * RC, rt, Mgt) for q in GAS_R]
            w = rg < 2 * RC
            vg = g[w, 4:7] - (g[w, 0, None] * g[w, 4:7]).sum(0) / g[w, 0].sum()
            row.append(np.sqrt((vg**2).sum(1).mean()) / cs)
        else:
            row += [np.nan] * (len(GAS_R) + 1)
        rows.append(row)
        if os.path.exists(fn + '.rung'):
            ru = np.fromfile(fn + '.rung', dtype='>i4')[1:]      # big-endian int32 after the count
            rungs = {k: np.bincount(x) for k, x in (('gas', ru[:ng]), ('dark', ru[sl]), ('star', ru[ng + nd:]))
                     if len(x)}
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    n = len(STAR_R)
    st = a[:, 7:7 + 3 * n].reshape(len(a), n, 3)
    return dict(t=a[:, 0], rho_c=a[:, 1], r50=a[:, 2], sigma=a[:, 3], P=a[:, 4], dE=(a[:, 5] - a[0, 5]) / abs(a[0, 6]),
                star_dr=st[:, :, 0], star_ph=st[:, :, 1], star_E=st[:, :, 2], gas_M=a[:, 7 + 3 * n:-1],
                gas_v=a[:, -1], rungs=rungs)


def plot_baryons(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    setup_rcparams()
    b, cv = params['D'], params['v']
    fig, ax = plt.subplots(2, 5, figsize=(25, 9.5))
    ylab = [r'dark: central density / $\rho_c$', r'dark: r$_{50}$ / r$_{50}$(truncated profile)',
            r'dark: core $\sigma_v$ / v$_c$(r$_c$)', r'all: energy drift (E - E$_1$)/|W$_1$|',
            r'all: |$\Sigma$ m v| / (M v$_c$)', r'stars: std(r / r$_0$)', 'stars: mean phase error [orbits]',
            r'stars: std(v$^2$/2 + $\phi$) / v$_c^2$', 'gas: M(<r) / analytic', r'gas: v$_{rms}$(<2 r$_c$) / c$_s$']
    for a_, yl in zip(ax.flat, ylab):
        a_.set_ylabel(yl)
        a_.set_xlabel('t [Myr]')
        a_.grid(alpha=0.25)
    for a_, y0 in zip(ax.flat, (1, 1, None, 0, None, None, 0, None, 1, None)):
        if y0 is not None:
            a_.axhline(y0, color='k', lw=1)
    ls = ['-', '--', ':', '-.']
    for k, (inp, lab) in enumerate(zip(inputs, labels)):
        s = baryons(inp, b, cv)
        c = f'C{k}'
        for a_, q in zip(ax[0], ('rho_c', 'r50', 'sigma', 'dE', 'P')):
            a_.plot(s['t'], s[q], 'o-', ms=3, color=c, label=lab if q == 'rho_c' else None)
        for j, q in enumerate(('star_dr', 'star_ph', 'star_E')):
            for i, r0 in enumerate(STAR_R):
                ax[1, j].plot(s['t'], s[q][:, i], ls[i], color=c, label=f'r$_0$ = {r0} r$_c$' if k == 0 else None)
        for i, q in enumerate(GAS_R):
            ax[1, 3].plot(s['t'], s['gas_M'][:, i], ls[i], color=c, label=f'r = {q} r$_c$' if k == 0 else None)
        ax[1, 4].plot(s['t'], s['gas_v'], 'o-', ms=3, color=c)
        name = os.path.basename(os.path.normpath(inp))
        print(f'{name}: t_end {s["t"][-1]:.0f} Myr  dark: rho_c end {s["rho_c"][-1]:.3f} (range {s["rho_c"].min():.3f}..'
              f'{s["rho_c"].max():.3f})  r50 end {s["r50"][-1]:.3f}  sigma_v/v_c end {s["sigma"][-1]:.4f}  all: dE/|W| '
              f'end {s["dE"][-1]:+.4f}  |P|/(M v_c) max {s["P"].max():.1e}')
        if np.isfinite(s['star_dr']).all():
            print(f'  stars (r0 {STAR_R} r_c) at the end: std(r/r0) {np.round(s["star_dr"][-1], 4).tolist()}  phase error '
                  f'{np.round(s["star_ph"][-1], 4).tolist()} orbits  E spread {np.round(s["star_E"][-1], 4).tolist()} v_c^2')
        if np.isfinite(s['gas_v']).all():
            dev = np.abs(s['gas_M'] - 1).max(0)
            print(f'  gas: M(<{GAS_R} r_c)/an end {np.round(s["gas_M"][-1], 4).tolist()} (max |dev| over the run '
                  f'{np.round(dev, 4).tolist()})  v_rms/c_s end {s["gas_v"][-1]:.4f} (max {s["gas_v"].max():.4f})')
        print('  rungs of the last output: ' + '  '.join(
            f'{t_} {np.nonzero(n_)[0].min()}..{np.nonzero(n_)[0].max()} (median {int(np.searchsorted(np.cumsum(n_), n_.sum() / 2))})'
            for t_, n_ in s['rungs'].items()))
    fig.suptitle(f'FDM soliton with baryons (case 5): b = {b:g} of the soliton mass in stars and in gas, c$_s$ = {cv:g} '
                 r'v$_c$(r$_c$); stars on circular orbits, gas hydrostatic')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax.flat), save=(f'{args.save}_baryons.png' if args.save else None))


# ---------------------------------------------------------------- driver
def plot(inputs, labels, params, args):
    if params['case'] == 0:
        fdm.plot('soliton', inputs, labels, save=args.save, ref_table=getattr(args, 'ref_table', None))
    elif params['case'] == 1:
        plot_packet(inputs, labels, args)
    elif params['case'] == 2:
        plot_equilibrium(inputs, labels, args)
    elif params['case'] == 3:
        plot_collision(inputs, labels, params, args)
    elif params['case'] == 4:
        plot_halo(inputs, labels, params, args)
    elif params['case'] == 5:
        plot_baryons(inputs, labels, params, args)
    else:
        raise SystemExit(f'soliton case {params["case"]} has no analysis')
    if args.movie and params['case'] not in (3, 4):
        print(f'soliton: --movie is for cases 3 and 4 (projected mass against the reference maps), not case '
              f'{params["case"]}')


def reference_table(inp, params, args):
    if params['case'] == 0:
        return fdm.reference_table('soliton', inp)
    if params['case'] == 1:
        s = packet(inp)
        return {'x': s['x'], 'r_core': s['r1'], 'rho_c': s['rho_c'], 'Ekin': s['Ekin']}
    if params['case'] == 2:
        s = equilibrium(inp)
        return {'x': s['t'], 'rho_c': s['rho_c'], 'r50': s['r50'], 'dE': s['dE']}
    if params['case'] == 4:
        s = halo(inp)
        return {'x': s['t'], 'rho_max': s['rho_max'], 'm1': s['m1'], 'm5': s['m5']}
    if params['case'] == 5:
        s = baryons(inp, params['D'], params['v'])
        return {'x': s['t'], 'rho_c': s['rho_c'], 'r50': s['r50'], 'dE': s['dE']}
    s = collision(inp)
    return {'x': s['t'], 'xR': s['xR'], 'mR': s['mR'], 'rhoR': s['rhoR'], 'M10': s['M10']}


def add_arguments(parser):
    parser.add_argument('--refdir', default=sref.REFDIR, help='case 3 reference directory (default %(default)s)')
    parser.add_argument('--map-times', type=float, nargs='+', default=None,
                        help='case 3 map times in Myr (default: 18, 35, 60, 100 %% of the run)')


if __name__ == "__main__":
    standalone_cli("soliton", PARAM_SPEC, plot, reference_table=reference_table,
                   description="FDM soliton: static core (case 0), free packet (1), ground-state equilibrium (2), "
                               "collision (3), multi-soliton halo (4), soliton with stars and gas (5).",
                   reg_keys=('a_med', 'r_core', 'Ekin', 'rho_c', 'r50', 'mR', 'rhoR', 'rho_max', 'm1'), add_time=False, add_arguments=add_arguments)
