#!/usr/bin/env python3
"""
FDM linear waves (IC_setup_fdmwave.py); the case (-a) is read from the run name.

Mode amplitudes from the particles (equal masses, no density estimate):
    delta(t) = (2/N) sum_j cos(k x_j),   V(t) = (2/N) sum_j v_x,j sin(k x_j).
case 0 (free dispersion) and case 1 (Jeans, oscillating mode n >= 2): rho0 (1 + A cos kx) at rest must oscillate
  as A cos(omega t), omega = (hbar/2m) k^2 (case 0) or omega^2 = (hbar/2m)^2 k^4 - 4 pi G rho0 (case 1).
  w_dyn/w_an     omega from the fit delta(t) = C cos(omega t) over all outputs
  w_stat/w_an    omega from the forces of the first output, omega^2 = k a_s / delta,
                 a_s = (2/N) sum_j a_x,j sin(k x_j) (no time integration; QP only (.aqp) in case 0, total
                 acceleration (.acc2: gravity + QP) in case 1)
  C/A            fitted amplitude against the IC amplitude
  envelope       sqrt(delta^2 + (k V / omega)^2) / C; it stays 1 without damping
case 1, growing mode n = 1: delta = A cosh(gamma t); gamma_dyn from the fit C cosh(gamma t), gamma_stat from
  -k a_s / delta, and delta / (A cosh(gamma_an t)).
case 1, quantum-pressure fraction f, omega^2 = f (hbar/2m)^2 k^4 - 4 pi G rho0 (f = 1 exact; rho0 from the particle
  masses, so any -d N_J works):
  f_stat   (k a_s / delta) / ((hbar/2m)^2 k^4) from the quantum-pressure force alone (.aqp) of the first output
  f_dyn    least-squares fit of C cos(omega t) or C cosh(gamma t) to delta(t) with f free (+- from the residual)
  g_stat   the gravity part (.acc2 - .aqp) of the first output's restoring force / (4 pi G rho0) (1 exact)
  k h      h = (3 N_s / (4 pi n))^(1/3) / 2 at the mean number density n (N_s = nSmooth from the .log; support 2h)
  A scan (runs of several n; pass any -b, e.g. -b 1: each run's n is read from its name) plots delta / A against
  t sqrt(4 pi G rho0) per mode with the f fit, f_dyn and f_stat against k h, and delta / A against (k/k_J)^2 at
  three times common to all runs. The scan and the f fit follow the Jeans instability test of Ida Martine
  Brynjulfsen (her setup: IC_setup_fdmwave -d 4.408).
case 2 (comoving Jeans crossing): delta(a) against the linear FDM ODE (IC_setup_fdmwave.comoving_growth, from
  delta(a_i) = A, delta'(a_i) = 0); the CDM solution (hbar/m = 0) for contrast. Reported: max |delta - ODE| / A
  and delta / ODE at the last output.
case 7 (comoving growing mode): delta(a) against A f(xi(a)) (IC_setup_fdmwave.schive_f, Schive 2025 Fig. 11),
  plotted against xi; each run's n is read from its name, so runs of several modes go into one plot (pass any
  -b, e.g. -b 1, for such a set: it is ignored). Reported: max |delta - A f| / A and its value at the last output.
case 3 (null test, uniform glass): per output the velocity dispersion / (hbar/m / dx) (dx the mean interparticle
  spacing), the density noise std(rho)/mean(rho) (.fdmrho) / its first-output value, E_kin / E_Q,1 and the energy
  drift (K + W + E_Q + sum m e - E_1) / E_Q,1 (E_Q = sum m (hbar/m)^2/8 |grad ln rho|^2 from .fdmgrad; e from .fdme
  with bFDMDiss). A run writes no t = 0 snapshot: the first output is the reference.
case 4 (two interfering waves): theta_j = K x_j - Omega t. Per output the pattern amplitude
  hypot(c, s) / A with c = (2/N) sum cos(theta), s = (2/N) sum sin(theta) (1 if exact), its phase error
  atan2(s, c) / 2 pi in wavelengths, and the rms of v_x - u(theta) in units of (hbar/m) K. Last output: the
  particle histogram in theta and v_x(theta) against the exact rho and u.
case 6 (three interfering waves, theta_j = K x_j): per output the errors of the two density modes,
  |<exp(-i q theta)> - exact| / (exact amplitude: 2/3 for q = 1, 1/3 for q = 2), and of the momentum modes,
  |<v_x exp(-i q theta)> - exact| (q = 1, 2 in quadrature) / ((hbar/m) K); time in T1 = 2 pi / w1. Reference: the same
  modes of the free evolution of the unsigned |1 + 2 cos theta| exp(2 i theta), which has the exact state's rho and u
  at t = 0 (a fluid code cannot tell them apart; it meets the exact modes every T1/4). Last output: the particle
  histogram in theta against both.
case 5 (two-stream): t = time - tau. Per output the width sqrt(2 <x^2>) / (sigma0 sqrt(f)), the velocity slope
  sum(x v_x) / sum(x^2) against t / (tau^2 f), and the kinetic energy / its t = -tau value (exact:
  t^2 / f x 2 / tau^2). Last output: v_x(x) in bins (median, 5-95 %) against the exact x / (2 tau).
Usage: python IC_analysis_fdmwave.py <run-dir> [...] [--labels ...] [--save out]
"""
import os
import re

import numpy as np
import readtipsy as tip

from IC_analysis_framework import Param, standalone_cli
from IC_analysis_general import find_files, setup_rcparams, finish_figure_with_legend, read_log_param
from IC_setup_fdmwave import (L_BOX, A_INIT, NW3_TEND, omega_free, comoving_units, comoving_growth, hbar_over_m_code,
                              setup_fdmwave, node_wave, two_stream, three_wave_modes, A7_INIT, A7_END, schive_xi,
                              schive_f)

PARAM_SPEC = [
    Param("a", "case", 0, int, "test case (0 free dispersion, 1 Jeans, 2 comoving Jeans crossing, 3 null test, 4 two-wave nodes, "
          "5 two-stream, 6 three-wave nodes, 7 comoving growing mode)"),
    Param("b", "mode", 1, int, "mode number n, k = 2 pi n / L",
          re.compile(r"case\d+n(\d+)a")),
    Param("c", "ampl", 3e-2, float, "IC density amplitude A",
          re.compile(r"case\d+n\d+a(\d*\.?\d+(?:e[+-]?\d+)?)")),
]
MASS_EV = setup_fdmwave.fdm_mass_eV


def series(run, mode, accs=()):
    """t (a in comoving runs), delta, V, the mean density, the particle number and, per acceleration file in accs
    that exists, the first output's static omega^2 = k a_s / delta for one run."""
    k = 2.0 * np.pi * mode / L_BOX
    t, dl, V, w2, rho0, n = [], [], [], {}, None, 0
    for fn in find_files(run):
        _, d, _, _, time = tip.readtipsy(fn)
        x, vx = d[:, 1].astype(float), d[:, 4].astype(float)
        c, s = np.cos(k * x), np.sin(k * x)
        t.append(float(np.atleast_1d(time)[0]))
        dl.append(2.0 * c.mean())
        V.append(2.0 * (vx * s).mean())
        if rho0 is None:
            n, rho0 = len(d), d[:, 0].astype(float).sum() / L_BOX**3
            for acc in accs:
                if os.path.exists(f'{fn}.{acc}'):
                    ax = tip.readtipsyVecaux(fn, acc)[0].reshape(n, 3)[:, 0].astype(float)
                    w2[acc] = k * 2.0 * (ax * s).mean() / dl[-1]
    if not t:
        raise SystemExit(f'no snapshots found for {run}')
    return k, np.array(t), np.array(dl), np.array(V), w2, rho0, n


def _fit(t, dl, w0, fun):
    """Least-squares C for each trial rate (0.05..1.5 w0), then the best rate: (rate, C)."""
    ws = w0 * np.linspace(0.05, 1.5, 29001)
    cw = fun(np.outer(ws, t))
    C = (cw @ dl) / (cw * cw).sum(1)
    i = int(np.argmin(((C[:, None] * cw - dl)**2).sum(1)))
    return ws[i], C[i]


def jeans_model(t, f, wq2, wg2):
    """Case 1: cos(omega t) or cosh(gamma t), omega^2 = -gamma^2 = f wq2 - wg2 (f broadcasts against t)."""
    w2 = np.asarray(f * wq2 - wg2, dtype=float)
    rt = np.sqrt(np.abs(w2)) * t
    return np.where(w2 > 0, np.cos(rt), np.cosh(rt))


def _fit_f(t, dl, wq2, wg2):
    """Case 1: least-squares (f, its 1-sigma half-width, C) of delta(t) = C jeans_model(t, f) over f = -0.5..2.5."""
    fs = np.linspace(-0.5, 2.5, 30001)
    cw = jeans_model(t, fs[:, None], wq2, wg2)
    C = (cw @ dl) / (cw * cw).sum(1)
    rss = ((C[:, None] * cw - dl)**2).sum(1)
    i = int(np.argmin(rss))
    inside = fs[rss <= rss[i] * (1.0 + 1.0 / max(len(t) - 2, 1))]
    return fs[i], 0.5 * (inside.max() - inside.min()), C[i]


def _run_mode(run, case, mode):
    """The mode number n in the run's name (case<case>n<n>a...), else mode."""
    m = re.search(rf'case{case}n(\d+)a', _name(run))
    return int(m.group(1)) if m else mode


def measure(run, params):
    case, mode, A = params["case"], params["mode"], params["ampl"]
    if case == 2:
        k, a, dl, _, _, _, _ = series(run, mode)
        _, nu = comoving_units(MASS_EV)
        ode = comoving_growth(k, nu, a, A)
        return dict(a=a, delta=dl, ode=ode, cdm=comoving_growth(k, 0.0, a, A), err=(dl - ode) / A)
    if case == 7:   # the run's own n (one plot holds runs of several modes)
        mode = _run_mode(run, 7, mode)
        k, a, dl, _, _, _, _ = series(run, mode)
        xi = schive_xi(k, comoving_units(MASS_EV)[1], a)
        ref = A * schive_f(xi)
        return dict(a=a, xi=xi, delta=dl, ref=ref, err=(dl - ref) / A, mode=mode)
    if case == 3:
        return null_test(run)
    if case not in (0, 1):
        raise SystemExit(f'fdmwave case {case} has no analysis')
    if case == 1:   # the run's own n (a scan holds runs of several modes)
        mode = _run_mode(run, 1, mode)
    k, t, dl, V, w2s, rho0, npart = series(run, mode, accs=('aqp',) if case == 0 else ('acc2', 'aqp'))
    w2 = w2s.get('aqp' if case == 0 else 'acc2', np.nan)
    nu = hbar_over_m_code(MASS_EV)
    wq2, wg2 = (0.5 * nu)**2 * k**4, 4. * np.pi * rho0
    w2a = wq2 if case == 0 else wq2 - wg2
    if w2a > 0:
        wa = np.sqrt(w2a)
        wd, Cd = _fit(t, dl, wa, np.cos)
        env = np.sqrt(dl**2 + (k * V / wd)**2) / abs(Cd)
        r = dict(t=t, delta=dl, wa=wa, wd=wd, C=Cd, env=env, grow=False,
                 wstat=np.sign(w2) * np.sqrt(abs(w2)), T=2 * np.pi / wa)
    else:
        ga = np.sqrt(-w2a)
        gd, Cd = _fit(t, dl, ga, np.cosh)
        r = dict(t=t, delta=dl, wa=ga, wd=gd, C=Cd, env=dl / (A * np.cosh(ga * t)), grow=True,
                 wstat=np.sign(-w2) * np.sqrt(abs(w2)), T=1.0 / ga)
    if case == 1:
        fd, fe, Cf = _fit_f(t, dl, wq2, wg2)
        nsm = read_log_param(run, 'nSmooth')
        h = 0.5 * (3. * nsm * L_BOX**3 / (4. * np.pi * npart))**(1. / 3.) if nsm else np.nan
        wqp = w2s.get('aqp', np.nan)
        r.update(mode=mode, k=k, kJ=(4. * wg2 / nu**2)**0.25, kh=k * h, wq2=wq2, wg2=wg2, f_stat=wqp / wq2,
                 f_dyn=fd, f_err=fe, C_f=Cf, g_stat=(wqp - w2) / wg2)
    return r


def null_test(run):
    """Case 3 time series (see the module docstring)."""
    nu = hbar_over_m_code(MASS_EV)
    T1 = 2 * np.pi / omega_free(2 * np.pi / L_BOX, MASS_EV)
    rows = []
    for fn in find_files(run):
        _, d, _, _, time = tip.readtipsy(fn)
        d = d.astype(float)
        n = len(d)
        m, v, phi = d[:, 0], d[:, 4:7], d[:, 8]
        dx = L_BOX / n**(1.0 / 3.0)
        rho = tip.readtipsyaux(fn, 'fdmrho')[0].astype(float)
        g = (tip.readtipsyVecaux(fn, 'fdmgrad')[0].reshape(n, 3).astype(float) if os.path.exists(fn + '.fdmgrad')
             else np.full((n, 3), np.nan))   # codes without .fdmgrad (GIZMO): no E_Q
        e = (m * tip.readtipsyaux(fn, 'fdme')[0]).sum() if os.path.exists(fn + '.fdme') else 0.0
        vb = (m[:, None] * v).sum(0) / m.sum()
        K = 0.5 * (m * ((v - vb)**2).sum(1)).sum()
        EQ = (m * nu**2 / 8 * (g**2).sum(1)).sum()
        rows.append([float(np.atleast_1d(time)[0]) / T1, np.sqrt(2 * K / m.sum()) / (nu / dx),
                     rho.std() / rho.mean(), K, EQ, K + EQ + 0.5 * (m * phi).sum() + e])
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    return dict(x=a[:, 0], sigma=a[:, 1], noise=a[:, 2] / a[0, 2], Ekin=a[:, 3] / a[0, 4],
                EQ=a[:, 4] / a[0, 4], dE=(a[:, 5] - a[0, 5]) / a[0, 4])


def node_test(run, mode, A):
    """Case 4 time series and the last output's profile (see the module docstring)."""
    nu = hbar_over_m_code(MASS_EV)
    K = 2 * np.pi * mode / L_BOX
    Om = 3 * omega_free(K, MASS_EV)
    rows, last = [], None
    for fn in find_files(run):
        _, d, _, _, time = tip.readtipsy(fn)
        t = float(np.atleast_1d(time)[0])
        th = K * d[:, 1].astype(float) - Om * t
        vx = d[:, 4].astype(float)
        c, s = 2 * np.cos(th).mean(), 2 * np.sin(th).mean()
        rows.append([t * Om / (2 * np.pi), np.hypot(c, s) / A, np.arctan2(s, c) / (2 * np.pi),
                     np.sqrt(((vx - node_wave(th, A, K, nu)[1])**2).mean()) / (nu * K)])
        last = (th, vx)
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    th = (last[0] + np.pi) % (2 * np.pi) - np.pi
    nb = 48
    ib = np.minimum(((th + np.pi) / (2 * np.pi) * nb).astype(int), nb - 1)
    cnt = np.bincount(ib, minlength=nb)
    vb = np.bincount(ib, weights=last[1], minlength=nb) / np.maximum(cnt, 1)
    vb[cnt == 0] = np.nan
    return dict(x=a[:, 0], amp=a[:, 1], phase=a[:, 2], verr=a[:, 3], thb=(np.arange(nb) + 0.5) / nb - 0.5,
                rhob=cnt * nb / len(th), vb=vb / (nu * K))


_TH = 2 * np.pi * np.arange(1024) / 1024


def three_unsigned_rho(w1t):
    """Case 6 reference: rho / rho0 on _TH (theta = K x) of the free evolution of the unsigned
    |1 + 2 cos theta| exp(2 i theta) (component q K evolves as exp(-i q^2 w1 t))."""
    q = np.fft.fftfreq(_TH.size, 1.0 / _TH.size)
    psi = np.fft.ifft(np.fft.fft(np.abs(1 + 2 * np.cos(_TH)) * np.exp(2j * _TH)) * np.exp(-1j * q * q * w1t))
    rho = np.abs(psi)**2
    return rho / rho.mean()


def three_exact_rho(w1t):
    """Case 6: exact rho / rho0 on _TH."""
    return np.abs(sum(np.exp(1j * (m * _TH - m * m * w1t)) for m in (1, 2, 3)))**2 / 3.0


def three_wave_test(run, mode):
    """Case 6 time series and the last output's profile (see the module docstring)."""
    nu = hbar_over_m_code(MASS_EV)
    K = 2 * np.pi * mode / L_BOX
    w1 = omega_free(K, MASS_EV)
    rows, last = [], None
    for fn in find_files(run):
        _, d, _, _, time = tip.readtipsy(fn)
        w1t = w1 * float(np.atleast_1d(time)[0])
        th = K * d[:, 1].astype(float)
        vx = d[:, 4].astype(float)
        f1, f2 = np.exp(-1j * th), np.exp(-2j * th)
        c1, c2, p1, p2 = three_wave_modes(w1t, K, nu)
        rows.append([w1t / (2 * np.pi), abs(f1.mean() - c1) / (2. / 3.), abs(f2.mean() - c2) / (1. / 3.),
                     np.hypot(abs((vx * f1).mean() - p1), abs((vx * f2).mean() - p2)) / (nu * K)])
        last = (th % (2 * np.pi), w1t)
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    nb = 48
    cnt = np.bincount(np.minimum((last[0] / (2 * np.pi) * nb).astype(int), nb - 1), minlength=nb)
    return dict(x=a[:, 0], e1=a[:, 1], e2=a[:, 2], ep=a[:, 3], thb=(np.arange(nb) + 0.5) / nb,
                rhob=cnt * nb / len(last[0]), w1t_last=last[1])


def stream_test(run):
    """Case 5 time series and the last output's v_x(x) (see the module docstring)."""
    sigma0, tau = two_stream(MASS_EV)
    rows, last = [], None
    for fn in find_files(run):
        _, d, _, _, time = tip.readtipsy(fn)
        t = float(np.atleast_1d(time)[0]) - tau
        x, vx = d[:, 1].astype(float), d[:, 4].astype(float)
        x = x - x.mean()
        vx = vx - vx.mean()
        f = 1 + (t / tau)**2
        rows.append([t / tau, np.sqrt(2 * (x * x).mean()) / (sigma0 * np.sqrt(f)), (x * vx).sum() / (x * x).sum() * tau,
                     (vx * vx).mean() / (sigma0**2 / (4 * tau**2))])
        last = (x / sigma0, vx * tau / sigma0)
    if not rows:
        raise SystemExit(f'no snapshots found for {run}')
    a = np.array(rows)
    xs, vs = last
    edges = np.linspace(-3.5, 3.5, 29)
    ib = np.digitize(xs, edges) - 1
    q = np.array([np.percentile(vs[ib == i], [5, 50, 95]) if (ib == i).sum() >= 5 else [np.nan] * 3
                  for i in range(len(edges) - 1)])
    return dict(x=a[:, 0], width=a[:, 1], slope=a[:, 2], Ekin=a[:, 3], xb=0.5 * (edges[1:] + edges[:-1]), vq=q)


def reference_table(run, params, args=None):
    if params["case"] == 4:
        r = node_test(run, params["mode"], params["ampl"])
        return {'x': r['x'], 'amp': r['amp'], 'phase': r['phase']}
    if params["case"] == 5:
        r = stream_test(run)
        return {'x': r['x'], 'width': r['width'], 'slope': r['slope'], 'Ekin': r['Ekin']}
    if params["case"] == 6:
        r = three_wave_test(run, params["mode"])
        return {'x': r['x'], 'moderr': np.hypot(r['e1'], r['e2']) / np.sqrt(2.)}
    r = measure(run, params)
    if params["case"] in (2, 7):
        return {'x': r['a'], 'delta': r['delta'] / params["ampl"]}
    if params["case"] == 3:
        return {'x': r['x'], 'sigma': r['sigma'], 'noise': r['noise'], 'Ekin': r['Ekin']}
    return {'x': r['t'] / r['T'], 'delta': r['delta'] / params["ampl"],
            'w_dyn': np.array([r['wd'] / r['wa']]), 'w_stat': np.array([r['wstat'] / r['wa']])}


def _name(inp):
    return os.path.basename(os.path.normpath(inp))


def plot_wave(inputs, labels, params, args):
    """Cases 0 and 1."""
    import matplotlib.pyplot as plt
    A, case, mode = params["ampl"], params["case"], params["mode"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    grow = False
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = measure(inp, params)
        grow = r['grow']
        tt = r['t'] / r['T']
        if j == 0:
            tf = np.linspace(0, tt.max(), 400)
            if grow:
                ax[0].plot(tf, np.cosh(tf), '-', color='k', lw=2, label=r'analytic $\cosh\gamma t$')
                ax[1].axhline(1.0, color='k', lw=2, label='analytic')
            else:
                ax[0].plot(tf, np.cos(2 * np.pi * tf), '-', color='k', lw=2, label=r'analytic $\cos\omega t$')
                ax[1].axhline(1.0, color='k', lw=2, label='no damping')
        c = f'C{j}'
        sym = r'\gamma' if grow else r'\omega'
        ax[0].plot(tt, r['delta'] / A, 'o', ms=5, color=c,
                   label=f'{lab}: ${sym}_{{dyn}}/{sym}$ {r["wd"]/r["wa"]:.4f}, '
                         f'${sym}_{{stat}}/{sym}$ {r["wstat"]/r["wa"]:.4f}, C/A {r["C"]/A:.3f}')
        ax[1].plot(tt, r['env'], 'o-', ms=4, color=c, label=lab)
        what = 'gamma' if grow else 'w'
        print(f'{_name(inp)}: {what}_dyn/{what}_an {r["wd"]/r["wa"]:.4f}  {what}_stat/{what}_an {r["wstat"]/r["wa"]:.4f}  '
              f'C/A {r["C"]/A:.4f}  {"delta/(A cosh)" if grow else "envelope"} {r["env"].min():.4f}..{r["env"].max():.4f}  '
              f't_end {tt.max():.2f} {"/gamma" if grow else "T"}')
        if case == 1:
            _print_f(inp, r)
    ref = getattr(args, 'ref_table', None)
    if ref is not None:
        ax[0].plot(ref['x'], ref['delta'], 'D', mfc='none', color='C3', label='reference')
    if grow:
        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$\delta_k(t)/A$')
        ax[1].set_ylabel(r'$\delta_k\,/\,(A\cosh\gamma t)$')
        xl = r'$\gamma t$,  $\gamma^2 = 4\pi G\rho_0 - (\hbar/2m)^2k^4$'
    else:
        ax[0].set_ylabel(r'$\delta_k(t)/A$')
        ax[1].set_ylabel(r'$\sqrt{\delta_k^2 + (kV/\omega)^2}\,/\,C$')
        xl = (r'$t / T$,  $T = 2\pi/\omega$,  ' +
              (r'$\omega = (\hbar/2m)k^2$' if case == 0 else r'$\omega^2 = (\hbar/2m)^2k^4 - 4\pi G\rho_0$'))
    for a_ in ax:
        a_.set_xlabel(xl)
        a_.grid(alpha=0.25)
    if case == 0:
        fig.suptitle(f'FDM free dispersion, mode n = {mode}, A = {A:g}')
        tag = 'dispersion'
    else:
        fig.suptitle(f'FDM Jeans test (static box), mode n = {r["mode"]}, k/k$_J$ = {r["k"] / r["kJ"]:.3f}, A = {A:g}: '
                     + ('growing' if grow else 'oscillating'))
        tag = 'jeans'
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_{tag}.png' if args.save else None))


def _print_f(inp, r):
    print(f'{_name(inp)}: n {r["mode"]}  k/k_J {r["k"] / r["kJ"]:.3f}  kh {r["kh"]:.3f}  f_stat {r["f_stat"]:.3f}  '
          f'f_dyn {r["f_dyn"]:.3f} +- {r["f_err"]:.3f}  g_stat {r["g_stat"]:.4f}')


def plot_jeans_scan(inputs, labels, params, args):
    """Case 1, runs of several n: delta(t) per mode with the f fit; f against k h; delta against (k/k_J)^2."""
    import matplotlib.pyplot as plt
    A = params["ampl"]
    rs = sorted(((measure(inp, params), inp) for inp in inputs), key=lambda q: q[0]['k'])
    for r, inp in rs:
        _print_f(inp, r)

    nc = min(3, len(rs))
    fig, axs = plt.subplots(int(np.ceil(len(rs) / nc)), nc, figsize=(5.2 * nc, 3.9 * np.ceil(len(rs) / nc)),
                            squeeze=False)
    used = list(axs.flat)[:len(rs)]
    for a_ in list(axs.flat)[len(rs):]:
        a_.set_visible(False)
    for a_, (r, inp) in zip(used, rs):
        wg = np.sqrt(r['wg2'])
        tf = np.linspace(0., r['t'].max(), 400)
        a_.plot(tf * wg, jeans_model(tf, 1.0, r['wq2'], r['wg2']), '-', color='k', lw=2, label='analytic (f = 1)')
        a_.plot(tf * wg, r['C_f'] / A * jeans_model(tf, r['f_dyn'], r['wq2'], r['wg2']), '--', color='C1', lw=1.5,
                label=r'fit with $f_{dyn}$')
        a_.plot(r['t'] * wg, r['delta'] / A, 'o', ms=3.5, color='C0', label='simulation')
        a_.set_title(f'n = {r["mode"]}, k/k$_J$ {r["k"] / r["kJ"]:.2f}, kh {r["kh"]:.2f}: '
                     f'f$_{{dyn}}$ {r["f_dyn"]:.2f}, f$_{{stat}}$ {r["f_stat"]:.2f}', fontsize=10)
        a_.set_xlabel(r'$t\,\sqrt{4\pi G\rho_0}$')
        a_.set_ylabel(r'$\delta_k/A$')
        a_.grid(alpha=0.25)
    fig.suptitle(f'FDM Jeans scan (static box), A = {A:g}: '
                 r'$\omega^2 = f\,(\hbar/2m)^2k^4 - 4\pi G\rho_0$')
    fig.tight_layout()
    finish_figure_with_legend(fig, used, save=(f'{args.save}_jeansmodes.png' if args.save else None))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    kh = np.array([r['kh'] for r, _ in rs])
    ax[0].axhline(1.0, color='k', lw=2, label='exact (f = 1)')
    ax[0].errorbar(kh, [r['f_dyn'] for r, _ in rs], yerr=[r['f_err'] for r, _ in rs], fmt='o', color='C0',
                   capsize=3, label=r'$f_{dyn}$: fit to $\delta_k(t)$')
    ax[0].plot(kh, [r['f_stat'] for r, _ in rs], 's', mfc='none', ms=8, color='C3',
               label=r'$f_{stat}$: quantum-pressure force of the first output')
    for r, _ in rs:
        ax[0].annotate(str(r['mode']), (r['kh'], r['f_dyn']), textcoords='offset points', xytext=(5, 5), fontsize=9)
    r0 = rs[0][0]
    ax[0].axvline(r0['kJ'] * r0['kh'] / r0['k'], color='0.5', ls='--', lw=1, label=r'$k = k_J$')
    ax[0].set_xlabel(r'$kh$  ($h$ at the mean density, kernel support $2h$)')
    ax[0].set_ylabel(r'$f$')

    tau_end = min((r['t'] * np.sqrt(r['wg2'])).max() for r, _ in rs)
    q = np.geomspace(min(r['k'] / r['kJ'] for r, _ in rs), max(r['k'] / r['kJ'] for r, _ in rs), 800)
    ax[1].axvline(1.0, color='0.5', ls='--', lw=1)
    for j, tau in enumerate(tau_end * np.array([1. / 3., 2. / 3., 1.])):
        c = f'C{j}'
        ax[1].plot(q**2, jeans_model(tau, 1.0, q**4, 1.0), '-', color=c, lw=1.5)
        ax[1].plot([(r['k'] / r['kJ'])**2 for r, _ in rs],
                   [np.interp(tau, r['t'] * np.sqrt(r['wg2']), r['delta']) / A for r, _ in rs], 'o', color=c,
                   label=rf'$t\,\sqrt{{4\pi G\rho_0}}$ = {tau:.3g} (line: analytic)')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$(k/k_J)^2$')
    ax[1].set_ylabel(r'$\delta_k/A$')
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.suptitle(f'FDM Jeans scan (static box), A = {A:g}: quantum-pressure fraction f per mode, '
                 r'$\omega^2 = f\,(\hbar/2m)^2k^4 - 4\pi G\rho_0$')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_jeansscan.png' if args.save else None))


def plot_comoving(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    A, mode = params["ampl"], params["mode"]
    L, nu = comoving_units(MASS_EV)
    k = 2 * np.pi * mode / L_BOX
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    af = np.geomspace(A_INIT, 1.0, 300)
    ax[0].plot(af, comoving_growth(k, nu, af, A) / A, '-', color='k', lw=2, label='linear FDM ODE')
    ax[0].plot(af, comoving_growth(k, 0.0, af, A) / A, ':', color='0.5', lw=2, label='CDM (no quantum pressure)')
    ax[1].axhline(0.0, color='k', lw=2, label='ODE')
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = measure(inp, params)
        c = f'C{j}'
        ax[0].plot(r['a'], r['delta'] / A, 'o', ms=4, color=c, label=lab)
        ax[1].plot(r['a'], r['err'], 'o-', ms=3, color=c, label=lab)
        print(f'{_name(inp)}: max|delta - ODE|/A {np.abs(r["err"]).max():.4f}  delta/ODE at a = {r["a"][-1]:.3f}: '
              f'{r["delta"][-1] / r["ode"][-1]:.4f}  (CDM/ODE {r["cdm"][-1] / r["ode"][-1]:.2f})')
    ref = getattr(args, 'ref_table', None)
    if ref is not None:
        ax[0].plot(ref['x'], ref['delta'], 'D', mfc='none', color='C3', label='reference')
    kJ = lambda a: (16 * np.pi * a)**0.25 / np.sqrt(nu)
    for a_ in ax:
        a_.set_xscale('log')
        a_.set_xlabel('expansion factor a')
        a_.grid(alpha=0.25)
        if k < kJ(1.0):
            ac = (k * np.sqrt(nu))**4 / (16 * np.pi)
            a_.axvline(ac, color='0.6', lw=1, ls='--')
    ax[0].set_ylabel(r'$\delta_k(a)/A$')
    ax[1].set_ylabel(r'$(\delta_k - \delta_{ODE})/A$')
    fig.suptitle(f'FDM comoving Jeans crossing (EdS, box {L:.1f} kpc comoving), mode n = {mode}, '
                 rf'k/k$_J$ {k / kJ(A_INIT):.2f} $\to$ {k / kJ(1.0):.2f}, A = {A:g}')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_comoving.png' if args.save else None))


def plot_schive(inputs, labels, params, args):
    """Case 7: the runs on the axes of Schive (2025) Fig. 11, f(xi) against xi (time runs to the left)."""
    import matplotlib.pyplot as plt
    A = params["ampl"]
    rs = [measure(inp, params) for inp in inputs]
    lo = min(0.8, min(r['xi'].min() for r in rs))
    hi = max(50., max(r['xi'].max() for r in rs))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    xs = np.geomspace(lo, hi, 3000)
    ax[0].plot(xs, schive_f(xs), '-', color='k', lw=1.5, label=r'$f(\xi)$ (Schive 2025 Eq. 77, Fig. 11)')
    ax[1].axhline(0.0, color='k', lw=1.5)
    for j, (inp, r, lab) in enumerate(zip(inputs, rs, labels)):
        c = f'C{j}'
        ax[0].plot(r['xi'], r['delta'] / A, 'o', ms=3, color=c, label=lab)
        ax[1].plot(r['xi'], r['err'], 'o-', ms=2, color=c, label=lab)
        print(f'{_name(inp)}: n = {r["mode"]}, xi {r["xi"][0]:.2f} -> {r["xi"][-1]:.2f}: max|delta - A f|/A '
              f'{np.abs(r["err"]).max():.4f}  end (delta - A f)/A {r["err"][-1]:+.4f}')
    for a_ in ax:
        a_.set_xscale('log')
        a_.set_xlim(lo, hi)
        a_.axvline(np.sqrt(6.), color='b', ls='--', lw=1)
        a_.set_xlabel(r'$\xi = (\hbar/m)\,k^2 / (H_0\,a^{1/2})$  (time runs to the left)')
        a_.grid(alpha=0.25)
    ax[0].set_ylabel(r'$\delta_k / A$')
    ax[1].set_ylabel(r'$(\delta_k - A f(\xi)) / A$')
    fig.suptitle(rf'FDM comoving growing mode (EdS, a = {A7_INIT:g} $\to$ {A7_END:g}, A = {A:g}) against '
                 r'$f(\xi)$; $\xi^2 = 6$ at $k = k_J$ (dashed)')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_schive.png' if args.save else None))


def plot_null(inputs, labels, args):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.0))
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = null_test(inp)
        c = f'C{j}'
        for a_, q in zip(ax, ('sigma', 'noise', 'Ekin', 'dE')):
            a_.plot(r['x'], r[q], 'o-', ms=3, color=c, label=lab)
        print(f'{_name(inp)}: sigma_v/(nu/dx) {r["sigma"][0]:.3e} -> {r["sigma"][-1]:.3e}  density noise x{r["noise"][-1]:.3f}  '
              f'E_kin/E_Q1 {r["Ekin"][0]:.4f} -> {r["Ekin"][-1]:.4f}  dE/E_Q1 {r["dE"][-1]:+.4f}  t_end {r["x"][-1]:.2f} T1')
    ax[1].axhline(1.0, color='k', lw=1.5, label='first output')
    ax[3].axhline(0.0, color='k', lw=1.5, label='conserved')
    for a_, yl in zip(ax, (r'$\sigma_v\,/\,(\hbar/m\,/\,\Delta x)$', r'density noise / first output',
                           r'$E_{kin}\,/\,E_{Q,1}$', r'$(E - E_1)\,/\,E_{Q,1}$')):
        a_.set_xlabel(r'$t / T_1$')
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    fig.suptitle('FDM null test: uniform glass, self-gravitating, box = 0.25 Jeans lengths (nothing should happen); '
                 r'$T_1$: period of the free n = 1 mode')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_null.png' if args.save else None))


def plot_node(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    A, mode = params["ampl"], params["mode"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.0))
    thf = np.linspace(-np.pi, np.pi, 400)
    rho_ex, u_ex = node_wave(thf, A, 1.0, 1.0)
    ax[0].axhline(1.0, color='k', lw=1.5, label='exact')
    ax[1].axhline(0.0, color='k', lw=1.5, label='exact')
    ax[2].plot(thf / (2 * np.pi), rho_ex, '-', color='k', lw=2, label='exact')
    ax[3].plot(thf / (2 * np.pi), u_ex, '-', color='k', lw=2, label='exact')
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = node_test(inp, mode, A)
        c = f'C{j}'
        ax[0].plot(r['x'], r['amp'], 'o-', ms=3, color=c, label=lab)
        ax[1].plot(r['x'], r['phase'], 'o-', ms=3, color=c, label=lab)
        ax[2].plot(r['thb'], r['rhob'], 'o', ms=4, color=c, label=lab)
        ax[3].plot(r['thb'], r['vb'], 'o', ms=4, color=c, label=lab)
        print(f'{_name(inp)}: amplitude/A {r["amp"][0]:.4f} -> {r["amp"][-1]:.4f} (min {r["amp"].min():.4f})  '
              f'phase error {r["phase"][-1]:+.4f} wavelengths  rms(v - u)/(nu K) {r["verr"][0]:.3e} -> {r["verr"][-1]:.3e}  '
              f't_end {r["x"][-1]:.2f} periods')
    for a_, xl, yl in zip(ax, (r'$t\,\Omega / 2\pi$', r'$t\,\Omega / 2\pi$', r'$\theta / 2\pi$', r'$\theta / 2\pi$'),
                          (r'pattern amplitude / A', r'phase error [wavelengths]', r'$\rho/\rho_0$ (last output)',
                           r'$v_x\,/\,(\hbar K/m)$ (last output)')):
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    fig.suptitle(f'FDM two interfering waves, K = 2$\pi$ {mode} / L, contrast A = {A:g}'
                 + (' (nodes)' if A == 1 else '') + r': $\rho = \rho_0(1 + A\cos\theta)$, $\theta = Kx - \Omega t$')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_nodes.png' if args.save else None))


def plot_three(inputs, labels, params, args):
    import matplotlib.pyplot as plt
    mode = params["mode"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.0))
    tf = np.linspace(0, NW3_TEND / (8 * np.pi), 801)          # t / T1
    ru = [three_unsigned_rho(2 * np.pi * t) for t in tf]
    cx = three_wave_modes(2 * np.pi * tf, 1., 1.)
    for a_, q, norm in ((ax[0], 1, 2. / 3.), (ax[1], 2, 1. / 3.)):
        eu = [abs(np.mean(r * np.exp(-1j * q * _TH)) - c) / norm for r, c in zip(ru, cx[q - 1])]
        a_.plot(tf, eu, '-', color='0.55', lw=1.4, label='unsigned |1 + 2cos θ|, undamped')
        a_.axhline(0.0, color='k', lw=1.5, label='exact')
    ax[2].axhline(0.0, color='k', lw=1.5, label='exact')
    w1t_last = None
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = three_wave_test(inp, mode)
        c = f'C{j}'
        ax[0].plot(r['x'], r['e1'], '-', lw=1.4, color=c, label=lab)
        ax[1].plot(r['x'], r['e2'], '-', lw=1.4, color=c, label=lab)
        ax[2].plot(r['x'], r['ep'], '-', lw=1.4, color=c, label=lab)
        ax[3].plot(r['thb'], r['rhob'], 'o', ms=4, color=c, label=lab)
        w1t_last = r['w1t_last']
        print(f'{_name(inp)}: density mode errors q=1 max {r["e1"].max():.3f} end {r["e1"][-1]:.3f}, q=2 max '
              f'{r["e2"].max():.3f} end {r["e2"][-1]:.3f}  momentum mode error end {r["ep"][-1]:.3e}  '
              f't_end {r["x"][-1]:.2f} T1')
    if w1t_last is not None:
        ax[3].plot(_TH / (2 * np.pi), three_exact_rho(w1t_last), '-', color='k', lw=2, label='exact (last output)')
        ax[3].plot(_TH / (2 * np.pi), three_unsigned_rho(w1t_last), '--', color='0.55', lw=1.4, label='unsigned')
    for a_, xl, yl in zip(ax, (r'$t / T_1$',) * 3 + (r'$Kx / 2\pi$',),
                          (r'density mode $q=1$: error / (2/3)', r'density mode $q=2$: error / (1/3)',
                           r'momentum modes: error / $(\hbar K/m)$', r'$\rho/\rho_0$ (last output)')):
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    fig.suptitle(rf'FDM three interfering waves (Hopkins 2019 Fig. 2), $k_m = mK$, $K = 2\pi\,{mode}/L$, equal amplitudes: '
                 r'discontinuous nodes at $t = 0, T_1/2, T_1, \ldots$ ($T_1 = 2\pi/\omega_1$)')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_threewave.png' if args.save else None))


def plot_stream(inputs, labels, args):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.0))
    tf = np.linspace(-1, 1, 200)
    ax[0].axhline(1.0, color='k', lw=1.5, label='exact')
    ax[1].plot(tf, tf / (1 + tf**2), '-', color='k', lw=2, label=r'exact $t\tau/(\tau^2 + t^2)$')
    ax[2].plot(tf, 2 * tf**2 / (1 + tf**2), '-', color='k', lw=2, label='exact')
    xf = np.linspace(-3.5, 3.5, 50)
    ax[3].plot(xf, 0.5 * xf, '-', color='k', lw=2, label=r'exact at $t = +\tau$')
    ax[3].plot(xf, -0.5 * xf, ':', color='0.5', lw=2, label=r'IC ($t = -\tau$)')
    for j, (inp, lab) in enumerate(zip(inputs, labels)):
        r = stream_test(inp)
        c = f'C{j}'
        ax[0].plot(r['x'], r['width'], 'o-', ms=3, color=c, label=lab)
        ax[1].plot(r['x'], r['slope'], 'o', ms=4, color=c, label=lab)
        ax[2].plot(r['x'], r['Ekin'], 'o', ms=4, color=c, label=lab)
        ax[3].errorbar(r['xb'], r['vq'][:, 1], yerr=[r['vq'][:, 1] - r['vq'][:, 0], r['vq'][:, 2] - r['vq'][:, 1]],
                       fmt='o', ms=4, color=c, capsize=2, label=lab)
        i0 = int(np.argmin(np.abs(r['x'])))
        print(f'{_name(inp)}: width/exact at t = {r["x"][i0]:+.2f} tau: {r["width"][i0]:.4f}, at {r["x"][-1]:+.2f} tau: '
              f'{r["width"][-1]:.4f}  slope tau {r["slope"][-1]:+.4f} (exact {r["x"][-1] / (1 + r["x"][-1]**2):+.4f})  '
              f'E_kin/E_kin(-tau) {r["Ekin"][-1]:.4f} (exact {2 * r["x"][-1]**2 / (1 + r["x"][-1]**2):.4f})')
    for a_, xl, yl in zip(ax, (r'$t/\tau$', r'$t/\tau$', r'$t/\tau$', r'$x/\sigma_0$'),
                          (r'width $\sqrt{2\langle x^2\rangle}$ / exact', r'velocity slope $\times\,\tau$',
                           r'$E_{kin}\,/\,E_{kin}(-\tau)$', r'$v_x\,\tau/\sigma_0$ (last output; median, 5-95 %)')):
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
        a_.grid(alpha=0.25)
    fig.suptitle(r'FDM two-stream intersection: a converging Gaussian bounces, $\rho\propto e^{-x^2/\sigma_0^2 f}/\sqrt{f}$, '
                 r'$u = xt/(\tau^2 f)$, $f = 1 + t^2/\tau^2$, $\tau = \sigma_0^2 m/\hbar$')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{args.save}_twostream.png' if args.save else None))


def plot(inputs, labels, params, args):
    setup_rcparams()
    if params["case"] == 1 and len({_run_mode(inp, 1, params["mode"]) for inp in inputs}) > 1:
        plot_jeans_scan(inputs, labels, params, args)
    elif params["case"] in (0, 1):
        plot_wave(inputs, labels, params, args)
    elif params["case"] == 2:
        plot_comoving(inputs, labels, params, args)
    elif params["case"] == 3:
        plot_null(inputs, labels, args)
    elif params["case"] == 4:
        plot_node(inputs, labels, params, args)
    elif params["case"] == 5:
        plot_stream(inputs, labels, args)
    elif params["case"] == 6:
        plot_three(inputs, labels, params, args)
    elif params["case"] == 7:
        plot_schive(inputs, labels, params, args)
    else:
        raise SystemExit(f'fdmwave case {params["case"]} has no analysis')


if __name__ == "__main__":
    standalone_cli("fdmwave", PARAM_SPEC, plot, reference_table=reference_table,
                   description="FDM linear waves: free dispersion (0), Jeans (1), comoving Jeans crossing (2), "
                               "null test (3), two-wave nodes (4), two-stream intersection (5), three-wave nodes (6), "
                               "comoving growing mode (7).",
                   reg_keys=('delta', 'sigma', 'Ekin', 'amp', 'width', 'moderr'), add_time=False)
