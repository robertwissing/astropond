#!/usr/bin/env python3
"""
Shared analysis for the FDM static operator tests (soliton, gaussOverdens,
1DdensityFront; Nori & Baldi 2018). Each IC_analysis_<test>.py is a thin call
to static_main().

The run is static (dDelta 1e-22), so the first snapshot is analysed. ChaNGa
(--enable-fdm) writes the quantum potential Q (.fdmq) and the quantum-pressure
acceleration a_QP = -grad Q (.aqp) of the dark particles. The analytic values
follow from the setup's own density, differentiated on a fine grid:
    Q = -(hbar/m)^2/4 (lap ln rho + |grad ln rho|^2 / 2),   a_s = -dQ/ds,
with s = r (spherical setups) or x (the 1D front).

Reported (interior particles only, a 4h layer at the open box faces removed,
where the one-sided kernel gives a spurious gradient):
  L2_Q, L2_a     ||num - analytic|| / ||analytic|| for Q and a_s
  L2_aperp       ||a_perp|| / ||a_analytic||, the force across s (noise)
  Pnet           |sum m a| / sum m |a|, the net quantum force (momentum)
Figure: binned medians and 16-84 % bands of Q and a_s against the analytic
profile.
"""
import importlib
import os

import numpy as np
import readtipsy as tip

from IC_analysis_framework import standalone_cli
from IC_analysis_general import find_files, setup_rcparams, finish_figure_with_legend
from IC_setup_fdmwave import hbar_over_m_code

GEOM = {'soliton': 'r', 'gaussOverdens': 'r', '1DdensityFront': 'x'}
HALFBOX = 5000.0      # kpc: all three setups use the open 10 Mpc box [-5000, 5000]^3
NBINS = 30


def load_setup(test):
    return getattr(importlib.import_module(f'IC_setup_{test}'), f'setup_{test}')()


def analytic(test):
    """(S, s_grid, Q(s), a_s(s)) from the setup's density on a fine grid."""
    S = load_setup(test)
    C = 0.25 * hbar_over_m_code(S.fdm_mass_eV)**2
    if GEOM[test] == 'r':
        # ds 0.5 kpc (finer grids lose a to round-off), mirrored about r = 0 without a point at 0,
        # so every difference is central and 2 (ln rho)'/r is finite at the centre
        s = np.linspace(-8999.75, 8999.75, 36000)
        L = np.log(S.getrho_vec(np.abs(s)))
        d1 = np.gradient(L, s)
        lap = np.gradient(d1, s) + 2.0 * d1 / s
        Q = -C * (lap + 0.5 * d1**2)
        a = -np.gradient(Q, s)
        return S, s[s > 0], Q[s > 0], a[s > 0]
    else:
        s = np.linspace(-5000.0, 5000.0, 20001)
        L = np.log(S.getrho_vec(s))
        d1 = np.gradient(L, s)
        lap = np.gradient(d1, s)
    Q = -C * (lap + 0.5 * d1**2)
    return S, s, Q, -np.gradient(Q, s)


def first_snapshot(run):
    files = find_files(run)
    if not files:
        raise SystemExit(f'no snapshots found for {run}')
    return files[0]


def measure(test, run):
    """Per-particle comparison on the first snapshot of `run`."""
    S, sg, Qg, ag = analytic(test)
    fn = first_snapshot(run)
    d = tip.readtipsy(fn)[1]
    n = len(d)
    m, x = d[:, 0].astype(float), d[:, 1:4].astype(float)
    rho = tip.readtipsyaux(fn, 'fdmrho')[0].astype(float)
    Q = tip.readtipsyaux(fn, 'fdmq')[0].astype(float)
    a = tip.readtipsyVecaux(fn, 'aqp')[0].reshape(n, 3).astype(float)
    h = 0.5 * (3.0 * S.ns * m / (4.0 * np.pi * rho))**(1.0 / 3.0)     # ns neighbours within 2h
    if GEOM[test] == 'r':
        s = np.linalg.norm(x, axis=1)
        e = x / np.maximum(s, 1e-30)[:, None]
    else:
        s = x[:, 0]
        e = np.tile([1.0, 0.0, 0.0], (n, 1))
    face = HALFBOX - np.abs(x).max(axis=1)                          # distance to the nearest face
    sel = face > 4.0 * h
    Qa, aa = np.interp(s, sg, Qg), np.interp(s, sg, ag)
    apar = (a * e).sum(1)
    aperp = np.linalg.norm(a - apar[:, None] * e, axis=1)
    l2 = lambda num, an: float(np.sqrt(((num - an)[sel]**2).sum() / (an[sel]**2).sum()))
    P = (m[:, None] * a).sum(0)
    return dict(fn=fn, s=s[sel], Q=Q[sel], a=apar[sel], h=h[sel], sg=sg, Qg=Qg, ag=ag,
                L2_Q=l2(Q, Qa), L2_a=l2(apar, aa),
                L2_aperp=float(np.sqrt((aperp[sel]**2).sum() / (aa[sel]**2).sum())),
                Pnet=float(np.linalg.norm(P) / (m * np.linalg.norm(a, axis=1)).sum()))


def binned(s, y, edges):
    idx = np.digitize(s, edges) - 1
    out = np.full((3, len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        yb = y[idx == b]
        if yb.size >= 5:
            out[:, b] = np.percentile(yb, [16, 50, 84])
    return 0.5 * (edges[1:] + edges[:-1]), out


def edges_for(test):
    return np.linspace(0.0, 3500.0, NBINS + 1) if GEOM[test] == 'r' else np.linspace(-3000.0, 3000.0, NBINS + 1)


def reference_table(test, run):
    r = measure(test, run)
    sc, Qb = binned(r['s'], r['Q'], edges_for(test))
    _, ab = binned(r['s'], r['a'], edges_for(test))
    return {'x': sc, 'Q_med': Qb[1], 'a_med': ab[1], 'L2_Q': np.array([r['L2_Q']]), 'L2_a': np.array([r['L2_a']])}


def plot(test, inputs, labels, save=None, ref_table=None):
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    sl = 'r [kpc]' if GEOM[test] == 'r' else 'x [kpc]'
    lo, hi = edges_for(test)[[0, -1]]
    for k, (inp, lab) in enumerate(zip(inputs, labels)):
        r = measure(test, inp)
        if k == 0:
            g = (r['sg'] >= lo) & (r['sg'] <= hi)
            ax[0].plot(r['sg'][g], r['Qg'][g], '-', color='k', lw=2, label='analytic')
            ax[1].plot(r['sg'][g], r['ag'][g], '-', color='k', lw=2, label='analytic')
        sc, Qb = binned(r['s'], r['Q'], edges_for(test))
        _, ab = binned(r['s'], r['a'], edges_for(test))
        c = f'C{k}'
        for j, b in enumerate((Qb, ab)):
            ax[j].fill_between(sc, b[0], b[2], color=c, alpha=0.2, lw=0)
            ax[j].plot(sc, b[1], 'o-', ms=4, color=c,
                       label=f'{lab}: L2 err Q {r["L2_Q"]:.1%}, a {r["L2_a"]:.1%}, '
                             f'a_perp {r["L2_aperp"]:.1%}')
        print(f'{os.path.basename(os.path.normpath(inp))}: L2_Q {r["L2_Q"]:.4f}  L2_a {r["L2_a"]:.4f}  '
              f'L2_aperp {r["L2_aperp"]:.4f}  Pnet {r["Pnet"]:.2e}  h {np.percentile(r["h"], 1):.0f}-'
              f'{np.percentile(r["h"], 99):.0f} kpc  ({r["fn"]})')
    if ref_table is not None:
        ax[0].plot(ref_table['x'], ref_table['Q_med'], 'D', mfc='none', color='C3', label='reference')
        ax[1].plot(ref_table['x'], ref_table['a_med'], 'D', mfc='none', color='C3', label='reference')
    ax[0].set_ylabel('Q  [code units]')
    ax[1].set_ylabel(r'$a_{QP}$ along $\hat s$  [code units]')
    for a_ in ax:
        a_.set_xlabel(sl)
        a_.axhline(0, color='0.5', lw=0.8)
        a_.grid(alpha=0.25)
    fig.suptitle(f'FDM static {test}: binned median and 16-84 % band vs analytic')
    fig.tight_layout()
    finish_figure_with_legend(fig, list(ax), save=(f'{save}_profile.png' if save else None))


def static_main(test, description):
    standalone_cli(
        test, [], lambda inputs, labels, params, args: plot(
            test, inputs, labels, save=args.save, ref_table=getattr(args, 'ref_table', None)),
        reference_table=lambda inp, params, args: reference_table(test, inp),
        description=description, reg_keys=('a_med',), add_time=False)
