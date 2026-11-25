#!/usr/bin/env python3
"""
Schroedinger-Poisson reference solutions for the soliton collision test (soliton -a 3 -b D -c v).

WHY. A head-on collision of two solitons has no closed form, and the particle schemes disagree with each other,
so the test is scored against the wave equation the Madelung fluid stands for:
    i dpsi/dt = -(nu/2) lap psi + (Phi/nu) psi,   lap Phi = 4 pi |psi|^2,   nu = hbar/m,  G = 1 (kpc, Msun).
The outcome of a bound collision depends on the relative phase of the two solitons (0: merger, pi: they bounce off
the destructive-interference node, pi/2: mass flows to one of them); a particle code carries no phase, so a bound
collision is scored against the band of phases. An unbound one is phase-independent once the solitons separate.

HOW. Pseudo-spectral split step (kick-drift-kick, exact free propagator, dt = CK dx^2/nu), isolated gravity by a
zero-padded FFT convolution (Hockney), a 2 kpc absorbing sponge at the box faces. The solitons are the untruncated
ground state of IC_setup_soliton.sp_groundstate (r_c = 1 kpc) at x = -+D/2 r_c, moving at +-v/2 v_c(r_c), the
right one multiplied by exp(i pi phase). The run ends when the particle run does (IC_setup_soliton.run_length).
Checked: dt/4 changes nothing (7e-6); dx 1/12 vs 1/8 kpc changes the peak
density by <= 2 % and the mass near the peaks by <= 0.01; a 48 x 32 x 32 box vs 32 x 20 x 20 changes them by
<= 1.5 % (bound case).

OUTPUT (reference_solution_highres/soliton/, picked up by IC_analysis_soliton.py):
  collision_D<D>_v<v>_phase<p>.csv     one row per output: t_Myr, M_box (mass left in the box / initial), M10
                                       (mass within 10 r_c of the origin / both solitons' mass), and per
                                       half (L: x < 0, R: x >= 0) the centre x (centre of mass within 1.5 r_c of
                                       the half's density peak, kpc), m (mass there / one soliton's mass) and rho
                                       (the half's peak density / rho_c), then K and W (code units).
  collision_D<D>_v<v>_phase<p>_maps.npz  projected density (along z, 0.25 kpc pixels, |y| <= 8 kpc) every
                                       --map-every outputs (default: at the particle run's snapshot times).

HALO (soliton -a 4 -b N -c R, --halo): the N solitons of IC_setup_soliton.halo_positions(N, R) at rest, soliton j
multiplied by exp(i theta_j); phase realisation 0 has all theta_j = 0, realisation p >= 1 draws them uniformly from
a generator seeded with 100 + p. Box 32^3 kpc. Output halo_N<N>_R<R>_phase<p>.csv: t_Myr, M_box, rho_max / rho_c,
the peak position (xc, yc, zc), m1, m2, m5, m10 (mass within 1, 2, 5, 10 kpc of the peak / the initial mass), K, W;
_prof.npz: the spherically averaged density around the peak (redges, rho [outputs x bins]) at every output;
_maps.npz as for the collision (|y| <= 16 kpc).

Usage:
    python soliton_reference.py -b 10 -c 6                 # phases 0 and 1 (unbound); 0, 1, 0.5 if bound
    python soliton_reference.py -b 8 -c 1 --phase 0 --workers 60
    python soliton_reference.py --halo                     # N 4, R 6: phase realisations 0, 1, 2
Cost at dx 1/8: ~4 h per phase on 60 cores (fast preset 96 x 32 x 32 kpc, slow 48 x 32 x 32 kpc).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from IC_setup_soliton import (RC, DDELTA, sp_groundstate, collision_energy, run_length, myr_per_code,  # noqa: E402
                               halo_positions)

REFDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                       "reference_solution_highres", "soliton"))
COLS = ['t_Myr', 'M_box', 'M10', 'xL', 'mL', 'rhoL', 'xR', 'mR', 'rhoR', 'K', 'W']
HALO_COLS = ['t_Myr', 'M_box', 'rho_max', 'xc', 'yc', 'zc', 'm1', 'm2', 'm5', 'm10', 'K', 'W']
HALO_R = (1.0, 2.0, 5.0, 10.0)                       # kpc, radii of the m columns
HALO_REDGES = np.geomspace(0.15, 16.0, 25)            # kpc, profile shells around the peak
HALO_BOX = 32.0
SPONGE = 2.0          # kpc
MAP_PIX, MAP_Y = 0.25, 8.0


def fmt(x):
    """runtest.sh's fmt: 2 decimals, trailing zeros stripped (10 -> '10', 0.5 -> '0.5')."""
    s = f'{float(x):.2f}'.rstrip('0').rstrip('.')
    return f'{float(x):g}' if s in ('0', '-0') and float(x) != 0 else s


def reference_path(D, v, phase, refdir=REFDIR):
    return os.path.join(refdir, f'collision_D{fmt(D)}_v{fmt(v)}_phase{fmt(phase)}.csv')


def halo_path(N, R, phase, refdir=REFDIR):
    return os.path.join(refdir, f'halo_N{fmt(N)}_R{fmt(R)}_phase{fmt(phase)}.csv')


def load(path):
    """{column: array} of a reference CSV plus 'meta' (the header's key = value lines); the column names are the
    header's last line."""
    meta, cols = {}, COLS
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if '=' in line:
                k, val = line[1:].split('=', 1)
                meta[k.strip()] = val.strip()
            elif ',' in line:
                cols = [c.strip() for c in line[1:].split(',')]
    data = np.loadtxt(path, delimiter=',', ndmin=2)
    out = {c: data[:, j] for j, c in enumerate(cols)}
    out['meta'] = meta
    return out


def default_box(D, v, tend_code, S):
    """Lx = 24 kpc + D + twice the distance each soliton travels if unbound (at least 48), Ly = Lz = 32 kpc."""
    s = 0.5 * v * S['vc'] * tend_code if collision_energy(D, v, S) > 0 else 0.0
    return (max(48.0, 4.0 * np.ceil((24.0 + D * RC + 2.0 * s) / 4.0)), 32.0, 32.0)


def solve(D, v, phase, dx, box, tend, tout, ck, workers, map_every, out):
    import scipy.fft as sfft
    S = sp_groundstate()
    NU, MYR = S['nu'], myr_per_code()
    Msol = S['M'][-1]
    N = [int(round(l / dx)) for l in box]
    ax = [(np.arange(n) + 0.5) * dx - 0.5 * n * dx for n in N]
    X, Y, Z = np.meshgrid(*ax, indexing='ij', sparse=True)
    amp = lambda r: np.sqrt(np.interp(r, S['r'], S['rho'], right=0.0))
    vel = 0.5 * v * S['vc']
    psi = (amp(np.sqrt((X + 0.5 * D * RC)**2 + Y**2 + Z**2)) * np.exp(1j * vel * X / NU)
           + amp(np.sqrt((X - 0.5 * D * RC)**2 + Y**2 + Z**2)) * np.exp(1j * (-vel * X / NU + np.pi * phase)))
    M0 = (np.abs(psi)**2).sum() * dx**3

    k2 = sum(np.meshgrid(*[(2 * np.pi * sfft.fftfreq(n, dx))**2 for n in N], indexing='ij', sparse=True))
    NP = [2 * n for n in N]
    rp = np.sqrt(sum(c**2 for c in np.meshgrid(*[np.minimum(np.arange(m), m - np.arange(m)) * dx for m in NP],
                                                indexing='ij', sparse=True)))
    g = -dx**3 / np.where(rp > 0, rp, 1.0)
    g[0, 0, 0] = -2.380077 * dx**2               # int_cell dV/r = 2.380077 dx^2
    Ghat = sfft.rfftn(g, workers=workers)
    del g, rp
    pad = np.zeros(NP)

    def potential(rho):
        pad[:N[0], :N[1], :N[2]] = rho
        return sfft.irfftn(sfft.rfftn(pad, workers=workers) * Ghat, s=NP, workers=workers)[:N[0], :N[1], :N[2]]

    depth = np.maximum(np.maximum(*[np.maximum(0.0, np.abs(a) - (0.5 * n * dx - SPONGE)) / SPONGE
                                    for a, n in zip((X, Y), N)]),
                       np.maximum(0.0, np.abs(Z) - (0.5 * N[2] * dx - SPONGE)) / SPONGE)
    Lt = tout / MYR
    nsub = int(np.ceil(Lt / (ck * dx**2 / NU)))
    dt = Lt / nsub
    sp_half = np.exp(-0.5 * dt * (5.0 * 2.5e4 / SPONGE) * depth**2)    # ~5 e-folds across the layer
    kprop = np.exp(-0.5j * NU * dt * k2)
    f = max(1, int(round(MAP_PIX / dx)))
    jy = np.abs(ax[1]) <= MAP_Y
    in10 = X**2 + Y**2 + Z**2 < (10.0 * RC)**2
    print(f'D {D} rc, v {v} v_c ({2 * vel:.1f}), phase {phase} pi, E_tot {collision_energy(D, v, S):.4e}; grid {N} '
          f'dx {dx} (box {box}), dt {dt:.3e} code, {nsub} steps per {tout:.3f} Myr output, t_end {tend:.1f} Myr, '
          f'M0 / 2 M_sol {M0 / (2 * Msol):.5f}, FFT workers {workers}', flush=True)

    def diag(t, psi, Phi):
        rho = np.abs(psi)**2
        K = 0.5 * NU**2 * (k2 * np.abs(sfft.fftn(psi, workers=workers))**2).sum() / psi.size * dx**3
        row = [t * MYR, rho.sum() * dx**3 / M0, (rho * in10).sum() * dx**3 / (2 * Msol)]
        for side in (ax[0] < 0, ax[0] >= 0):
            r_ = np.where(side[:, None, None], rho, 0.0)
            i = np.unravel_index(np.argmax(r_), N)
            s = ((X - ax[0][i[0]])**2 + (Y - ax[1][i[1]])**2 + (Z - ax[2][i[2]])**2 < (1.5 * RC)**2) \
                & side[:, None, None]
            m = (rho * s).sum() * dx**3
            row += [(rho * s * X).sum() * dx**3 / max(m, 1e-300), m / Msol, r_.max() / S['rhoc']]
        row += [K, 0.5 * (rho * Phi).sum() * dx**3]
        P = rho[:, jy].sum(axis=2) * dx
        P = P[:P.shape[0] // f * f, :P.shape[1] // f * f]
        return row, P.reshape(P.shape[0] // f, f, P.shape[1] // f, f).sum((1, 3)) * dx * dx

    def save(rows, maps, final):
        hdr = '\n'.join([f'soliton collision reference (soliton_reference.py): Schroedinger-Poisson, phase {phase} pi',
                         f'D = {D}', f'v = {v}', f'phase = {phase}', f'dx = {dx}', f'box = {box}',
                         f'rho_c = {S["rhoc"]:.8e}', f'M_sol = {Msol:.8e}', f'v_c = {S["vc"]:.6f}',
                         f'nu = {NU:.6f}', f'Myr_per_code = {MYR:.6f}', f'complete = {int(final)}',
                         ','.join(COLS)])
        np.savetxt(out + '.csv', np.array(rows), delimiter=',', header=hdr, fmt='%.8e')
        tm = np.array(rows)[::map_every, 0]
        xe = ax[0][0] - 0.5 * dx + np.arange(maps[0].shape[0] + 1) * f * dx
        ye = ax[1][jy][0] - 0.5 * dx + np.arange(maps[0].shape[1] + 1) * f * dx
        np.savez_compressed(out + '_maps.npz', t_Myr=tm, maps=np.array(maps, dtype=np.float32), xedges=xe, yedges=ye)

    Phi = potential(np.abs(psi)**2)
    r, P = diag(0.0, psi, Phi)
    rows, maps = [r], [P]
    kick = np.exp(-0.5j * dt / NU * Phi) * sp_half
    nout = int(round(tend / tout))
    t, t0 = 0.0, time.time()
    for io in range(1, nout + 1):
        for _ in range(nsub):                     # half kick with Phi(t_n), drift, half kick with Phi(t_n+1)
            psi *= kick
            psi = sfft.ifftn(sfft.fftn(psi, workers=workers) * kprop, workers=workers)
            Phi = potential(np.abs(psi)**2)
            kick = np.exp(-0.5j * dt / NU * Phi) * sp_half
            psi *= kick
            t += dt
        r, P = diag(t, psi, Phi)
        rows.append(r)
        if io % map_every == 0:
            maps.append(P)
        print(f't {r[0]:7.1f} Myr  M_box {r[1]:.5f}  M10 {r[2]:.4f}  L x {r[3]:+7.2f} m {r[4]:.3f} rho {r[5]:.3f}  '
              f'R x {r[6]:+7.2f} m {r[7]:.3f} rho {r[8]:.3f}  E {r[9] + r[10]:.5e}  wall {time.time() - t0:.0f} s',
              flush=True)
        if io % 10 == 0 or io == nout:
            save(rows, maps, io == nout)
    print('wrote', out + '.csv', flush=True)


def solve_halo(Nsol, R, phase, dx, tend, tout, ck, workers, map_every, out, box=HALO_BOX):
    """Case 4 reference (see the module docstring); the same integrator as solve()."""
    import scipy.fft as sfft
    S = sp_groundstate()
    NU, MYR = S['nu'], myr_per_code()
    pos = halo_positions(Nsol, R)
    th = np.zeros(len(pos)) if phase == 0 else np.random.default_rng(100 + int(phase)).uniform(0, 2 * np.pi, len(pos))
    N = [int(round(box / dx))] * 3
    ax = [(np.arange(n) + 0.5) * dx - 0.5 * n * dx for n in N]
    X, Y, Z = np.meshgrid(*ax, indexing='ij', sparse=True)
    amp = lambda r: np.sqrt(np.interp(r, S['r'], S['rho'], right=0.0))
    psi = sum(amp(np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2)) * np.exp(1j * t) for c, t in zip(pos, th))
    M0 = (np.abs(psi)**2).sum() * dx**3
    k2 = sum(np.meshgrid(*[(2 * np.pi * sfft.fftfreq(n, dx))**2 for n in N], indexing='ij', sparse=True))
    NP = [2 * n for n in N]
    rp = np.sqrt(sum(c**2 for c in np.meshgrid(*[np.minimum(np.arange(m), m - np.arange(m)) * dx for m in NP],
                                                indexing='ij', sparse=True)))
    g = -dx**3 / np.where(rp > 0, rp, 1.0)
    g[0, 0, 0] = -2.380077 * dx**2
    Ghat = sfft.rfftn(g, workers=workers)
    del g, rp
    pad = np.zeros(NP)

    def potential(rho):
        pad[:N[0], :N[1], :N[2]] = rho
        return sfft.irfftn(sfft.rfftn(pad, workers=workers) * Ghat, s=NP, workers=workers)[:N[0], :N[1], :N[2]]

    depth = np.maximum(np.maximum(*[np.maximum(0.0, np.abs(a) - (0.5 * n * dx - SPONGE)) / SPONGE
                                    for a, n in zip((X, Y), N)]),
                       np.maximum(0.0, np.abs(Z) - (0.5 * N[2] * dx - SPONGE)) / SPONGE)
    Lt = tout / MYR
    nsub = int(np.ceil(Lt / (ck * dx**2 / NU)))
    dt = Lt / nsub
    sp_half = np.exp(-0.5 * dt * (5.0 * 2.5e4 / SPONGE) * depth**2)
    kprop = np.exp(-0.5j * NU * dt * k2)
    f = max(1, int(round(MAP_PIX / dx)))
    vol = 4.0 / 3.0 * np.pi * np.diff(HALO_REDGES**3)
    print(f'halo: {len(pos)} solitons in {R} rc, phases {np.round(th, 3).tolist()}; grid {N} dx {dx}, dt {dt:.3e} '
          f'code, {nsub} steps per {tout:.3f} Myr output, t_end {tend:.1f} Myr, M0 / N M_sol '
          f'{M0 / (len(pos) * S["M"][-1]):.5f}, FFT workers {workers}', flush=True)

    def diag(t, psi, Phi):
        rho = np.abs(psi)**2
        K = 0.5 * NU**2 * (k2 * np.abs(sfft.fftn(psi, workers=workers))**2).sum() / psi.size * dx**3
        i = np.unravel_index(np.argmax(rho), N)
        c = [ax[j][i[j]] for j in range(3)]
        r = np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2)
        row = [t * MYR, rho.sum() * dx**3 / M0, rho.max() / S['rhoc'], *c]
        row += [(rho * (r < rr)).sum() * dx**3 / M0 for rr in HALO_R]
        row += [K, 0.5 * (rho * Phi).sum() * dx**3]
        prof = np.histogram(r.ravel(), HALO_REDGES, weights=rho.ravel())[0] * dx**3 / vol
        P = rho.sum(axis=2) * dx
        P = P[:P.shape[0] // f * f, :P.shape[1] // f * f]
        return row, prof, P.reshape(P.shape[0] // f, f, P.shape[1] // f, f).sum((1, 3)) * dx * dx

    def save(rows, profs, maps, final):
        hdr = '\n'.join([f'soliton halo reference (soliton_reference.py --halo): Schroedinger-Poisson, phase realisation '
                         f'{phase}', f'N = {Nsol}', f'R = {R}', f'phase = {phase}', f'theta = {np.round(th, 6).tolist()}',
                         f'dx = {dx}', f'box = {box}', f'rho_c = {S["rhoc"]:.8e}', f'M_sol = {S["M"][-1]:.8e}',
                         f'M0 = {M0:.8e}', f'nu = {NU:.6f}', f'Myr_per_code = {MYR:.6f}', f'complete = {int(final)}',
                         ','.join(HALO_COLS)])
        np.savetxt(out + '.csv', np.array(rows), delimiter=',', header=hdr, fmt='%.8e')
        np.savez_compressed(out + '_prof.npz', t_Myr=np.array(rows)[:, 0], redges=HALO_REDGES, rho=np.array(profs))
        e = ax[0][0] - 0.5 * dx + np.arange(maps[0].shape[0] + 1) * f * dx
        np.savez_compressed(out + '_maps.npz', t_Myr=np.array(rows)[::map_every, 0],
                            maps=np.array(maps, dtype=np.float32), xedges=e, yedges=e)

    Phi = potential(np.abs(psi)**2)
    r, pr, P = diag(0.0, psi, Phi)
    rows, profs, maps = [r], [pr], [P]
    kick = np.exp(-0.5j * dt / NU * Phi) * sp_half
    nout = int(round(tend / tout))
    t, t0 = 0.0, time.time()
    for io in range(1, nout + 1):
        for _ in range(nsub):
            psi *= kick
            psi = sfft.ifftn(sfft.fftn(psi, workers=workers) * kprop, workers=workers)
            Phi = potential(np.abs(psi)**2)
            kick = np.exp(-0.5j * dt / NU * Phi) * sp_half
            psi *= kick
            t += dt
        r, pr, P = diag(t, psi, Phi)
        rows.append(r)
        profs.append(pr)
        if io % map_every == 0:
            maps.append(P)
        print(f't {r[0]:7.1f} Myr  M_box {r[1]:.5f}  rho_max {r[2]:.3f}  peak ({r[3]:+.2f},{r[4]:+.2f},{r[5]:+.2f})  '
              f'm1 {r[6]:.3f} m2 {r[7]:.3f} m5 {r[8]:.3f} m10 {r[9]:.3f}  E {r[10] + r[11]:.5e}  '
              f'wall {time.time() - t0:.0f} s', flush=True)
        if io % 10 == 0 or io == nout:
            save(rows, profs, maps, io == nout)
    print('wrote', out + '.csv', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-b', type=float, default=None, dest='D',
                    help='separation / r_c (runtest -b, default 10); --halo: number of solitons (default 4)')
    ap.add_argument('-c', type=float, default=None, dest='v',
                    help='relative speed / v_c(r_c) (runtest -c, default 6); --halo: radius / r_c (default 6)')
    ap.add_argument('--halo', action='store_true', help='case 4, the multi-soliton halo')
    ap.add_argument('--phase', type=float, nargs='+', default=None,
                    help='relative phase(s) / pi (default 0 1, plus 0.5 for a bound collision)')
    ap.add_argument('--dx', type=float, default=0.125, help='grid spacing, kpc (default 0.125)')
    ap.add_argument('--box', type=float, nargs=3, default=None, help='Lx Ly Lz, kpc (default: see default_box)')
    ap.add_argument('--tend', type=float, default=None, help='Myr (default: the particle run length)')
    ap.add_argument('--tout', type=float, default=None, help='output interval, Myr (default dDelta, 7.9 Myr)')
    ap.add_argument('--ck', type=float, default=1.0 / 6.0, help='dt = ck dx^2 / nu (default 1/6)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='FFT threads')
    ap.add_argument('--map-every', type=int, default=None,
                    help="outputs between saved maps (default: the particle run's iOutInterval)")
    ap.add_argument('--outdir', default=REFDIR)
    a = ap.parse_args()
    S = sp_groundstate()
    if a.halo:
        Nsol, R = (a.D if a.D is not None else 4), (a.v if a.v is not None else 6.0)
        nsteps, freqout = run_length(4)
        tend = a.tend if a.tend is not None else nsteps * DDELTA * myr_per_code()
        tout = a.tout if a.tout is not None else DDELTA * myr_per_code()
        os.makedirs(a.outdir, exist_ok=True)
        for p in (a.phase if a.phase is not None else [0, 1, 2]):
            solve_halo(Nsol, R, int(p), a.dx, tend, tout, a.ck, a.workers, a.map_every or freqout,
                       halo_path(Nsol, R, int(p), a.outdir)[:-4], box=a.box[0] if a.box else HALO_BOX)
        return
    a.D = 10.0 if a.D is None else a.D
    a.v = 6.0 if a.v is None else a.v
    nsteps, freqout = run_length(3, a.D, a.v)
    tend_code = nsteps * DDELTA
    tend = a.tend if a.tend is not None else tend_code * myr_per_code()
    tout = a.tout if a.tout is not None else DDELTA * myr_per_code()
    box = tuple(a.box) if a.box else default_box(a.D, a.v, tend / myr_per_code(), S)
    phases = a.phase if a.phase is not None else ([0.0, 1.0] if collision_energy(a.D, a.v, S) > 0 else [0.0, 1.0, 0.5])
    os.makedirs(a.outdir, exist_ok=True)
    for p in phases:
        solve(a.D, a.v, p, a.dx, box, tend, tout, a.ck, a.workers, a.map_every or freqout,
              reference_path(a.D, a.v, p, a.outdir)[:-4])


if __name__ == '__main__':
    main()
