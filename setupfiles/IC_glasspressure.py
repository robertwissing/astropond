"""Glass-IC pressure post-step: runtest.sh runs it after IC_createsetup.py has built
datafiles/<IC>.00000 from a glass (distrib 2). Knobs in relax.config.

1. Pressure-bump check (always). The glass is force-free under its pseudo-pressure pP =
   (rho/target)^rhopow, not under uniform pressure. At sharp density features pP carries a
   bump b = ln pP (kernel-averaged: <glass>_IC.pPs); a warning is printed when max|b| exceeds
   GLASS_PBUMP_WARN. A uniform-P start then launches a coherent flow v/c_s ~ max|b|/gamma
   and settles to ~70-85% of the bump anyway; a pseudo-P start keeps the bump as a pressure
   error.
2. GLASS_START_P=pseudo: each particle's pressure is multiplied by pP/median(pP), so the run
   starts in the glass's force balance. The setup's (rho, u) columns give its intended pressure
   P = (gamma-1) rho u; the new u = P pP/median(pP) / ((gamma-1) rho_glass), as the code's t=0
   density is the glass's. Tested for a uniform setup pressure (static slab); for a non-uniform
   one it is the leading-order extension. Needs the IC particles to be the glass particles,
   in order and unmoved; a setup that crops, replicates or moves them is refused.
3. A force-matched glass (relax.config GLASS_FORCEMATCH=1) is relaxed under the real pressure, not pP:
   its .pPs measures rho/rho_t, not a pressure error, so the check is skipped and pseudo refused.
usage: python IC_glasspressure.py <IC>.00000 <glass>_IC <uniform|pseudo> <warn> [forcematch 0|1]
(pseudo rewrites <IC>.00000 in place)"""
import os
import sys

import numpy as np

import readtipsy as tip


def main():
    icf, glass, start, warn = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    if len(sys.argv) > 5 and sys.argv[5] == '1':
        if start != 'uniform':
            sys.exit('GLASS_START_P=pseudo with a force-matched glass: use GLASS_START_P=uniform')
        print('force-matched glass: relaxed under the real pressure; pseudo-pressure bump check skipped')
        return
    if os.path.exists(glass + '.pPs'):
        pps = tip.readtipsyaux(glass, 'pPs')[0].astype(np.float64)
        b = np.log(pps / np.median(pps))
        k = int(np.argmax(np.abs(b)))
        print(f'glass pressure bump: max|ln pPs| = {abs(b[k]):.2e} ({(np.abs(b) > warn).sum()} particles '
              f'above GLASS_PBUMP_WARN = {warn:g})')
        if abs(b[k]) > warn:
            print(f'WARNING: the glass holds a pressure bump of {abs(b[k]):.1%} at sharp density features. '
                  f'A uniform-P start launches a coherent flow v/c_s ~ {abs(b[k]):.1e}/gamma there; a '
                  f'pseudo-P start (GLASS_START_P=pseudo) starts quiet but keeps this pressure error. '
                  f'Widen the sharp features of the target to reduce it.')
    else:
        print(f'no {glass}.pPs (glass written before the pressure-bump check): check skipped')
    if start == 'uniform':
        return
    assert start == 'pseudo', start
    g, d, s, hdr, time = tip.readtipsy(icf)
    g = np.array(g, dtype=np.float64)
    gg = tip.readtipsy(glass)[0].astype(np.float64)
    pp = tip.readtipsyaux(glass, 'pP')[0].astype(np.float64)
    if not len(g) == len(gg) == len(pp):
        sys.exit(f'GLASS_START_P=pseudo: the IC has {len(g)} gas particles, the glass {len(gg)} '
                 f'(pP {len(pp)}): this setup crops or replicates the glass; use GLASS_START_P=uniform')
    shift = np.abs(g[:, 1:4] - gg[:, 1:4]).max()
    if shift > 1e-6 * np.ptp(gg[:, 1:4], axis=0).max():
        sys.exit(f'GLASS_START_P=pseudo: this setup moves the glass particles (max shift {shift:.3e}); '
                 f'use GLASS_START_P=uniform')
    if (g[:, 7] <= 0).any() or (gg[:, 7] <= 0).any():
        sys.exit('GLASS_START_P=pseudo: the IC or glass has rho <= 0 particles')
    fac = pp / np.median(pp) * g[:, 7] / gg[:, 7]
    g[:, 8] *= fac
    tip.writetipsy(g, d, s, icf, hdr, time)
    print(f'GLASS_START_P=pseudo: {icf} pressure x pP/median(pP): factor {fac.min():.4f} .. {fac.max():.4f}')


if __name__ == '__main__':
    main()
