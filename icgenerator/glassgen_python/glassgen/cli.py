"""CLI: functional replacement for the gasoline.gdicgen invocation in
runtest.sh (tipsy pre-IC + density table in -> relaxed tipsy glass out).

Example, mirroring runtest.sh ICCONFIG:
    PYTHONPATH=icgenerator/glassgen_python python -m glassgen.cli \
        -o datafiles/myglass -I datafiles/pre.00000 -s 64 -n 4800 -oi 4800 \
        --param datafiles/pre.param --table densitytable.npz

--table takes the self-describing .npz density-table format (AMR block
layout, or a 1D/2D/3D grid whose coordinate mapping/extent live in the
table header); convert a legacy C-generator
densitytable_xdr once with `python -m glassgen.density`. Without --table
the target density is uniform (total gas mass / box volume). Output:
${out}_IC (tipsy, gas positions/density/hsmooth updated, velocities
zeroed) and ${out}.NNNNN snapshots every -oi iterations.
"""
import argparse
import os
import sys

import numpy as np

_SETUPFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', '..', 'setupfiles')
sys.path.insert(0, _SETUPFILES)

from .density import CallableDensity, TableDensity
from .params import (box_from_param, density_from_param, expert_help,
                     read_param)
from .relax import relax
from .kernels import KERNELS
from . import sph_core
from .sph_core import GRADIENTS


def _target_readout(density, pos, h):
    """Read the same target used by the relax pseudo-pressure."""
    if getattr(density, 'smooth_readout', False):
        return density.rho0_h(pos, h)
    return density.rho0(pos)


def write_glass_aux(out, pos, mass, h, rho, density, box, nsmooth, rhopow,
                    sigma_f, gradient, nsmoothmin, kernel):
    """Write the glass's fit/residual diagnostics as tipsy aux arrays beside
    ${out}_IC (never read back by the pipeline):
      .rho0   analytic target at each particle
      .rho0h  kernel-averaged target (--kernel-match only: the target used)
      .pP     residual pseudo-pressure (rho/target)^rhopow the relax balanced -
              a pressure field the glass is force-free under (same operator)
      .pPs    pP kernel-averaged over the neighbours: the pressure bump the glass
              holds at sharp density features, above the per-particle noise
              (runtest.sh warns from it, relax.config GLASS_PBUMP_WARN)
      .fres   residual relax force as dP/P across h, |a| rho h / pP (~0)
      .funi   force a UNIFORM-pressure IC starts with, |a| rho h / P
              (a h / c_s^2 = funi / gamma)"""
    import readtipsy as tip
    from .diagnostics import residual_force, smoothed_pP
    pos = np.asarray(pos, float)
    if box is not None:
        box = np.asarray(box, float)
        pos = pos - box * np.rint(pos / box)
    mass = np.broadcast_to(np.asarray(mass, float), (len(pos),))
    rho, h = np.asarray(rho, float), np.asarray(h, float)
    target = _target_readout(density, pos, h)
    pP = (rho / target) ** rhopow
    kw = dict(rhopow=rhopow, nsmooth=nsmooth, sigma_f=sigma_f, gradient=gradient,
              nsmoothmin=nsmoothmin, kernel=kernel)
    fres = residual_force(pos, mass, h, rho, target, box, **kw) * rho * h / pP
    funi = residual_force(pos, mass, h, rho, target, box, uniform_pP=True,
                          **kw) * rho * h
    pPs = smoothed_pP(pos, mass, h, rho, pP, box, nsmooth=nsmooth,
                      nsmoothmin=nsmoothmin, kernel=kernel)
    aux = {'rho0': density.rho0(pos), 'pP': pP, 'pPs': pPs, 'fres': fres, 'funi': funi}
    if getattr(density, 'smooth_readout', False):
        aux['rho0h'] = target
    for label, x in aux.items():
        tip.writetipsyaux(np.asarray(x, float), label, f'{out}_IC')
    print(f'wrote {out}_IC.{{{",".join(aux)}}}: median fres {np.median(fres):.2e}, '
          f'funi {np.median(funi):.2e} (p99 {np.percentile(funi, 99):.2e}), '
          f'pressure bump max|ln pPs| {np.abs(np.log(pPs / np.median(pPs))).max():.2e}',
          flush=True)


def write_denserr_plot(out, pos, mass, h, rho, density, box, nsmooth,
                       sigma_f=1.0, gradient='onesided-isph', rhopow=2.0,
                       nsmoothmin=32, kernel='wendlandc2'):
    """Write ${out}_denserr.png: the density-error |1-rho/rho0| and residual
    descent-force |icvel| distributions of the final glass (the force whose RMS
    the controller drives to equilibrium / stop), each annotated mean/p95/max.
    Self-contained (matplotlib only); the force reuses diagnostics.residual_force."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .diagnostics import residual_force

    pos = np.asarray(pos, float)
    if box is not None:
        box = np.asarray(box, float)
        pos = pos - box * np.rint(pos / box)
    rho = np.asarray(rho, float)
    mass = np.broadcast_to(np.asarray(mass, float), (len(pos),))  # callers pass the scalar mf
    rho0 = _target_readout(density, pos, h)
    err = np.abs(1.0 - rho / rho0)
    force = residual_force(pos, mass, h, rho, rho0, box, rhopow=rhopow,
                           nsmooth=nsmooth, sigma_f=sigma_f,
                           gradient=gradient, nsmoothmin=nsmoothmin,
                           kernel=kernel)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, lab in (
            (axes[0], err, r'$|1-\rho/\rho_0|$'),
            (axes[1], force, r'residual force $|\mathrm{icvel}|$')):
        ax.hist(data, bins=80, histtype='step', lw=1.8, color='C0')
        ax.set_yscale('log')
        ax.set_xlabel(lab); ax.set_ylabel('count')
        ax.set_title(f'mean {data.mean():.4g}  p95 {np.percentile(data,95):.4g}  '
                     f'max {data.max():.4g}')
    fig.suptitle(f'{os.path.basename(out)}: glass density error + residual force')
    fig.tight_layout()
    path = f'{out}_denserr.png'
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}  '
          f'(denserr mean/p95/max {err.mean():.4g}/{np.percentile(err,95):.4g}/'
          f'{err.max():.4g})', flush=True)


def _print_stages(res):
    for l in res.levels:
        print(f'  stage {l["name"]:<6} N={l["n"]:<8} niter={l["niter"]:<5} '
              f'{l["wall"]:.0f}s  denserr(mn/p95/mx)='
              f'{l["denserr_mean"]:.4f}/{l["denserr_p95"]:.4f}/'
              f'{l["denserr_max"]:.4f}  force(mn/p95/mx)='
              f'{l["force_mean"]:.2e}/{l["force_p95"]:.2e}/'
              f'{l["force_max"]:.2e}')


def main(argv=None):
    hide = expert_help(argv)
    ap = argparse.ArgumentParser(
        description='glassgen: glass-relaxation IC generator')
    ap.add_argument('--help-extra', action='help',
                    help='show the full help, including the hidden expert '
                         'options (relax internals, legacy compat)')
    ap.add_argument('-o', dest='out', required=True,
                    help='output prefix (writes ${out}_IC)')
    ap.add_argument('-I', dest='infile', default=None,
                    help='input tipsy pre-IC (omit it and pass --table + '
                         '--count to sample the initial particles from the '
                         'table itself)')
    ap.add_argument('--count', type=int, default=None, metavar='N',
                    help='build from the table alone: sample N initial '
                         'particles from --table (no -I) and relax toward it '
                         '- THE glassgen table builder (glassresample '
                         '--op reconstruct drives the same path)')
    ap.add_argument('-s', dest='nsmooth', type=int, default=64)
    ap.add_argument('-n', dest='max_iter', type=int, default=4800)
    ap.add_argument('-oi', dest='oi', type=int, default=0,
                    help=hide('snapshot output interval (0 = none)'))
    ap.add_argument('--table', default=None,
                    help='target density table (.npz, self-describing): an '
                         'AMR table (glasstable/glassresample) or a 1D/2D/3D '
                         'grid whose coordinate mapping lives in the table '
                         'header. Convert a legacy densitytable_xdr once '
                         'with `python -m glassgen.density`. Default: '
                         'uniform density')
    ap.add_argument('--force-table', dest='force_table', default=None,
                    help='force-matched relax: the real pressure A(r) rho^Gamma '
                         'plus the table g(r) (normally grad P_t / rho_t) instead '
                         'of the pseudo-pressure, so the glass reproduces the '
                         'target pressure acceleration (.npz with r, P, g, rho, '
                         'closure; see glassgen/forcematch.py)')
    ap.add_argument('--target-shape-errp', dest='target_shape_errp', type=float, default=0.0,
                    help='replace a 1D two-plateau .param contact by the curvature step '
                         'whose glass holds this pressure error max|ln pPs| (glassgen/shaping.py); '
                         '0 = the raw target')
    ap.add_argument('--kernel-match', action='store_true',
                    help='compare the pseudo-pressure density with the target '
                         'averaged over each particle kernel. Use for a smooth '
                         'profile whose faint tails or transition are narrower '
                         'than the local particle resolution; the point target '
                         'is recovered as resolution increases')
    ap.add_argument('--kernel', default='wendlandc2', choices=KERNELS,
                    help='SPH kernel of the relaxation (density, force, ISPH '
                         'matrix, kernel-match stencil); default wendlandc2')
    ap.add_argument('--sigma', type=float, default=0.0,
                    help='density partition exponent sigma (gasoline '
                         'CORRPARTITION dFacQ); only 0 = the plain SPH density '
                         'estimate is implemented')
    ap.add_argument('--sigma-f', dest='sigma_f', type=float, default=1.0,
                    help='force/gradient volume exponent sigma_f (gasoline '
                         'CORRPARTITION dFacQf): 1 = GDSPH (default), 0 = TSPH')
    ap.add_argument('--gradient', default='onesided-isph',
                    choices=sorted(GRADIENTS),
                    help='pair-force gradient: onesided-isph (default, '
                         'gasoline GDFORCE+ISPH, which forces ONESIDED), '
                         'onesided-sph (ONESIDED only), average-sph (the C '
                         'icgenerator / gasoline without either), average-isph')
    ap.add_argument('--isph-boundterm', dest='isph_boundterm', type=float, default=2.0,
                    help='ISPH fallback: plain gradient where the boundary term '
                         '|T^-1 I| h exceeds this (gasoline boundterm0; default 2.0)')
    ap.add_argument('--gradh', action='store_true',
                    help="gasoline's grad-h owner factor f_i = 3/tr(C_i T_i) on the force "
                         '(CORRPARTITION_GRADH_EXACT; one-sided gradients only)')
    ap.add_argument('--coarse-fft', dest='coarse_fft', type=float, default=0.0, metavar='TOP',
                    help='FFT coarse-grid correction for the long-wavelength modes (periodic '
                         'box, with --no-progressive): the re-anneal ladder starts at TOP*h '
                         'instead of the box, and every rung end and the polish entry apply '
                         'the correction (glassgen.coarse). 0 = off (default)')
    ap.add_argument('--coarse-fft-every', dest='coarse_fft_every', type=int, default=0, metavar='N',
                    help='--coarse-fft transport mode for a smooth profile far from the pre-IC '
                         '(soliton): also apply the correction every N iterations outside polish, '
                         'R from the median h, step capped by its strain (relax.coarse_every). '
                         'Not for sharp compact objects (areablob runs away). 0 = off (default)')
    ap.add_argument('--coarse-fft-serial', dest='coarse_fft_serial', action='store_true',
                    help='--coarse-fft: deposit on one thread, so reruns are bitwise identical '
                         '(A/B tests); default parallel, ~1%% faster, reruns differ like seeds')
    ap.add_argument('--neighbours', choices=('constant', 'variable'),
                    default='constant',
                    help='constant = kNN h from the nSmooth-th neighbour '
                         '(gasoline default); variable = h from m/rho with '
                         'a ball gather (CORRPARTITION-style h)')
    ap.add_argument('--rhopow', type=float, default=2.0, help='ICrhopow')
    ap.add_argument('--icr0', type=float, default=1.0, help=hide('ICR0'))
    ap.add_argument('--icrloop0', type=float, default=0.5,
                    help=hide('ICRLOOP0'))
    ap.add_argument('--icr0rate', type=float, default=1.5,
                    help=hide('ICR0Rate'))
    ap.add_argument('--margin', type=int, default=8,
                    help=hide('extra kNN candidates kept per particle; larger '
                              'values widen the Verlet skin (fewer rebuilds) '
                              'at slightly higher per-iteration cost'))
    ap.add_argument('--transpose-free', dest='one_sided', action='store_true',
                    help=hide('SPH-EXA-style transpose-free force walk: '
                              'faster, pair forces no longer exactly '
                              'antisymmetric'))
    ap.add_argument('--h-mode', dest='hmode', default=None,
                    choices=('knn', 'measured', 'target'),
                    help=hide('smoothing length for the force kernel + move '
                              'step: knn (default), measured '
                              '(=(m*hfact3/rho_sph)^1/3, the C density-based '
                              'h), or target (=(m*hfact3/rho0)^1/3, '
                              'noise-free analytic h); overrides '
                              '--neighbours. knn is the production mode'))
    ap.add_argument('--nsmoothmin', type=int, default=32,
                    help=hide('floor on density-based h: never gather fewer '
                              'than ~nsmoothmin real neighbours '
                              '(measured/target only)'))
    ap.add_argument('--no-progressive', dest='progressive',
                    action='store_false',
                    help='disable the DEFAULT multi-resolution path and run a '
                         'single full-resolution relax. The default coarsens '
                         'the input to the auto density-chosen coarse count, '
                         'relaxes it, then cascades (x2 splits + confined heals) '
                         'back up to full res.')
    ap.add_argument('--coarse-target', dest='coarse_target', default='auto',
                    help='progressive coarse particle count: "auto" picks '
                         'N_coarse_min from the target density (resolves the '
                         'steepest feature); an integer forces a coarse count. '
                         'The merge factor is floor_pow2(N_full/target); '
                         'factor 1 => no merge / direct relax (default auto)')
    ap.add_argument('--alpha', type=float, default=4.0,
                    help='cascade resolvability tolerance: the auto coarse '
                         'count bound (default 4.0)')
    ap.add_argument('--nx-full', dest='nx_full', type=int, default=0,
                    help=hide('full target resolution (default: '
                              'round(N**(1/3)) from the input)'))
    ap.add_argument('--inter-iter', dest='inter_iter', type=int, default=10,
                    help=hide('icgen iterations after each intermediate x2 '
                              'split in the progressive cascade (default 10)'))
    ap.add_argument('--momentum', type=float, default=0.5,
                    help=hide('heavy-ball coefficient on the post-split '
                              'confined stages, active ONLY in the polish '
                              'phase (hardwired 2026-07-22; default 0.5; '
                              '0 = off)'))
    ap.add_argument('--icr0-stop', dest='icr0_stop', type=float, default=3e-3,
                    help='median-smoothed icr0 stop (max step as a fraction of '
                         'h) - the quality dial: 3e-3 fast/good, ~1e-3 tight. '
                         '0 = off (then -n / --polish-iter ends the run)')
    ap.add_argument('--polish-iter', dest='polish_iter', type=int, default=0,
                    help='fixed polish-iteration budget on the fine stage '
                         '(0 = off; composes with --icr0-stop)')
    ap.add_argument('--polish-down', dest='polish_down', type=float,
                    default=0.1,
                    help=hide('polish-phase icr0 shrink coefficient (C '
                              'default 0.1 = a 47:1 ratchet vs --polish-up '
                              'that freezes the run shortly after polish '
                              'entry; ~0.01 keeps icr0 mobile for deep '
                              'fixed-budget polishes)'))
    ap.add_argument('--polish-up', dest='polish_up', type=float,
                    default=0.002,
                    help=hide('polish-phase icr0 grow coefficient (C '
                              'default 0.002)'))
    ap.add_argument('--no-self-bias', dest='self_bias', action='store_false',
                    help=hide('disable the Wendland C2 self-density bias '
                              'correction (Wzero10); density then uses the '
                              'bare W0 self term'))
    ap.add_argument('--envelope', choices=('auto', 'on', 'off'), default='auto',
                    help=hide('free-surface / open-box vacuum padding: '
                              'pad the vacuum side of '
                              'the gas with a mass-consistent background so '
                              'edge kernels fill, relax the union, then cull '
                              'the background. "auto" (default) = on for an '
                              '--open box, off for a filled periodic box '
                              '(byte-identical to no envelope); "on" forces '
                              'it for a periodic box that contains vacuum'))
    ap.add_argument('--nedge', dest='bg_c', type=int, default=0,
                    help='envelope: resolvability contour count - INTEGRAL: '
                         'rho_edge = where the cumulative faint-end mass '
                         'first exceeds nedge*m (the whole discarded tail in '
                         'particle masses, NOT a per-kernel count); 0 = use '
                         '--nsmooth (-s)')
    ap.add_argument('--rho-edge-max', dest='rho_edge_max', type=float,
                    default=None,
                    help=hide('envelope contract: require the computed '
                              'resolution-dependent rho_edge to be strictly '
                              'below this value; useful when a fixed physical '
                              'buffer must remain on the gas side of the cull'))
    ap.add_argument('--bg-over', dest='bg_over', type=float, default=0.0,
                    help=hide('envelope background oversubscription as a '
                              'fraction of the requested gas count; supplies '
                              'the post-relaxation cull skin without changing '
                              'the target or occupancy contour (default 0)'))
    ap.add_argument('--nframe', dest='n_frame', type=float, default=None,
                    help='envelope: floor backability - expected '
                         'particles per edge kernel at the profile floor '
                         '(rho_frame = rho_edge*nframe/nsmooth, i.e. '
                         'k = (nsmooth/nframe)**(1/6); default 16 ~ k1.26)')
    ap.add_argument('--nl0', dest='nl0', type=float, default=None,
                    help='envelope: scaffold launch resolvability '
                         'N_L(0) - the near-gas e-fold length in kernel '
                         'supports; tail_reach is solved per build (default '
                         '1.35; avoid ~1, the metastable resonance band)')
    ap.add_argument('--open-pad', dest='open_pad', type=float, default=0.1,
                    help='envelope: vacuum margin (fraction of the gas bbox per '
                         'side) of the synthesised periodic frame for an --open '
                         'IC (default 0.1)')
    ap.add_argument('--plot', action='store_true',
                    help='after relaxation, write ${out}_denserr.png: the '
                         'density-error |1-rho/rho0| and residual-force |icvel| '
                         'distributions of the final glass (mean/p95/max)')
    ap.add_argument('--history-plot', dest='history_plot', action='store_true',
                    help='after relaxation, write ${out}_history.png: the '
                         'convergence history (RMS residual force, density '
                         'error, step icr0 vs iteration). For the progressive '
                         'path this is the FINAL stage; add --no-progressive '
                         'for the whole single-relax trajectory.')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--param', help='gasoline .param file to take the box from')
    grp.add_argument('--box', nargs=3, type=float, metavar=('LX', 'LY', 'LZ'))
    grp.add_argument('--open', action='store_true', help='no periodic box')
    args = ap.parse_args(argv)
    sph_core.ISPH_BOUNDTERM = args.isph_boundterm
    if args.isph_boundterm != 2.0:
        print(f'ISPH boundary-term threshold {args.isph_boundterm}', flush=True)
    sph_core.GRADH = args.gradh
    if args.gradh:
        print('grad-h owner factor 3/tr(C T) on', flush=True)
    if args.hmode is None:
        args.hmode = 'knn' if args.neighbours == 'constant' else 'measured'

    if (args.rho_edge_max is not None and
            (not np.isfinite(args.rho_edge_max) or args.rho_edge_max <= 0.0)):
        ap.error('--rho-edge-max must be positive')
    if not np.isfinite(args.bg_over) or args.bg_over < 0.0:
        ap.error('--bg-over must be finite and non-negative')

    if args.target_shape_errp > 0.0 and (args.table or args.force_table or args.open
                                         or args.count is not None or not args.param):
        ap.error('--target-shape-errp shapes a 1D .param contact: it needs --param, '
                 'and no --table, --force-table, --open or --count')

    if args.count is not None:
        if not args.table:
            ap.error('--count builds from a table: pass --table FILE.npz')
        if args.force_table:
            ap.error('--force-table is not wired into the --count table build')
        if args.infile:
            ap.error('--count samples the initial particles from the table '
                     'itself; drop -I (or drop --count to relax the pre-IC)')
        if args.kernel_match:
            ap.error('--count already applies its N-aware kernel-matched table '
                     'builder; do not also pass --kernel-match')
    elif not args.infile:
        ap.error('-I is required (or pass --table + --count to build from '
                 'the table alone)')

    import readtipsy as tip

    if args.infile:
        gas, dark, star, header, time = tip.readtipsy(args.infile)
        if len(gas) == 0:
            sys.exit('no gas particles in input')
        print(f'read {args.infile}: {len(gas)} gas particles', flush=True)
        pos = gas[:, 1:4].astype(np.float64)
        mass = gas[:, 0].astype(np.float64)
    else:                                # --table + --count: no input particles
        gas = np.zeros((0, 12), dtype=np.float32)
        dark = np.zeros((0, 9), dtype=np.float32)
        star = np.zeros((0, 11), dtype=np.float32)
        header = np.array([0, 3, 0, 0, 0, 0], dtype=np.int32)
        time = np.zeros(1, dtype=np.float64)
        pos = mass = None

    param_dict = read_param(args.param) if args.param else None
    if args.open:
        box = None
    elif args.box:
        box = np.array(args.box)
    else:
        box = box_from_param(param_dict)

    if args.table:
        if not args.table.endswith('.npz'):
            sys.exit(f'{args.table}: --table takes the self-describing .npz '
                     'density-table format. Convert a legacy '
                     'densitytable_xdr once with:\n'
                     f'  python -m glassgen.density {args.table} '
                     'table.npz --direction r_sph')
        # self-describing .npz: AMR block layout (glasstable/glassresample
        # --save-table, targeted directly = reconstruct-in-place) or a
        # uniform/mapped grid carrying its own coordinate mapping
        from .reconstruct import describe_table, load_table
        density = load_table(args.table)
        print(f'loaded density table {args.table} '
              f'({describe_table(density)})', flush=True)
    elif param_dict is not None:
        # read the density profile straight from the .param (profiles 1/2/3),
        # exactly as the C gdicgen does - no density table needed for these.
        density = density_from_param(param_dict)
        if args.target_shape_errp > 0.0:
            from .shaping import curvature_target
            try:
                density, beta2, merr = curvature_target(param_dict, mass.mean(), args.nsmooth,
                                                        args.target_shape_errp)
            except ValueError as err:
                sys.exit(f'--target-shape-errp: {err}')
            print(f'target shape: curvature step for err_P {args.target_shape_errp:g}: B2 {beta2:.4g}, '
                  f'mass error {merr:.1e}', flush=True)
    else:
        # --box without a .param: uniform target (total gas mass / box volume)
        if box is None:
            sys.exit('uniform density needs a box (--box)')
        rho_mean = mass.sum() / np.prod(box)
        density = CallableDensity(lambda p, r=rho_mean: np.full(len(p), r))

    if args.count is not None:
        # ---- THE table builder (no input particles): sample --count from the
        # table, N-aware envelope, progressive relax, cull + renorm - the same
        # path glassresample --op reconstruct drives. -------------------------
        if getattr(density, 'dim', 3) != 3:
            sys.exit(f'--count samples cells from a 3D table; {args.table} '
                     f'is a {density.dim}D mapped table - relax a pre-IC '
                     'against it with -I instead')
        from .progressive import build_from_table
        ct = args.coarse_target
        ct = ct if ct == 'auto' else int(ct)
        print(f'table build: sampling {args.count} particles from the table '
              f'(envelope {args.envelope}, cascade '
              f'{"on" if args.progressive else "off"})', flush=True)
        res = build_from_table(
            density, args.count, box=box, nsmooth=args.nsmooth,
            bg_c=args.bg_c, envelope=args.envelope,
            n_frame=args.n_frame, nl0=args.nl0,
            bg_over=args.bg_over,
            cascade=args.progressive, open_pad=args.open_pad, verbose=True,
            coarse_target=ct, alpha=args.alpha, inter_iter=args.inter_iter,
            max_iter=args.max_iter, momentum=args.momentum,
            icr0_stop=args.icr0_stop, polish_iter=args.polish_iter,
            polish_down=args.polish_down, polish_up=args.polish_up,
            cap=2.0, start_icr0=2.0,
            kernel=args.kernel, sigma=args.sigma,
            sigma_f=args.sigma_f,
            gradient=args.gradient, rhopow=args.rhopow, icr0rate=args.icr0rate,
            margin=args.margin, one_sided=args.one_sided, hmode=args.hmode,
            nsmoothmin=args.nsmoothmin, self_bias=args.self_bias)
        a = res.envelope_audit
        nf, mf = len(res.pos), a['mf']
        tb = res.table_build
        if 'rho_edge' in tb:
            print(f'envelope: m={tb["m"]:.3e} rho_edge={tb["rho_edge"]:.3e} '
                  f'rho_frame={tb["rho_frame"]:.3e} '
                  f'delta_bg={tb["delta_bg"]:.3e}; cull {a["n_full"]} -> {nf} '
                  f'(culled {a["n_culled"]}, bg {a["n_background"]})',
                  flush=True)
        g = np.zeros((nf, 12), dtype=np.float32)
        g[:, 0] = mf
        g[:, 1:4] = res.pos
        g[:, 7] = res.rho
        g[:, 9] = res.h
        hdr = header.copy()
        hdr[0] = nf
        hdr[2] = nf
        tip.writetipsy(g, dark, star, f'{args.out}_IC', hdr, time)
        write_glass_aux(args.out, res.pos, mf, res.h, res.rho, density, box,
                        args.nsmooth, args.rhopow, args.sigma_f,
                        args.gradient, args.nsmoothmin, args.kernel)
        err = np.abs(1.0 - res.rho /
                     _target_readout(density, res.pos, res.h))
        print(f'done (table build, count={args.count}): final N={nf}, '
              f'per-particle mass {mf:.3e}, mean|1-rho/rho0|={err.mean():.3e},'
              f' wrote {args.out}_IC')
        _print_stages(res)
        if args.plot:
            write_denserr_plot(args.out, res.pos, mf, res.h, res.rho, density,
                               box, args.nsmooth, sigma_f=args.sigma_f,
                               gradient=args.gradient,
                               kernel=args.kernel,
                               rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
        if args.history_plot:
            from .diagnostics import plot_history
            plot_history(res.history, f'{args.out}_history.png',
                         title=f'{os.path.basename(args.out)} (all levels)',
                         stop_reason=res.stop_reason,
                         levels=res.level_history)
            print(f'  wrote {args.out}_history.png', flush=True)
        return

    def write_snapshot(path, pos_now, rho_now, h_now):
        g = gas.copy()
        g[:, 1:4] = pos_now
        g[:, 4:7] = 0.0
        g[:, 7] = rho_now
        g[:, 9] = h_now
        tip.writetipsy(g.astype(gas.dtype), dark, star, path, header, time)

    # --- envelope (free-surface / open-box padding) ---------------------------
    #   auto: on for an open box, off for a filled periodic box (-> byte-identical
    #   to no envelope). An open box ALWAYS routes through relax_progressive (a
    #   plain open relax cannot contain the free surface); --no-progressive there
    #   means cascade off = a single full-res relax of the placed ensemble.
    envelope_on = args.envelope == 'on' or (args.envelope == 'auto' and box is None)
    env = None
    open_seed_table = None
    if envelope_on:
        from .amrtable import AMRTable
        from .envelope import (build_open_density_envelope,
                               build_table_envelope)
        bg_c = args.bg_c or args.nsmooth
        physical_table = (isinstance(density, AMRTable) or
                          (isinstance(density, TableDensity) and
                           density.dim == 3))
        if box is None:
            # Open -I runs use the proven --count prescription.  The input
            # supplies N/M/header/extent, not a seed worth preserving: draw a
            # fresh density-weighted seed from the physical 3-D table or from a
            # Cartesian volume quadrature of an analytic/mapped target.
            if physical_table:
                env = build_table_envelope(
                    density, float(mass.sum()), len(pos), box=None, c=bg_c,
                    nsmooth=args.nsmooth, pad_frac=args.open_pad,
                    n_frame=args.n_frame, nl0=args.nl0,
                    bg_over=args.bg_over)
                open_seed_table = density
            else:
                env = build_open_density_envelope(
                    density, pos, float(mass.sum()), len(pos), c=bg_c,
                    nsmooth=args.nsmooth, pad_frac=args.open_pad,
                    n_frame=args.n_frame, nl0=args.nl0,
                    bg_over=args.bg_over)
                open_seed_table = env['seed_table']
        elif isinstance(density, AMRTable):
            # periodic AMR table: table-native envelope, retaining the
            # supplied -I positions.
            env = build_table_envelope(density, float(mass.sum()), len(pos),
                                       box=box, c=bg_c,
                                       n_frame=args.n_frame, nl0=args.nl0,
                                       bg_over=args.bg_over)
        else:
            # MIXED boundaries: the .param/--box frame stays periodic (a
            # slab's z wrap) and the target's free surfaces live inside it,
            # carried by the same scaffold + cull the open builds use.
            # The -I seed is kept: the caller's box already frames the relax.
            env = build_open_density_envelope(
                density, pos, float(mass.sum()), len(pos), box=box, c=bg_c,
                nsmooth=args.nsmooth, pad_frac=args.open_pad,
                n_frame=args.n_frame, nl0=args.nl0,
                bg_over=args.bg_over)
        print(f'envelope: m={env["m"]:.3e} rho_edge={env["rho_edge"]:.3e} '
              f'rho_frame={env["rho_frame"]:.3e} delta_bg={env["delta_bg"]:.3e} '
              f'background N={len(env["bg_pos"])} '
              f'(top-up {env.get("n_bg_topup", 0)})', flush=True)
        if (args.rho_edge_max is not None and
                env['rho_edge'] >= args.rho_edge_max):
            sys.exit(
                f'envelope rho_edge={env["rho_edge"]:.6g} is not below the '
                f'required {args.rho_edge_max:.6g}; increase the particle '
                'resolution or widen/lower the disposable target skirt')
        if len(env['bg_pos']) == 0 and args.envelope == 'auto':
            print('envelope: no sub-rho_edge cells -> filled box, disabling',
                  flush=True)
            env = None
            open_seed_table = None
    elif args.rho_edge_max is not None:
        sys.exit('--rho-edge-max requires an active envelope')

    if env is not None and open_seed_table is not None:
        from .envelope import sample_table
        n_seed = len(pos)
        gas_mass = float(mass.sum())
        pos = sample_table(open_seed_table, n_seed, seed=0)
        mass = np.full(n_seed, gas_mass / n_seed)
        qn = env.get('quadrature_ngrid')
        detail = (f' from {qn}^3 Cartesian quadrature'
                  if qn is not None else ' from the physical 3-D table')
        print(f'open envelope: drew {n_seed} density-weighted seed particles'
              f'{detail}; -I supplies count/mass/extent only', flush=True)

    # Kernel matching must wrap the COMPLETE relaxation target.  In envelope
    # mode that is the base+scaffold PaddedDensity, not merely the raw gas
    # profile used above to determine the sharp occupancy contour.  This
    # mirrors progressive.build_from_table and preserves a pointwise contour
    # while comparing SPH rho to the kernel average on both sides of it.
    if args.kernel_match:
        from .density import KernelMatchedDensity
        target = env['density'] if env is not None else density
        density = KernelMatchedDensity(target, kernel=args.kernel,
                                       nsmooth=args.nsmooth)
        if env is not None:
            env['density'] = density
        print('kernel-matched target readout enabled', flush=True)

    if args.coarse_fft > 0.0 and (args.progressive or env is not None or box is None):
        sys.exit('--coarse-fft needs a periodic box, --no-progressive and no envelope')
    if args.coarse_fft_every > 0 and args.coarse_fft <= 0.0:
        sys.exit('--coarse-fft-every needs --coarse-fft')
    force_table = None
    if args.force_table:
        if args.coarse_fft > 0.0:
            sys.exit('--force-table and --coarse-fft cannot be combined')
        from .forcematch import ForceTable
        force_table = ForceTable(args.force_table)
        print(f'force-matched relax: {args.force_table} (closure Gamma={force_table.gamma})',
              flush=True)
    if args.progressive or env is not None:
        if box is None and env is None:
            sys.exit('--progressive needs a periodic box (--param or --box), or '
                     'use --open with the envelope (default for --open)')
        from .progressive import relax_progressive
        nx_full = args.nx_full or int(round(len(gas) ** (1.0 / 3.0)))
        # merge the input down to the density-chosen coarse count (auto
        # N_coarse_min, alpha), relax, then cascade (x2 splits + confined heals,
        # momentum on the post-split stages) back to the input count 1:1.
        # --coarse-target overrides 'auto' with an explicit count.
        ct = args.coarse_target
        ct = ct if ct == 'auto' else int(ct)
        res = relax_progressive(
            pos, mass, density, box, nx_full=nx_full, coarse_seed='merge',
            coarse_target=ct, alpha=args.alpha, inter_iter=args.inter_iter,
            nsmooth=args.nsmooth, max_iter=args.max_iter,
            momentum=args.momentum, icr0_stop=args.icr0_stop,
            polish_iter=args.polish_iter,
            polish_down=args.polish_down, polish_up=args.polish_up,
            cap=2.0, start_icr0=2.0,
            kernel=args.kernel, sigma=args.sigma,
            sigma_f=args.sigma_f,
            gradient=args.gradient, rhopow=args.rhopow, icr0rate=args.icr0rate,
            margin=args.margin, one_sided=args.one_sided, hmode=args.hmode,
            nsmoothmin=args.nsmoothmin, self_bias=args.self_bias,
            envelope=env, open_pad=args.open_pad, cascade=args.progressive,
            force_table=force_table, verbose=True)
        if env is not None and res.envelope_audit is not None:
            a = res.envelope_audit           # background culled, survivors renormed
            nf, mf = a['n_survivor'], a['mf']
            print(f'envelope cull: {a["n_full"]} -> {nf} '
                  f'(culled {a["n_culled"]}), gas mass {a["gas_mass"]:.3e}, '
                  f'per-particle mass {mf:.3e}', flush=True)
        else:
            nf = len(res.pos)
            mf = mass.sum() / nf  # equal-mass glass
        g = np.zeros((nf, 12), dtype=gas.dtype)
        g[:, 0] = mf
        g[:, 1:4] = res.pos
        g[:, 7] = res.rho
        g[:, 9] = res.h
        hdr = header.copy()
        hdr[0] = nf
        hdr[2] = nf
        tip.writetipsy(g, dark, star, f'{args.out}_IC', hdr, time)
        write_glass_aux(args.out, res.pos, mf, res.h, res.rho, density, box,
                        args.nsmooth, args.rhopow, args.sigma_f,
                        args.gradient, args.nsmoothmin, args.kernel)
        err = np.abs(1.0 - res.rho /
                     _target_readout(density, res.pos, res.h))
        print(f'done (progressive -> nx_full={nx_full}): final N={nf}, '
              f'mean|1-rho/rho0|={err.mean():.3e}, wrote {args.out}_IC')
        _print_stages(res)
        if args.plot:
            write_denserr_plot(args.out, res.pos, mf, res.h, res.rho, density,
                               box, args.nsmooth, sigma_f=args.sigma_f,
                               gradient=args.gradient,
                               kernel=args.kernel,
                               rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
        if args.history_plot:
            from .diagnostics import plot_history
            plot_history(res.history, f'{args.out}_history.png',
                         title=f'{os.path.basename(args.out)} (all levels)',
                         stop_reason=res.stop_reason,
                         levels=res.level_history)
            print(f'  wrote {args.out}_history.png', flush=True)
        return

    callback = None
    if args.oi > 0:
        def callback(st):
            write_snapshot(f'{args.out}.{st.it:05d}', st.pos, st.rho, st.h)

    res = relax(pos, mass, density, box=box, nsmooth=args.nsmooth,
                kernel=args.kernel, sigma=args.sigma,
            sigma_f=args.sigma_f,
            gradient=args.gradient, rhopow=args.rhopow,
                icr0=args.icr0, icrloop0=args.icrloop0,
                icr0rate=args.icr0rate, max_iter=args.max_iter,
                margin=args.margin, one_sided=args.one_sided,
                hmode=args.hmode, nsmoothmin=args.nsmoothmin,
                icr0_stop=args.icr0_stop, polish_iter=args.polish_iter,
                polish_down=args.polish_down, polish_up=args.polish_up,
                self_bias=args.self_bias, coarse_fft=args.coarse_fft,
                coarse_serial=args.coarse_fft_serial, coarse_every=args.coarse_fft_every,
                force_table=force_table,
                callback=callback, callback_every=args.oi or args.max_iter,
                verbose=True)

    write_snapshot(f'{args.out}_IC', res.pos, res.rho, res.h)
    write_glass_aux(args.out, res.pos, mass, res.h, res.rho, density, box,
                    args.nsmooth, args.rhopow, args.sigma_f,
                    args.gradient, args.nsmoothmin, args.kernel)
    err = np.abs(1.0 - res.rho /
                 _target_readout(density, res.pos, res.h))
    print(f'done: {res.niter} iterations, converged={res.converged}, '
          f'mean|1-rho/rho0|={err.mean():.3e}, wrote {args.out}_IC')
    if args.plot:
        write_denserr_plot(args.out, res.pos, mass, res.h, res.rho, density,
                           box, args.nsmooth, sigma_f=args.sigma_f,
                               gradient=args.gradient,
                               kernel=args.kernel,
                           rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
    if args.history_plot:
        from .diagnostics import plot_history
        plot_history(res.history, f'{args.out}_history.png',
                     title=os.path.basename(args.out),
                     stop_reason=res.stop_reason)
        print(f'  wrote {args.out}_history.png', flush=True)


if __name__ == '__main__':
    main()
