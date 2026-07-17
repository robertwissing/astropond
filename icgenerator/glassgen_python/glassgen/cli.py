"""CLI: functional replacement for the gasoline.gdicgen invocation in
runtest.sh (tipsy pre-IC + density table in -> relaxed tipsy glass out).

Example, mirroring runtest.sh ICCONFIG:
    PYTHONPATH=icgenerator/glassgen_python python -m glassgen.cli \
        -o datafiles/myglass -I datafiles/pre.00000 -s 64 -n 4800 -oi 4800 \
        --param datafiles/pre.param --table densitytable_xdr --direction r_sph

Without --table the target density is uniform (total gas mass / box
volume). Output: ${out}_IC (tipsy, gas positions/density/hsmooth updated,
velocities zeroed) and ${out}.NNNNN snapshots every -oi iterations.
"""
import argparse
import os
import sys

import numpy as np

_SETUPFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', '..', 'setupfiles')
sys.path.insert(0, _SETUPFILES)

from .density import MAPPINGS, CallableDensity, TableDensity
from .params import box_from_param, density_from_param, read_param
from .relax import relax


def write_denserr_plot(out, pos, mass, h, rho, density, box, nsmooth,
                       flavour='gdforce', rhopow=2.0, nsmoothmin=32):
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
    rho0 = density.rho0(pos)
    err = np.abs(1.0 - rho / rho0)
    force = residual_force(pos, mass, h, rho, rho0, box, rhopow=rhopow,
                           nsmooth=nsmooth, flavour=flavour,
                           nsmoothmin=nsmoothmin)

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
    ap = argparse.ArgumentParser(
        description='glassgen: glass-relaxation IC generator')
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
                    help='snapshot output interval (0 = none)')
    ap.add_argument('--table', default=None,
                    help='target density table: a densitytable_xdr file (1D/2D/'
                         '3D, indexed by --direction) OR a 3D AMR table .npz '
                         '(glasstable/glassresample --save-table) targeted '
                         'directly. Default: uniform density')
    ap.add_argument('--direction', default='r_sph', choices=MAPPINGS,
                    help='coordinate a 1D table is indexed by')
    ap.add_argument('--flavour', default='gdforce')
    ap.add_argument('--rhopow', type=float, default=2.0, help='ICrhopow')
    ap.add_argument('--icr0', type=float, default=1.0, help='ICR0')
    ap.add_argument('--icrloop0', type=float, default=0.5, help='ICRLOOP0')
    ap.add_argument('--icr0rate', type=float, default=1.5, help='ICR0Rate')
    ap.add_argument('--margin', type=int, default=8,
                    help='extra kNN candidates kept per particle; larger '
                         'values widen the Verlet skin (fewer rebuilds) at '
                         'slightly higher per-iteration cost')
    ap.add_argument('--one-sided', action='store_true',
                    help='SPH-EXA-style transpose-free force walk: faster, '
                         'pair forces no longer exactly antisymmetric')
    ap.add_argument('--h-mode', dest='hmode', default='knn',
                    choices=('knn', 'measured', 'target'),
                    help='smoothing length for the force kernel + move step: '
                         'knn (default), measured (=(m*hfact3/rho_sph)^1/3, '
                         'the C density-based h), or target '
                         '(=(m*hfact3/rho0)^1/3, noise-free analytic h)')
    ap.add_argument('--nsmoothmin', type=int, default=32,
                    help='floor on density-based h: never gather fewer than '
                         '~nsmoothmin real neighbours (measured/target only)')
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
                    help='resolvability tolerance: the auto coarse count '
                         'bound AND (with --scaffold cubic) the vacuum-'
                         'scaffold reach (default 4.0)')
    ap.add_argument('--nx-full', dest='nx_full', type=int, default=0,
                    help='full target resolution (default: round(N**(1/3)) '
                         'from the input)')
    ap.add_argument('--inter-iter', dest='inter_iter', type=int, default=10,
                    help='icgen iterations after each intermediate x2 split in '
                         'the progressive cascade (default 10)')
    ap.add_argument('--momentum', type=float, default=0.5,
                    help='heavy-ball coefficient on the post-split confined '
                         'stages (default 0.5; 0 = off)')
    ap.add_argument('--icr0-stop', dest='icr0_stop', type=float, default=3e-3,
                    help='median-smoothed icr0 stop (max step as a fraction of '
                         'h) - the quality dial: 3e-3 fast/good, ~1e-3 tight. '
                         '0 = off (then -n / --polish-iter ends the run)')
    ap.add_argument('--polish-iter', dest='polish_iter', type=int, default=0,
                    help='fixed polish-iteration budget on the fine stage '
                         '(0 = off; composes with --icr0-stop)')
    ap.add_argument('--no-self-bias', dest='self_bias', action='store_false',
                    help='disable the Wendland C2 self-density bias correction '
                         '(Wzero10); density then uses the bare W0 self term')
    ap.add_argument('--envelope', choices=('auto', 'on', 'off'), default='auto',
                    help='free-surface / open-box vacuum padding (see '
                         'RECONSTRUCT_PLAN.md): pad the vacuum side of the '
                         'gas with a mass-consistent background so edge kernels '
                         'fill, relax the union, then cull the background. '
                         '"auto" (default) = on for an --open box, off for a '
                         'filled periodic box (byte-identical to no envelope); '
                         '"on"/"off" force it.')
    ap.add_argument('--nedge', '--bg-c', dest='bg_c', type=int, default=0,
                    help='envelope: resolvability contour count - INTEGRAL: '
                         'rho_edge = where the cumulative faint-end mass '
                         'first exceeds nedge*m (the whole discarded tail in '
                         'particle masses, NOT a per-kernel count); 0 = use '
                         '--nsmooth (-s). --bg-c is a legacy alias.')
    ap.add_argument('--nframe', dest='n_frame', type=float, default=None,
                    help='envelope (M20): floor backability - expected '
                         'particles per edge kernel at the profile floor '
                         '(rho_frame = rho_edge*nframe/nsmooth, i.e. '
                         'k = (nsmooth/nframe)**(1/6); default 16 ~ k1.26). '
                         'Exclusive with the legacy --bg-k.')
    ap.add_argument('--nl0', dest='nl0', type=float, default=None,
                    help='envelope (M20): scaffold launch resolvability '
                         'N_L(0) - the near-gas e-fold length in kernel '
                         'supports; tail_reach is solved per build (default '
                         '1.35; avoid ~1, the metastable resonance band)')
    ap.add_argument('--bg-k', dest='bg_k', type=float, default=None,
                    help='LEGACY: backing coarseness k directly (rho_frame = '
                         'rho_edge/k**6, Delta_bg = k*(m/rho_edge)**(1/3)); '
                         'bypasses --nframe. The uniform/analytic path '
                         'defaults to k=2 (untouched by M20).')
    ap.add_argument('--tail-reach', dest='tail_reach', type=float,
                    default=None, metavar='S',
                    help='LEGACY: explicit vacuum-profile steepness TR '
                         '(bypasses the --nl0 solve); table targets only')
    ap.add_argument('--scaffold', default='exp', choices=('exp', 'cubic'),
                    help="envelope vacuum scaffold: 'exp' (default: strictly-"
                         "decreasing rho_edge->rho_frame profile + sampled "
                         "backing) or 'cubic' (falsified M14 arm, kept as "
                         "documented-dead; --bg-k does not apply)")
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

    if args.count is not None:
        if not args.table:
            ap.error('--count builds from a table: pass --table FILE.npz')
        if args.infile:
            ap.error('--count samples the initial particles from the table '
                     'itself; drop -I (or drop --count to relax the pre-IC)')
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
        if args.table.endswith('.npz'):
            # a 3D AMR density table (glasstable/glassresample --save-table):
            # target the captured field directly (reconstruct-in-place).
            from .amrtable import AMRTable
            density = AMRTable.load(args.table)
            print(f'loaded AMR density table {args.table} '
                  f'({density.ncell} leaves)', flush=True)
        else:
            density = TableDensity.from_xdr(args.table, mapping=args.direction)
    elif param_dict is not None:
        # read the density profile straight from the .param (profiles 1/2/3),
        # exactly as the C gdicgen does - no density table needed for these.
        density = density_from_param(param_dict)
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
        from .progressive import build_from_table
        ct = args.coarse_target
        ct = ct if ct == 'auto' else int(ct)
        print(f'table build: sampling {args.count} particles from the table '
              f'(envelope {args.envelope}, cascade '
              f'{"on" if args.progressive else "off"})', flush=True)
        res = build_from_table(
            density, args.count, box=box, nsmooth=args.nsmooth,
            bg_c=args.bg_c, bg_k=args.bg_k, envelope=args.envelope,
            n_frame=args.n_frame, nl0=args.nl0, tail_reach=args.tail_reach,
            scaffold=args.scaffold,
            cascade=args.progressive, open_pad=args.open_pad, verbose=True,
            coarse_target=ct, alpha=args.alpha, inter_iter=args.inter_iter,
            max_iter=args.max_iter, momentum=args.momentum,
            icr0_stop=args.icr0_stop, polish_iter=args.polish_iter,
            cap=2.0, start_icr0=2.0,
            flavour=args.flavour, rhopow=args.rhopow, icr0rate=args.icr0rate,
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
        err = np.abs(1.0 - res.rho / density.rho0(res.pos))
        print(f'done (table build, count={args.count}): final N={nf}, '
              f'per-particle mass {mf:.3e}, mean|1-rho/rho0|={err.mean():.3e},'
              f' wrote {args.out}_IC')
        _print_stages(res)
        if args.plot:
            write_denserr_plot(args.out, res.pos, mf, res.h, res.rho, density,
                               box, args.nsmooth, flavour=args.flavour,
                               rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
        if args.history_plot:
            from .diagnostics import plot_history
            plot_history(res.history, f'{args.out}_history.png',
                         title=f'{os.path.basename(args.out)} (final stage)',
                         stop_reason=res.stop_reason)
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
    if envelope_on:
        from .amrtable import AMRTable
        from .envelope import build_table_envelope, build_uniform_envelope
        bg_c = args.bg_c or args.nsmooth
        if isinstance(density, AMRTable):
            # table-native: rho_edge/shell/background straight off the leaf set
            env = build_table_envelope(density, float(mass.sum()), len(pos),
                                       box=box, c=bg_c, k=args.bg_k,
                                       n_frame=args.n_frame, nl0=args.nl0,
                                       tail_reach=args.tail_reach,
                                       scaffold=args.scaffold,
                                       alpha=args.alpha)
        else:
            # uniform/analytic path: untouched by M20 - keeps the legacy k=2
            env = build_uniform_envelope(density, pos, float(mass.sum()),
                                         len(pos), c=bg_c,
                                         k=(2.0 if args.bg_k is None
                                            else args.bg_k),
                                         scaffold=args.scaffold,
                                         alpha=args.alpha)
        print(f'envelope: m={env["m"]:.3e} rho_edge={env["rho_edge"]:.3e} '
              f'rho_frame={env["rho_frame"]:.3e} delta_bg={env["delta_bg"]:.3e} '
              f'background N={len(env["bg_pos"])}', flush=True)
        if len(env['bg_pos']) == 0 and args.envelope == 'auto':
            print('envelope: no sub-rho_edge cells -> filled box, disabling',
                  flush=True)
            env = None

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
            polish_iter=args.polish_iter, cap=2.0, start_icr0=2.0,
            flavour=args.flavour, rhopow=args.rhopow, icr0rate=args.icr0rate,
            margin=args.margin, one_sided=args.one_sided, hmode=args.hmode,
            nsmoothmin=args.nsmoothmin, self_bias=args.self_bias,
            envelope=env, open_pad=args.open_pad, cascade=args.progressive,
            verbose=True)
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
        err = np.abs(1.0 - res.rho / density.rho0(res.pos))
        print(f'done (progressive -> nx_full={nx_full}): final N={nf}, '
              f'mean|1-rho/rho0|={err.mean():.3e}, wrote {args.out}_IC')
        _print_stages(res)
        if args.plot:
            write_denserr_plot(args.out, res.pos, mf, res.h, res.rho, density,
                               box, args.nsmooth, flavour=args.flavour,
                               rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
        if args.history_plot:
            from .diagnostics import plot_history
            plot_history(res.history, f'{args.out}_history.png',
                         title=f'{os.path.basename(args.out)} (final stage)',
                         stop_reason=res.stop_reason)
            print(f'  wrote {args.out}_history.png', flush=True)
        return

    callback = None
    if args.oi > 0:
        def callback(st):
            write_snapshot(f'{args.out}.{st.it:05d}', st.pos, st.rho, st.h)

    res = relax(pos, mass, density, box=box, nsmooth=args.nsmooth,
                flavour=args.flavour, rhopow=args.rhopow,
                icr0=args.icr0, icrloop0=args.icrloop0,
                icr0rate=args.icr0rate, max_iter=args.max_iter,
                margin=args.margin, one_sided=args.one_sided,
                hmode=args.hmode, nsmoothmin=args.nsmoothmin,
                icr0_stop=args.icr0_stop, polish_iter=args.polish_iter,
                self_bias=args.self_bias,
                callback=callback, callback_every=args.oi or args.max_iter,
                verbose=True)

    write_snapshot(f'{args.out}_IC', res.pos, res.rho, res.h)
    err = np.abs(1.0 - res.rho / density.rho0(res.pos))
    print(f'done: {res.niter} iterations, converged={res.converged}, '
          f'mean|1-rho/rho0|={err.mean():.3e}, wrote {args.out}_IC')
    if args.plot:
        write_denserr_plot(args.out, res.pos, mass, res.h, res.rho, density,
                           box, args.nsmooth, flavour=args.flavour,
                           rhopow=args.rhopow, nsmoothmin=args.nsmoothmin)
    if args.history_plot:
        from .diagnostics import plot_history
        plot_history(res.history, f'{args.out}_history.png',
                     title=os.path.basename(args.out),
                     stop_reason=res.stop_reason)
        print(f'  wrote {args.out}_history.png', flush=True)


if __name__ == '__main__':
    main()
