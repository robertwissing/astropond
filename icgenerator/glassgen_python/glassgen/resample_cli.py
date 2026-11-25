"""Snapshot resampler CLI: --op merge | split | reconstruct (tipsy + aux).

split/merge act on one or more particle types (gas, dark, star) in a single
invocation - by default ALL types present in the input, so the relative mass
(and softening) resolution between components is preserved. Types not
selected are carried through unchanged. Reads the tipsy snapshot plus its
aux sibling files (BField, BFieldx/y/z, iord, ...), resamples each selected
type's particle fields and its slice of every aux file, and writes a new
snapshot with the changed counts.

    # upres a full galaxy IC by 2**3 = 8x (all types in one run):
    python -m glassgen.resample_cli -I ic.00000 -o ic_x8 \
        --op split --ntimes 3 --param ic.param

    # coarsen only the gas and dark by 2x:
    python -m glassgen.resample_cli -I ic.00000 -o ic_coarse \
        --particletype gas dark --op merge --param ic.param

--ntimes N doubles each selected type's count N times (factor 2**N) on split
and halves it N times on merge. Merged masses are renormalised to the input
mass distribution scaled by the count ratio, so a uniform-mass input stays
uniform-mass and total mass is conserved (see resample.coarsen). The
gravitational softening follows the mass resolution: dark/star `eps` is
scaled by (2**N)**(-1/3) on split, (2**N)**(+1/3) on merge. Gas has no
softening column - its `hsmooth` is set at runtime from the local density /
neighbour count, so it is carried through unscaled.

--op reconstruct (GAS only, IC-grade): rebuild the gas at an ARBITRARY
--count N, decoupled from the input count. A thin wrapper around the
glassgen table builder: capture
the input gas density on a raw AMR table (sph_interp), sample + envelope +
progressive relax in glassgen (build_from_table), then transfer fields
(vel/u/...) either by particle scatter or by a
conservative raw-AMR state plus vertex imprint - the output has no history
to inherit. Positions are rebuilt from scratch, so this is for
ICs / quasi-equilibrium snapshots, not mid-run states with strong coherent
flows. Only the gas is processed; dark/star (and their aux slices) pass
through unchanged.

    # rebuild the gas of a galaxy IC at 2M particles:
    python -m glassgen.resample_cli -I ic.00000 -o ic_2M \
        --op reconstruct --count 2000000 --param ic.param
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

_SETUPFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', '..', 'setupfiles')
sys.path.insert(0, _SETUPFILES)

from . import resample
from .params import box_from_param, read_param

# tipsy column layout per particle type: mass=0, pos=1:4, vel=4:7 handled
# explicitly; these are the remaining per-particle scalar columns carried as
# inherited aux through the resample.
_TYPE_NCOL = {'gas': 12, 'dark': 9, 'star': 11}
# col 9 of gas is the softening eps (NOT hsmooth; the SPH smoothing length is
# never stored - it is recomputed at runtime from density / nSmooth).
_TYPE_SCALARS = {
    'gas':  {'rho': 7, 'u': 8, 'eps': 9, 'metals': 10, 'phi': 11},
    'dark': {'eps': 7, 'phi': 8},
    'star': {'metals': 7, 'tform': 8, 'eps': 9, 'phi': 10},
}
# extensive columns: halved on split / summed on merge. Gas col 8 ("u") is
# TEMPERATURE (intensive): inherited on split, mass-weighted mean on merge.
_TYPE_HALVE = {'gas': set(), 'dark': set(), 'star': set()}
# softening columns that follow the mass resolution (scaled by factor**-1/3)
_TYPE_EPS = {'gas': ['eps'], 'dark': ['eps'], 'star': ['eps']}


def _default_output_prefix(infile, op, ntimes):
    """Default output name, including the actual split/merge count factor."""
    stem = re.sub(r'\.\d+$', '', infile)
    suffix = 'recon' if op == 'reconstruct' else f'{op}_{2 ** ntimes}x'
    return stem + '_' + suffix


def _read_aux(infile):
    """Read all aux sibling files of `infile`. Each aux file spans every
    particle ([gas|dark|star] concatenated under a single count header).
    Returns a list of dicts: label, kind in {'scalar','vector','int'},
    data (Ntot,) or (Ntot,3)."""
    import readtipsy as tip
    out = []
    for path in sorted(glob.glob(infile + '.*')):
        label = path[len(infile) + 1:]
        if any(b in label for b in tip.banauxfiles):
            continue
        if label.endswith('.param') or label == 'param':
            continue
        if label in tip.vecauxfiles:
            data, _ = tip.readtipsyVecaux(infile, label)
            data = np.asarray(data, dtype=np.float64).reshape(-1, 3)
            kind = 'vector'
        else:
            data, _ = tip.readtipsyaux(infile, label)
            data = np.asarray(data)
            kind = 'int' if label in tip.intauxfiles else 'scalar'
        out.append(dict(label=label, kind=kind, data=data))
    return out


def _write_aux(meta, outfile):
    import readtipsy as tip
    for m in meta:
        data = m['data']
        if m['kind'] == 'vector':
            tip.writetipsyVecaux(np.asarray(data, dtype=np.float64), m['label'],
                                 outfile)
        else:
            tip.writetipsyaux(np.asarray(data), m['label'], outfile)


# .param mass keys scaled with a gas resample (each by 2**ntimes); the ratio
# between them is whatever the input sets (not assumed to be 10x).
_PARAM_MASS_KEYS = ('dInitGasMass', 'dInitStarMass')
_MASS_RE = re.compile(
    r'^(\s*(?:%s)\s*=\s*)([-+0-9.eEdD]+)(.*)$' % '|'.join(_PARAM_MASS_KEYS))


def _write_scaled_param(param_path, out_prefix, scale_masses, mass_scale):
    """Copy `param_path` to `${out_prefix}.param`. When scale_masses, multiply
    the (active) dInitGasMass / dInitStarMass values by mass_scale, preserving
    each line's key, spacing and trailing comment."""
    with open(param_path) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if scale_masses and not line.lstrip().startswith('#'):
            m = _MASS_RE.match(line.rstrip('\n'))
            if m:
                val = float(m.group(2).replace('d', 'e').replace('D', 'e'))
                line = f'{m.group(1)}{val * mass_scale:.10g}{m.group(3)}\n'
        out.append(line)
    outparam = f'{out_prefix}.param'
    with open(outparam, 'w') as f:
        f.writelines(out)
    return outparam


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='split/merge particle types of a tipsy snapshot '
                    '(particle fields + all aux files); types not selected '
                    'pass through unchanged')
    ap.add_argument('--help-extra', action='help',
                    help='show the full help, including any hidden expert '
                         'options')
    ap.add_argument('-I', dest='infile', required=True, help='input tipsy')
    ap.add_argument('-o', dest='out', default=None,
                    help='output prefix (writes ${out}.00000 + aux). '
                         'Default: <input minus .NNNNN>_<op>_<factor>x for '
                         'split/merge, or <input>_recon for reconstruct')
    ap.add_argument('--particletype', nargs='+', default=None,
                    choices=('gas', 'dark', 'star'),
                    help='particle type(s) to resample (one or more). '
                         'Default: all types present in the input')
    ap.add_argument('--op', choices=('split', 'merge', 'reconstruct'),
                    default=None,
                    help='split (2x count per pass), merge (1/2x count per '
                         'pass), or reconstruct (rebuild the GAS at an '
                         'arbitrary --count against its captured density '
                         'table). No default: asked interactively when omitted')
    ap.add_argument('--ntimes', '--nsplits', dest='ntimes', type=int,
                    default=1,
                    help='split/merge: number of passes - each pass doubles '
                         '(split) or halves (merge) the selected count, i.e. '
                         'factor 2**ntimes. Default: 1')
    ap.add_argument('--count', type=int, default=None, metavar='N',
                    help='reconstruct: the target gas particle count '
                         '(required with --op reconstruct; decoupled from the '
                         'input count)')
    ap.add_argument('-s', dest='nsmooth', type=int, default=64,
                    help='neighbour count used to find each particle\'s '
                         'split direction / merge partner. Default: 64')
    ap.add_argument('--dist', type=float, default=0.34,
                    help='split only: the two children are placed at '
                         '+/- dist * (local spacing, see --scale) from the '
                         'parent, perpendicular to the neighbour direction. '
                         'Default: 0.34 - with the jr orient this maximizes '
                         'the min child pair spacing (0.84 dx, 96%% of a '
                         'relaxed glass; the packing ceiling FALLS on both '
                         'sides). Density noise scales as dist^2 (~0.3 is '
                         '~20%% quieter, ~15%% less spacing); values below '
                         '~0.3 degenerate under repeated splitting')
    ap.add_argument('--orient', default=None,
                    help='split only: child pair orientation algorithm '
                         '(glassgen.resample.split orient values). Default: '
                         'jr96 - iterative parallel azimuth relaxation '
                         '(active-set randomized Jacobi, early-stopping): '
                         'vs the old random azimuth it cuts the mean split '
                         'density error ~20%%, the worst-case error ~40%%, '
                         'and lifts the min pair spacing 0.55 -> 0.75 dx '
                         '(glass benchmark). Use "random" for the pre-2026 '
                         'behaviour')
    ap.add_argument('--scale', default='vol', choices=('nn', 'vol', 'h'),
                    help='split only: the local spacing the offset is a '
                         'fraction of. vol = (m/rho)^(1/3) (default; '
                         'kernel-weighted, ~20x less scatter than nn and '
                         'the best-behaved in density gradients), nn = '
                         'nearest-neighbour distance (pre-2026 behaviour), '
                         'h = from the smoothing length')
    rec = ap.add_argument_group(
        'reconstruct op (gas only)',
        'rebuild the gas at --count N: capture the input gas density on a '
        'raw AMR table, then run the glassgen table builder (sample + '
        'N-aware envelope + progressive relax + cull/renorm to M_gas) and '
        'transfer fields from the input onto the final positions.')
    rec.add_argument('--grid-npx', type=int, default=None,
                     help='AMR BASE grid cells per axis (default ~N^(1/3)/8, '
                          'power of two - refinement supplies locality). '
                          'Must stay coarser than the particle sampling or '
                          'the table encodes the input\'s particle noise as '
                          'target structure')
    rec.add_argument('--capture-method', default=None,
                     choices=('petkova', 'sph-gather', 'sph-scatter'),
                     help='grid deposit for the capture. Default: petkova '
                          '(exact mass-conserving cell averages - right for '
                          'periodic boxes). For OPEN boxes try sph-gather: '
                          'the table then carries the same kernel-truncation '
                          'deficit at the gas edge as the relax\'s own '
                          'density estimate, so the edge biases cancel '
                          'instead of driving a persistent edge force')
    rec.add_argument('--field-transfer', choices=('scatter', 'table'),
                     default='table',
                     help='field transfer after reconstruction: table '
                          '(default) captures conservative rho*A channels '
                          'on the raw AMR grid and imprints A=vertex(rho*A)/'
                          'vertex(rho) after relaxation; scatter is the '
                          'particle-to-particle control/fallback')
    rec.add_argument('--field-interp',
                     choices=('vertex', 'limited', 'flat'), default='vertex',
                     help='table field readout at final particle '
                          'positions (default: vertex). Numerator and rho '
                          'use the same operator before division')
    rec.add_argument('--magnetic-projection',
                     choices=('none', 'hodge', 'vector-potential'),
                     default='none',
                     help='transfer of any BField aux (never rides the '
                          'generic scatter/table path); every mode works on '
                          'the density AMR capture grid. none (default): '
                          'conservative volume-carrier cell B, vertex '
                          'read-back at the final positions - no projection; '
                          'clean divergence afterwards in the consumer\'s '
                          'operator (mhd_divBclean). hodge: open AMR face '
                          'capture + Hodge/solenoidal projection, RT0 '
                          'readout. vector-potential: meshless Biot-Savart '
                          'particle A -> conservative cell A -> B = curl of '
                          'the vertex A interpolant (open boxes only)')
    rec.add_argument('--energy-mode', choices=('internal', 'total'),
                     default='internal',
                     help='table transfer: internal preserves deposited '
                          'u (default); total also deposits kinetic energy '
                          'and converts unresolved within-cell velocity '
                          'variance to heat. Requires --field-transfer table')
    rec.add_argument('--momentum-closure', choices=('boost', 'none'),
                     default=None,
                     help='table transfer: apply the minimum-change '
                          'uniform velocity boost that matches full raw-table '
                          'linear momentum (default for table), or none for '
                          'an uncorrected A/B')
    state_io = rec.add_mutually_exclusive_group()
    state_io.add_argument('--state-table', default=None, metavar='FILE.npz',
                          help='reuse a previously captured raw AMR '
                               'state table (still reads the input snapshot '
                               'for gas mass and non-gas families)')
    state_io.add_argument('--save-state', default=None, metavar='FILE.npz',
                          help='save the captured raw AMR state before '
                               'starting relaxation')
    rec.add_argument('--icr0-stop', dest='icr0_stop', type=float,
                     default=None, metavar='X',
                     help='median-smoothed icr0 stop (max step as a fraction '
                          'of h) for the relax - the quality dial: 3e-3 '
                          '(default) fast/good, ~1e-3 tight; capped at 4800 '
                          'iterations either way')
    rec.add_argument('--momentum', type=float, default=None, metavar='B',
                     help='heavy-ball coefficient on the relax move, applied '
                          'ONLY in the polish phase (hardwired 2026-07-22; '
                          'default 0 = off). NB with the default icr0-stop '
                          'the envelope build barely polishes before the '
                          'stop fires')
    rec.add_argument('--nedge', dest='bg_c', type=int, default=0,
                     help='envelope: resolvability contour count (INTEGRAL: '
                          'rho_edge = where the cumulative faint-end mass '
                          'first exceeds nedge*m; 0 = use --nsmooth/-s)')
    rec.add_argument('--nframe', dest='n_frame', type=float, default=None,
                     help='floor backability - expected particles per '
                          'edge kernel at the profile floor (rho_frame = '
                          'rho_edge*nframe/nsmooth; default 16 ~ k1.26)')
    rec.add_argument('--nl0', dest='nl0', type=float, default=None,
                     help='scaffold launch resolvability N_L(0) in '
                          'kernel supports; tail_reach solved per build '
                          '(default 1.35; avoid ~1, the resonance band)')
    rec.add_argument('--alpha', type=float, default=None,
                     help='cascade resolvability tolerance '
                          '(auto_coarse_count; default 2.0)')
    rec.add_argument('--no-cascade', dest='cascade', action='store_false',
                     help='disable the multi-resolution cascade and relax '
                          'the placed gas+background ensemble in a single '
                          'full-resolution pass (same envelope + cull)')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--param', help='gasoline .param file to take the box from')
    grp.add_argument('--box', nargs=3, type=float, metavar=('LX', 'LY', 'LZ'))
    grp.add_argument('--open', action='store_true', help='no periodic box')
    args = ap.parse_args(argv)

    if args.ntimes < 1:
        sys.exit('--ntimes must be >= 1')
    factor = 2 ** args.ntimes

    if args.op is None:
        if sys.stdin.isatty():
            while args.op not in ('split', 'merge', 'reconstruct'):
                args.op = input('no --op given - split, merge or '
                                'reconstruct? ').strip().lower()
        else:
            ap.error('--op {split,merge,reconstruct} is required when not '
                     'running interactively')
    if args.op == 'reconstruct':
        if args.count is None:
            ap.error('--op reconstruct needs --count N (the target gas '
                     'particle count)')
        if args.count < 1:
            ap.error('--count must be >= 1')
        if args.ntimes != 1:
            ap.error('--ntimes does not apply to reconstruct '
                     '(--count is the target, not a factor)')
    else:
        for val, name in ((args.count, '--count'),
                          (args.grid_npx, '--grid-npx'),
                          (args.capture_method, '--capture-method'),
                          (args.icr0_stop, '--icr0-stop'),
                          (args.momentum, '--momentum')):
            if val is not None:
                ap.error(f'{name} requires --op reconstruct')
        if args.magnetic_projection != 'none':
            ap.error('--magnetic-projection requires --op reconstruct')
    if args.capture_method is None:
        args.capture_method = 'petkova'
    if (args.op == 'reconstruct' and args.field_transfer == 'table'
            and args.capture_method != 'petkova'):
        ap.error('--field-transfer table requires --capture-method petkova')
    if (args.op == 'reconstruct' and args.energy_mode == 'total'
            and args.field_transfer != 'table'):
        ap.error('--energy-mode total requires --field-transfer table')
    if (args.op == 'reconstruct' and args.momentum_closure == 'boost'
            and args.field_transfer != 'table'):
        ap.error('--momentum-closure boost requires --field-transfer table')
    if (args.op == 'reconstruct'
            and (args.state_table or args.save_state)
            and args.field_transfer != 'table'):
        ap.error('--state-table/--save-state require --field-transfer table')
    if args.icr0_stop is None:
        args.icr0_stop = 3e-3
    if args.momentum is None:
        args.momentum = 0.0
    if args.orient is None:
        args.orient = 'jr96'

    if args.out is None:
        args.out = _default_output_prefix(args.infile, args.op, args.ntimes)
        print(f'no -o given: writing to {args.out}.00000', flush=True)

    import readtipsy as tip

    t_all = time.perf_counter()
    gas, dark, star, header, snaptime = tip.readtipsy(args.infile)
    fam = {'gas': gas, 'dark': dark, 'star': star}
    counts = {k: len(v) for k, v in fam.items()}
    ng, nd, ns = counts['gas'], counts['dark'], counts['star']
    ntot = ng + nd + ns
    order = ('gas', 'dark', 'star')
    bounds = {'gas': (0, ng), 'dark': (ng, ng + nd), 'star': (ng + nd, ntot)}

    if args.op == 'reconstruct':
        # gas only: dark/star have no SPH density for the run to trip over -
        # they (and their aux slices) pass through unchanged
        if args.particletype not in (None, ['gas']):
            sys.exit('--op reconstruct rebuilds GAS only; drop '
                     '--particletype (or pass --particletype gas)')
        if counts['gas'] == 0:
            sys.exit(f'no gas particles in {args.infile}')
        ptypes = ['gas']
    elif args.particletype:
        ptypes = []
        for t in args.particletype:            # dedupe, keep given order
            if t not in ptypes:
                ptypes.append(t)
        for t in ptypes:
            if counts[t] == 0:
                sys.exit(f'no {t} particles in {args.infile}')
    else:
        ptypes = [t for t in order if counts[t] > 0]
        print('no --particletype given: resampling all types present '
              f'({", ".join(ptypes)})', flush=True)
    print(f'read {args.infile}: {ng} gas, {nd} dark, {ns} star '
          f'({ntot} total)', flush=True)

    if args.open:
        box = None
        boxdesc = 'open (no periodic wrap)'
    elif args.box:
        box = np.array(args.box)
        boxdesc = 'periodic %g x %g x %g (--box)' % tuple(box)
    else:
        box = box_from_param(read_param(args.param))
        boxdesc = ('open (no periodic box in %s)' % args.param if box is None
                   else 'periodic %g x %g x %g (from %s)' % (*box, args.param))

    # say up front what will run and with which settings - the phases below
    # (capture / pairing / relax) can each take minutes on ~1e6 particles
    if args.op == 'reconstruct':
        budget = f'icr0 < {args.icr0_stop:g}, max 4800 iters'
        print(f'plan: reconstruct gas -> {args.count} particles (raw AMR '
              f'capture, {args.capture_method} deposit'
              + (f', {args.grid_npx} cells/axis' if args.grid_npx else '')
              + ' -> glassgen table build: envelope '
              f'{"cascade" if args.cascade else "single relax"}, {budget}, '
              f'{args.field_transfer} field transfer at the final positions'
              + (f' ({args.field_interp})' if args.field_transfer == 'table'
                 else '')
              + (f', energy {args.energy_mode}'
                 if args.field_transfer == 'table' else '')
              + (f', momentum {args.momentum_closure or "boost"}'
                 if args.field_transfer == 'table' else '')
              + f', magnetic {args.magnetic_projection}' + '); '
              f'box {boxdesc}; nsmooth {args.nsmooth}', flush=True)
    else:
        arrow = ('x%d' if args.op == 'split' else '1/%d') % factor
        print(f'plan: {args.op} {", ".join(ptypes)} '
              f'({args.ntimes} pass{"es" if args.ntimes > 1 else ""}, '
              f'count {arrow}); box {boxdesc}; nsmooth {args.nsmooth}',
              flush=True)
    if args.op == 'split':
        print(f'  placement: orient {args.orient}, dist {args.dist:g}, '
              f'scale {args.scale}', flush=True)

    # aux sibling files span every particle: sliced per type below
    auxmeta = _read_aux(args.infile)
    for m in auxmeta:
        if len(m['data']) != ntot:
            sys.exit(f"aux file .{m['label']} has {len(m['data'])} rows, "
                     f"expected {ntot} (gas+dark+star)")

    newaux = {}     # ptype -> resampled aux dict (incl. aux_<label> blocks)
    for ptype in ptypes:
        sel = fam[ptype]
        # selected type's particle fields -> resample aux dict
        pos = sel[:, 1:4].astype(np.float64)
        mass = sel[:, 0].astype(np.float64)
        aux = {'vel': sel[:, 4:7].astype(np.float64)}
        for key, col in _TYPE_SCALARS[ptype].items():
            aux[key] = sel[:, col].astype(np.float64)
        halve = set(_TYPE_HALVE[ptype])

        lo, hi = bounds[ptype]
        for m in auxmeta:
            key = 'aux_' + m['label']
            aux[key] = m['data'][lo:hi].astype(np.float64)
            # B-field aux are extensive (flux split), matching the SPH split
            if m['label'].startswith('BField') or m['label'].startswith('CurlB'):
                halve.add(key)

        t0 = time.perf_counter()
        n_in, dt = len(sel), sel.dtype
        if args.op == 'reconstruct':    # gas rebuild via the glassgen builder
            from . import reconstruct
            state_cap = None
            if args.state_table:
                from .amrtable import AMRStateTable
                state_cap = AMRStateTable.load(args.state_table)
            # the input rows are fully unpacked into pos/mass/aux float64 -
            # release the float32 originals so the heavy build does not carry
            # them (gas-only processing; dark/star stay untouched in fam)
            fam[ptype] = None
            del sel
            # a from-scratch draw is a COLD start - the builder runs the
            # regular glassgen glass build (anneal ladder / cascade), not the
            # split/merge confined heal. max_iter 4800; icr0_stop usually
            # finishes it earlier.
            rkw = dict(icr0_stop=args.icr0_stop, max_iter=4800,
                       callback_every=10, momentum=args.momentum)
            field_payload = aux
            tu_fac = 1.0
            if args.field_transfer == 'table':
                # BField aux stays in: reconstruct splits magnetic components
                # out of the generic transfer and moves them by
                # --magnetic-projection. Recomputed/resolution fields are not hydrodynamic state
                # channels. eps is restored below then scaled to output mass;
                # phi is intentionally reset for the target gravity solve.
                field_payload = {k: v for k, v in aux.items()
                                 if k not in ('eps', 'phi')}
                if args.energy_mode == 'total':
                    # Tipsy col 8 is temperature, whereas kinetic energy is
                    # in code-specific-energy units. Convert T->u for closure
                    # and convert back before writing the snapshot.
                    from sph_interp.particles import read_param_tufac
                    tu_fac = read_param_tufac(args.infile)
                    if tu_fac is None:
                        from IC_analysis_general import read_tufac
                        tu_fac = read_tufac(args.infile, np.nan)
                    if not np.isfinite(tu_fac) or tu_fac <= 0.0:
                        sys.exit('--energy-mode total needs a valid dTuFac '
                                 '(u=dTuFac*T) in the sibling .param/.log; '
                                 'this snapshot has no compatible energy-unit '
                                 'conversion')
                    field_payload = dict(field_payload)
                    field_payload['u'] = aux['u'] * tu_fac
            print(f'reconstruct: {n_in} gas -> {args.count} target '
                  f'(capture -> glassgen build_from_table, envelope '
                  f'{"cascade" if args.cascade else "single relax"})...',
                  flush=True)
            ct = time.perf_counter()
            pos2, mass2, aux2, _isf, ea = reconstruct.reconstruct_resample(
                pos, mass, args.count, field_payload, box=box,
                nsmooth=args.nsmooth,
                method=args.capture_method, base_npx=args.grid_npx,
                field_transfer=args.field_transfer,
                field_interp=args.field_interp,
                energy_mode=args.energy_mode,
                momentum_closure=args.momentum_closure,
                magnetic_projection=args.magnetic_projection,
                amr_cap=state_cap, save_state=args.save_state,
                bg_c=args.bg_c,
                n_frame=args.n_frame, nl0=args.nl0, cascade=args.cascade,
                alpha=args.alpha,
                relax_kw=rkw,
                progress=lambda s: print(
                    f'  iter {s.it}: denserr {s.dens_err:.4f} icr0 '
                    f'{s.icr0:.1e} ({time.perf_counter() - ct:.0f}s)',
                    flush=True))
            if args.field_transfer == 'table':
                if args.energy_mode == 'total':
                    aux2['u'] = aux2['u'] / tu_fac
                aux2['eps'] = np.full(
                    len(pos2),
                    np.average(aux['eps'], weights=mass))
                aux2['phi'] = np.zeros(len(pos2), dtype=np.float64)
            n2 = len(pos2)
            print(f'  -> {n2} gas (target {args.count}): bg {ea["n_background"]} '
                  f'seeded, culled {ea["n_culled"]}, mf {ea["mf"]:.3e}, '
                  f'rho_edge {ea["rho_edge"]:.2e}, rho_frame {ea["rho_frame"]:.2e} '
                  f'({time.perf_counter() - t0:.0f}s)', flush=True)
            density_capture = ea.get('density_capture')
            if density_capture is not None:
                print(
                    '  raw AMR mass: pre-closure residual '
                    f'{density_capture["relative_error_before_closure"]:+.3e}; '
                    'uniform closure x'
                    f'{density_capture["uniform_closure_factor"]:.9g}; '
                    'final residual '
                    f'{density_capture["relative_error_after_closure"]:+.3e}',
                    flush=True)
        elif args.op == 'split':
            print(f'splitting {n_in} {ptype} particles '
                  f'in {args.ntimes} pass'
                  f'{"es" if args.ntimes > 1 else ""} '
                  f'(orient {args.orient})...', flush=True)
            split_pass_t = [time.perf_counter()]

            def split_progress(p, na, nb):
                now = time.perf_counter()
                print(f'  split {p}/{args.ntimes}: {na} -> {nb} particles '
                      f'({now - split_pass_t[0]:.0f}s)', flush=True)
                split_pass_t[0] = now

            pos2, mass2, aux2 = resample.split(
                pos, mass, aux, box=box, factor=factor,
                nsmooth=args.nsmooth, dist=args.dist, halve_fields=halve,
                orient=args.orient, scale=args.scale,
                progress=split_progress)
        else:
            # coarsen (not merge): iterate to EXACTLY count//factor
            print(f'merging {n_in} {ptype} particles (mutual-nearest '
                  'pairing passes, kNN rebuild each pass)...', flush=True)
            pos2, mass2, aux2 = resample.coarsen(
                pos, mass, aux, box=box, factor=factor, halve_fields=halve,
                progress=lambda p, na, nb, tgt: print(
                    f'  pass {p}: {na} -> {nb} (target {tgt})', flush=True))
        n2 = len(pos2)
        if args.op != 'reconstruct':
            print(f'{args.op} {ptype}: {n_in} -> {n2} particles '
                  f'({time.perf_counter() - t0:.0f}s)', flush=True)

        # softening follows the mass resolution: eps *= (m_new/m_old)^(1/3).
        # split/merge -> factor**(-/+1/3); reconstruct -> realised count ratio
        if args.op == 'reconstruct':
            eps_scale = (n_in / max(n2, 1)) ** (1.0 / 3.0)
        else:
            eps_scale = factor ** (-1.0 / 3.0) if args.op == 'split' \
                else factor ** (1.0 / 3.0)
        for key in _TYPE_EPS[ptype]:
            aux2[key] = aux2[key] * eps_scale

        # rebuild the selected type's tipsy array
        arr = np.zeros((n2, _TYPE_NCOL[ptype]), dtype=dt)
        arr[:, 0] = mass2
        arr[:, 1:4] = pos2
        arr[:, 4:7] = aux2['vel']
        for key, col in _TYPE_SCALARS[ptype].items():
            arr[:, col] = aux2[key]
        fam[ptype] = arr
        newaux[ptype] = aux2

    new_counts = {k: len(v) for k, v in fam.items()}
    ng2, nd2, ns2 = new_counts['gas'], new_counts['dark'], new_counts['star']
    ntot2 = ng2 + nd2 + ns2
    hdr = header.copy()
    hdr[0] = ntot2     # nbodies
    hdr[2] = ng2       # ngas
    hdr[3] = nd2       # ndark
    hdr[4] = ns2       # nstar
    outsnap = f'{args.out}.00000'
    print(f'writing {outsnap} ({ntot2} particles'
          + (f' + {len(auxmeta)} aux files' if auxmeta else '') + ')...',
          flush=True)
    tip.writetipsy(fam['gas'], fam['dark'], fam['star'], outsnap, hdr,
                   snaptime)

    # reassemble each aux file [gas|dark|star] with the resampled slices
    for m in auxmeta:
        if m['kind'] == 'int':
            # ids must stay unique across a count change: regenerate globally
            m['data'] = np.arange(ntot2, dtype=np.int32)
            continue
        parts = []
        for f in order:
            if f in newaux:
                parts.append(newaux[f]['aux_' + m['label']])
            else:
                flo, fhi = bounds[f]
                parts.append(m['data'][flo:fhi])
        m['data'] = np.concatenate(parts, axis=0)
    _write_aux(auxmeta, outsnap)
    print(f'wrote {outsnap} ({ntot2} total)'
          + (f' + {len(auxmeta)} aux files' if auxmeta else '')
          + f'  [total {time.perf_counter() - t_all:.0f}s]')

    # companion .param: copy it forward; scale the mass keys with a gas
    # resample (each by 2**ntimes), so dInit{Gas,Star}Mass track the new
    # gas particle mass. dark/star-only resamples copy the param verbatim.
    if args.param:
        if args.op == 'reconstruct':
            # gas mass went M_gas/N_in -> M_gas/N_out (realised count ratio)
            scale = ng / max(ng2, 1)
        elif args.op == 'split':
            scale = 1.0 / factor
        else:
            scale = float(factor)
        scale_masses = 'gas' in ptypes
        outparam = _write_scaled_param(
            args.param, args.out, scale_masses=scale_masses,
            mass_scale=scale)
        print(f'wrote {outparam}'
              + (f' (dInitGasMass/dInitStarMass x{scale:g})'
                 if scale_masses else ' (copy)'))


if __name__ == '__main__':
    main()
