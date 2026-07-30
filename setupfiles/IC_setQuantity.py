#!/usr/bin/env python3
"""Edit quantities in a tipsy IC: chainable operations over particles and aux.

The generalisation of setQuantity.py's 19-entry numbered menu. Three things
changed beyond the cleanup:

  * OPERATIONS CHAIN. Several ops apply in one pass, in the order given, so a
    real IC-prep recipe is one reproducible command instead of N runs:

        IC_setQuantity.py in.00000 -o out.00000 \\
            center --on potential-min \\
            set --field temp --value 1e4 \\
            remove --type gas --where 'r>50'

  * ONE SELECTION VOCABULARY. Every op takes `--where EXPR` (repeatable,
    ANDed) over the same quantities - r, rcyl, x, y, z, absz, v, rho, temp,
    mass, metals, eps, phi, tform, index. The old script had a different
    numbered sub-menu per operation ("1: r 2: y 3: z 4: rcyl 5: nr"), and its
    "y" option in fact filtered on |x|.

  * ONE SCALAR SETTER. `set --field` replaces the six near-identical options
    (velocity value, softening, tform, temperature, mass, metals), with
    --mode set|add|multiply and a selection, instead of one menu entry each
    that always applied to every particle.

Aux files ride along through every operation: removing particles drops their
aux rows, appending zero-fills them, and any rotation turns the packed vector
aux AND its scalar component twins (BFieldx/y/z, wx/wy/wz) together.

Split/merge are NOT here: `glassresample` does them for all particle types
with proper aux and .param handling (glassresample IC.00000 --op split).

Run bare for an interactive menu. `--list-ops` prints the operation table.
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'icgenerator', 'glassgen_python'))

from tipsysnap import (FIELD_COLUMN, FIELDS, NCOL,  # noqa: E402
                       QUANTITIES, Snapshot)

ALL = ('gas', 'dark', 'star')


def _types(arg):
    return ALL if 'all' in arg else tuple(arg)


def _masks(snap, types, where):
    """{type: boolean mask} for the given selection, reported per type."""
    out = {}
    for t in types:
        if len(snap.data[t]):
            out[t] = snap.select(t, where)
        else:
            out[t] = np.zeros(0, dtype=bool)
    if where:
        got = ', '.join(f'{t} {int(out[t].sum())}/{len(out[t])}'
                        for t in types if len(out[t]))
        print(f"    selection {' and '.join(where)} -> {got or 'nothing'}")
    return out


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def op_set(snap, a):
    """Set / add / multiply a scalar particle field."""
    types = [t for t in _types(a.type) if t in FIELD_COLUMN[a.field]]
    skipped = [t for t in _types(a.type) if t not in FIELD_COLUMN[a.field]]
    if skipped:
        print(f'    note: {", ".join(skipped)} have no {a.field!r} column, '
              'skipped')
    if not types:
        raise SystemExit(f'no selected type has a {a.field!r} column '
                         f'(it exists for: {sorted(FIELD_COLUMN[a.field])})')
    masks = _masks(snap, types, a.where)
    for t in types:
        col = FIELD_COLUMN[a.field][t]
        m = masks[t]
        if not m.any():
            continue
        if a.mode == 'set':
            snap.data[t][m, col] = a.value
        elif a.mode == 'add':
            snap.data[t][m, col] += a.value
        else:
            snap.data[t][m, col] *= a.value
        print(f'    {t}: {a.field} {a.mode} {a.value:g} on {int(m.sum())} '
              'particles')


def op_velocity(snap, a):
    """Set a velocity field: constant, radial, rotation, shear or zero."""
    types = _types(a.type)
    masks = _masks(snap, types, a.where)
    for t in types:
        arr = snap.data[t]
        m = masks[t]
        if not len(arr) or not m.any():
            continue
        pos = arr[m, 1:4]
        if a.mode == 'zero':
            v = np.zeros((int(m.sum()), 3))
        elif a.mode == 'constant':
            v = np.tile(_axis_vector(a.axis) * a.value, (int(m.sum()), 1))
        elif a.mode == 'radial':
            r = np.linalg.norm(pos, axis=1)
            unit = np.zeros_like(pos)
            ok = r > 0
            unit[ok] = pos[ok] / r[ok, None]
            v = a.value * (r[:, None] ** a.power) * unit
        elif a.mode == 'rotation':
            # omega x r about --axis; --power tilts it away from solid body
            # (power 0 = solid body v ~ R, power -1 = flat rotation curve)
            omega = _axis_vector(a.axis) * a.value
            v = np.cross(np.tile(omega, (len(pos), 1)), pos)
            if a.power:
                rc = np.hypot(pos[:, 0], pos[:, 1])
                v *= np.where(rc > 0, rc ** a.power, 0.0)[:, None]
        elif a.mode == 'shear':
            # v_y = -q Omega x, the shearing-box / keplerian shear
            v = np.zeros((len(pos), 3))
            v[:, 1] = -a.qshear * a.value * pos[:, 0]
        else:
            raise SystemExit(f'unknown velocity mode {a.mode!r}')
        if a.add:
            arr[np.flatnonzero(m)[:, None], [4, 5, 6]] += v
        else:
            arr[np.flatnonzero(m)[:, None], [4, 5, 6]] = v
        print(f'    {t}: velocity {a.mode} '
              f'{"added to" if a.add else "set on"} {int(m.sum())} particles')


def _axis_vector(axis):
    return np.eye(3)['xyz'.index(axis)]


def op_center(snap, a):
    """Recentre positions (and optionally velocities) on a chosen origin."""
    types = _types(a.type)
    if a.on == 'potential-min':
        gas = snap.data['gas']
        if not len(gas):
            raise SystemExit('--on potential-min needs gas particles')
        origin = gas[np.argmin(gas[:, 11]), 1:4].copy()
    elif a.on == 'density-max':
        gas = snap.data['gas']
        if not len(gas):
            raise SystemExit('--on density-max needs gas particles')
        origin = gas[np.argmax(gas[:, 7]), 1:4].copy()
    elif a.on == 'mass':
        origin = snap.centre_of_mass(types=types)
    else:
        origin = np.zeros(3)
    snap.translate(-origin, types=types)
    print(f'    centred on {a.on} at {origin}')
    if a.velocity:
        v = snap.centre_of_mass_velocity(types=types)
        snap.boost(-v, types=types)
        print(f'    removed bulk velocity {v}')


def op_remove(snap, a):
    types = _types(a.type)
    if not a.where:
        raise SystemExit('remove needs at least one --where (refusing to '
                         'delete every particle of a type by accident)')
    masks = _masks(snap, types, a.where)
    for t in types:
        m = masks[t]
        if m.any():
            snap.remove(t, m)
            print(f'    {t}: removed {int(m.sum())}, {len(snap.data[t])} left')


def op_keep(snap, a):
    types = _types(a.type)
    if not a.where:
        raise SystemExit('keep needs at least one --where')
    masks = _masks(snap, types, a.where)
    for t in types:
        m = masks[t]
        if len(m):
            snap.keep(t, m)
            print(f'    {t}: kept {int(m.sum())}')


def op_rotate(snap, a):
    snap.rotate(a.axis, a.angle, types=_types(a.type))
    vecs = [lab for lab, _ in snap.vector_columns()]
    print(f'    rotated {a.angle:g} deg about {a.axis}; vector aux rotated: '
          f'{vecs or "none"}')


def op_permute(snap, a):
    """Permute the coordinate axes (generalises 'change direction')."""
    order = [{'x': 0, 'y': 1, 'z': 2}[c] for c in a.order]
    if sorted(order) != [0, 1, 2]:
        raise SystemExit('--order must be a permutation of xyz, e.g. zxy')
    for t in _types(a.type):
        arr = snap.data[t]
        if len(arr):
            arr[:, 1:4] = arr[:, 1:4][:, order]
            arr[:, 4:7] = arr[:, 4:7][:, order]
    for lab, (get, put) in snap.vector_columns():
        put(get()[:, order])
    print(f'    axes permuted to {a.order} (positions, velocities, vector aux)')


def op_time(snap, a):
    snap.time = np.array([float(a.value)])
    print(f'    time set to {a.value:g}')


def op_bfield(snap, a):
    """Operate on the magnetic field aux: scale, zero, B/rho, B*rho, smooth."""
    # A snapshot commonly stores B TWICE - packed (.BField) and as scalar
    # component twins (.BFieldx/y/z). Both must be updated or a consumer that
    # reads the other representation silently gets the old field. Take the
    # packed copy as the source of truth and write the result back through
    # every representation.
    targets = [(k, acc) for k, acc in snap.vector_columns()
               if k.startswith(a.label)]
    if not targets:
        present = [k for k, _ in snap.vector_columns()]
        raise SystemExit(f'no {a.label} aux in this snapshot '
                         f'(vectors present: {present or "none"})')
    key = ', '.join(k for k, _ in targets)
    get, put = targets[0][1]
    B = get()
    for other_key, (g2, _p2) in targets[1:]:
        if not np.allclose(g2(), B, atol=0, rtol=1e-6):
            print(f'    WARNING: {a.label} and {other_key} disagreed on input '
                  f'(using {targets[0][0]}); they are now made consistent')
    ngas = len(snap.data['gas'])
    lo, hi = snap.span('gas')
    masks = _masks(snap, ('gas',), a.where)
    m = masks['gas']
    rows = np.zeros(len(B), dtype=bool)
    rows[lo:hi] = m

    if a.op == 'zero':
        B[rows] = 0.0
    elif a.op == 'scale':
        B[rows] *= a.value
    elif a.op in ('over-rho', 'times-rho'):
        rho = snap.data['gas'][:, 7]
        if np.any(rho <= 0) and a.op == 'over-rho':
            raise SystemExit('non-positive gas density: B/rho is undefined')
        f = (1.0 / rho) if a.op == 'over-rho' else rho
        sel = np.zeros(len(B))
        sel[lo:hi] = np.where(m, f, 1.0)
        sel[:lo] = 1.0
        sel[hi:] = 1.0
        B *= sel[:, None]
    elif a.op == 'smooth':
        B = _smooth_bfield(snap, B, lo, hi, m, a)
    else:
        raise SystemExit(f'unknown bfield op {a.op!r}')
    for _k, (_g, p) in targets:
        p(B)
    mag = np.linalg.norm(B[lo:hi], axis=1)
    print(f'    {key}: {a.op} applied to {int(m.sum())}/{ngas} gas; '
          f'|B| max {mag.max():.6e} rms {np.sqrt((mag ** 2).mean()):.6e}')


def _smooth_bfield(snap, B, lo, hi, mask, a):
    """SPH kernel smoothing of B over the gas, in a glassgen flavour.

    The menu entry the old script advertised but never implemented. Uses the
    same operator family as IC_setBField.py and mhd_divBclean, so a smoothed
    field stays in one discrete convention: the smoothed value is the
    Shepard-normalised SPH interpolant of B (`flavours.interpolate`, the
    family's zeroth member), blended by --strength (1 = fully smoothed).

    One neighbour build serves every pass: smoothing never moves a particle,
    so the state is reused rather than rebuilt.
    """
    from glassgen.flavours import interpolate, operator_state

    gas = snap.data['gas']
    pos = np.ascontiguousarray(gas[:, 1:4])
    mass = np.ascontiguousarray(gas[:, 0])
    box = None if a.box is None else np.asarray(a.box, dtype=np.float64)
    if box is not None and box.size == 1:
        box = np.repeat(box, 3)
    st = operator_state(pos, mass, box=box, nsmooth=a.nsmooth)
    Bg = np.ascontiguousarray(B[lo:hi], dtype=np.float64)
    for _ in range(max(1, a.passes)):
        out, st = interpolate(pos, mass, Bg, state=st, flavour=a.flavour)
        Bg = np.where(mask[:, None],
                      (1.0 - a.strength) * Bg + a.strength * out, Bg)
    B[lo:hi] = Bg
    return B


def op_spin(snap, a):
    """Write a spin (w) aux: an angular-velocity field with a radial profile."""
    gas = snap.data['gas']
    if not len(gas):
        raise SystemExit('spin needs gas particles')
    pos = gas[:, 1:4]
    r = np.linalg.norm(pos, axis=1)
    if a.profile == 'uniform':
        prof = np.ones(len(r))
    elif a.profile == 'gaussian':
        prof = np.exp(-(r / a.scale_radius) ** 2)
    elif a.profile == 'exp':
        prof = np.exp(-r / a.scale_radius)
    else:
        raise SystemExit(f'unknown spin profile {a.profile!r}')
    w = (a.value * prof)[:, None] * _axis_vector(a.axis)
    full = np.zeros((snap.n, 3))
    lo, hi = snap.span('gas')
    full[lo:hi] = w
    for i, c in enumerate('xyz'):
        label = 'w' + c
        existing = next((m for m in snap.aux if m['label'] == label), None)
        if existing is None:
            snap.aux.append(dict(label=label, kind='scalar',
                                 data=full[:, i].copy()))
        else:
            existing['data'] = full[:, i].copy()
    print(f'    spin {a.value:g} about {a.axis} with a {a.profile} profile '
          f'-> wx/wy/wz aux')


def op_add(snap, a):
    """Append one particle (or a binary pair) of the given type."""
    t = a.type
    row = np.zeros(NCOL[t])
    row[0] = a.mass
    row[1:4] = a.pos
    row[4:7] = a.vel
    row[FIELD_COLUMN['eps'][t]] = a.softening
    if t == 'gas':
        row[7] = a.rho
        row[8] = a.temp
    if t == 'star' and a.tform is not None:
        row[8] = a.tform
    snap.append(t, row)
    print(f'    added 1 {t} particle: m={a.mass:g} pos={a.pos} vel={a.vel} '
          f'eps={a.softening:g}')
    if a.companion_mass is not None:
        row2 = row.copy()
        row2[0] = a.companion_mass
        row2[1:4] = a.companion_pos
        row2[4:7] = a.companion_vel
        row2[FIELD_COLUMN['eps'][t]] = a.companion_softening or a.softening
        snap.append(t, row2)
        print(f'    added companion: m={a.companion_mass:g} '
              f'pos={a.companion_pos} vel={a.companion_vel}')


OPS = {
    'set': (op_set, 'set/add/multiply a scalar field on selected particles'),
    'velocity': (op_velocity, 'set a velocity field (constant/radial/'
                              'rotation/shear/zero)'),
    'center': (op_center, 'recentre on the potential minimum, density peak, '
                          'centre of mass or origin'),
    'remove': (op_remove, 'delete selected particles (and their aux rows)'),
    'keep': (op_keep, 'keep only selected particles'),
    'rotate': (op_rotate, 'rotate positions, velocities and all vector aux'),
    'permute': (op_permute, 'permute the coordinate axes'),
    'time': (op_time, 'set the snapshot time'),
    'bfield': (op_bfield, 'scale/zero/divide-by-rho/smooth the magnetic aux'),
    'spin': (op_spin, 'write a spin (wx/wy/wz) aux with a radial profile'),
    'add': (op_add, 'append a particle (optionally a binary companion)'),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(sp):
    sp.add_argument('--type', nargs='+', default=['all'],
                    choices=['all', 'gas', 'dark', 'star'],
                    help='particle types to act on (default all)')
    sp.add_argument('--where', action='append', default=[], metavar='EXPR',
                    help="selection like 'r>50' or 'rho<=1e-3'; repeatable "
                         '(ANDed). Quantities: ' + ', '.join(sorted(QUANTITIES)))


def build_parser():
    """(global parser, {op name: op parser}).

    Deliberately NOT argparse subparsers: a `nargs='?'` positional followed by
    a subparser is ambiguous (argparse hands the filename to the subcommand
    slot), and subparsers cannot repeat anyway - which the chaining needs.
    Each operation gets its own parser and the argv is split by hand.
    """
    ops = ' '.join(OPS)
    p = argparse.ArgumentParser(
        prog=os.environ.get('IC_TOOL_NAME'),   # set by the bin/ wrapper
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'operations (chainable, in order): {ops}\n'
               f'  <op> --help   flags for one operation\n'
               f'  --list-ops    one-line description of each')
    p.add_argument('infile', nargs='?')
    p.add_argument('-o', dest='out', default=None,
                   help='output snapshot (default: _mod before the .NNNNN)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--list-ops', action='store_true',
                   help='print the operation table and exit')

    class Sub:
        """Minimal stand-in for argparse's subparser registry."""

        def __init__(self):
            self.choices = {}

        def add_parser(self, name, help=None):
            sp = argparse.ArgumentParser(
                prog=name, description=f'{name}: {help}')
            self.choices[name] = sp
            return sp

    sub = Sub()

    sp = sub.add_parser('set', help=OPS['set'][1])
    sp.add_argument('--field', required=True, choices=sorted(FIELDS))
    sp.add_argument('--value', type=float, required=True)
    sp.add_argument('--mode', choices=('set', 'add', 'multiply'),
                    default='set')
    _add_common(sp)

    sp = sub.add_parser('velocity', help=OPS['velocity'][1])
    sp.add_argument('--mode', required=True,
                    choices=('constant', 'radial', 'rotation', 'shear',
                             'zero'))
    sp.add_argument('--value', type=float, default=0.0,
                    help='speed, angular speed, or shear rate Omega')
    sp.add_argument('--axis', choices=('x', 'y', 'z'), default='z')
    sp.add_argument('--power', type=float, default=0.0,
                    help='radial power: v ~ r^power (radial/rotation modes)')
    sp.add_argument('--qshear', type=float, default=1.5,
                    help='shear parameter q (default 3/2, keplerian)')
    sp.add_argument('--add', action='store_true',
                    help='add to the existing velocity instead of replacing')
    _add_common(sp)

    sp = sub.add_parser('center', help=OPS['center'][1])
    sp.add_argument('--on', choices=('potential-min', 'density-max', 'mass',
                                     'origin'), default='potential-min')
    sp.add_argument('--velocity', action='store_true',
                    help='also remove the bulk velocity')
    _add_common(sp)

    for name in ('remove', 'keep'):
        sp = sub.add_parser(name, help=OPS[name][1])
        _add_common(sp)

    sp = sub.add_parser('rotate', help=OPS['rotate'][1])
    sp.add_argument('--axis', choices=('x', 'y', 'z'), required=True)
    sp.add_argument('--angle', type=float, required=True, help='degrees')
    _add_common(sp)

    sp = sub.add_parser('permute', help=OPS['permute'][1])
    sp.add_argument('--order', required=True,
                    help='new axis order, e.g. zxy')
    _add_common(sp)

    sp = sub.add_parser('time', help=OPS['time'][1])
    sp.add_argument('--value', type=float, required=True)

    sp = sub.add_parser('bfield', help=OPS['bfield'][1])
    sp.add_argument('--op', required=True,
                    choices=('zero', 'scale', 'over-rho', 'times-rho',
                             'smooth'))
    sp.add_argument('--value', type=float, default=1.0,
                    help='factor for --op scale')
    sp.add_argument('--label', default='BField',
                    help='aux label to act on (default BField)')
    sp.add_argument('--nsmooth', type=int, default=64)
    sp.add_argument('--passes', type=int, default=1,
                    help='number of smoothing passes (--op smooth)')
    sp.add_argument('--strength', type=float, default=1.0,
                    help='blend factor per pass, 1 = fully smoothed')
    sp.add_argument('--box', type=float, nargs='+', default=None,
                    help='periodic box for the smoothing kernel')
    sp.add_argument('--flavour', default='gdforce',
                    help='operator flavour for the smoothing kernel '
                         '(gdforce = m_j/rho_j, the textbook SPH interpolant '
                         'and what this op has always used)')
    _add_common(sp)

    sp = sub.add_parser('spin', help=OPS['spin'][1])
    sp.add_argument('--value', type=float, required=True)
    sp.add_argument('--axis', choices=('x', 'y', 'z'), default='z')
    sp.add_argument('--profile', choices=('uniform', 'gaussian', 'exp'),
                    default='gaussian')
    sp.add_argument('--scale-radius', type=float, default=1.0)
    _add_common(sp)

    sp = sub.add_parser('add', help=OPS['add'][1])
    sp.add_argument('--type', choices=('gas', 'dark', 'star'), default='star')
    sp.add_argument('--mass', type=float, required=True)
    sp.add_argument('--pos', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    sp.add_argument('--vel', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    sp.add_argument('--softening', type=float, default=0.0)
    sp.add_argument('--rho', type=float, default=0.0, help='gas only')
    sp.add_argument('--temp', type=float, default=0.0, help='gas only')
    sp.add_argument('--tform', type=float, default=None, help='star only')
    sp.add_argument('--companion-mass', type=float, default=None,
                    help='append a second particle (binary/sink pair)')
    sp.add_argument('--companion-pos', type=float, nargs=3,
                    default=[0.0, 0.0, 0.0])
    sp.add_argument('--companion-vel', type=float, nargs=3,
                    default=[0.0, 0.0, 0.0])
    sp.add_argument('--companion-softening', type=float, default=None)
    return p, sub.choices


def split_chain(argv, opnames):
    """Split argv into the global part and one argv per chained operation."""
    first = len(argv)
    for i, tok in enumerate(argv):
        if tok in opnames:
            first = i
            break
    head, rest = argv[:first], argv[first:]
    chunks = []
    cur = None
    prev = None
    for tok in rest:
        # a token only STARTS a new operation if it is not the value of a
        # preceding flag - otherwise `set --mode set` would split on its own
        # argument, and `--field mass` on a field named like an op
        starts = tok in opnames and not (prev is not None
                                         and prev.startswith('-'))
        if starts:
            if cur is not None:
                chunks.append(cur)
            cur = [tok]
        else:
            cur.append(tok)
        prev = tok
    if cur:
        chunks.append(cur)
    return head, chunks


def interactive(opparsers):
    print('Operations:')
    for i, (name, (_fn, doc)) in enumerate(OPS.items(), 1):
        print(f'  {i:2d}: {name:<9} {doc}')
    print('\n(split/merge live in glassresample: '
          'glassresample IC.00000 --op split)')
    raw = input('\nOperation [name or number]: ').strip()
    names = list(OPS)
    if raw.isdigit() and 1 <= int(raw) <= len(names):
        raw = names[int(raw) - 1]
    if raw not in OPS:
        raise SystemExit(f'unknown operation {raw!r}')
    print(f'\nflags for {raw} (blank for defaults):')
    opparsers[raw].print_help()
    extra = input('\nflags: ').strip()
    return [raw] + (extra.split() if extra else [])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p, opparsers = build_parser()
    opnames = set(OPS)

    if '--list-ops' in argv:
        for name, (_fn, doc) in OPS.items():
            print(f'{name:<10} {doc}')
        return

    head, chunks = split_chain(argv, opnames)
    args = p.parse_args(head)

    # `<op> --help` before we require a file to exist
    for chunk in chunks:
        if '--help' in chunk or '-h' in chunk:
            opparsers[chunk[0]].parse_args(chunk[1:])

    infile = args.infile
    if infile is None:
        infile = input('File name: ').strip()
    if not os.path.isfile(infile):
        raise SystemExit(f'no such snapshot: {infile}')
    out = args.out
    if out is None:
        out = (re.sub(r'(\.\d+)$', r'_mod\1', infile)
               if re.search(r'\.\d+$', infile) else infile + '_mod')
    if os.path.abspath(out) == os.path.abspath(infile):
        raise SystemExit('output would overwrite the input')

    if not chunks:
        chunks = [interactive(opparsers)]

    # parse every operation BEFORE touching the file, so a typo in the last
    # op of a chain does not leave a half-edited snapshot behind
    plan = [(chunk[0], opparsers[chunk[0]].parse_args(chunk[1:]))
            for chunk in chunks]

    snap = Snapshot.read(infile)
    c = snap.counts
    print(f'read {infile}: {c["gas"]} gas, {c["dark"]} dark, {c["star"]} star'
          f'; aux {[m["label"] for m in snap.aux] or "none"}')

    for name, a in plan:
        print(f'\n[{name}]')
        try:
            OPS[name][0](snap, a)
        except ValueError as e:          # bad selection/field -> clean message
            raise SystemExit(f'{name}: {e}')

    c = snap.counts
    print(f'\nresult: {c["gas"]} gas, {c["dark"]} dark, {c["star"]} star '
          f'({snap.n} total)')
    if args.dry_run:
        print('--dry-run: nothing written')
        return
    wrote = snap.write(out)
    print(f'wrote {out}' + (('\n  ' + '\n  '.join(wrote)) if wrote else ''))


if __name__ == '__main__':
    main()
