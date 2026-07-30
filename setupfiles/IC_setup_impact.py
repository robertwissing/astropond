#!/usr/bin/env python3
"""Set up a two-body impact / merger from one or two tipsy snapshots.

The generalisation of setQuantity.py's "13: Impact (clone)": that version could
only clone ONE object, took the masses as typed-in numbers, hard-coded a
head-on escape-velocity encounter, and offset the impactor by hand. Here:

  * TWO inputs (target and impactor). Omitting the impactor clones the target,
    which is the old behaviour and the right default for an equal-mass merger.
  * masses are MEASURED from the particles, not typed in (--mass-within
    restricts the measurement to a radius; --mass-value overrides).
  * the encounter is a proper two-body problem placed in the CENTRE-OF-MASS
    frame with momentum conservation, so an unequal-mass pair gets the correct
    velocity split instead of an equal-and-opposite guess.
  * every VECTOR the snapshots carry is rotated with them - positions,
    velocities, packed vector aux (BField) AND scalar component twins
    (BFieldx/y/z, wx/wy/wz). Rotating only some of those leaves two
    disagreeing copies of B in the IC.

TWO ORBIT PARAMETERISATIONS (--orbit), because galaxy mergers and planetary
impacts are quoted differently:

  keplerian   --separation r --pericentre rp --ecc e
              The exact two-body solution: semi-latus rectum p = rp (1 + e),
              angular momentum L = sqrt(G M p), speed from vis-viva
              v^2 = G M (2/r - 1/a) with a = rp/(1-e) (v^2 = 2 G M / r for the
              parabolic e = 1). The velocity is decomposed into a tangential
              part L/r and an inward radial part, so the pair is ON the
              specified orbit at the specified separation - it will actually
              reach that pericentre. This is what a galaxy-merger orbit means.

  simple      --separation r --vinf V --impact-parameter b
              The impact-angle convention: relative speed
              v = sqrt(v_esc^2 + vinf^2) with v_esc = sqrt(2 G M / r) directed
              along the line of centres, and the impactor offset TRANSVERSELY
              by b (in units of the separation by default, so b = sin of the
              impact angle; --impact-parameter-abs for a length). This
              reproduces setQuantity's geometry, cleaned up.

Units are code units with G = 1 by default (the gasoline convention); pass
--G for anything else. Velocities/positions are whatever the snapshots use.

    IC_setup_impact.py galaxy.00000 -o merger.00000 \
        --orbit keplerian --separation 217 --pericentre 20 --ecc 1.0 \
        --incline-impactor x 60 --spin-flip

    IC_setup_impact.py target.00000 impactor.00000 -o impact.00000 \
        --orbit simple --separation 5 --vinf 0 --impact-parameter 0.5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tipsysnap import Snapshot, concatenate  # noqa: E402


def measure_mass(snap, within=None, types=('gas', 'dark', 'star'),
                 centre=None):
    """Total mass, optionally only inside a radius about `centre`."""
    if within is None:
        return snap.total_mass(types=types)
    c = np.zeros(3) if centre is None else np.asarray(centre)
    tot = 0.0
    for t in types:
        a = snap.data[t]
        if len(a):
            r = np.linalg.norm(a[:, 1:4] - c, axis=1)
            tot += float(a[r < within, 0].sum())
    return tot


def measure_radius(snap, types=('gas', 'dark', 'star'), percentile=99.0,
                   centre=None):
    """(r_percentile, r_max) of the particle distribution about `centre`.

    The percentile, not the maximum, is the working radius of an object: a
    handful of ejected particles can sit ten times further out than anything
    dynamically meaningful, and letting them set the size would veto every
    sensible orbit. The maximum comes back too so the caller can say how far
    the tail actually reaches.
    """
    c = np.zeros(3) if centre is None else np.asarray(centre)
    r = np.concatenate([
        np.linalg.norm(snap.data[t][:, 1:4] - c, axis=1)
        for t in types if len(snap.data[t])] or [np.zeros(1)])
    return float(np.percentile(r, percentile)), float(r.max())


def check_overlap(r1, r2, separation, pericentre, allow, label1='target',
                  label2='impactor'):
    """Warn (or refuse) when the two objects interpenetrate at t = 0.

    The orbit code is about masses and never looks at how big anything is, so
    nothing else in this script would notice that a 20 kpc separation between
    two 30 kpc discs means the IC starts with the two already merged.

    Checked at the INITIAL separation, and for a keplerian orbit also at
    pericentre - an orbit that starts clear but is specified to pass inside
    the sum of the radii is the more common way to get this wrong, and it is
    not necessarily an error (that IS a merger), so it is only ever a note.
    """
    total = r1 + r2
    ok = True
    print(f'\nextent: {label1} r99 {r1:.4g}, {label2} r99 {r2:.4g}, '
          f'sum {total:.4g} vs separation {separation:.4g}')
    if total > separation:
        msg = (f'the two objects OVERLAP at the initial separation: '
               f'r99({label1}) + r99({label2}) = {total:.4g} > '
               f'{separation:.4g}. The IC would start already merged.')
        if allow:
            print(f'  WARNING: {msg}\n           (--allow-overlap given, '
                  'continuing)')
        else:
            raise SystemExit(
                f'  ERROR: {msg}\n  Raise --separation above {total:.4g}, '
                'measure the radii on fewer particle types with '
                '--mass-types, or pass --allow-overlap if this is deliberate.')
        ok = False
    elif total > 0.5 * separation:
        print(f'  NOTE: the gap between the two surfaces is only '
              f'{separation - total:.4g}, less than their combined size - '
              'tides are already significant at t = 0.')
    if pericentre is not None and total > pericentre:
        print(f'  NOTE: pericentre {pericentre:.4g} is inside the combined '
              f'radius {total:.4g}, so this orbit is a collision by '
              'construction (often the point, but say so deliberately).')
    return ok


def keplerian_velocity(G, M, r, rp, e):
    """(v_radial, v_tangential) of the relative orbit at separation r.

    v_radial is NEGATIVE (approaching). Raises when the requested separation
    is inside pericentre or the orbit cannot reach r, rather than silently
    returning a NaN velocity that would produce a nonsense IC.
    """
    if rp <= 0:
        raise SystemExit('--pericentre must be > 0')
    if e < 0:
        raise SystemExit('--ecc must be >= 0')
    if r < rp:
        raise SystemExit(f'--separation {r:g} is inside the pericentre '
                         f'{rp:g}; the pair would already be past closest '
                         'approach')
    p = rp * (1.0 + e)                      # semi-latus rectum
    L = np.sqrt(G * M * p)                  # specific angular momentum
    if abs(e - 1.0) < 1e-12:
        v2 = 2.0 * G * M / r                # parabolic
    else:
        a = rp / (1.0 - e)                  # negative for e > 1
        v2 = G * M * (2.0 / r - 1.0 / a)
    if v2 <= 0:
        raise SystemExit(
            f'a bound orbit with pericentre {rp:g} and e={e:g} has apocentre '
            f'{rp * (1 + e) / (1 - e):g} < separation {r:g}: it never reaches '
            'that separation. Raise --ecc or lower --separation.')
    vt = L / r
    radicand = v2 - vt ** 2
    if radicand < 0:
        if radicand > -1e-9 * v2:
            radicand = 0.0                  # exactly at pericentre
        else:
            raise SystemExit(
                f'inconsistent orbit: tangential speed exceeds the total at '
                f'r={r:g} (rp={rp:g}, e={e:g})')
    return -np.sqrt(radicand), vt


def simple_velocity(G, M, r, vinf):
    """Relative speed along the line of centres: sqrt(v_esc^2 + vinf^2)."""
    vesc = np.sqrt(2.0 * G * M / r)
    return np.sqrt(vesc ** 2 + float(vinf) ** 2), vesc


def apply_inclinations(snap, incls, label):
    for axis, deg in incls:
        snap.rotate(axis, deg)
        print(f'  {label}: rotated {deg:g} deg about {axis}')


def build_parser():
    p = argparse.ArgumentParser(
        prog=os.environ.get('IC_TOOL_NAME'),   # set by the bin/ wrapper
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('target', help='target snapshot')
    p.add_argument('impactor', nargs='?', default=None,
                   help='impactor snapshot (default: clone the target)')
    p.add_argument('-o', dest='out', default=None,
                   help='output snapshot (default: <target>_impact.NNNNN)')

    g = p.add_argument_group('orbit')
    g.add_argument('--orbit', choices=('keplerian', 'simple'),
                   default='keplerian')
    g.add_argument('--separation', type=float, default=None,
                   help='initial separation of the two centres (required)')
    g.add_argument('--pericentre', type=float, default=None,
                   help='keplerian: pericentre distance of the relative orbit')
    g.add_argument('--ecc', type=float, default=1.0,
                   help='keplerian: eccentricity (1 = parabolic, default)')
    g.add_argument('--vinf', type=float, default=0.0,
                   help='simple: relative speed at infinity (default 0)')
    g.add_argument('--impact-parameter', type=float, default=0.0,
                   help='simple: transverse offset, in units of the '
                        'separation (sin of the impact angle); 0 = head-on')
    g.add_argument('--impact-parameter-abs', action='store_true',
                   help='treat --impact-parameter as a length, not a fraction')
    g.add_argument('--offset-axis', choices=('y', 'z'), default='y',
                   help='axis the impact-parameter offset is applied along '
                        '(default y: in the plane)')
    g.add_argument('--G', type=float, default=1.0,
                   help='gravitational constant in code units (default 1)')

    g = p.add_argument_group('masses (measured from the particles by default)')
    g.add_argument('--mass-within', type=float, default=None,
                   help='measure each mass only inside this radius of its own '
                        'centre (e.g. a virial radius)')
    g.add_argument('--mass-target', type=float, default=None,
                   help='override the measured target mass')
    g.add_argument('--mass-impactor', type=float, default=None,
                   help='override the measured impactor mass')
    g.add_argument('--mass-types', nargs='+', default=['gas', 'dark', 'star'],
                   choices=['gas', 'dark', 'star'],
                   help='particle types counted in the mass (default all)')

    g = p.add_argument_group('overlap check')
    g.add_argument('--radius-percentile', type=float, default=99.0,
                   help='percentile of the particle radii taken as an '
                        "object's size (default 99; 100 = the outermost "
                        'particle, which a few strays can dominate)')
    g.add_argument('--radius-target', type=float, default=None,
                   help='override the measured target radius')
    g.add_argument('--radius-impactor', type=float, default=None,
                   help='override the measured impactor radius')
    g.add_argument('--allow-overlap', action='store_true',
                   help='downgrade the overlap error to a warning (the two '
                        'objects genuinely interpenetrating at t=0 is '
                        'occasionally what you want)')
    g.add_argument('--no-overlap-check', dest='overlap_check',
                   action='store_false',
                   help='skip the check entirely')

    g = p.add_argument_group('orientation')
    g.add_argument('--incline-target', nargs=2, action='append', default=[],
                   metavar=('AXIS', 'DEG'),
                   help='rotate the target DEG degrees about AXIS (repeatable)')
    g.add_argument('--incline-impactor', nargs=2, action='append', default=[],
                   metavar=('AXIS', 'DEG'),
                   help='rotate the impactor (repeatable, applied in order)')
    g.add_argument('--spin-flip', action='store_true',
                   help='flip the impactor 180 deg about z (retrograde '
                        'encounter)')

    g = p.add_argument_group('framing / output')
    g.add_argument('--centre-each', choices=('none', 'mass', 'potential-min'),
                   default='mass',
                   help='recentre each object on its own centre before '
                        'placing it (default mass)')
    g.add_argument('--no-com-frame', dest='com_frame', action='store_false',
                   help='place the target at the origin at rest instead of '
                        'putting the pair in its centre-of-mass frame')
    g.add_argument('--dry-run', action='store_true')
    return p


def centre_object(snap, mode, label):
    if mode == 'none':
        return np.zeros(3)
    if mode == 'mass':
        c = snap.centre_of_mass()
    else:
        gas = snap.data['gas']
        if not len(gas):
            raise SystemExit(f'{label}: --centre-each potential-min needs gas')
        c = gas[np.argmin(gas[:, 11]), 1:4].copy()
    snap.translate(-c)
    v = snap.centre_of_mass_velocity()
    snap.boost(-v)
    print(f'  {label}: centred on {mode} at {c}, bulk velocity {v} removed')
    return c


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.separation is None:
        ap.error('--separation is required')
    if args.orbit == 'keplerian' and args.pericentre is None:
        ap.error('--orbit keplerian requires --pericentre')

    out = args.out
    if out is None:
        import re
        out = (re.sub(r'(\.\d+)$', r'_impact\1', args.target)
               if re.search(r'\.\d+$', args.target)
               else args.target + '_impact')
    if os.path.abspath(out) == os.path.abspath(args.target):
        ap.error('output would overwrite the target snapshot')

    print(f'target:   {args.target}')
    target = Snapshot.read(args.target)
    if args.impactor:
        print(f'impactor: {args.impactor}')
        impactor = Snapshot.read(args.impactor)
    else:
        print('impactor: (clone of the target)')
        impactor = Snapshot.read(args.target)
    for name, s in (('target', target), ('impactor', impactor)):
        c = s.counts
        print(f'  {name}: {c["gas"]} gas, {c["dark"]} dark, {c["star"]} star; '
              f'aux {[m["label"] for m in s.aux] or "none"}')

    # ---- centre each object on itself ----------------------------------
    centre_object(target, args.centre_each, 'target')
    centre_object(impactor, args.centre_each, 'impactor')

    # ---- orientation ----------------------------------------------------
    apply_inclinations(target, [(a, float(d)) for a, d in args.incline_target],
                       'target')
    incl = [(a, float(d)) for a, d in args.incline_impactor]
    if args.spin_flip:
        incl.append(('z', 180.0))
    apply_inclinations(impactor, incl, 'impactor')

    # ---- masses ---------------------------------------------------------
    types = tuple(args.mass_types)
    M1 = (args.mass_target if args.mass_target is not None
          else measure_mass(target, args.mass_within, types))
    M2 = (args.mass_impactor if args.mass_impactor is not None
          else measure_mass(impactor, args.mass_within, types))
    if M1 <= 0 or M2 <= 0:
        raise SystemExit(f'non-positive mass measured (M1={M1:g}, M2={M2:g}); '
                         'check --mass-within / --mass-types')
    M = M1 + M2
    where = (f' within r < {args.mass_within:g}' if args.mass_within
             else ' (total)')
    print(f'\nmasses{where}, types {types}:')
    print(f'  M_target {M1:.6e}   M_impactor {M2:.6e}   '
          f'total {M:.6e}   mass ratio q = {M2 / M1:.4g}')

    # ---- do the two objects actually fit at this separation? ------------
    # measured on the same particle types the masses were, and after
    # centring, so the radii are about each object's own centre
    R1, R1max = measure_radius(target, types, args.radius_percentile)
    R2, R2max = measure_radius(impactor, types, args.radius_percentile)
    if args.radius_target is not None:
        R1 = float(args.radius_target)
    if args.radius_impactor is not None:
        R2 = float(args.radius_impactor)
    if args.overlap_check:
        check_overlap(R1, R2, float(args.separation),
                      args.pericentre if args.orbit == 'keplerian' else None,
                      args.allow_overlap)
        for name, rp, rmax in (('target', R1, R1max),
                               ('impactor', R2, R2max)):
            if rmax > 3.0 * rp:
                print(f'  ({name}: outermost particle at {rmax:.4g}, '
                      f'{rmax / rp:.1f}x the r{args.radius_percentile:g} - a '
                      'sparse tail, ignored by the check above)')

    # ---- orbit ----------------------------------------------------------
    r = float(args.separation)
    sep_vec = np.array([r, 0.0, 0.0])
    off_axis = 'xyz'.index(args.offset_axis)
    if args.orbit == 'keplerian':
        vr, vt = keplerian_velocity(args.G, M, r, args.pericentre, args.ecc)
        v_rel = np.zeros(3)
        v_rel[0] = vr                                # along the separation
        v_rel[off_axis] = vt                         # perpendicular
        kind = ('parabolic' if abs(args.ecc - 1) < 1e-12 else
                'bound' if args.ecc < 1 else 'hyperbolic')
        print(f'\norbit: keplerian, e = {args.ecc:g} ({kind}), '
              f'pericentre {args.pericentre:g}, separation {r:g}')
        print(f'  |v_rel| {np.linalg.norm(v_rel):.6e}   radial {vr:.6e}   '
              f'tangential {vt:.6e}  (along {args.offset_axis})')
    else:
        speed, vesc = simple_velocity(args.G, M, r, args.vinf)
        v_rel = np.array([-speed, 0.0, 0.0])
        b = args.impact_parameter * (1.0 if args.impact_parameter_abs else r)
        sep_vec[off_axis] = b
        print(f'\norbit: simple, separation {r:g}, v_inf {args.vinf:g}')
        print(f'  v_esc {vesc:.6e}   |v_rel| {speed:.6e}   '
              f'transverse offset {b:.6e} along {args.offset_axis}')

    # ---- place the pair --------------------------------------------------
    if args.com_frame:
        # centre-of-mass frame: r_i and v_i split by the mass fractions, so
        # total momentum is exactly zero and the COM sits at the origin
        r1 = -(M2 / M) * sep_vec
        r2 = +(M1 / M) * sep_vec
        v1 = -(M2 / M) * v_rel
        v2 = +(M1 / M) * v_rel
    else:
        r1 = np.zeros(3)
        r2 = sep_vec
        v1 = np.zeros(3)
        v2 = v_rel
    target.translate(r1)
    target.boost(v1)
    impactor.translate(r2)
    impactor.boost(v2)
    print(f'\nplacement ({"COM frame" if args.com_frame else "target at rest"}):')
    print(f'  target   at {r1}  v {v1}')
    print(f'  impactor at {r2}  v {v2}')

    merged = concatenate(target, impactor)
    c = merged.counts
    print(f'\ncombined: {c["gas"]} gas, {c["dark"]} dark, {c["star"]} star '
          f'({merged.n} total)')
    p_tot = sum(merged.data[t][:, 0:1].T @ merged.data[t][:, 4:7]
                for t in ('gas', 'dark', 'star') if len(merged.data[t]))
    print(f'  total momentum {np.asarray(p_tot).ravel()}  '
          f'(should be ~0 in the COM frame)')
    print(f'  centre of mass {merged.centre_of_mass()}')

    if args.dry_run:
        print('\n--dry-run: nothing written')
        return
    wrote = merged.write(out, copy_unmatched_aux=False)
    print(f'\nwrote {out}' + (('\n  ' + '\n  '.join(wrote)) if wrote else ''))
    audit = dict(target=args.target, impactor=args.impactor or args.target,
                 clone=args.impactor is None, out=out, orbit=args.orbit,
                 separation=r, pericentre=args.pericentre, ecc=args.ecc,
                 vinf=args.vinf, impact_parameter=args.impact_parameter,
                 G=args.G, M_target=M1, M_impactor=M2, mass_ratio=M2 / M1,
                 v_rel=v_rel.tolist(), r_target=r1.tolist(),
                 r_impactor=r2.tolist(), v_target=v1.tolist(),
                 v_impactor=v2.tolist(), com_frame=args.com_frame,
                 radius_percentile=args.radius_percentile,
                 radius_target=R1, radius_impactor=R2,
                 radius_target_max=R1max, radius_impactor_max=R2max,
                 counts=c)
    with open(out + '_impact.json', 'w') as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f'  {out}_impact.json  (orbit + masses)')


if __name__ == '__main__':
    main()
