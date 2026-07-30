"""Tipsy snapshot + aux as one editable object, with a general selection and
rotation vocabulary.

The thing every IC-editing script re-implements badly: a snapshot is not just
three particle arrays, it is those arrays PLUS a set of aux sibling files that
span all three types under a single count header. Any edit that changes the
particle count or their order (removing, appending, concatenating two
snapshots) must carry every aux row with it, or the IC silently acquires a
magnetic field belonging to different particles.

    snap = Snapshot.read('ic.00000')
    mask = snap.select('gas', ['r>50'])        # general selection vocabulary
    snap.remove('gas', mask)                   # particles AND aux rows
    snap.rotate('x', 30.0)                     # positions, velocities, AND
                                               # every vector aux (B, spin)
    snap.write('out.00000')

SELECTION: `select` takes expression strings like 'r>50', 'rho<=1e-3',
'absz>2', 'index==17', ANDed together. The available quantities are derived
from the type's own columns (see QUANTITIES), so the same vocabulary drives
removal, masked field edits and diagnostics instead of each having its own
numbered sub-menu.

ROTATION: rotating a snapshot means rotating every VECTOR it carries, not just
positions. `rotate` does positions, velocities, every aux of kind 'vector'
(BField, CurlB), and any x/y/z-suffixed scalar aux triple (BFieldx/y/z,
wx/wy/wz) - the last is why a naive per-column rotation is wrong.

Tipsy column layout (the authority for the column maps below):
    gas  (12): mass x y z vx vy vz rho temp eps metals phi
    dark  (9): mass x y z vx vy vz eps phi
    star (11): mass x y z vx vy vz metals tform eps phi
NB gas column 9 is the gravitational SOFTENING, not the SPH smoothing length
(which is never stored - it is recomputed at runtime from density/nSmooth).
"""
import glob
import os
import re
import shutil

import numpy as np

import readtipsy as tip

TYPES = ('gas', 'dark', 'star')
NCOL = {'gas': 12, 'dark': 9, 'star': 11}

# per-type scalar field name -> column index
FIELD_COLUMN = {
    'mass': {'gas': 0, 'dark': 0, 'star': 0},
    'rho': {'gas': 7},
    'temp': {'gas': 8},
    'metals': {'gas': 10, 'star': 7},
    'tform': {'star': 8},
    'eps': {'gas': 9, 'dark': 7, 'star': 9},
    'phi': {'gas': 11, 'dark': 8, 'star': 10},
}
FIELDS = tuple(FIELD_COLUMN)

# selection quantities: name -> callable(arr, type) -> (N,) values
QUANTITIES = {
    'x': lambda a, t: a[:, 1],
    'y': lambda a, t: a[:, 2],
    'z': lambda a, t: a[:, 3],
    'absx': lambda a, t: np.abs(a[:, 1]),
    'absy': lambda a, t: np.abs(a[:, 2]),
    'absz': lambda a, t: np.abs(a[:, 3]),
    'r': lambda a, t: np.linalg.norm(a[:, 1:4], axis=1),
    'rcyl': lambda a, t: np.hypot(a[:, 1], a[:, 2]),
    'vx': lambda a, t: a[:, 4],
    'vy': lambda a, t: a[:, 5],
    'vz': lambda a, t: a[:, 6],
    'v': lambda a, t: np.linalg.norm(a[:, 4:7], axis=1),
    'index': lambda a, t: np.arange(len(a), dtype=np.float64),
}
for _name in FIELDS:
    def _mk(nm):
        def q(a, t):
            col = FIELD_COLUMN[nm].get(t)
            if col is None:
                raise ValueError(f'{t} particles have no {nm!r} column')
            return a[:, col]
        return q
    QUANTITIES[_name] = _mk(_name)

_EXPR = re.compile(r'^\s*([A-Za-z_]\w*)\s*(>=|<=|==|!=|>|<)\s*'
                   r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*$')
_OPS = {'>': np.greater, '<': np.less, '>=': np.greater_equal,
        '<=': np.less_equal, '==': np.equal, '!=': np.not_equal}


class Snapshot:
    """Tipsy particle arrays + every aux sibling file, edited together."""

    def __init__(self, data, header, time, aux, path=None):
        self.data = data                     # {'gas': (n,12), ...}
        self.header = np.asarray(header).copy()
        self.time = time
        self.aux = aux                       # [{label, kind, data}, ...]
        self.path = path

    # -- construction ----------------------------------------------------

    @classmethod
    def read(cls, path):
        gas, dark, star, header, time = tip.readtipsy(path)
        data = {'gas': np.asarray(gas, dtype=np.float64).reshape(-1, 12),
                'dark': np.asarray(dark, dtype=np.float64).reshape(-1, 9),
                'star': np.asarray(star, dtype=np.float64).reshape(-1, 11)}
        snap = cls(data, header, time, read_aux(path), path=path)
        snap._check_aux()
        return snap

    def _check_aux(self):
        """Aux files must span every particle; a mismatch means the aux
        belongs to a different snapshot and MUST NOT be carried silently."""
        n = self.n
        for m in self.aux:
            if len(m['data']) != n:
                raise ValueError(
                    f"aux .{m['label']} has {len(m['data'])} rows but the "
                    f'snapshot has {n} particles - it belongs to a different '
                    'file')

    # -- geometry --------------------------------------------------------

    @property
    def n(self):
        return sum(len(self.data[t]) for t in TYPES)

    @property
    def counts(self):
        return {t: len(self.data[t]) for t in TYPES}

    def span(self, t):
        """(lo, hi) row range of type `t` inside a whole-snapshot aux array."""
        lo = 0
        for u in TYPES:
            if u == t:
                return lo, lo + len(self.data[u])
            lo += len(self.data[u])
        raise KeyError(t)

    def sync_header(self):
        c = self.counts
        self.header[0] = c['gas'] + c['dark'] + c['star']
        self.header[2], self.header[3], self.header[4] = (
            c['gas'], c['dark'], c['star'])

    # -- selection -------------------------------------------------------

    def select(self, t, exprs):
        """Boolean mask over type `t` from ANDed expressions like 'r>50'."""
        arr = self.data[t]
        mask = np.ones(len(arr), dtype=bool)
        for e in (exprs or []):
            m = _EXPR.match(e)
            if not m:
                raise ValueError(
                    f'cannot parse selection {e!r}; expected QUANTITY OP VALUE '
                    f"e.g. 'r>50' (quantities: {sorted(QUANTITIES)})")
            name, op, val = m.group(1), m.group(2), float(m.group(3))
            if name not in QUANTITIES:
                raise ValueError(f'unknown selection quantity {name!r} '
                                 f'(available: {sorted(QUANTITIES)})')
            mask &= _OPS[op](QUANTITIES[name](arr, t), val)
        return mask

    # -- edits -----------------------------------------------------------

    def keep(self, t, mask):
        """Keep only `mask` particles of type t, dropping their aux rows too."""
        mask = np.asarray(mask, dtype=bool)
        lo, hi = self.span(t)
        keep_all = np.ones(self.n, dtype=bool)
        keep_all[lo:hi] = mask
        self.data[t] = self.data[t][mask]
        for m in self.aux:
            m['data'] = m['data'][keep_all]
        self.sync_header()

    def remove(self, t, mask):
        self.keep(t, ~np.asarray(mask, dtype=bool))

    def append(self, t, rows, aux_rows=None):
        """Append particles of type t (rows (k,ncol)); aux_rows maps label ->
        (k,...) with zeros assumed for any label not supplied."""
        rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        if rows.shape[1] != NCOL[t]:
            raise ValueError(f'{t} rows must have {NCOL[t]} columns')
        k = len(rows)
        lo, hi = self.span(t)
        self.data[t] = np.vstack([self.data[t], rows])
        for m in self.aux:
            block = (np.zeros((k, 3)) if m['kind'] == 'vector'
                     else np.zeros(k, dtype=m['data'].dtype))
            if aux_rows and m['label'] in aux_rows:
                block = np.asarray(aux_rows[m['label']]).reshape(block.shape)
            m['data'] = np.concatenate([m['data'][:hi], block, m['data'][hi:]])
        self.sync_header()

    def vector_columns(self):
        """Every (x,y,z) vector this snapshot carries in its aux files.

        Yields (getter, setter) pairs over (N,3) arrays: aux of kind 'vector'
        directly, plus scalar aux triples whose labels differ only by a
        trailing x/y/z (BFieldx/y/z, wx/wy/wz). Missing the second kind is the
        classic rotation bug - the packed vector gets rotated and its
        component twins do not, leaving two disagreeing copies of B.
        """
        by_label = {m['label']: m for m in self.aux}
        out = []
        for m in self.aux:
            if m['kind'] == 'vector':
                out.append((m['label'], _direct_accessor(m)))
        seen = set()
        for m in self.aux:
            lab = m['label']
            if m['kind'] != 'scalar' or not lab.endswith('x'):
                continue
            stem = lab[:-1]
            trio = [by_label.get(stem + c) for c in 'xyz']
            if all(t is not None and t['kind'] == 'scalar' for t in trio) \
                    and stem not in seen:
                seen.add(stem)
                out.append((stem + '{x,y,z}', _triple_accessor(trio)))
        return out

    def rotate(self, axis, degrees, types=TYPES, about=None):
        """Rotate positions, velocities and EVERY vector aux about `axis`."""
        R = rotation_matrix(axis, degrees)
        origin = (np.zeros(3) if about is None
                  else np.asarray(about, dtype=np.float64))
        for t in types:
            a = self.data[t]
            if not len(a):
                continue
            a[:, 1:4] = (a[:, 1:4] - origin) @ R.T + origin
            a[:, 4:7] = a[:, 4:7] @ R.T
        for _label, (get, put) in self.vector_columns():
            put(get() @ R.T)
        return R

    def translate(self, shift, types=TYPES):
        shift = np.asarray(shift, dtype=np.float64).reshape(3)
        for t in types:
            if len(self.data[t]):
                self.data[t][:, 1:4] += shift

    def boost(self, dv, types=TYPES):
        dv = np.asarray(dv, dtype=np.float64).reshape(3)
        for t in types:
            if len(self.data[t]):
                self.data[t][:, 4:7] += dv

    def total_mass(self, types=TYPES, mask=None):
        return float(sum(self.data[t][:, 0].sum() if mask is None
                         else self.data[t][mask[t], 0].sum()
                         for t in types if len(self.data[t])))

    def centre_of_mass(self, types=TYPES):
        num = np.zeros(3)
        den = 0.0
        for t in types:
            a = self.data[t]
            if len(a):
                num += (a[:, 0:1] * a[:, 1:4]).sum(axis=0)
                den += a[:, 0].sum()
        return num / den if den > 0 else num

    def centre_of_mass_velocity(self, types=TYPES):
        num = np.zeros(3)
        den = 0.0
        for t in types:
            a = self.data[t]
            if len(a):
                num += (a[:, 0:1] * a[:, 4:7]).sum(axis=0)
                den += a[:, 0].sum()
        return num / den if den > 0 else num

    # -- output ----------------------------------------------------------

    def write(self, out, copy_unmatched_aux=True):
        self.sync_header()
        self._check_aux()
        tip.writetipsy(self.data['gas'], self.data['dark'], self.data['star'],
                       out, self.header, time=self.time)
        write_aux(self.aux, out)
        wrote = [f"{out}.{m['label']}" for m in self.aux]
        if copy_unmatched_aux and self.path:
            known = {m['label'] for m in self.aux}
            for path in glob.glob(self.path + '.*'):
                lab = path[len(self.path) + 1:]
                if lab and lab not in known and '/' not in lab \
                        and not lab.endswith('.param'):
                    shutil.copyfile(path, out + '.' + lab)
        return wrote


def _direct_accessor(m):
    def get():
        return np.asarray(m['data'], dtype=np.float64).reshape(-1, 3)

    def put(v):
        m['data'] = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    return get, put


def _triple_accessor(trio):
    def get():
        return np.stack([np.asarray(t['data'], dtype=np.float64)
                         for t in trio], axis=1)

    def put(v):
        v = np.asarray(v, dtype=np.float64)
        for i, t in enumerate(trio):
            t['data'] = v[:, i]
    return get, put


def rotation_matrix(axis, degrees):
    """Right-handed rotation matrix about 'x'|'y'|'z' by `degrees`."""
    a = np.radians(float(degrees))
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be x, y or z (got {axis!r})")


def read_aux(infile):
    """All aux sibling files of `infile` as [{label, kind, data}].

    Shares glassgen.resample_cli's reader so a snapshot edited here and one
    resampled by glassresample agree on what an aux file is (vector vs scalar
    vs int, and which labels are banned).
    """
    import contextlib
    import io

    import readtipsy as _tip
    out = []
    # readtipsy's readers print a line per file; that is debug noise here and
    # it buries the actual operation log
    quiet = contextlib.redirect_stdout(io.StringIO())
    for path in sorted(glob.glob(infile + '.*')):
        label = path[len(infile) + 1:]
        if not label or '/' in label:
            continue
        if any(b in label for b in _tip.banauxfiles):
            continue
        if label.endswith('.param') or label == 'param':
            continue
        if os.path.isdir(path):
            continue
        if label in _tip.vecauxfiles:
            with quiet:
                data, _ = _tip.readtipsyVecaux(infile, label)
            data = np.asarray(data, dtype=np.float64).reshape(-1, 3)
            kind = 'vector'
        else:
            with quiet:
                data, _ = _tip.readtipsyaux(infile, label)
            data = np.asarray(data)
            kind = 'int' if label in _tip.intauxfiles else 'scalar'
        out.append(dict(label=label, kind=kind, data=data))
    return out


def write_aux(meta, outfile):
    for m in meta:
        if m['kind'] == 'vector':
            tip.writetipsyVecaux(np.asarray(m['data'], dtype=np.float64),
                                 m['label'], outfile)
        else:
            tip.writetipsyaux(np.asarray(m['data']), m['label'], outfile)


def concatenate(a, b):
    """Concatenate two snapshots type by type, aux included.

    Both must carry the SAME aux labels (an aux present on only one side would
    otherwise be silently zero-filled for half the particles); the caller is
    told which labels differ rather than being handed a corrupt IC.
    """
    la = {m['label'] for m in a.aux}
    lb = {m['label'] for m in b.aux}
    if la != lb:
        raise ValueError('the two snapshots carry different aux files: '
                         f'only in first {sorted(la - lb)}, only in second '
                         f'{sorted(lb - la)}')
    data = {t: np.vstack([a.data[t], b.data[t]]) for t in TYPES}
    aux = []
    for m in a.aux:
        other = next(o for o in b.aux if o['label'] == m['label'])
        blocks = []
        for t in TYPES:
            lo_a, hi_a = a.span(t)
            lo_b, hi_b = b.span(t)
            blocks.append(m['data'][lo_a:hi_a])
            blocks.append(other['data'][lo_b:hi_b])
        aux.append(dict(label=m['label'], kind=m['kind'],
                        data=np.concatenate(blocks)))
    out = Snapshot(data, a.header.copy(), a.time, aux, path=a.path)
    out.sync_header()
    return out
