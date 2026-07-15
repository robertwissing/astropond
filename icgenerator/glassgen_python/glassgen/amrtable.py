"""Relax-ready AMR density table (the CONSUME side of the capture pipeline).

`AMRTable` wraps an ALREADY-DEPOSITED AMR leaf set (per-cell densities) as a
glassgen `DensityField` the relax can query at every particle every iteration.
It does NO interpolation itself - the deposit is done by sph_interp (see
`reconstruct.capture_amr_raw`, which drives the same engine as
`setupfiles/IC_interpolate_to_grid.py`). Storage is per-level blocks of sorted
int64 linear cell keys + leaf values; a point query costs one binary search per
populated level (levels tried in descending leaf count), with trilinear
('linear') or piecewise-constant ('flat') readout.

Built from an sph_interp AMRGrid via `from_grid` (duck-typed - only reads
`lo/hi/cell_level/base_npx/cell_lo/cell_size/periodic/ncell`), and round-trips
through .npz with `save()`/`load()` WITHOUT sph_interp installed.

Moved here from `reconstruct.py` so glassgen owns the relax-target lookup
structure (the envelope port); `reconstruct.py` re-exports `AMRTable` for
backwards compatibility.
"""
import numpy as np
from numba import njit, prange

from .density import DensityField


@njit(cache=True, inline='always')
def _pwc_lookup(tx, ty, tz, dims, offsets, keys, vals, order):
    """Piecewise-constant AMR lookup at fractional domain coords in [0,1):
    walk the levels in `order` (descending leaf count), compute the linear
    cell key at that level and binary-search the level's sorted key block.
    A space-filling leaf set guarantees exactly one hit. Returns
    (value, level); level -1 = miss."""
    nlev = dims.shape[0]
    for oi in range(nlev):
        lv = order[oi]
        a = offsets[lv]
        b = offsets[lv + 1]
        if a == b:
            continue
        nx, ny, nz = dims[lv, 0], dims[lv, 1], dims[lv, 2]
        ix = int(tx * nx)
        iy = int(ty * ny)
        iz = int(tz * nz)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1
        if iz < 0:
            iz = 0
        elif iz >= nz:
            iz = nz - 1
        key = (ix * ny + iy) * nz + iz
        while a < b:
            mid = (a + b) // 2
            if keys[mid] < key:
                a = mid + 1
            else:
                b = mid
        if a < offsets[lv + 1] and keys[a] == key:
            return vals[a], lv
    return -1.0, np.int64(-1)


@njit(cache=True, inline='always')
def _lvl_lookup(tx, ty, tz, lv, dims, aoffsets, akeys, avals):
    """Level-consistent lookup at fractional domain coords: the value of the
    LEVEL-`lv` VIEW of the field - the leaf value if the containing level-lv
    cell is a leaf, the mass-conserving RESTRICTED average if that region is
    refined finer, or the covering coarser leaf's value if it is covered
    coarser (walk lv..0 over the augmented blocks, first hit wins). This is
    what the trilinear stencil nodes must read: point-sampling a much finer
    leaf from inside a coarse cell stretches a point value across the whole
    coarse width (the 2026-07-11 galaxy z-sheet bleed, 3+ decades).
    Returns (value, found)."""
    for l in range(lv, -1, -1):
        a = aoffsets[l]
        b = aoffsets[l + 1]
        if a == b:
            continue
        nx, ny, nz = dims[l, 0], dims[l, 1], dims[l, 2]
        ix = int(tx * nx)
        iy = int(ty * ny)
        iz = int(tz * nz)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= ny:
            iy = ny - 1
        if iz < 0:
            iz = 0
        elif iz >= nz:
            iz = nz - 1
        key = (ix * ny + iy) * nz + iz
        while a < b:
            mid = (a + b) // 2
            if akeys[mid] < key:
                a = mid + 1
            else:
                b = mid
        if a < aoffsets[l + 1] and akeys[a] == key:
            return avals[a], True
    return -1.0, False


@njit(cache=True, inline='always')
def _lvl_neighbour(ii, jj, kk, nx, ny, nz, periodic, lv, dims, aoffsets,
                   akeys, avals):
    """Level-consistent value of the cell (ii,jj,kk) at level lv, for the
    'limited' readout's axis neighbours. Periodic wraps; a non-periodic
    out-of-domain index returns (0., False) - the caller falls back to the
    centre value (zero-gradient at the domain edge)."""
    if periodic:
        ii %= nx
        jj %= ny
        kk %= nz
    elif ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
        return 0.0, False
    return _lvl_lookup((ii + 0.5) / nx, (jj + 0.5) / ny, (kk + 0.5) / nz,
                       lv, dims, aoffsets, akeys, avals)


@njit(cache=True, inline='always')
def _amr_readout(tx, ty, tz, periodic, dims, offsets, keys, vals, order,
                 interp, aoffsets, akeys, avals):
    """Single-point AMR readout at fractional domain coords (see _amr_query
    for the semantics). Returns (value, flat_value, level); level -1 = miss.
    flat_value is the containing leaf's own cell average - the occupancy
    decision must use THAT (a linear readout inside a truly-empty coarse
    cell ramps toward a dense neighbour's average and fabricates target
    density over the whole coarse width - the 2026-07-11 galaxy z-sheet).
    Shared by the plain table query and the fused padded-target query
    (envelope._padded_amr_query). Stencil nodes are read LEVEL-CONSISTENTLY
    (:func:`_lvl_lookup` on the augmented restricted blocks aoffsets/akeys/
    avals), never point-sampled from finer leaves.

    interp code: 0 = flat (containing leaf's average), 1 = linear
    (trilinear at the containing level), 2 = limited (MUSCL: cell average
    + per-axis minmod-limited slope from the 6 face neighbours, result
    clamped to the neighbour range - linear-exact in smooth regions,
    flat at sharp jumps, never ramps above a bright neighbour: the cheap
    monotone option of plan task M13)."""
    v, lv = _pwc_lookup(tx, ty, tz, dims, offsets, keys, vals, order)
    if lv < 0 or interp == 0:
        return v, v, lv
    nx, ny, nz = dims[lv, 0], dims[lv, 1], dims[lv, 2]
    if interp == 2:
        fx = tx * nx
        fy = ty * ny
        fz = tz * nz
        ix = min(max(int(fx), 0), nx - 1)
        iy = min(max(int(fy), 0), ny - 1)
        iz = min(max(int(fz), 0), nz - 1)
        acc = v
        vlo = v
        vhi = v
        for a in range(3):
            if a == 0:
                vm, okm = _lvl_neighbour(ix - 1, iy, iz, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                vp, okp = _lvl_neighbour(ix + 1, iy, iz, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                off = fx - ix - 0.5
            elif a == 1:
                vm, okm = _lvl_neighbour(ix, iy - 1, iz, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                vp, okp = _lvl_neighbour(ix, iy + 1, iz, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                off = fy - iy - 0.5
            else:
                vm, okm = _lvl_neighbour(ix, iy, iz - 1, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                vp, okp = _lvl_neighbour(ix, iy, iz + 1, nx, ny, nz,
                                         periodic, lv, dims, aoffsets,
                                         akeys, avals)
                off = fz - iz - 0.5
            if not okm:
                vm = v
            if not okp:
                vp = v
            d1 = v - vm
            d2 = vp - v
            if d1 * d2 > 0.0:               # minmod: 0 at local extrema
                s = d1 if abs(d1) < abs(d2) else d2
                acc += s * off
            if vm < vlo:
                vlo = vm
            elif vm > vhi:
                vhi = vm
            if vp < vlo:
                vlo = vp
            elif vp > vhi:
                vhi = vp
        if acc < vlo:                       # corner sums stay in the
            acc = vlo                       # neighbour range (positivity)
        elif acc > vhi:
            acc = vhi
        return acc, v, lv
    fx = tx * nx - 0.5
    fy = ty * ny - 0.5
    fz = tz * nz - 0.5
    ix0 = np.floor(fx)
    iy0 = np.floor(fy)
    iz0 = np.floor(fz)
    wx1 = fx - ix0
    wy1 = fy - iy0
    wz1 = fz - iz0
    acc = 0.0
    for cx in range(2):
        ntx = (ix0 + cx + 0.5) / nx
        wx = wx1 if cx == 1 else 1.0 - wx1
        if periodic:
            ntx -= np.floor(ntx)
        for cy in range(2):
            nty = (iy0 + cy + 0.5) / ny
            wy = wy1 if cy == 1 else 1.0 - wy1
            if periodic:
                nty -= np.floor(nty)
            for cz in range(2):
                ntz = (iz0 + cz + 0.5) / nz
                wz = wz1 if cz == 1 else 1.0 - wz1
                if periodic:
                    ntz -= np.floor(ntz)
                nv, ok = _lvl_lookup(ntx, nty, ntz, lv, dims, aoffsets,
                                     akeys, avals)
                if not ok:
                    nv = v      # off-domain node (open box): clamp
                acc += wx * wy * wz * nv
    return acc, v, lv


@njit(cache=True, parallel=True)
def _amr_query(pos, lo, box, periodic, dims, offsets, keys, vals, order,
               interp, aoffsets, akeys, avals, out):
    """AMR table readout. interp=0 (flat): the containing leaf's value
    (flat within a cell). interp=1 (linear): trilinear blend at the
    containing leaf's resolution - the 8 stencil nodes are the cell centres
    of the containing level's virtual uniform grid, each read
    LEVEL-CONSISTENTLY: the level's own leaf, the restricted average where
    the node's region is refined finer, or the covering coarser leaf
    (2026-07-11 fix - point-sampling a finer leaf from a coarse cell
    smeared refined disk densities decades above the coarse cell's true
    average across its whole width: the galaxy z-sheet artifact). Inside a
    uniform-level region this is exactly trilinear-on-cell-centres and
    unchanged; across level jumps it interpolates between same-scale
    averages. interp=2 (limited): MUSCL cell-average + minmod slopes (see
    _amr_readout). Returns the number of misses."""
    n = pos.shape[0]
    nmiss = 0
    for p in prange(n):
        tx = (pos[p, 0] - lo[0]) / box[0]
        ty = (pos[p, 1] - lo[1]) / box[1]
        tz = (pos[p, 2] - lo[2]) / box[2]
        if periodic:
            tx -= np.floor(tx)
            ty -= np.floor(ty)
            tz -= np.floor(tz)
        v, _, lv = _amr_readout(tx, ty, tz, periodic, dims, offsets, keys,
                                vals, order, interp, aoffsets, akeys, avals)
        if lv < 0:
            out[p] = -1.0
            nmiss += 1
            continue
        out[p] = v
    return nmiss


@njit(cache=True, parallel=True)
def _pwc_level_batch(pos, lo, box, periodic, dims, offsets, keys, vals,
                     order, out_lv):
    """Containing-leaf LEVEL per query point; -1 where the point lies
    outside an open (non-periodic) domain (the clamped _pwc_lookup would
    otherwise report the boundary cell). Used by the 'vertex' corner build
    to find the coarsest leaf touching each vertex."""
    n = pos.shape[0]
    for p in prange(n):
        tx = (pos[p, 0] - lo[0]) / box[0]
        ty = (pos[p, 1] - lo[1]) / box[1]
        tz = (pos[p, 2] - lo[2]) / box[2]
        if periodic:
            tx -= np.floor(tx)
            ty -= np.floor(ty)
            tz -= np.floor(tz)
        elif (tx < 0.0 or tx >= 1.0 or ty < 0.0 or ty >= 1.0
              or tz < 0.0 or tz >= 1.0):
            out_lv[p] = -1
            continue
        _, lv = _pwc_lookup(tx, ty, tz, dims, offsets, keys, vals, order)
        out_lv[p] = lv


@njit(cache=True, inline='always')
def _pwc_index(tx, ty, tz, dims, offsets, keys, order):
    """Like _pwc_lookup but returns (global slot index, level); the slot
    indexes the leaf-block arrays (vals / corner table rows)."""
    nlev = dims.shape[0]
    for oi in range(nlev):
        lv = order[oi]
        a = offsets[lv]
        b = offsets[lv + 1]
        if a == b:
            continue
        nx, ny, nz = dims[lv, 0], dims[lv, 1], dims[lv, 2]
        ix = min(max(int(tx * nx), 0), nx - 1)
        iy = min(max(int(ty * ny), 0), ny - 1)
        iz = min(max(int(tz * nz), 0), nz - 1)
        key = (ix * ny + iy) * nz + iz
        while a < b:
            mid = (a + b) // 2
            if keys[mid] < key:
                a = mid + 1
            else:
                b = mid
        if a < offsets[lv + 1] and keys[a] == key:
            return a, lv
    return np.int64(-1), np.int64(-1)


@njit(cache=True, inline='always')
def _corner_tri(tx, ty, tz, idx, lv, dims, corners):
    """Trilinear blend of leaf `idx`'s 8 corner values at fractional domain
    coords inside it (corner k = a + 2b + 4c). The 'vertex' readout's
    per-point evaluation, shared by _vertex_query and the fused padded
    kernels."""
    nx, ny, nz = dims[lv, 0], dims[lv, 1], dims[lv, 2]
    fx = tx * nx
    fy = ty * ny
    fz = tz * nz
    ux = min(max(fx - int(min(max(int(fx), 0), nx - 1)), 0.0), 1.0)
    uy = min(max(fy - int(min(max(int(fy), 0), ny - 1)), 0.0), 1.0)
    uz = min(max(fz - int(min(max(int(fz), 0), nz - 1)), 0.0), 1.0)
    c = corners[idx]
    return ((1 - ux) * (1 - uy) * (1 - uz) * c[0]
            + ux * (1 - uy) * (1 - uz) * c[1]
            + (1 - ux) * uy * (1 - uz) * c[2]
            + ux * uy * (1 - uz) * c[3]
            + (1 - ux) * (1 - uy) * uz * c[4]
            + ux * (1 - uy) * uz * c[5]
            + (1 - ux) * uy * uz * c[6]
            + ux * uy * uz * c[7])


@njit(cache=True, parallel=True)
def _vertex_query(pos, lo, box, periodic, dims, offsets, keys, order,
                  corners, out):
    """Conforming corner-trilinear readout ('vertex', plan M13 route a):
    locate the containing leaf, trilinear-blend its 8 CORNER values (corner
    k = a + 2b + 4c at offsets (a,b,c)) at the local cell coordinate.
    Corner values are built once with hanging vertices SLAVED to the coarse
    face's own interpolant, so both sides of every face restrict to the
    same face polynomial -> continuous across level jumps by construction.
    Returns the number of misses."""
    n = pos.shape[0]
    nmiss = 0
    for p in prange(n):
        tx = (pos[p, 0] - lo[0]) / box[0]
        ty = (pos[p, 1] - lo[1]) / box[1]
        tz = (pos[p, 2] - lo[2]) / box[2]
        if periodic:
            tx -= np.floor(tx)
            ty -= np.floor(ty)
            tz -= np.floor(tz)
        idx, lv = _pwc_index(tx, ty, tz, dims, offsets, keys, order)
        if idx < 0:
            out[p] = -1.0
            nmiss += 1
            continue
        out[p] = _corner_tri(tx, ty, tz, idx, lv, dims, corners)
    return nmiss


def _restricted_blocks(dims, offsets, keys, vals):
    """Augmented per-level 'level view' blocks for level-consistent stencil
    lookups (:func:`_lvl_lookup`): each level's sorted block = the level's
    LEAVES plus the mass-conserving RESTRICTED average of every level cell
    that is refined finer (parent = mean of its 8 children, bottom-up; the
    space-filling leaf set guarantees complete sibling groups, so the mean
    is the exact volume-weighted average). Cells covered by a COARSER leaf
    have no entry - the lookup falls through to the covering leaf. Returns
    (aoffsets, akeys, avals)."""
    nlev = dims.shape[0]
    akeys = [None] * nlev
    avals = [None] * nlev
    ik = np.empty(0, dtype=np.int64)         # internal cells at this level
    iv = np.empty(0, dtype=np.float64)
    for l in range(nlev - 1, -1, -1):
        ck = np.concatenate([keys[offsets[l]:offsets[l + 1]], ik])
        cv = np.concatenate([vals[offsets[l]:offsets[l + 1]], iv])
        srt = np.argsort(ck, kind='stable')
        akeys[l] = np.ascontiguousarray(ck[srt])
        avals[l] = np.ascontiguousarray(cv[srt])
        if l == 0:
            break
        ny, nz = int(dims[l, 1]), int(dims[l, 2])
        iz = ck % nz
        iy = (ck // nz) % ny
        ix = ck // (ny * nz)
        pny, pnz = int(dims[l - 1, 1]), int(dims[l - 1, 2])
        pk = ((ix >> 1) * pny + (iy >> 1)) * pnz + (iz >> 1)
        upk, inv = np.unique(pk, return_inverse=True)
        pv = np.zeros(len(upk))
        np.add.at(pv, inv, cv)
        ik, iv = upk, pv / 8.0
    aoffsets = np.zeros(nlev + 1, dtype=np.int64)
    aoffsets[1:] = np.cumsum([len(a) for a in akeys])
    return (aoffsets, np.ascontiguousarray(np.concatenate(akeys)),
            np.ascontiguousarray(np.concatenate(avals)))


class AMRTable(DensityField):
    """Relax-ready density table on an AMR leaf set.

    Storage: per-level blocks of sorted int64 linear cell keys + the leaf
    density values (petkova capture -> exact cell averages), plus the
    domain (lo, box) and per-level grid dims. A point query costs one
    binary search per populated level (levels tried in descending
    leaf-count order). Readout `interp`:
      'linear' (default) - trilinear blend at the containing leaf's
          resolution, stencil nodes read LEVEL-CONSISTENTLY (leaf,
          restricted average of finer refinement, or covering coarser
          leaf - never a point sample of a finer leaf): continuous inside
          uniform-level regions, no zeroth-order steps fed to the relax
          force at cell faces, no level-jump bleed (2026-07-11).
      'flat'   - the containing leaf's value (true piecewise-constant).
      'limited' - MUSCL slope-limited (2026-07-13, plan M13): cell average
          + per-axis minmod slope from the 6 face neighbours (level-
          consistent), clamped to the neighbour range. Exact on smooth
          linear fields like 'linear', but NEVER ramps toward a bright
          neighbour inside a faint cell (the graded z-line sawtooth lobes)
          - flat-like at sharp jumps. Still discontinuous at level faces
          (the basis switch; true continuity = 'vertex').
      'vertex' - conforming corner-trilinear (2026-07-13, plan M13 route
          a): the cell field is promoted ONCE to per-leaf CORNER values
          (mean of the leaves sharing each vertex), hanging vertices (fine
          corners on a coarser face/edge) SLAVED to the coarse cell's own
          interpolant at that position - both sides of every face restrict
          to the same face polynomial, so the readout is CONTINUOUS
          everywhere, including across level jumps. Not monotone (a
          corner mean next to a bright cell lifts the faint side ~1 cell;
          combine judgment with 'limited' if lobes matter more than
          continuity). Corner table built lazily on first query.
    Built from an sph_interp AMRGrid via `from_grid`; round-trips through
    .npz with save()/load() so a table can be captured once and reused
    without sph_interp installed."""

    _INTERP_CODE = {'flat': 0, 'linear': 1, 'limited': 2, 'vertex': 3}

    def __init__(self, lo, box, dims, offsets, keys, vals, order, periodic,
                 interp='linear'):
        self.lo = np.ascontiguousarray(lo, dtype=np.float64)
        self.box = np.ascontiguousarray(box, dtype=np.float64)
        self.dims = np.ascontiguousarray(dims, dtype=np.int64)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self.keys = np.ascontiguousarray(keys, dtype=np.int64)
        self.vals = np.ascontiguousarray(vals, dtype=np.float64)
        self.order = np.ascontiguousarray(order, dtype=np.int64)
        self.periodic = bool(periodic)
        if interp == 'linear':
            # RETIRED 2026-07-13 (user): trilinear overshoots across level
            # jumps (detached lobes 1-3 decades above truth on thin-disk
            # flanks - the bumps_flat_vs_vertex probe); 'limited' is the
            # same stencil with minmod slopes (== linear in smooth regions,
            # flat at jumps). The trilinear code stays as the internal
            # substrate (limited/vertex build on it; diagnostics may set
            # .interp='linear' directly).
            import warnings
            warnings.warn("interp='linear' is retired (level-jump overshoot"
                          " lobes); using 'limited'", stacklevel=2)
            interp = 'limited'
        if interp not in self._INTERP_CODE:
            raise ValueError(f"unknown interp {interp!r} (use 'flat', "
                             "'limited' or 'vertex')")
        self.interp = interp
        self._corners_cache = None      # lazy 'vertex' corner table
        # level-view blocks for level-consistent stencil nodes (recomputed on
        # every construction - cheap, and _with_values shell tables get their
        # own restriction of the replaced values automatically)
        self.aoffsets, self.akeys, self.avals = _restricted_blocks(
            self.dims, self.offsets, self.keys, self.vals)

    @classmethod
    def from_grid(cls, amr, values, interp='limited'):
        """Index an sph_interp AMRGrid + per-leaf values for point queries.
        Requires the space-filling leaf set AMRGrid.build produces by
        default (empty_max_level=None) - every point must be in a leaf."""
        values = np.asarray(values, dtype=np.float64)
        if len(values) != amr.ncell:
            raise ValueError(f'{len(values)} values for {amr.ncell} leaves')
        lo = np.asarray(amr.lo, dtype=np.float64)
        box = np.asarray(amr.hi, dtype=np.float64) - lo
        lvl = np.asarray(amr.cell_level, dtype=np.int64)
        nlev = int(lvl.max()) + 1
        base = np.asarray(amr.base_npx, dtype=np.int64)
        dims = base[None, :] * (2 ** np.arange(nlev, dtype=np.int64))[:, None]
        # integer coords of each leaf at ITS level (centre / cell width:
        # lands mid-cell, immune to edge rounding)
        centre = amr.cell_lo + 0.5 * amr.cell_size
        nl = dims[lvl]
        ijk = np.floor((centre - lo) / (box / nl)).astype(np.int64)
        ijk = np.clip(ijk, 0, nl - 1)
        keys = (ijk[:, 0] * nl[:, 1] + ijk[:, 1]) * nl[:, 2] + ijk[:, 2]
        srt = np.lexsort((keys, lvl))           # by level, then key
        counts = np.bincount(lvl, minlength=nlev)
        offsets = np.zeros(nlev + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)
        order = np.argsort(-counts, kind='stable')
        return cls(lo, box, dims, offsets, keys[srt], values[srt], order,
                   all(amr.periodic), interp=interp)

    @property
    def ncell(self):
        return len(self.keys)

    def leaves(self):
        """Decode the leaf set back to geometry for inspection/export:
        returns (centres (N,3), sizes (N,3) per-axis cell widths,
        levels (N,), values (N,)) in key-block order (aligned with
        self.vals)."""
        n = self.ncell
        centres = np.empty((n, 3))
        sizes = np.empty((n, 3))
        levels = np.empty(n, dtype=np.int64)
        for lv in range(self.dims.shape[0]):
            a, b = self.offsets[lv], self.offsets[lv + 1]
            if a == b:
                continue
            nx, ny, nz = self.dims[lv]
            k = self.keys[a:b]
            iz = k % nz
            iy = (k // nz) % ny
            ix = k // (nz * ny)
            cw = self.box / self.dims[lv]
            centres[a:b] = self.lo + (np.stack([ix, iy, iz], 1) + 0.5) * cw
            sizes[a:b] = cw
            levels[a:b] = lv
        return centres, sizes, levels, self.vals

    def _corner_table(self):
        """(ncell, 8) conforming corner values for the 'vertex' readout
        (corner k = a + 2b + 4c at unit offsets (a,b,c)), built lazily:
        every vertex gets the mean of the leaves sharing it as a corner,
        then HANGING vertices (not representable on the coarsest touching
        leaf's lattice) are slaved, coarsest slaving level first, to that
        leaf's own trilinear at the vertex position - the FEM hanging-node
        constraint that makes the per-leaf corner-trilinear readout
        continuous across level jumps (2:1 grading not required, any
        space-filling leaf set works)."""
        if self._corners_cache is not None:
            return self._corners_cache
        nlev = self.dims.shape[0]
        centres, sizes, _, vals = self.leaves()
        fdims = self.dims[nlev - 1]
        cw = self.box / fdims                     # finest cell width
        offs = np.array([[k & 1, (k >> 1) & 1, (k >> 2) & 1]
                         for k in range(8)], dtype=np.float64)
        cpos = (centres[:, None, :] - 0.5 * sizes[:, None, :]
                + sizes[:, None, :] * offs[None])
        Xi = np.rint((cpos - self.lo) / cw).astype(np.int64)   # (n,8,3)
        if self.periodic:
            Xi %= fdims[None, None, :]            # identify seam vertices
            vny, vnz = int(fdims[1]), int(fdims[2])
        else:
            vny, vnz = int(fdims[1]) + 1, int(fdims[2]) + 1
        vid = (Xi[..., 0] * vny + Xi[..., 1]) * vnz + Xi[..., 2]
        uvid, inv = np.unique(vid.reshape(-1), return_inverse=True)
        vval = (np.bincount(inv, weights=np.repeat(vals, 8))
                / np.bincount(inv))
        # coarsest touching leaf per vertex: probe the 8 surrounding octants
        Z = uvid % vnz
        Y = (uvid // vnz) % vny
        X = uvid // (vny * vnz)
        vpos = self.lo + np.stack([X, Y, Z], 1) * cw
        probes = (vpos[:, None, :]
                  + 0.25 * cw * (2.0 * offs - 1.0)[None]).reshape(-1, 3)
        lvq = np.empty(len(probes), dtype=np.int64)
        _pwc_level_batch(np.ascontiguousarray(probes), self.lo, self.box,
                         self.periodic, self.dims, self.offsets, self.keys,
                         self.vals, self.order, lvq)
        lvq8 = np.where(lvq < 0, nlev, lvq).reshape(-1, 8)
        lmin = np.minimum(lvq8.min(1), nlev - 1)
        oct_of_min = lvq8.argmin(1)
        step = np.left_shift(np.int64(1), nlev - 1 - lmin)
        hang = ~((X % step == 0) & (Y % step == 0) & (Z % step == 0))
        for lc in np.unique(lmin[hang]):
            sel = np.where(hang & (lmin == lc))[0]
            st = np.int64(1) << np.int64(nlev - 1 - int(lc))
            nl = self.dims[int(lc)]
            # the slaving cell C = the leaf the lmin-achieving octant probe
            # fell in (a point strictly INSIDE C): flooring the vertex's own
            # coords instead picks the wrong/out-of-grid cell when v lies on
            # C's UPPER face along a divisible axis
            ppos = (vpos[sel]
                    + 0.25 * cw * (2.0 * offs[oct_of_min[sel]] - 1.0))
            frac = (ppos - self.lo) / self.box
            if self.periodic:
                frac -= np.floor(frac)
            cix = np.clip((frac * nl).astype(np.int64), 0, nl - 1)
            cx0, cy0, cz0 = cix[:, 0] * st, cix[:, 1] * st, cix[:, 2] * st
            dX, dY, dZ = X[sel] - cx0, Y[sel] - cy0, Z[sel] - cz0
            if self.periodic:                    # seam: v wrapped, C not
                dX %= int(fdims[0])
                dY %= int(fdims[1])
                dZ %= int(fdims[2])
            tx, ty, tz = dX / st, dY / st, dZ / st
            acc = np.zeros(len(sel))
            for k in range(8):
                a, b, c = k & 1, (k >> 1) & 1, (k >> 2) & 1
                gx, gy, gz = cx0 + a * st, cy0 + b * st, cz0 + c * st
                if self.periodic:
                    gx, gy, gz = (gx % int(fdims[0]), gy % vny, gz % vnz)
                cvid = (gx * vny + gy) * vnz + gz
                j = np.minimum(np.searchsorted(uvid, cvid), len(uvid) - 1)
                assert (uvid[j] == cvid).all(), \
                    'slaving-cell corner not in the vertex set'
                w = ((tx if a else 1.0 - tx) * (ty if b else 1.0 - ty)
                     * (tz if c else 1.0 - tz))
                acc += w * vval[j]
            vval[sel] = acc
        self._corners_cache = np.ascontiguousarray(vval[inv].reshape(-1, 8))
        return self._corners_cache

    def rho0(self, pos):
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        out = np.empty(len(pos))
        if self.interp == 'vertex':
            nmiss = _vertex_query(pos, self.lo, self.box, self.periodic,
                                  self.dims, self.offsets, self.keys,
                                  self.order, self._corner_table(), out)
        else:
            nmiss = _amr_query(pos, self.lo, self.box, self.periodic,
                               self.dims, self.offsets, self.keys,
                               self.vals, self.order,
                               self._INTERP_CODE[self.interp],
                               self.aoffsets, self.akeys, self.avals, out)
        if nmiss:
            raise ValueError(f'{nmiss}/{len(pos)} query points not covered '
                             'by the AMR table (leaf set not space-filling?)')
        return out

    __call__ = rho0

    def save(self, path):
        np.savez(path, lo=self.lo, box=self.box, dims=self.dims,
                 offsets=self.offsets, keys=self.keys, vals=self.vals,
                 order=self.order, periodic=np.array([self.periodic]),
                 interp=np.array([self.interp]))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        interp = str(d['interp'][0]) if 'interp' in d.files else 'limited'
        if interp == 'linear':                # pre-retirement save:
            interp = 'limited'                # remap quietly, no warning
        return cls(d['lo'], d['box'], d['dims'], d['offsets'], d['keys'],
                   d['vals'], d['order'], bool(d['periodic'][0]),
                   interp=interp)
