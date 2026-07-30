"""Target geometries for interpolation.

A *target* defines the cells that particle quantities are deposited onto. The
interpolation *method* (regular SPH or Petkova exact) is orthogonal: any method
can in principle deposit onto any target, because a cell is a polyhedron either
way.

Implemented: `UniformGrid` (3D regular cubic cells), `Projection2D` (2D pixel
grid), `AMRGrid` (octree-refined cubic cells over a base grid), and `VoronoiGrid`
(unstructured polyhedral cells, the native Petkova case).
"""

from dataclasses import dataclass
import importlib.util
import os
import contextlib
import subprocess
import tempfile
import numpy as np
import numba

from .domain import kernel_support_bounds

try:                                  # optional fast parallel mesher (OpenMP voro++)
    import multivoro as _multivoro
    _HAS_MULTIVORO = True
except Exception:                     # pragma: no cover - optional dependency
    _multivoro = None
    _HAS_MULTIVORO = False

# optional AREPO Voronoi mesher (codes/arepo) — a shell-out (write HDF5 snapshot
# -> run `Arepo param.txt 14` mesh-dump -> parse), in _arepo/arepo_mesh.py. AREPO is
# the reference moving-mesh code; it tessellates collapsed high-dynamic-range cores
# robustly (its native regime) where multivoro's parallel solver under-cuts. It is
# the preferred backend when built (see `arepo-mesh-backend`), so `auto` picks it
# whenever codes/arepo/AREPO/Arepo is present, falling back to multivoro then
# pyvoro.
try:
    from ._arepo import arepo_mesh as _arepo_mesh
    _HAS_AREPO = _arepo_mesh.has_arepo()
except Exception:                     # pragma: no cover - optional dependency
    _arepo_mesh = None
    _HAS_AREPO = False

_DEFAULT_THREADS = os.cpu_count() or 1

# De-sliver floor as a fraction of the median nearest-neighbour spacing, PER
# MESHER. multivoro (parallel voro++) needs ~4x pyvoro's floor on a collapsed,
# high-dynamic-range core: its independent per-cell search loses completeness when
# the local spacing collapses abruptly, so cells are UNDER-CUT and OVERLAP (mass
# error +2260% / 100% core overlap at frac 0.05, vs pyvoro's graceful +0.21%).
# A larger floor pulls the core dynamic range back into the regime multivoro can
# resolve. Serial pyvoro keeps voro++'s completeness guarantee, so 0.05 (just
# enough to clear true slivers) conserves. See `multivoro-dynamic-range-defect`.
# arepo at 0.0: AREPO's Delaunay/Voronoi handles the collapsed high-dynamic-range
# core natively (its native regime) -- no de-sliver needed; the ConvexHull
# reconstruction is exact on the resulting cells.
_DESLIVER_FRAC = {"pyvoro": 0.05, "multivoro": 0.20, "arepo": 0.0}

# Relative noise tolerance on the periodic volume budget |Sum(cell vol)/box - 1|.
# The signal it must separate: a genuinely insufficient image shell inflates the
# budget by >= ~1% (+5% sparse N=32, +7.6% clustered 1500 gens), while mesher
# float noise in the face-computed volumes measures +2e-16 (multivoro, 40k
# collapsed merger core) to ~5e-12 (synthetic clustered) with rare run-dependent
# excursions to ~1e-6 -- four decades of separation, so 1e-5 splits them safely.
# The check is also ONE-SIDED: an unclosed boundary cell can only ADD volume
# (it bloats into the pad instead of being cut by its missing image), so only a
# POSITIVE excess is ever fixable by re-tessellating at a wider shell; a deficit
# is mesher volume error and re-building is a pure waste (~10 s per doubling at
# 40k generators).
_VOL_BUDGET_TOL = 1e-5

# Refinement budget for a periodic mesh whose cells are too big to unwrap (see
# `_periodic_cell_rad_limit`). One round is normally enough -- the limit is a
# geometric bound, not an iteration -- so the rounds only cover refinement
# opening up a neighbouring cell. The growth cap is what keeps a very thin box
# from silently turning a 1500-cell request into a 100k-cell mesh; past it the
# ghost-image path takes over and meshes the original generators exactly.
_REFINE_ROUNDS = 3
_REFINE_MAX_GROWTH = 8.0


def _resolve_backend(backend):
    """Map the requested backend to the effective mesher. 'auto' prefers AREPO when
    it is built (codes/arepo/AREPO/Arepo present) -- it tessellates collapsed cores
    robustly -- then multivoro if installed, else pyvoro."""
    if backend in ("arepo", "multivoro", "pyvoro"):
        return backend
    # auto
    if _HAS_AREPO:
        return "arepo"
    if _HAS_MULTIVORO:
        return "multivoro"
    return "pyvoro"


def _backend_availability():
    """[(name, available, why), ...] for every mesher, in the SAME preference order
    `auto` walks (arepo -> multivoro -> pyvoro). Lets a CLI report which backends
    `auto` skipped and why, instead of silently landing on the last rung."""
    if _arepo_mesh is None:
        arepo_why = "sph_interp._arepo did not import"
    elif _HAS_AREPO:
        arepo_why = f"built at {_arepo_mesh._AREPO_EXE}"
    else:
        arepo_why = f"not built ({_arepo_mesh._AREPO_EXE})"
    # find_spec, not import: pyvoro is imported lazily in `_tessellate` and we do
    # not want to pay for it just to report availability.
    has_pyvoro = importlib.util.find_spec("pyvoro") is not None
    return [
        ("arepo", _HAS_AREPO, arepo_why),
        ("multivoro", _HAS_MULTIVORO,
         "installed" if _HAS_MULTIVORO else "not installed"),
        ("pyvoro", has_pyvoro, "installed" if has_pyvoro else "not installed"),
    ]


def _resolve_threads(n_threads):
    """Resolve the requested worker count: <=0 (or falsy) -> all cores, else the
    given count. Used by the multivoro OpenMP backend (arepo uses `_arepo_ranks`,
    which defaults to serial; see `VoronoiGrid._from_arepo`)."""
    return int(n_threads) if n_threads and n_threads > 0 else _DEFAULT_THREADS


# AREPO's domain decomposition aborts when a rank gets too few generators
# (empirically ~625/rank meshes, ~312/rank aborts), so keep a conservative floor.
_AREPO_MIN_PER_RANK = 2000


def _arepo_ranks(n_threads, npts):
    """MPI rank count for an AREPO mesh dump. Default (n_threads<=0) is SERIAL:
    AREPO's MPI meshing anti-scales for this dump-only use (ghost-point import
    grows with rank count, the dump is written serially by rank 0, and mpirun
    launch costs seconds), peaking ~np=16 even at 800k generators -- while the
    Python cell reconstruction runs on a fork pool of all physical cores (see
    `_arepo/arepo_mesh._reconstruct_cells`). So ranks add overhead for little
    gain, and too many abort. An explicit n_threads>1 is honoured but capped so
    each rank keeps at least `_AREPO_MIN_PER_RANK` generators. The dumped mesh
    is identical either way (see the cell-id grouping in `_arepo/arepo_mesh.py`)."""
    if not n_threads or n_threads <= 1:
        return 1
    return max(1, min(int(n_threads), npts // _AREPO_MIN_PER_RANK))


@numba.njit(cache=True)
def _mv_flatten(VV, vstart, FF, fstart, genx, geny, genz):
    """Build the Voronoi CSR (cf_start, fv_start, fvx/fvy/fvz, volume, cell_rad)
    from multivoro's concatenated per-cell vertices (VV, offsets vstart) and flat
    face-vertex runs (FF = [k, local_ids..k, ...] per cell, offsets fstart). Face
    vertex indices are LOCAL to the cell. Volume = signed tets fanned from the
    generator; degenerate faces (k<3) skipped. The hot inner loop, in numba."""
    N = vstart.shape[0] - 1
    nface = 0
    nfv = 0
    for c in range(N):                          # pass 1: count faces (k>=3) + verts
        i = fstart[c]; end = fstart[c + 1]
        while i < end:
            k = FF[i]; i += 1 + k
            if k >= 3:
                nface += 1; nfv += k
    cf_start = np.zeros(N + 1, np.int64)
    fv_start = np.zeros(nface + 1, np.int64)
    fvx = np.empty(nfv); fvy = np.empty(nfv); fvz = np.empty(nfv)
    volume = np.empty(N); cell_rad = np.empty(N)
    fidx = 0; vptr = 0
    for c in range(N):                          # pass 2: fill
        gx = genx[c]; gy = geny[c]; gz = genz[c]
        vs = vstart[c]; nv = vstart[c + 1] - vs
        rmax = 0.0
        for j in range(nv):
            dx = VV[vs + j, 0] - gx; dy = VV[vs + j, 1] - gy; dz = VV[vs + j, 2] - gz
            d = dx * dx + dy * dy + dz * dz
            if d > rmax:
                rmax = d
        cell_rad[c] = rmax ** 0.5
        vol6 = 0.0
        i = fstart[c]; end = fstart[c + 1]
        while i < end:
            k = FF[i]; base = i + 1; i += 1 + k
            if k < 3:
                continue
            for t in range(k):
                gi = vs + FF[base + t]
                fvx[vptr] = VV[gi, 0]; fvy[vptr] = VV[gi, 1]; fvz[vptr] = VV[gi, 2]
                vptr += 1
            fidx += 1
            fv_start[fidx] = vptr
            i0 = vs + FF[base]
            a0x = VV[i0, 0] - gx; a0y = VV[i0, 1] - gy; a0z = VV[i0, 2] - gz
            for t in range(1, k - 1):
                i1 = vs + FF[base + t]; i2 = vs + FF[base + t + 1]
                a1x = VV[i1, 0] - gx; a1y = VV[i1, 1] - gy; a1z = VV[i1, 2] - gz
                a2x = VV[i2, 0] - gx; a2y = VV[i2, 1] - gy; a2z = VV[i2, 2] - gz
                vol6 += (a0x * (a1y * a2z - a1z * a2y)
                         - a0y * (a1x * a2z - a1z * a2x)
                         + a0z * (a1x * a2y - a1y * a2x))
        volume[c] = abs(vol6) / 6.0
        cf_start[c + 1] = fidx
    return cf_start, fv_start, fvx, fvy, fvz, volume, cell_rad


@contextlib.contextmanager
def _silence_cstd():
    """Mute C-level stdout+stderr (voro++ prints 'Order 4 vertex memory scaled
    up...' progress noise per cell) for the duration of the block."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


@dataclass
class UniformGrid:
    """A regular 3D grid of cubic cells spanning [lo, hi] with `npx` cells/axis.

    lo, hi : (3,) lower/upper corners of the domain.
    npx    : int or (3,) number of cells per axis.

    Cell centres are lo + (i + 0.5) * pixwidth. Use `centered(box, npx)` to get a
    grid spanning [-box/2, +box/2] (the convention of the periodic test boxes).
    """
    lo: np.ndarray
    hi: np.ndarray
    npx: object = 64

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)
        npx = self.npx
        if np.isscalar(npx):
            npx = (int(npx), int(npx), int(npx))
        self.npx = tuple(int(x) for x in npx)
        if any(n <= 0 for n in self.npx):
            raise ValueError("npx must be positive")
        if np.any(self.hi <= self.lo):
            raise ValueError("hi must exceed lo on every axis")

    @classmethod
    def centered(cls, box, npx=64):
        """Grid spanning [-box/2, +box/2] for a box size (scalar or (3,))."""
        box = np.asarray(box, dtype=np.float64).reshape(-1)
        if box.size == 1:
            box = np.repeat(box, 3)
        half = 0.5 * box
        return cls(lo=-half, hi=half, npx=npx)

    @property
    def box(self):
        return self.hi - self.lo

    @property
    def pixwidth(self):
        return self.box / np.asarray(self.npx, dtype=np.float64)

    @property
    def cell_volume(self):
        pw = self.pixwidth
        return float(pw[0] * pw[1] * pw[2])

    def edges(self):
        """Per-axis cell edge coordinates: list of three (npx[a]+1,) arrays."""
        return [np.linspace(self.lo[a], self.hi[a], self.npx[a] + 1)
                for a in range(3)]

    def centres(self):
        """Per-axis cell centre coordinates: list of three (npx[a],) arrays."""
        pw = self.pixwidth
        return [self.lo[a] + (np.arange(self.npx[a]) + 0.5) * pw[a]
                for a in range(3)]


@dataclass
class AMRGrid:
    """Adaptive (octree-refined) cubic cells over a base grid.

    A forest of octrees: the domain [lo,hi] is tiled by a `base_npx` base grid
    (level 0); a refinement criterion recursively splits cells into 8 children
    (down to `max_level`) where the data warrants it. The result is a flat list
    of LEAF cells (true memory-efficient AMR — storage scales with the number of
    leaves, not base_npx**3 * 8**max_level). Build one with `AMRGrid.build`.

    Geometry arrays (one row per leaf, sorted by the leaf's base-grid cell):
      cell_lo    (Ncell,3)  lower corner of each leaf
      cell_size  (Ncell,3)  edge lengths of each leaf
      cell_level (Ncell,)   refinement level (0 = base)
      cell_base  (Ncell,)   flat base-grid index the leaf descends from
    Lookup for the deposit (which leaves live in base cell b):
      base_start (nbase,)   first leaf index of base cell b (leaves are sorted)
      base_count (nbase,)   number of leaves in base cell b
    Every method deposits by looping the base cells a particle's kernel overlaps
    (O(1) index arithmetic on the base grid), then that base cell's few leaves.
    """
    lo: np.ndarray
    hi: np.ndarray
    base_npx: object
    cell_lo: np.ndarray
    cell_size: np.ndarray
    cell_level: np.ndarray
    cell_base: np.ndarray
    base_start: np.ndarray
    base_count: np.ndarray
    periodic: object = (False, False, False)
    cell_count: np.ndarray = None        # (Ncell,) particles in each leaf (0 = empty)

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)
        nb = self.base_npx
        if np.isscalar(nb):
            nb = (int(nb), int(nb), int(nb))
        self.base_npx = tuple(int(x) for x in nb)
        p = self.periodic
        if np.isscalar(p):
            p = (bool(p), bool(p), bool(p))
        self.periodic = tuple(bool(x) for x in p)

    @property
    def ncell(self):
        return self.cell_lo.shape[0]

    @property
    def box(self):
        return self.hi - self.lo

    @property
    def base_pw(self):
        return (self.hi - self.lo) / np.asarray(self.base_npx, dtype=np.float64)

    def centres(self):
        """(Ncell,3) leaf-cell centres."""
        return self.cell_lo + 0.5 * self.cell_size

    def volumes(self):
        """(Ncell,) leaf-cell volumes."""
        return self.cell_size[:, 0] * self.cell_size[:, 1] * self.cell_size[:, 2]

    def edges(self):
        """(cell_lo, cell_hi): the lower/upper corners of every leaf."""
        return self.cell_lo, self.cell_lo + self.cell_size

    @classmethod
    def build(cls, particles, base_npx=16, max_level=4, criterion=None,
              lo=None, hi=None, periodic=None, empty_max_level=None,
              refine_empty=False):
        """Build an AMR octree over `particles` by refining a base grid.

        particles : Particles (uses pos, mass, h).
        base_npx  : base-grid cells per axis (int or (3,)); the level-0 tiling.
        max_level : maximum refinement depth (leaf size = base / 2**level).
        criterion : callable(count, mass, hmin, size, level) -> bool[Mcells],
                    True where a cell should be refined. Defaults to
                    `refine_smoothing_length()`. See `refine_mass` /
                    `refine_count`. Empty cells (count==0) are never refined.
        lo, hi    : domain corners. Default: the centered `particles.box` if
                    there is one, else `kernel_support_bounds(pos, h)` -- the
                    smallest box holding every 2h support, so an OPEN capture
                    never clips a kernel. A CLI wanting a density-aware pad
                    should resolve `open_domain_bounds` itself and pass it.
        periodic  : periodicity (default: particles.periodic).
        refine_empty : allow the criterion to refine EMPTY (count==0) cells.
                    Default False: empties are never split (count-style
                    criteria would not terminate on them). Set True for
                    self-terminating region criteria (e.g. a contour-band
                    correction pass targeting kernel-spill cells whose
                    deposited value is nonzero despite holding no particle
                    centres) - the criterion must bound its own depth.
        empty_max_level : deepest level at which an EMPTY (count==0) leaf is kept.
                    None (default) keeps empty leaves at every level, so the mesh
                    is space-filling -- required by RAMSES / structured exports.
                    Set 0 to keep empty cells only at the base (GridSize) level and
                    drop all finer empty siblings; an intermediate value grades the
                    void fill, approximating AREPO ADDBACKGROUNDGRID (which only
                    creates empty void nodes down to its top-domain-tree depth, not
                    through the resolved gas). Occupied leaves are always kept; the
                    dropped regions are absorbed by neighbours in a Voronoi deposit.
        """
        if criterion is None:
            criterion = refine_smoothing_length()
        pos = np.ascontiguousarray(particles.pos, dtype=np.float64)
        mass = np.ascontiguousarray(particles.mass, dtype=np.float64)
        h = np.ascontiguousarray(particles.h, dtype=np.float64)
        N = pos.shape[0]

        if lo is None or hi is None:
            if particles.box is not None:
                half = 0.5 * np.asarray(particles.box, dtype=np.float64).reshape(3)
                lo, hi = -half, half
            else:
                # OPEN capture: the mesh must contain every particle's 2h
                # support, not just its centre. The bare particle bounding box
                # clips half the support of every extremal particle -- silent
                # mass loss at the boundary, and the open-AMR Hodge solve
                # refuses the domain outright. `kernel_support_bounds` is the
                # package's own resolver for exactly this; never hand-roll a
                # percentage pad here (see `open_domain_bounds` for the
                # density-aware modes a CLI should offer instead).
                lo, hi = kernel_support_bounds(pos, h)
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if periodic is None:
            periodic = particles.periodic

        nb = base_npx
        if np.isscalar(nb):
            nb = (int(nb), int(nb), int(nb))
        nb = tuple(int(x) for x in nb)
        nbx, nby, nbz = nb
        nbase = nbx * nby * nbz
        bpw = (hi - lo) / np.array(nb, dtype=np.float64)

        # particle -> base cell (clip strays into the edge cells)
        ix = np.clip(((pos[:, 0] - lo[0]) / bpw[0]).astype(np.int64), 0, nbx - 1)
        iy = np.clip(((pos[:, 1] - lo[1]) / bpw[1]).astype(np.int64), 0, nby - 1)
        iz = np.clip(((pos[:, 2] - lo[2]) / bpw[2]).astype(np.int64), 0, nbz - 1)
        base_flat = (ix * nby + iy) * nbz + iz

        # active cells start as the full base grid (so the mesh tiles the domain)
        idx = np.arange(nbase)
        ac_lo = np.empty((nbase, 3))
        ac_lo[:, 0] = lo[0] + (idx // (nby * nbz)) * bpw[0]
        ac_lo[:, 1] = lo[1] + ((idx // nbz) % nby) * bpw[1]
        ac_lo[:, 2] = lo[2] + (idx % nbz) * bpw[2]
        ac_size = np.tile(bpw, (nbase, 1))
        ac_base = np.arange(nbase, dtype=np.int64)

        pidx = np.arange(N)              # particles still in play
        pcell = base_flat.copy()         # their active-cell index

        leaf_lo, leaf_size, leaf_level, leaf_base, leaf_count = [], [], [], [], []
        for level in range(max_level + 1):
            M = ac_lo.shape[0]
            if M == 0:
                break
            count = np.bincount(pcell, minlength=M).astype(np.int64)
            msum = np.bincount(pcell, weights=mass[pidx], minlength=M)
            hmn = np.full(M, np.inf)
            if pidx.size:
                np.minimum.at(hmn, pcell, h[pidx])
            if level == max_level:
                refine = np.zeros(M, dtype=bool)
            else:
                if getattr(criterion, 'needs_lo', False):
                    refine = np.asarray(criterion(count, msum, hmn, ac_size,
                                                  level, ac_lo), dtype=bool)
                else:
                    refine = np.asarray(criterion(count, msum, hmn, ac_size,
                                                  level), dtype=bool)
                if not refine_empty:
                    refine &= count > 0  # never refine empty cells
            leafm = ~refine
            if empty_max_level is not None and level > empty_max_level:
                # drop fine empty siblings; keep occupied leaves (bggrid-like).
                # Their space is absorbed by neighbours in a Voronoi deposit.
                leafm = leafm & (count > 0)
            if leafm.any():
                leaf_lo.append(ac_lo[leafm])
                leaf_size.append(ac_size[leafm])
                leaf_level.append(np.full(int(leafm.sum()), level, dtype=np.int64))
                leaf_base.append(ac_base[leafm])
                leaf_count.append(count[leafm])
            if not refine.any():
                break

            rc = np.where(refine)[0]
            nref = rc.shape[0]
            newsize = ac_size[rc] * 0.5
            child_lo = np.empty((nref * 8, 3))
            child_size = np.empty((nref * 8, 3))
            child_base = np.repeat(ac_base[rc], 8)
            for k in range(8):           # child k = ox + 2*oy + 4*oz
                ox, oy, oz = k & 1, (k >> 1) & 1, (k >> 2) & 1
                child_lo[k::8, 0] = ac_lo[rc, 0] + ox * newsize[:, 0]
                child_lo[k::8, 1] = ac_lo[rc, 1] + oy * newsize[:, 1]
                child_lo[k::8, 2] = ac_lo[rc, 2] + oz * newsize[:, 2]
                child_size[k::8] = newsize

            keepp = refine[pcell]        # particles whose cell was refined
            pidx2 = pidx[keepp]
            parent = pcell[keepp]
            rel = (pos[pidx2] - ac_lo[parent]) / ac_size[parent]
            oct_ = ((rel[:, 0] >= 0.5).astype(np.int64)
                    + 2 * (rel[:, 1] >= 0.5).astype(np.int64)
                    + 4 * (rel[:, 2] >= 0.5).astype(np.int64))
            parent_to_child = np.full(M, -1, dtype=np.int64)
            parent_to_child[rc] = np.arange(nref, dtype=np.int64) * 8
            pcell = parent_to_child[parent] + oct_
            ac_lo, ac_size, ac_base, pidx = child_lo, child_size, child_base, pidx2

        if leaf_lo:
            cell_lo = np.concatenate(leaf_lo)
            cell_size = np.concatenate(leaf_size)
            cell_level = np.concatenate(leaf_level)
            cell_base = np.concatenate(leaf_base)
            cell_count = np.concatenate(leaf_count)
        else:
            cell_lo = np.zeros((0, 3)); cell_size = np.zeros((0, 3))
            cell_level = np.zeros(0, dtype=np.int64); cell_base = np.zeros(0, dtype=np.int64)
            cell_count = np.zeros(0, dtype=np.int64)

        order = np.argsort(cell_base, kind="stable")   # group leaves by base cell
        cell_lo, cell_size = cell_lo[order], cell_size[order]
        cell_level, cell_base = cell_level[order], cell_base[order]
        cell_count = cell_count[order]
        base_count = np.bincount(cell_base, minlength=nbase).astype(np.int64)
        base_start = np.zeros(nbase, dtype=np.int64)
        base_start[1:] = np.cumsum(base_count)[:-1]

        return cls(lo=lo, hi=hi, base_npx=nb, cell_lo=cell_lo, cell_size=cell_size,
                   cell_level=cell_level, cell_base=cell_base,
                   base_start=base_start, base_count=base_count, periodic=periodic,
                   cell_count=cell_count)


# --------------------------------------------------------------------------- #
#  Refinement criteria for AMRGrid.build — pluggable, add more freely.
#  Each is criterion(count, mass, hmin, size, level) -> bool[Mcells]:
#    count (M,)   particles in the cell        mass  (M,)   summed particle mass
#    hmin  (M,)   min smoothing length in cell size  (M,3) cell edge lengths
#    level int    current refinement level
#  Return True where the cell should be split. Empty cells are filtered by the
#  builder, so a criterion may ignore the count==0 case.
# --------------------------------------------------------------------------- #

def refine_smoothing_length(factor=1.0):
    """Refine until a cell is no larger than `factor` * the local smoothing
    length, i.e. resolution tracks the SPH resolution (fine where h is small /
    particles are dense, coarse in voids). factor=1 -> cell <~ h; factor=2 ->
    cell <~ 2h (the kernel support)."""
    def crit(count, mass, hmin, size, level):
        return size.max(axis=1) > factor * hmin
    return crit


def refine_mass(threshold):
    """Refine while a cell's summed particle mass exceeds `threshold` (with
    equal-mass particles this is equivalent to a particles-per-cell cap of
    threshold / m_particle)."""
    def crit(count, mass, hmin, size, level):
        return mass > threshold
    return crit


def refine_count(n_target):
    """Refine while a cell holds more than `n_target` particles."""
    def crit(count, mass, hmin, size, level):
        return count > n_target
    return crit


def refine_band(rho_lo, rho_hi, m_out, beta=0.5):
    """Contour-resolving refinement: refine an occupied cell whose mean
    density (summed particle mass / volume) lies in [rho_lo, rho_hi] while
    the cell is larger than the local sampling scale ``beta *
    (m_out/rho_cell)**(1/3)`` (the inter-particle spacing of an output
    ensemble of per-particle mass ``m_out`` at that density). Aimed at a
    band around a classification contour (e.g. an envelope's rho_edge): a
    coarse cell straddling the contour averages both sides into one value
    and any one-bit gas/vacuum verdict on it is wrong over part of the cell
    - splitting it lets sub-cells answer separately, down to the resolution
    at which the verdict's consumer can act. Self-terminating: children off
    the contour leave the band, children on it shrink below their own
    spacing scale. Combine with a sampling criterion via `refine_any`."""
    def crit(count, mass, hmin, size, level):
        rho = mass / size.prod(axis=1)
        band = (rho >= rho_lo) & (rho <= rho_hi)
        with np.errstate(divide='ignore'):
            floor = beta * np.cbrt(m_out / np.maximum(rho, 1e-300))
        return band & (size.max(axis=1) > floor)
    return crit


def refine_any(*crits):
    """OR-combine refinement criteria (refine where any one fires).
    Propagates the `needs_lo` marker: sub-criteria that declare it receive
    the cell lower corners as a sixth argument."""
    def crit(count, mass, hmin, size, level, lo=None):
        out = np.zeros(len(count), dtype=bool)
        for c in crits:
            if getattr(c, 'needs_lo', False):
                out |= np.asarray(c(count, mass, hmin, size, level, lo),
                                  dtype=bool)
            else:
                out |= np.asarray(c(count, mass, hmin, size, level),
                                  dtype=bool)
        return out
    crit.needs_lo = any(getattr(c, 'needs_lo', False) for c in crits)
    return crit


def refine_all(*crits):
    """AND-combine refinement criteria (refine only where ALL of them fire).

    The complement of `refine_any`, and the natural way to gate an aggressive
    geometric criterion on something that says whether the data can support the
    extra resolution -- see `refine_occupied`.
    """
    def crit(count, mass, hmin, size, level, lo=None):
        out = np.ones(len(count), dtype=bool)
        for c in crits:
            if getattr(c, 'needs_lo', False):
                out &= np.asarray(c(count, mass, hmin, size, level, lo),
                                  dtype=bool)
            else:
                out &= np.asarray(c(count, mass, hmin, size, level),
                                  dtype=bool)
        return out
    crit.needs_lo = any(getattr(c, 'needs_lo', False) for c in crits)
    return crit


def refine_occupied(criterion, min_count=1):
    """Gate `criterion` on a cell actually holding more than `min_count`
    particles.

    `refine_smoothing_length` refines wherever h is small, with no reference to
    whether any particle is present to justify it -- so in a galaxy with a large
    dynamic range it splits deep into regions the data does not sample. Measured
    on an evolved 462k-particle merger cutout (base 16, max_level 4):

        refine_smoothing_length(0.5)              680,940 leaves, 86.3% EMPTY,
                                                  median 0 particles/leaf
        refine_occupied(same, min_count=1)         ~10x fewer leaves, and every
                                                  split is backed by particles

    An empty leaf still receives a kernel-weighted value from whatever particles
    overlap it, but with no local sampling to constrain it -- that is shot noise,
    and on a vector field it survives into the divergence-free projection as
    small-scale structure that was never in the data. It also costs: the THB
    De Rham operators scale with leaf count, so 3M leaves needs ~80 GB where
    300k needs ~8 GB.

    This does NOT replace the anti-aliasing motivation for refining on h; it
    keeps it exactly where there are particles and stops it in vacuum.
    """
    return refine_all(criterion, refine_count(int(min_count)))


def _amr_leaf_keys(amr):
    """Per-level sorted int64 linear cell keys of an AMRGrid's leaf set
    (key = (ix*ny + iy)*nz + iz at the leaf's own level), plus the (nlev,3)
    per-level grid dims. Keys are computed from the leaf CENTRE / cell width
    (lands mid-cell, immune to edge rounding)."""
    lvl = np.asarray(amr.cell_level, dtype=np.int64)
    nlev = int(lvl.max()) + 1
    base = np.asarray(amr.base_npx, dtype=np.int64)
    dims = base[None, :] * (2 ** np.arange(nlev, dtype=np.int64))[:, None]
    centre = amr.cell_lo + 0.5 * amr.cell_size
    nl = dims[lvl]
    box = amr.hi - amr.lo
    ijk = np.floor((centre - amr.lo) / (box / nl)).astype(np.int64)
    ijk = np.clip(ijk, 0, nl - 1)
    keys = (ijk[:, 0] * nl[:, 1] + ijk[:, 1]) * nl[:, 2] + ijk[:, 2]
    return [np.sort(keys[lvl == l]) for l in range(nlev)], dims


# the 26 face/edge/corner neighbour offsets (2:1 balance is enforced across
# ALL of them: a cell-centre trilinear stencil reaches diagonal neighbours)
_NB26 = np.array([(dx, dy, dz)
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)],
                 dtype=np.int64)


def _balance_leaf_keys(levels, dims, periodic):
    """Enforce 2:1 balance on a per-level leaf-key partition IN PLACE-ish
    (returns new per-level sorted key arrays): no leaf may touch (face,
    edge or corner) a leaf more than one level coarser. Leaves violating
    the rule are split into their 8 children; splitting propagates, so
    iterate to the fixpoint (bounded by the level span). Pure integer key
    arithmetic - no geometry, no particles."""
    levels = [np.asarray(k, dtype=np.int64) for k in levels]
    nlev = len(levels)
    while True:
        marks = [[] for _ in range(nlev)]
        for L in range(nlev - 1, 1, -1):
            k = levels[L]
            if k.size == 0:
                continue
            nx, ny, nz = (int(dims[L, 0]), int(dims[L, 1]), int(dims[L, 2]))
            iz = k % nz
            iy = (k // nz) % ny
            ix = k // (ny * nz)
            nb = (np.stack([ix, iy, iz], 1)[:, None, :]
                  + _NB26[None, :, :]).reshape(-1, 3)
            valid = np.ones(len(nb), dtype=bool)
            for a, n_a in enumerate((nx, ny, nz)):
                if periodic[a]:
                    nb[:, a] %= n_a
                else:
                    valid &= (nb[:, a] >= 0) & (nb[:, a] < n_a)
            nb = nb[valid]
            rem = np.unique((nb[:, 0] * ny + nb[:, 1]) * nz + nb[:, 2])
            # containing-leaf level of each neighbour coord: the leaf set
            # partitions space, so membership is at exactly one level. Hits
            # at L-1 or finer are compliant; walk M = L-2 .. 0 and mark the
            # first (only) containment as a violation.
            for M in range(L - 2, -1, -1):
                if rem.size == 0:
                    break
                lm = levels[M]
                if lm.size == 0:
                    continue
                sh = L - M
                izr = rem % nz
                iyr = (rem // nz) % ny
                ixr = rem // (ny * nz)
                kM = (((ixr >> sh) * dims[M, 1] + (iyr >> sh)) * dims[M, 2]
                      + (izr >> sh))
                idx = np.searchsorted(lm, kM)
                hit = ((idx < lm.size)
                       & (lm[np.minimum(idx, lm.size - 1)] == kM))
                if hit.any():
                    marks[M].append(np.unique(kM[hit]))
                    rem = rem[~hit]
        if not any(marks):
            return levels
        for M in range(nlev):
            if not marks[M]:
                continue
            mk = np.unique(np.concatenate(marks[M]))
            ny, nz = int(dims[M, 1]), int(dims[M, 2])
            iz = mk % nz
            iy = (mk // nz) % ny
            ix = mk // (ny * nz)
            cny, cnz = int(dims[M + 1, 1]), int(dims[M + 1, 2])
            ch = []
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        ch.append(((2 * ix + a) * cny + (2 * iy + b)) * cnz
                                  + (2 * iz + c))
            levels[M] = np.setdiff1d(levels[M], mk, assume_unique=True)
            levels[M + 1] = np.sort(np.concatenate([levels[M + 1],
                                                    np.concatenate(ch)]))


def grade_2to1(amr, particles):
    """Return a 2:1-BALANCED copy of an AMRGrid: refine leaves until no leaf
    touches (face, edge or corner) a leaf more than one level finer, so level
    jumps decay geometrically away from refined regions instead of a fine
    bright cell abutting a huge coarse one (a trilinear/level-consistent
    readout across a many-level jump stretches the fine region's values over
    the whole coarse cell width - the glassgen galaxy z-staircase).

    The balance is computed in pure integer key space (split-only, iterated
    to the fixpoint), then the grid is REBUILT via `AMRGrid.build` with a
    criterion that refines exactly the balanced tree's internal nodes -
    cell_count / base ordering / deposit lookups all come out consistent.
    The caller must redeposit values (the leaf set changed). Returns `amr`
    unchanged (same object) if the grid is already balanced."""
    levels, dims = _amr_leaf_keys(amr)
    ncell0 = sum(k.size for k in levels)
    levels = _balance_leaf_keys(levels, dims, amr.periodic)
    if sum(k.size for k in levels) == ncell0:
        return amr
    # internal nodes of the balanced tree: ancestors of every leaf
    nlev = len(levels)
    internal = [np.empty(0, dtype=np.int64) for _ in range(nlev)]
    carry = np.empty(0, dtype=np.int64)
    for L in range(nlev - 1, 0, -1):
        ks = np.concatenate([levels[L], carry])
        ny, nz = int(dims[L, 1]), int(dims[L, 2])
        iz = ks % nz
        iy = (ks // nz) % ny
        ix = ks // (ny * nz)
        carry = np.unique(((ix >> 1) * dims[L - 1, 1] + (iy >> 1))
                          * dims[L - 1, 2] + (iz >> 1))
        internal[L - 1] = carry
    lo = amr.lo
    box = amr.hi - amr.lo

    def crit(count, mass, hmin, size, level, lo_cells):
        out = np.zeros(len(count), dtype=bool)
        if level >= nlev or internal[level].size == 0:
            return out
        nl = dims[level]
        centre = lo_cells + 0.5 * size
        ijk = np.floor((centre - lo) / (box / nl)).astype(np.int64)
        ijk = np.clip(ijk, 0, nl - 1)
        key = (ijk[:, 0] * nl[1] + ijk[:, 1]) * nl[2] + ijk[:, 2]
        idx = np.searchsorted(internal[level], key)
        return ((idx < internal[level].size)
                & (internal[level][np.minimum(idx, internal[level].size - 1)]
                   == key))

    crit.needs_lo = True
    out = AMRGrid.build(particles, base_npx=amr.base_npx, max_level=nlev - 1,
                        lo=amr.lo, hi=amr.hi, periodic=amr.periodic,
                        criterion=crit, refine_empty=True)
    want = sum(k.size for k in levels)
    if out.ncell != want:
        raise RuntimeError(f'graded rebuild produced {out.ncell} leaves, '
                           f'balance says {want}')
    return out


@dataclass
class VoronoiGrid:
    """Voronoi polyhedral cells (via pyvoro) — a third target geometry.

    A cell is an arbitrary convex polyhedron about a generator point, exactly
    what the Petkova exact integral was designed for (it integrates the kernel
    over any polyhedron face-by-face; the cube of UniformGrid/AMRGrid is just a
    special case). The SPH method evaluates the kernel at the generator.

    Build one with `VoronoiGrid.from_points(points, bounds)` (arbitrary mesh
    generators) or `VoronoiGrid.from_particles(particles)` (tessellate the
    particles themselves). pyvoro is required only for the build, not the deposit.

    Geometry is stored flat (numba-friendly), two-level CSR so cells/faces can
    have any number of faces/vertices:
      gen        (Ncell,3)  generator point of each cell
      volume     (Ncell,)   cell volume
      cell_rad   (Ncell,)   circumradius (max vertex distance from generator);
                            a particle's kernel can reach a cell only if
                            |x_particle - gen| < 2h + cell_rad (used to prune).
      cf_start   (Ncell+1,) CSR offsets: cell c's faces are [cf_start[c],cf_start[c+1])
      fv_start   (Nface+1,) CSR offsets: face f's vertices are [fv_start[f],fv_start[f+1])
      fvx/fvy/fvz(Nfv,)     ABSOLUTE vertex coords, inlined per face occurrence
                            (shared vertices are duplicated — the integral only
                            ever reads them in face-winding order, so this keeps
                            the kernel index-indirection-free).
    Face winding is pyvoro's own (the Petkova sign logic depends on it, the same
    convention `_CUBE_FACES` reproduces for the cube).

    NOTE: the deposit is NON-PERIODIC (absolute coordinates). For a periodic box,
    `from_particles` builds a periodic tessellation if asked, but boundary cells
    receive only the in-domain part of a kernel; interior mass is conserved.
    """
    gen: np.ndarray
    volume: np.ndarray
    cell_rad: np.ndarray
    cf_start: np.ndarray
    fv_start: np.ndarray
    fvx: np.ndarray
    fvy: np.ndarray
    fvz: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    periodic: object = (False, False, False)
    cell_parent: object = None      # (Ntotal,) ghost->central index; None = none
    ncentral: object = None         # number of real (central) cells

    def __post_init__(self):
        self.gen = np.ascontiguousarray(self.gen, dtype=np.float64).reshape(-1, 3)
        self.volume = np.ascontiguousarray(self.volume, dtype=np.float64).ravel()
        self.cell_rad = np.ascontiguousarray(self.cell_rad, dtype=np.float64).ravel()
        self.cf_start = np.ascontiguousarray(self.cf_start, dtype=np.int64).ravel()
        self.fv_start = np.ascontiguousarray(self.fv_start, dtype=np.int64).ravel()
        self.fvx = np.ascontiguousarray(self.fvx, dtype=np.float64).ravel()
        self.fvy = np.ascontiguousarray(self.fvy, dtype=np.float64).ravel()
        self.fvz = np.ascontiguousarray(self.fvz, dtype=np.float64).ravel()
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)
        p = self.periodic
        if np.isscalar(p):
            p = (bool(p), bool(p), bool(p))
        self.periodic = tuple(bool(x) for x in p)
        if self.cell_parent is None:                  # plain (non-ghost) mesh
            self.ncentral = self.gen.shape[0]
        else:
            self.cell_parent = np.ascontiguousarray(self.cell_parent,
                                                    dtype=np.int64).ravel()
            self.ncentral = int(self.ncentral)

    @property
    def ncell(self):
        """Number of REAL cells (central; ghosts are deposit bookkeeping only)."""
        return self.ncentral

    @property
    def ntotal(self):
        """Total stored cells including periodic ghost images (for the deposit)."""
        return self.gen.shape[0]

    @property
    def box(self):
        return self.hi - self.lo

    def fold_to_central(self, arr):
        """Sum a per-(total-)cell deposit array down onto the central cells:
        ghost-image contributions are added back into their parent cell. Returns
        the first `ncentral` rows when there are no ghosts."""
        arr = np.asarray(arr)
        if self.cell_parent is None:
            return arr[:self.ncentral]
        out = np.zeros((self.ncentral,) + arr.shape[1:], dtype=arr.dtype)
        np.add.at(out, self.cell_parent, arr)
        return out

    def centres(self):
        """(Ncell,3) generator points of the REAL cells (central)."""
        return self.gen[:self.ncentral]

    def volumes(self):
        """(Ncell,) volumes of the REAL cells (central)."""
        return self.volume[:self.ncentral]

    def edges(self):
        """Voronoi cells are polyhedra, not axis-aligned boxes — there is no
        per-axis edge array. Returns the flat face/vertex CSR instead."""
        return self.cf_start, self.fv_start, self.fvx, self.fvy, self.fvz

    @staticmethod
    def _median_nn(pts):
        """Median nearest-neighbour distance of the generators — a robust local
        resolution scale for the de-sliver floor. Preferred over the mean spacing
        (domain_vol/N)**(1/3) because that is inflated by empty volume on a
        CONCENTRATED system (a galaxy/merger with a dense disk in a large empty
        halo), where it overshoots the real spacing ~100x and the floor would
        merge away genuine structure. The median NN distance instead tracks the
        typical spacing of where the generators actually are. Ignores exact
        duplicates (NN==0) so a handful of coincident points don't drag it to 0."""
        from scipy.spatial import cKDTree
        n = pts.shape[0]
        if n < 2:
            return 0.0
        d, _ = cKDTree(pts).query(pts, k=2, workers=-1)
        nn = d[:, 1]
        pos = nn[nn > 0.0]
        if pos.size == 0:
            return 0.0
        return float(np.median(pos))

    @staticmethod
    def _merge_close(pts, tol):
        """Collapse near-coincident generators (sub-`tol` clusters) into one
        point at their centroid, so voro++ does not have to cut degenerate sliver
        faces between them. Returns `(newpts, nmerged)`.

        WHY: on a collapsed core (high dynamic range) the densest particles sit
        at separations << the smoothing length. pyvoro/voro++ cannot tessellate
        such near-coincident generators cleanly — the bisecting faces are
        ill-conditioned and the stored polyhedra develop local GAPS and OVERLAPS
        (they no longer partition space). The signed-tet integrator is exact on
        well-formed cells, so a broken partition is the sole remaining source of
        the Voronoi Petkova conservation error (~0.2% on mhdcollapse). Merging the
        coincident generators removes the slivers and restores Sum frac = 1.
        Connected components are merged transitively (union-find over all pairs
        within `tol`), so a tight chain collapses to a single representative."""
        from scipy.spatial import cKDTree
        n = pts.shape[0]
        if tol <= 0.0 or n < 2:
            return pts, 0
        pairs = cKDTree(pts).query_pairs(tol, output_type='ndarray')
        if len(pairs) == 0:
            return pts, 0
        parent = np.arange(n)

        def find(a):
            root = a
            while parent[root] != root:
                root = parent[root]
            while parent[a] != root:           # path compression
                parent[a], a = root, parent[a]
            return root

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        roots = np.array([find(i) for i in range(n)], dtype=np.int64)
        _, inv = np.unique(roots, return_inverse=True)
        m = inv.max() + 1
        newpts = np.zeros((m, 3))
        np.add.at(newpts, inv, pts)
        newpts /= np.bincount(inv, minlength=m)[:, None]
        return np.ascontiguousarray(newpts), n - m

    @staticmethod
    def _periodic_cell_rad_limit(L, per):
        """Largest cell circumradius a periodic min-image unwrap can resolve.

        A cell is rebuilt around its generator by unwrapping each vertex to its
        nearest image, so the resolvable quantity is the cell's DIAMETER, not its
        radius: two vertices of one cell more than a half-period apart have no
        consistent unwrap and the reconstructed volume is wrong. Hence
        2*cell_rad <= L/2, i.e. cell_rad <= L/4 -- a QUARTER period, not a half.

        Measured on the thin synthetic boxes (native AREPO, 1500..40000
        generators over 1x0.094x0.094 and 1x1x0.094): the budget is still open at
        max cell_rad = 0.66 half-periods (+1.5e-04) and closes to +1e-06 by 0.54,
        so the closure sits at ~L/4 and not at the L/2 the old comments assumed.
        Refining to the L/2 bound is what left slab2d stuck at +0.50%.
        """
        pl = [float(L[a]) for a in range(3) if per[a]]
        return 0.25 * min(pl) if pl else float("inf")

    @classmethod
    def _insert_for_half_period(cls, pts, lo, hi, per, mesh, limit, h):
        """Add generators to the cells that are too big for the periodic unwrap.

        Returns ``(points, n_added)``. `mesh` is anything carrying this class's
        CSR geometry (`gen`, `cell_rad`, `cf_start`, `fv_start`, `fvx/fvy/fvz`)
        -- a VoronoiGrid or the raw dict from the AREPO mesher.

        TWO strategies, cheapest first:

        1. LATTICE. Candidates are a cell-centred cubic lattice of spacing `h`
           (cubic in PHYSICAL space -- an n^3 lattice over a thin box has
           anisotropic spacing and would refine the wrong axis), kept where they
           fall inside an offending cell's circumsphere and dropped within h/2 of
           an existing generator so the insertion cannot make slivers. A lattice
           of spacing h has covering radius 0.87h, so h = limit bounds the
           refined cells at 0.87*limit. Regular spacing is what makes this the
           CHEAP option: every inserted generator uses its full radius, so it
           needs the fewest cells to get under the bound (measured on the
           1500-generator slabs: 5.4x the cells, against 7.9x for strategy 2).

        2. SPLIT. The circumsphere in (1) is a superset of the cell, so on a
           SLIVER -- large circumradius, small volume, neighbours packed close --
           every candidate can be dropped for sitting on an existing generator,
           and (1) returns nothing. Splitting cannot: it inserts at the midpoint
           between the offending cell's generator and each of its far vertices.
           A Voronoi cell is convex and both endpoints lie in it, so the midpoint
           does too, and the nearest generator to a point inside a cell is that
           cell's own -- at cell_rad/2 > limit/2, so a split can no more make a
           sliver than the lattice can. Every offending cell has a vertex beyond
           `limit` by definition, so this ALWAYS yields at least one point.

        This REFINES the mesh rather than reproducing the caller's generator set:
        the returned tessellation has more cells than points supplied. That is
        the same trade `periodic_background` already makes, and it is the only
        fix available for a cell that is too big -- no image shell can shrink it.
        """
        L = hi - lo
        gen = np.asarray(mesh["gen"] if isinstance(mesh, dict) else mesh.gen)
        cell_rad = np.asarray(
            mesh["cell_rad"] if isinstance(mesh, dict) else mesh.cell_rad)
        n_in = len(pts)
        gen, cell_rad = gen[:n_in], cell_rad[:n_in]
        bad = np.where(cell_rad > limit)[0]
        if bad.size == 0:
            return pts, 0

        from scipy.spatial import cKDTree
        n = np.maximum(1, np.ceil(L / float(h)).astype(np.int64))
        ax = [lo[a] + (np.arange(n[a]) + 0.5) * (L[a] / n[a]) for a in range(3)]
        cand = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        # Wrapped coordinates so the "near an existing generator" test sees
        # across periodic faces; a mixed-periodicity box falls back to the plain
        # tree (it can only keep a candidate it should have dropped, which the
        # h/2 spacing test then handles on the next round).
        cu = np.mod(cand - lo, L)
        kw = {"boxsize": L} if all(bool(x) for x in per) else {}
        d_bad, j = cKDTree(np.mod(gen[bad] - lo, L), **kw).query(
            cu, k=1, workers=-1)
        keep = d_bad <= cell_rad[bad][j]
        d_gen, _ = cKDTree(np.mod(gen - lo, L), **kw).query(cu, k=1, workers=-1)
        keep &= d_gen >= 0.5 * float(h)
        if keep.any():
            return (np.ascontiguousarray(
                np.concatenate((pts, cand[keep]), axis=0)), int(keep.sum()))
        return cls._split_oversize_cells(pts, lo, hi, mesh, limit, bad)

    @staticmethod
    def _split_oversize_cells(pts, lo, hi, mesh, limit, bad, kmax=2,
                              cos_sep=0.5):
        """Strategy 2 of `_insert_for_half_period`: midpoint splits inside the
        offending cells.

        Splits go along the cell's ELONGATION, which is what de-slivers it: the
        farthest vertex is by definition the long axis, and a generator halfway
        out puts the new bisector across that axis at cell_rad/4. Vertices are
        taken farthest-first, and the 2nd..kth are accepted only if they point
        at least 60 degrees (cos_sep) off every direction already taken -- a
        genuinely DIFFERENT long axis, not another point down the same one. On a
        needle, where every far vertex points the same way, that correctly
        yields the single split the cell needs instead of stacking generators
        along the spike.

        kmax=2 measured cheapest on the thin slabs (1 costs an extra round, >2
        over-inserts without closing sooner); the angular rule beats a plain
        distance rule at every kmax (slab2d, kmax=3: 9.2x the cells vs 10.3x).
        """
        L = hi - lo
        g = mesh["gen"] if isinstance(mesh, dict) else mesh.gen
        cf = mesh["cf_start"] if isinstance(mesh, dict) else mesh.cf_start
        fv = mesh["fv_start"] if isinstance(mesh, dict) else mesh.fv_start
        vx = mesh["fvx"] if isinstance(mesh, dict) else mesh.fvx
        vy = mesh["fvy"] if isinstance(mesh, dict) else mesh.fvy
        vz = mesh["fvz"] if isinstance(mesh, dict) else mesh.fvz
        add = []
        for i in bad:
            # faces of a cell are contiguous, so its vertex block is too
            v0, v1 = int(fv[cf[i]]), int(fv[cf[i + 1]])
            V = np.column_stack((vx[v0:v1], vy[v0:v1], vz[v0:v1]))
            rel = V - g[i]
            d = np.linalg.norm(rel, axis=1)
            far = np.argsort(d)[::-1]
            dirs = []
            for j in far[d[far] > limit]:
                u = rel[j] / d[j]
                if any(float(u @ w) >= cos_sep for w in dirs):
                    continue                  # same long axis, already split
                dirs.append(u)
                add.append(g[i] + 0.5 * rel[j])
                if len(dirs) >= kmax:
                    break
        if not add:
            return pts, 0
        # A cell whose unwrap already failed can have vertices outside the box;
        # wrap so an inserted generator is always a point of the domain.
        add = np.mod(np.asarray(add) - lo, L) + lo
        return (np.ascontiguousarray(np.concatenate((pts, add), axis=0)),
                len(add))

    @staticmethod
    def _periodic_background(lo, hi, n):
        """Return an ``n^3`` cell-centred guard lattice spanning the domain.

        These are cheap background/filler generators, not replacements for the
        supplied generators. In a periodic mesh their covering radius is smaller
        than half a period, so every cell has an unambiguous minimum image. In an
        open mesh they fill the empty capture volume and keep the AREPO mesher's
        internal periodic closure far from the source kernels.
        """
        n = int(n)
        if n < 2:
            raise ValueError("periodic_background must be 0 or at least 2")
        frac = (np.arange(n, dtype=np.float64) + 0.5) / n
        axes = [lo[a] + frac * (hi[a] - lo[a]) for a in range(3)]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        return np.ascontiguousarray(
            np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))

    @staticmethod
    def _periodic_coverage_radius(pts, lo, hi, background_n):
        """Estimate the largest empty-space radius in normalized periodic units.

        A small regular probe lattice makes this independent of the physical box
        aspect ratio. The comparison radius is the worst-case covering radius of
        the proposed background lattice. This check is O(N log N) with only a few
        hundred queries, negligible beside a Voronoi build.
        """
        from scipy.spatial import cKDTree
        unit = np.mod((pts - lo) / (hi - lo), 1.0)
        nprobe = max(8, 2 * int(background_n))
        frac = (np.arange(nprobe, dtype=np.float64) + 0.5) / nprobe
        xx, yy, zz = np.meshgrid(frac, frac, frac, indexing="ij")
        probes = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        dist, _ = cKDTree(unit, boxsize=1.0).query(probes, k=1, workers=-1)
        radius = float(np.max(dist))
        background_radius = np.sqrt(3.0) / (2.0 * int(background_n))
        return radius, background_radius

    @classmethod
    def from_points(cls, points, bounds=None, periodic=False, dispersion=None,
                    ghost_pad=0.25, merge_tol=None, backend="auto", n_threads=-1,
                    periodic_background="auto"):
        """Tessellate `points` (Ncell,3) into Voronoi cells.

        backend    : mesher — 'auto' (arepo if codes/arepo/AREPO/Arepo is built,
                     else
                     multivoro if installed for parallel OpenMP voro++, else pyvoro),
                     'arepo', 'multivoro', or 'pyvoro'. arepo tessellates collapsed
                     high-dynamic-range cores robustly (its native regime), so auto
                     prefers it; multivoro is ~2x faster single-threaded and up to
                     ~40x with threads. Identical CSR cells out either way.
        n_threads  : multivoro thread count; <=0 -> all cores. For arepo it is the
                     MPI rank count but defaults to SERIAL (AREPO meshing is not
                     the bottleneck; see `_arepo_ranks`); ignored by pyvoro
                     (single-threaded).

        bounds     : [[xlo,xhi],[ylo,yhi],[zlo,zhi]] domain (default: padded
                     point extent). pyvoro requires every point strictly inside.
        periodic   : bool or (3,) — a true PERIODIC tessellation via GHOST IMAGES
                     (see `_from_points_periodic`): boundary generators are
                     replicated into their neighbouring periodic images, the lot
                     is tessellated non-periodically, and the central cells are
                     then the exact periodic Voronoi cells (Sum volume = box
                     volume exactly, no boundary padding). The ghost cells are
                     retained for the deposit (folded back into their parent by
                     `fold_to_central`), so a kernel crossing a face wraps.
        periodic_background
                   : background/guard lattice. ``"auto"`` (default) adds a
                     cell-centred ``16^3`` lattice for every OPEN mesh; for a
                     periodic mesh it first measures the empty-space radius and
                     adds the lattice only when coverage is sparse. An integer
                     >=2 always adds that ``n^3`` lattice; 0 disables it. The
                     guards prevent a generator in a concentrated system from
                     owning a huge cell. For AREPO open meshes, the guard-filled
                     physical box is tessellated internally with native
                     non-cubic periodic closure, while the returned target and
                     deposit remain open. Supplied generators are never split
                     or removed (apart from near-duplicate merging).
        dispersion : pyvoro block-sizing hint (~ the typical generator spacing);
                     default = the mean spacing (domain_volume / N)**(1/3). This
                     is what sizes voro++'s internal block grid — setting it to a
                     large value (e.g. the box size) collapses the mesh into one
                     giant block and makes the build O(N^2) (and memory-hungry).
                     PERFORMANCE ONLY: the result must not depend on it. (It
                     appeared to until 2026-07-27, because passing `radii=[]`
                     made pyvoro substitute a radius of `dispersion/10` and run
                     a RADICAL tessellation — see `_tessellate`. Both mesher
                     calls now pass explicit zero radii, which is what makes
                     the tessellation an ordinary Voronoi one.)
        ghost_pad  : periodic only — the image shell width as a fraction of the
                     box per axis (generators within this distance of a boundary
                     are replicated across it). Acts as a FLOOR: the shell is
                     auto-grown to at least 3x the mean generator spacing, so
                     SPARSE meshes (few generators -> large cells) still close
                     their boundary cells correctly (otherwise Sum cell vol > box
                     vol, e.g. +5% at N=32). 0.25 already covers dense/near-
                     uniform meshes; raise it only for strongly clustered ones.
        merge_tol  : collapse generators closer than this distance into one (see
                     `_merge_close`) BEFORE tessellating, to keep voro++ from
                     emitting degenerate sliver cells (or failing outright on exact
                     duplicates) on a high-dynamic-range core. Default None -> a
                     BACKEND-AWARE fraction of the MEDIAN nearest-neighbour distance
                     (`_median_nn` x `_DESLIVER_FRAC`): 5% for pyvoro, 20% for
                     multivoro (whose parallel solver under-cuts/overlaps core cells
                     under abrupt-spacing dynamic range and so needs the spacing
                     pulled up further). An absolute resolution floor that bites only
                     near-coincident generators and is a no-op on well-resolved
                     meshes. The larger multivoro floor merges ~10% more core cells,
                     which caps the very densest cell's peak density (the volume-
                     weighted field PDF is unchanged); set merge_tol explicitly to
                     override. NOTE: a
                     LOCAL/relative floor cannot work here: the slivers form exactly
                     where the local spacing collapses, so a local threshold shrinks
                     with it and never reaches them (and merging them unevenly makes
                     voro++ fail) — the floor must be absolute. median-NN (not the
                     mean spacing (vol/N)**(1/3)) keeps it robust on CONCENTRATED
                     systems where empty volume inflates the mean ~100x. 0 disables
                     the sliver de-merge but EXACT/near-coincident generators are
                     STILL collapsed (an unconditional ~1e-9*span floor) so voro++
                     never crashes on duplicates. NOTE: merging can make Ncell
                     smaller than the supplied N, while the periodic background
                     can make it up to ``n^3`` larger.
        """
        # Own the array: wrapping below must not mutate a caller's positions when
        # their input already happens to be a contiguous float64 ndarray.
        pts = np.array(points, dtype=np.float64, order="C", copy=True).reshape(-1, 3)
        N = pts.shape[0]
        if N == 0:
            raise ValueError("need at least one generator point")
        if bounds is None:
            plo = pts.min(axis=0); phi = pts.max(axis=0)
            span = phi - plo
            pad = 0.05 * np.where(span > 0, span, 1.0) + 1e-6
            lo = plo - pad; hi = phi + pad
        else:
            lo = np.array([float(bounds[a][0]) for a in range(3)])
            hi = np.array([float(bounds[a][1]) for a in range(3)])
        per = periodic
        if np.isscalar(per):
            per = [bool(per)] * 3
        else:
            per = [bool(x) for x in per]

        L = hi - lo
        if np.any(L <= 0.0):
            raise ValueError("each upper Voronoi bound must exceed its lower bound")

        # Put every periodic coordinate into the canonical box before merging or
        # meshing. Snapshot roundoff (and some IC writers) can leave points just
        # beyond a face; they are the same periodic position, not out-of-domain
        # generators.
        for a in range(3):
            if per[a]:
                pts[:, a] = lo[a] + np.mod(pts[:, a] - lo[a], L[a])

        # A NON-periodic axis has no such canonical position: a generator
        # outside [lo, hi] there is simply outside the domain, and the mesher
        # has no cell to give it. voro++ does notice -- but it reports
        # "number of cells found was not equal to the number of particles",
        # which is the SAME opaque message it gives for coincident generators
        # (see the tol_dup note below). One message, two unrelated causes, and
        # only the other one is handled. So name this one here, before the
        # mesher runs. Roundoff-scale excursions are clamped rather than
        # refused, matching how the periodic branch above forgives an IC
        # writer that lands a hair beyond a face.
        oob_tol = 1e-9 * float(np.max(L))
        for a in range(3):
            if per[a]:
                continue
            out = np.maximum(lo[a] - pts[:, a], pts[:, a] - hi[a])
            worst = float(out.max()) if len(out) else 0.0
            if worst <= 0.0:
                continue
            if worst <= oob_tol:
                # strictly INSIDE, not onto the face: voro++ owns the half-open
                # box and refuses a generator sitting exactly on the upper wall
                # (same "number of cells found..." error), so clamping to hi
                # would trade one opaque failure for another
                np.clip(pts[:, a], lo[a] + oob_tol, hi[a] - oob_tol,
                        out=pts[:, a])
                continue
            n_out = int((out > oob_tol).sum())
            raise ValueError(
                f"{n_out} generator(s) lie outside the domain on the "
                f"non-periodic {'xyz'[a]} axis "
                f"[{lo[a]:.6g}, {hi[a]:.6g}]: worst excursion {worst:.6g} "
                f"({worst / L[a]:.2%} of the axis length). A non-periodic "
                f"axis has no wrap, so these points have no cell. Widen "
                f"`bounds`, mark the axis periodic, or drop them.")

        # A tiny regular background is more robust than recursively splitting the
        # real generator set. Auto-detect the risk BEFORE tessellating: retrying a
        # doomed 40k pyvoro build would cost another ~200 seconds.
        auto_background = isinstance(periodic_background, str)
        if auto_background:
            if periodic_background.lower() != "auto":
                raise ValueError("periodic_background must be 'auto', 0, or >=2")
            background_n = 16
        else:
            background_n = int(periodic_background)
        if background_n < 0 or background_n == 1:
            raise ValueError("periodic_background must be 0 or at least 2")
        # Open captures always get the guard in auto mode: unlike a true
        # periodic volume, their empty exterior is intentional and must be
        # represented by finite background cells. Periodic captures retain the
        # coverage test so an already space-filling particle set is left alone.
        add_background = background_n > 0
        coverage_radius = None
        background_radius = None
        if add_background and auto_background and all(per):
            # Detect only HALF-PERIOD-SCALE voids. Comparing against the 4^3
            # lattice covering radius avoids adding 4096 fillers to an ordinary
            # lower-N but already space-filling mesh. Once triggered, use 16^3:
            # the thin [151,143,18.6] regression needs this resolution for native
            # AREPO (12^3 leaves +4.71e-4 volume error; 16^3 gives +2.65e-7).
            coverage_radius, background_radius = cls._periodic_coverage_radius(
                pts, lo, hi, 4)
            add_background = coverage_radius > background_radius
        if add_background:
            bg = cls._periodic_background(lo, hi, background_n)
            pts = np.ascontiguousarray(np.concatenate((pts, bg), axis=0))
            N = pts.shape[0]
            if auto_background and all(per):
                print(
                    "VoronoiGrid: sparse periodic coverage "
                    f"(normalized empty radius {coverage_radius:.3f} > "
                    f"{background_radius:.3f}); adding {background_n}^3 = "
                    f"{background_n ** 3} background generators to prevent "
                    "half-period cells",
                    flush=True)
            elif auto_background:
                print(
                    f"VoronoiGrid: adding open-domain {background_n}^3 = "
                    f"{background_n ** 3} background guard generators",
                    flush=True)
            else:
                print(
                    f"VoronoiGrid: adding forced {background_n}^3 = "
                    f"{background_n ** 3} background guard generators",
                    flush=True)

        # EXACT/near-coincident generators are ALWAYS collapsed, independent of
        # merge_tol (incl. merge_tol=0): two generators voro++ cannot tell apart
        # have no valid distinct cells and make it FAIL outright ("number of cells
        # found was not equal to the number of particles"). This floor sits ~1e-9
        # of the domain span -- ~1e3x above voro++'s internal ~1e-12 relative
        # tolerance, far below any real spacing -- so it removes only true
        # duplicates, never structure. (merge_tol=0 thus still BUILDS on data with
        # duplicates; it only switches off the larger sliver de-merge below.)
        tol_dup = 1e-9 * float(np.max(L))
        # de-sliver: merge near-coincident generators so the mesh stays a clean
        # partition (the collapsed-core conservation fix). Default is a BACKEND-
        # AWARE fraction of median-NN: multivoro needs ~4x pyvoro's floor (its
        # parallel solver under-cuts/overlaps cells on an abrupt-spacing core where
        # serial pyvoro stays valid) -- see `_DESLIVER_FRAC`.
        effective_backend = _resolve_backend(backend)
        if merge_tol is None:
            merge_tol = (_DESLIVER_FRAC[effective_backend]
                         * cls._median_nn(pts))
        tol_eff = max(float(merge_tol), tol_dup)
        if tol_eff > 0.0:
            pts, _ = cls._merge_close(pts, tol_eff)
            N = pts.shape[0]

        if any(per):
            # AREPO is a periodic moving-mesh code: mesh the PHYSICAL periodic box
            # NATIVELY (per-axis LONG_Y/LONG_Z) instead of ghost-generator padding +
            # a cubic clip, which mangles thin non-cubic boxes (see
            # `_from_arepo_periodic`). Other backends use the ghost-image path.
            if effective_backend == "arepo":
                return cls._from_arepo_periodic(pts, lo, hi, per, ghost_pad,
                                                n_threads)
            return cls._from_points_periodic(pts, lo, hi, per, dispersion,
                                             ghost_pad, effective_backend,
                                             n_threads)

        if effective_backend == "arepo":
            if add_background:
                return cls._from_arepo_guarded_open(
                    pts, lo, hi, n_threads)
            if backend == "arepo":
                raise ValueError(
                    "open backend='arepo' requires a background guard; use "
                    "periodic_background='auto'/N, or a wall-bounded backend")
            # With guards explicitly disabled, preserve true open walls via a
            # wall-bounded backend instead of accepting AREPO's periodic cells.
            effective_backend = "multivoro" if _HAS_MULTIVORO else "pyvoro"
        return cls._tessellate(
            pts, lo, hi, lo, hi, dispersion, effective_backend, n_threads)

    @classmethod
    def _from_points_periodic(cls, pts, lo, hi, per, dispersion, ghost_pad,
                              backend="auto", n_threads=-1, _refine_left=2):
        """Periodic Voronoi by GHOST IMAGES: replicate near-boundary generators
        into their periodic images, tessellate the lot non-periodically over a
        padded box, keep the N central cells as the real (periodic) cells and the
        images as ghosts (`cell_parent` -> central index). The central cells are
        complete and exact, and tile EXACTLY one period (Sum volume = box vol)."""
        N = pts.shape[0]
        L = hi - lo
        # image shell half-width. The shell MUST span the Voronoi-neighbour reach
        # (~ a few generator spacings) or boundary cells are not cut by their
        # periodic images -> they bloat and Sum(cell vol) > box vol (badly wrong
        # for SPARSE meshes, e.g. +5% at N=32 with the fixed 0.25). So grow the
        # shell to at least 3x the mean generator spacing; the user's ghost_pad
        # fraction acts as a floor (and still covers near-uniform dense meshes).
        # Neither floor sees CLUSTERED generators (a merger: everything in two
        # galaxies, boundary voids >> mean spacing -> +7.6% at 1500 gens), so the
        # budget is VERIFIED below and the shell auto-doubled until it closes:
        # an unclosed boundary cell can only ADD volume (it runs into the pad
        # instead of being cut by its missing image), so on a fully periodic box
        # Sum(central vol) > box vol detects an insufficient shell exactly.
        # MIXED PERIODICITY: only a periodic axis gets a pad. Padding a
        # NON-periodic axis (which is what this did until 2026-07-27) is a
        # silent catastrophe: nothing is imaged across it, so every boundary
        # cell simply runs out into the pad instead of being cut, and the
        # central cells end up tiling the PADDED box. Measured on 1500 random
        # generators in a unit box: +50.0% volume per non-periodic axis
        # (per=(T,T,F) gave 1.500, per=(T,F,F) gave 2.250, both exactly the
        # padded volume). Clipping the pad puts a voro++ wall back on lo/hi
        # there, which is the correct open-boundary treatment and is what the
        # fully-open path does anyway.
        pad_axis = np.asarray(per, dtype=bool)
        spacing = float((np.prod(L) / max(N, 1)) ** (1.0 / 3.0))
        margin = np.where(pad_axis, np.maximum(ghost_pad * L, 3.0 * spacing),
                          0.0)
        margin0 = margin.copy()          # deposit reach: the ghost-CELL keep shell
        # Shell ceiling: ONE full period per padded axis -- the point at which
        # the image set is COMPLETE. Every generator has then been replicated
        # into all 26 neighbouring boxes, so a wider shell can only add images of
        # images, which say nothing new about a mesh whose cells fit in a box.
        # A half-period cap (what this was until 2026-07-29) is wrong twice over
        # on a high-aspect box. It assumes the largest cell fits in a
        # half-period, which a quasi-1D box breaks -- slab1d is 1 x 0.094 x
        # 0.094, so the thin axes hold only ~5 generator spacings and a boundary
        # cell's Voronoi reach runs past L/2; at 1500 generators the budget stuck
        # at +2.3e-05 and closed to EXACTLY 0 only once the shell reached ~L.
        # Worse, the cap was `max(min(margin, 3L), 0.5L)`, and there the
        # 3x-spacing floor (0.054) already exceeded 0.5L (0.047) -- so
        # margin_max came out EQUAL to the starting margin, the doubling loop
        # could not grow by even one step on those axes, and the first iteration
        # raised "even at the maximum shell" having never actually widened it.
        # The starting margin is still honoured if it is wider than a period (a
        # nearly-empty box, where 3*spacing alone exceeds L); dense meshes start
        # at 3*spacing << L and stop long before the ceiling either way.
        margin_max = np.where(pad_axis, np.maximum(margin, L), 0.0)
        while True:
            margin = np.minimum(margin, margin_max)
            # central generators first, then the periodic images that fall inside
            # the padded box [lo-margin, hi+margin].
            gens = [pts]
            parents = [np.arange(N, dtype=np.int64)]
            # Image ORDER follows the margin: +-1 period reaches only margin = L,
            # so a wider shell has to replicate further out or it is padded but
            # unpopulated -- the cells run into empty pad exactly as if the shell
            # had never grown, which is the failure the growth loop is trying to
            # fix. ceil(margin/L) is the smallest order that fills the pad.
            offs = [tuple(range(-k, k + 1)) if per[a] else (0,)
                    for a, k in ((a, max(1, int(np.ceil(margin[a] / L[a]))))
                                 for a in range(3))]
            for ox in offs[0]:
                for oy in offs[1]:
                    for oz in offs[2]:
                        if ox == 0 and oy == 0 and oz == 0:
                            continue
                        shift = np.array([ox * L[0], oy * L[1], oz * L[2]])
                        q = pts + shift
                        keep = np.ones(N, dtype=bool)
                        for a in range(3):
                            keep &= (q[:, a] >= lo[a] - margin[a]) & \
                                    (q[:, a] <= hi[a] + margin[a])
                        if keep.any():
                            gens.append(q[keep])
                            parents.append(np.where(keep)[0].astype(np.int64))
            gen_all = np.ascontiguousarray(np.concatenate(gens, axis=0))
            parent_all = np.concatenate(parents)
            plo = lo - margin; phi = hi + margin
            # tessellate the padded ghosted set; store on the PHYSICAL box.
            vg = cls._tessellate(gen_all, plo, phi, lo, hi, dispersion,
                                 backend, n_threads)
            vg.periodic = tuple(bool(x) for x in per)
            vg.cell_parent = np.ascontiguousarray(parent_all, dtype=np.int64)
            vg.ncentral = N
            # The budget applies under MIXED periodicity too, now that the pad
            # is clipped: the tessellation domain is exactly [lo, hi] on every
            # non-periodic axis, so the central cells still tile exactly one
            # period. It was skipped here while the pad was unclipped, which
            # is why a +50%-per-axis error could go unnoticed.
            verr = float(vg.volumes().sum()) / float(np.prod(L)) - 1.0
            if verr <= _VOL_BUDGET_TOL:
                # closed (or below noise). A deficit beyond noise cannot be a
                # shell problem (see _VOL_BUDGET_TOL) -- it is mesher volume
                # error, and re-tessellating at a wider shell cannot fix it, so
                # fail rather than grow.
                if verr < -_VOL_BUDGET_TOL:
                    raise RuntimeError(
                        f"periodic Voronoi volume budget short by {verr:+.2e}: "
                        f"the {_resolve_backend(backend)!r} cells sum to LESS "
                        f"than the box, i.e. they leave gaps and do not "
                        f"partition it. A deficit is mesher volume error, not "
                        f"an image-shell problem, so growing the shell cannot "
                        f"fix it. The mesh is unusable — every conservation "
                        f"property downstream assumes the cells tile the box "
                        f"exactly. Try another backend "
                        f"(`backend='multivoro'` or `'auto'`).")
                break
            if np.all(margin >= margin_max):
                # Last resort: REFINE. Growing the shell is the exact fix and is
                # tried first -- it tessellates the caller's generators and
                # nothing else -- but a cell that is genuinely bigger than the
                # shell ceiling cannot be closed by any shell, and there the
                # only move left is to make the cells smaller. Cost is why this
                # is second, not first: it changes the returned mesh.
                if _refine_left > 0:
                    lim = cls._periodic_cell_rad_limit(L, per)
                    new, nadd = cls._insert_for_half_period(
                        pts, lo, hi, per, vg, lim, lim)
                    if nadd:
                        print(f"VoronoiGrid: ghost-image budget {verr:+.2e} at "
                              f"the maximum shell; adding {nadd} generators "
                              f"inside the over-large cells and rebuilding",
                              flush=True)
                        return cls._from_points_periodic(
                            new, lo, hi, per, dispersion, ghost_pad, backend,
                            n_threads, _refine_left - 1)
                raise RuntimeError(
                    f"periodic Voronoi volume budget off by {verr:+.2e} even at "
                    f"the maximum (one-period) image shell {margin_max}: the "
                    f"{_resolve_backend(backend)!r} boundary cells are still "
                    f"not being cut by their periodic images, so the central "
                    f"cells do not tile the box. This needs a mesh whose "
                    f"largest cell fits within one period — if {N} "
                    f"generators over this box genuinely cannot give one, the "
                    f"box is too sparsely sampled to tessellate periodically.")
            margin = 2.0 * margin
        # The auto-grown shell is needed only WHILE tessellating (to close the
        # central cells); the deposit only ever reaches ghost cells within a
        # kernel of the box (the `margin0` heuristic, as before). Keeping the
        # whole grown shell is expensive downstream — on a clustered coarse mesh
        # it multiplies the giant boundary void cells ~26x, and via the
        # 2h+max(cell_rad) candidate radius made the petkova deposit ~14x
        # slower — so prune ghost cells farther than margin0 from the box
        # (per-axis, slackened by each cell's circumradius so a giant cell that
        # still touches the reach shell is kept).
        if np.any(margin > margin0) and vg.gen.shape[0] > N:
            g = vg.gen; rad = vg.cell_rad
            keep = np.ones(g.shape[0], dtype=bool)
            for a in range(3):
                keep &= (g[:, a] >= lo[a] - margin0[a] - rad) & \
                        (g[:, a] <= hi[a] + margin0[a] + rad)
            keep[:N] = True
            if not keep.all():
                nf = np.diff(vg.cf_start)
                fmask = np.repeat(keep, nf)
                nv = np.diff(vg.fv_start)
                vmask = np.repeat(fmask, nv)
                cf = np.zeros(int(keep.sum()) + 1, np.int64)
                cf[1:] = np.cumsum(nf[keep])
                fv = np.zeros(int(fmask.sum()) + 1, np.int64)
                fv[1:] = np.cumsum(nv[fmask])
                vg.gen = np.ascontiguousarray(g[keep])
                vg.volume = np.ascontiguousarray(vg.volume[keep])
                vg.cell_rad = np.ascontiguousarray(rad[keep])
                vg.cf_start = cf
                vg.fv_start = fv
                vg.fvx = np.ascontiguousarray(vg.fvx[vmask])
                vg.fvy = np.ascontiguousarray(vg.fvy[vmask])
                vg.fvz = np.ascontiguousarray(vg.fvz[vmask])
                vg.cell_parent = np.ascontiguousarray(vg.cell_parent[keep])
        return vg

    @classmethod
    def from_particles(cls, particles, bounds=None, periodic=None, dispersion=None,
                       ghost_pad=0.25, merge_tol=None, backend="auto", n_threads=-1,
                       periodic_background="auto"):
        """Tessellate the particle positions themselves into Voronoi cells.

        bounds default to the particle box (centered) when known, else the padded
        particle extent; periodic defaults to particles.periodic. For a periodic
        box the bounds ARE the period (no edge padding — ghost images handle the
        boundary), so Sum cell volume = box volume exactly.

        merge_tol and periodic_background forward to `from_points`.
        """
        pos = np.ascontiguousarray(particles.pos, dtype=np.float64)
        if periodic is None:
            periodic = particles.periodic
        per_any = (periodic if np.isscalar(periodic) else any(periodic))
        if bounds is None and particles.box is not None:
            half = 0.5 * np.asarray(particles.box, dtype=np.float64).reshape(3)
            if per_any:                                  # period == the box, exactly
                bounds = [[-half[a], half[a]] for a in range(3)]
            else:
                plo = pos.min(axis=0); phi = pos.max(axis=0)
                lo = np.minimum(-half, plo) - 1e-6
                hi = np.maximum(half, phi) + 1e-6
                bounds = [[lo[a], hi[a]] for a in range(3)]
        return cls.from_points(pos, bounds=bounds, periodic=periodic,
                               dispersion=dispersion, ghost_pad=ghost_pad,
                               merge_tol=merge_tol, backend=backend,
                               n_threads=n_threads,
                               periodic_background=periodic_background)

    @staticmethod
    def _from_pyvoro(cells, lo, hi, periodic):
        """Flatten a pyvoro `compute_voronoi` result into the CSR arrays."""
        N = len(cells)
        gen = np.empty((N, 3)); volume = np.empty(N); cell_rad = np.empty(N)
        cf_start = np.zeros(N + 1, dtype=np.int64)
        fv_start_list = [0]
        fvx, fvy, fvz = [], [], []
        nface = 0
        for c, cell in enumerate(cells):
            verts = np.asarray(cell["vertices"], dtype=np.float64)
            g = np.asarray(cell["original"], dtype=np.float64)
            gen[c] = g
            volume[c] = float(cell["volume"])
            cell_rad[c] = (np.sqrt(((verts - g) ** 2).sum(axis=1)).max()
                           if verts.shape[0] else 0.0)
            for face in cell["faces"]:
                vi = face["vertices"]
                if len(vi) < 3:
                    continue                       # skip degenerate faces
                for idx in vi:
                    fvx.append(verts[idx, 0])
                    fvy.append(verts[idx, 1])
                    fvz.append(verts[idx, 2])
                fv_start_list.append(len(fvx))
                nface += 1
            cf_start[c + 1] = nface
        return VoronoiGrid(
            gen=gen, volume=volume, cell_rad=cell_rad, cf_start=cf_start,
            fv_start=np.asarray(fv_start_list, dtype=np.int64),
            fvx=np.asarray(fvx), fvy=np.asarray(fvy), fvz=np.asarray(fvz),
            lo=lo, hi=hi, periodic=periodic)

    @staticmethod
    def _from_multivoro(cells, pts, lo, hi, periodic):
        """Flatten a multivoro result into the CSR arrays. multivoro returns a
        list of `Cell` (get_vertices/get_face_vertices/get_neighbors) in INPUT
        ORDER, but gives neither the generator nor the cell volume — so the
        generator is the input point and the volume is computed here from the
        polyhedron faces (signed tets fanned from the generator). `get_face_
        vertices()` is a flat list [k0, v..(k0), k1, v..(k1), ...] of per-face
        vertex-index runs. The signed-tet Petkova integrator re-derives face
        orientation from the generator, so the face winding need not match
        pyvoro's."""
        N = len(cells)
        gen = np.ascontiguousarray(np.asarray(pts, dtype=np.float64)[:N])
        # Collect the per-cell vertices/faces via the (fast) nanobind accessors,
        # concatenated with offsets, then do the CSR assembly + volume in numba
        # (the per-vertex/per-tet work is the cost; the accessors are ~0.1s/40k).
        vlist = [None] * N
        flist = [None] * N
        vcounts = np.empty(N, dtype=np.int64)
        fcounts = np.empty(N, dtype=np.int64)
        for c in range(N):
            cell = cells[c]
            v = np.ascontiguousarray(cell.get_vertices(), dtype=np.float64)
            f = np.ascontiguousarray(cell.get_face_vertices(), dtype=np.int64)
            vlist[c] = v; flist[c] = f
            vcounts[c] = v.shape[0]; fcounts[c] = f.shape[0]
        VV = (np.concatenate(vlist, axis=0) if N else np.zeros((0, 3)))
        FF = (np.concatenate(flist) if N else np.zeros(0, dtype=np.int64))
        vstart = np.zeros(N + 1, dtype=np.int64); vstart[1:] = np.cumsum(vcounts)
        fstart = np.zeros(N + 1, dtype=np.int64); fstart[1:] = np.cumsum(fcounts)
        cf_start, fv_start, fvx, fvy, fvz, volume, cell_rad = _mv_flatten(
            VV, vstart, FF, fstart,
            np.ascontiguousarray(gen[:, 0]), np.ascontiguousarray(gen[:, 1]),
            np.ascontiguousarray(gen[:, 2]))
        return VoronoiGrid(
            gen=gen, volume=volume, cell_rad=cell_rad, cf_start=cf_start,
            fv_start=fv_start, fvx=fvx, fvy=fvy, fvz=fvz,
            lo=lo, hi=hi, periodic=periodic)

    @classmethod
    def _from_arepo(cls, pts, dom_lo, dom_hi, grid_lo, grid_hi, periodic,
                    n_threads):
        """Tessellate via AREPO (codes/arepo) through the `_arepo` shell-out helper
        (write HDF5 snapshot -> run `Arepo param.txt 14` mesh dump -> parse). AREPO
        runs PERIODIC over a cubic box (its native regime); each cell is rebuilt as
        the ConvexHull of its periodic-unwrapped vertices -> exact volume + clean
        faces (AREPO's raw face lists need it; see `_arepo/arepo_mesh.py`).
        n_threads defaults to SERIAL (see `_arepo_ranks`: AREPO meshing is not the
        bottleneck, so MPI ranks rarely pay off); an explicit >1 launches
        `mpirun -np n_threads` (capped by generators-per-rank)."""
        if not _HAS_AREPO:
            raise RuntimeError(
                "backend='arepo' needs the built Arepo executable in codes/arepo "
                "(see setupfiles/sph_interp/_arepo/arepo_mesh.py)")
        nt = _arepo_ranks(n_threads, len(pts))
        m = _arepo_mesh.tessellate(pts, dom_lo, dom_hi, n_threads=nt)
        return VoronoiGrid(
            gen=np.ascontiguousarray(m["gen"]), volume=m["volume"],
            cell_rad=m["cell_rad"], cf_start=m["cf_start"], fv_start=m["fv_start"],
            fvx=m["fvx"], fvy=m["fvy"], fvz=m["fvz"],
            lo=grid_lo, hi=grid_hi, periodic=periodic)

    @classmethod
    def _from_arepo_guarded_open(cls, pts, lo, hi, n_threads):
        """AREPO mesh for an open, support-complete, guard-filled capture box.

        AREPO can only close a Voronoi tessellation periodically. The regular
        guard lattice fills the otherwise empty open capture region, so we may
        use AREPO's native non-cubic periodic closure to construct a clean
        partition of exactly ``[lo, hi]``. The returned grid is deliberately
        marked OPEN: particle kernels are not wrapped, and support completeness
        guarantees that no source kernel reaches the internal closure faces.
        Thus periodicity is only a mesher implementation detail in zero-source
        guard territory.
        """
        if not _HAS_AREPO:
            raise RuntimeError(
                "backend='arepo' needs the built Arepo executable in codes/arepo "
                "(see setupfiles/sph_interp/_arepo/arepo_mesh.py)")
        nt = _arepo_ranks(n_threads, len(pts))
        m = _arepo_mesh.tessellate(
            pts, lo, hi, n_threads=nt, periodic=(True, True, True))
        expected = float(np.prod(hi - lo))
        actual = float(np.sum(m["volume"]))
        verr = actual / expected - 1.0
        if abs(verr) > _VOL_BUDGET_TOL:
            raise RuntimeError(
                "guarded open AREPO Voronoi volume budget failed: "
                f"sum(V)/domain-1 = {verr:+.3e}; refusing overlapping cells")
        print(
            "VoronoiGrid: AREPO closure is periodic; returned Petkova target "
            "is open (no kernel wrapping)",
            flush=True)
        return VoronoiGrid(
            gen=np.ascontiguousarray(m["gen"]), volume=m["volume"],
            cell_rad=m["cell_rad"], cf_start=m["cf_start"],
            fv_start=m["fv_start"], fvx=m["fvx"], fvy=m["fvy"],
            fvz=m["fvz"], lo=lo, hi=hi,
            periodic=(False, False, False))

    @classmethod
    def _arepo_refine_to_budget(cls, pts, lo, hi, per, m, verr, nt):
        """Insert generators into the over-large cells until the native periodic
        budget closes. Returns ``(points, mesh, verr)`` -- the last mesh built,
        which the caller re-checks (an unclosed budget then takes the fallback).

        Each round targets `_periodic_cell_rad_limit` and re-meshes; the limit is
        a hard geometric bound, so one round normally does it (measured on the
        1500-generator coarse meshes: slab1d +1.36% -> +1e-06 with 261 points
        added, slab2d +12.88% -> +0.00% with 6612). The extra rounds cover the
        case where refining one region opens a neighbouring cell up.

        `_REFINE_MAX_GROWTH` is the escape hatch: a very high aspect box can need
        more cells than the caller could possibly want (the count needed scales
        as (L_long/L_short)^2 for a quasi-2D box), and silently returning a mesh
        20x the requested size is worse than handing the job to the ghost path,
        which tessellates the ORIGINAL generators exactly.
        """
        L = hi - lo
        limit = cls._periodic_cell_rad_limit(L, per)
        n_in = len(pts)
        h = limit
        for _ in range(_REFINE_ROUNDS):
            new, nadd = cls._insert_for_half_period(
                pts, lo, hi, per, m, limit, h)
            if nadd == 0:
                break
            if len(new) > _REFINE_MAX_GROWTH * n_in:
                print(f"VoronoiGrid: native AREPO refinement would need "
                      f"> {_REFINE_MAX_GROWTH:g}x the {n_in} generators "
                      f"supplied; leaving it to the ghost-image path",
                      flush=True)
                break
            pts = new
            m = _arepo_mesh.tessellate(pts, lo, hi, n_threads=nt, periodic=per)
            verr = float(np.sum(m["volume"])) / float(np.prod(L)) - 1.0
            print(f"VoronoiGrid: native AREPO budget {verr:+.3%} after adding "
                  f"{nadd} generators inside cells wider than the periodic "
                  f"unwrap limit ({len(pts)} cells now)", flush=True)
            if abs(verr) <= _VOL_BUDGET_TOL:
                break
            h *= 0.7
        return pts, m, verr

    @classmethod
    def _from_arepo_periodic(cls, pts, lo, hi, per, ghost_pad, n_threads):
        """NATIVE periodic AREPO Voronoi. Unlike the ghost-generator path
        (`_from_points_periodic`), AREPO meshes the PHYSICAL non-cubic box directly
        with per-axis periods (LONG_Y/LONG_Z), so a thin/anisotropic box tessellates
        correctly (the cubic-clip ghost path put the short-axis period wrong ->
        malformed cells, e.g. KH [1,1,0.306] petkova +340%). AREPO returns the N true
        periodic cells (each rebuilt around its generator by the per-face min-image
        unwrap). We then add periodic GHOST CELLS -- boundary cells translated by
        +/-box per periodic axis -- so a kernel crossing a face still finds the cell
        on the other side; `cell_parent`/`fold_to_central` sum ghost deposits back
        into their central cell (identical bookkeeping to the ghost-image path)."""
        if not _HAS_AREPO:
            raise RuntimeError(
                "backend='arepo' needs the built Arepo executable in codes/arepo "
                "(see setupfiles/sph_interp/_arepo/arepo_mesh.py)")
        nt = _arepo_ranks(n_threads, len(pts))    # serial by default (see `_from_arepo`)
        m = _arepo_mesh.tessellate(pts, lo, hi, n_threads=nt, periodic=per)
        # cell volumes are recomputed from the min-image-UNWRAPPED cell vertices
        # (ConvexHull in arepo_mesh._reconstruct_cells); a cell spanning more than
        # a half-period (sparse/CLUSTERED mesh: boundary voids of a merger) makes
        # that unwrap ambiguous and the volumes wrong (+20.6% on a 1500-generator
        # merger mesh). Detect via the closed-box budget and fall back to the
        # ghost-image path, which self-verifies its shell (`_from_points_periodic`).
        # Tolerance `_VOL_BUDGET_TOL` (1e-5, same as the ghost path): the eps
        # clip + hull reconstruction carries ~1e-6 relative volume noise on a
        # healthy mesh (a collapsed-core 40k mesh trips 1e-9 at "+0.000%"),
        # while a real unwrap failure is >= 0.1%. Two-sided here (unlike the
        # ghost path) because a bad unwrap can move volume either way.
        if all(per):
            verr = float(np.sum(m["volume"])) / float(np.prod(hi - lo)) - 1.0
            # A cell too big for the unwrap is a MESH problem, and no amount of
            # re-meshing the same generators fixes it -- so refine: insert
            # generators into the offending cells until every cell fits, and keep
            # the native per-axis-period tessellation. Switching to the ghost
            # path (what this did unconditionally until 2026-07-29) abandons
            # AREPO's correct thin-box periods for a backend whose cubic-clip
            # history on thin boxes is exactly why the native path exists.
            # Refinement CHANGES the mesh -- more cells come back than
            # generators went in -- so it is capped and the ghost path stays as
            # the fallback for the case where the refinement would explode.
            if abs(verr) > _VOL_BUDGET_TOL:
                pts, m, verr = cls._arepo_refine_to_budget(
                    pts, lo, hi, per, m, verr, nt)
            if abs(verr) > _VOL_BUDGET_TOL:
                import warnings
                backend = "multivoro" if _HAS_MULTIVORO else "pyvoro"
                warnings.warn(
                    f"native AREPO periodic mesh volume budget off by {verr:+.3%} "
                    f"(cell-spanning-half-period unwrap limit) and refinement "
                    f"did not close it; rebuilding via the "
                    f"ghost-image path [{backend}]")
                # re-apply the FALLBACK backend's de-sliver floor: the arepo
                # entry path merged with _DESLIVER_FRAC["arepo"]=0.0, and
                # multivoro without its 20%-median-NN floor under-cuts/overlaps
                # collapsed-core cells (the dynamic-range defect, +2260%-scale
                # volume garbage).
                mtol = _DESLIVER_FRAC[backend] * cls._median_nn(pts)
                if mtol > 0.0:
                    pts, _ = cls._merge_close(pts, mtol)
                return cls._from_points_periodic(pts, lo, hi, per, None,
                                                 ghost_pad, backend, n_threads)
        gen = np.ascontiguousarray(m["gen"]); vol = m["volume"]; crad = m["cell_rad"]
        cf = np.ascontiguousarray(m["cf_start"], dtype=np.int64)
        fvx = m["fvx"]; fvy = m["fvy"]; fvz = m["fvz"]
        N = gen.shape[0]
        L = hi - lo
        # Replicate boundary cells across each periodic face. The image shell width
        # matches the ghost-generator path: max(ghost_pad*L, 3*mean-spacing), a
        # FLOOR that covers a kernel's boundary reach; over-replication is harmless
        # (untouched ghosts deposit 0 and fold to 0). AREPO faces are all triangles,
        # so each cell's vertex block is contiguous (3 verts/face).
        spacing = float((np.prod(L) / max(N, 1)) ** (1.0 / 3.0))
        margin = np.maximum(ghost_pad * L, 3.0 * spacing)
        offs = [(-1, 0, 1) if per[a] else (0,) for a in range(3)]
        par_list = []; shift_list = []
        for ox in offs[0]:
            for oy in offs[1]:
                for oz in offs[2]:
                    if ox == 0 and oy == 0 and oz == 0:
                        continue
                    s = np.array([ox * L[0], oy * L[1], oz * L[2]])
                    q = gen + s
                    keep = np.ones(N, dtype=bool)
                    for a in range(3):
                        keep &= (q[:, a] >= lo[a] - margin[a]) & \
                                (q[:, a] <= hi[a] + margin[a])
                    if keep.any():
                        idx = np.where(keep)[0]
                        par_list.append(idx)
                        shift_list.append(np.repeat(s[None, :], idx.size, axis=0))

        nf = np.diff(cf)                                   # faces per central cell
        if par_list:
            par = np.concatenate(par_list).astype(np.int64)
            shifts = np.concatenate(shift_list, axis=0)    # (G,3)
            # per-ghost vertex slices into the (triangle) coord arrays: cell c spans
            # verts [3*cf[c], 3*cf[c+1]); gather them all, then translate by shift.
            vcnt = 3 * nf[par]                             # verts per ghost
            gidx = np.concatenate([np.arange(3 * cf[c], 3 * cf[c + 1]) for c in par]) \
                if par.size else np.zeros(0, np.int64)
            vshift = np.repeat(shifts, vcnt, axis=0)       # (sum vcnt, 3)
            g_gen = gen[par] + shifts
            g_vol = vol[par]; g_crad = crad[par]
            new_gen = np.concatenate([gen, g_gen], axis=0)
            new_vol = np.concatenate([vol, g_vol])
            new_crad = np.concatenate([crad, g_crad])
            new_fvx = np.concatenate([fvx, fvx[gidx] + vshift[:, 0]])
            new_fvy = np.concatenate([fvy, fvy[gidx] + vshift[:, 1]])
            new_fvz = np.concatenate([fvz, fvz[gidx] + vshift[:, 2]])
            all_nf = np.concatenate([nf, nf[par]])
            cell_parent = np.concatenate([np.arange(N, dtype=np.int64), par])
        else:
            new_gen, new_vol, new_crad = gen, vol, crad
            new_fvx, new_fvy, new_fvz = fvx, fvy, fvz
            all_nf = nf
            cell_parent = np.arange(N, dtype=np.int64)
        # rebuild CSR: all faces are triangles, so fv_start is a fixed 3-stride.
        new_cf = np.zeros(all_nf.size + 1, np.int64); new_cf[1:] = np.cumsum(all_nf)
        tface = int(new_cf[-1])
        new_fv = np.arange(tface + 1, dtype=np.int64) * 3
        g = VoronoiGrid(
            gen=new_gen, volume=new_vol, cell_rad=new_crad,
            cf_start=new_cf, fv_start=new_fv,
            fvx=new_fvx, fvy=new_fvy, fvz=new_fvz,
            lo=lo, hi=hi, periodic=tuple(bool(x) for x in per))
        g.cell_parent = np.ascontiguousarray(cell_parent, dtype=np.int64)
        g.ncentral = N
        return g

    # Relative tolerance on "the cells tile the mesher domain". Measured on
    # pyvoro and multivoro across open/periodic/mixed geometries with and
    # without background guards, N = 400..5000: worst |ratio-1| = 4.4e-16, i.e.
    # exact. arepo is the loose one at ~5e-7, because it rebuilds each cell as
    # a ConvexHull of float32 dumped vertices. 1e-5 sits far above that floor
    # and far below every failure this has actually caught (1.5x, 147x, 1901x,
    # 10^4x), so it cannot cry wolf on a healthy mesh.
    _TESS_BUDGET_TOL = 1.0e-5

    @classmethod
    def _tessellate(cls, pts, dom_lo, dom_hi, grid_lo, grid_hi, dispersion,
                    backend, n_threads):
        """Raw NON-periodic tessellation over [dom_lo, dom_hi], BUDGET-CHECKED.

        A Voronoi tessellation partitions its container, so the cell volumes
        must sum to the domain volume. Everything downstream is built on that:
        petkova's mass budget, the periodic volume budget, the AMR capture
        ledgers. Until 2026-07-27 nothing asserted it here, and two separate
        bugs - the mixed-periodicity pad (task 3) and the pyvoro `radii=[]`
        call (task 13) - each returned a grossly non-partitioning mesh that the
        library then used without complaint. This is the one check that would
        have caught both, at the source, in a single run.
        """
        vg = cls._tessellate_raw(pts, dom_lo, dom_hi, grid_lo, grid_hi,
                                 dispersion, backend, n_threads)
        dom = float(np.prod(np.asarray(dom_hi, dtype=np.float64)
                            - np.asarray(dom_lo, dtype=np.float64)))
        got = float(vg.volumes().sum())
        if dom > 0.0 and abs(got / dom - 1.0) > cls._TESS_BUDGET_TOL:
            raise RuntimeError(
                f"Voronoi backend {_resolve_backend(backend)!r} did not "
                f"partition its domain: {pts.shape[0]} generators over a "
                f"volume of {dom:.6g} produced cells summing to {got:.6g} "
                f"({got / dom:.6g} x). The mesh is unusable - every "
                f"conservation property downstream assumes the cells tile the "
                f"domain exactly. Try another backend "
                f"(`backend='multivoro'` or `'auto'`).")
        return vg

    @classmethod
    def _tessellate_raw(cls, pts, dom_lo, dom_hi, grid_lo, grid_hi, dispersion,
                        backend, n_threads):
        """The tessellation itself: `pts` over the mesher domain
        [dom_lo, dom_hi] -> CSR VoronoiGrid stored with bounds [grid_lo, grid_hi]
        (the two differ for the periodic ghost path: padded mesher box vs physical
        box). `backend`: 'auto' (arepo if codes/arepo/AREPO/Arepo is built, else
        multivoro if installed, else pyvoro), 'arepo' (via the shell-out mesh dump),
        'multivoro', or 'pyvoro'. `n_threads`: multivoro thread count, or arepo MPI
        rank count (>1 -> mpirun); <=0 -> all cores (multivoro)."""
        N = pts.shape[0]
        eff = _resolve_backend(backend)          # auto -> arepo/multivoro/pyvoro
        if eff == "arepo":
            return cls._from_arepo(pts, dom_lo, dom_hi, grid_lo, grid_hi,
                                   (False, False, False), n_threads)
        use_mv = eff == "multivoro"
        if backend == "multivoro" and not _HAS_MULTIVORO:
            raise ImportError("backend='multivoro' but the multivoro package is "
                              "not installed")
        if use_mv:
            nt = _resolve_threads(n_threads)
            lim = np.array([dom_lo, dom_hi], dtype=np.float64)   # (2,3): lower,upper
            with _silence_cstd():
                # radii = explicit zeros: an ORDINARY Voronoi tessellation, not
                # a radical/power one. Keep it explicit here as well as in the
                # pyvoro branch below - passing nothing is what broke pyvoro
                # (see the long note there), and the two branches should be
                # obviously making the same request of their mesher.
                cells = _multivoro.compute_voronoi(
                    np.ascontiguousarray(pts, dtype=np.float64), limits=lim,
                    radii=np.zeros(N), periodic_boundaries=(False, False, False),
                    n_threads=nt)
            return cls._from_multivoro(cells, pts, grid_lo, grid_hi,
                                       (False, False, False))
        import pyvoro
        if dispersion is None:
            dispersion = float((np.prod(np.asarray(dom_hi) - np.asarray(dom_lo))
                                / max(N, 1)) ** (1.0 / 3.0))
        limits = [[dom_lo[a], dom_hi[a]] for a in range(3)]
        # RADII MUST BE EXPLICIT ZEROS, never the empty list. pyvoro always
        # drives voro++'s container_poly (the RADICAL/power container), and on
        # an empty list it substitutes a nonzero uniform radius of its own:
        #
        #     if len(radii) != len(points):
        #         radii = [dispersion / 10.] * len(points)
        #
        # A power diagram with equal weights is mathematically the ordinary
        # Voronoi diagram, but voro++ sizes its neighbour-search cutoff from
        # the radii, and with that substituted radius the search terminates
        # early: cells come back under-cut with ~7 faces instead of the ~15.5
        # of a 3D Poisson cell, and the volumes do not sum to the box. Because
        # the substituted radius is tied to `dispersion`, the error also tracks
        # a parameter that is supposed to be a pure performance hint - which is
        # what made this look like a broken library rather than a bad call.
        # Measured on uniform random points in a unit cube, sum(vol) vs 1.0:
        # N=400 147.2, N=800 284.6, N=1000 312.4, N=5000 1901.4 -> all exactly
        # 1.0 with explicit zeros. (multivoro was always given real zeros,
        # which is why only this backend was affected.)
        cells = pyvoro.compute_voronoi(pts.tolist(), limits, dispersion,
                                       radii=[0.0] * N,
                                       periodic=[False, False, False])
        return cls._from_pyvoro(cells, grid_lo, grid_hi, (False, False, False))


def voronoi_candidates(gen, pos, radii):
    """Per-particle candidate cell lists for a Voronoi deposit, as flat CSR.

    For each particle, every generator within `radii[i]` of it is a candidate
    cell (a parallel cKDTree range search — sklearn's BallTree.query_radius is
    single-threaded and was the dominant cost of the whole Voronoi deposit on
    ~1e6-particle snapshots). Returns (cand_start, cand_count, cand_cell)
    where particle i's candidate cell indices are
    cand_cell[cand_start[i] : cand_start[i] + cand_count[i]].
    """
    from scipy.spatial import cKDTree
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    N = pos.shape[0]
    tree = cKDTree(np.ascontiguousarray(gen, dtype=np.float64))
    ind = tree.query_ball_point(
        pos, np.ascontiguousarray(radii, dtype=np.float64), workers=-1)
    counts = np.fromiter((len(a) for a in ind), dtype=np.int64, count=N)
    start = np.zeros(N, dtype=np.int64)
    if N:
        start[1:] = np.cumsum(counts)[:-1]
    cand = (np.concatenate(ind).astype(np.int64) if counts.sum()
            else np.zeros(0, dtype=np.int64))
    return start, counts, cand


def generator_buckets(gen, rad=None, nb=64):
    """Uniform bucket grid over the generators, for FUSED range searches
    inside a numba deposit (no per-particle Python candidate lists — building
    those via cKDTree.query_ball_point costs minutes at >=1e8 pairs, and on
    clustered meshes the circumradius-inflated query radius makes almost all
    of them false candidates anyway).

    Returns (blo, bw, bstart, border, bradmax):
      blo, bw       (3,)      bucket-grid origin / bucket width per axis
                              (covers the full generator extent incl. ghosts);
      bstart        (nb^3+1,) CSR offsets: bucket b holds generators
                              border[bstart[b] : bstart[b+1]];
      border        (Ngen,)   generator indices, bucket-sorted;
      bradmax       (nb^3,)   max cell circumradius per bucket (zeros when
                              `rad` is None) — lets the walker prune a bucket
                              by dist(particle, bucket box) >= reach + bradmax.
    Bucket index of point q along axis a: min(int((q[a]-blo[a])/bw[a]), nb-1).
    """
    gen = np.asarray(gen, dtype=np.float64).reshape(-1, 3)
    M = gen.shape[0]
    blo = gen.min(axis=0) if M else np.zeros(3)
    bhi = gen.max(axis=0) if M else np.ones(3)
    bw = np.maximum((bhi - blo) / nb, 1e-300)
    ib = np.minimum(((gen - blo) / bw).astype(np.int64), nb - 1)
    key = (ib[:, 0] * nb + ib[:, 1]) * nb + ib[:, 2]
    border = np.argsort(key, kind="stable").astype(np.int64)
    counts = np.bincount(key, minlength=nb * nb * nb)
    bstart = np.zeros(nb * nb * nb + 1, dtype=np.int64)
    bstart[1:] = np.cumsum(counts)
    bradmax = np.zeros(nb * nb * nb, dtype=np.float64)
    if rad is not None and M:
        np.maximum.at(bradmax, key, np.asarray(rad, dtype=np.float64).ravel())
    return (np.ascontiguousarray(blo), np.ascontiguousarray(bw),
            bstart, border, bradmax)


@dataclass
class Projection2D:
    """A 2D pixel grid in the (axis1, axis2) plane for direct rendering.

    Fast SPLASH-style deposit (scales as npx^2 * N, no 3D cube):
      mode='column'     line-of-sight integral (surface density for 'rho');
      mode='average'    kernel-weighted LOS average (column(A)/column(1));
      mode='rhocolumn'  density-weighted LOS average (mass-weighted);
      mode='slice'      field value on the plane at `zslice` (3D kernel).

    axis1, axis2 : world axes (0,1,2) mapped to image x,y; the remaining axis is
                   the line of sight (`los_axis`).
    lo, hi       : (2,) image extent in the (axis1, axis2) plane.
    npx          : int or (2,) pixels.
    zslice       : LOS coordinate for mode='slice'.
    """
    axis1: int
    axis2: int
    lo: np.ndarray
    hi: np.ndarray
    npx: object = 512
    mode: str = "column"
    zslice: float = 0.0

    def __post_init__(self):
        if self.axis1 == self.axis2 or self.axis1 not in (0, 1, 2) or self.axis2 not in (0, 1, 2):
            raise ValueError("axis1, axis2 must be distinct axes in {0,1,2}")
        if self.mode not in ("column", "average", "rhocolumn", "slice"):
            raise ValueError(f"unknown mode {self.mode!r}")
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(2)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(2)
        npx = self.npx
        if np.isscalar(npx):
            npx = (int(npx), int(npx))
        self.npx = tuple(int(x) for x in npx)
        if any(n <= 0 for n in self.npx):
            raise ValueError("npx must be positive")
        if np.any(self.hi <= self.lo):
            raise ValueError("hi must exceed lo on both axes")

    @classmethod
    def centered(cls, box, axis1, axis2, npx=512, mode="column", zslice=0.0):
        """Plane spanning the (axis1, axis2) extent of a centered box [-box/2,box/2]."""
        box = np.asarray(box, dtype=np.float64).reshape(-1)
        if box.size == 1:
            box = np.repeat(box, 3)
        half = 0.5 * np.array([box[axis1], box[axis2]])
        return cls(axis1=axis1, axis2=axis2, lo=-half, hi=half, npx=npx,
                   mode=mode, zslice=zslice)

    @property
    def los_axis(self):
        return ({0, 1, 2} - {self.axis1, self.axis2}).pop()

    @property
    def box(self):
        return self.hi - self.lo

    @property
    def pixwidth(self):
        return self.box / np.asarray(self.npx, dtype=np.float64)

    def edges(self):
        return [np.linspace(self.lo[a], self.hi[a], self.npx[a] + 1) for a in range(2)]

    def centres(self):
        pw = self.pixwidth
        return [self.lo[a] + (np.arange(self.npx[a]) + 0.5) * pw[a] for a in range(2)]
