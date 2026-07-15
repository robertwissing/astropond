"""Density-table capture for reconstruct-mode resampling.

The bridge between sph_interp (grid build + kernel deposit) and the glassgen
relaxation (which wants a rho0 target it can evaluate at every particle every
iteration). See RECONSTRUCT_PLAN.md.

    table = capture_density(pos, mass, box=box)          # AMR by default
    res   = relax(pos2, mass2, table, box=box, ...)      # table IS the target
    table.save('rho0.npz'); table = AMRTable.load('rho0.npz')

capture_density() computes the particle SPH density with glassgen's OWN
kernel convention (Wendland C2, same nsmooth as the relax - so the kernel
smoothing bias of the table matches the bias of the relax's density estimate
and cancels to first order in gradients), hands the particles to sph_interp,
and wraps the deposited grid as a relax-ready DensityField:

  grid='amr'     -> AMRTable: petkova-deposited (exact cell-average,
                    mass-conserving) leaf values; readout 'linear' (default,
                    trilinear blend at the containing leaf's resolution) or
                    'flat'; point query via per-level sorted linear cell
                    keys (binary search, numba parallel; ~100/300 Mq/s
                    linear/flat - as cheap as trilinear). Storage scales
                    with the LEAF count - the whole point for galaxies /
                    cosmo boxes where a uniform table blows up.
  grid='uniform' -> glassgen.density.TableDensity (trilinear 3D table).

sph_interp is located by walking up to the testsuite root (glassgen_python
stays standalone: importing this module without sph_interp raises only when
capture is actually called; AMRTable.load needs nothing but numpy/numba).
"""
import os
import re
import sys

import numpy as np

from .amrtable import AMRTable          # re-exported (moved out of this module)
from .density import TableDensity
from .envelope import (_fill_empty, _pad_periodic_box, _with_values,
                       edge_floor_spacing, frame_open_box, table_cells)
from .kernels import W0, wzero_wc2
from .neighbors import NeighborList
from .sph_core import density_loop


def _find_setupfiles():
    """Locate <testsuite root>/setupfiles via the fixed layout (walk up from
    this package). None if absent."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, 'setupfiles')
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _import_sph_interp():
    """Import sph_interp, locating it via the fixed testsuite layout
    (<root>/setupfiles/sph_interp vs <root>/icgenerator/glassgen_python)."""
    try:
        import sph_interp
        return sph_interp
    except ImportError:
        pass
    setup = _find_setupfiles()
    if setup and os.path.exists(os.path.join(setup, 'sph_interp',
                                             '__init__.py')):
        sys.path.insert(0, setup)
        import sph_interp
        return sph_interp
    raise ImportError(
        'sph_interp not found (no setupfiles/sph_interp above '
        f'{os.path.dirname(os.path.abspath(__file__))}); pass a '
        'precomputed table instead (AMRTable.load / TableDensity)')


def _import_readtipsy():
    setup = _find_setupfiles()
    if setup is None or not os.path.exists(os.path.join(setup,
                                                        'readtipsy.py')):
        raise ImportError('readtipsy.py not found (no setupfiles above '
                          f'{os.path.dirname(os.path.abspath(__file__))})')
    if setup not in sys.path:
        sys.path.insert(0, setup)
    import readtipsy
    return readtipsy


def particle_density(pos, mass, box=None, nsmooth=64):
    """SPH density + smoothing length with glassgen's own convention
    (kNN h, Wendland C2, wzero self-correction) - the SAME estimate the
    relax uses, so table and relax biases cancel. Returns (rho, h)."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    periodic = box is not None
    boxarr = np.ones(3) if box is None else np.asarray(box, dtype=np.float64)
    nl = NeighborList(boxarr if periodic else None, nsmooth,
                      build_transpose=False)
    idx, h, _, _ = nl.update(pos)
    rho = np.empty(len(pos))
    density_loop(pos, mass, h, idx, boxarr, periodic, rho,
                 W0 * wzero_wc2(nsmooth))
    return rho, h


def capture_density(pos, mass, box=None, grid='amr', nsmooth=64,
                    method='petkova', base_npx=None, max_level=16,
                    part_per_cell=8, npx=None, empty='decay',
                    decay_cells=2.0, floor_frac=1e-3, interp='vertex',
                    contour_n=None, contour_band=8.0, contour_beta=0.5,
                    contour_passes=2, grade=True):
    """Capture the particle density field on a 3D grid, relax-ready.

    pos, mass : the particles whose density defines the target (capture the
                INPUT snapshot BEFORE splitting/merging).
    box       : (3,) periodic box or None (open: domain = particle extent).
    grid      : 'amr' (default) -> AMRTable; 'uniform' -> TableDensity.
    nsmooth   : neighbour count for the particle density estimate; keep it
                equal to the relax's nsmooth (bias cancellation).
    method    : 'petkova' (default: EXACT kernel-volume integral -> true
                cell-averaged density, mass-conserving by construction - the
                right deposit for a table), 'sph-gather' (kernel estimate at
                cell centres; fast, approximate) or 'sph-scatter'
                (conservative mass deposit; table value = deposited mass /
                cell volume).
    base_npx  : AMR base cells on the LONGEST axis (scalar: the other axes get
                proportionally fewer so cells stay ~CUBIC on a non-cubic
                domain - plan task M3), or an explicit (3,) per-axis tuple.
                Default: ~N^(1/3)/8 rounded to a power of two (>= 8) -
                coarse; refinement supplies locality.
    max_level : AMR refinement depth cap (leaf = base / 2^level).
    part_per_cell : refine while a cell holds more than this many particles
                (refine_count). ~8 keeps the table coarser than the particle
                sampling everywhere - finer would encode particle noise as
                target structure (see plan, design decision 2).
    npx       : uniform-grid cells on the longest axis (scalar -> ~cubic
                cells, as base_npx) or (3,). Default: ~N^(1/3)/2, power of 2.
    empty     : policy for non-positive cells (empty space) - a rho0 of 0
                is a singularity for the relax pseudo-pressure
                pP=(rho/rho0)^2, and its log-compressed step normalization
                means ONE particle in a near-zero cell can crush every
                other particle's step (global stall):
                'edge'  - fill each empty cell with the value of the
                    nearest occupied cell (periodic-aware): CONSTANT
                    extrapolation of the gas edge. Simple, but on an
                    OPEN-box galaxy it wraps the gas in a constant
                    edge-density skirt the sparse escaped/edge particles
                    can never match -> a persistent irreducible relax force
                    that stalls the step controller (observed on the
                    isolated-galaxy IC, denserr floor ~0.24).
                'decay' (default) - fill each empty cell with its nearest
                    occupied value DECAYED toward a floor,
                    rho0 = max(floor, rho_edge * exp(-d / lambda)), where
                    d is the periodic-aware distance to that occupied cell
                    and lambda = decay_cells * (the empty cell's own
                    width). This is the accretion-disk pkdUpdateDensity
                    trick (gasoline, ICdensprofile>=4): the low-density
                    region is a smooth taper from the resolved edge down to
                    a floor, not a constant, so a sparse particle one cell
                    out feels a target near its actual density and the
                    irreducible edge force vanishes. floor = floor_frac x
                    median(occupied) bounds the rho/rho0 dynamic range away
                    from the rho0=0 singularity.
                'min'  - floor at the smallest positive cell value (can be
                    a ~1e-12-of-peak kernel tail on a galaxy table: keeps
                    positivity but not the dynamic range).
                'zero' - keep zeros (table is then NOT relax-safe).
    decay_cells : 'decay' fill length in units of the empty cell's width
                (lambda = decay_cells x cell width). Larger -> the edge
                density reaches further into the vacuum before tapering.
    floor_frac : 'decay'/vacuum floor as a fraction of the median occupied
                density (bounds the dynamic range; keeps rho0 > 0).
    interp    : AMR readout mode: 'vertex' (default since 2026-07-13, user
                decision - conforming corner trilinear, continuous across
                level jumps, tracks sub-cell gradients; fused, costs ~
                linear), 'linear', 'flat' or 'limited' (see AMRTable).
    """
    si, parts, n, boxarr = _build_parts(pos, mass, box, nsmooth)

    if grid == 'uniform':
        if npx is None:
            npx = _pow2(max(n, 64) ** (1.0 / 3.0) / 2.0, lo=4)
        if np.isscalar(npx):
            # ~cubic cells on a non-cubic domain (plan task M3, as in the AMR
            # path): the requested count applies to the longest axis.
            span = (boxarr if boxarr is not None
                    else np.asarray(pos.max(0) - pos.min(0), dtype=np.float64))
            cell = float(span.max()) / int(npx)
            npx = tuple(max(1, int(round(span[i] / cell))) for i in range(3))
        if boxarr is not None:
            tgt = si.UniformGrid.centered(boxarr, npx)
        else:
            tgt = si.UniformGrid(lo=pos.min(0), hi=pos.max(0), npx=npx)
        res = si.interpolate(parts, tgt, method=method, values=['rho'])
        vals = _grid_rho(res, tgt.cell_volume, method)
        ax = tgt.centres()
        cx, cy, cz = np.meshgrid(*ax, indexing='ij')
        centres = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
        cell_len = float(np.cbrt(np.prod(
            [np.diff(a).mean() if len(a) > 1 else 1.0 for a in ax])))
        sizes = np.full(len(centres), cell_len)
        vals = _fill_empty(vals.reshape(-1), empty, centres, boxarr,
                           sizes=sizes, decay_cells=decay_cells,
                           floor_frac=floor_frac).reshape(vals.shape)
        return TableDensity(ax, vals)

    if grid != 'amr':
        raise ValueError(f"unknown grid {grid!r} (use 'amr' or 'uniform')")
    contour = None
    if contour_n:
        m_out = float(np.asarray(mass, dtype=np.float64).sum()) / int(contour_n)
        re_est = rho_edge_particles(parts.rho, mass, m_out, c=nsmooth)
        contour = dict(rho_lo=re_est / contour_band,
                       rho_hi=re_est * contour_band, m_out=m_out,
                       beta=contour_beta, band=contour_band,
                       passes=contour_passes, c=nsmooth)
    amr, vals_raw = _capture_amr_grid(si, parts, n, method, base_npx,
                                      max_level, part_per_cell,
                                      contour=contour, grade=grade)
    vols = amr.volumes()
    vals = _fill_empty(vals_raw, empty, amr.centres(), boxarr,
                       sizes=np.cbrt(vols), decay_cells=decay_cells,
                       floor_frac=floor_frac)
    return AMRTable.from_grid(amr, vals, interp=interp)


def _build_parts(pos, mass, box, nsmooth):
    """Shared capture preamble: import sph_interp, compute the particle SPH
    density/h with glassgen's own convention, wrap as sph_interp Particles.
    Returns (sph_interp module, Particles, n, boxarr)."""
    si = _import_sph_interp()
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    rho, h = particle_density(pos, mass, box, nsmooth)
    boxarr = None if box is None else np.asarray(box, dtype=np.float64)
    parts = si.Particles(pos=pos, mass=mass, rho=rho, h=h,
                         values={'rho': rho}, box=boxarr,
                         periodic=box is not None)
    return si, parts, len(pos), boxarr


def rho_edge_particles(rho, mass, m_out, c=64):
    """Particle-field estimate of the envelope's ``rho_edge`` BEFORE any
    grid exists (the capture-side twin of :func:`envelope.rho_edge`, same
    mass integral on the particle SPH densities instead of cell values):
    sort particles by density and cut the cumulative faint-end mass at
    ``c*m_out``. Used to aim the contour-refinement band; the envelope
    still derives its own rho_edge from the finished table."""
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    order = np.argsort(rho, kind='stable')
    cum = np.cumsum(mass[order])
    idx = min(int(np.searchsorted(cum, c * float(m_out), side='left')),
              len(order) - 1)
    return float(rho[order[idx]])


def _capture_amr_grid(si, parts, n, method='sph-gather', base_npx=None,
                      max_level=16, part_per_cell=8, contour=None,
                      grade=False):
    """Build the AMR grid and deposit the per-leaf density BEFORE any
    empty-cell fill. Returns (AMRGrid, vals_raw) - vals_raw is the raw
    deposited cell density (0 in empty cells); the N-aware envelope
    (envelope.build_table_envelope) derives its own rho_edge/rho_floor from it.

    Refinement is CRITERION-driven: refine_count(part_per_cell) splits any cell
    with > part_per_cell (8) particles, so leaves hold <= 8 particles and the
    process self-terminates - leaf count stays below N (occupied leaves carry
    several particles each; ~338k leaves for ~800k particles on the isolated
    galaxy). part_per_cell is fixed at 8 (enough particles for a meaningful cell
    average). max_level is only a finite BACKSTOP for the degenerate case the
    criterion cannot resolve - > 8 coincident particles (same point / closer
    than the finest cell) that never fall to <= 8 no matter how deep. That case
    costs only O(max_level) (one refining chain), so the cap is set generous
    (16 -> box/16/2**16 ~ 1e6 cells/axis) and is essentially never reached on
    real data.

    contour : optional (rho_lo, rho_hi, m_out, beta) - plan task M10: also
    refine occupied cells whose mean density straddles the classification
    band [rho_lo, rho_hi] while larger than the local output sampling scale
    beta*(m_out/rho)**(1/3) (sph_interp.refine_band). Resolves the
    rho_edge contour so the flat occupancy verdict is not quantized on
    coarse straddling cells (the rectangular cull bites / phantom ramps).

    grade : enforce 2:1 balance on the finished leaf set (sph_interp.
    grade_2to1: no leaf touches a leaf more than one level coarser, level
    jumps decay geometrically away from refined regions) and redeposit.
    Count-driven refinement leaves the razor-thin galaxy disk's max-level
    cells abutting level-0/1 vacuum cells; any readout across that jump
    stretches midplane values over the whole coarse width (the 2026-07-12
    z-staircase: plateaus decades above the SPH truth across |z|<4)."""
    if base_npx is None:
        base_npx = _pow2(max(n, 64) ** (1.0 / 3.0) / 8.0, lo=8)
    if np.isscalar(base_npx):
        # ~CUBIC cells on a non-cubic domain (plan task M3): the requested
        # count applies to the LONGEST axis and the others get proportionally
        # fewer cells. A scalar count on all three axes would make the cell
        # aspect track the domain aspect at EVERY octree level (children
        # inherit it): the isolated galaxy captured 8:1 cells - z oversampled
        # ~8x per level (leaf waste, and extra vertical target sharpness the
        # N-particle kernel cannot represent) while x/y stayed marginal, the
        # linear-readout gradients direction-biased, and the envelope decay
        # length (cbrt(vol)) a mix of both scales.
        span = (np.asarray(parts.box, dtype=np.float64).reshape(3)
                if parts.box is not None
                else parts.pos.max(0) - parts.pos.min(0))
        cell = float(span.max()) / int(base_npx)
        base_npx = tuple(max(1, int(round(span[i] / cell))) for i in range(3))
    crit = si.refine_count(part_per_cell)
    if contour is not None:
        rho_lo, rho_hi, m_out, beta = (contour['rho_lo'], contour['rho_hi'],
                                       contour['m_out'], contour['beta'])
        crit = si.refine_any(crit, si.refine_band(rho_lo, rho_hi, m_out,
                                                  beta=beta))
    amr = si.AMRGrid.build(parts, base_npx=base_npx, max_level=max_level,
                           criterion=crit)
    res = si.interpolate(parts, amr, method=method, values=['rho'])
    vals_raw = _grid_rho(res, amr.volumes(), method)
    if contour is not None:
        # correction passes (plan task M10, two-pass design): the in-flight
        # band criterion only sees particle-CENTRE mass, but the verdict
        # consumes the DEPOSITED values - kernel-spill cells (nonzero petkova
        # value, zero centres) straddle the band invisibly. Re-flag on the
        # deposited values with the cell-based rho_edge (the envelope's own
        # cut, no estimator), rebuild refining the flagged regions
        # (refine_empty=True: spill cells hold no centres), redeposit;
        # iterate until clean or passes exhausted.
        from .envelope import rho_edge as _rho_edge_cells
        for _ in range(int(contour.get('passes', 2))):
            re_c = _rho_edge_cells(vals_raw, amr.volumes(), contour['m_out'],
                                   c=contour.get('c', 64))
            width = amr.cell_size.max(axis=1)
            v = np.asarray(vals_raw, dtype=np.float64)
            floor = beta * np.cbrt(contour['m_out'] / np.maximum(v, 1e-300))
            sel = ((v >= re_c / contour['band'])
                   & (v <= re_c * contour['band'])
                   & (width > floor * (1.0 + 1e-9))
                   & (amr.cell_level < max_level))
            if not sel.any():
                break
            region = _refine_regions_criterion(
                amr.cell_lo[sel] + 0.5 * amr.cell_size[sel],
                0.5 * amr.cell_size[sel], floor[sel], amr,
                np.where(sel, floor, 0.0))
            amr = si.AMRGrid.build(parts, base_npx=amr.base_npx,
                                   max_level=max_level, lo=amr.lo, hi=amr.hi,
                                   criterion=si.refine_any(crit, region),
                                   refine_empty=True)
            res = si.interpolate(parts, amr, method=method, values=['rho'])
            vals_raw = _grid_rho(res, amr.volumes(), method)
    if grade:
        graded = si.grade_2to1(amr, parts)
        if graded is not amr:
            amr = graded
            res = si.interpolate(parts, amr, method=method, values=['rho'])
            vals_raw = _grid_rho(res, amr.volumes(), method)
    return amr, vals_raw


def _refine_regions_criterion(centres, halfs, floors, amr, floor_leaf_vals):
    """Criterion refining every cell that overlaps a flagged leaf box, down
    to that box's own floor size (M10 correction pass). Overlap on the
    shared dyadic tiling has only two cases: the cell nests inside a flagged
    leaf (exact point-in-leaf lookup via a flat AMRTable holding the
    per-leaf floors) or flagged boxes nest inside the cell (cKDTree ball
    query on the flagged centres). Self-terminating: refinement stops at
    the flag's floor, so refine_empty=True is safe."""
    from scipy.spatial import cKDTree
    ftab = AMRTable.from_grid(amr, floor_leaf_vals, interp='flat')
    tree = cKDTree(centres)
    min_flag_w = float((2.0 * halfs).max(axis=1).min())

    def crit(count, mass, hmin, size, level, lo):
        ctr = lo + 0.5 * size
        w = size.max(axis=1)
        # case A: cell inside a flagged leaf -> refine while above its floor
        fl = ftab.rho0(ctr)
        out = (fl > 0.0) & (w > fl * (1.0 + 1e-9))
        # case B: flagged leaves inside the (coarser) cell -> must reach them
        big = w > min_flag_w * (1.0 - 1e-9)
        if big.any():
            r = 0.5 * np.linalg.norm(size[big], axis=1) * (1.0 + 1e-9)
            hits = tree.query_ball_point(ctr[big], r)
            bidx = np.where(big)[0]
            for k, hh in zip(bidx, hits):
                if out[k] or not hh:
                    continue
                inside = np.all(np.abs(centres[hh] - ctr[k])
                                <= 0.5 * size[k] + 1e-12, axis=1)
                out[k] = bool(inside.any())
        return out

    crit.needs_lo = True
    return crit


def capture_amr_raw(pos, mass, box=None, nsmooth=64, method='petkova',
                    base_npx=None, max_level=16, part_per_cell=8,
                    contour_n=None, contour_band=8.0, contour_beta=0.5,
                    contour_passes=2, grade=True):
    """Raw AMR capture: the space-filling leaf set + per-leaf density BEFORE
    any empty-cell fill (unlike capture_density, which returns a finished,
    filled table). `AMRTable.from_grid(amr, vals_raw)` makes it the EXACT
    build target. Returns (AMRGrid, vals_raw).

    contour_n : target OUTPUT particle count -> contour-resolving refinement
    (plan task M10): estimate rho_edge(N) from the particle field
    (:func:`rho_edge_particles`) and additionally refine occupied cells in
    the band [rho_edge/contour_band, rho_edge*contour_band] down to
    contour_beta*(m_out/rho)**(1/3) (the local output spacing). None = the
    pure part_per_cell capture (pre-M10 behaviour).

    grade : 2:1-balance the finished leaf set + redeposit (see
    :func:`_capture_amr_grid`)."""
    si, parts, n, _ = _build_parts(pos, mass, box, nsmooth)
    contour = None
    if contour_n:
        m_out = float(np.asarray(mass, dtype=np.float64).sum()) / int(contour_n)
        re_est = rho_edge_particles(parts.rho, mass, m_out, c=nsmooth)
        contour = dict(rho_lo=re_est / contour_band,
                       rho_hi=re_est * contour_band, m_out=m_out,
                       beta=contour_beta, band=contour_band,
                       passes=contour_passes, c=nsmooth)
    return _capture_amr_grid(si, parts, n, method=method, base_npx=base_npx,
                             max_level=max_level, part_per_cell=part_per_cell,
                             contour=contour, grade=grade)


def reconstruct_resample(pos_in, mass_in, count, fields, box=None, nsmooth=64,
                         method='petkova', base_npx=None, max_level=16,
                         part_per_cell=8, relax_kw=None, seed=0, progress=None,
                         amr_cap=None, pad_frac=0.1, cascade=True,
                         bg_c=None, bg_k=2.0, bg_tail=False, kernel_match=True,
                         contour_refine=True, contour_band=8.0,
                         contour_beta=0.5, target_interp='vertex',
                         grade=True, shell_start='local', decay_cells=2.0,
                         scaffold='exp', alpha=None, decay_spacing=None,
                         decay_local=False, bg_mode='lattice',
                         flat_floor=False, tail_reach=6.0,
                         km_hmode='particle', km_hscale=1.0, verbose=False):
    """The `glassresample --op reconstruct` engine - a THIN WRAPPER around
    the glassgen table builder (decided 2026-07-10): capture the input gas
    density (exact raw AMR table), hand it to
    :func:`glassgen.progressive.build_from_table` (sample `count` positions
    from the table + N-aware envelope + progressive relax + spatial cull /
    renorm to the input gas mass), then gather per-survivor fields from the
    input at the FINAL positions.

      pos_in, mass_in : the INPUT gas.
      count           : target gas particle count (may be < or > the input).
      fields          : dict name -> (Nin,)/(Nin,3) input gas fields, gathered
                        onto the survivors (the glass has no history to inherit).
      cascade         : False -> single full-res relax of the placed ensemble
                        (the --no-progressive fallback), same envelope + cull.
      bg_c, bg_k      : envelope knobs (rho_edge cut count; rho_floor=re/k**3).
      scaffold, alpha : 'exp' (M12 taper + lattice) or 'cubic' (2026-07-13
                        resolvability scaffold, lattice retired; alpha =
                        the shared cascade constant, default 2.0).
      bg_tail         : backed-tail experiment (2026-07-11): background over
                        the whole frame down to rho_frame, frame cap grows
                        ~k on capped axes (envelope.build_table_envelope).
      contour_refine  : M10 contour-resolving capture (default ON): refine
                        cells straddling the rho_edge(count) band so the
                        flat occupancy verdict is not cell-quantized
                        (capture_amr_raw contour_n=count). Ignored when
                        amr_cap is passed (the table is already built).
      amr_cap         : optional precomputed (amr, vals_raw) to skip capture.

    Returns (pos, mass, aux, is_filler, audit); final count is the post-cull
    occupancy, near but not exactly `count`. is_filler is all-False (survivors
    are reconstructed gas; the background was culled). audit['timings'] holds
    the per-stage wall clock (capture / src_rho / build / gather, seconds) and
    audit['levels'] the per-relax-level (name, n, niter, wall) breakdown;
    verbose=True additionally prints relax's own per-phase profile
    (neighbours/density/pP/force/step) at the end of every level."""
    import time as _time
    from .progressive import build_from_table
    pos_in = np.ascontiguousarray(pos_in, dtype=np.float64)
    mass_in = np.ascontiguousarray(mass_in, dtype=np.float64)
    timings = {}
    # OPEN IC (box is None): capture in WORLD coordinates on the raw gas bbox
    # (open kernels, like glasstable) - the periodic frame for the relax is
    # sized AFTER capture by the envelope (build_table_envelope frame_box:
    # per-axis margins + the kernel-fits-the-box cap), because its Delta_bg
    # needs rho_edge(N) from the table (plan task M2). The old order (frame
    # first at a fixed pad_frac, capture in the padded box) fixed the z
    # period before Delta_bg was known: the 80k galaxy run's background
    # kernels wrapped its box_z 2.6x. relax_progressive frames/unframes, so
    # everything here stays in the IC's own coordinates.
    t0 = _time.perf_counter()
    if amr_cap is None:
        amr, vals_raw = capture_amr_raw(
            pos_in, mass_in, box=box, nsmooth=nsmooth, method=method,
            base_npx=base_npx, max_level=max_level, part_per_cell=part_per_cell,
            contour_n=(count if contour_refine else None),
            contour_band=contour_band, contour_beta=contour_beta, grade=grade)
    else:
        amr, vals_raw = amr_cap
    timings['capture'] = _time.perf_counter() - t0
    M_gas = float(mass_in.sum())
    # target readout: 'linear' (trilinear) OVERSHOOTS across coarse->fine
    # level jumps (the razor-thin disk vertical-smear defect, 2026-07-12);
    # 'flat' (PWC cell average) has no overshoot. The shell/occupancy is
    # built flat regardless (build_table_envelope); this sets the OCCUPIED
    # BASE readout the relax fills.
    table = AMRTable.from_grid(amr, vals_raw, interp=target_interp)
    # per-survivor fields: gather from the input at the FINAL positions
    t0 = _time.perf_counter()
    src_rho, _ = particle_density(pos_in, mass_in, box=box, nsmooth=nsmooth)
    timings['src_rho'] = _time.perf_counter() - t0

    def _hook(final_pos):
        t = _time.perf_counter()
        out = gather_fields(final_pos, pos_in, mass_in, src_rho, fields,
                            box=box, nsmooth=nsmooth)
        timings['gather'] = _time.perf_counter() - t
        return out

    rkw = dict(relax_kw or {})
    if progress is not None:
        rkw.setdefault('callback', progress)
    t0 = _time.perf_counter()
    res = build_from_table(table, count, box=box, gas_mass=M_gas,
                           nsmooth=nsmooth, bg_c=bg_c, bg_k=bg_k,
                           bg_tail=bg_tail,
                           envelope='on', cascade=cascade, seed=seed,
                           open_pad=pad_frac, kernel_match=kernel_match,
                           shell_start=shell_start, decay_cells=decay_cells,
                           scaffold=scaffold, alpha=alpha,
                           decay_spacing=decay_spacing,
                           decay_local=decay_local, bg_mode=bg_mode,
                           flat_floor=flat_floor, tail_reach=tail_reach,
                           km_hmode=km_hmode, km_hscale=km_hscale,
                           filler_field_hook=_hook, verbose=verbose, **rkw)
    timings['build'] = _time.perf_counter() - t0
    pos_f = res.pos
    aux_f = res.aux
    a = dict(res.envelope_audit)
    mass_f = np.full(len(pos_f), a['mf'])
    is_filler = np.zeros(len(pos_f), dtype=bool)
    a.update({k: v for k, v in res.table_build.items()
              if k in ('rho_edge', 'rho_frame', 'delta_bg', 'm',
                       'n_bg_tail', 'scaffold', 'alpha')},
             relax_iters=int(res.niter), relax_stop=str(res.stop_reason),
             timings=timings,
             levels=[dict(name=lv.get('name'), n=lv.get('n'),
                          niter=lv.get('niter'), wall=lv.get('wall'))
                     for lv in getattr(res, 'levels', [])])
    return pos_f, mass_f, aux_f, is_filler, a


def gather_fields(qpos, src_pos, src_mass, src_rho, fields, box=None,
                  nsmooth=64):
    """SPH-gather fields from source particles onto query positions,
    Shepard-normalized so a constant field is reproduced exactly:
        f(x) = sum_j w_j f_j / sum_j w_j,  w_j = (m_j/rho_j) W(r_j/H)
    with H = the nsmooth-th source-neighbour distance and W a Wendland-C2
    shape. `fields` maps name -> (Nsrc,) or (Nsrc,ncomp); returns name ->
    (Nq,) / (Nq,ncomp). Gives relaxed / filler particles physical
    vel/u/B/temperature from the INPUT snapshot (tasks 8, 9, 11)."""
    from scipy.spatial import cKDTree
    qpos = np.ascontiguousarray(qpos, dtype=np.float64).reshape(-1, 3)
    src_pos = np.ascontiguousarray(src_pos, dtype=np.float64).reshape(-1, 3)
    src_mass = np.asarray(src_mass, dtype=np.float64)
    src_rho = np.asarray(src_rho, dtype=np.float64)
    k = min(nsmooth, len(src_pos))
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        lo = np.minimum(qpos.min(0), src_pos.min(0))
        tree = cKDTree(np.mod(src_pos - lo, box), boxsize=box)
        dist, nn = tree.query(np.mod(qpos - lo, box), k=k)
    else:
        tree = cKDTree(src_pos)
        dist, nn = tree.query(qpos, k=k)
    if k == 1:
        dist, nn = dist[:, None], nn[:, None]
    H = dist[:, -1:]
    H = np.where(H > 0.0, H, 1.0)
    q = np.clip(dist / H, 0.0, 1.0)
    W = (1.0 - q) ** 4 * (1.0 + 4.0 * q)          # Wendland C2 (shape only)
    wj = (src_mass / src_rho)[nn] * W             # (Nq, k)
    wsum = wj.sum(1)
    wsum = np.where(wsum > 0.0, wsum, 1.0)
    out = {}
    for name, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            out[name] = (wj * arr[nn]).sum(1) / wsum
        else:
            out[name] = (wj[:, :, None] * arr[nn]).sum(1) / wsum[:, None]
    return out


def _pow2(x, lo=4):
    return max(lo, 2 ** int(round(np.log2(max(x, 1.0)))))


def save_table(table, path):
    """Save a captured table (AMRTable or 3D uniform TableDensity) as .npz.
    Both formats load back through load_table() without sph_interp."""
    if isinstance(table, AMRTable):
        table.save(path)
    elif isinstance(table, TableDensity):
        if table.dim != 3:
            raise ValueError(f'expected a 3D table, got dim={table.dim}')
        np.savez(path, ax0=table.axes[0], ax1=table.axes[1],
                 ax2=table.axes[2], values=table.values)
    else:
        raise TypeError(f'cannot save table of type {type(table).__name__}')


def load_table(path):
    """Load a table written by save_table() (or AMRTable.save): sniffs the
    .npz keys - AMR block layout -> AMRTable, axes+values -> TableDensity."""
    with np.load(path) as d:
        files = set(d.files)
        if 'keys' in files:
            pass                        # AMR layout: reload via the class
        elif 'values' in files:
            return TableDensity((d['ax0'], d['ax1'], d['ax2']), d['values'])
        else:
            raise ValueError(
                f'{path}: not a density table (.npz needs either the '
                'AMRTable block layout or ax0/ax1/ax2 + values)')
    return AMRTable.load(path)


def describe_table(table):
    """One-line human description for CLI logging."""
    if isinstance(table, AMRTable):
        nlev = int((np.diff(table.offsets) > 0).sum())
        return (f'AMR, {table.ncell} leaves, {nlev} levels, '
                f'base {"x".join(str(d) for d in table.dims[0])}, '
                f'{table.interp} readout')
    if isinstance(table, TableDensity):
        return 'uniform ' + 'x'.join(str(len(a)) for a in table.axes)
    return type(table).__name__


def _grid_rho(res, volumes, method):
    """Cell density from a GridResult: petkova/gather expose it as the
    'rho' field (petkova = exact mass-conserving cell average); scatter =
    accumulated mass (res.norm) / cell volume (conservative)."""
    if method == 'sph-scatter':
        return np.asarray(res.norm, dtype=np.float64) / volumes
    return np.asarray(res.data['rho'], dtype=np.float64)


def apply_count_overlay(table, count, c=64, k=2.0, decay_cells=2.0,
                        shell_start='local', scaffold='exp', alpha=None,
                        decay_spacing=None, decay_local=False,
                        flat_floor=False, tail_reach=6.0):
    """The N-aware envelope fill on a captured table: returns
    ``(filled_table, info)`` - a NEW table (the input is untouched). Uses the
    SAME rho_edge scheme as the glassgen builder
    (:func:`envelope.edge_floor_spacing` + the decay ``_fill_empty`` with
    ``floor=rho_edge``): from ``m = M_gas / count``, every cell below
    ``rho_edge`` is emptied and refilled - with ``flat_floor=False`` (the M18
    default) as ONE strictly-decreasing profile down to ``rho_frame =
    rho_edge/k**6`` at the frame edge (frame sized by the same M2 rule the
    builder uses; rho_floor retired, two-rung ladder); cells above keep their
    exact values. Different N -> different overlay on the
    SAME exact table (larger N -> lower rho_edge/floor). This is the density
    target the relax sees at that count (the leaf-set view of the builder's
    PaddedDensity shell; the analytic outside-domain tail continuation is
    not representable in the leaf set and is omitted here).

    ``decay_spacing`` (eta) / ``decay_local`` mirror
    :func:`envelope.build_table_envelope` exactly: eta puts the taper
    lambda in OUTPUT-spacing units (``lambda = eta*(m/rho_edge)**(1/3)``,
    N-invariant reach, overrides ``decay_cells``); ``decay_local``
    evaluates it at the launch cell's own density (floored at rho_edge,
    the M16 skirt). ``bg_mode='sample'`` has no table-side effect (it
    changes the builder's background PARTICLES, not the density target),
    so it has no counterpart here.

    ``flat_floor=False`` (+ ``tail_reach`` steepness, plan M18 2026-07-14)
    retires the flat rho_floor shelf: the 'decay' fill becomes ONE
    strictly-decreasing profile down to rho_frame at the frame corner
    (mirrors :func:`envelope.build_table_envelope`), so glasstable renders
    the true de-flattened target. Default True = unchanged."""
    if decay_local and decay_spacing is None:
        raise ValueError('decay_local needs decay_spacing (the eta taper); '
                         'the decay_cells taper is capture-cell based')
    centres, sizes, _, vals = table_cells(table)
    vals = np.asarray(vals, dtype=np.float64)
    vols = sizes.prod(1)
    m = float((vals * vols).sum()) / int(count)
    re, delta_bg = edge_floor_spacing(vals, vols, m, c=c, k=k)
    fbox = table.box if getattr(table, 'periodic', False) is True \
        or (hasattr(table, 'periodic') and bool(np.ravel(table.periodic)[0])) \
        else None
    # frame extent for the M6 tail scale, sized as the builder does (M2 rule;
    # taper reach from the gas-edge cell size, nsmooth ~ c)
    if fbox is None:
        cell_edge = np.cbrt(vols[(vals >= re) & (vals <= 10.0 * re)])
        if cell_edge.size == 0:
            cell_edge = np.cbrt(vols[vals >= re])
        cell0 = float(np.median(cell_edge)) if cell_edge.size else 0.0
        lam0 = (decay_spacing * (m / re) ** (1.0 / 3.0)
                if decay_spacing is not None else decay_cells * cell0)
        taper_len = lam0 * 3.0 * max(
            np.log(max(k, 1.0 + 1e-9)), 0.1) + cell0
        dom_lo = (centres - 0.5 * sizes).min(0)
        dom_hi = (centres + 0.5 * sizes).max(0)
        frame_extent, _ = frame_open_box(dom_lo - taper_len,
                                         dom_hi + taper_len, delta_bg,
                                         nsmooth=c, taper_len=taper_len, k=k)
    else:
        frame_extent = np.asarray(fbox, dtype=np.float64).reshape(3)
    rho_frame = re / k ** 6               # two-rung ladder: rho_edge -> rho_frame
    if shell_start not in ('edge', 'local'):
        raise ValueError(f"unknown shell_start {shell_start!r} "
                         "(use 'edge' or 'local')")
    if scaffold not in ('exp', 'cubic'):
        raise ValueError(f"unknown scaffold {scaffold!r} "
                         "(use 'exp' or 'cubic')")
    if scaffold == 'cubic':
        from .envelope import ALPHA_DEFAULT, _fill_cubic
        if alpha is None:
            alpha = ALPHA_DEFAULT
        filled = _fill_cubic(vals.copy(), centres, fbox, m, alpha, re,
                             fold=k ** 3,
                             tail_scale=0.5 * float(np.max(frame_extent)),
                             tail_floor=rho_frame)
        n_emptied = int((vals < re).sum())
    else:
        # taper length, matching build_table_envelope exactly: eta puts
        # lambda in OUTPUT-spacing units (N-invariant), overriding the
        # capture-cell decay_cells; decay_local takes lambda from each launch
        # cell's OWN density (floored at rho_edge, the M16 skirt).
        if decay_spacing is not None:
            if decay_local:
                fill_sizes = decay_spacing * (
                    m / np.maximum(vals, re)) ** (1.0 / 3.0)
            else:
                fill_sizes = np.full(len(vals),
                                     decay_spacing * (m / re) ** (1.0 / 3.0))
            fill_dc = 1.0
        else:
            fill_sizes = np.cbrt(vols)
            fill_dc = decay_cells
        filled = _fill_empty(vals.copy(), 'decay', centres, fbox,
                             sizes=fill_sizes, decay_cells=fill_dc,
                             floor=re,
                             cap=(re if shell_start == 'edge' else None),
                             tail_scale=0.5 * float(np.max(frame_extent)),
                             tail_floor=rho_frame,
                             flat_floor=flat_floor,
                             tail_reach=tail_reach)
        n_emptied = int((vals < re).sum())
    occ = float((vals >= re).mean())
    info = dict(m=m, rho_edge=re, rho_frame=rho_frame,
                delta_bg=delta_bg, occ_frac=occ, scaffold=scaffold,
                flat_floor=flat_floor, tail_reach=tail_reach,
                n_emptied=n_emptied, n_cells=len(vals))
    return _with_values(table, filled), info


def table_to_tipsy(table, out, m=None):
    """Write a density table as a gas-only tipsy snapshot `${out}.00000`
    (one gas particle per cell) so it can be viewed with the usual tipsy
    tooling: pos = cell centre, mass = rho * cell volume (mass-weighted
    renders are physical), rho column = the table density (what the relax
    reads as rho0), temp column = refinement level (0 for uniform tables),
    and col 12 (phi) = half the cell width (the per-level display scale).
    eps (col 10) = SPLASH's smoothing length h, set to the CELL WIDTH
    (`EPS_HFAC * sizes.max(1)`): tied to each leaf's own size so SPLASH's ~2h
    kernel spans the cell and blends with its neighbours, erasing the AMR grid
    in the render (fine cells stay sharp in the disk, coarse cells smooth the
    vacuum). Raise EPS_HFAC for heavier smoothing. NB empty-space cells carry
    the edge-fill value (relax safety), so the exported total mass slightly
    exceeds the gas mass - the constant-density skirt around the gas IS the
    fill. `m` is accepted for API compatibility but unused."""
    tip = _import_readtipsy()
    centres, sizes, levels, vals = table_cells(table)
    n = len(vals)
    g = np.zeros((n, 12), dtype=np.float32)
    g[:, 0] = vals * sizes.prod(1)
    g[:, 1:4] = centres
    g[:, 7] = vals
    g[:, 8] = levels
    EPS_HFAC = 1.0                       # SPLASH h = EPS_HFAC * cell width
    g[:, 9] = EPS_HFAC * sizes.max(1)    # col 10 (eps/h): grid-adaptive smoothing
    g[:, 11] = 0.5 * sizes.max(1)        # col 12 (phi): half cell width
    outsnap = f'{out}.00000'
    tip.writetipsy(g, np.zeros((0, 9), np.float32),
                   np.zeros((0, 11), np.float32), outsnap,
                   np.array([n, 3, n, 0, 0, 0], dtype=np.int32))
    return outsnap


def _export_table(table, out, what):
    """Write a table's tipsy render `${out}.00000` + a summary line."""
    outsnap = table_to_tipsy(table, out)
    _, _, levels, vals = table_cells(table)
    lv, cnt = np.unique(levels, return_counts=True)
    print(f'wrote {outsnap} ({what}, {len(vals)} cells as gas particles; '
          'cells per level: '
          + ', '.join(f'{a}: {b}' for a, b in zip(lv, cnt)) + ')', flush=True)
    return outsnap


def main(argv=None):
    """CLI (the `glasstable` executable): density-TABLE GENERATOR.

    On a snapshot, capture the RAW gas density table (exact petkova cell
    averages, genuine zeros in vacuum) and write BOTH the table and its
    tipsy render:

        ${out}_raw.npz  +  ${out}_raw.00000

    then apply the N-aware envelope fill - the SAME rho_edge/rho_floor
    logic the glassgen builder relaxes against - for --count N (default:
    the snapshot's gas count) and write the filled pair:

        ${out}_n${count}.npz  +  ${out}_n${count}.00000

    so the density render of the interpolated table AT THAT COUNT can be
    inspected before building. --raw-only skips the envelope pair;
    repeated --count runs on the saved raw .npz are cheap:

        python -m glassgen.reconstruct ic.00000 --param ic.param
        python -m glassgen.reconstruct --table ic_table_raw.npz --count 2e6
    """
    import argparse
    ap = argparse.ArgumentParser(
        description='density-table generator: capture a snapshot\'s gas '
                    'density as a RAW table (.npz + tipsy render), plus the '
                    'N-aware envelope-filled table at --count (default: the '
                    'snapshot gas count) - the target the glassgen builder '
                    'would relax against. Render: one gas particle per cell, '
                    'pos=centre, mass=rho*V, rho col=table density, temp '
                    'col=AMR level.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--table', metavar='FILE.npz',
                     help='precomputed raw table (a ${out}_raw.npz from an '
                          'earlier run): skip the capture, just render / '
                          'apply the --count envelope')
    src.add_argument('-I', dest='infile',
                     help='tipsy snapshot: capture the raw table from its gas')
    ap.add_argument('-o', dest='out', default=None,
                    help='output prefix. Default: input basename + _table '
                         '(writes ${out}_raw.npz/.00000 + '
                         '${out}_n${count}.npz/.00000)')
    ap.add_argument('--grid', default='amr', choices=('amr', 'uniform'))
    ap.add_argument('--grid-npx', type=int, default=None,
                    help='cells per axis (AMR base / uniform grid)')
    ap.add_argument('--capture-method', default='petkova',
                    choices=('petkova', 'sph-gather', 'sph-scatter'))
    ap.add_argument('-s', dest='nsmooth', type=int, default=64)
    ap.add_argument('--count', type=int, default=None,
                    help='target gas particle count N for the envelope-filled '
                         'table (m=M_gas/N -> rho_edge from the cumulative '
                         'faint tail at bg-c*m, cells below '
                         'rho_floor=rho_edge/bg-k**3 emptied + decay-tapered). '
                         'Default: the snapshot gas count; required with '
                         '--table unless --raw-only')
    ap.add_argument('--raw-only', dest='raw_only', action='store_true',
                    help='write only the raw table + render (no envelope pair)')
    ap.add_argument('--no-contour-refine', dest='contour_refine',
                    action='store_false',
                    help='disable the M10 contour-resolving capture (extra '
                         'refinement of cells straddling the rho_edge(count) '
                         'band, so the occupancy verdict is not quantized '
                         'on coarse cells)')
    ap.add_argument('--contour-band', dest='contour_band', type=float,
                    default=8.0, help='M10 band half-width factor around the '
                                      'rho_edge estimate (default 8)')
    ap.add_argument('--contour-beta', dest='contour_beta', type=float,
                    default=0.5, help='M10 size floor = beta * local output '
                                      'spacing (m/rho)^(1/3) (default 0.5)')
    ap.add_argument('--grade', action='store_true', default=True,
                    help='2:1-balance the AMR leaf set (no leaf touches a '
                         'leaf more than one level finer) + redeposit: level '
                         'jumps decay geometrically away from refined '
                         'regions, so the linear readout cannot stretch fine '
                         'bright values across a huge coarse neighbour '
                         '(DEFAULT since 2026-07-13)')
    ap.add_argument('--no-grade', dest='grade', action='store_false',
                    help='ungraded capture (pre-M11 leaf set)')
    ap.add_argument('--bg-c', dest='bg_c', type=int, default=0,
                    help='envelope: particles per resolved kernel for rho_edge '
                         '(0 = use --nsmooth/-s)')
    ap.add_argument('--bg-k', dest='bg_k', type=float, default=2.0,
                    help='envelope: background coarseness; rho_floor = '
                         'rho_edge/k**3 (default 2.0)')
    ap.add_argument('--shell-start', dest='shell_start', default='local',
                    choices=('edge', 'local'),
                    help="vacuum-taper start: 'local' (default) = the "
                         "nearest occupied cell's own value (local edge "
                         "continuation - the exterior pressure-support "
                         "scaffold); 'edge' = capped at the global rho_edge "
                         "(the pre-M12 scheme)")
    ap.add_argument('--scaffold', default='exp', choices=('exp', 'cubic'),
                    help="vacuum scaffold: 'exp' (default) or 'cubic' "
                         "(resolvability profile; --bg-k/--shell-start "
                         "do not apply)")
    ap.add_argument('--alpha', type=float, default=None,
                    help='resolvability constant for --scaffold cubic '
                         '(default 2.0)')
    ap.add_argument('--decay-cells', dest='decay_cells', type=float,
                    default=2.0,
                    help="vacuum-taper reach in units of the empty cell's own "
                         'width (lambda = decay_cells x cell width; default '
                         '2.0). Capture-cell based, so N-dependent - prefer '
                         '--decay-spacing for a resolution-invariant reach')
    ap.add_argument('--decay-spacing', dest='decay_spacing', type=float,
                    default=None, metavar='ETA',
                    help='vacuum-taper reach ETA in OUTPUT-spacing units '
                         '(lambda = ETA*(m/rho_edge)**(1/3), N-invariant; '
                         'overrides --decay-cells). eta~5 =~ 2 kernel supports')
    ap.add_argument('--decay-local', dest='decay_local', action='store_true',
                    help="evaluate the ETA taper at the launch cell's own "
                         'density (floored at rho_edge, the M16 skirt); '
                         'requires --decay-spacing')
    ap.add_argument('--no-floor-shelf', dest='flat_floor',
                    action='store_false', default=False,
                    help='retire the flat rho_floor shelf (plan M18 - now the '
                         'DEFAULT): the vacuum fill is ONE strictly-decreasing '
                         'profile down to rho_frame at the frame corner - '
                         'no mid-field plateau (parking lot), so anneal-'
                         'kicked strays keep a home-pointing gradient. '
                         '(kept as a back-compat no-op alias)')
    ap.add_argument('--floor-shelf', dest='flat_floor',
                    action='store_true',
                    help='LEGACY: restore the old flat rho_floor shelf fill '
                         '(pre-M18). NB the two-rung floor=rho_edge trigger '
                         'makes this path untested - diagnostics only')
    ap.add_argument('--tail-reach', dest='tail_reach', type=float,
                    default=6.0, metavar='S',
                    help='steepness knob for --no-floor-shelf: r_scale = '
                         'tail_scale/S (larger S = steeper near the gas, '
                         'shallower far out; default 6.0). Ignored with the '
                         'shelf on')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--param', help='gasoline .param file for the box')
    grp.add_argument('--box', nargs=3, type=float,
                     metavar=('LX', 'LY', 'LZ'))
    grp.add_argument('--open', action='store_true', help='no periodic box')
    args = ap.parse_args(argv)

    if args.table:
        for val, name in ((args.param, '--param'), (args.box, '--box'),
                          (args.grid_npx, '--grid-npx')):
            if val:
                ap.error(f'{name} only applies when capturing with -I')
        # no snapshot -> no count default: without --count just render it raw
        table = load_table(args.table)
        print(f'loaded {args.table} ({describe_table(table)})', flush=True)
        out = args.out or re.sub(r'(_raw)?\.npz$', '', args.table)
        _export_table(table, f'{out}_raw', 'raw table')
    else:
        if not (args.param or args.box or args.open):
            ap.error('capturing with -I needs --param, --box or --open')
        if args.open:
            box = None
        elif args.box:
            box = np.asarray(args.box, dtype=np.float64)
        else:
            from .params import box_from_param, read_param
            box = box_from_param(read_param(args.param))
        tip = _import_readtipsy()
        gas, _, _, _, _ = tip.readtipsy(args.infile)
        if len(gas) == 0:
            sys.exit(f'no gas particles in {args.infile}')
        if args.count is None:
            args.count = len(gas)          # default: the snapshot's gas count
        print(f'capturing raw table from {len(gas)} gas particles '
              f'({args.grid}, {args.capture_method}, box '
              f'{"open" if box is None else box})...', flush=True)
        # RAW capture (empty='zero'): exact cell averages, genuine zeros in
        # vacuum - the N-dependent fill is applied as an overlay below, never
        # baked into the saved table.
        kw = dict(grid=args.grid, nsmooth=args.nsmooth,
                  method=args.capture_method, empty='zero')
        if args.grid == 'amr':
            kw['grade'] = args.grade      # uniform grids have no levels
        if args.grid == 'amr' and args.contour_refine:
            kw.update(contour_n=args.count, contour_band=args.contour_band,
                      contour_beta=args.contour_beta)
        if args.grid_npx:
            kw['base_npx' if args.grid == 'amr' else 'npx'] = args.grid_npx
        table = capture_density(gas[:, 1:4].astype(np.float64),
                                gas[:, 0].astype(np.float64), box=box, **kw)
        print(f'captured ({describe_table(table)})', flush=True)
        out = args.out or re.sub(r'\.\d+$', '', args.infile) + '_table'
        save_table(table, f'{out}_raw.npz')
        print(f'wrote {out}_raw.npz', flush=True)
        _export_table(table, f'{out}_raw', 'raw table')

    if args.raw_only or not args.count:
        return
    # N-aware envelope pair: the target the glassgen builder relaxes against
    filled, info = apply_count_overlay(
        table, args.count, c=(args.bg_c or args.nsmooth), k=args.bg_k,
        shell_start=args.shell_start, scaffold=args.scaffold,
        alpha=args.alpha, decay_cells=args.decay_cells,
        decay_spacing=args.decay_spacing, decay_local=args.decay_local,
        flat_floor=args.flat_floor, tail_reach=args.tail_reach)
    floor_mode = ('floor-shelf' if info['flat_floor']
                  else f'no-floor-shelf(reach={info["tail_reach"]:g})')
    print(f'count={args.count}: m={info["m"]:.3e} '
          f'rho_edge={info["rho_edge"]:.3e} rho_floor={info["rho_floor"]:.3e} '
          f'delta_bg={info["delta_bg"]:.3g} occ(cells>=rho_edge)='
          f'{info["occ_frac"]:.3f} emptied {info["n_emptied"]}/'
          f'{info["n_cells"]} cells ({info["scaffold"]} scaffold, {floor_mode})',
          flush=True)
    save_table(filled, f'{out}_n{args.count}.npz')
    print(f'wrote {out}_n{args.count}.npz', flush=True)
    _export_table(filled, f'{out}_n{args.count}',
                  f'envelope fill at count {args.count}')


if __name__ == '__main__':
    main()
