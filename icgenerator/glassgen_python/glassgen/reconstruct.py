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
import warnings

import numpy as np

from .amrtable import AMRTable          # re-exported (moved out of this module)
from .density import TableDensity
from .envelope import (NL0_DEFAULT, _fill_empty, _pad_periodic_box,
                       _with_values, check_scaffold_resolvability,
                       edge_floor_spacing, frame_open_box, resolve_bg_k,
                       solve_tail_reach, table_cells)
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


def _capture_fill_kw(empty, vals, boxarr, pos, floor_frac):
    """Tail anchors for a capture-time 'decay' fill (C2): the no-floor
    profile runs from the nearest occupied value down to the floor_frac
    floor at the domain corner. Other fill modes need nothing."""
    if empty != 'decay':
        return {}
    v = np.asarray(vals, dtype=np.float64).ravel()
    pv = v[v > 0.0]
    if pv.size == 0:
        return {}
    fl = floor_frac * float(np.median(pv))
    span = (np.asarray(boxarr, dtype=np.float64) if boxarr is not None
            else np.asarray(pos.max(0) - pos.min(0), dtype=np.float64))
    return dict(floor=fl, tail_scale=0.5 * float(np.max(span)),
                tail_floor=fl)


def capture_density(pos, mass, box=None, grid='amr', nsmooth=64,
                    method='petkova', base_npx=None, max_level=16,
                    part_per_cell=8, npx=None, empty='decay',
                    floor_frac=1e-3, interp='vertex',
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
                'decay' (default) - fill each empty cell with the no-floor
                    profile (the C2-surviving fill, see
                    :func:`envelope._fill_empty`): a strictly-decreasing
                    ramp from the nearest occupied value down to the
                    floor_frac floor at the domain corner, so a sparse
                    particle one cell out feels a target near its actual
                    density with a home-pointing gradient everywhere.
                    (The legacy exp taper this mode used pre-2026-07-17
                    was retired with the taper family, CLEANUP_PLAN C2.)
                'min'  - floor at the smallest positive cell value (can be
                    a ~1e-12-of-peak kernel tail on a galaxy table: keeps
                    positivity but not the dynamic range).
                'zero' - keep zeros (table is then NOT relax-safe; the
                    production reconstruct captures RAW and applies the
                    N-aware overlay separately).
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
        vals = _fill_empty(vals.reshape(-1), empty, centres, boxarr,
                           floor_frac=floor_frac,
                           **_capture_fill_kw(empty, vals, boxarr, pos,
                                              floor_frac)).reshape(vals.shape)
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
    vals = _fill_empty(vals_raw, empty, amr.centres(), boxarr,
                       floor_frac=floor_frac,
                       **_capture_fill_kw(empty, vals_raw, boxarr, pos,
                                          floor_frac))
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
                         bg_c=None, bg_k=None, kernel_match=True,
                         contour_refine=True, contour_band=8.0,
                         contour_beta=0.5, target_interp='vertex',
                         grade=True,
                         alpha=None,
                         tail_reach=None, n_frame=None, nl0=None,
                         bg_over=0.0,
                         km_hmode='particle', km_hscale=1.0, verbose=False):
    """The `glassresample --op reconstruct` engine - a THIN WRAPPER around
    the glassgen table builder (decided 2026-07-10): capture the input gas
    density (exact raw AMR table), hand it to
    :func:`glassgen.progressive.build_from_table` (sample `count` positions
    from the table + N-aware envelope + progressive relax + spatial cull /
    renorm to the input gas mass), then conservatively SCATTER per-survivor
    fields from the input onto the FINAL positions (scatter_fields, M22).

      pos_in, mass_in : the INPUT gas.
      count           : target gas particle count (may be < or > the input).
      fields          : dict name -> (Nin,)/(Nin,3) input gas fields,
                        scattered conservatively onto the survivors (the
                        glass has no history to inherit; totals conserved
                        exactly over the resolved region, truncation budget
                        + residuals in audit['fields']).
      cascade         : False -> single full-res relax of the placed ensemble
                        (the --no-progressive fallback), same envelope + cull.
      bg_c, bg_k      : envelope knobs (rho_edge cut count; two-rung ladder
                        rho_frame = rho_edge/k**6). The background is always
                        sampled backing (mass-contract closure; the lattice
                        and bg_tail were retired 2026-07-16, CLEANUP_PLAN C1).
      alpha : the cascade resolvability tolerance (auto_coarse_count;
                        the cubic scaffold that shared it was DELETED
                        record; alpha = the shared cascade constant, 2.0).
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
    # per-survivor fields: conservative scatter from the input onto the
    # FINAL positions (M22). The truncation split needs rho_edge BEFORE the
    # build - computed here in LOCKSTEP with build_table_envelope (same m,
    # same c, same k resolve on the same cells).
    t0 = _time.perf_counter()
    src_rho, _ = particle_density(pos_in, mass_in, box=box, nsmooth=nsmooth)
    timings['src_rho'] = _time.perf_counter() - t0
    _, sizes_e, _, vals_e = table_cells(table)
    rho_edge_pre, _ = edge_floor_spacing(
        np.asarray(vals_e, dtype=np.float64), sizes_e.prod(1),
        M_gas / int(count), c=(bg_c or nsmooth),
        k=resolve_bg_k(bg_k, n_frame, nsmooth=(bg_c or nsmooth)))
    field_audit = {}

    def _hook(final_pos):
        t = _time.perf_counter()
        out, fa = scatter_fields(final_pos, M_gas / max(len(final_pos), 1),
                                 pos_in, mass_in, src_rho, fields,
                                 rho_edge=rho_edge_pre, box=box,
                                 nsmooth=nsmooth)
        field_audit.update(fa)
        timings['scatter'] = _time.perf_counter() - t
        return out

    rkw = dict(relax_kw or {})
    if progress is not None:
        rkw.setdefault('callback', progress)
    t0 = _time.perf_counter()
    res = build_from_table(table, count, box=box, gas_mass=M_gas,
                           nsmooth=nsmooth, bg_c=bg_c, bg_k=bg_k,
                           envelope='on', cascade=cascade, seed=seed,
                           open_pad=pad_frac, kernel_match=kernel_match,
                           alpha=alpha,
                           tail_reach=tail_reach,
                           n_frame=n_frame, nl0=nl0, bg_over=bg_over,
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
                       'scaffold', 'alpha')},
             relax_iters=int(res.niter), relax_stop=str(res.stop_reason),
             timings=timings, fields=dict(field_audit),
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
        dist, nn = tree.query(np.mod(qpos - lo, box), k=k, workers=-1)
    else:
        tree = cKDTree(src_pos)
        dist, nn = tree.query(qpos, k=k, workers=-1)
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


def scatter_fields(qpos, m_out, src_pos, src_mass, src_rho, fields,
                   rho_edge=None, box=None, nsmooth=64):
    """Conservative donor-normalized SCATTER of input fields onto the
    survivors - THE default field transfer (RECONSTRUCT_PLAN M22,
    2026-07-19; replaces the Shepard gather above, which is kept for
    comparison/tests).

    Every INPUT particle in the resolved region (src_rho >= rho_edge)
    donates its conserved content m_i*A_i to the receivers its kernel
    reaches (sph_interp.sph_method.scatter_to_points: per-donor shares
    sum to 1, nearest-receiver fallback so every donor lands), and each
    receiver takes the MASS-WEIGHTED AVERAGE: A_j = content_j/recv_mass_j.

    Division convention (A/B'd on the galaxy, 2026-07-19, RECONSTRUCT_PLAN
    M22): dividing by a UNIFORM mass share instead makes the written
    totals exact but writes A_j = r_j*Abar_j with r = the received/uniform
    mass ratio (spread [0,3.8] at 20k) - measured KE +31%, coverage noise
    injected into the dynamics. The average keeps every particle's value
    the smoothed local input field (a constant field is reproduced
    EXACTLY, fill included); the written-IC totals then drift by the
    DELIVERED-GEOMETRY difference (the reconstructed disk is thicker, so
    equal-mass particles sample more hot off-plane gas: E_th +17% here,
    +19% under the legacy gather - a representation limit, NOT transfer
    error; closing it uniformly was measured at x0.85 on every u and
    rejected). The audit reports TWO tiers: resid_* = the scatter's own
    content conservation (machine zeros - nothing is lost in transfer)
    and drift_*/ke_*/lz_* = written-IC totals vs the interior input
    (geometry diagnostics - watch them like the wing excess). Donors
    below rho_edge are NOT scattered: their content is the explicit
    truncation budget (deliberately unrepresented, never smeared onto the
    resolved region).

    Returns (out, audit): out maps name -> (Nq,)/(Nq,ncomp) intensive
    values; audit holds the truncation budget (trunc_*), per-field
    conservation residuals (should be ~1e-16), the KE deficit from
    velocity smoothing (ke_*, when a 3-vector 'vel' field is present,
    with L_z in/out about the interior COM) and the received/output mass
    ratio statistics (recv_ratio_*)."""
    from scipy.spatial import cKDTree
    _import_sph_interp()
    from sph_interp.sph_method import scatter_to_points
    qpos = np.ascontiguousarray(qpos, dtype=np.float64).reshape(-1, 3)
    src_pos = np.ascontiguousarray(src_pos, dtype=np.float64).reshape(-1, 3)
    src_mass = np.asarray(src_mass, dtype=np.float64).ravel()
    src_rho = np.asarray(src_rho, dtype=np.float64).ravel()
    m_out = float(m_out)
    interior = (src_rho >= float(rho_edge)) if rho_edge is not None \
        else np.ones(len(src_pos), dtype=bool)
    # donor kernel: 2h = the SPH support holding nsmooth neighbours
    h_src = 0.5 * (3.0 * nsmooth * src_mass
                   / (4.0 * np.pi * np.maximum(src_rho, 1e-300))) ** (1. / 3.)
    # receiver integration element: local spacing cubed (+ the kernel floor
    # scatter_to_points wants, from the same NN pass)
    if len(qpos) > 1:
        d1 = cKDTree(qpos).query(qpos, k=2, workers=-1)[0][:, 1]
        gen_vol, hmin = d1 ** 3, 0.5 * float(np.median(d1))
    else:
        gen_vol, hmin = None, None
    fin = {k: np.asarray(v, dtype=np.float64)[interior]
           for k, v in fields.items()}
    content, recv_mass = scatter_to_points(
        src_pos[interior], h_src[interior], src_mass[interior], fin,
        qpos, gen_vol=gen_vol, hmin=hmin, box=box,
        periodic=(box is not None,) * 3)
    m_sc = float(src_mass[interior].sum()) / max(len(qpos), 1)
    starved = recv_mass == 0.0                    # nobody reached them
    rs = np.where(starved, 1.0, recv_mass)
    out = {k: (v / rs[:, None] if v.ndim == 2 else v / rs)
           for k, v in content.items()}

    audit = dict(n_donor=int(interior.sum()),
                 n_trunc=int((~interior).sum()),
                 n_starved=int(starved.sum()),
                 trunc_mass=float(src_mass[~interior].sum()),
                 m_scatter=m_sc,
                 mass_renorm_trunc=float(m_out / m_sc),
                 recv_ratio_min=float(np.min(recv_mass / m_sc)),
                 recv_ratio_med=float(np.median(recv_mass / m_sc)),
                 recv_ratio_max=float(np.max(recv_mass / m_sc)))
    # residuals from the raw scatter content (the machine-zero invariant;
    # the starved-receiver fill below is continuity, not content)
    for k, arr in fields.items():
        arr2 = np.asarray(arr, dtype=np.float64).reshape(len(src_pos), -1)
        c_in = src_mass[interior, None] * arr2[interior]
        t_in = c_in.sum(0)
        t_out = content[k].reshape(len(qpos), -1).sum(0)
        t_tr = (src_mass[~interior, None] * arr2[~interior]).sum(0)
        scale = np.abs(t_in) + np.abs(c_in).sum(0)
        audit[f'resid_{k}'] = float(np.max(np.abs(t_out - t_in)
                                           / np.maximum(scale, 1e-300)))
        audit[f'trunc_{k}'] = [float(x) for x in t_tr]
        w_out = (m_out * out[k].reshape(len(qpos), -1)).sum(0)
        audit[f'drift_{k}'] = float(np.max(np.abs(w_out - t_in)
                                           / np.maximum(scale, 1e-300)))
    for k, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:      # a velocity: KE + L_z
            ke_in = 0.5 * (src_mass[interior]
                           * (arr[interior] ** 2).sum(1)).sum()
            ke_out = 0.5 * m_out * (out[k] ** 2).sum()
            com = np.average(src_pos[interior], axis=0,
                             weights=src_mass[interior])
            q = src_pos[interior] - com
            lz_in = (src_mass[interior] * (q[:, 0] * arr[interior][:, 1]
                                           - q[:, 1] * arr[interior][:, 0])
                     ).sum()
            qq = qpos - com
            lz_out = m_out * (qq[:, 0] * out[k][:, 1]
                              - qq[:, 1] * out[k][:, 0]).sum()
            audit[f'ke_in_{k}'] = float(ke_in)
            audit[f'ke_out_{k}'] = float(ke_out)
            audit[f'lz_in_{k}'] = float(lz_in)
            audit[f'lz_out_{k}'] = float(lz_out)
    # starved receivers (typically 0 at production relax quality): no donor
    # kernel reached them and they were nobody's nearest - leaving zeros
    # would punch pressure/velocity holes in the IC, so fill their INTENSIVE
    # values from the nearest interior donor (pure continuity; the audit
    # above deliberately excludes this content, and n_starved reports it).
    if starved.any() and interior.any():
        nn = cKDTree(src_pos[interior]).query(qpos[starved], k=1,
                                              workers=-1)[1]
        for k, arr in fields.items():
            arr = np.asarray(arr, dtype=np.float64)
            out[k][starved] = arr[interior][nn]
    return out, audit


def _pow2(x, lo=4):
    return max(lo, 2 ** int(round(np.log2(max(x, 1.0)))))


def save_table(table, path):
    """Save a density table (AMRTable, or TableDensity of any dim) as the
    self-describing .npz (CLEANUP_PLAN C6): AMR keeps its block layout,
    uniform/mapped tables carry axes + values + mapping in the header.
    Both load back through load_table() without sph_interp."""
    if isinstance(table, AMRTable):
        table.save(path)
    elif isinstance(table, TableDensity):
        table.save(path)
    else:
        raise TypeError(f'cannot save table of type {type(table).__name__}')


def load_table(path):
    """Load a .npz density table: sniffs the keys - AMR block layout ->
    AMRTable; values (+axes, and the 1D/2D coordinate mapping in the
    header) -> TableDensity. Legacy 3D saves without a mapping key load
    fine (3D is (x,y,z)-indexed)."""
    with np.load(path) as d:
        files = set(d.files)
        if 'keys' in files:
            pass                        # AMR layout: reload via the class
        elif 'values' in files:
            pass                        # TableDensity layout
        else:
            raise ValueError(
                f'{path}: not a density table (.npz needs either the '
                'AMRTable block layout or ax0.. + values [+ mapping])')
        amr = 'keys' in files
    return AMRTable.load(path) if amr else TableDensity.load(path)


def describe_table(table):
    """One-line human description for CLI logging."""
    if isinstance(table, AMRTable):
        nlev = int((np.diff(table.offsets) > 0).sum())
        return (f'AMR, {table.ncell} leaves, {nlev} levels, '
                f'base {"x".join(str(d) for d in table.dims[0])}, '
                f'{table.interp} readout')
    if isinstance(table, TableDensity):
        desc = 'uniform ' + 'x'.join(str(len(a)) for a in table.axes)
        if table.mapping is not None:
            desc += f', mapping {table.mapping}'
        return desc
    return type(table).__name__


def _grid_rho(res, volumes, method):
    """Cell density from a GridResult: petkova/gather expose it as the
    'rho' field (petkova = exact mass-conserving cell average); scatter =
    accumulated mass (res.norm) / cell volume (conservative)."""
    if method == 'sph-scatter':
        return np.asarray(res.norm, dtype=np.float64) / volumes
    return np.asarray(res.data['rho'], dtype=np.float64)


def apply_count_overlay(table, count, c=64, k=None,
                        tail_reach=None, n_frame=None, nl0=None):
    """The N-aware envelope fill on a captured table: returns
    ``(filled_table, info)`` - a NEW table (the input is untouched). Uses the
    SAME rho_edge scheme as the glassgen builder
    (:func:`envelope.edge_floor_spacing` + the decay ``_fill_empty`` with
    ``floor=rho_edge``): from ``m = M_gas / count``, every cell below
    ``rho_edge`` is emptied and refilled as ONE strictly-decreasing profile
    down to ``rho_frame = rho_edge/k**6`` at the frame edge (frame sized by
    the same M2 rule the builder uses; steepness from the M20 nl0 solve or
    a legacy explicit ``tail_reach``); cells above keep their exact values.
    Different N -> different overlay on the SAME exact table (larger N ->
    lower rho_edge/frame). This is the density target the relax sees at
    that count (the leaf-set view of the builder's PaddedDensity shell;
    the analytic outside-domain tail continuation is not representable in
    the leaf set and is omitted here). The builder's background (sampled
    backing) changes PARTICLES, not the density target, so it has no
    counterpart here. (The legacy taper family - flat_floor shelf,
    shell_start, decay_cells/decay_spacing/decay_local - retired
    2026-07-17, CLEANUP_PLAN C2.)"""
    centres, sizes, _, vals = table_cells(table)
    vals = np.asarray(vals, dtype=np.float64)
    vols = sizes.prod(1)
    m = float((vals * vols).sum()) / int(count)
    # M20: k from n_frame unless legacy-explicit (lockstep with
    # build_table_envelope - same resolve, same solve below)
    k = resolve_bg_k(k, n_frame, nsmooth=c)
    re, delta_bg = edge_floor_spacing(vals, vols, m, c=c, k=k)
    fbox = table.box if getattr(table, 'periodic', False) is True \
        or (hasattr(table, 'periodic') and bool(np.ravel(table.periodic)[0])) \
        else None
    # frame extent for the M6 tail scale, sized as the builder does (M2 rule,
    # nsmooth ~ c). taper_len retired 2026-07-16 (CLEANUP_PLAN C1a) - it must
    # stay in lockstep with build_table_envelope or the rendered target and
    # the relaxed target disagree on tail_scale.
    if fbox is None:
        dom_lo = (centres - 0.5 * sizes).min(0)
        dom_hi = (centres + 0.5 * sizes).max(0)
        frame_extent, _ = frame_open_box(dom_lo, dom_hi, delta_bg,
                                         nsmooth=c, k=k)
    else:
        frame_extent = np.asarray(fbox, dtype=np.float64).reshape(3)
    rho_frame = re / k ** 6               # two-rung ladder: rho_edge -> rho_frame
    # M20: solve tail_reach from nl0 (or take the legacy explicit value) -
    # MUST stay in lockstep with build_table_envelope or the rendered
    # target and the relaxed target disagree on the profile.
    nl0_eff = p_eff = None
    tail_scale = 0.5 * float(np.max(frame_extent))
    spacing_edge = (m / re) ** (1.0 / 3.0)
    if tail_reach is None:
        tail_reach = solve_tail_reach(
            tail_scale, spacing_edge,
            NL0_DEFAULT if nl0 is None else float(nl0), k, nsmooth=c)
    elif nl0 is not None:
        warnings.warn('explicit tail_reach overrides nl0 (legacy path)')
    nl0_eff, p_eff = check_scaffold_resolvability(
        tail_scale, spacing_edge, tail_reach, k, nsmooth=c)
    filled = _fill_empty(vals.copy(), 'decay', centres, fbox,
                         floor=re, cap=None,
                         tail_scale=tail_scale,
                         tail_floor=rho_frame,
                         tail_reach=tail_reach)
    n_emptied = int((vals < re).sum())
    occ = float((vals >= re).mean())
    info = dict(m=m, rho_edge=re, rho_frame=rho_frame,
                delta_bg=delta_bg, occ_frac=occ,
                tail_reach=tail_reach, bg_k=k,
                n_emptied=n_emptied, n_cells=len(vals))
    if nl0_eff is not None:               # M20: launch N_L + exponent
        info['nl0'] = nl0_eff
        info['profile_p'] = p_eff
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
    ap.add_argument('--help-extra', action='help',
                    help='show the full help, including the hidden expert '
                         'options')
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
                         'faint tail at nedge*m; cells below rho_edge refilled '
                         'with the strictly-decreasing rho_edge->rho_frame '
                         'profile, rho_frame=rho_edge*nframe/nsmooth). '
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
    ap.add_argument('--nedge', dest='bg_c', type=int, default=0,
                    help='envelope: resolvability contour count (INTEGRAL: '
                         'rho_edge = where the cumulative faint-end mass '
                         'first exceeds nedge*m - the whole discarded tail '
                         'in particle masses, NOT a per-kernel count; 0 = '
                         'use --nsmooth/-s)')
    ap.add_argument('--nframe', dest='n_frame', type=float, default=None,
                    help='M20: floor backability - expected particles per '
                         'edge kernel at the profile floor (rho_frame = '
                         'rho_edge*nframe/nsmooth, k=(nsmooth/nframe)^(1/6); '
                         'default 16 ~ k1.26)')
    ap.add_argument('--nl0', dest='nl0', type=float, default=None,
                    help='M20: scaffold launch resolvability N_L(0) - the '
                         'near-gas e-fold length in kernel supports; '
                         'tail_reach is solved per build (default 1.35; '
                         'avoid ~1, the metastable resonance band)')
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
        if getattr(table, 'dim', 3) != 3:
            sys.exit(f'{args.table}: glasstable renders/overlays 3D tables; '
                     f'this is a {table.dim}D mapped table - relax against '
                     'it directly (glassrelax --table)')
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
        table, args.count, c=(args.bg_c or args.nsmooth),
        n_frame=args.n_frame, nl0=args.nl0)
    floor_mode = f'no-floor(reach={info["tail_reach"]:g})'
    print(f'count={args.count}: m={info["m"]:.3e} '
          f'rho_edge={info["rho_edge"]:.3e} rho_frame={info["rho_frame"]:.3e} '
          f'delta_bg={info["delta_bg"]:.3g} occ(cells>=rho_edge)='
          f'{info["occ_frac"]:.3f} emptied {info["n_emptied"]}/'
          f'{info["n_cells"]} cells ({floor_mode})',
          flush=True)
    save_table(filled, f'{out}_n{args.count}.npz')
    print(f'wrote {out}_n{args.count}.npz', flush=True)
    _export_table(filled, f'{out}_n{args.count}',
                  f'envelope fill at count {args.count}')


if __name__ == '__main__':
    main()
