"""Free-surface / vacuum envelope for the glassgen relax (the ported core).

When a glass is built against a density field that has a free surface or vacuum
(open-box ICs, galaxy tables), the edge particles have truncation-deficient
kernels on the vacuum side -> their SPH density can never reach the target ->
an irreducible pseudo-pressure force stalls the confined relax. The fix is to
pad the vacuum side with a mass-consistent BACKGROUND of ordinary particles so
edge particles get a full kernel, relax the union, then remove the background
by a spatial vacuum cull at the end.

This module owns the N-aware floor + placement + cull. Everything here is
resolution-driven: given the exact (petkova) density field and the target mass
resolution ``m = M_gas / N_final``,

    rho_edge  = the density where the cumulative faint-end mass Sum(rho*V)
                first exceeds c*m  (c ~ nsmooth particles = one kernel): below
                it there is < one kernel of mass -> unresolvable -> background.
    rho_frame = rho_edge / k**6          (k = background coarseness vs the edge;
                the deep-vacuum floor reached at the frame corner)
    Delta_bg  = k * (m / rho_edge)**(1/3),   background particle mass m_bg = m.

The shell target DECAYS from rho_edge at the gas boundary to rho_frame outward
(a density -> pressure gradient the relax works the background into), and the
background is a jittered lattice whose number density tracks that decay.

Nothing here needs sph_interp: the density field / occupancy / shell targets are
passed in as any object exposing ``rho0(pos)`` (an ``AMRTable``, a
``TableDensity``, or a plain callable).
"""
import warnings

import numpy as np
from numba import njit, prange

from .amrtable import AMRTable, _amr_readout, _corner_tri, _pwc_index
from .density import CallableDensity, DensityField, TableDensity

NSMOOTH_DEFAULT = 64        # matches the glassgen relax default nsmooth
ALPHA_DEFAULT = 2.0         # resolvability constant: local spacing must obey
MAPPED_NGRID_DEFAULT = 128  # Cartesian volume quadrature for open mapped targets
#                             Delta(rho) <= alpha * rho/|grad rho| - the SAME
#                             alpha as progressive.auto_coarse_count (cascade
#                             coarse-level bound, calibrated on mhdcollapse).


def _eval(fn, pos):
    """Evaluate a density-field-like object (has ``rho0``) or a plain callable
    at ``pos`` (n,3) -> (n,)."""
    if hasattr(fn, 'rho0'):
        return np.asarray(fn.rho0(pos), dtype=np.float64)
    return np.asarray(fn(pos), dtype=np.float64)


def rho_edge(vals, vols, m, c=NSMOOTH_DEFAULT):
    """N-aware edge density from a cell density field + the target mass res.

    Sort cells ascending by density and walk the cumulative faint-end mass
    ``cumsum(rho*V)``; ``rho_edge`` is the density at which it first reaches
    ``c*m`` (``c`` particles ~ one kernel). Everything fainter carries less than
    one kernel of mass and is unresolvable at mass resolution ``m`` -> treated
    as background. Robust (a mass integral, immune to a single faint outlier),
    box-independent (a contour of the distribution), and resolution-aware
    (larger N -> smaller ``m`` -> lower ``rho_edge``).

    Zero/negative cells (pure vacuum) carry no mass, so the returned value is
    always a positive density (the first cell whose tail clears the threshold).
    If the WHOLE field holds less than ``c*m`` (N too small to resolve even one
    kernel), returns the maximum density (degenerate: everything is background).
    """
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    vols = np.asarray(vols, dtype=np.float64).reshape(-1)
    if len(vals) == 0:
        raise ValueError('rho_edge: empty density field')
    order = np.argsort(vals, kind='stable')
    cum = np.cumsum(vals[order] * vols[order])
    idx = int(np.searchsorted(cum, c * m, side='left'))
    idx = min(idx, len(order) - 1)
    return float(vals[order[idx]])


def edge_floor_spacing(vals, vols, m, c=NSMOOTH_DEFAULT, k=2.0):
    """Convenience: (rho_edge, Delta_bg) from the density field and (m, c, k).
    ``Delta_bg = k*(m/rho_edge)**1/3``. The vacuum field decays
    rho_edge -> rho_frame = rho_edge/k**6.
    """
    re = rho_edge(vals, vols, m, c=c)
    delta_bg = k * (m / re) ** (1.0 / 3.0)
    return re, delta_bg


def kernel_support(spacing, nsmooth=NSMOOTH_DEFAULT):
    """SPH kernel support radius of a particle in a region of uniform spacing
    ``Delta``: the radius enclosing ``nsmooth`` neighbours at number density
    ``Delta**-3``, i.e. ``(3*nsmooth/4pi)**(1/3) * Delta`` (~2.47*Delta at
    nsmooth=64). This is the reach of the SPARSEST kernels in an envelope
    build (the background's) - the quantity the periodic frame must contain
    (:func:`frame_open_box`)."""
    return (3.0 * nsmooth / (4.0 * np.pi)) ** (1.0 / 3.0) * spacing


#: Default floor backability count and scaffold launch resolvability
#: (the measured k1.25-tr12 anchor).
NFRAME_DEFAULT = 16.0
NL0_DEFAULT = 1.35


def resolve_bg_k(k=None, n_frame=None, nsmooth=NSMOOTH_DEFAULT):
    """Background coarseness from the floor-BACKABILITY count.

    ``n_frame`` = expected particles of mass m per EDGE-KERNEL volume
    (V_e = nsmooth*m/rho_edge, so n(rho) = nsmooth*rho/rho_edge) at the
    profile floor: rho_frame = rho_edge*(n_frame/nsmooth), i.e.
    k = (nsmooth/n_frame)**(1/6). n_frame=1 <=> k=2 (the floor AT the
    unbackable density - a structurally thin buffer); n_frame=16 (default)
    <=> k~1.26, a fat backed buffer. Legacy escape: pass ``k`` explicitly
    (mutually exclusive with n_frame)."""
    if k is not None:
        if n_frame is not None:
            raise ValueError('pass either the legacy k or n_frame, not both')
        return float(k)
    nf = NFRAME_DEFAULT if n_frame is None else float(n_frame)
    if not 0.0 < nf <= nsmooth:
        raise ValueError(f'n_frame must be in (0, nsmooth={nsmooth}]; got {nf}')
    return float((nsmooth / nf) ** (1.0 / 6.0))


def solve_tail_reach(tail_scale, spacing_edge, nl0, k,
                     nsmooth=NSMOOTH_DEFAULT):
    """tail_reach (TR) that LAUNCHES the no-floor profile at ``nl0``
    kernel supports.

    The profile rho(d) = rho_edge*(1+d/rs)**(-p) (rs = tail_scale/TR,
    p = 6*ln(k)/ln(1+TR)) has near-gas e-fold length L(0) = rs/p; solve
    L(0) = nl0 * 2h_e with 2h_e = kernel_support(spacing_edge). N_L ~ 1
    is the measured metastable RESONANCE; pinning the
    launch in kernel units keeps a config off it at every N (2h_e grows
    with N, so the solved TR shrinks). L(0) is strictly decreasing in TR
    -> unique root (geometric bisection). If nl0 exceeds the TR->0 limit
    tail_scale/(6*ln k * 2h_e) it is unreachable on this frame: the
    shallowest profile is returned with a warning."""
    two_h = kernel_support(spacing_edge, nsmooth)
    ln_dyn = 6.0 * np.log(float(k))
    target = float(nl0) * two_h
    if target * ln_dyn >= float(tail_scale):
        warnings.warn(
            f'nl0={nl0} unreachable on this frame (max launch e-fold '
            f'{float(tail_scale) / ln_dyn:.4g} < requested {target:.4g}); '
            'using the shallowest profile')
        return 1e-3
    lo, hi = 1e-3, 1e9
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if float(tail_scale) * np.log1p(mid) / (mid * ln_dyn) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def check_scaffold_resolvability(tail_scale, spacing_edge, tail_reach, k,
                                 nsmooth=NSMOOTH_DEFAULT,
                                 band=(0.85, 1.15), n_backed=4.0):
    """Guard on the scaffold's resolvability curve
    N_L(d) = nl0*(1+d/rs)**(1-p/3).

    Warns on the measured failure geometries (in kernel units): (a) p >= 3 - N_L flat or DESCENDING, so the profile
    lingers in/returns to the resonance band far out where backing is
    thin (the k2.0-tr3 cell: p = 3 exactly, the whole vacuum resonant);
    (b) the LAUNCH sits in the band while the floor backing is thin
    (n_frame_eff = nsmooth/k**6 below ~10): the k1.5-tr6 valley - the
    same launch on a fat k1.25-style buffer (k1.25-tr24) is stable;
    (c) the climb crosses the band beyond the backed depth, where the
    target holds fewer than ``n_backed`` expected particles per edge
    kernel and the drawn gradient is not felt (KNN-h saturation - the
    k2.0-tr12/24 huge-sigma cells). Returns (nl0, p)."""
    two_h = kernel_support(spacing_edge, nsmooth)
    tr = max(float(tail_reach), 1e-12)
    rs = float(tail_scale) / tr
    p = 6.0 * np.log(float(k)) / np.log1p(tr)
    nl0 = rs / (p * two_h)
    n_frame_eff = nsmooth / float(k) ** 6
    if p >= 3.0:
        warnings.warn(
            f'scaffold N_L(d) is {"flat" if p == 3.0 else "DESCENDING"} '
            f'(p={p:.2f} >= 3): the profile re-enters low N_L far out where '
            'backing is thin (the k2-tr3 pathology) - raise n_frame or '
            'nl0')
        return nl0, p
    if band[0] <= nl0 <= band[1] and n_frame_eff < 10.0:
        warnings.warn(
            f'scaffold LAUNCHES in the N_L resonance band (nl0={nl0:.2f}) '
            f'on a thin floor buffer (n_frame_eff={n_frame_eff:.1f} < 10): '
            'the valley geometry (k1.5-tr6) - move nl0 off ~1 or '
            'raise n_frame')
        return nl0, p
    expo = 1.0 - p / 3.0

    def _d_at(nl):                       # climb distance to N_L = nl
        if nl <= nl0:
            return 0.0
        return rs * ((nl / nl0) ** (1.0 / expo) - 1.0)

    d0, d1 = _d_at(band[0]), min(_d_at(band[1]), float(tail_scale))
    if d1 > d0:                          # the curve crosses the band
        d_backed = rs * ((nsmooth / n_backed) ** (1.0 / p) - 1.0)
        if d1 > d_backed:
            warnings.warn(
                'scaffold crosses the N_L resonance band where it is '
                f'UNBACKED: in-band d = [{d0:.4g}, {d1:.4g}] vs backed '
                f'depth {d_backed:.4g} (< {n_backed:g} expected particles '
                f'per edge kernel); launch nl0={nl0:.2f}, p={p:.2f} - move '
                'nl0 off ~1 or raise n_frame')
    return nl0, p


def frame_open_box(span_lo, span_hi, delta_bg, nsmooth=NSMOOTH_DEFAULT,
                   pad_frac=0.1, k=2.0, cap_delta=None):
    """Size the periodic frame for an OPEN (free-surface) build: per-axis
    margins AND a global kernel-fits-the-box cap, so no kernel is ever larger
    than any box axis (a period shorter than a kernel support DIAMETER lets
    the kernel wrap onto its own image).

      margin_i = max(pad_frac*span_i, H_edge)   (H_edge = support
                 of an edge-gas kernel, Delta_edge = Delta_bg/k)
      box_i    = span_i + 2*margin_i
      box_i    = max(box_i, 2*H_bg)                        (H_bg = support of
                 a background kernel at spacing Delta_bg)

    span_lo/span_hi : world bbox that must sit inside the frame (gas domain +
    background region). Returns (box (3,), centre (3,)): shift positions by
    ``-centre`` to enter the centred periodic frame.

    cap_delta : spacing of the SPARSEST kernels the frame must contain
    (defaults to ``delta_bg``). The sampled backing's faintest fillers can
    sit at spacing up to ``k*delta_bg`` (toward ``rho_frame``); pass
    ``cap_delta=k*delta_bg`` if a small/thin domain shows kernel wrap."""
    span_lo = np.asarray(span_lo, dtype=np.float64).reshape(3)
    span_hi = np.asarray(span_hi, dtype=np.float64).reshape(3)
    span = span_hi - span_lo
    h_bg = kernel_support(delta_bg, nsmooth)
    h_cap = h_bg if cap_delta is None else kernel_support(cap_delta, nsmooth)
    h_edge = h_bg / max(k, 1e-12)
    margin = np.maximum(pad_frac * span, h_edge)
    box = np.maximum(span + 2.0 * margin, 2.0 * h_cap)
    centre = 0.5 * (span_lo + span_hi)
    return np.ascontiguousarray(box), np.ascontiguousarray(centre)


class PaddedDensity(DensityField):
    """Composite target: exact base density in the gas, shell target in vacuum.

    ``rho0(pos) = base(pos)             where occupancy(pos) >= 0.5 (gas)
                = min(shell(pos), cap)  elsewhere (decay-filled shell)``

    Keeps the exact/analytic gas density (not degraded to the coarse capture
    grid) and only overrides the vacuum/shell with the decay-to-floor envelope.
    ``base``/``occupancy``/``shell`` are any objects exposing ``rho0`` (an
    ``AMRTable``/``TableDensity``) or plain callables.

    cap : clamp on the VACUUM side, normally ``rho_edge``. The shell is only ever read below the occupancy contour, where
        the target is <= rho_edge by definition; without the clamp, trilinear
        bleed from dense gas cells into the first vacuum cells makes the
        shell read O(gas density) right outside the contour while the base
        side hands over exactly rho_edge - an O(1) value seam in the target
        that the relax force (and auto_coarse_count) sees as a step."""

    def __init__(self, base, occupancy, shell, cap=None):
        self.base = base
        self.occupancy = occupancy
        self.shell = shell
        self.cap = cap

    def rho0(self, pos):
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        out = np.empty(len(pos))
        gas = _eval(self.occupancy, pos) >= 0.5
        if gas.any():
            out[gas] = _eval(self.base, pos[gas])
        nvac = ~gas
        if nvac.any():
            sv = _eval(self.shell, pos[nvac])
            out[nvac] = sv if self.cap is None else np.minimum(sv, self.cap)
        return out

    __call__ = rho0


class _TailedShell(DensityField):
    """Continue the shell's vacuum decay OUTSIDE the table domain.

    The shell is a cell table over the capture domain; beyond its bbox the
    AMR/grid readout clamps to the boundary cell, so the frame's outer vacuum
    (which :func:`frame_open_box`'s kernel-fits-the-box rule can make much
    larger than the domain) reads FLAT - no
    home-pointing gradient for a kicked stray out there. This wrapper clamps
    the query into the domain, reads the shell value ``v`` there, and decays
    it with the excess distance ``d`` (query -> domain bbox) as
    ``v * (1 + d/r_scale)^-p`` with the exponent solved PER QUERY so the
    decay lands exactly on ``floor`` at ``d = d_max`` (the farthest frame
    corner): continuous at the domain faces, monotone home-pointing outside,
    the floor is reached only AT the frame edge whatever the boundary value
    was."""

    def __init__(self, shell, dom_lo, dom_hi, r_scale, d_max, floor):
        self.shell = shell
        self.dom_lo = np.asarray(dom_lo, dtype=np.float64)
        self.dom_hi = np.asarray(dom_hi, dtype=np.float64)
        self.r_scale = float(r_scale)
        self.d_max = float(d_max)
        self.floor = float(floor)
        # ln(1 + d_max/r_scale): the log-span the per-query exponent divides
        self._span = float(np.log1p(self.d_max / self.r_scale))

    def rho0(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        cl = np.clip(pos, self.dom_lo, self.dom_hi)
        v = _eval(self.shell, cl)
        d2 = ((pos - cl) ** 2).sum(1)
        out = d2 > 0.0
        if out.any():
            p = np.log(np.maximum(v[out] / self.floor, 1.0)) / self._span
            fac = (1.0 + np.sqrt(d2[out]) / self.r_scale) ** (-p)
            v[out] = np.maximum(v[out] * fac, self.floor)
        return v

    __call__ = rho0


@njit(cache=True, inline='always')
def _padded_point(x, y, z, lo, box, periodic, dims, offsets, keys,
                  vals_base, vals_shell, order, interp, aoffsets, akeys,
                  avals_base, avals_shell, corners_base, corners_shell,
                  rho_edge_val, cap, tail_floor, r_scale, inv_span):
    """Padded-target value at ONE point on an AMR leaf pair (base + shell
    vals on the SAME cell geometry): base lookup -> occupancy cut (the
    containing cell's FLAT average >= rho_edge AND inside the domain; a
    linear readout inside a truly-empty coarse cell ramps toward a dense
    neighbour and would classify fabricated density as occupied gas) ->
    occupied returns the LINEAR base value -> else shell lookup at the
    domain-clamped point + the outside-domain tail + the rho_edge cap. Semantically identical to
    PaddedDensity(base, occ_flat, _TailedShell(shell)).
    Returns (value, ok); ok=False = AMR miss."""
    d = 0.0
    if not periodic:
        # clamp into the domain; d = excess distance (0 inside)
        cx = min(max(x, lo[0]), lo[0] + box[0])
        cy = min(max(y, lo[1]), lo[1] + box[1])
        cz = min(max(z, lo[2]), lo[2] + box[2])
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        x, y, z = cx, cy, cz
    tx = (x - lo[0]) / box[0]
    ty = (y - lo[1]) / box[1]
    tz = (z - lo[2]) / box[2]
    if periodic:
        tx -= np.floor(tx)
        ty -= np.floor(ty)
        tz -= np.floor(tz)
    if interp == 3:
        # 'vertex': ONE leaf locate serves the flat occupancy verdict
        # (vals_base[idx] = the containing leaf's own average) and both
        # corner-trilinear evaluations - cheaper than the linear stencil
        # (8 array reads vs 8 level-walk lookups)
        idx, lv = _pwc_index(tx, ty, tz, dims, offsets, keys, order)
        if idx < 0:
            return -1.0, False
        if d == 0.0 and vals_base[idx] >= rho_edge_val:
            return _corner_tri(tx, ty, tz, idx, lv, dims,
                               corners_base), True
        sv = _corner_tri(tx, ty, tz, idx, lv, dims, corners_shell)
    else:
        if d == 0.0:
            tv, tvf, lv = _amr_readout(tx, ty, tz, periodic, dims, offsets,
                                       keys, vals_base, order, interp,
                                       aoffsets, akeys, avals_base)
            if lv < 0:
                return -1.0, False
            if tvf >= rho_edge_val:         # occupied (flat): interp base
                return tv, True
        sv, svf, lv = _amr_readout(tx, ty, tz, periodic, dims, offsets,
                                   keys, vals_shell, order, interp,
                                   aoffsets, akeys, avals_shell)
        if lv < 0:
            return -1.0, False
    if d > 0.0:
        if inv_span > 0.0:                  # decay tail beyond the domain
            r = sv / tail_floor
            if r > 1.0:
                sv = sv * (1.0 + d / r_scale) ** (-np.log(r) * inv_span)
            if sv < tail_floor:
                sv = tail_floor
    if sv > cap:
        sv = cap
    return sv, True


@njit(cache=True, parallel=True)
def _padded_amr_query(pos, lo, box, periodic, dims, offsets, keys,
                      vals_base, vals_shell, order, interp, aoffsets, akeys,
                      avals_base, avals_shell, corners_base, corners_shell,
                      rho_edge_val, cap, tail_floor, r_scale, inv_span,
                      out):
    """One-pass padded-target readout (:func:`_padded_point` per point) with
    no numpy temporaries/subset indexing (the generic composition costs ~9x
    the raw table query). Returns the number of misses."""
    n = pos.shape[0]
    nmiss = 0
    for p in prange(n):
        v, ok = _padded_point(pos[p, 0], pos[p, 1], pos[p, 2], lo, box,
                              periodic, dims, offsets, keys, vals_base,
                              vals_shell, order, interp, aoffsets, akeys,
                              avals_base, avals_shell, corners_base,
                              corners_shell, rho_edge_val, cap,
                              tail_floor, r_scale, inv_span)
        out[p] = v
        if not ok:
            nmiss += 1
    return nmiss


@njit(cache=True, parallel=True)
def _padded_amr_stencil(pos, sup, offs, wts, lo, box, periodic, dims,
                        offsets, keys, vals_base, vals_shell, order, interp,
                        aoffsets, akeys, avals_base, avals_shell,
                        corners_base, corners_shell,
                        rho_edge_val, cap, tail_floor, r_scale, inv_span,
                        out):
    """Kernel-matched readout fused with the padded query: per particle,
    the weighted stencil sum ``sum_q w_q * padded(x_i + sup_i * u_q)``
    (density.KernelMatchedDensity semantics, sup = 2h) evaluated entirely
    in-register - no (n*K,3) offset-position block, no (n,K) value matrix.
    Returns the number of misses."""
    n = pos.shape[0]
    k = offs.shape[0]
    nmiss = 0
    for p in prange(n):
        s = sup[p]
        acc = 0.0
        bad = False
        for q in range(k):
            v, ok = _padded_point(pos[p, 0] + s * offs[q, 0],
                                  pos[p, 1] + s * offs[q, 1],
                                  pos[p, 2] + s * offs[q, 2],
                                  lo, box, periodic, dims, offsets, keys,
                                  vals_base, vals_shell, order, interp,
                                  aoffsets, akeys, avals_base, avals_shell,
                                  corners_base, corners_shell,
                                  rho_edge_val, cap, tail_floor, r_scale,
                                  inv_span)
            if not ok:
                bad = True
            acc += wts[q] * v
        out[p] = acc
        if bad:
            nmiss += 1
    return nmiss


class PaddedTableDensity(PaddedDensity):
    """Fused fast path of :class:`PaddedDensity` for an AMRTable target:
    the shell shares the base table's cell geometry (``_with_values``), so
    the whole padded readout (occupancy cut, shell + tail, cap) runs as
    ONE numba kernel (:func:`_padded_amr_query`) with zero numpy
    temporaries. ``base``/``occupancy``/``shell`` keep the same objects the
    generic composition would use (introspection, cull, tests); only
    ``rho0`` is overridden. ``tail`` disabled (d_max=0) = clamp-flat outside
    the domain, as the generic path without a _TailedShell.

    occ_edge : the occupancy threshold (always ``rho_edge``).
    cap : vacuum-side clamp. Default (the sentinel) = ``occ_edge`` (right
        for an edge-capped shell); pass None for the local-start shell (the
        production fill), whose values legitimately exceed ``rho_edge`` next to bright
        surfaces."""

    _CAP_EDGE = object()                 # sentinel: cap = occ_edge

    def __init__(self, base, occupancy, shell, occ_edge, shell_table,
                 tail_floor=0.0, r_scale=1.0, d_max=0.0, cap=_CAP_EDGE):
        if cap is PaddedTableDensity._CAP_EDGE:
            cap = occ_edge
        super().__init__(base, occupancy, shell, cap=cap)
        self.occ_edge = float(occ_edge)
        self.shell_table = shell_table
        self.tail_floor = float(tail_floor)
        self.r_scale = float(r_scale)
        self.d_max = float(d_max)
        self._inv_span = (1.0 / float(np.log1p(d_max / r_scale))
                          if d_max > 0.0 else 0.0)

    @property
    def _cap_val(self):
        return np.inf if self.cap is None else float(self.cap)

    _DUMMY_CORNERS = np.zeros((1, 8))

    def _corner_pair(self):
        """(corners_base, corners_shell) for the 'vertex' fused path (built
        lazily on the two tables); dummies for the other interps (numba
        needs consistent argument types)."""
        if self.base.interp == 'vertex':
            return (self.base._corner_table(),
                    self.shell_table._corner_table())
        return self._DUMMY_CORNERS, self._DUMMY_CORNERS

    def rho0(self, pos):
        t = self.base
        cb, cs = self._corner_pair()
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        out = np.empty(len(pos))
        nmiss = _padded_amr_query(
            pos, t.lo, t.box, t.periodic, t.dims, t.offsets, t.keys,
            t.vals, self.shell_table.vals, t.order,
            AMRTable._INTERP_CODE[t.interp],
            t.aoffsets, t.akeys, t.avals, self.shell_table.avals,
            cb, cs,
            self.occ_edge, self._cap_val, self.tail_floor, self.r_scale,
            self._inv_span, out)
        if nmiss:
            raise ValueError(f'{nmiss}/{len(pos)} padded-query points not '
                             'covered by the AMR table')
        return out

    def rho0_stencil(self, pos, support, offsets, weights):
        """Fused kernel-matched readout: the weighted stencil sum
        ``sum_q w_q * rho0(pos_i + support_i * u_q)`` in one numba pass
        (density.KernelMatchedDensity delegates here - no offset-position
        block, no value matrix)."""
        t = self.base
        cb, cs = self._corner_pair()
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        support = np.ascontiguousarray(support, dtype=np.float64).reshape(-1)
        out = np.empty(len(pos))
        nmiss = _padded_amr_stencil(
            pos, support, offsets, weights, t.lo, t.box, t.periodic,
            t.dims, t.offsets, t.keys, t.vals, self.shell_table.vals,
            t.order, AMRTable._INTERP_CODE[t.interp],
            t.aoffsets, t.akeys, t.avals,
            self.shell_table.avals, cb, cs, self.occ_edge, self._cap_val,
            self.tail_floor, self.r_scale, self._inv_span, out)
        if nmiss:
            raise ValueError(f'{nmiss}/{len(pos)} stencil points not '
                             'covered by the AMR table')
        return out

    __call__ = rho0


def place_background(occupancy, shell, rho_edge_val, rho_cut, m,
                     region_lo, region_hi, jitter=0.3, seed=0, rho_min=None,
                     n_extra=0):
    """Jittered cubic background lattice in the SHELL (N-aware form).

    Lattice spacing = the finest shell spacing ``(m/rho_edge)**(1/3)`` over
    ``[region_lo, region_hi]`` (the gas bbox + a shell margin). A site is KEPT
    iff it is vacuum (``occupancy < 0.5``) AND above the cutoff
    (``shell(site) > rho_min``, default ``rho_cut``), then
    THINNED by ``shell(site)/rho_edge`` so the surviving number density tracks
    the decaying shell target (grades from ~the finest spacing at the gas edge
    out to ``Delta_bg`` where the taper meets the floor). Deep vacuum
    (``shell <= rho_min``) gets no particle; gas cells get none (occupied).
    ``n_extra`` adds independently sampled particles with the same shell
    density law.  It is the analytic-envelope counterpart of the table
    builder's ``bg_over`` top-up and leaves the target field unchanged.
    All background particles carry mass ``m``. Returns
    ``(pos (n,3), mass (n,))``.

    NB the table/reconstruct path uses sampled backing via the mass-contract
    closure instead; this lattice is the background of
    :func:`build_uniform_envelope`, the analytic/uniform build path.
    ``rho_min`` lowers the vacuum cutoff below ``rho_cut``.
    """
    rng = np.random.default_rng(seed)
    region_lo = np.asarray(region_lo, dtype=np.float64).reshape(3)
    region_hi = np.asarray(region_hi, dtype=np.float64).reshape(3)
    if rho_min is None:
        rho_min = rho_cut
    d_fine = (m / rho_edge_val) ** (1.0 / 3.0)
    axes = [np.arange(region_lo[i] + 0.5 * d_fine, region_hi[i], d_fine)
            for i in range(3)]
    if any(len(a) == 0 for a in axes):
        return np.zeros((0, 3)), np.zeros(0)
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    grid = grid + (rng.random(grid.shape) - 0.5) * jitter * d_fine
    occ = _eval(occupancy, grid)
    rho = _eval(shell, grid)
    keep = (occ < 0.5) & (rho > rho_min * (1.0 + 1e-9))
    prob = np.clip(rho / rho_edge_val, 0.0, 1.0)
    keep &= rng.random(len(grid)) < prob
    pos = np.ascontiguousarray(grid[keep])

    # The lattice above is the analytic-envelope mass closure.  A relaxed
    # free surface retains a one-kernel skin outside the sharp cull contour,
    # so an exact closure can deliver fewer than n_final survivors.  Sample
    # an explicit oversubscription from the SAME vacuum-shell density; this
    # supplies the skin without widening or densifying the physical gas
    # buffer.  The default zero returns the lattice alone.
    n_extra = int(n_extra)
    if n_extra < 0:
        raise ValueError("n_extra must be non-negative")
    if n_extra:
        extra_chunks = []
        remaining = n_extra
        attempted = 0
        span = region_hi - region_lo
        attempt_limit = max(1_000_000, 10_000 * n_extra)
        while remaining:
            ntrial = max(4096, 8 * remaining)
            trial = region_lo + rng.random((ntrial, 3)) * span
            trial_rho = _eval(shell, trial)
            accept = ((_eval(occupancy, trial) < 0.5) &
                      (trial_rho > rho_min * (1.0 + 1e-9)))
            accept &= (rng.random(ntrial) <
                       np.clip(trial_rho / rho_edge_val, 0.0, 1.0))
            accepted = trial[accept]
            if len(accepted):
                take = min(remaining, len(accepted))
                extra_chunks.append(accepted[:take])
                remaining -= take
            attempted += ntrial
            if attempted >= attempt_limit and remaining:
                raise ValueError("could not sample the requested analytic "
                                 "envelope background top-up")
        pos = np.ascontiguousarray(
            np.concatenate([pos, *extra_chunks], axis=0))
    mass = np.full(len(pos), m)
    return pos, mass


def cull_vacuum(pos, mass, aux, occupancy, total_mass):
    """Spatial vacuum cull + mass renormalisation.

    Keep every particle whose FINAL position is resolved gas
    (``occupancy(pos) >= 0.5``) - a background particle that meshed into the gas
    is kept, a gas particle that strayed into vacuum is culled (pure occupancy,
    never a tag test). Renormalise the survivors so their total mass equals
    ``total_mass`` exactly. Returns ``(pos, mass, aux, audit)``."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    keep = _eval(occupancy, pos) >= 0.5
    pos_f = np.ascontiguousarray(pos[keep])
    mass_f = np.ascontiguousarray(mass[keep])
    aux_f = {key: np.asarray(v)[keep] for key, v in (aux or {}).items()}
    m_sum = float(mass_f.sum())
    renorm = total_mass / m_sum if m_sum > 0.0 else 1.0
    mass_f = mass_f * renorm
    audit = dict(n_in=int(len(pos)), n_survivor=int(keep.sum()),
                 n_culled=int((~keep).sum()), renorm=renorm,
                 M_out=float(mass_f.sum()))
    return pos_f, mass_f, aux_f, audit


def table_cells(table):
    """Cell geometry of a table for sampling/envelope/export: (centres (N,3),
    sizes (N,3) per-axis widths, levels (N,), values (N,)) in the table's own
    storage order (aligned with an AMRTable's ``vals``). AMRTable -> its leaf
    set; 3D TableDensity -> the full grid at level 0."""
    if isinstance(table, AMRTable):
        return table.leaves()
    if isinstance(table, TableDensity) and table.dim == 3:
        ax = table.axes
        widths = [np.diff(a).mean() if len(a) > 1 else 1.0 for a in ax]
        cx, cy, cz = np.meshgrid(*ax, indexing='ij')
        centres = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
        n = len(centres)
        sizes = np.tile(np.asarray(widths, dtype=np.float64), (n, 1))
        return (centres, sizes, np.zeros(n, dtype=np.int64),
                table.values.reshape(-1))
    raise TypeError(f'cannot decode cells of {type(table).__name__}')


def _with_values(table, vals):
    """A new table with the same cell geometry but replaced values (aligned
    with :func:`table_cells` order). Supports AMRTable and 3D TableDensity."""
    vals = np.ascontiguousarray(vals, dtype=np.float64)
    if isinstance(table, AMRTable):
        out = AMRTable(table.lo, table.box, table.dims, table.offsets,
                       table.keys, vals, table.order, table.periodic,
                       interp=table.interp)
    elif isinstance(table, TableDensity):
        out = TableDensity(table.axes, vals.reshape(table.values.shape),
                           mapping=None)
    else:
        raise TypeError(f'cannot rebuild table of type {type(table).__name__}')
    for name in ('capture_domain_audit', 'capture_particle_lo',
                 'capture_particle_hi', 'capture_padding_mode'):
        if hasattr(table, name):
            setattr(out, name, getattr(table, name))
    return out


def table_mass(table):
    """Total mass of the table field, sum(rho*V) over cells (negative cell
    values, if any, are treated as vacuum)."""
    _, sizes, _, vals = table_cells(table)
    return float((np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
                  * sizes.prod(1)).sum())


def sample_table(table, n, seed=0):
    """Draw ``n`` positions distributed like the table density: pick cells
    with probability proportional to their mass (rho*V), then a uniform point
    inside each. THE initial draw for a from-scratch table build
    (``./glassgen --table --count`` / ``glassresample --op reconstruct``);
    the relax refines this seed toward the table."""
    centres, sizes, _, vals = table_cells(table)
    w = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None) * sizes.prod(1)
    tot = float(w.sum())
    if tot <= 0.0:
        raise ValueError('cannot sample: table density is all non-positive')
    rng = np.random.default_rng(seed)
    n = int(n)
    idx = rng.choice(len(w), size=n, p=w / tot)
    return np.ascontiguousarray(
        centres[idx] + (rng.random((n, 3)) - 0.5) * sizes[idx])


def build_table_envelope(table, gas_mass, n_final, box=None,
                         c=NSMOOTH_DEFAULT, k=None,
                         seed=0, nsmooth=NSMOOTH_DEFAULT,
                         pad_frac=0.1, tail_reach=None,
                         n_frame=None, nl0=None, bg_over=0.0,
                         base_density=None):
    """Envelope dict for a TABLE target (AMRTable or 3D TableDensity) - the
    table-native analog of :func:`build_uniform_envelope`, driven by the
    glassgen table builder (:func:`glassgen.progressive.build_from_table`).

    From ``m = gas_mass/n_final``: ``rho_edge`` (cumulative faint-tail cut at
    ``c*m`` on the cell values); the vacuum decays ``rho_edge -> rho_frame =
    rho_edge/k**6``. Occupancy is analytic and sharp on the table
    (``table.rho0(pos) >= rho_edge``); the shell is the same cell set with the
    sub-``rho_edge`` cells emptied (the fill trigger is ``floor=rho_edge``) and
    refilled by :func:`_fill_empty`: ONE strictly-decreasing profile from the
    local edge value down to ``rho_frame`` at the frame corner (in the cells
    inside the domain, analytically outside it via :class:`_TailedShell`) -
    no plateau anywhere, so a re-anneal-kicked stray always feels a
    home-pointing gradient. Steepness comes from the launch knob ``nl0``
    (tail_reach solved per build) or a legacy explicit ``tail_reach``.
    The background is SAMPLED backing: the mass-contract closure is the sole
    sampler - ``n_bg = (M_target - gas_mass)/m`` fillers drawn from the
    vacuum cells w.p. ``shell*V`` at particle mass m, so backing number
    density tracks the target right up to bright surfaces (no ``delta_bg``
    quantization; ``delta_bg`` serves as a diagnostic and the frame-sizing
    kernel scale).
    ``bg_over`` OVERSUBSCRIBES the closure by ``bg_over*n_final`` extra
    fillers (same shell-mass sampling) to cancel the systematic post-cull
    count undershoot (renorm f > 1); set it ~ the observed f-1 (0 = exact
    contract, the default).
    NB the cut is computed on the values as stored - capture the table RAW
    (genuine zeros) for an exact cut; a pre-filled (decay) table shifts
    ``rho_edge`` slightly low.

    For an OPEN build (``box is None``) the dict also carries ``frame_box``/
    ``frame_centre`` - the periodic frame sized by :func:`frame_open_box`
    (per-axis margins + the kernel-fits-the-box cap from
    ``Delta_bg``); ``nsmooth``/``pad_frac`` feed that rule. Returns the
    ``envelope=`` dict for :func:`glassgen.progressive.relax_progressive`.

    The fill always LAUNCHES from the nearest occupied cell's own value
    (continuous at the surface by construction).

    ``base_density`` is an optional exact gas-side evaluator for a table that
    exists only as a Cartesian volume quadrature (for example a 3-D sampling
    of a mapped ``(R, |z|)`` target).  Cell values/volumes still determine the
    edge, shell mass, and sampled backing, while occupancy and the padded
    target read ``base_density`` inside the table domain.  The default keeps
    the physical table as both quadrature and gas target."""
    centres, sizes, _, vals = table_cells(table)
    vals = np.asarray(vals, dtype=np.float64)
    vols = sizes.prod(1)
    base = table if base_density is None else base_density
    m = float(gas_mass) / int(n_final)
    # k from the floor-backability count n_frame unless legacy-explicit
    k = resolve_bg_k(k, n_frame, nsmooth=nsmooth)
    re, delta_bg = edge_floor_spacing(vals, vols, m, c=c, k=k)
    shell_cap = None                    # local-start launch
    occ_cells = vals >= re
    # occupancy is DOMAIN-AWARE: outside the table's bbox the AMR/grid readout
    # clamps to the boundary cell, so a stray beyond the domain could read
    # >= rho_edge and dodge the vacuum cull.
    # It is also FLAT (cell averages): the linear readout inside a
    # truly-empty coarse cell ramps toward a dense neighbour's average and
    # marks fabricated density as occupied gas.
    dom_lo = (centres - 0.5 * sizes).min(0)
    dom_hi = (centres + 0.5 * sizes).max(0)
    occ_view = base
    if base_density is None and isinstance(table, AMRTable) \
            and table.interp != 'flat':
        occ_view = AMRTable(table.lo, table.box, table.dims, table.offsets,
                            table.keys, table.vals, table.order,
                            table.periodic, interp='flat')

    def _occ(p, t=occ_view, e=re, lo=dom_lo, hi=dom_hi):
        p = np.asarray(p, dtype=np.float64)
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        return ((_eval(t, p) >= e) & inside).astype(np.float64)

    occupancy = CallableDensity(_occ)
    # A raw capture may deliberately extend well beyond the physical object
    # (full compact-support, input-rho-edge, or constant padding).  Those
    # sub-rho_edge leaves are scaffold storage, not part of the resolved gas
    # extent.  Size the relaxation frame from the ACTIVE cell boxes so raw
    # capture padding cannot silently inflate rho_frame's periodic frame.
    cell_lo = centres - 0.5 * sizes
    cell_hi = centres + 0.5 * sizes
    region_lo = cell_lo[occ_cells].min(0)
    region_hi = cell_hi[occ_cells].max(0)
    particle_lo = getattr(table, 'capture_particle_lo', None)
    particle_hi = getattr(table, 'capture_particle_hi', None)
    capture_padded = particle_lo is not None and particle_hi is not None
    if capture_padded:
        # Include every input particle centre even when a faint outlier falls
        # below the output-count rho_edge classification, but exclude the
        # deliberately added raw-table padding beyond that source bbox.
        region_lo = np.minimum(region_lo, particle_lo)
        region_hi = np.maximum(region_hi, particle_hi)
    else:
        # Backward compatibility: an older/external table has no way to tell
        # us which part of its domain was deliberate capture padding.
        region_lo, region_hi = dom_lo, dom_hi
    # frame BEFORE the fill (the tail needs its extent): open build ->
    # frame_open_box; explicit periodic box -> the box IS the frame.
    frame_box_ = frame_centre = None
    if box is None:
        frame_box_, frame_centre = frame_open_box(
            region_lo, region_hi, delta_bg, nsmooth=nsmooth,
            pad_frac=pad_frac, k=k)
        frame_extent = frame_box_
        frame_lo = frame_centre - 0.5 * frame_box_
        frame_hi = frame_centre + 0.5 * frame_box_
    else:
        frame_extent = np.asarray(box, dtype=np.float64).reshape(3)
        frame_lo = -0.5 * frame_extent
        frame_hi = 0.5 * frame_extent
    # Exact per-cell volume inside the relaxation frame.  Usually this is the
    # full leaf volume; the intersection matters when an exported raw table is
    # wider than the active-cell-derived frame.
    overlap_size = np.maximum(
        np.minimum(cell_hi, frame_hi) - np.maximum(cell_lo, frame_lo), 0.0)
    frame_cell_vols = overlap_size.prod(1)
    # whole-frame vacuum tail: the shell decays as a power law, reaching
    # rho_frame only at the frame edge - a kicked stray anywhere in the
    # frame feels a home-pointing gradient.
    rho_frame = re / k ** 6               # vacuum decays rho_edge -> rho_frame
    tail_scale = 0.5 * float(np.max(frame_extent))
    # the profile steepness comes from the launch-resolvability knob
    # nl0 (tail_reach solved per build, so the launch N_L is pinned at
    # every N) unless a legacy explicit tail_reach bypasses the solve;
    # either way the resonance-band guard runs on the resulting profile.
    nl0_eff = p_eff = None
    spacing_edge = (m / re) ** (1.0 / 3.0)
    if tail_reach is None:
        tail_reach = solve_tail_reach(
            tail_scale, spacing_edge,
            NL0_DEFAULT if nl0 is None else float(nl0), k,
            nsmooth=nsmooth)
    elif nl0 is not None:
        warnings.warn('explicit tail_reach overrides nl0 (legacy path)')
    nl0_eff, p_eff = check_scaffold_resolvability(
        tail_scale, spacing_edge, tail_reach, k, nsmooth=nsmooth)
    shell_vals = _fill_empty(vals.copy(), 'decay', centres, box,
                             floor=re, cap=shell_cap,
                             tail_scale=tail_scale,
                             tail_floor=rho_frame,
                             tail_reach=tail_reach)
    shell = _with_values(table, shell_vals)
    # outside the table domain the readout clamps flat -> continue the decay
    # analytically out to the frame (open builds only; a periodic box wraps)
    padded_shell = shell
    d_out, r_scale = 0.0, 1.0
    if box is None:
        dom_half = 0.5 * (dom_hi - dom_lo)
        outside_reach = np.maximum(
            np.maximum(dom_lo - frame_lo, frame_hi - dom_hi), 0.0)
        d_out = float(np.linalg.norm(outside_reach))
        if d_out > 0.0:
            r_scale = max(float(np.linalg.norm(dom_half)), 1e-12)
            padded_shell = _TailedShell(shell, dom_lo, dom_hi, r_scale,
                                        d_out, rho_frame)
    if base_density is None and isinstance(table, AMRTable):
        # fused one-kernel readout (same values as the generic composition;
        # the relax evaluates this every particle every iteration). All
        # four interps run fused - 'vertex' passes the two corner tables.
        padded = PaddedTableDensity(table, occupancy, padded_shell, re,
                                    shell, tail_floor=rho_frame,
                                    r_scale=r_scale, d_max=d_out,
                                    cap=shell_cap)
    else:
        padded = PaddedDensity(base, occupancy, padded_shell,
                               cap=shell_cap)
    # MASS-CONTRACT CLOSURE (the sole background sampler): the ensemble
    # must carry the target's WHOLE mass integral - occupied base + in-domain
    # shell + the analytic outside-domain tail - as particles inside the
    # EXISTING frame (no box growth, which would risk kernel wrap).
    # Exact placement does not matter (the relax redistributes); what matters
    # is that no mass has to be borrowed from the gas: n_bg = (M_target -
    # M_gas)/m fillers are sampled from the shell's own mass distribution
    # (vacuum cells w.p. shell*V, uniform inside the cell - the dense skirt
    # gets most of them, the far tail's mass representatives just sit closer
    # in). Backing number density tracks the target right up to the bright
    # surfaces a fixed lattice spacing could never represent, with no
    # delta_bg quantization.
    # bg_over: OVERSUBSCRIBE the closure by bg_over*N extra
    # fillers beyond the exact contract. The exact closure systematically
    # undershoots the delivered count (renorm f 1.02-1.05): at stop time the
    # scaffold holds the contract mass PLUS the strays/skin layer, so the
    # interior is short by ~f-1 of N. Excess ensemble mass is force-free at a
    # uniform rho/rho0 offset and distributes ~prop rho0, so ~M_occ/M_target
    # (~95%) of the extra lands inside the contour - bg_over ~ (f-1) aims the
    # post-cull survivor count at N (renorm f ~ 1).
    M_occ = float((vals[occ_cells] * frame_cell_vols[occ_cells]).sum())
    M_shell_in = float(
        (shell_vals[~occ_cells] * frame_cell_vols[~occ_cells]).sum())
    M_out = 0.0
    if box is None and d_out > 0.0:
        rng = np.random.default_rng(seed + 7)
        q = frame_lo + rng.random((200_000, 3)) * frame_box_
        outside = ~np.all((q >= dom_lo) & (q <= dom_hi), axis=1)
        v_frame = float(np.prod(frame_box_))
        v_out = max(0.0, v_frame - float(frame_cell_vols.sum()))
        if outside.any() and v_out > 0.0:
            M_out = float(_eval(padded_shell, q[outside]).mean()) * v_out
    M_target = M_occ + M_shell_in + M_out
    n_bg_want = max(0, int(round((M_target - float(gas_mass)) / m
                                 + float(bg_over) * int(n_final))))
    bg_pos, bg_mass = np.zeros((0, 3)), np.zeros(0)
    rng = np.random.default_rng(seed + 13)
    if n_bg_want > 0:
        vac = np.where((~occ_cells) & (frame_cell_vols > 0.0))[0]
        w = shell_vals[vac] * frame_cell_vols[vac]
        if len(vac) == 0 or float(w.sum()) <= 0.0:
            raise ValueError('envelope frame has scaffold mass but no '
                             'in-table vacuum cells to sample')

        def _draw_from_vacuum_cells(n_draw):
            pick = rng.choice(len(vac), size=int(n_draw), p=w / w.sum())
            cells = vac[pick]
            sample_lo = np.maximum(cell_lo[cells], frame_lo)
            sample_hi = np.minimum(cell_hi[cells], frame_hi)
            return (sample_lo
                    + rng.random((len(cells), 3)) * (sample_hi - sample_lo))

        if base_density is None:
            # Preserve the established table path's deterministic draw exactly.
            # Its flat cell occupancy makes every point in ``vac`` vacuum.
            bg_pos = _draw_from_vacuum_cells(n_bg_want)
        else:
            # A quadrature cell is classified by its centre, but the exact
            # mapped contour can cross it.  Reject sub-cell draws that land on
            # the gas side so every scaffold seed obeys the exact occupancy
            # contract used by the final cull.
            chunks = []
            remaining = n_bg_want
            attempted = 0
            attempt_limit = max(1_000_000, 1000 * n_bg_want)
            while remaining:
                ntrial = max(1024, 2 * remaining)
                trial = _draw_from_vacuum_cells(ntrial)
                accepted = trial[_eval(occupancy, trial) < 0.5]
                if len(accepted):
                    take = min(remaining, len(accepted))
                    chunks.append(accepted[:take])
                    remaining -= take
                attempted += ntrial
                if attempted >= attempt_limit and remaining:
                    raise ValueError('could not sample mapped-envelope backing '
                                     'outside the exact occupancy contour')
            bg_pos = np.ascontiguousarray(np.concatenate(chunks, axis=0))
        bg_mass = np.full(len(bg_pos), m)
    env = dict(density=padded, occupancy=occupancy, bg_pos=bg_pos,
               bg_mass=bg_mass, gas_mass=float(gas_mass),
               rho_edge=re, rho_frame=rho_frame,
               delta_bg=delta_bg, m=m)
    env['M_target'] = M_target
    requested_topup = int(round(float(bg_over) * int(n_final)))
    env['n_bg_topup'] = min(requested_topup, len(bg_pos))
    env['n_bg_contract'] = len(bg_pos) - env['n_bg_topup']
    env['bg_over'] = float(bg_over)
    env['bg_k'] = k
    env['tail_reach'] = tail_reach
    env['active_lo'], env['active_hi'] = region_lo, region_hi
    env['capture_padded'] = bool(capture_padded)
    if nl0_eff is not None:                      # launch N_L + exponent
        env['nl0'] = nl0_eff
        env['profile_p'] = p_eff
    if frame_box_ is not None:                   # open build: periodic frame
        env['frame_box'], env['frame_centre'] = frame_box_, frame_centre
    return env


def density_quadrature_table(density, pos, gas_mass, n_final, ngrid=None,
                             nsmooth=NSMOOTH_DEFAULT, box=None):
    """Sample an evaluator into volume-correct Cartesian cells.

    This is the representation adapter needed by the proven table-envelope
    backend when the physical target is analytic or a mapped 1-D/2-D table and
    therefore has no Cartesian cell volumes.  The returned ``TableDensity`` is
    a quadrature/seed object only; :func:`build_open_density_envelope` retains
    ``density`` itself as the exact gas-side target and occupancy readout.

    The domain matches :func:`build_uniform_envelope`: the realised input bbox
    plus one typical kernel support and one ``2*N**(1/3)``-grid cell.  The
    domain margin is independent of ``ngrid`` so convergence scans change only
    the quadrature, not the physical integration region.  Axes are true CELL
    CENTRES, making :func:`table_cells` recover the intended domain and exact
    ``dx*dy*dz`` volumes.

    ``box`` is the periodic relaxation frame, when there is one.  Particles
    wrap inside it, so the quadrature domain is grown to contain it: every
    reachable point must have a cell volume and a scaffold value.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
        raise ValueError('density quadrature needs non-empty (N,3) positions')
    n_final = int(n_final)
    if n_final <= 0:
        raise ValueError('n_final must be positive')
    gas_mass = float(gas_mass)
    if not np.isfinite(gas_mass) or gas_mass <= 0.0:
        raise ValueError('gas_mass must be finite and positive')
    if ngrid is None:
        ngrid = MAPPED_NGRID_DEFAULT
    ngrid = int(ngrid)
    if ngrid < 8:
        raise ValueError('ngrid must be at least 8 cells per axis')

    m = gas_mass / n_final
    particle_lo = pos.min(0)
    particle_hi = pos.max(0)
    span = particle_hi - particle_lo
    # Physical margin from a 2*N**(1/3) grid, independent of the quadrature
    # ngrid.
    margin_grid = int(np.clip(round(2.0 * n_final ** (1.0 / 3.0)), 24, 128))
    margin_cell = float(np.max(span)) / margin_grid
    rho_typ = max(float(np.median(_eval(density, pos))), 1e-300)
    margin = margin_cell + kernel_support((m / rho_typ) ** (1.0 / 3.0),
                                          nsmooth)
    region_lo = particle_lo - margin
    region_hi = particle_hi + margin
    if box is not None:
        half_box = 0.5 * np.asarray(box, dtype=np.float64).reshape(3)
        region_lo = np.minimum(region_lo, -half_box)
        region_hi = np.maximum(region_hi, half_box)
    widths = (region_hi - region_lo) / ngrid
    axes = tuple(region_lo[d] + (np.arange(ngrid) + 0.5) * widths[d]
                 for d in range(3))
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    centres = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vals = _eval(density, centres).reshape((ngrid, ngrid, ngrid))
    table = TableDensity(axes, vals, mapping=None)
    # build_table_envelope uses these to size the active frame from all source
    # particle centres while excluding the deliberate quadrature padding.
    table.capture_particle_lo = particle_lo.copy()
    table.capture_particle_hi = particle_hi.copy()
    table.capture_padding_mode = 'mapped-cartesian-quadrature'
    table.quadrature_region_lo = region_lo.copy()
    table.quadrature_region_hi = region_hi.copy()
    table.quadrature_ngrid = ngrid
    return table


def build_open_density_envelope(density, pos, gas_mass, n_final,
                                c=NSMOOTH_DEFAULT, ngrid=None, seed=0,
                                nsmooth=NSMOOTH_DEFAULT, pad_frac=0.1,
                                n_frame=None, nl0=None, bg_over=0.0,
                                box=None):
    """Free-surface envelope adapter for an analytic or mapped evaluator.

    Construct only the Cartesian volume representation missing from such a
    target, then delegate the complete scaffold and sampled mass closure to
    :func:`build_table_envelope`.  The exact evaluator remains the gas target
    and final cull predicate.  ``seed_table`` in the returned dict is the table
    from which an open ``-I`` CLI run draws a fresh ``N=len(-I)`` seed.

    ``box=None`` is the open build: :func:`build_table_envelope` sizes its own
    periodic frame.  Passing a ``box`` instead keeps a target's free surfaces
    inside a frame the caller already fixed - the mixed-boundary case, where
    one axis is genuinely periodic (a slab's ``z`` wrap) and the surfaces sit
    in the interior.  The frame is then the box, and the background is sampled
    only inside it.
    """
    quadrature = density_quadrature_table(
        density, pos, gas_mass, n_final, ngrid=ngrid, nsmooth=nsmooth,
        box=box)
    env = build_table_envelope(
        quadrature, gas_mass, n_final, box=box, c=c, seed=seed,
        nsmooth=nsmooth, pad_frac=pad_frac, n_frame=n_frame, nl0=nl0,
        bg_over=bg_over, base_density=density)
    env['seed_table'] = quadrature
    env['quadrature_ngrid'] = quadrature.quadrature_ngrid
    env['quadrature_mass'] = table_mass(quadrature)
    env['quadrature_region_lo'] = quadrature.quadrature_region_lo
    env['quadrature_region_hi'] = quadrature.quadrature_region_hi
    return env


def build_uniform_envelope(density, pos, gas_mass, n_final,
                           c=NSMOOTH_DEFAULT, k=2.0, ngrid=None,
                           jitter=0.3, seed=0,
                           nsmooth=NSMOOTH_DEFAULT, pad_frac=0.1,
                           tail_reach=6.0, bg_over=0.0):
    """Assemble the envelope dict for ``./glassgen`` (glassgen-native, no
    sph_interp) from the analytic/table target ``density`` and the gas pre-IC.

    All fields are in OPEN/world coordinates (relax_progressive frames the padded
    periodic box itself). Steps:

    - A uniform grid over the gas bbox + a kernel-support shell margin
      samples ``vals = density.rho0(centres)``.
    - ``rho_edge``/``rho_frame``/``Delta_bg`` come from ``m = gas_mass/n_final``
      (:func:`edge_floor_spacing`). The vacuum fill is the no-floor profile
      with a LITERAL ``tail_reach`` (default 6.0; no ``nl0`` solve here).
    - **occupancy** is analytic and sharp: ``density.rho0(pos) >= rho_edge``
      (no grid-graining; assumes the target tapers below ``rho_edge`` outside the
      gas - true for free-surface tables, NOT for a table that clamps to a
      nonzero edge value at large radius).
    - **shell** is a 3D ``TableDensity`` of the grid with sub-``rho_edge``
      cells refilled by the no-floor profile (:func:`_fill_empty` 'decay').
    - **background** is placed by :func:`place_background` over the same region.

    ``bg_over`` oversubscribes the background by ``bg_over*n_final`` sampled
    shell particles, matching :func:`build_table_envelope`.  This compensates
    a measured post-relaxation surface skin without changing the occupancy
    contour or physical target.

    Returns the ``envelope=`` dict consumed by
    :func:`glassgen.progressive.relax_progressive`: keys ``density`` (a
    :class:`PaddedDensity`), ``occupancy``, ``bg_pos``, ``bg_mass``, ``gas_mass``
    (plus ``rho_edge``/``rho_frame``/``delta_bg``/``m``, ``frame_box``/
    ``frame_centre`` and ``bg_over``/``n_bg_topup`` for reporting). ``bg_pos`` may
    be empty if the target has no sub-``rho_edge`` cells (a filled box) - then the
    caller can treat the build as envelope-off."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    m = float(gas_mass) / int(n_final)
    lo = pos.min(0)
    hi = pos.max(0)
    span = hi - lo
    if ngrid is None:                    # ~2 cells per gas mean-spacing, clamped
        ngrid = int(round(2.0 * n_final ** (1.0 / 3.0)))
    ngrid = int(np.clip(ngrid, 24, 128))
    # shell margin: one kernel support at the gas's typical output spacing,
    # plus a cell (kernel support is the physically motivated clearance scale)
    cell0 = float(np.max(span)) / ngrid
    rho_typ = max(float(np.median(_eval(density, pos))), 1e-300)
    margin = cell0 + kernel_support((m / rho_typ) ** (1.0 / 3.0), nsmooth)
    region_lo = lo - margin
    region_hi = hi + margin
    axes = [np.linspace(region_lo[i], region_hi[i], ngrid) for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    centres = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    cell_w = np.array([a[1] - a[0] for a in axes])
    dv = float(np.prod(cell_w))
    vals = _eval(density, centres)
    re, delta_bg = edge_floor_spacing(
        vals, np.full(len(vals), dv), m, c=c, k=k)
    # analytic sharp occupancy (world coords)
    occupancy = CallableDensity(
        lambda p, d=density, e=re: (_eval(d, p) >= e).astype(np.float64))
    # periodic frame: the region already includes the shell margin, so only
    # the pad_frac/H_edge margins + the kernel-fits-the-box cap apply on top
    frame_box_, frame_centre = frame_open_box(
        region_lo, region_hi, delta_bg, nsmooth=nsmooth,
        pad_frac=pad_frac, k=k)
    # shell = grid with sub-rho_edge cells refilled by the no-floor profile,
    # launched from the local edge value
    shell_cap = None
    rho_frame = re / k ** 6               # vacuum decays rho_edge -> rho_frame
    tail_scale = 0.5 * float(np.max(frame_box_))
    shell_vals = _fill_empty(
        vals.copy(), 'decay', centres, box=None,
        floor=re, cap=shell_cap,
        tail_scale=tail_scale, tail_floor=rho_frame,
        tail_reach=tail_reach)
    shell = TableDensity(tuple(axes),
                         shell_vals.reshape(ngrid, ngrid, ngrid), mapping=None)
    # continue the decay beyond the grid (the readout clamps flat there)
    reg_half = 0.5 * (region_hi - region_lo)
    d_out = float(np.linalg.norm(
        np.maximum(0.5 * frame_box_ - reg_half, 0.0)))
    padded_shell = shell
    if d_out > 0.0:
        r_scale = max(float(np.linalg.norm(reg_half)), 1e-12)
        padded_shell = _TailedShell(shell, region_lo, region_hi, r_scale,
                                    d_out, rho_frame)
    padded = PaddedDensity(density, occupancy, padded_shell, cap=shell_cap)
    bg_over = float(bg_over)
    if not np.isfinite(bg_over) or bg_over < 0.0:
        raise ValueError("bg_over must be finite and non-negative")
    n_bg_topup = int(round(bg_over * int(n_final)))
    bg_pos, bg_mass = place_background(occupancy, shell, re, rho_frame,
                                       m, region_lo, region_hi,
                                       jitter=jitter, seed=seed,
                                       n_extra=n_bg_topup)
    env = dict(density=padded, occupancy=occupancy, bg_pos=bg_pos,
               bg_mass=bg_mass, gas_mass=float(gas_mass),
               rho_edge=re, rho_frame=rho_frame,
               delta_bg=delta_bg, m=m,
               frame_box=frame_box_, frame_centre=frame_centre)
    env['bg_over'] = bg_over
    env['n_bg_topup'] = n_bg_topup
    return env


# --------------------------------------------------------------------------- #
#  Table fill helpers (re-exported by reconstruct.py)
# --------------------------------------------------------------------------- #

def _fill_empty(vals, empty, centres, box, floor_frac=1e-3, floor=None,
                cap=None, tail_scale=None, tail_floor=None, tail_reach=6.0):
    """Make empty-space cells relax-safe. A cell counts as EMPTY when its
    density is below the vacuum floor - so both genuine zeros AND faint
    sub-floor cells qualify. (A petkova capture never yields exact zeros: its
    vacuum cells carry tiny-but-positive densities, so a ``<=0`` test would miss
    them entirely and leave the vacuum unfilled - hence the floor test.)
      'decay' - the no-floor vacuum profile (the production fill): ONE
          strictly-decreasing profile from
          the (capped) nearest-occupied value down to ``tail_floor``
          (rho_frame) at the frame corner ``d = tail_scale``, mirroring
          :meth:`_TailedShell.rho0` so the in-domain fill and the
          outside-domain tail share a form and join seamlessly at the
          domain face: ``rho(d) = edge * (1 + d/r_scale)**-p`` with the
          per-cell exponent ``p`` solved so ``rho(tail_scale) = tail_floor``
          exactly. No plateau anywhere => the mid-field always drains (a
          flat rho0 region is a parking lot).
          The residual flat is only at ``tail_floor`` beyond the frame
          corner (benign - culled).
      'edge'  - constant nearest-occupied fill (no taper; diagnostics only).
      'min'   - global minimum positive value.
      'zero'  - keep as-is (NOT relax-safe; raw captures).
    floor : explicit fill trigger (the N-aware ``rho_edge``); cells below it
        are emptied and refilled. When None it is ``floor_frac *
        median(positive density)`` (relative fallback).
    cap : clamp on the fill's START value (the nearest-occupied density).
        None = launch from the local value; ``cap=rho_edge`` clamps the
        launch at the edge density.
    tail_scale, tail_floor : the frame-corner distance and rho_frame - the
        profile's far anchor. REQUIRED for 'decay'.
    tail_reach : steepness: ``r_scale = tail_scale/tail_reach`` (larger =>
        steeper near the gas, shallower far out). Solved per build from the
        launch-resolvability knob nl0 upstream; explicit values are
        the legacy override."""
    vals = np.array(vals, dtype=np.float64)
    if empty == 'zero':
        return vals
    if empty not in ('edge', 'decay', 'min'):
        raise ValueError(f"unknown empty policy {empty!r}")
    posv = vals[vals > 0.0]
    if posv.size == 0:                       # nothing occupied anywhere
        vals[:] = 1.0 if floor is None else floor
        return vals
    if floor is None:
        floor = floor_frac * float(np.median(posv))
    bad = vals < floor                       # empty = below the vacuum floor
    if not bad.any():
        return vals
    good = ~bad
    if not good.any():
        vals[:] = floor
        return vals
    if empty == 'min':
        vals[bad] = vals[good].min()
        return vals
    from scipy.spatial import cKDTree
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if box is not None:
        lo = centres.min(axis=0)
        wrapped = np.mod(centres - lo, box)
        tree = cKDTree(wrapped[good], boxsize=box)
        dist, nn = tree.query(wrapped[bad], workers=-1)
    else:
        tree = cKDTree(centres[good])
        dist, nn = tree.query(centres[bad], workers=-1)
    edge = vals[good][nn]
    if cap is not None:
        edge = np.minimum(edge, cap)
    if empty == 'edge':
        vals[bad] = edge
        return vals
    # 'decay' = the no-floor profile (see docstring)
    if tail_scale is None or tail_floor is None:
        raise ValueError("'decay' needs tail_scale and tail_floor (the "
                         'frame-corner distance and rho_frame)')
    rs = float(tail_scale) / max(float(tail_reach), 1e-12)
    span = max(np.log1p(float(tail_scale) / rs), 1e-12)
    p = np.log(np.maximum(edge / tail_floor, 1.0)) / span
    fac = (1.0 + dist / rs) ** (-p)
    vals[bad] = np.maximum(edge * fac, tail_floor)
    return vals


def _pad_periodic_box(pos, pad_frac=0.1):
    """Synthesise a periodic box (glassgen convention: domain CENTRED at the
    origin, [-L/2, L/2]) enclosing `pos` with a `pad_frac` vacuum margin per
    side, for relaxing an OPEN-domain IC. The glassgen relax MUST be periodic:
    an open boundary cannot contain the pseudo-pressure forces, so near-
    isolated particles (huge h) run away to infinity. A padded periodic box contains them (an
    escapee wraps into vacuum and is culled), while the margin - larger than
    the edge kernel reach - keeps the gas from feeling its periodic images.
    Returns (box (3,), centre (3,)): shift pos -= centre to enter the frame,
    += centre to leave it."""
    lo = pos.min(0)
    hi = pos.max(0)
    centre = 0.5 * (lo + hi)
    box = (hi - lo) * (1.0 + 2.0 * pad_frac)
    return np.ascontiguousarray(box), np.ascontiguousarray(centre)
