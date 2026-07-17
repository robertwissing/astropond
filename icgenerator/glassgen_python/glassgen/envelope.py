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
background is a jittered lattice whose number density tracks that decay. See
RECONSTRUCT_PLAN.md.

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
#                             Delta(rho) <= alpha * rho/|grad rho| - the SAME
#                             alpha as progressive.auto_coarse_count (cascade
#                             coarse-level bound, calibrated on mhdcollapse);
#                             the cubic scaffold is that criterion's marginal
#                             solution (scaffold='cubic', 2026-07-13 redesign)


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
    ``Delta_bg = k*(m/rho_edge)**1/3``. Two-rung ladder (2026-07-14): the vacuum
    field decays rho_edge -> rho_frame = rho_edge/k**6; rho_floor retired (it was
    the redundant fill trigger + shelf bottom).
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


#: M20 defaults (RECONSTRUCT_PLAN): floor backability count and scaffold
#: launch resolvability - the measured k1.25-tr12 anchor cell.
NFRAME_DEFAULT = 16.0
NL0_DEFAULT = 1.35


def resolve_bg_k(k=None, n_frame=None, nsmooth=NSMOOTH_DEFAULT):
    """M20: background coarseness from the floor-BACKABILITY count.

    ``n_frame`` = expected particles of mass m per EDGE-KERNEL volume
    (V_e = nsmooth*m/rho_edge, so n(rho) = nsmooth*rho/rho_edge) at the
    profile floor: rho_frame = rho_edge*(n_frame/nsmooth), i.e.
    k = (nsmooth/n_frame)**(1/6). n_frame=1 <=> the old k=2 default (the
    floor AT the unbackable density - a structurally thin buffer);
    n_frame=16 (default) <=> k~1.26, the fat backed buffer of the M19
    grid's most reproducible cell. Legacy escape: pass ``k`` explicitly
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
    """M20: tail_reach (TR) that LAUNCHES the no-floor profile at ``nl0``
    kernel supports.

    The profile rho(d) = rho_edge*(1+d/rs)**(-p) (rs = tail_scale/TR,
    p = 6*ln(k)/ln(1+TR)) has near-gas e-fold length L(0) = rs/p; solve
    L(0) = nl0 * 2h_e with 2h_e = kernel_support(spacing_edge). N_L ~ 1
    is the measured metastable RESONANCE (the M19 valley); pinning the
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
    """M20 guard on the scaffold's resolvability curve
    N_L(d) = nl0*(1+d/rs)**(1-p/3).

    Warns on the measured failure geometries (the M19 grid, decoded in
    kernel units): (a) p >= 3 - N_L flat or DESCENDING, so the profile
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
            'backing is thin (the M19 k2-tr3 pathology) - raise n_frame or '
            'nl0')
        return nl0, p
    if band[0] <= nl0 <= band[1] and n_frame_eff < 10.0:
        warnings.warn(
            f'scaffold LAUNCHES in the N_L resonance band (nl0={nl0:.2f}) '
            f'on a thin floor buffer (n_frame_eff={n_frame_eff:.1f} < 10): '
            'the M19 valley geometry (k1.5-tr6) - move nl0 off ~1 or '
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
    """Size the periodic frame for an OPEN (free-surface) build - plan task M2
    (2026-07-10): per-axis margins AND a global kernel-fits-the-box cap, so no
    kernel is ever larger than any box axis (a period shorter than a kernel
    support DIAMETER lets the kernel wrap onto its own image: the 80k galaxy
    run had box_z = 22.4 against background supports of ~30).

      margin_i = max(pad_frac*span_i, H_edge)   (H_edge = support
                 of an edge-gas kernel, Delta_edge = Delta_bg/k)
      box_i    = span_i + 2*margin_i
      box_i    = max(box_i, 2*H_bg)                        (H_bg = support of
                 a background kernel at spacing Delta_bg)

    span_lo/span_hi : world bbox that must sit inside the frame (gas domain +
    background region). Returns (box (3,), centre (3,)): shift positions by
    ``-centre`` to enter the centred periodic frame.

    cap_delta : spacing of the SPARSEST kernels the frame must contain
    (defaults to ``delta_bg``). No caller overrides it since the backed-tail
    experiment was retired (2026-07-16, CLEANUP_PLAN C1); kept because the
    sampled backing's faintest fillers can in principle sit at spacing up to
    ``k*delta_bg`` (toward ``rho_frame``) - if a small/thin domain ever shows
    kernel wrap, pass ``cap_delta=k*delta_bg`` (see the plan's RECHECK
    section, caveat 2)."""
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

    cap : clamp on the VACUUM side, normally ``rho_edge`` (2026-07-10 seam
        fix). The shell is only ever read below the occupancy contour, where
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
    """Continue the shell's vacuum decay OUTSIDE the table domain (task M6).

    The shell is a cell table over the capture domain; beyond its bbox the
    AMR/grid readout clamps to the boundary cell, so the frame's outer vacuum
    (which M2's kernel-fits-the-box rule can make much larger than the
    domain, e.g. the galaxy's box_z ~59 vs domain ~19) reads FLAT - no
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


class _CubicTail(DensityField):
    """Continue the CUBIC scaffold outside the table domain (scaffold='cubic').

    The cubic profile is u-linear: u = rho**(-1/3) grows with distance at the
    universal slope 1/(3*alpha*m**(1/3)) regardless of the launch value, so
    the continuation past the domain bbox needs only the boundary value ``v``
    and the excess distance ``d``:

        rho(d) = (v**(-1/3) + d/scale)**-3,   scale = 3*alpha*m**(1/3)

    - exact (the same profile the in-domain fill follows), continuous at the
    domain faces, monotone home-pointing, no per-query exponent solve and no
    separate floor: the frame-corner value IS the derived rho_frame."""

    def __init__(self, shell, dom_lo, dom_hi, scale):
        self.shell = shell
        self.dom_lo = np.asarray(dom_lo, dtype=np.float64)
        self.dom_hi = np.asarray(dom_hi, dtype=np.float64)
        self.scale = float(scale)

    def rho0(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        cl = np.clip(pos, self.dom_lo, self.dom_hi)
        v = _eval(self.shell, cl)
        d2 = ((pos - cl) ** 2).sum(1)
        out = (d2 > 0.0) & (v > 0.0)
        if out.any():
            v[out] = (v[out] ** (-1.0 / 3.0)
                      + np.sqrt(d2[out]) / self.scale) ** -3.0
        return v

    __call__ = rho0


def _fill_cubic(vals, centres, box, m, alpha, rho_edge_val, fold=None,
                tail_scale=None, tail_floor=None):
    """CUBIC resolvability scaffold (2026-07-13 redesign): every sub-rho_edge
    cell gets ``max(raw, taper)`` where the taper is the STEEPEST profile an
    equal-mass glass at mass ``m`` can hold, integrated from the nearest
    occupied (>= rho_edge) cell's own UNCAPPED value rho0 (M12 local start):

        |grad ln rho| = 1/(alpha*Delta(rho)),  Delta = (m/rho)**(1/3)
        =>  rho(s) = rho0 * (1 + s/(3*alpha*h0))**-3,  h0 = (m/rho0)**(1/3)
        =>  rho(s)**(-1/3) = rho0**(-1/3) + s/(3*alpha*m**(1/3))

    It replaces the exp taper (lambda in capture-cell units - N-dependent,
    the ladder's delta_bg defect) and rho_floor: reach and shape are set by
    the LOCAL output spacing only. ``max(raw, .)`` keeps real sub-contour
    structure (the disk flare) wherever it beats the taper.

    fold, tail_scale, tail_floor : the CARRY CUT (2026-07-13 20k smoke
    verdict). The marginal profile is the steepest a glass can HOLD - i.e.
    maximal parking: carried to the frame it is a fat (~s**-3, log-divergent
    per e-fold), fully backed equilibrium 10-25x above the exp+M6 far field,
    and the 20k A/B parked 22% of the ensemble beyond r=15 and HALVED the
    cusp. So the scaffold is only CARRIED (backed) for one ``fold`` drop -
    s_cut = 3*alpha*h0*(fold**(1/3)-1), ~2 output kernel supports at the
    launch spacing for fold=8 (the dc6 calibration) - and beyond it hands
    over to the steep M6 guidance tail (per-cell exponent to ``tail_floor``
    exactly at ``tail_scale``, the frame edge): effectively massless, pure
    home-pointing gradient, so kicked strays drain instead of parking.
    fold=None (or no tail params) = the pure marginal profile everywhere
    (the analytic form; used by tests and _CubicTail)."""
    vals = np.array(vals, dtype=np.float64)
    good = vals >= rho_edge_val
    bad = ~good
    if not bad.any():
        return vals
    if not good.any():
        raise ValueError('cubic scaffold: no occupied cells (rho_edge too '
                         'high for this table)')
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
    launch = vals[good][nn]
    scale = 3.0 * alpha * m ** (1.0 / 3.0)
    taper = (launch ** (-1.0 / 3.0) + dist / scale) ** -3.0
    if fold is not None and tail_scale is not None and tail_floor is not None:
        # carry cut: cubic skirt down to launch/fold, then the M6 steep
        # guidance tail (continuous at the cut; the shallower-of-two max
        # never applies here - the tail is strictly steeper by design)
        h0 = (m / launch) ** (1.0 / 3.0)
        s_cut = 3.0 * alpha * h0 * (fold ** (1.0 / 3.0) - 1.0)
        cut_val = launch / fold
        beyond = dist > s_cut
        if beyond.any():
            p = np.log(np.maximum(cut_val[beyond] / tail_floor, 1.0)) \
                / np.log(np.maximum(tail_scale / s_cut[beyond], np.exp(0.5)))
            tailv = cut_val[beyond] * (dist[beyond] / s_cut[beyond]) ** (-p)
            taper[beyond] = np.maximum(tailv, tail_floor)
    vals[bad] = np.maximum(np.clip(vals[bad], 0.0, None), taper)
    return vals


@njit(cache=True, inline='always')
def _padded_point(x, y, z, lo, box, periodic, dims, offsets, keys,
                  vals_base, vals_shell, order, interp, aoffsets, akeys,
                  avals_base, avals_shell, corners_base, corners_shell,
                  rho_edge_val, cap, tail_floor, r_scale, inv_span,
                  cubic_scale):
    """Padded-target value at ONE point on an AMR leaf pair (base + shell
    vals on the SAME cell geometry): base lookup -> occupancy cut (the
    containing cell's FLAT average >= rho_edge AND inside the domain;
    2026-07-11 - the linear readout inside a truly-empty coarse cell ramps
    toward a dense neighbour and would classify fabricated density as
    occupied gas: the galaxy z-sheet) -> occupied returns the LINEAR base
    value -> else shell lookup at the domain-clamped point + the M6
    outside-domain tail + the rho_edge cap. Semantically identical to
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
        if cubic_scale > 0.0:               # cubic scaffold: u-linear
            if sv > 0.0:                    # continuation (_CubicTail)
                sv = (sv ** (-1.0 / 3.0) + d / cubic_scale) ** -3.0
        elif inv_span > 0.0:                # M6 tail beyond the domain
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
                      cubic_scale, out):
    """One-pass padded-target readout (:func:`_padded_point` per point) with
    no numpy temporaries/subset indexing - the generic composition measured
    ~9x the raw table query on the 2026-07-10 galaxy timing probe (pP was
    94% of the relax iteration). Returns the number of misses."""
    n = pos.shape[0]
    nmiss = 0
    for p in prange(n):
        v, ok = _padded_point(pos[p, 0], pos[p, 1], pos[p, 2], lo, box,
                              periodic, dims, offsets, keys, vals_base,
                              vals_shell, order, interp, aoffsets, akeys,
                              avals_base, avals_shell, corners_base,
                              corners_shell, rho_edge_val, cap,
                              tail_floor, r_scale, inv_span, cubic_scale)
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
                        cubic_scale, out):
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
                                  inv_span, cubic_scale)
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
    the whole padded readout (occupancy cut, shell + M6 tail, cap) runs as
    ONE numba kernel (:func:`_padded_amr_query`) with zero numpy
    temporaries. ``base``/``occupancy``/``shell`` keep the same objects the
    generic composition would use (introspection, cull, tests); only
    ``rho0`` is overridden. ``tail`` disabled (d_max=0) = clamp-flat outside
    the domain, as the generic path without a _TailedShell.

    occ_edge : the occupancy threshold (always ``rho_edge``).
    cap : vacuum-side clamp. Default (the sentinel) = ``occ_edge`` (the
        2026-07-10 seam fix, right for the 'edge'-start shell); pass None
        for the LOCAL-start shell (``shell_start='local'``), whose values
        legitimately exceed ``rho_edge`` next to bright surfaces."""

    _CAP_EDGE = object()                 # sentinel: cap = occ_edge

    def __init__(self, base, occupancy, shell, occ_edge, shell_table,
                 tail_floor=0.0, r_scale=1.0, d_max=0.0, cap=_CAP_EDGE,
                 cubic_scale=0.0):
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
        # >0 = cubic-scaffold outside-domain continuation (3*alpha*m**(1/3),
        # the u-linear slope; see _CubicTail) - takes precedence over the M6
        # tail parameters in the fused readout
        self.cubic_scale = float(cubic_scale)

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
            self._inv_span, self.cubic_scale, out)
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
            self.tail_floor, self.r_scale, self._inv_span, self.cubic_scale,
            out)
        if nmiss:
            raise ValueError(f'{nmiss}/{len(pos)} stencil points not '
                             'covered by the AMR table')
        return out

    __call__ = rho0


def place_background(occupancy, shell, rho_edge_val, rho_cut, m,
                     region_lo, region_hi, jitter=0.3, seed=0, rho_min=None):
    """Jittered cubic background lattice in the SHELL (task 9, N-aware form).

    Lattice spacing = the finest shell spacing ``(m/rho_edge)**(1/3)`` over
    ``[region_lo, region_hi]`` (the gas bbox + a shell margin). A site is KEPT
    iff it is vacuum (``occupancy < 0.5``) AND above the cutoff
    (``shell(site) > rho_min``, default ``rho_floor`` - the taper only), then
    THINNED by ``shell(site)/rho_edge`` so the surviving number density tracks
    the decaying shell target (grades from ~the finest spacing at the gas edge
    out to ``Delta_bg`` where the taper meets the floor). Deep vacuum
    (``shell <= rho_min``) gets no particle; gas cells get none (occupied).
    All background particles carry mass ``m``. Returns ``(pos (n,3), mass (n,))``.

    NB the table/reconstruct path no longer uses the lattice (retired
    2026-07-16, CLEANUP_PLAN C1 - sampled backing via the mass-contract
    closure is unconditional there); this survives as the natural background
    of :func:`build_uniform_envelope`, the analytic/uniform build path.
    ``rho_min`` lowers the vacuum cutoff below ``rho_cut`` (the retired
    backed-tail experiment was its only caller; kept for generality).
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
    mass = np.full(len(pos), m)
    return pos, mass


def cull_vacuum(pos, mass, aux, occupancy, total_mass):
    """Spatial vacuum cull + mass renormalisation (task 9f/g).

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
        return AMRTable(table.lo, table.box, table.dims, table.offsets,
                        table.keys, vals, table.order, table.periodic,
                        interp=table.interp)
    if isinstance(table, TableDensity):
        return TableDensity(table.axes, vals.reshape(table.values.shape),
                            mapping=None)
    raise TypeError(f'cannot rebuild table of type {type(table).__name__}')


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
                         c=NSMOOTH_DEFAULT, k=None, decay_cells=2.0,
                         seed=0, nsmooth=NSMOOTH_DEFAULT,
                         pad_frac=0.1, shell_start='local',
                         scaffold='exp', alpha=ALPHA_DEFAULT,
                         decay_spacing=None, decay_local=False,
                         flat_floor=False, tail_reach=None,
                         n_frame=None, nl0=None):
    """Envelope dict for a TABLE target (AMRTable or 3D TableDensity) - the
    table-native analog of :func:`build_uniform_envelope`, driven by the
    glassgen table builder (:func:`glassgen.progressive.build_from_table`).

    From ``m = gas_mass/n_final``: ``rho_edge`` (cumulative faint-tail cut at
    ``c*m`` on the cell values). Two-rung ladder (2026-07-14): rho_floor is
    RETIRED as a separate rung - the vacuum decays ``rho_edge -> rho_frame =
    rho_edge/k**6``. Occupancy is analytic and sharp on the table
    (``table.rho0(pos) >= rho_edge``); the shell is the same cell set with the
    sub-``rho_edge`` cells emptied (the fill trigger is ``floor=rho_edge``) and
    refilled by :func:`_fill_empty`. With ``flat_floor=False`` (the M18 DEFAULT)
    that fill is ONE strictly-decreasing profile from the edge value down to
    ``rho_frame`` at the frame corner (in the cells inside the domain,
    analytically outside it via :class:`_TailedShell`) - no rho_floor plateau,
    so a re-anneal-kicked stray always feels a home-pointing gradient. (Legacy
    ``flat_floor=True`` keeps the old exp taper + M6 tail with the flat shelf.)
    The background is SAMPLED backing (lattice retired 2026-07-16, CLEANUP_PLAN
    C1): the mass-contract closure is the sole sampler - ``n_bg = (M_target -
    gas_mass)/m`` fillers drawn from the vacuum cells w.p. ``shell*V`` at
    particle mass m, so backing number density tracks the target right up to
    bright surfaces (the k15/ladder cusp lever, no ``delta_bg`` quantization;
    ``delta_bg`` survives as a diagnostic and the frame-sizing kernel scale).
    NB the cut is computed on the values as stored - capture the table RAW
    (genuine zeros) for an exact cut; a pre-filled (decay) table shifts
    ``rho_edge`` slightly low.

    For an OPEN build (``box is None``) the dict also carries ``frame_box``/
    ``frame_centre`` - the periodic frame sized by :func:`frame_open_box`
    (task M2: per-axis margins + the kernel-fits-the-box cap from
    ``Delta_bg``); ``nsmooth``/``pad_frac`` feed that rule. Returns the
    ``envelope=`` dict for :func:`glassgen.progressive.relax_progressive`.

    shell_start : where the vacuum taper STARTS (2026-07-13, the M11 A/B
    follow-up; 'local' is the DEFAULT since the 2026-07-13 galaxy verdict -
    crater gone, best arm on every metric).
      'edge' - start capped at the global ``rho_edge`` (the
        2026-07-10 seam fix) and clamp the padded vacuum side there too.
        Right for surfaces whose local density ~ rho_edge (the smoke ball);
        on a razor-thin disk, whose surface density varies ~5 decades along
        the occupancy contour, the cap itself creates a multi-decade step at
        the bright parts (centre: 20 -> rho_edge between adjacent cells) and
        leaves the exterior with no support scaffold near the level a real
        transit particle can realize.
      'local' (default) - start at the NEAREST OCCUPIED CELL's own value (uncapped
        fill, no padded vacuum clamp): a local edge continuation
        ``rho_local * exp(-d/lambda)`` - the glassgen_old pkdUpdateDensity
        structure (decision 11) and the explicit version of the exterior
        pressure support the pre-M11 capture smear provided by accident.
        Continuous at the surface by construction; still decays to the SAME
        N-aware ``rho_frame`` (larger start -> the profile just decays
        further out). Occupancy, the cull, and rho_edge / rho_frame are
        UNCHANGED (raw-value based) - only the relax's guidance field in the
        vacuum changes.

    scaffold : the vacuum-scaffold construction (2026-07-13 redesign).
      'exp' (default) - the M12 local-start taper + M6 power-law tail
        (legacy ``flat_floor=True``) or the M18 strictly-decreasing profile
        (default), backed by closure-sampled fillers, parametrized by
        k / shell_start / tail_reach as documented above.
      'cubic' - the RESOLVABILITY scaffold: the marginal profile
        ``rho(s) = rho_local*(1 + s/(3*alpha*h0))**-3`` (:func:`_fill_cubic`)
        from the occupied surface, CARRIED for one k**3 fold (s_cut =
        3*alpha*h0*(k-1), ~2 output kernel supports at the launch spacing -
        the dc6 calibration), then handed to the steep M6 guidance tail
        (2026-07-13 smoke verdict: the marginal profile carried to the frame
        is a fat backed equilibrium that parks strays and halves the cusp);
        shell value = max(raw, taper). The lattice is RETIRED: the whole
        background is mass-weighted sampling of the scaffold at particle
        mass m (the closure mechanism as the sole sampler), so backing
        number density tracks the target right up to bright surfaces - the
        dense-backing limit the k15/ladder arms pointed at, with no
        delta_bg quantization. decay_cells / shell_start do not apply
        (local start is inherent); rho_frame / delta_bg keep their
        exp-arm formulas as tail parameters / diagnostics.
        ``alpha`` is the ONE reach knob, shared with
        progressive.auto_coarse_count (cascade).
        VERDICT (20k A/B 2026-07-13): FALSIFIED as a target shape - the
        marginal profile is ~10x shallower than exp mid-field, and backed
        shallow field = parking lot (cusp 0.32/0.42 vs exp 0.77; 22-27%
        of the ensemble parked beyond r=15). Kept for the record; the
        redesign continued as exp + sampled backing (now the only path).

    decay_spacing : exp-scaffold lambda in OUTPUT-SPACING units (the pivot
        after the cubic verdict): ``lambda = decay_spacing*(m/rho_edge)
        **(1/3)`` - N-invariant reach, replacing ``decay_cells`` (whose
        capture-cell units made the scaffold RELATIVELY sparser as N grew:
        the resolution ladder's cusp inversion). None = the legacy
        decay_cells behaviour.
    decay_local : evaluate the eta taper length at the LAUNCH cell's own
        density instead of the global rho_edge (M16, the sink fix):
        lambda_i = decay_spacing*(m/rho_i)**(1/3) - each vacuum cell
        inherits the lambda of its nearest occupied cell. A GLOBAL lambda
        is calibrated to the faintest resolvable gas, so above a surface
        orders brighter than rho_edge the skirt barely decays: a
        near-flat shelf of demanded mass (the table_res_probe vertical
        panel) that the cusp drains into when the backing does not fill
        it. Locally, the skirt shrinks to ~eta output spacings OF THAT
        SURFACE (enough support for one kernel, no more); at rho_edge
        launches nothing changes. Requires decay_spacing.

    (bg_mode/bg_tail RETIRED 2026-07-16, CLEANUP_PLAN C1: the jittered
    lattice and the backed-tail experiment are gone - sampled backing via
    the mass-contract closure is the sole, unconditional mechanism. The
    ladder verdict: the lattice's cusp decays with N (delta_bg grows
    relatively sparser), the sampled backing's is ~N-invariant.)"""
    if shell_start not in ('edge', 'local'):
        raise ValueError(f"unknown shell_start {shell_start!r} "
                         "(use 'edge' or 'local')")
    if scaffold not in ('exp', 'cubic'):
        raise ValueError(f"unknown scaffold {scaffold!r} "
                         "(use 'exp' or 'cubic')")
    if decay_local and decay_spacing is None:
        raise ValueError('decay_local needs decay_spacing (the eta taper); '
                         'the decay_cells taper is capture-cell based')
    centres, sizes, _, vals = table_cells(table)
    vals = np.asarray(vals, dtype=np.float64)
    vols = sizes.prod(1)
    m = float(gas_mass) / int(n_final)
    # M20: k from the floor-backability count n_frame unless legacy-explicit
    k = resolve_bg_k(k, n_frame, nsmooth=nsmooth)
    re, delta_bg = edge_floor_spacing(vals, vols, m, c=c, k=k)
    # cubic scaffold is inherently local-start: never capped
    shell_cap = re if (shell_start == 'edge'
                       and scaffold == 'exp') else None
    # occupancy is DOMAIN-AWARE: outside the table's bbox the AMR/grid readout
    # clamps to the boundary cell, so a stray beyond the domain could read
    # >= rho_edge and dodge the vacuum cull (seen when the capture stopped
    # covering the whole periodic frame - task M2's capture-first order).
    # It is also FLAT (cell averages, 2026-07-11): the linear readout inside
    # a truly-empty coarse cell ramps toward a dense neighbour's average and
    # marks fabricated density as occupied gas (the galaxy z-sheet: 15% of
    # the survivors parked in cells whose true average is zero).
    dom_lo = (centres - 0.5 * sizes).min(0)
    dom_hi = (centres + 0.5 * sizes).max(0)
    occ_view = table
    if isinstance(table, AMRTable) and table.interp != 'flat':
        occ_view = AMRTable(table.lo, table.box, table.dims, table.offsets,
                            table.keys, table.vals, table.order,
                            table.periodic, interp='flat')

    def _occ(p, t=occ_view, e=re, lo=dom_lo, hi=dom_hi):
        p = np.asarray(p, dtype=np.float64)
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        return ((_eval(t, p) >= e) & inside).astype(np.float64)

    occupancy = CallableDensity(_occ)
    # background region = table domain + the shell taper reach, so gas at the
    # domain faces gets background too (the taper hits the floor at
    # 3*lambda*ln(k), lambda = decay_cells x the gas-EDGE cell size)
    if decay_spacing is not None:
        # eta mode (post-cubic pivot): lambda in OUTPUT-spacing units -
        # N-invariant scaffold reach, independent of the capture grid.
        # (Feeds fill_sizes below; it no longer sizes any region - taper_len
        # was retired 2026-07-16, see below.)
        lam_const = decay_spacing * (m / re) ** (1.0 / 3.0)
    # taper_len RETIRED (2026-07-16, CLEANUP_PLAN C1a). It was the reach of
    # the old exponential skirt and did two jobs, both dead: (a) it inflated
    # region_lo/hi, the LATTICE placement region - but the lattice is legacy
    # (sampled backing places from the shell's own mass distribution), and
    # (b) it was a floor in frame_open_box's margin - where H_edge always
    # wins at every setting in use (7.2 vs 11.2 on the 20k galaxy). Its only
    # surviving effect was to fatten the frame by 2*taper_len per axis, which
    # inflated tail_scale = 0.5*max(frame) and so STRETCHED the vacuum
    # profile's length scale rs = tail_scale/tail_reach (the exponent p is
    # independent of tail_scale). Removing it tightens the frame to what
    # kernel-safety + padding require and compresses the vacuum profile, so
    # the scaffold drains faster - the direction M18 wants.
    region_lo = dom_lo
    region_hi = dom_hi
    # frame BEFORE the fill (M6 needs its extent): open build -> M2 rule;
    # explicit periodic box -> the box IS the frame.
    frame_box_ = frame_centre = None
    if box is None:
        frame_box_, frame_centre = frame_open_box(
            region_lo, region_hi, delta_bg, nsmooth=nsmooth,
            pad_frac=pad_frac, k=k)
        frame_extent = frame_box_
    else:
        frame_extent = np.asarray(box, dtype=np.float64).reshape(3)
    # M6 whole-frame vacuum tail: past the exponential taper the shell keeps
    # decaying as a power law, reaching rho_frame (= rho_floor/k**3) only at
    # the frame edge - a kicked stray anywhere in the frame feels a
    # home-pointing gradient instead of the flat rho_floor.
    rho_frame = re / k ** 6               # two-rung ladder: rho_edge -> rho_frame
    tail_scale = 0.5 * float(np.max(frame_extent))
    # M20: the profile steepness comes from the launch-resolvability knob
    # nl0 (tail_reach solved per build, so the launch N_L is pinned at
    # every N) unless a legacy explicit tail_reach bypasses the solve;
    # either way the resonance-band guard runs on the resulting profile.
    nl0_eff = p_eff = None
    if flat_floor:
        tail_reach = 6.0 if tail_reach is None else float(tail_reach)
    else:
        spacing_edge = (m / re) ** (1.0 / 3.0)
        if tail_reach is None:
            tail_reach = solve_tail_reach(
                tail_scale, spacing_edge,
                NL0_DEFAULT if nl0 is None else float(nl0), k,
                nsmooth=nsmooth)
        elif nl0 is not None:
            warnings.warn('explicit tail_reach overrides nl0 (legacy path)')
        if scaffold == 'exp':
            nl0_eff, p_eff = check_scaffold_resolvability(
                tail_scale, spacing_edge, tail_reach, k, nsmooth=nsmooth)
    if scaffold == 'cubic':
        # cubic skirt carried for one k**3 fold, then the SAME steep M6
        # guidance tail as the exp arm (2026-07-13 20k smoke verdict: the
        # marginal profile carried to the frame is a fat backed equilibrium
        # - 22% of the ensemble parked beyond r=15 and the cusp HALVED;
        # dense backing belongs at the surface, the deep vacuum needs
        # drainage, not carrying)
        shell_vals = _fill_cubic(vals.copy(), centres, box, m, alpha, re,
                                 fold=k ** 3, tail_scale=tail_scale,
                                 tail_floor=rho_frame)
    else:
        if decay_spacing is not None:
            if decay_local:
                # M16 local-lambda skirt: _fill_empty takes lambda from the
                # NEAREST-OCCUPIED cell's sizes entry, so stamping each
                # cell with eta spacings of ITS OWN density hands every
                # vacuum cell the launch surface's local spacing (bright
                # cusp ~ (m/rho_cusp)^(1/3), faint edge = lam_const).
                # Launch density floored at rho_edge: the edge-calibrated
                # reach is the CEILING - sub-rho_edge launches (scaffold
                # cells above rho_floor) must not get a longer skirt than
                # the working global arm (first probe: uncapped grew the
                # scaffold mass +13%, the faint outer disk outweighing
                # the cusp shrink).
                fill_sizes = decay_spacing * (
                    m / np.maximum(vals, re)) ** (1.0 / 3.0)
            else:
                fill_sizes = np.full(len(vals), lam_const)
            fill_dc = 1.0
        else:
            fill_sizes = np.cbrt(vols)
            fill_dc = decay_cells
        shell_vals = _fill_empty(vals.copy(), 'decay', centres, box,
                                 sizes=fill_sizes,
                                 decay_cells=fill_dc,
                                 floor=re, cap=shell_cap,
                                 tail_scale=tail_scale,
                                 tail_floor=rho_frame,
                                 flat_floor=flat_floor,
                                 tail_reach=tail_reach)
    shell = _with_values(table, shell_vals)
    # outside the table domain the readout clamps flat -> continue the decay
    # analytically out to the frame (open builds only; a periodic box wraps)
    padded_shell = shell
    d_out, r_scale = 0.0, 1.0
    if box is None:
        dom_half = 0.5 * (dom_hi - dom_lo)
        d_out = float(np.linalg.norm(
            np.maximum(0.5 * frame_extent - dom_half, 0.0)))
        if d_out > 0.0:
            r_scale = max(float(np.linalg.norm(dom_half)), 1e-12)
            padded_shell = _TailedShell(shell, dom_lo, dom_hi, r_scale,
                                        d_out, rho_frame)
    if isinstance(table, AMRTable):
        # fused one-kernel readout (same values as the generic composition;
        # the relax evaluates this every particle every iteration). All
        # four interps run fused - 'vertex' passes the two corner tables
        # (2026-07-13 port: the 80k vertex arm spent 75% of the loop on
        # the generic path, 1651 vs 522s).
        padded = PaddedTableDensity(table, occupancy, padded_shell, re,
                                    shell, tail_floor=rho_frame,
                                    r_scale=r_scale, d_max=d_out,
                                    cap=shell_cap)
    else:
        padded = PaddedDensity(table, occupancy, padded_shell,
                               cap=shell_cap)
    # MASS-CONTRACT CLOSURE (2026-07-13, user; the SOLE background sampler
    # since the lattice retirement 2026-07-16, CLEANUP_PLAN C1): the ensemble
    # must carry the target's WHOLE mass integral - occupied base + in-domain
    # shell + the analytic outside-domain tail - as particles inside the
    # EXISTING frame (no box growth; that was bg_tail's kernel-wrap trap).
    # Exact placement does not matter (the relax redistributes); what matters
    # is that no mass has to be borrowed from the gas: n_bg = (M_target -
    # M_gas)/m fillers are sampled from the shell's own mass distribution
    # (vacuum cells w.p. shell*V, uniform inside the cell - the dense skirt
    # gets most of them, the far tail's mass representatives just sit closer
    # in). Backing number density tracks the target right up to the bright
    # surfaces a fixed lattice spacing could never represent - the
    # k15/ladder cusp lever, with no delta_bg quantization.
    occ_cells = vals >= re
    M_occ = float((vals[occ_cells] * vols[occ_cells]).sum())
    M_shell_in = float((shell_vals[~occ_cells] * vols[~occ_cells]).sum())
    M_out = 0.0
    if box is None and d_out > 0.0:
        rng = np.random.default_rng(seed + 7)
        fr_lo = frame_centre - 0.5 * frame_box_
        q = fr_lo + rng.random((200_000, 3)) * frame_box_
        outside = ~np.all((q >= dom_lo) & (q <= dom_hi), axis=1)
        v_frame = float(np.prod(frame_box_))
        v_out = v_frame - float(np.prod(dom_hi - dom_lo))
        if outside.any() and v_out > 0.0:
            M_out = float(_eval(padded_shell, q[outside]).mean()) * v_out
    M_target = M_occ + M_shell_in + M_out
    n_bg_want = max(0, int(round((M_target - float(gas_mass)) / m)))
    bg_pos, bg_mass = np.zeros((0, 3)), np.zeros(0)
    rng = np.random.default_rng(seed + 13)
    if n_bg_want > 0:
        vac = np.where(~occ_cells)[0]
        w = shell_vals[vac] * vols[vac]
        pick = rng.choice(len(vac), size=n_bg_want, p=w / w.sum())
        cells = vac[pick]
        bg_pos = (centres[cells] - 0.5 * sizes[cells]
                  + rng.random((len(cells), 3)) * sizes[cells])
        bg_mass = np.full(len(bg_pos), m)
    env = dict(density=padded, occupancy=occupancy, bg_pos=bg_pos,
               bg_mass=bg_mass, gas_mass=float(gas_mass),
               rho_edge=re, rho_frame=rho_frame,
               delta_bg=delta_bg, m=m, scaffold=scaffold, alpha=alpha,
               decay_spacing=decay_spacing,
               decay_local=bool(decay_local))
    env['M_target'] = M_target
    env['n_bg_topup'] = len(bg_pos)
    env['bg_k'] = k
    env['tail_reach'] = tail_reach
    if nl0_eff is not None:                      # M20: launch N_L + exponent
        env['nl0'] = nl0_eff
        env['profile_p'] = p_eff
    if frame_box_ is not None:                   # open build: the M2 frame
        env['frame_box'], env['frame_centre'] = frame_box_, frame_centre
    return env


def build_uniform_envelope(density, pos, gas_mass, n_final,
                           c=NSMOOTH_DEFAULT, k=2.0, ngrid=None,
                           decay_cells=2.0, jitter=0.3, seed=0,
                           nsmooth=NSMOOTH_DEFAULT, pad_frac=0.1,
                           shell_start='local', scaffold='exp',
                           alpha=ALPHA_DEFAULT,
                           flat_floor=True, tail_reach=6.0):
    """Assemble the envelope dict for ``./glassgen`` (glassgen-native, no
    sph_interp) from the analytic/table target ``density`` and the gas pre-IC.

    All fields are in OPEN/world coordinates (relax_progressive frames the padded
    periodic box itself). Steps:

    - A uniform grid over the gas bbox + a shell margin (large enough to hold the
      ``rho_edge -> rho_floor`` taper) samples ``vals = density.rho0(centres)``.
    - ``rho_edge``/``rho_floor``/``Delta_bg`` come from ``m = gas_mass/n_final``
      (:func:`edge_floor_spacing`).
    - **occupancy** is analytic and sharp: ``density.rho0(pos) >= rho_edge``
      (no grid-graining; assumes the target tapers below ``rho_edge`` outside the
      gas - true for free-surface tables, NOT for a table that clamps to a
      nonzero edge value at large radius).
    - **shell** is a 3D ``TableDensity`` of the grid decayed edge->floor
      (:func:`_fill_empty` 'decay', floored exactly at ``rho_floor``).
    - **background** is placed by :func:`place_background` over the same region.

    Returns the ``envelope=`` dict consumed by
    :func:`glassgen.progressive.relax_progressive`: keys ``density`` (a
    :class:`PaddedDensity`), ``occupancy``, ``bg_pos``, ``bg_mass``, ``gas_mass``
    (plus ``rho_edge``/``rho_floor``/``delta_bg`` for reporting). ``bg_pos`` may
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
    # provisional cell width (pre-margin) sets the shell margin: the taper hits
    # the floor at d = lam*3*ln(k), lam = decay_cells*cell_width
    cell0 = float(np.max(span)) / ngrid
    margin = decay_cells * cell0 * 3.0 * max(np.log(max(k, 1.0 + 1e-9)), 0.1) + cell0
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
    # M2 frame: the region already includes the taper margin, so only the
    # pad_frac/H_edge margins + the kernel-fits-the-box cap apply on top
    # (this path keeps its OWN region_lo/hi margin - see above; it never used
    # the retired taper_len knob, it passed 0.0)
    frame_box_, frame_centre = frame_open_box(
        region_lo, region_hi, delta_bg, nsmooth=nsmooth,
        pad_frac=pad_frac, k=k)
    # shell = grid decayed edge->floor, floored exactly at rho_floor and the
    # taper started at the local edge value ('local', default - the exterior
    # pressure-support scaffold, 2026-07-13) or capped at rho_edge ('edge',
    # the 2026-07-10 seam fix), plus the M6 whole-frame power-law tail down
    # to rho_frame at the frame edge (see build_table_envelope)
    if shell_start not in ('edge', 'local'):
        raise ValueError(f"unknown shell_start {shell_start!r} "
                         "(use 'edge' or 'local')")
    if scaffold not in ('exp', 'cubic'):
        raise ValueError(f"unknown scaffold {scaffold!r} "
                         "(use 'exp' or 'cubic')")
    shell_cap = re if (shell_start == 'edge'
                       and scaffold == 'exp') else None
    rho_frame = re / k ** 6               # two-rung ladder: rho_edge -> rho_frame
    tail_scale = 0.5 * float(np.max(frame_box_))
    dv_full = np.full(len(centres), dv)
    if scaffold == 'cubic':
        shell_vals = _fill_cubic(vals.copy(), centres, None, m, alpha, re,
                                 fold=k ** 3, tail_scale=tail_scale,
                                 tail_floor=rho_frame)
    else:
        shell_vals = _fill_empty(
            vals.copy(), 'decay', centres, box=None,
            sizes=np.full(len(centres), float(np.max(cell_w))),
            decay_cells=decay_cells, floor=re, cap=shell_cap,
            tail_scale=tail_scale, tail_floor=rho_frame,
            flat_floor=flat_floor, tail_reach=tail_reach)
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
    if scaffold == 'cubic':
        # lattice retired: mass-weighted scaffold sampling with the closure
        # count (see build_table_envelope)
        occ_cells = vals >= re
        M_occ = float((vals[occ_cells] * dv_full[occ_cells]).sum())
        M_shell_in = float(
            (shell_vals[~occ_cells] * dv_full[~occ_cells]).sum())
        M_out = 0.0
        if d_out > 0.0:
            rng = np.random.default_rng(seed + 7)
            fr_lo = frame_centre - 0.5 * frame_box_
            q = fr_lo + rng.random((200_000, 3)) * frame_box_
            outside = ~np.all((q >= region_lo) & (q <= region_hi), axis=1)
            v_out = (float(np.prod(frame_box_))
                     - float(np.prod(region_hi - region_lo)))
            if outside.any() and v_out > 0.0:
                M_out = float(_eval(padded_shell,
                                    q[outside]).mean()) * v_out
        M_target = M_occ + M_shell_in + M_out
        n_bg = max(0, int(round((M_target - float(gas_mass)) / m)))
        rng = np.random.default_rng(seed + 13)
        vac = np.where(~occ_cells)[0]
        if n_bg > 0 and len(vac):
            w = shell_vals[vac] * dv_full[vac]
            pick = rng.choice(len(vac), size=n_bg, p=w / w.sum())
            cells = vac[pick]
            bg_pos = (centres[cells] - 0.5 * cell_w
                      + rng.random((len(cells), 3)) * cell_w)
            bg_mass = np.full(len(bg_pos), m)
        else:
            bg_pos, bg_mass = np.zeros((0, 3)), np.zeros(0)
    else:
        bg_pos, bg_mass = place_background(occupancy, shell, re, rho_frame,
                                           m, region_lo, region_hi,
                                           jitter=jitter, seed=seed)
    env = dict(density=padded, occupancy=occupancy, bg_pos=bg_pos,
               bg_mass=bg_mass, gas_mass=float(gas_mass),
               rho_edge=re, rho_frame=rho_frame,
               delta_bg=delta_bg, m=m, scaffold=scaffold, alpha=alpha,
               frame_box=frame_box_, frame_centre=frame_centre)
    if scaffold == 'cubic':
        env['M_target'] = M_target
    return env


# --------------------------------------------------------------------------- #
#  Moved from reconstruct.py (re-exported there for back-compat)
# --------------------------------------------------------------------------- #

def _fill_empty(vals, empty, centres, box, sizes=None, decay_cells=2.0,
                floor_frac=1e-3, floor=None, cap=None,
                tail_scale=None, tail_floor=None,
                flat_floor=True, tail_reach=6.0):
    """Make empty-space cells relax-safe. A cell counts as EMPTY when its
    density is below the vacuum floor - so both genuine zeros AND faint
    sub-floor cells qualify. (A petkova capture never yields exact zeros: its
    vacuum cells carry tiny-but-positive densities, so a ``<=0`` test would miss
    them entirely and leave the vacuum unfilled - hence the floor test.)
      'decay' - nearest occupied value tapered toward the floor,
          rho0 = max(floor, rho_edge * exp(-d/lambda)) with lambda =
          decay_cells x the empty cell's width: a smooth edge->vacuum ramp
          (accretion-disk pkdUpdateDensity), so sparse edge particles feel a
          target near their own density instead of a constant edge skirt.
      'edge'  - constant nearest-occupied fill (no taper).
      'min'   - global minimum positive value.
      'zero'  - keep as-is (NOT relax-safe).
    floor : explicit vacuum floor (e.g. the N-aware ``rho_floor``); cells below
        it are emptied and the decay bottoms out here. When None it is
        ``floor_frac * median(positive density)`` (the legacy relative floor).
    cap : clamp on the fill's START value (the nearest-occupied density), used
        by the N-aware envelope with ``cap=rho_edge``: the plan's shell is
        ``rho_edge * exp(-d/lambda)``, but the nearest occupied CELL next to a
        dense gas surface holds the interior density (>> rho_edge), which
        would start the taper far too high and seam against the base field at
        the occupancy contour (2026-07-10 fix). None = uncapped (legacy).
    tail_scale, tail_floor : whole-frame vacuum tail (plan task M6). The
        exponential taper hits ``floor`` after only ~3*lambda*ln(k) (a few
        edge cells), leaving the deep vacuum FLAT - a re-anneal-kicked stray
        out there feels no gradient pointing home (the M4-rerun movie
        finding). With these set, beyond the taper the fill continues as a
        gentle POWER LAW ``floor * (d/d0)^-p`` (glassgen_old accretion-disk
        pkdUpdateDensity: gaussian core, then a matched power-law tail down
        to mindens), with the per-cell exponent solved so the tail reaches
        ``tail_floor`` only at distance ``tail_scale`` (the frame edge) -
        every vacuum point keeps a home-pointing gradient while the extra
        dynamic range stays bounded (floor -> tail_floor, ~k**3). None/None
        = the flat floor (legacy).
    flat_floor : retire the flat rho_floor shelf (plan 2026-07-14). The 'decay'
        exp+tail construction above bottoms out on a WIDE plateau at ``floor``
        (rho_floor): the max(., floor) clamp and the tail's max(d/d0,1) clamp
        pin the whole mid-field flat, and a flat rho0 region is a parking lot -
        an anneal-kicked disk particle that lands there feels zero gradient,
        glasses out at rho_floor and is culled (lost mass + renorm inflation).
        With ``flat_floor=False`` the 'decay' fill is instead ONE strictly-
        decreasing profile from the (capped) nearest-occupied value down to
        ``tail_floor`` (rho_frame) at the frame corner ``d = tail_scale``,
        mirroring :meth:`_TailedShell.rho0` so the in-domain fill and the
        outside-domain tail share a form and join seamlessly at the domain
        face:  ``rho(d) = edge * (1 + d/r_scale)**-p`` with the per-cell
        exponent ``p`` solved so ``rho(tail_scale) = tail_floor`` exactly. No
        max(., rho_floor) clamp anywhere => no plateau; the mid-field always
        drains. Requires ``tail_scale``/``tail_floor`` (the frame-corner
        distance and rho_frame). The residual flat is only at ``tail_floor``
        beyond the frame corner (benign - culled).
    tail_reach : the ONE steepness knob for ``flat_floor=False``. Sets
        ``r_scale = tail_scale/tail_reach``: larger tail_reach => smaller
        r_scale => steeper near the gas, shallower far out (the two-slope
        shape carried by a single monotone function). Ignored when
        ``flat_floor=True``."""
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
    if not flat_floor:
        # De-flattened 'decay' (retire the rho_floor shelf, plan 2026-07-14):
        # ONE strictly-decreasing profile edge*(1+d/r_scale)^-p from the capped
        # nearest-occupied value down to tail_floor at the frame corner
        # d=tail_scale (mirrors _TailedShell.rho0 - same form as the outside-
        # domain tail, seamless join). r_scale=tail_scale/tail_reach is the
        # steepness knob; no max(., rho_floor) clamp => no mid-field plateau.
        if tail_scale is None or tail_floor is None:
            raise ValueError('flat_floor=False needs tail_scale and '
                             'tail_floor (the frame-corner distance and '
                             'rho_frame)')
        rs = float(tail_scale) / max(float(tail_reach), 1e-12)
        span = max(np.log1p(float(tail_scale) / rs), 1e-12)
        p = np.log(np.maximum(edge / tail_floor, 1.0)) / span
        fac = (1.0 + dist / rs) ** (-p)
        vals[bad] = np.maximum(edge * fac, tail_floor)
        return vals
    # 'decay': exponential taper edge->floor over lambda = decay_cells cells.
    # lambda is set by the NEAREST-OCCUPIED (gas-edge) cell size, NOT the empty
    # cell's own width: on an AMR leaf set the deep-vacuum cells are huge coarse
    # cells, and tying lambda to their own size would give them a huge decay
    # length -> they never reach the floor -> a spurious second, higher "floor"
    # in the far corners. The physical decay scale is the gas surface's own
    # resolution, so far vacuum (any cell size) taper to the SAME floor.
    if sizes is not None:
        sz = np.asarray(sizes, dtype=np.float64).reshape(-1)
        lam = decay_cells * sz[good][nn]
    else:                                # no cell size: fall back to spacing
        lam = np.full(bad.sum(), decay_cells * np.median(dist))
    lam = np.maximum(lam, np.finfo(float).tiny)
    expv = edge * np.exp(-dist / lam)
    if tail_scale is not None and tail_floor is not None \
            and tail_floor < floor:
        # M6 whole-frame tail: past d0 (where the exponential crosses the
        # floor) continue as floor*(d/d0)^-p, p solved per cell so the tail
        # bottoms out at tail_floor exactly at d = tail_scale. Continuous at
        # d0 (both laws = floor) and always the shallower of the two beyond
        # it, so max(exp, tail) selects exp near the gas, tail in the deep
        # vacuum. Degenerate guards: edge==floor -> d0=lam (tail starts at
        # once); d0 within e^0.5 of tail_scale -> exponent clamped (a frame
        # barely larger than the taper just decays faster).
        d0 = np.maximum(lam * np.log(np.maximum(edge / floor, 1.0)), lam)
        p = np.log(floor / tail_floor) \
            / np.log(np.maximum(tail_scale / d0, np.exp(0.5)))
        tailv = floor * np.maximum(dist / d0, 1.0) ** (-p)
        vals[bad] = np.maximum(expv, np.maximum(tailv, tail_floor))
    else:
        vals[bad] = np.maximum(expv, floor)
    return vals


def _pad_periodic_box(pos, pad_frac=0.1):
    """Synthesise a periodic box (glassgen convention: domain CENTRED at the
    origin, [-L/2, L/2]) enclosing `pos` with a `pad_frac` vacuum margin per
    side, for relaxing an OPEN-domain IC. The glassgen relax MUST be periodic:
    an open boundary cannot contain the pseudo-pressure forces, so near-
    isolated particles (huge h) run away to infinity (verified on the isolated
    galaxy - 1% escaped past R=60). A padded periodic box contains them (an
    escapee wraps into vacuum and is culled), while the margin - larger than
    the edge kernel reach - keeps the gas from feeling its periodic images.
    Returns (box (3,), centre (3,)): shift pos -= centre to enter the frame,
    += centre to leave it."""
    lo = pos.min(0)
    hi = pos.max(0)
    centre = 0.5 * (lo + hi)
    box = (hi - lo) * (1.0 + 2.0 * pad_frac)
    return np.ascontiguousarray(box), np.ascontiguousarray(centre)
