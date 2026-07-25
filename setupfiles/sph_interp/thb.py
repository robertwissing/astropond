"""Truncated hierarchical B-splines (THB) - 1D core.

The bedrock of the hierarchical spline De Rham complex readout on the 2:1
octree (FIELD_RECONSTRUCTION_PLAN.md, "THB-spline De Rham complex readout",
rung 3).  This module builds the 1D truncated hierarchical basis; the
tensor-product spaces, the hierarchical exterior derivative, and the global
L2 projector are layered on top.

Construction (Kraft selection + Giannelli-Juttler-Speleers truncation) on a
periodic dyadic hierarchy of uniform degree-``p`` B-spline spaces
``V^0 subset V^1 subset ...`` (spacing halves each level):

- a level-l basis function is ACTIVE if its support lies in the level-l
  refined region Omega^l but NOT entirely in the finer Omega^{l+1} (the
  finest level keeps every function whose support is in Omega^L);
- each active coarse function is TRUNCATED: expressed in the finest basis via
  the dyadic subdivision (two-scale) relation, with the coefficients of all
  finer ACTIVE functions zeroed.

Truncation restores partition of unity (``sum of active THB == 1``), which
plain hierarchical B-splines lack - that identity is the module's correctness
gate.  Reference: Giannelli, Juttler & Speleers (2012), "THB-splines".
"""

import numpy as np
from scipy.special import comb

from .spline_derham import _stencil_1d, _deboor_weights, _deboor_deriv_weights

try:                                                   # numba: the leaf-transfer
    from numba import njit, prange                     # descent hot path (below)
    from numba.core import types
    from numba.typed import Dict
    _HAVE_NUMBA = True
except Exception:                                      # pragma: no cover
    _HAVE_NUMBA = False


def subdivision_mask(p):
    """Dyadic two-scale mask ``a_k = 2^{-p} C(p+1, k)``, k=0..p+1, so that
    ``B^l_i = sum_k a_k B^{l+1}_{2i+k}`` (uniform-knot B-spline refinement)."""
    k = np.arange(p + 2)
    return (2.0 ** (-p)) * comb(p + 1, k)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _leaf_transfer_numba(lleaf, cell, deg, dims, masks, lev_offset,
                             code2coeff, npatch, count_only=False):
        """numba core of :meth:`THB3D._leaf_transfer`: T_leaf as
        ``(coeff_ids (nc,), Tmat (nc, npatch))`` - Tmat row a = the weight
        vector, over the ``(p+1)^3`` leaf-covering keys, of active coefficient
        ``coeff_ids[a]``.  ONE reverse truncated-subdivision descent leaf->0,
        independent of a point's sub-position; shared by every point in the
        leaf.

        ``count_only=True`` runs the SAME traversal but skips every weight
        update (which key is active and which restricts does not depend on the
        weights), returning ``nc`` with an empty Tmat at ~1/npatch of the cost.
        :func:`_eval_operators_numba` uses it to lay out its output exactly
        before the parallel pass - deliberately the same code path rather than
        a second copy of the descent, so the two can never disagree.

        Dict-free: the patch at each level is exactly the ``(p+1)^3`` covering
        box of the ancestor cell, so it lives in a dense array indexed by the
        per-axis REDUCED offset ``r_a = off_a % s_a``, ``s_a = min(p_a+1,
        dims[l][a])`` (the reduction folds periodic self-aliasing when an axis
        has fewer cells than the spline support - e.g. a base depth of 2).
        ``code2coeff`` maps the global cell code (``lev_offset[l] + flat(i,j,k)``,
        unique per level) to the global THB coefficient index; absent = not
        active.  Each active function lives at one level and each reduced key is
        visited once, so contributions append with no dedup."""
        p0 = deg[0]; p1 = deg[1]; p2 = deg[2]
        # --- INIT: unit vectors on the leaf-level covering box (reduced) ---
        dL0 = dims[lleaf, 0]; dL1 = dims[lleaf, 1]; dL2 = dims[lleaf, 2]
        s0 = p0 + 1 if p0 + 1 < dL0 else dL0
        s1 = p1 + 1 if p1 + 1 < dL1 else dL1
        s2 = p2 + 1 if p2 + 1 < dL2 else dL2
        w = 1 if count_only else npatch          # weight-vector width (0 = off)
        pv = np.zeros((npatch, w))
        if not count_only:
            jj = 0
            for a in range(p0 + 1):
                r0 = a % s0
                for b in range(p1 + 1):
                    r1 = b % s1
                    for c in range(p2 + 1):
                        r2 = c % s2
                        pv[(r0 * s1 + r1) * s2 + r2, jj] += 1.0
                        jj += 1
        # contributions (append-only: one hit per active coeff)
        ccap = (lleaf + 1) * npatch
        cv = np.zeros((ccap, w))
        cids = np.empty(ccap, dtype=np.int64)
        nc = 0
        cur_s0 = s0; cur_s1 = s1; cur_s2 = s2
        # --- descend: current patch at level l+1 -> active check + restrict ---
        for l in range(lleaf - 1, -1, -1):
            lv1 = l + 1
            d10 = dims[lv1, 0]; d11 = dims[lv1, 1]; d12 = dims[lv1, 2]
            dl0 = dims[l, 0]; dl1 = dims[l, 1]; dl2 = dims[l, 2]
            sh = lleaf - lv1
            a1_0 = cell[0] >> sh; a1_1 = cell[1] >> sh; a1_2 = cell[2] >> sh
            b0 = (a1_0 >> 1) - p0; b1 = (a1_1 >> 1) - p1; b2 = (a1_2 >> 1) - p2
            ns0 = p0 + 1 if p0 + 1 < dl0 else dl0
            ns1 = p1 + 1 if p1 + 1 < dl1 else dl1
            ns2 = p2 + 1 if p2 + 1 < dl2 else dl2
            nv = np.zeros((npatch, w))
            for r0 in range(cur_s0):
                ci = (a1_0 - p0 + r0) % d10
                for r1 in range(cur_s1):
                    cj = (a1_1 - p1 + r1) % d11
                    for r2 in range(cur_s2):
                        ck = (a1_2 - p2 + r2) % d12
                        rj = (r0 * cur_s1 + r1) * cur_s2 + r2
                        code = lev_offset[lv1] + (ci * d11 + cj) * d12 + ck
                        coeff = code2coeff[code] if code in code2coeff else -1
                        if coeff >= 0:                      # active -> a coeff
                            cids[nc] = coeff
                            if not count_only:
                                for t in range(npatch):
                                    cv[nc, t] = pv[rj, t]
                            nc += 1
                        elif not count_only:                # restrict to level l
                            for aa in range(ci % 2, p0 + 2, 2):
                                pi_ = ((ci - aa) // 2) % dl0
                                r0p = ((pi_ - b0) % dl0) % ns0
                                wa = masks[0, aa]
                                for bb in range(cj % 2, p1 + 2, 2):
                                    pj_ = ((cj - bb) // 2) % dl1
                                    r1p = ((pj_ - b1) % dl1) % ns1
                                    wab = wa * masks[1, bb]
                                    for cc in range(ck % 2, p2 + 2, 2):
                                        pk_ = ((ck - cc) // 2) % dl2
                                        r2p = ((pk_ - b2) % dl2) % ns2
                                        wt = wab * masks[2, cc]
                                        rjp = (r0p * ns1 + r1p) * ns2 + r2p
                                        for t in range(npatch):
                                            nv[rjp, t] += wt * pv[rj, t]
            pv = nv
            cur_s0 = ns0; cur_s1 = ns1; cur_s2 = ns2
        # --- level 0 active check ---
        d00 = dims[0, 0]; d01 = dims[0, 1]; d02 = dims[0, 2]
        a0_0 = cell[0] >> lleaf; a0_1 = cell[1] >> lleaf; a0_2 = cell[2] >> lleaf
        for r0 in range(cur_s0):
            ci = (a0_0 - p0 + r0) % d00
            for r1 in range(cur_s1):
                cj = (a0_1 - p1 + r1) % d01
                for r2 in range(cur_s2):
                    ck = (a0_2 - p2 + r2) % d02
                    rj = (r0 * cur_s1 + r1) * cur_s2 + r2
                    code = lev_offset[0] + (ci * d01 + cj) * d02 + ck
                    coeff = code2coeff[code] if code in code2coeff else -1
                    if coeff >= 0:
                        cids[nc] = coeff
                        if not count_only:
                            for t in range(npatch):
                                cv[nc, t] = pv[rj, t]
                        nc += 1
        return cids[:nc].copy(), cv[:nc].copy()

    @njit(cache=True)
    def _bspline_w_into(frac, deg, out):
        """Uniform-knot degree-``deg`` B-spline values at local ``frac`` into
        ``out[0:deg+1]`` (Piegl-Tiller A2.2; == :func:`_deboor_weights`)."""
        out[0] = 1.0
        for j in range(1, deg + 1):
            saved = 0.0
            for r in range(j):
                right = (r + 1.0) - frac
                left = frac + (j - r - 1.0)
                temp = out[r] / (right + left)
                out[r] = saved + right * temp
                saved = left * temp
            out[j] = saved

    @njit(cache=True)
    def _bspline_dw_into(frac, deg, out, tmp):
        """d/du of the degree-``deg`` covering B-splines into ``out[0:deg+1]``
        (``N'_{i,p}=N_{i,p-1}-N_{i+1,p-1}``; == :func:`_deboor_deriv_weights`).
        ``tmp`` holds the degree-(deg-1) values (length >= deg)."""
        if deg == 0:
            out[0] = 0.0
            return
        _bspline_w_into(frac, deg - 1, tmp)
        for r in range(deg + 1):
            t1 = tmp[r - 1] if r >= 1 else 0.0
            t2 = tmp[r] if r <= deg - 1 else 0.0
            out[r] = t1 - t2

    @njit(cache=True)
    def _leaf_level_cell(x, dims, lev_offset, refined, L):
        """Deepest present level at ``x`` (level-0 cell units) and its cell:
        descend while the cell is refined (numba port of THB3D._leaf_level).
        ``refined`` = typed set of refined-cell codes (levels 0..L-1)."""
        l = 0
        while l < L:
            fac = 2.0 ** l
            m0 = np.int64(np.floor(x[0] * fac)) % dims[l, 0]
            m1 = np.int64(np.floor(x[1] * fac)) % dims[l, 1]
            m2 = np.int64(np.floor(x[2] * fac)) % dims[l, 2]
            code = lev_offset[l] + (m0 * dims[l, 1] + m1) * dims[l, 2] + m2
            if code in refined:
                l += 1
            else:
                break
        fac = 2.0 ** l
        c0 = np.int64(np.floor(x[0] * fac)) % dims[l, 0]
        c1 = np.int64(np.floor(x[1] * fac)) % dims[l, 1]
        c2 = np.int64(np.floor(x[2] * fac)) % dims[l, 2]
        return l, c0, c1, c2

    @njit(parallel=True, cache=True)
    def _eval_operators_numba(pts, deriv_axes, deg, dims, masks, lev_offset,
                              code2coeff, refined, npatch, L, nchunk,
                              vtemplate):
        """The WHOLE per-leaf fast operator in ONE njit call: assign each point
        to its leaf, group (sort by leaf code), run the reduced-offset descent
        ONCE per leaf, apply each point's (p+1)^3 B-spline (or d/dx_axis) weights
        to T_leaf, and emit COO ``(rows, cols, vals)`` of ``E[q,i]=beta_i(x_q)``
        - no per-leaf Python dispatch, no Python by_leaf/BW glue.  Reproduces
        :meth:`THB3D.eval_operator_fast` (the per-leaf Python build).

        THREADED over leaves (the descent was 78% of the build and the build is
        now the dominant cost, the solves having been threaded).  Leaves are
        independent, so the only obstacle to ``prange`` was that a shared
        append cursor serialises them.  Removed by laying the output out
        EXACTLY before emitting: a ``count_only`` descent (same code path, no
        weight updates, ~1/npatch of the cost) gives each leaf's ``nc``, whose
        prefix sum over points is each point's slot.  Every thread then writes
        only inside its own points' slots - no cursor, no atomics, no
        per-thread buffers to merge, and NO possible overflow (the bound is
        exact, not an estimate).  As a bonus the buffer is now the tight bound
        rather than the old ``npts*(npatch+8)`` allowance.

        BYTE-IDENTICAL to the serial build: a point's row is computed from its
        own leaf's T_leaf alone, in the same order, and the final CSR is
        assembled by :func:`_gather_csr` from ``src``/``cnt`` per POINT - so
        neither the thread count nor the chunking can be observed in the
        output.  (``nchunk`` only sets the work split.)

        ``deriv_axes`` is a (K,) int array (-1 = value, 0/1/2 = d/dx_axis): ALL K
        operators come out of the SAME descent and the SAME structural pattern,
        as ``vals (K, nnz)`` sharing one ``(rows, cols)``.  That is exact - the
        descent depends only on the leaf, and the sparsity pattern only on which
        basis functions cover the leaf - and it is the memory-relevant part: the
        curl needs 2 derivative operators per S1 component, so they cost one
        index array between them instead of one each (and one descent, not two).

        MEMORY: indices are emitted as int32 (a Q x n_basis operator never
        exceeds 2^31 on either axis here) and the buffer is the exact
        per-leaf-nc bound (see above).  The buffers are returned WHOLE with the
        used count ``out``; the caller slices views and lets :func:`_gather_csr`
        write the final arrays (an internal ``[:out].copy()`` would double the
        peak at exactly the wrong moment).  Values are bit-for-bit what the
        single-operator build gave.  The only difference is the PATTERN: an
        entry is kept when it is nonzero for ANY of the K (the shared pattern is
        their union), so an operator can carry an explicit stored zero where
        another is nonzero.  That is numerically inert - the ACTION is the same.

        Emission is per point CONTIGUOUS (a point's whole row lands in one
        block) and the leaf's coefficient ids are sorted first, so the buffers
        are already CSR rows in canonical column order - just in leaf order
        rather than point order.  ``cnt``/``src`` (per point) let the caller
        gather them into CSR with no sort and no big permutation array."""
        Q = pts.shape[0]
        K = deriv_axes.shape[0]
        p0 = deg[0]; p1 = deg[1]; p2 = deg[2]
        # --- leaf assignment + group key (per point independent) ---
        lleaf = np.empty(Q, dtype=np.int64)
        cell = np.empty((Q, 3), dtype=np.int64)
        key = np.empty(Q, dtype=np.int64)
        for q in prange(Q):
            ll, c0, c1, c2 = _leaf_level_cell(pts[q], dims, lev_offset,
                                              refined, L)
            lleaf[q] = ll; cell[q, 0] = c0; cell[q, 1] = c1; cell[q, 2] = c2
            key[q] = lev_offset[ll] + (c0 * dims[ll, 1] + c1) * dims[ll, 2] + c2
        order = np.argsort(key, kind="mergesort")
        # --- leaf runs in the sorted order ---
        rstart = np.empty(Q + 1, dtype=np.int64)
        nruns = 0
        g = 0
        while g < Q:
            rstart[nruns] = g
            k = key[order[g]]
            h = g + 1
            while h < Q and key[order[h]] == k:
                h += 1
            nruns += 1
            g = h
        rstart[nruns] = Q
        # --- EXACT layout: count-only descent per leaf, then prefix sum ---
        rnc = np.empty(nruns, dtype=np.int64)
        for r in prange(nruns):
            cid0, _t0 = _leaf_transfer_numba(
                lleaf[order[rstart[r]]], cell[order[rstart[r]]], deg, dims,
                masks, lev_offset, code2coeff, npatch, True)
            rnc[r] = cid0.shape[0]
        src = np.zeros(Q, dtype=np.int64)      # per-point slot start
        cnt = np.zeros(Q, dtype=np.int32)      # per-point row length (<= nc)
        pos = 0
        for r in range(nruns):
            n = rnc[r]
            for gg in range(rstart[r], rstart[r + 1]):
                src[order[gg]] = pos
                pos += n
        cap = pos
        # `vals` is allocated here in the caller's output dtype (carried in by
        # the empty `vtemplate`): accumulation stays float64 either way, and
        # storing float32 is exactly the cast the caller used to do at the end -
        # so it halves the biggest transient buffer instead of adding a second
        # copy of it.
        cols = np.empty(cap, dtype=np.int32)
        vals = np.empty((K, cap), vtemplate.dtype)
        # --- chunks of leaf runs, balanced by POINT count (the per-point apply
        # is the other half of the work, and readout points cluster unevenly) --
        cstart = np.empty(nchunk + 1, dtype=np.int64)
        cstart[0] = 0
        ci = 1
        acc_pts = 0
        for r in range(nruns):
            acc_pts += rstart[r + 1] - rstart[r]
            while ci < nchunk and acc_pts * nchunk >= Q * ci:
                cstart[ci] = r + 1
                ci += 1
        while ci <= nchunk:
            cstart[ci] = nruns
            ci += 1
        for c in prange(nchunk):
            w0 = np.empty((K, p0 + 1)); w1 = np.empty((K, p1 + 1))
            w2 = np.empty((K, p2 + 1))
            tmp = np.empty(max(p0, p1, p2) + 1)
            bw = np.empty((K, npatch))
            accs = np.empty(K)
            for r in range(cstart[c], cstart[c + 1]):
                g0 = order[rstart[r]]
                ll = lleaf[g0]
                cids, Tmat = _leaf_transfer_numba(
                    ll, cell[g0], deg, dims, masks, lev_offset, code2coeff,
                    npatch, False)
                nc = cids.shape[0]
                if nc == 0:
                    continue
                # ascending order of this leaf's coefficient ids: emitting
                # through it makes every row come out in canonical CSR column
                # order, so no post-hoc sort is ever needed.  Indexed through,
                # NOT materialised - permuting Tmat itself would copy
                # (nc x npatch) doubles per leaf.
                perm = np.argsort(cids)
                scale = 2.0 ** ll
                for gg in range(rstart[r], rstart[r + 1]):
                    q = order[gg]
                    out = src[q]
                    f0 = pts[q, 0] * scale; f0 -= np.floor(f0)
                    f1 = pts[q, 1] * scale; f1 -= np.floor(f1)
                    f2 = pts[q, 2] * scale; f2 -= np.floor(f2)
                    for kk in range(K):
                        da = deriv_axes[kk]
                        if da == 0:
                            _bspline_dw_into(f0, p0, w0[kk], tmp)
                            for rr in range(p0 + 1):
                                w0[kk, rr] *= scale
                        else:
                            _bspline_w_into(f0, p0, w0[kk])
                        if da == 1:
                            _bspline_dw_into(f1, p1, w1[kk], tmp)
                            for rr in range(p1 + 1):
                                w1[kk, rr] *= scale
                        else:
                            _bspline_w_into(f1, p1, w1[kk])
                        if da == 2:
                            _bspline_dw_into(f2, p2, w2[kk], tmp)
                            for rr in range(p2 + 1):
                                w2[kk, rr] *= scale
                        else:
                            _bspline_w_into(f2, p2, w2[kk])
                        t = 0
                        for i0 in range(p0 + 1):
                            for i1 in range(p1 + 1):
                                wij = w0[kk, i0] * w1[kk, i1]
                                for i2 in range(p2 + 1):
                                    bw[kk, t] = wij * w2[kk, i2]
                                    t += 1
                    for a in range(nc):
                        ai = perm[a]             # ascending coefficient id
                        keep = False
                        for kk in range(K):
                            acci = 0.0
                            for tt in range(npatch):
                                acci += bw[kk, tt] * Tmat[ai, tt]
                            accs[kk] = acci
                            if acci != 0.0:
                                keep = True
                        if not keep:
                            continue
                        cols[out] = cids[ai]
                        for kk in range(K):
                            vals[kk, out] = accs[kk]
                        out += 1
                    cnt[q] = np.int32(out - src[q])
        out = 0
        for q in range(Q):
            out += cnt[q]
        return cols, vals, cnt, src, out

    @njit(cache=True)
    def _gather_csr(cols, vals, cnt, src, indptr, out_indices, out_data):
        """Reorder the leaf-grouped blocks into CSR (point) order by straight
        block copies - the emission is already per-point contiguous and column-
        sorted, so this is a memcpy shuffle: no sort, no permutation array, and
        it writes STRAIGHT into the final (already correctly typed) arrays, so
        nothing of size nnz is ever materialised twice."""
        K = vals.shape[0]
        for q in range(cnt.shape[0]):
            n = cnt[q]
            if n == 0:
                continue
            s = src[q]
            d = indptr[q]
            for i in range(n):
                out_indices[d + i] = cols[s + i]
            for kk in range(K):
                for i in range(n):
                    out_data[kk, d + i] = vals[kk, s + i]


def _subdivide(coeff, p, n_fine):
    """One coarse->fine subdivision step (periodic): length-n coarse coeffs ->
    length-2n fine coeffs via the two-scale mask."""
    a = subdivision_mask(p)
    n = coeff.shape[0]
    fine = np.zeros(n_fine)
    idx0 = 2 * np.arange(n)
    for k in range(p + 2):
        np.add.at(fine, (idx0 + k) % n_fine, a[k] * coeff)
    return fine


class THB1D:
    """1D periodic truncated hierarchical B-spline space of degree ``p``.

    ``refined`` is a list of length ``L`` (number of refinement steps);
    ``refined[l]`` is a boolean array over the ``n0 * 2**l`` level-``l`` cells
    marking which level-``l`` cells are subdivided to level ``l+1``.  Nesting
    (a cell can refine only inside the coarser refined region) is enforced.
    """

    def __init__(self, n0, degree, refined):
        self.n0 = int(n0)
        self.p = int(degree)
        self.L = len(refined)
        self.nl = [self.n0 * 2 ** l for l in range(self.L + 1)]
        # domain hierarchy Omega^l (boolean over level-l cells), nested
        self.omega = [np.ones(self.n0, dtype=bool)]
        self.refined = []
        for l in range(self.L):
            ref = np.asarray(refined[l], dtype=bool).copy()
            if ref.shape[0] != self.nl[l]:
                raise ValueError(f'refined[{l}] must have {self.nl[l]} cells')
            ref &= self.omega[l]                       # enforce nesting
            self.refined.append(ref)
            child = np.repeat(ref, 2)                  # child cells exist iff parent refined
            self.omega.append(child)
        self._select_active()
        self._build_truncated()

    # -- Kraft active-function selection -----------------------------------
    def _support_cells(self, i, l):
        return (i + np.arange(self.p + 1)) % self.nl[l]

    def _select_active(self):
        self.active = []
        for l in range(self.L + 1):
            act = []
            nl = self.nl[l]
            for i in range(nl):
                cells = self._support_cells(i, l)
                if not self.omega[l][cells].all():     # support in Omega^l?
                    continue
                if l < self.L and self.refined[l][cells].all():
                    continue                           # support fully refined -> deactivate
                act.append(i)
            self.active.append(np.array(act, dtype=np.int64))
        # flat list of (level, index) for the global basis ordering
        self.basis = [(l, int(i)) for l in range(self.L + 1)
                      for i in self.active[l]]
        # boolean active-mask per level for the truncation zeroing
        self._active_mask = []
        for l in range(self.L + 1):
            m = np.zeros(self.nl[l], dtype=bool)
            m[self.active[l]] = True
            self._active_mask.append(m)

    # -- truncated finest-level coefficient rows ---------------------------
    def _build_truncated(self):
        """Each active THB function as a coefficient vector on the FINEST
        level (length n_L): subdivide from its level to L, zeroing finer
        active coefficients at every step (the truncation)."""
        rows = []
        for (l0, i) in self.basis:
            c = np.zeros(self.nl[l0])
            c[i] = 1.0
            for l in range(l0, self.L):
                c = _subdivide(c, self.p, self.nl[l + 1])
                # truncate: drop contributions to finer ACTIVE functions
                c[self._active_mask[l + 1]] = 0.0
            rows.append(c)
        self.coeffs_L = (np.array(rows) if rows
                         else np.zeros((0, self.nl[self.L])))

    # -- evaluation --------------------------------------------------------
    def eval_matrix(self, x_cell0):
        """Values of every active THB function at domain coordinates
        ``x_cell0`` (in level-0 cell units).  Returns ``(n_basis, M)``."""
        x = np.ascontiguousarray(x_cell0, dtype=np.float64).reshape(-1)
        uL = x * (2 ** self.L)                          # level-L cell units
        w, idx = _stencil_1d(uL, self.p, self.nl[self.L], periodic=True)
        m = x.shape[0]
        out = np.zeros((len(self.basis), m))
        for s in range(self.p + 1):
            out += self.coeffs_L[:, idx[s]] * w[s][None, :]
        return out

    @property
    def n_basis(self):
        return len(self.basis)


# ==========================================================================
# 3D tensor-product THB space (prototype: dense finest-level expansion)
# ==========================================================================

def _subdivide_axis(c, p, axis, n_fine):
    """Subdivide a coefficient array along one axis (periodic): that axis'
    length n -> n_fine=2n via the degree-``p`` two-scale mask."""
    a = subdivision_mask(p)
    c = np.moveaxis(c, axis, 0)
    n = c.shape[0]
    fine = np.zeros((n_fine,) + c.shape[1:])
    idx0 = 2 * np.arange(n)
    for k in range(p + 2):
        np.add.at(fine, (idx0 + k) % n_fine, a[k] * c)
    return np.moveaxis(fine, 0, axis)


def _window_all(region, w, axis):
    """Periodic rolling-AND: out[i] = all(region[i : i+w]) along ``axis``."""
    out = region.copy()
    for s in range(1, w):
        out &= np.roll(region, -s, axis=axis)
    return out


def _window_all_3d(region, widths):
    out = region
    for axis in range(3):
        out = _window_all(out, widths[axis], axis)
    return out


class THB3D:
    """3D periodic truncated hierarchical B-spline space, per-axis degree
    ``degrees=(px,py,pz)`` (so it also serves the degree-dropped De Rham
    components).  ``refined[l]`` is a boolean 3D array over the level-``l``
    cells marking those subdivided to level ``l+1`` (nesting enforced).

    PROTOTYPE scope: each active THB function is materialised as dense
    coefficients on the FINEST uniform level (exact, but memory ~ prod(dims_L))
    - fine for shallow graded test meshes, NOT the deep galaxy octree (that
    needs the sparse hierarchical operators, a later rung-3 increment)."""

    def __init__(self, base_dims, degrees, refined):
        self.base = tuple(int(b) for b in base_dims)
        self.deg = tuple(int(d) for d in degrees)
        self.L = len(refined)
        self.dims = [tuple(b * 2 ** l for b in self.base)
                     for l in range(self.L + 1)]
        self.omega = [np.ones(self.dims[0], dtype=bool)]
        self.refined = []
        for l in range(self.L):
            ref = np.asarray(refined[l], dtype=bool).copy()
            if ref.shape != self.dims[l]:
                raise ValueError(f'refined[{l}] must have shape {self.dims[l]}')
            ref &= self.omega[l]
            self.refined.append(ref)
            child = np.repeat(np.repeat(np.repeat(ref, 2, 0), 2, 1), 2, 2)
            self.omega.append(child)
        self._select_active()
        self._build_expansion()

    def _widths(self):
        return tuple(d + 1 for d in self.deg)

    def _select_active(self):
        self.active_idx = []          # list of (K,3) int arrays per level
        self._active_mask = []
        w = self._widths()
        for l in range(self.L + 1):
            in_omega = _window_all_3d(self.omega[l], w)
            if l < self.L:
                in_ref = _window_all_3d(self.refined[l], w)
                act = in_omega & ~in_ref
            else:
                act = in_omega
            self._active_mask.append(act)
            self.active_idx.append(np.argwhere(act))
        self.basis = [(l, tuple(int(v) for v in ijk))
                      for l in range(self.L + 1) for ijk in self.active_idx[l]]

    def _build_expansion(self):
        """Sparse matrix C (n_basis x prod(dims_L)): row a = the truncated
        finest-level coefficients of active THB function a."""
        from scipy import sparse
        nL = int(np.prod(self.dims[self.L]))
        rows = []
        for (l0, ijk) in self.basis:
            c = np.zeros(self.dims[l0])
            c[ijk] = 1.0
            for l in range(l0, self.L):
                for axis in range(3):
                    c = _subdivide_axis(c, self.deg[axis], axis,
                                        self.dims[l + 1][axis])
                c[self._active_mask[l + 1]] = 0.0        # truncation
            rows.append(sparse.csr_matrix(c.reshape(1, nL)))
        self.C = (sparse.vstack(rows).tocsr() if rows
                  else sparse.csr_matrix((0, nL)))

    def expand_to_finest(self, coeff):
        """THB coefficients (n_basis,) -> finest-level dense coeff array."""
        flat = np.asarray(coeff, dtype=np.float64) @ self.C
        return np.asarray(flat).reshape(self.dims[self.L])

    def eval_matrix(self, positions_cell0):
        """Every active THB function at 3D coordinates ``positions_cell0``
        (in level-0 cell units, (M,3)).  Returns ``(n_basis, M)``."""
        pos = np.ascontiguousarray(positions_cell0, dtype=np.float64
                                   ).reshape(-1, 3)
        fac = 2 ** self.L
        stncl = [_stencil_1d(pos[:, a] * fac, self.deg[a],
                             self.dims[self.L][a], periodic=True)
                 for a in range(3)]
        m = pos.shape[0]
        # finest-level tensor evaluation weight per basis, via C.  Densify the
        # (prototype-scale) expansion once - repeated sparse column slicing in
        # this 64-term loop is the hotspot (the deep-octree path won't use this
        # dense finest expansion).
        if not hasattr(self, "_Cdense"):
            self._Cdense = np.asarray(self.C.todense())
        Cd = self._Cdense
        vals = np.zeros((self.n_basis, m))
        dimsL = self.dims[self.L]
        for a in range(self.deg[0] + 1):
            wa, ia = stncl[0][0][a], stncl[0][1][a]
            for b in range(self.deg[1] + 1):
                wb, ib = stncl[1][0][b], stncl[1][1][b]
                wab = wa * wb
                for cc in range(self.deg[2] + 1):
                    wc, ic = stncl[2][0][cc], stncl[2][1][cc]
                    flat = (ia * dimsL[1] + ib) * dimsL[2] + ic     # (M,)
                    vals += Cd[:, flat] * (wab * wc)[None, :]
        return vals

    @property
    def n_basis(self):
        return len(self.basis)

    # -- SCALABLE localized evaluation (no dense finest expansion) ----------
    def _active_coeff_dicts(self, coeff):
        """Per-level {(i,j,k): coeff} dicts of the active THB coefficients."""
        dicts, off = [], 0
        for l in range(self.L + 1):
            idx = self.active_idx[l]
            dicts.append({tuple(int(v) for v in idx[k]): float(coeff[off + k])
                          for k in range(len(idx))})
            off += len(idx)
        return dicts

    def _is_refined(self, l, cell):
        return bool(self.refined[l][tuple(cell)])

    def _is_active(self, l, key):
        return bool(self._active_mask[l][tuple(key)])

    def _leaf_level(self, x):
        """Deepest level whose cell at ``x`` is refined (present)."""
        l = 0
        while l < self.L:
            cell = tuple(int(np.floor(x[a] * 2 ** l)) % self.dims[l][a]
                         for a in range(3))
            if self._is_refined(l, cell):
                l += 1
            else:
                break
        return l

    def _covering(self, x, dims_l, level):
        """The (p+1) covering B-spline indices per axis at ``x`` (level)."""
        cell = np.floor(x * 2 ** level).astype(np.int64)
        return [(cell[a] - self.deg[a] + np.arange(self.deg[a] + 1)) % dims_l[a]
                for a in range(3)]

    def _leaf_patch(self, x, actco, masks):
        """Localized truncated subdivide-accumulate: the leaf-level coefficient
        patch (dict) covering ``x`` plus its leaf level.  Bounded work, no
        finest grid."""
        from collections import defaultdict
        lleaf = self._leaf_level(x)
        cov = self._covering(x, self.dims[0], 0)
        patch = {}
        for ia in cov[0]:
            for ib in cov[1]:
                for ic in cov[2]:
                    key = (int(ia), int(ib), int(ic))
                    if key in actco[0]:
                        patch[key] = actco[0][key]
        for l in range(lleaf):
            dn = self.dims[l + 1]
            sub = defaultdict(float)
            for (i, j, k), co in patch.items():
                if co == 0.0:
                    continue
                for a in range(self.deg[0] + 2):
                    ci = (2 * i + a) % dn[0]
                    wa = masks[0][a]
                    for b in range(self.deg[1] + 2):
                        cj = (2 * j + b) % dn[1]
                        wab = wa * masks[1][b]
                        for c in range(self.deg[2] + 2):
                            ck = (2 * k + c) % dn[2]
                            sub[(ci, cj, ck)] += wab * masks[2][c] * co
            covn = self._covering(x, dn, l + 1)
            patch = {}
            for ia in covn[0]:
                for ib in covn[1]:
                    for ic in covn[2]:
                        key = (int(ia), int(ib), int(ic))
                        if key in actco[l + 1]:      # truncate coarse + place active
                            patch[key] = actco[l + 1][key]
                        elif key in sub:
                            patch[key] = sub[key]
        return patch, lleaf

    def eval_field_local(self, coeff, points):
        """Evaluate the THB field (coeffs ``coeff``) at ``points`` WITHOUT the
        dense finest expansion (the memory-scalable primitive).  Matches
        ``coeff @ eval_matrix(points)`` exactly."""
        actco = self._active_coeff_dicts(np.asarray(coeff, dtype=np.float64))
        masks = [subdivision_mask(self.deg[a]) for a in range(3)]
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = np.empty(len(pts))
        for pi in range(len(pts)):
            x = pts[pi]
            patch, lleaf = self._leaf_patch(x, actco, masks)
            xs = x * 2 ** lleaf
            frac = xs - np.floor(xs)
            w = [_deboor_weights(np.array([frac[a]]), self.deg[a])[:, 0]
                 for a in range(3)]
            covL = self._covering(x, self.dims[lleaf], lleaf)
            val = 0.0
            for a in range(self.deg[0] + 1):
                for b in range(self.deg[1] + 1):
                    wab = w[0][a] * w[1][b]
                    for c in range(self.deg[2] + 1):
                        key = (int(covL[0][a]), int(covL[1][b]),
                               int(covL[2][c]))
                        val += wab * w[2][c] * patch.get(key, 0.0)
            out[pi] = val
        return out

    def _basis_index(self):
        """Map (level, i, j, k) -> global THB coefficient index."""
        if not hasattr(self, "_bidx"):
            self._bidx = {(l,) + ijk: n
                          for n, (l, ijk) in enumerate(self.basis)}
        return self._bidx

    def _numba_tables(self):
        """Flat numba-friendly hierarchy tables for :func:`_leaf_transfer_numba`
        (built once, cached): ``dims`` (L+1, 3); ``masks`` (3, max deg+2);
        ``lev_offset`` (L+1) = per-level base of a globally-unique cell code
        ``lev_offset[l] + (i*dims[l,1]+j)*dims[l,2]+k``; and ``code2coeff`` (a
        numba typed dict) mapping that code -> global THB coefficient index for
        the ACTIVE functions (absent = not active).  Encodes both the
        membership test (``_is_active``) and the index (``_basis_index``) so the
        njit descent needs neither the dense masks nor the sparse sets.
        ``refined`` (a typed set of refined-cell codes, levels 0..L-1) drives
        the numba leaf-level walk."""
        if getattr(self, "_nbtab", None) is None:
            dims = np.array(self.dims, dtype=np.int64)              # (L+1, 3)
            lev_offset = np.zeros(self.L + 1, dtype=np.int64)
            acc = 0
            for l in range(self.L + 1):
                lev_offset[l] = acc
                acc += int(dims[l, 0]) * int(dims[l, 1]) * int(dims[l, 2])
            maxmask = max(d + 2 for d in self.deg)
            masks = np.zeros((3, maxmask))
            for a in range(3):
                m = subdivision_mask(self.deg[a])
                masks[a, :m.shape[0]] = m
            c2c = Dict.empty(types.int64, types.int64)
            for gi, (l, ijk) in enumerate(self.basis):
                code = (int(lev_offset[l])
                        + (ijk[0] * int(dims[l, 1]) + ijk[1]) * int(dims[l, 2])
                        + ijk[2])
                c2c[np.int64(code)] = np.int64(gi)
            refined = Dict.empty(types.int64, types.int64)          # code -> 1
            rset = getattr(self, "refined_set", None)               # THBSparse
            for l in range(self.L):
                cells = (rset[l] if rset is not None
                         else (tuple(int(v) for v in c)
                               for c in np.argwhere(self.refined[l])))
                for (i, j, k) in cells:
                    code = (int(lev_offset[l])
                            + (i * int(dims[l, 1]) + j) * int(dims[l, 2]) + k)
                    refined[np.int64(code)] = np.int64(1)
            self._nbtab = (dims, masks, lev_offset, c2c, refined)
        return self._nbtab

    def _point_contributions(self, x, masks, bidx, deriv_axis):
        """{coeff_index: weight} = the row of the eval (or grad) operator at
        ``x`` (i.e. beta_i(x) [or d/dx_axis]).  The transpose of the localized
        subdivide-accumulate; used by both :meth:`assemble_local` and
        :meth:`eval_operator`."""
        from collections import defaultdict
        lleaf = self._leaf_level(x)
        xs = x * 2 ** lleaf
        frac = xs - np.floor(xs)
        w = []
        for a in range(3):
            if a == deriv_axis:
                w.append(_deboor_deriv_weights(frac[a], self.deg[a])
                         * (2 ** lleaf))
            else:
                w.append(_deboor_weights(np.array([frac[a]]),
                                         self.deg[a])[:, 0])
        covL = self._covering(x, self.dims[lleaf], lleaf)
        patch_adj = defaultdict(float)
        for a in range(self.deg[0] + 1):
            for b in range(self.deg[1] + 1):
                wab = w[0][a] * w[1][b]
                for c in range(self.deg[2] + 1):
                    key = (int(covL[0][a]), int(covL[1][b]), int(covL[2][c]))
                    patch_adj[key] += wab * w[2][c]
        contrib = defaultdict(float)
        for l in range(lleaf - 1, -1, -1):
            dl = self.dims[l]
            covn = self._covering(x, self.dims[l + 1], l + 1)
            covset = {(int(ia), int(ib), int(ic))
                      for ia in covn[0] for ib in covn[1] for ic in covn[2]}
            coarse = defaultdict(float)
            for key in covset:
                v = patch_adj.get(key, 0.0)
                if v == 0.0:
                    continue
                if self._is_active(l + 1, key):            # active -> a coeff
                    contrib[bidx[(l + 1,) + key]] += v
                else:                                      # restrict to level l
                    ci, cj, ck = key
                    for a in range(ci % 2, self.deg[0] + 2, 2):
                        pi_ = ((ci - a) // 2) % dl[0]
                        wa = masks[0][a]
                        for b in range(cj % 2, self.deg[1] + 2, 2):
                            pj_ = ((cj - b) // 2) % dl[1]
                            wab = wa * masks[1][b]
                            for cc in range(ck % 2, self.deg[2] + 2, 2):
                                pk_ = ((ck - cc) // 2) % dl[2]
                                coarse[(pi_, pj_, pk_)] += wab * masks[2][cc] * v
            patch_adj = coarse
        cov0 = self._covering(x, self.dims[0], 0)
        for key in {(int(ia), int(ib), int(ic))     # dedup: aliased covering
                    for ia in cov0[0]               # keys (periodic wrap when
                    for ib in cov0[1]               # dims < p+1) are ONE coeff
                    for ic in cov0[2]}:
            v = patch_adj.get(key, 0.0)
            if v != 0.0 and self._is_active(0, key):
                contrib[bidx[(0,) + key]] += v
        return contrib

    def assemble_local(self, values, points, deriv_axis=None):
        """Adjoint of :meth:`eval_field_local` (``deriv_axis=None``) or of
        :meth:`eval_grad_local` (``deriv_axis=axis``): ``out_i = sum_q values_q
        beta_i(x_q)`` [or its ``d/dx_axis``].  Satisfies the adjoint identity
        ``<eval[_grad](c), y> == <c, assemble(y[, axis])>``."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        masks = [subdivision_mask(self.deg[a]) for a in range(3)]
        bidx = self._basis_index()
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = np.zeros(self.n_basis)
        for pi in range(len(pts)):
            for idx, wt in self._point_contributions(
                    pts[pi], masks, bidx, deriv_axis).items():
                out[idx] += values[pi] * wt
        return out

    def eval_operator(self, points, deriv_axis=None):
        """Sparse ``(Q, n_basis)`` matrix ``E[q,i] = beta_i(x_q)`` (or its
        ``d/dx_axis``) built ONCE from the localized descent.  Then eval = E@c,
        adjoint = E.T@y, mass = E.T diag(w) E - all fast sparse algebra, reused
        across CG iterations instead of re-walking the tree each time."""
        from scipy import sparse
        masks = [subdivision_mask(self.deg[a]) for a in range(3)]
        bidx = self._basis_index()
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rows, cols, vals = [], [], []
        for pi in range(len(pts)):
            for idx, wt in self._point_contributions(
                    pts[pi], masks, bidx, deriv_axis).items():
                rows.append(pi)
                cols.append(idx)
                vals.append(wt)
        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(len(pts), self.n_basis))

    def _leaf_transfer(self, lleaf, cell, masks, bidx):
        """T_leaf: {coeff_index -> weight-vector over the (p+1)^3 leaf-covering
        keys}.  ONE reverse descent (vector-valued), shared by every point in
        the leaf (the descent is leaf-only, independent of sub-position)."""
        from collections import defaultdict
        p = self.deg
        npatch = (p[0] + 1) * (p[1] + 1) * (p[2] + 1)
        dL = self.dims[lleaf]
        cov = [(cell[a] - p[a] + np.arange(p[a] + 1)) % dL[a] for a in range(3)]
        patch = defaultdict(lambda: np.zeros(npatch))
        jj = 0
        for a in range(p[0] + 1):
            for b in range(p[1] + 1):
                for c in range(p[2] + 1):
                    patch[(int(cov[0][a]), int(cov[1][b]), int(cov[2][c]))][jj] += 1.0
                    jj += 1
        contrib = defaultdict(lambda: np.zeros(npatch))
        for l in range(lleaf - 1, -1, -1):
            dl = self.dims[l]
            anc1 = tuple(cell[a] >> (lleaf - (l + 1)) for a in range(3))
            covn = [(anc1[a] - p[a] + np.arange(p[a] + 1)) % self.dims[l + 1][a]
                    for a in range(3)]
            new = defaultdict(lambda: np.zeros(npatch))
            for key in {(int(ia), int(ib), int(ic))    # dedup aliased keys
                        for ia in covn[0]              # (see _point_contributions)
                        for ib in covn[1]
                        for ic in covn[2]}:
                if key not in patch:
                    continue
                vec = patch[key]
                if self._is_active(l + 1, key):
                    contrib[bidx[(l + 1,) + key]] += vec
                else:
                    ci, cj, ck = key
                    for a in range(ci % 2, p[0] + 2, 2):
                        pi_ = ((ci - a) // 2) % dl[0]; wa = masks[0][a]
                        for b in range(cj % 2, p[1] + 2, 2):
                            pj_ = ((cj - b) // 2) % dl[1]; wab = wa * masks[1][b]
                            for cc in range(ck % 2, p[2] + 2, 2):
                                pk_ = ((ck - cc) // 2) % dl[2]
                                new[(pi_, pj_, pk_)] += wab * masks[2][cc] * vec
            patch = new
        anc0 = tuple(cell[a] >> lleaf for a in range(3))
        cov0 = [(anc0[a] - p[a] + np.arange(p[a] + 1)) % self.dims[0][a]
                for a in range(3)]
        for key in {(int(ia), int(ib), int(ic))        # dedup aliased keys
                    for ia in cov0[0]
                    for ib in cov0[1]
                    for ic in cov0[2]}:
            if key in patch and self._is_active(0, key):
                contrib[bidx[(0,) + key]] += patch[key]
        return contrib

    def eval_operator_fast(self, points, deriv_axis=None, dtype=np.float64):
        """Same as :meth:`eval_operator` but the leaf-only descent is done ONCE
        per unique leaf (T_leaf) and reused for all points in it - the per-leaf
        perf pass.  With numba the ENTIRE build (leaf assignment, grouping, the
        descent, the (p+1)^3 B-spline weighting, and COO assembly) runs in ONE
        njit call (:func:`_eval_operators_numba`) - no per-leaf dispatch, no
        Python by_leaf/BW glue.  Falls back to the pure-Python per-leaf build
        (grouping + :meth:`_leaf_transfer`) when numba is unavailable.

        ``dtype`` sets the sparse ``data`` type (indices are int32 already):
        ``float32`` halves operator memory (the readout is smooth so single
        precision is ample) - opt-in for memory-bound deep grids; the default
        ``float64`` is byte-exact.

        Need several operators at the SAME points (the curl needs 2 derivative
        axes per component)?  Use :meth:`eval_operators_fast` - one descent and
        one shared index array instead of one of each."""
        return self.eval_operators_fast(points, [deriv_axis], dtype)[0]

    def eval_operators_fast(self, points, deriv_axes, dtype=np.float64):
        """K operators at the SAME points from ONE descent, SHARING one
        ``indices``/``indptr`` pair.  ``deriv_axes`` is a list whose entries are
        ``None`` (the value operator) or an axis 0/1/2 (d/dx_axis).

        Why it matters: the descent and the sparsity pattern depend only on the
        LEAF, not on which derivative is taken, so K separate calls redo the
        same descent K times and store K copies of the same index array.  Here
        the curl's 2 derivative operators per S1 component cost ONE index array
        between them - 12 bytes/nnz instead of 16 at float32 data (-25%), and
        one descent instead of two.  Values are bit-identical to the
        single-operator build; the shared pattern is the union of the K
        patterns, so an operator may carry an explicit stored zero (inert)."""
        from scipy import sparse
        p = self.deg
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        axes = [(-1 if a is None else int(a)) for a in deriv_axes]
        if not _HAVE_NUMBA:
            return [self._eval_operator_fast_py(pts, None if a < 0 else a,
                                                dtype) for a in axes]
        dims_a, masks_a, lev_off, c2c, refined = self._numba_tables()
        npatch = int((p[0] + 1) * (p[1] + 1) * (p[2] + 1))
        npts = len(pts)
        # Work split for the threaded leaf loop.  Several chunks per thread so
        # a chunk that draws deep (expensive) leaves cannot stall the pass;
        # invisible in the output, which is byte-identical for any nchunk.
        try:
            from numba import get_num_threads
            nchunk = max(1, 4 * get_num_threads())
        except Exception:                              # pragma: no cover
            nchunk = 8
        nchunk = int(min(nchunk, max(1, npts)))
        cols, vals, cnt, src, out = _eval_operators_numba(
            np.ascontiguousarray(pts), np.array(axes, dtype=np.int64),
            np.array(p, dtype=np.int64), dims_a, masks_a, lev_off, c2c,
            refined, np.int64(npatch), np.int64(self.L), np.int64(nchunk),
            np.empty(0, dtype=dtype))
        if out > np.iinfo(np.int32).max:            # would overflow indptr
            raise MemoryError(f"THB operator has {out} nonzeros, above the "
                              f"int32 CSR limit; reduce nq or cap the level")
        # leaf-grouped blocks -> CSR, in place into the final arrays.  Going
        # through csr_matrix((v,(r,c))) instead would rebuild the whole COO->CSR
        # conversion once PER operator and hold a second copy of everything.
        indptr = np.zeros(npts + 1, dtype=np.int32)
        np.cumsum(cnt, out=indptr[1:], dtype=np.int32)
        indices = np.empty(out, dtype=np.int32)
        data = np.empty((len(axes), out), dtype=dtype)
        _gather_csr(cols, vals, cnt, src, indptr, indices, data)
        del cols, vals
        mats = []
        for k in range(len(axes)):
            m = sparse.csr_matrix((data[k], indices, indptr),
                                  shape=(npts, self.n_basis), copy=False)
            m.has_sorted_indices = True
            mats.append(m)
        return mats

    def _eval_operator_fast_py(self, pts, deriv_axis, dtype=np.float64):
        """Pure-Python fallback for :meth:`eval_operator_fast` (no numba): group
        points by leaf, one :meth:`_leaf_transfer` per leaf, apply per-point
        B-spline weights."""
        from collections import defaultdict
        from scipy import sparse
        p = self.deg
        masks = [subdivision_mask(a) for a in p]
        bidx = self._basis_index()
        by_leaf = defaultdict(list)
        for q in range(len(pts)):
            x = pts[q]
            ll = self._leaf_level(x)
            cell = tuple(int(np.floor(x[a] * 2 ** ll)) % self.dims[ll][a]
                         for a in range(3))
            by_leaf[(ll, cell)].append(q)
        rows, cols, vals = [], [], []
        for (ll, cell), qlist in by_leaf.items():
            T = self._leaf_transfer(ll, cell, masks, bidx)
            if not T:
                continue
            coeff_ids = np.fromiter(T.keys(), dtype=np.int64, count=len(T))
            Tmat = np.array([T[c] for c in coeff_ids])          # (ncoeff, npatch)
            qs = np.asarray(qlist)
            frac = pts[qs] * 2 ** ll - np.floor(pts[qs] * 2 ** ll)   # (npts,3)
            W = []
            for a in range(3):
                if a == deriv_axis:
                    wa = np.array([_deboor_deriv_weights(f, p[a])
                                   for f in frac[:, a]]) * (2 ** ll)   # (npts,p+1)
                else:
                    wa = _deboor_weights(frac[:, a], p[a]).T            # (npts,p+1)
                W.append(wa)
            BW = (W[0][:, :, None, None] * W[1][:, None, :, None]
                  * W[2][:, None, None, :]).reshape(len(qs), -1)      # (npts,npatch)
            Emat = BW @ Tmat.T                                         # (npts,ncoeff)
            nz = np.nonzero(Emat)
            rows.extend(qs[nz[0]].tolist())
            cols.extend(coeff_ids[nz[1]].tolist())
            vals.extend(Emat[nz].tolist())
        return sparse.csr_matrix(
            (np.asarray(vals, dtype=dtype), (rows, cols)),
            shape=(len(pts), self.n_basis))

    def eval_grad_local(self, coeff, points, axis):
        """d/dx_axis of the THB field at ``points`` (level-0 units), localized.
        The leaf-level patch is the field near x, so its analytic B-spline
        derivative is the field's derivative (du/dx = 2**lleaf)."""
        actco = self._active_coeff_dicts(np.asarray(coeff, dtype=np.float64))
        masks = [subdivision_mask(self.deg[a]) for a in range(3)]
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = np.empty(len(pts))
        for pi in range(len(pts)):
            x = pts[pi]
            patch, lleaf = self._leaf_patch(x, actco, masks)
            xs = x * 2 ** lleaf
            frac = xs - np.floor(xs)
            w = []
            for a in range(3):
                if a == axis:
                    w.append(_deboor_deriv_weights(frac[a], self.deg[a])
                             * (2 ** lleaf))       # du/dx
                else:
                    w.append(_deboor_weights(np.array([frac[a]]),
                                             self.deg[a])[:, 0])
            covL = self._covering(x, self.dims[lleaf], lleaf)
            val = 0.0
            for a in range(self.deg[0] + 1):
                for b in range(self.deg[1] + 1):
                    wab = w[0][a] * w[1][b]
                    for c in range(self.deg[2] + 1):
                        key = (int(covL[0][a]), int(covL[1][b]),
                               int(covL[2][c]))
                        val += wab * w[2][c] * patch.get(key, 0.0)
            out[pi] = val
        return out


# ==========================================================================
# Sparse-octree THB: same localized operators, no dense per-level arrays
# ==========================================================================

class THBSparse(THB3D):
    """Scalable THB space: refined regions and active functions are stored as
    SPARSE SETS (no dense ``omega``/``refined``/``_active_mask``/``C`` arrays,
    no dense finest quadrature), so it works for deep hierarchies where the
    finest grid is astronomically large.  It reuses THB3D's localized
    evaluation / derivative / adjoint verbatim - those only need the membership
    tests :meth:`_is_refined` / :meth:`_is_active`, overridden here to hit
    sets.

    ``refined[l]`` may be a boolean 3D array (converted) or an iterable of
    ``(i,j,k)`` refined-cell tuples.  The active-function ordering matches
    :class:`THB3D` (lexicographic per level), so coefficients are compatible.
    """

    def __init__(self, base_dims, degrees, refined):
        self.base = tuple(int(b) for b in base_dims)
        self.deg = tuple(int(d) for d in degrees)
        self.L = len(refined)
        self.dims = [tuple(b * 2 ** l for b in self.base)
                     for l in range(self.L + 1)]
        self.refined_set = []
        for l in range(self.L):
            r = refined[l]
            if isinstance(r, np.ndarray):
                cells = (tuple(int(v) for v in c) for c in np.argwhere(r))
            else:
                cells = (tuple(int(v) for v in c) for c in r)
            # enforce nesting: a cell can refine only where its parent is present
            self.refined_set.append({c for c in cells if self._present(l, c)})
        self._select_active_sparse()

    # -- sparse membership -------------------------------------------------
    def _present(self, l, cell):
        if l == 0:
            return True
        parent = (cell[0] // 2, cell[1] // 2, cell[2] // 2)
        return parent in self.refined_set[l - 1]

    def _is_refined(self, l, cell):
        return tuple(cell) in self.refined_set[l]

    def _is_active(self, l, key):
        return tuple(key) in self.active_set[l]

    # -- sparse Kraft selection -------------------------------------------
    def _present_cells(self, l):
        """Cells that EXIST at level ``l``: every level-0 cell, and deeper the
        8 children of each cell refined at ``l-1``."""
        if l == 0:
            return [(i, j, k) for i in range(self.base[0])
                    for j in range(self.base[1]) for k in range(self.base[2])]
        out = []
        for (pi, pj, pk) in self.refined_set[l - 1]:
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        out.append((2 * pi + di, 2 * pj + dj, 2 * pk + dk))
        return out

    def _candidate_functions(self, l, present=None):
        """Functions whose support touches the present region at level ``l``."""
        dims = self.dims[l]
        if present is None:
            present = self._present_cells(l)
        cand = set()
        for (ci, cj, ck) in present:
            for a in range(self.deg[0] + 1):
                ki = (ci - a) % dims[0]
                for b in range(self.deg[1] + 1):
                    kj = (cj - b) % dims[1]
                    for c in range(self.deg[2] + 1):
                        cand.add((ki, kj, (ck - c) % dims[2]))
        return cand

    def _support_cells(self, key, l):
        dims = self.dims[l]
        return [(key[0] + s0) % dims[0] for s0 in range(self.deg[0] + 1)], \
               [(key[1] + s1) % dims[1] for s1 in range(self.deg[1] + 1)], \
               [(key[2] + s2) % dims[2] for s2 in range(self.deg[2] + 1)]

    def _select_active_sparse(self):
        """Kraft selection: a function is active where its whole support is
        present but not entirely inside the finer region.

        The two membership tests run over every candidate's (p+1)^3 support -
        millions of lookups, and the dominant cost of building a deep space.
        They are done on INTEGER CELL CODES rather than (i,j,k) tuples: same
        sets, but int hashing instead of tuple hashing and no per-cell tuple
        allocation.  ``present`` is also built once per level and shared with
        the candidate sweep, instead of re-deriving each cell's parent through
        :meth:`_present`.  Selection loop 1.11 -> 0.71 s and the whole
        3-component space build 2.95 -> 1.86 s on the galaxy-shaped L3 mesh;
        the active sets are IDENTICAL (checked against the tuple version on
        four meshes)."""
        self.active_set, self.active_idx = [], []
        p0, p1, p2 = self.deg
        for l in range(self.L + 1):
            d0, d1, d2 = self.dims[l]
            present_cells = self._present_cells(l)
            present = {(i * d1 + j) * d2 + k for (i, j, k) in present_cells}
            refined = (set() if l >= self.L else
                       {(i * d1 + j) * d2 + k
                        for (i, j, k) in self.refined_set[l]})
            act = set()
            for key in self._candidate_functions(l, present_cells):
                sx = [(key[0] + s) % d0 for s in range(p0 + 1)]
                sy = [(key[1] + s) % d1 for s in range(p1 + 1)]
                sz = [(key[2] + s) % d2 for s in range(p2 + 1)]
                ok, allref = True, l < self.L
                for i in sx:
                    for j in sy:
                        ij = (i * d1 + j) * d2
                        for k in sz:
                            code = ij + k
                            if code not in present:
                                ok = False
                                break
                            if allref and code not in refined:
                                allref = False
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok or allref:   # absent support / fully refined -> drop
                    continue
                act.add(key)
            self.active_set.append(act)
            self.active_idx.append(
                np.array(sorted(act), dtype=np.int64).reshape(-1, 3) if act
                else np.zeros((0, 3), dtype=np.int64))
        self.basis = [(l, tuple(int(v) for v in ijk))
                      for l in range(self.L + 1) for ijk in self.active_idx[l]]

    # -- leaves + per-leaf Gauss quadrature (scalable, no finest grid) ------
    def leaves(self):
        """Active leaves ``(level, (i,j,k))``: present cells not further
        refined (all present cells at the finest level)."""
        out = []
        for i in range(self.base[0]):
            for j in range(self.base[1]):
                for k in range(self.base[2]):
                    cell = (i, j, k)
                    if self.L == 0 or cell not in self.refined_set[0]:
                        out.append((0, cell))
        for l in range(1, self.L + 1):
            for (pi, pj, pk) in self.refined_set[l - 1]:
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            cell = (2 * pi + di, 2 * pj + dj, 2 * pk + dk)
                            if l == self.L or cell not in self.refined_set[l]:
                                out.append((l, cell))
        return out

    def leaf_quadrature(self, nq):
        """Tensor Gauss quadrature over every active leaf (level-0 cell units).
        Returns (points (Q,3), weights (Q,)); integral ~ sum w f(points).

        Filled into ONE preallocated (Q,3) array: a per-leaf list + concatenate
        holds two full copies at the peak (plus ~400k array objects) on a deep
        galaxy octree."""
        gx, gw = np.polynomial.legendre.leggauss(nq)
        gx = 0.5 * (gx + 1.0)
        gw = 0.5 * gw
        gp = np.array([(a, b, c) for a in gx for b in gx for c in gx])
        gwt = np.array([wa * wb * wc for wa in gw for wb in gw for wc in gw])
        leaves = self.leaves()
        n, m = len(leaves), len(gp)
        if n == 0:
            return np.zeros((0, 3)), np.zeros(0)
        pts = np.empty((n * m, 3))
        wts = np.empty(n * m)
        for i, (l, cell) in enumerate(leaves):
            h = 2.0 ** (-l)
            s = i * m
            np.multiply(gp, h, out=pts[s:s + m])
            pts[s:s + m] += np.asarray(cell, dtype=np.float64) * h
            np.multiply(gwt, h ** 3, out=wts[s:s + m])
        return pts, wts


# ==========================================================================
# Bridge: sph_interp AMRGrid -> THBSparse hierarchy
# ==========================================================================

def amr_refined_sets(amr):
    """Extract ``(base_dims, refined)`` from an sph_interp ``AMRGrid`` for
    :class:`THBSparse`.  ``refined[l]`` = the level-``l`` cells that are
    subdivided (i.e. have a deeper leaf below them) = every leaf's ancestors.
    The dyadic octree maps directly onto the THB dyadic hierarchy."""
    base = tuple(int(b) for b in amr.base_npx)
    lev = np.asarray(amr.cell_level, dtype=np.int64).reshape(-1)
    L = int(lev.max()) if lev.size else 0
    lo = np.asarray(amr.lo, dtype=np.float64).reshape(3)
    base_pw = np.asarray(amr.base_pw, dtype=np.float64).reshape(3)
    cell_lo = np.asarray(amr.cell_lo, dtype=np.float64)
    refined = [set() for _ in range(L)]
    for n in range(cell_lo.shape[0]):
        lv = int(lev[n])
        if lv == 0:
            continue
        cw = base_pw / (2 ** lv)
        idx = np.round((cell_lo[n] - lo) / cw).astype(np.int64)
        for l in range(lv):
            sh = lv - l
            refined[l].add((int(idx[0]) >> sh, int(idx[1]) >> sh,
                            int(idx[2]) >> sh))
    return base, refined
