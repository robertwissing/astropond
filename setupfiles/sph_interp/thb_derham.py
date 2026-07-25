"""Hierarchical (THB) spline De Rham readout - projector + curl (rung 3).

Builds on `thb.THB3D` (the truncated hierarchical basis) and the uniform
De Rham operators in `spline_derham` (exact curl on the finest level).  The
vector potential A is L2-projected into the hierarchical H(curl) space S1
(three degree-dropped THB components), then read back as ``B = curl A`` -
smooth and exactly divergence-free on the graded mesh.

Two objectives share this core (FIELD_RECONSTRUCTION_PLAN, "TWO OBJECTIVES,
SAME CORE"):
  - OBJ-A (this module): project A into S1, then curl.  Optimal for A.
  - OBJ-B (curl-fit, follow-up): fit A so ``curl A`` matches a target B
    directly (the smooth Hodge projection of B) - reuses the S2 mass matrix
    and the curl incidence.

PROTOTYPE scope: the readout expands each THB function onto the finest uniform
level (exact but ~prod(dims_L) memory) and applies the uniform circulant
curl there.  Correct and fully validating on shallow graded meshes; the deep
galaxy octree needs the sparse hierarchical incidence (a later increment).

Coordinates are LEVEL-0 CELL UNITS throughout: the domain is [0, base_a] per
axis with unit level-0 spacing, so the finest spacing is ``2**-L`` and the
curl uses ``1/h_finest = 2**L``.
"""

import numpy as np

from .thb import THB3D, THBSparse
from ._spmv import spmv, spmv_t, col_sq
from .spline_derham import (_A_component_degrees, _B_component_degrees,
                            _diff, _eval_scalar)


def _gauss_1d(nq):
    """Gauss-Legendre nodes/weights mapped to [0, 1)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    return 0.5 * (x + 1.0), 0.5 * w


def _weighted_col_sq(E, w, chunk=1 << 20):
    """``diag_j = sum_q w_q E[q,j]^2`` (the Jacobi preconditioner of ``E^T W E``)
    without materialising anything of size nnz: ``E.multiply(E)`` would build a
    full COO copy of E (24 B/nnz) purely to be reduced away again.  Accumulated
    in float64 over row chunks, so the result is independent of E's dtype.

    Uses the threaded kernel when numba is available (20x: 77 -> 3.5 ms on the
    L3 operator); the chunked bincount below is the fallback and the
    cross-check.  They agree to 9e-15 - a reassociation, and this is a
    PRECONDITIONER, which changes the iterate path, not the system."""
    fast = col_sq(E, w)
    if fast is not None:
        return fast
    out = np.zeros(E.shape[1], dtype=np.float64)
    indptr = E.indptr
    for lo in range(0, E.shape[0], chunk):
        hi = min(lo + chunk, E.shape[0])
        s, e = indptr[lo], indptr[hi]
        if s == e:
            continue
        d = E.data[s:e].astype(np.float64)
        d *= d
        d *= np.repeat(w[lo:hi], np.diff(indptr[lo:hi + 1]))
        out += np.bincount(E.indices[s:e], weights=d, minlength=E.shape[1])
    return out


def _finest_quadrature(base, L, nq):
    """Tensor Gauss quadrature over every finest cell, in level-0 cell units.
    Returns (points (Q,3), weights (Q,)); integral ~ sum w f(points)."""
    gx, gw = _gauss_1d(nq)
    hf = 2.0 ** (-L)                       # finest cell width (level-0 units)
    axes_pts, axes_w = [], []
    for a in range(3):
        ncell = base[a] * 2 ** L
        edges = np.arange(ncell) * hf                       # (ncell,)
        pts = (edges[:, None] + hf * gx[None, :]).ravel()   # (ncell*nq,)
        wts = np.tile(gw * hf, ncell)
        axes_pts.append(pts)
        axes_w.append(wts)
    PX, PY, PZ = np.meshgrid(*axes_pts, indexing="ij")
    WX, WY, WZ = np.meshgrid(*axes_w, indexing="ij")
    pts = np.column_stack([PX.ravel(), PY.ravel(), PZ.ravel()])
    w = (WX * WY * WZ).ravel()
    return pts, w


class THBDeRham:
    """Hierarchical spline De Rham readout on a periodic dyadic THB hierarchy.

    base   : (3,) level-0 cell counts.
    degree : top (S0) degree p (cubic default).
    refined: THB refinement (list of boolean 3D arrays, see THB3D); [] = the
             uniform space (reproduces the rung-1 complex).
    nq     : Gauss points per finest cell per axis (>= p+1 for exact mass).
    sparse : use the scalable THBSparse space + per-leaf quadrature (no dense
             per-level arrays / finest grid).  Only the *_local / *_matfree
             methods work in sparse mode (the dense project_A / curl_readout,
             which need the finest expansion, are disabled).
    """

    def __init__(self, base, degree=3, refined=None, nq=None, sparse=False,
                 op_dtype=np.float64):
        self.base = tuple(int(b) for b in base)
        self.p = int(degree)
        self.refined = refined or []
        self.L = len(self.refined)
        self.hf = 2.0 ** (-self.L)                     # finest spacing, lvl-0 units
        self.dimsL = tuple(b * 2 ** self.L for b in self.base)
        self.nq = int(nq) if nq else self.p + 1
        self.degA = _A_component_degrees(self.p)
        self.degB = _B_component_degrees(self.p)
        self.sparse = bool(sparse)
        # sparse-operator data dtype: float32 halves the (Q x nnz) grad/eval
        # operator memory for deep grids (smooth readout -> single precision is
        # ample).  float64 (default) keeps the matfree solves byte-exact.
        self.op_dtype = np.dtype(op_dtype)
        if self.sparse:
            self.S1 = [THBSparse(self.base, self.degA[c], self.refined)
                       for c in range(3)]
            # per-leaf Gauss quadrature (the S1 components share the geometry)
            self.qpts, self.qw = self.S1[0].leaf_quadrature(self.nq)
        else:
            self.S1 = [THB3D(self.base, self.degA[c], self.refined)
                       for c in range(3)]
            self._build_projector()

    # -- L2 (Galerkin) projector: mass matrix per S1 component -------------
    def _build_projector(self):
        self.qpts, self.qw = _finest_quadrature(self.base, self.L, self.nq)
        self._E = []            # eval of each S1 component's basis at quad pts
        self._M = []            # mass matrices (n_basis x n_basis)
        self._Mfac = []         # Cholesky-ish solve prep
        for c in range(3):
            E = self.S1[c].eval_matrix(self.qpts)          # (n_basis, Q)
            EW = E * self.qw[None, :]
            M = EW @ E.T
            self._E.append(E)
            self._M.append(M)
            self._Mfac.append(EW)

    def project_A(self, A_at_quad):
        """L2-project a vector potential into S1.  ``A_at_quad`` is (Q, 3):
        the field sampled at ``self.qpts`` (analytic values, or the table
        cell field sampled there).  Returns the 3 THB coefficient vectors."""
        A = np.ascontiguousarray(A_at_quad, dtype=np.float64)
        coeffs = []
        for c in range(3):
            b = self._Mfac[c] @ A[:, c]
            coeffs.append(np.linalg.solve(self._M[c], b))
        return coeffs

    # -- readout: B = curl A on the finest level, evaluated at positions ---
    def _A_finest(self, coeffs):
        return np.stack([self.S1[c].expand_to_finest(coeffs[c])
                         for c in range(3)], axis=-1)

    def _curl_finest(self, A_finest):
        h = np.full(3, self.hf)
        Ax, Ay, Az = A_finest[..., 0], A_finest[..., 1], A_finest[..., 2]
        Bx = _diff(Az, 1, h[1]) - _diff(Ay, 2, h[2])
        By = _diff(Ax, 2, h[2]) - _diff(Az, 0, h[0])
        Bz = _diff(Ay, 0, h[0]) - _diff(Ax, 1, h[1])
        return Bx, By, Bz

    def curl_readout(self, coeffs, positions):
        """B = curl(THB A) at ``positions`` (level-0 cell units, (M,3))."""
        Bx, By, Bz = self._curl_finest(self._A_finest(coeffs))
        pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        u = [np.mod(pos[:, a] / self.hf, self.dimsL[a]) for a in range(3)]
        return np.stack([_eval_scalar(Bx, u, self.degB[0], self.dimsL),
                         _eval_scalar(By, u, self.degB[1], self.dimsL),
                         _eval_scalar(Bz, u, self.degB[2], self.dimsL)], axis=1)

    def project_and_curl(self, A_at_quad, positions):
        return self.curl_readout(self.project_A(A_at_quad), positions)

    def _eval_ops(self):
        """Sparse eval operators E_c (Q, n_basis_c) per S1 component, built
        once (the localized descent walked once, then reused as fast sparse
        matmuls).  NOTE: holds all 3 at once - :meth:`project_A_matfree` builds
        them one at a time instead, since the components decouple there."""
        if not hasattr(self, "_Eop"):
            self._Eop = [self.S1[c].eval_operator_fast(self.qpts,
                                                       dtype=self.op_dtype)
                         for c in range(3)]
        return self._Eop

    def project_A_matfree(self, A_at_quad, rtol=1e-10, maxiter=500,
                          solver="direct"):
        """SCALABLE L2 projection of A into S1 (VARIANT 1).  Builds the sparse
        eval operator once (no finest grid, no dense mass matrix materialised as
        n_basis^2 dense), then solves the SPD system ``E^T W E a = E^T W A``.
        Matches :meth:`project_A`.

        ``solver='direct'`` (default) forms the sparse mass matrix and factors
        it (``spsolve``) - exact, fine up to ~1e4 DOF.  ``solver='cg'`` never
        forms it: Jacobi-preconditioned CG with ``M v = E^T W (E v)``.  The mass
        matrix is SPD and diagonal scaling removes the graded-mesh level scaling
        (coarse functions integrate 2^(3l) more than fine ones), so CG converges
        in O(10) iterations - and the factorisation FILL-IN, not the matrix, is
        what makes the direct path infeasible on a deep octree (n_basis ~ 1e5).
        """
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator, cg, spsolve
        A = np.ascontiguousarray(A_at_quad, dtype=np.float64)
        coeffs = []
        cached = hasattr(self, "_Eop")
        for c in range(3):
            # The three components DECOUPLE (block-diagonal mass matrix), so
            # build this component's operator, solve, and drop it - peak is ONE
            # eval operator instead of the three `_eval_ops` would cache.
            Ec = (self._Eop[c] if cached
                  else self.S1[c].eval_operator_fast(self.qpts,
                                                     dtype=self.op_dtype))
            b = spmv_t(Ec, self.qw * A[:, c])       # E^T W A
            if solver == "direct":
                # scale the ROWS of E by w, sharing E's index arrays, instead of
                # E.T.multiply(w) - which materialises a whole COO copy of E
                # (24 B/nnz) just to weight it.
                Ew = sparse.csr_matrix(
                    (Ec.data * np.repeat(self.qw, np.diff(Ec.indptr)),
                     Ec.indices, Ec.indptr), shape=Ec.shape, copy=False)
                M = (Ec.T @ Ew).tocsc()
                del Ew
                coeffs.append(spsolve(M, b))
                del M
            elif solver == "cg":
                n = Ec.shape[1]
                diag = _weighted_col_sq(Ec, self.qw)
                diag[diag <= 0.0] = 1.0             # unreached basis fn -> no-op
                # threaded matvecs (see _spmv): the mass solve is ~90% CG, and
                # CG is ~all of these two products.
                def _mass(v, Ec=Ec):
                    r = spmv(Ec, v)
                    r *= self.qw                     # in place, r is a temp
                    return spmv_t(Ec, r)

                K = LinearOperator((n, n), dtype=np.float64, matvec=_mass)
                P = LinearOperator((n, n), dtype=np.float64,
                                   matvec=lambda v, d=diag: v / d)
                a, info = cg(K, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=P)
                if info != 0:
                    import warnings
                    res = np.linalg.norm(K @ a - b) / max(np.linalg.norm(b),
                                                          1e-300)
                    warnings.warn(f"THB mass CG did not converge (component "
                                  f"{c}, info={info}, rel-residual {res:.2e})")
                coeffs.append(a)
            else:
                raise ValueError(f"unknown solver {solver!r} "
                                 f"(expected 'direct' or 'cg')")
            if not cached:
                del Ec                              # release before the next
        return coeffs

    def project_and_curl_matfree(self, A_at_quad, positions, Bbar=None,
                                 solver="cg", rtol=1e-8, maxiter=500,
                                 return_A=False):
        """SCALABLE VARIANT 1 end to end: L2-project A into S1, read back
        ``B = curl A`` at ``positions`` - the scalable twin of
        :meth:`project_and_curl`, and the A-fed counterpart of
        :meth:`bfit_matfree` (which fits B directly).

        ``A_at_quad`` is (Q,3) at ``self.qpts``; a per-leaf (table/Petkova)
        A is just its leaf value broadcast to that leaf's quad points, exactly
        as :meth:`cells_to_quad` does for a finest-grid cell field.

        NOTE ON THE MEAN: ``curl`` of the periodic S1 space is mean-free, so
        this readout carries NO uniform field - unlike :meth:`bfit_matfree`,
        which splits the target's ``Bbar`` off and restores it analytically.
        Pass ``Bbar`` to add one back (the A cannot supply it).

        ``return_A=True`` also returns the S1 coefficients (for
        :meth:`helicity` / :meth:`eval_A`)."""
        coeffs = self.project_A_matfree(A_at_quad, rtol=rtol, maxiter=maxiter,
                                        solver=solver)
        B = self.curl_readout_fast(coeffs, positions, Bbar=Bbar)
        return (B, coeffs) if return_A else B

    def free_operators(self):
        """Drop the cached quad-point sparse operators (``_Eop`` = the value ops
        the projector uses, ``_Gop`` = the 6 grad ops the curl-fit uses).  They
        are the memory wall on deep grids (Q x nnz each), so a driver that runs
        BOTH variants on one space should free between them."""
        for name in ("_Eop", "_Gop"):
            if hasattr(self, name):
                delattr(self, name)

    def bfit_matfree(self, B_at_quad, positions, Bbar=None, rtol=1e-8,
                     maxiter=1000, return_A=False):
        """SCALABLE curl-fit (variant 2): fit A so ``curl A`` matches B via
        matrix-free CG on the curl-curl system ``K = G^T W G`` (G = curl at the
        quad points, G^T via the grad-adjoints), then read ``B_h = curl A +
        Bbar``.  K is singular on the gauge (grad S0) but the RHS is consistent,
        so CG converges; ``B_h`` is gauge-independent.  Mean split + restored.

        ``return_A=True`` also returns the fitted S1 potential coefficients
        (the 3 component vectors) - the THB path has A explicitly, so
        :meth:`helicity` (H = <A, curl A>) is a natural topology diagnostic.

        THE RETURNED A IS ONLY DEFINED UP TO GAUGE.  K is singular on grad S0,
        so which point of the solution manifold CG lands on depends on solver
        details (the Jacobi preconditioner below moves it by ~40% on a graded
        mesh, measured).  Everything physical is unaffected: ``B = curl A``
        matched to 3e-10, and :meth:`helicity` evaluated on the QUADRATURE RULE
        matched to 3e-13 - it really is gauge-invariant.  But helicity SAMPLED
        at scattered points (particle positions) inherits the gauge through its
        own sampling error - 1.5e-3 at 2k points, 9.3e-5 at 60k, i.e. it shrinks
        with the sample, so it is estimator noise, not a leak.  Do not read more
        precision into a sampled H than the sampling supports."""
        from scipy.sparse.linalg import LinearOperator, cg
        B = np.ascontiguousarray(B_at_quad, dtype=np.float64)
        if Bbar is None:
            Bbar = np.average(B, axis=0, weights=self.qw)
        Bbar = np.asarray(Bbar, dtype=np.float64).reshape(3)
        fluc = B - Bbar[None, :]
        counts = [self.S1[c].n_basis for c in range(3)]
        offs = np.cumsum([0] + counts)
        # sparse grad operators G[(c,axis)] = d/d axis of S1[c] at the quad pts.
        # curl uses only the 6 OFF-diagonal (c != axis) ops; the 3 diagonals
        # (dAx/dx, dAy/dy, dAz/dz) never appear, so don't build them.  The two
        # ops of a component come out of ONE descent and SHARE one index array
        # (eval_operators_fast) - the pattern depends on the leaf, not on which
        # derivative is taken.
        if not hasattr(self, "_Gop"):
            self._Gop = {}
            for c in range(3):
                ax1, ax2 = (c + 1) % 3, (c + 2) % 3
                self._Gop[c, ax1], self._Gop[c, ax2] = \
                    self.S1[c].eval_operators_fast(self.qpts, [ax1, ax2],
                                                   dtype=self.op_dtype)
        Gy_Az, Gz_Ay = self._Gop[2, 1], self._Gop[1, 2]
        Gz_Ax, Gx_Az = self._Gop[0, 2], self._Gop[2, 0]
        Gx_Ay, Gy_Ax = self._Gop[1, 0], self._Gop[0, 1]

        def split(a):
            return [a[offs[c]:offs[c + 1]] for c in range(3)]

        def Gmul(a):                     # concatenated a -> curl at quad (Q,3)
            a0, a1, a2 = split(a)
            return np.stack([spmv(Gy_Az, a2) - spmv(Gz_Ay, a1),
                             spmv(Gz_Ax, a0) - spmv(Gx_Az, a2),
                             spmv(Gx_Ay, a1) - spmv(Gy_Ax, a0)], axis=1)

        def Gtmul(r):                    # (Q,3) -> concatenated coeffs
            rx = np.ascontiguousarray(r[:, 0])       # spmv_t wants contiguous
            ry = np.ascontiguousarray(r[:, 1])
            rz = np.ascontiguousarray(r[:, 2])
            return np.concatenate([spmv_t(Gz_Ax, ry) - spmv_t(Gy_Ax, rz),
                                   spmv_t(Gx_Ay, rz) - spmv_t(Gz_Ay, rx),
                                   spmv_t(Gy_Az, rx) - spmv_t(Gx_Az, ry)])

        def Kmul(a):
            r = Gmul(a)
            r *= self.qw[:, None]        # in place: the (Q,3) product is a pure
            return Gtmul(r)              # temporary, no need for a second copy

        n = int(offs[-1])
        K = LinearOperator((n, n), matvec=Kmul)
        fluc *= self.qw[:, None]                     # fluc is already a copy
        f = Gtmul(fluc)
        del fluc
        # JACOBI PRECONDITIONER.  Component c enters the curl through exactly
        # two derivative operators whose contributions land in DIFFERENT output
        # components, so the diagonal has no cross terms:
        #     diag(K)_jj = sum_q w_q [ (Ga)_qj^2 + (Gb)_qj^2 ].
        # It costs one pass over the operators (~0.5s at 48k leaves) and cut
        # that solve from 593 to 145 iterations - 170s -> 42s, MEASURED.  A
        # preconditioner changes the iterate path, not the system: both runs
        # converge to the same rtol and agreed to 3.5e-5 on B (at rtol 1e-6).
        diag = np.concatenate([
            _weighted_col_sq(self._Gop[c, (c + 1) % 3], self.qw)
            + _weighted_col_sq(self._Gop[c, (c + 2) % 3], self.qw)
            for c in range(3)])
        diag[diag <= 0.0] = 1.0            # gauge/unreached dof -> no-op
        P = LinearOperator((n, n), dtype=np.float64,
                           matvec=lambda v, d=diag: v / d)
        a, info = cg(K, f, rtol=rtol, atol=0.0, maxiter=maxiter, M=P)
        A_coeffs = split(a)
        # the quad-point operators are dead now; the readout below builds its
        # own at `positions`, so free ~6 operators' worth before it does.
        self.free_operators()
        B = self.curl_readout_fast(A_coeffs, positions) + Bbar[None, :]
        return (B, A_coeffs) if return_A else B

    def curl_readout_fast(self, coeffs, positions, Bbar=None):
        """B = curl A at ``positions`` using the per-leaf fast grad operators
        (built once, matmul) - faster than :meth:`curl_readout_local` when
        query points cluster in leaves.  Matches it exactly.

        Accumulated ONE COMPONENT AT A TIME: component c enters the curl through
        exactly two derivatives (``B[a] += dA_c/dx_b``, ``B[b] -= dA_c/dx_a`` for
        a=(c+1)%3, b=(c+2)%3), so its 2 operators are built (sharing one index
        array), applied, and dropped before the next component - peak is 2
        operators instead of all 6."""
        pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        B = np.zeros((len(pos), 3))
        for c in range(3):
            a, b = (c + 1) % 3, (c + 2) % 3
            Ga, Gb = self.S1[c].eval_operators_fast(pos, [a, b],
                                                    dtype=self.op_dtype)
            B[:, a] += spmv(Gb, coeffs[c])
            B[:, b] -= spmv(Ga, coeffs[c])
            del Ga, Gb
        if Bbar is not None:
            B = B + np.asarray(Bbar, dtype=np.float64).reshape(1, 3)
        return B

    def eval_A(self, coeffs, positions):
        """The vector potential A (the S1 field of ``coeffs``) at ``positions``
        (level-0 cell units, (M,3)) -> (M,3).  Companion to
        :meth:`curl_readout_fast` (B = curl A); together they give the helicity
        integrand A.B.  One component's operator alive at a time."""
        pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        A = np.empty((len(pos), 3))
        for c in range(3):
            E = self.S1[c].eval_operator_fast(pos, dtype=self.op_dtype)
            A[:, c] = spmv(E, coeffs[c])
            del E
        return A

    def helicity(self, coeffs, positions, volumes, B=None):
        """Magnetic helicity ``H = int A.(curl A) dV ~ sum_a V_a A_a.B_a`` of the
        FITTED fluctuation field (B = curl A, mean-free - Bbar carries no
        helicity here), sampled at ``positions`` with quadrature/particle
        ``volumes``.  Gauge-invariant: curl A is exactly div-free and mean-free
        on the periodic domain, so the gauge shift ``int (grad chi).B = -int chi
        div B = 0``.  A topology-preservation diagnostic; the THB readout is the
        one with A explicit (unlike the RT0 / vertex-B methods).

        Pass ``B`` (the MEAN-FREE ``curl A`` at the same ``positions``) if the
        caller already has it - otherwise it is recomputed here, which rebuilds
        all 6 grad operators for nothing."""
        A = self.eval_A(coeffs, positions)
        if B is None:
            B = self.curl_readout_fast(coeffs, positions)  # mean-free curl A
        return float(np.sum(np.asarray(volumes, dtype=np.float64)
                            * np.sum(A * np.asarray(B, dtype=np.float64),
                                     axis=1)))

    def curl_readout_local(self, coeffs, positions, Bbar=None):
        """SCALABLE ``B = curl A`` at ``positions`` via localized THB
        derivatives (no dense finest expansion).  Matches :meth:`curl_readout`.
        curl A = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy)."""
        Ax, Ay, Az = coeffs
        pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        Bx = self.S1[2].eval_grad_local(Az, pos, 1) \
            - self.S1[1].eval_grad_local(Ay, pos, 2)
        By = self.S1[0].eval_grad_local(Ax, pos, 2) \
            - self.S1[2].eval_grad_local(Az, pos, 0)
        Bz = self.S1[1].eval_grad_local(Ay, pos, 0) \
            - self.S1[0].eval_grad_local(Ax, pos, 1)
        B = np.stack([Bx, By, Bz], axis=1)
        if Bbar is not None:
            B = B + np.asarray(Bbar, dtype=np.float64).reshape(1, 3)
        return B

    # -- TABLE cell-field RHS (Petkova-leaf values) -----------------------
    def _quad_cell_index(self):
        """Flat finest-cell index of each quad point (mesh-only, cached).
        With per-cell Gauss quadrature, sampling a piecewise-constant cell
        field at the quad points yields exactly ``b_i = sum_cell v_cell
        int_cell beta_i`` - the table cell-integral RHS."""
        if not hasattr(self, "_qci"):
            idx = np.mod(np.floor(self.qpts / self.hf).astype(np.int64),
                         np.asarray(self.dimsL))
            self._qci = ((idx[:, 0] * self.dimsL[1] + idx[:, 1])
                         * self.dimsL[2] + idx[:, 2])
        return self._qci

    def cells_to_quad(self, cell_field):
        """Sample a FINEST-grid cell field (nx,ny,nz[,k]) at the quad points
        (piecewise-constant).  An AMR leaf at level l is just its value
        broadcast to the 2^(3(L-l)) finest cells it covers."""
        cf = np.ascontiguousarray(cell_field, dtype=np.float64)
        extra = cf.shape[3:]
        return cf.reshape((-1,) + extra)[self._quad_cell_index()]

    def project_A_cells(self, cell_A):
        """VARIANT 1 from a table: project a finest-grid cell A into S1."""
        return self.project_A(self.cells_to_quad(cell_A))

    def bfit_cells(self, cell_B, positions, Bbar=None):
        """VARIANT 2 from a table: fit A so curl A matches a finest-grid
        cell B, read B_h at positions."""
        return self.bfit(self.cells_to_quad(cell_B), positions, Bbar=Bbar)

    # -- VARIANT 2: curl-fit (smooth Hodge of B) --------------------------
    def _zero_coeffs(self):
        return [np.zeros(self.S1[c].n_basis) for c in range(3)]

    def _build_curlfit(self):
        """Assemble the mesh-only curl-least-squares operator: column (c,j) is
        `curl(psi_{c,j})` sampled at the quad points (component-major flatten),
        weighted by sqrt(quad weight).  Cached; only the RHS changes per B."""
        if hasattr(self, "_pinv"):
            return
        counts = [self.S1[c].n_basis for c in range(3)]
        self._offsets = np.cumsum([0] + counts)
        ntot = int(self._offsets[-1])
        Q = len(self.qpts)
        G = np.zeros((3 * Q, ntot))
        col = 0
        for c in range(3):
            for j in range(counts[c]):
                coeffs = self._zero_coeffs()
                coeffs[c][j] = 1.0
                G[:, col] = self.curl_readout(coeffs, self.qpts).T.ravel()
                col += 1
        self._G = G
        self._sw = np.sqrt(np.tile(self.qw, 3))              # (3Q,)
        # weighted least-squares min ||sqrt(W)(G a - Bfluc)||; pinv is mesh-only.
        # K = G^T W G is singular (gauge = grad S0); the pseudo-inverse takes
        # the minimum-norm A, and B_h = curl A is gauge-independent anyway.
        # rcond MUST cut the gauge/near-null modes: they sit ~1e-9*sigma_max or
        # below while the real curl modes are ~0.2*sigma_max (a wide gap), so
        # the default 1e-15 keeps them and blows the solution norm up to ~1e13
        # (a 1/sigma amplification of noise) - measured.  1e-7 cleans it.
        self._pinv = np.linalg.pinv(self._sw[:, None] * G, rcond=1e-7)

    def bfit(self, B_at_quad, positions, Bbar=None):
        """Fit the spline A so ``curl A`` best matches the target B (L2), then
        read ``B_h = curl A + Bbar``.  ``B_at_quad`` is (Q,3) at ``self.qpts``.

        The mean is split off first: ``curl A`` is mean-free on a periodic
        domain (Bbar is L2-orthogonal to the curl range), so Bbar cannot be
        fit and MUST be restored analytically.  ``Bbar=None`` uses the
        quadrature-weighted mean of B."""
        self._build_curlfit()
        B = np.ascontiguousarray(B_at_quad, dtype=np.float64)
        if Bbar is None:
            Bbar = np.average(B, axis=0, weights=self.qw)
        Bbar = np.asarray(Bbar, dtype=np.float64).reshape(3)
        fluc = (B - Bbar[None, :]).T.ravel()                 # component-major
        a_flat = self._pinv @ (self._sw * fluc)
        coeffs = [a_flat[self._offsets[c]:self._offsets[c + 1]]
                  for c in range(3)]
        return self.curl_readout(coeffs, positions) + Bbar[None, :]
