"""Threaded sparse matrix-vector products for the THB solves.

Why this module exists.  Profiling the THB De Rham pass showed the CG solves
are 84-89% of the runtime and the operator build only 5-16% - and inside CG
essentially all of it is ``E @ v`` / ``E.T @ y`` on the quadrature operators.
scipy's CSR/CSC matvec is single-threaded, and worse, when the operator data
is float32 (``op_dtype``, the memory default on deep grids) and the CG vectors
are float64, scipy UPCASTS the whole ``data`` array to float64 on EVERY call -
a full nnz-sized temporary per matvec, and a measured 2.2x slowdown (19.2M
nnz: 23.6 ms mixed vs 10.5 ms matched).  These kernels read float32 data in
place and accumulate in float64, so mixed precision costs nothing and the
loops run over all cores.

PRECISION.  Both kernels accumulate in float64 regardless of the stored dtype
(a float32 -> float64 widening is exact, so no precision is lost relative to
scipy's upcast-then-multiply).  Neither is bit-identical to scipy, and both
differences were measured rather than assumed:

  * :func:`spmv` (``A @ x``) sums each row sequentially in stored order, just
    as scipy does, and rows are independent so threading cannot change a sum.
    It still disagrees with scipy in the LAST ULP (max |diff| 4.4e-16 on the
    19.2M-nnz L3 operator) because LLVM contracts ``acc += a*x`` into an FMA,
    which does not round the product.  Checked against exact rational
    arithmetic on the worst-disagreeing rows: with float32 data the FMA result
    is closer to exact in 10 of 12, with float64 data it is a coin flip
    (5 of 12) - i.e. this is <=1 ulp of rounding, never a loss.
  * :func:`spmv_t` (``A.T @ y``) is a scatter, so a threaded version cannot
    reproduce scipy's global sequential order.  It splits the ROWS into a
    fixed number of blocks (derived from the matrix shape alone - NOT from the
    thread count, so the result is reproducible on any machine and across
    runs), accumulates each block into its own buffer, and reduces the buffers
    in block order.  That is a reassociation of a float64 sum: measured max
    relative deviation from scipy is 6e-15 (float32 data) / 8e-15 (float64) on
    the 19.2M-nnz L3 operator, and blocked summation is if
    anything slightly more accurate than one long sequential accumulation.

Both kernels are deterministic: the same call gives byte-identical output on
every run and at any thread count.  The solves they feed converge to a
residual tolerance, and the tightest assertions in the suite are at 1e-12.

Set ``SPH_INTERP_SPMV=0`` in the environment to fall back to scipy everywhere
(the reference path for cross-checks).
"""

import os

import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                      # pragma: no cover
    _HAVE_NUMBA = False

_ENABLED = os.environ.get("SPH_INTERP_SPMV", "1") not in ("0", "false", "no")

# Row-block count for the transpose scatter.  Fixed by the matrix, not by the
# hardware: reproducible results everywhere.  Buffers are nblk x n_cols
# float64, so the cap keeps that under ~128 MB whatever the basis size.
_MAX_BLOCKS = 32
_MIN_ROWS_PER_BLOCK = 4096
_MAX_BUF_BYTES = 128 << 20


def _n_blocks(nrows, ncols):
    """Deterministic block count for :func:`spmv_t` (matrix shape only)."""
    nblk = min(_MAX_BLOCKS, max(1, nrows // _MIN_ROWS_PER_BLOCK))
    while nblk > 1 and nblk * ncols * 8 > _MAX_BUF_BYTES:
        nblk //= 2
    return nblk


if _HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _csr_matvec(indptr, indices, data, x, out):
        for i in prange(out.shape[0]):
            acc = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                acc += data[k] * x[indices[k]]
            out[i] = acc

    @njit(parallel=True, cache=True)
    def _csr_rmatvec(indptr, indices, data, y, out, bounds, nblk):
        """``out = A.T @ y`` by row blocks, each into its own buffer."""
        n = out.shape[0]
        buf = np.empty((nblk, n))
        for b in prange(nblk):
            for j in range(n):                    # zeroed by the owning thread
                buf[b, j] = 0.0
            for i in range(bounds[b], bounds[b + 1]):
                yi = y[i]
                for k in range(indptr[i], indptr[i + 1]):
                    buf[b, indices[k]] += data[k] * yi
        for j in prange(n):                       # reduce in BLOCK order
            s = 0.0
            for b in range(nblk):
                s += buf[b, j]
            out[j] = s

    @njit(parallel=True, cache=True)
    def _csr_col_sq(indptr, indices, data, w, out, bounds, nblk):
        """``out_j = sum_i w_i A[i,j]^2`` - same blocking as the transpose."""
        n = out.shape[0]
        buf = np.empty((nblk, n))
        for b in prange(nblk):
            for j in range(n):
                buf[b, j] = 0.0
            for i in range(bounds[b], bounds[b + 1]):
                wi = w[i]
                for k in range(indptr[i], indptr[i + 1]):
                    v = np.float64(data[k])
                    buf[b, indices[k]] += wi * v * v
        for j in prange(n):
            s = 0.0
            for b in range(nblk):
                s += buf[b, j]
            out[j] = s


def _usable(A):
    from scipy import sparse
    return (_HAVE_NUMBA and _ENABLED and sparse.isspmatrix_csr(A)
            and A.indptr.shape[0] == A.shape[0] + 1)


def _bounds(nrows, nblk):
    return np.linspace(0, nrows, nblk + 1).astype(np.int64)


def spmv(A, x, out=None):
    """``A @ x`` for a CSR ``A``, threaded and float64-accumulated.

    Row-sequential like scipy, up to FMA contraction (<=1 ulp; see module
    doc), and free of scipy's float32->float64 data copy."""
    if not _usable(A):
        return A @ x
    x = np.ascontiguousarray(x, dtype=np.float64)
    if out is None:
        out = np.empty(A.shape[0], dtype=np.float64)
    _csr_matvec(A.indptr, A.indices, A.data, x, out)
    return out


def spmv_t(A, y, out=None):
    """``A.T @ y`` for a CSR ``A``, threaded and float64-accumulated.

    Deterministic; a blocked reassociation of scipy's sum (see module doc)."""
    if not _usable(A):
        return A.T @ y
    y = np.ascontiguousarray(y, dtype=np.float64)
    if out is None:
        out = np.empty(A.shape[1], dtype=np.float64)
    nblk = _n_blocks(A.shape[0], A.shape[1])
    _csr_rmatvec(A.indptr, A.indices, A.data, y, out,
                 _bounds(A.shape[0], nblk), nblk)
    return out


def col_sq(A, w):
    """``diag_j = sum_i w_i A[i,j]^2`` (the Jacobi diagonal), threaded."""
    if not _usable(A):
        return None
    w = np.ascontiguousarray(w, dtype=np.float64)
    out = np.empty(A.shape[1], dtype=np.float64)
    nblk = _n_blocks(A.shape[0], A.shape[1])
    _csr_col_sq(A.indptr, A.indices, A.data, w, out,
                _bounds(A.shape[0], nblk), nblk)
    return out
