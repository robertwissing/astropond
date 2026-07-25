"""Threaded sparse matvec kernels (sph_interp._spmv) vs scipy.

The THB solves are ~2/3 CG and CG is ~all matvec, so these kernels replace
scipy's single-threaded (and, for float32 data + float64 vectors,
silently-upcasting) products.  Neither is bit-identical to scipy - the forward
one differs by FMA contraction, the transpose by a blocked reassociation - so
what is locked here is that both stay at the rounding level and that the
transpose is reproducible whatever the thread count.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from glassgen import reconstruct

try:
    reconstruct._import_sph_interp()
    from sph_interp import _spmv
    HAVE_SPH_INTERP = True
except ImportError:
    _spmv = None
    HAVE_SPH_INTERP = False

needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason="sph_interp not found in testsuite layout")


def _mat(rng, Q=3000, n=400, per_row=18, dtype=np.float64):
    rows = np.repeat(np.arange(Q), per_row)
    cols = rng.integers(0, n, size=Q * per_row)
    vals = rng.standard_normal(Q * per_row)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(Q, n)).astype(dtype)
    A.sum_duplicates()
    return A


@needs_sph_interp
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spmv_matches_scipy(dtype):
    rng = np.random.default_rng(0)
    A = _mat(rng, dtype=dtype)
    x = rng.standard_normal(A.shape[1])
    y = rng.standard_normal(A.shape[0])
    fwd, ref = _spmv.spmv(A, x), A @ x
    assert fwd.dtype == np.float64          # f32 data never truncates the sum
    np.testing.assert_allclose(fwd, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
    tr, reft = _spmv.spmv_t(A, y), A.T @ y
    np.testing.assert_allclose(tr, reft, rtol=0,
                               atol=1e-12 * np.abs(reft).max())
    # the Jacobi diagonal helper, against an independent dense reduction
    w = np.abs(rng.standard_normal(A.shape[0]))
    got = _spmv.col_sq(A, w)
    if got is not None:
        D = np.asarray(A.todense(), dtype=np.float64)
        np.testing.assert_allclose(got, (w[:, None] * D ** 2).sum(axis=0),
                                   rtol=1e-12)


@needs_sph_interp
def test_spmv_is_thread_count_invariant():
    # spmv_t accumulates into per-block buffers; the block count is fixed by
    # the MATRIX (not by numba's thread count) precisely so that the answer is
    # reproducible on any machine.  Same bytes at 1, 2 and N threads.
    numba = pytest.importorskip("numba")
    rng = np.random.default_rng(1)
    A = _mat(rng, Q=20000, n=900)
    y = rng.standard_normal(A.shape[0])
    x = rng.standard_normal(A.shape[1])
    n0 = numba.get_num_threads()
    try:
        ref_t, ref_f = _spmv.spmv_t(A, y), _spmv.spmv(A, x)
        for nt in (1, 2, max(1, n0)):
            numba.set_num_threads(nt)
            np.testing.assert_array_equal(_spmv.spmv_t(A, y), ref_t)
            np.testing.assert_array_equal(_spmv.spmv(A, x), ref_f)
    finally:
        numba.set_num_threads(n0)


@needs_sph_interp
def test_spmv_falls_back_for_non_csr():
    # a CSC (or anything else) must go through scipy untouched, not silently
    # be read as if it were CSR
    rng = np.random.default_rng(2)
    A = _mat(rng, Q=500, n=80)
    x = rng.standard_normal(A.shape[1])
    np.testing.assert_allclose(_spmv.spmv(A.tocsc(), x), A @ x)
    assert _spmv.col_sq(A.tocsc(), np.ones(A.shape[0])) is None
