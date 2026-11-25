"""Truncated hierarchical B-splines (1D core): partition of unity + selection."""

import numpy as np
import pytest

from glassgen import reconstruct

try:
    reconstruct._import_sph_interp()
    import sph_interp.thb as thb
    HAVE_SPH_INTERP = True
except ImportError:
    thb = None
    HAVE_SPH_INTERP = False

needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason="sph_interp not found in testsuite layout")


@needs_sph_interp
def test_subdivision_mask_sums_to_two():
    # two-scale mask: sum a_k = 2 (a B-spline refines into shifted copies)
    for p in (1, 2, 3):
        assert np.isclose(thb.subdivision_mask(p).sum(), 2.0)


@needs_sph_interp
def test_no_refinement_is_uniform_space():
    # with no refinement THB reduces to the uniform periodic B-spline space
    t = thb.THB1D(n0=8, degree=3, refined=[])
    assert t.n_basis == 8
    x = np.linspace(0, 8, 200, endpoint=False)
    pou = t.eval_matrix(x).sum(axis=0)
    np.testing.assert_allclose(pou, 1.0, atol=1e-12)


@needs_sph_interp
@pytest.mark.parametrize("degree", [2, 3])
def test_partition_of_unity_local_refinement(degree):
    # a local refined patch: THB must still be a partition of unity everywhere
    n0 = 8
    ref0 = np.zeros(n0, dtype=bool)
    ref0[3:6] = True                       # refine a few interior cells
    t = thb.THB1D(n0=n0, degree=degree, refined=[ref0])
    assert t.n_basis > n0                  # extra fine functions were activated
    x = np.linspace(0, n0, 500, endpoint=False)
    pou = t.eval_matrix(x).sum(axis=0)
    np.testing.assert_allclose(pou, 1.0, atol=1e-11)


@needs_sph_interp
def test_partition_of_unity_two_levels():
    # two nested refinement levels (2:1-graded), PoU still holds
    n0 = 12
    ref0 = np.zeros(n0, dtype=bool)
    ref0[4:9] = True
    ref1 = np.zeros(2 * n0, dtype=bool)
    ref1[10:16] = True                     # refine inside the level-0 patch
    t = thb.THB1D(n0=n0, degree=3, refined=[ref0, ref1])
    assert len(t.active[0]) and len(t.active[1]) and len(t.active[2])
    x = np.linspace(0, n0, 800, endpoint=False)
    pou = t.eval_matrix(x).sum(axis=0)
    np.testing.assert_allclose(pou, 1.0, atol=1e-10)


@needs_sph_interp
def test_thb3d_no_refinement_partition_of_unity():
    t = thb.THB3D(base_dims=(4, 4, 4), degrees=(3, 3, 3), refined=[])
    assert t.n_basis == 4 * 4 * 4
    q = np.random.default_rng(0).uniform(0, 4, size=(60, 3))
    pou = t.eval_matrix(q).sum(axis=0)
    np.testing.assert_allclose(pou, 1.0, atol=1e-11)


@needs_sph_interp
@pytest.mark.parametrize("degrees", [(3, 3, 3), (2, 3, 3), (3, 2, 3)])
def test_thb3d_local_refinement_partition_of_unity(degrees):
    # a refined interior box; per-axis (degree-dropped) spaces exercise the
    # De Rham components. THB partition of unity must hold everywhere.
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    t = thb.THB3D(base_dims=base, degrees=degrees, refined=[ref0])
    assert t.n_basis > 4 * 4 * 4
    q = np.random.default_rng(1).uniform(0, 4, size=(300, 3))
    pou = t.eval_matrix(q).sum(axis=0)
    np.testing.assert_allclose(pou, 1.0, atol=1e-10)


@needs_sph_interp
def test_thb3d_expand_to_finest_matches_eval():
    # a THB field expanded to the finest uniform level, then evaluated with the
    # uniform de Boor eval, matches the direct THB evaluation
    from sph_interp.spline_derham import _eval_scalar
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    t = thb.THB3D(base_dims=base, degrees=(3, 3, 3), refined=[ref0])
    coeff = np.random.default_rng(2).standard_normal(t.n_basis)
    cL = t.expand_to_finest(coeff)
    q = np.random.default_rng(3).uniform(0, 4, size=(120, 3))
    direct = coeff @ t.eval_matrix(q)
    dimsL = t.dims[t.L]
    fac = 2 ** t.L
    u = [np.mod(q[:, a] * fac, dimsL[a]) for a in range(3)]
    viafinest = _eval_scalar(cL, u, (3, 3, 3), dimsL)
    np.testing.assert_allclose(direct, viafinest, atol=1e-10)


@needs_sph_interp
@pytest.mark.parametrize("degrees", [(3, 3, 3), (2, 3, 3), (3, 2, 3)])
def test_thb3d_local_eval_matches_dense(degrees):
    # the memory-scalable localized subdivide-accumulate reproduces the dense
    # finest-expansion evaluation exactly, on a two-level graded mesh
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    ref1 = np.zeros((8, 8, 8), dtype=bool)
    ref1[3:6, 3:6, 3:6] = True
    t = thb.THB3D(base, degrees, [ref0, ref1])
    rng = np.random.default_rng(1)
    c = rng.standard_normal(t.n_basis)
    q = rng.uniform(0.1, 3.9, (100, 3))
    np.testing.assert_allclose(t.eval_field_local(c, q), c @ t.eval_matrix(q),
                               atol=1e-12)


@needs_sph_interp
@pytest.mark.parametrize("degrees", [(3, 3, 3), (2, 3, 3)])
def test_thb_sparse_matches_dense(degrees):
    # the sparse-octree THB (no dense per-level arrays) reproduces the dense
    # THB3D exactly: same basis ordering, eval, derivative, adjoint
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    ref1 = np.zeros((8, 8, 8), dtype=bool)
    ref1[3:6, 3:6, 3:6] = True
    d = thb.THB3D(base, degrees, [ref0, ref1])
    s = thb.THBSparse(base, degrees, [ref0, ref1])
    assert s.n_basis == d.n_basis and s.basis == d.basis
    rng = np.random.default_rng(5)
    c = rng.standard_normal(d.n_basis)
    q = rng.uniform(0.1, 3.9, (60, 3))
    y = rng.standard_normal(len(q))
    np.testing.assert_allclose(s.eval_field_local(c, q),
                               d.eval_field_local(c, q), atol=1e-12)
    np.testing.assert_allclose(s.eval_grad_local(c, q, 1),
                               d.eval_grad_local(c, q, 1), atol=1e-12)
    np.testing.assert_allclose(s.assemble_local(y, q),
                               d.assemble_local(y, q), atol=1e-12)


@needs_sph_interp
def test_eval_operator_matches_local():
    # the once-built sparse eval/grad operator reproduces the localized
    # eval and derivative (E @ c == eval, E_grad @ c == eval_grad)
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    t = thb.THBSparse(base, (3, 3, 3), [ref0])
    rng = np.random.default_rng(7)
    c = rng.standard_normal(t.n_basis)
    q = rng.uniform(0.2, 3.8, (50, 3))
    np.testing.assert_allclose(t.eval_operator(q) @ c,
                               t.eval_field_local(c, q), atol=1e-12)
    np.testing.assert_allclose(t.eval_operator(q, deriv_axis=1) @ c,
                               t.eval_grad_local(c, q, 1), atol=1e-12)
    # per-leaf fast build must equal the per-point build
    for dax in (None, 1):
        d = (t.eval_operator(q, deriv_axis=dax)
             - t.eval_operator_fast(q, deriv_axis=dax))
        assert np.max(np.abs(d.toarray())) < 1e-12


@needs_sph_interp
def test_thb_sparse_deep_hierarchy():
    # a depth the dense THB3D cannot represent (finest grid ~ 6e4 per axis):
    # THBSparse builds + evaluates with no dense arrays.  PoU stays exact.
    base = (16, 15, 2)
    refs, cur = [], {(8, 7, 1)}
    for _ in range(12):
        refs.append(set(cur))
        cur = {(2 * i, 2 * j, 2 * k) for (i, j, k) in cur}
    s = thb.THBSparse(base, (3, 3, 3), refs)
    assert s.n_basis > 0
    pou = s.eval_field_local(np.ones(s.n_basis),
                             np.array([[8.3, 7.4, 1.2], [0.5, 0.5, 0.5]]))
    np.testing.assert_allclose(pou, 1.0, atol=1e-10)


@needs_sph_interp
def test_amr_to_thbsparse_roundtrip():
    # a real sph_interp AMR octree -> THBSparse hierarchy: the leaf sets must
    # be identical (the octree-wiring bridge to real captures)
    si = reconstruct._import_sph_interp()
    from sph_interp.thb import amr_refined_sets, THBSparse
    rng = np.random.default_rng(0)
    pos = np.concatenate([rng.normal(0, 0.08, (3000, 3)),
                          rng.uniform(-0.5, 0.5, (1500, 3))])
    mass = np.full(len(pos), 1.0 / len(pos))
    P = si.Particles(pos=pos, mass=mass, rho=np.ones(len(pos)),
                     h=np.full(len(pos), 0.05), box=np.ones(3), periodic=True)
    amr = si.AMRGrid.build(P, base_npx=4, max_level=3,
                           criterion=si.refine_count(30))
    base, refined = amr_refined_sets(amr)
    s = THBSparse(base, (3, 3, 3), refined)
    amr_leaves = set()
    for n in range(amr.ncell):
        lv = int(amr.cell_level[n])
        cw = amr.base_pw / 2 ** lv
        idx = tuple(np.round((amr.cell_lo[n] - amr.lo) / cw).astype(int))
        amr_leaves.add((lv, idx))
    assert set((l, tuple(c)) for (l, c) in s.leaves()) == amr_leaves


@needs_sph_interp
def test_thb3d_local_grad_matches_fd():
    # the localized analytic derivative matches finite differences
    base = (4, 4, 4)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:3, 1:3, 1:3] = True
    t = thb.THB3D(base, (3, 3, 3), [ref0])
    rng = np.random.default_rng(2)
    c = rng.standard_normal(t.n_basis)
    q = rng.uniform(0.6, 3.4, (40, 3))
    h = 1e-5
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = h
        fd = (t.eval_field_local(c, q + e) - t.eval_field_local(c, q - e)) / (2 * h)
        an = t.eval_grad_local(c, q, axis)
        assert np.max(np.abs(an - fd)) < 1e-6 * (np.abs(an).max() + 1e-9)


@needs_sph_interp
@pytest.mark.parametrize("cls_name", ["THB3D", "THBSparse"])
def test_operators_match_ground_truth_aliasing(cls_name):
    # A shallow base axis (dims < spline support p+1) makes the periodic
    # covering wrap onto itself, so several stencil offsets alias to ONE
    # coefficient.  The galaxy base is (16, 15, 2) -> the z-axis (2 cells,
    # cubic) aliases.  eval_field_local / eval_matrix are the ground truth;
    # eval_operator (per-point) and eval_operator_fast (per-leaf, numba) must
    # both reproduce it for the value AND every derivative, without
    # double-counting aliased keys.
    cls = getattr(thb, cls_name)
    base, deg = (16, 15, 2), (2, 3, 3)          # an A-component degree triple
    ref0 = np.zeros(base, dtype=bool)
    ref0[7:9, 6:8, 0:1] = True
    t = cls(base, deg, [ref0])
    rng = np.random.default_rng(3)
    c = rng.standard_normal(t.n_basis)
    q = rng.uniform(0.05, 0.95, (60, 3)) * np.array(base)
    ref = t.eval_field_local(c, q)
    assert np.max(np.abs(t.eval_operator(q) @ c - ref)) < 1e-11
    assert np.max(np.abs(t.eval_operator_fast(q) @ c - ref)) < 1e-11
    for ax in range(3):
        gloc = t.eval_grad_local(c, q, ax)
        assert np.max(np.abs(t.eval_operator(q, deriv_axis=ax) @ c - gloc)) < 1e-10
        assert np.max(np.abs(t.eval_operator_fast(q, deriv_axis=ax) @ c - gloc)) < 1e-10


@needs_sph_interp
def test_nesting_enforced():
    # a level-1 refinement outside the level-0 refined region is dropped
    n0 = 8
    ref0 = np.zeros(n0, dtype=bool)
    ref0[2:4] = True
    ref1 = np.ones(2 * n0, dtype=bool)     # ask to refine everywhere at L1
    t = thb.THB1D(n0=n0, degree=3, refined=[ref0, ref1])
    # only children of refined level-0 cells 2,3 -> level-1 cells 4..7 survive
    assert t.refined[1].sum() == 4
    assert t.omega[2].sum() == 8


@needs_sph_interp
@pytest.mark.parametrize("cls_name", ["THB3D", "THBSparse"])
def test_multi_operator_build_shares_pattern_and_matches(cls_name):
    # eval_operators_fast returns K operators from ONE descent, SHARING one
    # index array (the descent and the sparsity pattern depend on the leaf, not
    # on which derivative is taken).  Values must equal the single-op build
    # exactly, and the CSR must be canonical (sorted columns) - the shared
    # arrays are handed to scipy with has_sorted_indices=True, so unsorted
    # columns would silently corrupt every downstream sparse op.
    base = (6, 5, 4)
    deg = (3, 2, 3)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:4, 1:3, :] = True
    cls = getattr(thb, cls_name)
    if cls_name == "THBSparse":
        t = cls(base, deg, [{tuple(int(v) for v in ijk)
                             for ijk in np.argwhere(ref0)}])
    else:
        t = cls(base, deg, [ref0])
    rng = np.random.default_rng(5)
    q = rng.uniform(0.05, 0.95, (200, 3)) * np.array(base)
    axes = [None, 0, 1, 2]
    ops = t.eval_operators_fast(q, axes)
    for k, ax in enumerate(axes):
        single = t.eval_operator_fast(q, ax)
        d = ops[k] - single
        assert (d.nnz == 0 or np.max(np.abs(d.data)) == 0.0), ax
    for k in range(1, len(axes)):                 # one index array for all K
        assert np.shares_memory(ops[k].indices, ops[0].indices)
        assert np.shares_memory(ops[k].indptr, ops[0].indptr)
    for op in ops:                                 # canonical CSR
        assert op.has_sorted_indices
        for r in range(op.shape[0]):
            cols = op.indices[op.indptr[r]:op.indptr[r + 1]]
            assert np.all(np.diff(cols) > 0), "columns not strictly ascending"


@needs_sph_interp
@pytest.mark.parametrize("base,deg,nref",
                         [((6, 5, 4), (3, 2, 3), 1), ((16, 15, 2), (3, 3, 3), 2)])
def test_count_only_descent_matches_full(base, deg, nref):
    # THE load-bearing invariant of the THREADED operator build: it lays its
    # output out from a count_only descent (which skips every weight update)
    # and then writes into those exact slots in parallel.  If count_only ever
    # returned a different nc from the full descent, threads would overrun
    # each other's rows.  Same code path by construction - locked anyway.
    if not thb._HAVE_NUMBA:
        pytest.skip("numba required")
    rng = np.random.default_rng(11)
    refined = []
    for l in range(nref):
        dims = tuple(b * 2 ** l for b in base)
        r = np.zeros(dims, dtype=bool)
        r[tuple(slice(1, max(2, d // 2)) for d in dims)] = True
        refined.append({tuple(int(v) for v in ijk) for ijk in np.argwhere(r)})
    t = thb.THBSparse(base, deg, refined)
    dims_a, masks_a, lev_off, c2c, refined_t = t._numba_tables()
    npatch = int(np.prod([d + 1 for d in deg]))
    deg_a = np.array(deg, dtype=np.int64)
    for (lev, cell) in t.leaves():
        cell_a = np.array(cell, dtype=np.int64)
        full, _ = thb._leaf_transfer_numba(lev, cell_a, deg_a, dims_a, masks_a,
                                           lev_off, c2c, npatch, False)
        cnt, _ = thb._leaf_transfer_numba(lev, cell_a, deg_a, dims_a, masks_a,
                                          lev_off, c2c, npatch, True)
        assert cnt.shape[0] == full.shape[0], (lev, cell)
        np.testing.assert_array_equal(cnt, full)


@needs_sph_interp
def test_operator_build_is_thread_count_invariant():
    # The build is threaded over leaves; the work split must be invisible in
    # the output (rows are laid out per POINT, so no chunking can be observed).
    # Byte-identical, not just close - a race would show up here first.
    numba = pytest.importorskip("numba")
    base, deg = (6, 5, 4), (3, 2, 3)
    ref0 = np.zeros(base, dtype=bool)
    ref0[1:4, 1:3, :] = True
    t = thb.THBSparse(base, deg, [{tuple(int(v) for v in ijk)
                                   for ijk in np.argwhere(ref0)}])
    rng = np.random.default_rng(13)
    q = rng.uniform(0.05, 0.95, (4000, 3)) * np.array(base)
    n0 = numba.get_num_threads()
    try:
        ref = t.eval_operators_fast(q, [None, 0, 2], dtype=np.float32)
        for nt in (1, 2, max(1, n0)):
            numba.set_num_threads(nt)
            got = t.eval_operators_fast(q, [None, 0, 2], dtype=np.float32)
            for k in range(3):
                np.testing.assert_array_equal(got[k].data, ref[k].data)
                np.testing.assert_array_equal(got[k].indices, ref[k].indices)
                np.testing.assert_array_equal(got[k].indptr, ref[k].indptr)
    finally:
        numba.set_num_threads(n0)
