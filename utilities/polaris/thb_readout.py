"""THB De Rham curl-fit readout of a cell-centred magnetic field -- M3b of
POLARIS_EXPORT_PLAN.md.

One entry point, `thb_curl_readout`, which takes an `AMRGrid` plus the per-leaf
B from an ordinary deposit and returns a **pointwise divergence-free** B at
arbitrary positions, by fitting a THB-spline vector potential A so that
`curl A` best matches the deposited field in L2, then evaluating `curl A` where
asked. `div B = 0` then holds exactly and pointwise, not just in a discrete
sense, because the C2 spline basis makes it an identity.

This wraps the VARIANT 2 (`bfit_matfree`) path validated in
`icgenerator/glassgen_python/recon_m8/magnetic_galaxy_amr_thb.py`; the plumbing
here is the leaf-order bookkeeping that script keeps privately, lifted so both
the POLARIS export and the generic grid CLI can call one implementation.

Two things worth knowing:

* **`Bbar` is handled automatically for the curl-fit.** `curl A` is mean-free,
  so the ordered mean field is L2-orthogonal to the fit and must be restored
  analytically -- `bfit_matfree` does that itself from the quadrature-weighted
  mean when `Bbar=None`. (The plan's "`Bbar` must be passed" warning is about
  the A-PROJECTION variant, which cannot recover it; that run had
  `aproj_bbar = none` and lost the ordered field. Curl-fit is safe by
  construction, and this module only ever uses curl-fit.)
* **A divergence-free field is not optional for POLARIS.** With `div B != 0` the
  polarized-RT solver stalls -- 312 "too many steps" escalations, 73 skipped
  cells, no FITS after 10 minutes on a grid that takes 23 s with a projected
  field. See the plan's §8b.

Memory is the real constraint: the sparse operators are (Q x nnz) and peaked at
~15 GB on the 405k-leaf full-resolution galaxy. `op_dtype=float32` halves that
and costs ~1e-7 in the readout, which is far below the capture error.
"""

import time

import numpy as np


def _thb_space(amr, degree=3, nq=2, op_dtype=np.float32):
    """The THB De Rham space on an AMR hierarchy."""
    from sph_interp.thb import amr_refined_sets
    import sph_interp.thb_derham as td

    base, refined = amr_refined_sets(amr)
    return td.THBDeRham(base, degree=degree, refined=refined, nq=nq,
                        sparse=True, op_dtype=op_dtype)


def _leaf_cell_index(R, amr):
    """AMR leaf index for every THB leaf, in `R.S1[0].leaves()` order.

    The two structures describe the same octree but enumerate it differently, so
    a field indexed by AMR leaf has to be permuted before it can be sampled at
    the THB quadrature points. Keyed on (level, integer cell index), which is
    exact -- no float comparison of corners.
    """
    lo = np.asarray(amr.lo, dtype=np.float64)
    bpw = np.asarray(amr.base_pw, dtype=np.float64)
    key_of = {}
    for n in range(amr.ncell):
        lv = int(amr.cell_level[n])
        cw = bpw / 2.0 ** lv
        idx = tuple(np.round((amr.cell_lo[n] - lo) / cw).astype(int))
        key_of[(lv, idx)] = n
    try:
        return np.array([key_of[(lv, tuple(cell))]
                         for lv, cell in R.S1[0].leaves()], dtype=np.int64)
    except KeyError as e:                      # pragma: no cover - structural
        raise RuntimeError(
            "THB leaf set does not match the AMR leaf set -- the hierarchy was "
            f"rebuilt or is not dyadic (missing {e.args[0]})") from None


def _leaf_field_at_quad(R, leaf_field, cell_of_leaf):
    """Per-leaf vector field -> the per-leaf Gauss quadrature points.

    With per-cell Gauss quadrature, sampling a piecewise-constant leaf field at
    the quad points IS the cell-integral right-hand side, so no extra weighting
    is needed.
    """
    nq3 = R.nq ** 3
    return np.column_stack([
        np.repeat(np.asarray(leaf_field[i], dtype=np.float64)[cell_of_leaf], nq3)
        for i in range(3)])


def thb_curl_readout(amr, B_leaf, positions=None, degree=3, nq=2,
                     op_dtype=np.float32, rtol=1e-6, maxiter=1000,
                     verbose=True, audit=True):
    """Divergence-free B at `positions` from a per-leaf deposited B.

    Parameters
    ----------
    amr : AMRGrid                    the hierarchy the field was deposited on
    B_leaf : (3,) sequence of (Ncell,) arrays, or an (Ncell,3) array
    positions : (M,3) or None        where to read B out; default `amr.centres()`,
                                     which is POLARIS's convention for ids 4,5,6
    rtol : float                     CG tolerance of the curl-fit

    Returns
    -------
    (B, info) with B of shape (M,3) and `info` carrying the fit diagnostics:
    `Bbar` (the restored ordered mean), `rel_l2` against the deposited field at
    the same points, `energy_over_source`, `helicity`, and timings.
    """
    B_leaf = np.asarray(B_leaf, dtype=np.float64)
    if B_leaf.ndim == 2 and B_leaf.shape[0] == amr.ncell and B_leaf.shape[1] == 3:
        B_leaf = B_leaf.T
    if B_leaf.shape != (3, amr.ncell):
        raise ValueError(f"B_leaf must be (3,{amr.ncell}) or ({amr.ncell},3), "
                         f"got {np.shape(B_leaf)}")
    if positions is None:
        positions = amr.centres()
    positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)

    t0 = time.time()
    R = _thb_space(amr, degree=degree, nq=nq, op_dtype=np.dtype(op_dtype))
    col = _leaf_cell_index(R, amr)
    bpw = np.asarray(amr.base_pw, dtype=np.float64)
    lo = np.asarray(amr.lo, dtype=np.float64)
    # the THB space works in level-0-cell units, not physical units
    pos_cell = (positions - lo) / bpw
    if verbose:
        print(f"  THB space: {R.S1[0].n_basis:,} S1x basis fns, "
              f"Q={len(R.qpts):,} quad pts  [{time.time() - t0:.0f}s]",
              flush=True)

    Bq = _leaf_field_at_quad(R, B_leaf, col)
    t1 = time.time()
    B, A = R.bfit_matfree(Bq, pos_cell, rtol=rtol, maxiter=maxiter,
                          return_A=True)
    fit_s = time.time() - t1

    info = {"method": "thb-curlfit", "rtol": rtol,
            "n_basis": int(R.S1[0].n_basis), "n_quad": int(len(R.qpts)),
            "space_s": round(t1 - t0, 1), "fit_s": round(fit_s, 1)}
    # Bbar is what bfit_matfree restored: the quadrature-weighted mean of the
    # input. Recording it makes a lost ordered field obvious in the sidecar.
    info["Bbar"] = np.average(Bq, axis=0, weights=R.qw).tolist()

    if audit:
        # NOTE THE BASELINE. Everything here is measured against the DEPOSITED
        # leaf field, i.e. it isolates what the readout does. It is NOT
        # comparable to the plan's `rel_l2_B` / `energy_over_source`, which are
        # measured against the SPH PARTICLE field and therefore also carry the
        # capture/volume-filling loss of the deposit itself. Mixing the two would
        # make the readout look ~10x better than it is. Hence the `_vs_deposit`
        # suffix on every key.
        vol = np.asarray(amr.volumes(), dtype=np.float64)
        src = B_leaf.T
        if positions.shape[0] == amr.ncell:
            num = float(np.sum(vol * np.sum((B - src) ** 2, axis=1)))
            den = float(np.sum(vol * np.sum(src ** 2, axis=1)))
            info["rel_l2_vs_deposit"] = (float(np.sqrt(num / den)) if den > 0
                                         else float("nan"))
            e_new = float(np.sum(vol * np.sum(B ** 2, axis=1)))
            info["energy_over_deposit"] = e_new / den if den > 0 else float("nan")
            for c, nm in enumerate("xyz"):
                s = float(np.sum(vol * src[:, c]))
                if abs(s) > 0:
                    info[f"mean_B{nm}_over_deposit"] = float(
                        np.sum(vol * B[:, c]) / s)
        try:
            info["helicity"] = float(np.mean(bpw)
                                     * R.helicity(A, pos_cell, vol))
        except Exception:                       # pragma: no cover - diagnostic
            pass
    R.free_operators()
    if verbose:
        bits = [f"fit {fit_s:.0f}s"]
        for k in ("rel_l2_vs_deposit", "energy_over_deposit", "helicity"):
            if k in info:
                bits.append(f"{k}={info[k]:.4g}")
        print("  THB curl-fit: " + ", ".join(bits), flush=True)
    return B, info
