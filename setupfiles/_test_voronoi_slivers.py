"""Regression guard for the sub-h sliver mesh-build fix (Voronoi Petkova).

On a *collapsed*, high-dynamic-range snapshot (mhdcollapse at t=0.00800, rho up to
~1e8) the densest generators sit at separations << their local spacing.  pyvoro's
voro++ cannot tessellate those near-coincident generators cleanly: the bisecting
faces are ill-conditioned and the stored polyhedra develop local GAPS and OVERLAPS
(they stop partitioning space), which leaks ~+0.2% mass from the otherwise-exact
Petkova deposit.  The fix (`VoronoiGrid.from_points(merge_tol=...)`, default a
BACKEND-AWARE fraction of the median nearest-neighbour spacing -- 5% for pyvoro,
20% for multivoro) merges those coincident generators before tessellating.  The
parallel multivoro backend fails far harder (under-cut/overlapping core cells,
~+2260% / 100% overlap) and needs the larger floor; this test covers both.

This test reproduces the bug AND verifies the fix in one run:

  1. INTEGRATOR CONTROL: UniformGrid + AMRGrid Petkova on the SAME particles
     conserve to ~0%.  The signed-tet integrator is shared by all three geometries
     and an axis-aligned mesh cannot have a broken partition, so this proves the
     integral is sound and any Voronoi error is mesh-specific.

  2. BUG, FIX OFF (`merge_tol=0`): the Voronoi Petkova deposit gains ~+0.2% mass,
     and a Monte-Carlo point-in-polyhedron probe around the smallest cell finds a
     measurably broken core partition (points in 0 cells = GAPS, >1 cell =
     OVERLAPS) -> the cause is the mesh, not the integral.

  3. FIX ON (default `merge_tol`): the de-slivered mesh conserves to <0.05% AND the
     MC probe finds a clean partition (~0% gaps/overlaps).

Run from setupfiles/:  python _test_voronoi_slivers.py   (~75s; exits nonzero on
any regression -- control breaks, bug stops reproducing, or fix stops working.)
"""
import numpy as np, time, sys
import numba
from sklearn.neighbors import BallTree
from sph_interp import (from_tipsy, interpolate, UniformGrid, AMRGrid,
                        VoronoiGrid, refine_smoothing_length)
from sph_interp.targets import _HAS_MULTIVORO

SNAP = ('../test_cases/mhdcollapse/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
        'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')
NVORO = 40000           # generators (subsampled particles), matches interp_demo
CONS_TOL = 5e-4         # 0.05% conservation tolerance (the guard)
SEED = 0

failures = []           # real regressions -> nonzero exit
def check(cond, msg):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        failures.append(msg)
    return ok


@numba.njit(cache=True)
def _count_containing(px, py, pz, cand, gen, cf_start, fv_start,
                      fvx, fvy, fvz, eps):
    """For each sample point, count how many of its candidate cells' polyhedra
    contain it (convex cell -> point is interior side of every face plane,
    oriented by the generator). 1 = clean, 0 = gap, >1 = overlap.  Face normals
    are unit-normalized so `eps` is a true signed-distance slack (a point up to
    `eps` outside a face still counts as inside, so roundoff on shared faces does
    not manufacture false gaps)."""
    npts = px.shape[0]
    K = cand.shape[1]
    out = np.zeros(npts, dtype=np.int64)
    for i in range(npts):
        x = px[i]; y = py[i]; z = pz[i]
        for k in range(K):
            c = cand[i, k]
            if c < 0:
                continue
            gx = gen[c, 0]; gy = gen[c, 1]; gz = gen[c, 2]
            inside = True
            for f in range(cf_start[c], cf_start[c + 1]):
                v0 = fv_start[f]
                nv = fv_start[f + 1] - v0
                if nv < 3:
                    continue
                ax = fvx[v0]; ay = fvy[v0]; az = fvz[v0]
                bx = fvx[v0 + 1]; by = fvy[v0 + 1]; bz = fvz[v0 + 1]
                cx = fvx[v0 + 2]; cy = fvy[v0 + 2]; cz = fvz[v0 + 2]
                # face normal = (b-a) x (c-a)
                e1x = bx - ax; e1y = by - ay; e1z = bz - az
                e2x = cx - ax; e2y = cy - ay; e2z = cz - az
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x
                nmag = (nx * nx + ny * ny + nz * nz) ** 0.5
                if nmag <= 0.0:
                    continue                    # degenerate triangle
                nx /= nmag; ny /= nmag; nz /= nmag
                # orient so the generator is on the +side
                sg = nx * (gx - ax) + ny * (gy - ay) + nz * (gz - az)
                sp = nx * (x - ax) + ny * (y - ay) + nz * (z - az)
                if sg < 0.0:
                    sp = -sp
                if sp < -eps:
                    inside = False
                    break
            if inside:
                out[i] += 1
    return out


def main():
    print(f"Loading {SNAP.split('/')[-1]}", flush=True)
    p = from_tipsy(SNAP, fields=("rho",), periodic=True)
    print(f"  N={p.n}, box={np.round(p.box, 4)}, periodic={p.periodic}, "
          f"rho in [{p.rho.min():.3g},{p.rho.max():.3g}], "
          f"mass total={p.mass.sum():.6g}", flush=True)
    Mtot = p.mass.sum()

    half = 0.5 * np.asarray(p.box)
    vper = p.periodic if any(np.atleast_1d(p.periodic)) else False
    bounds = [[-half[a], half[a]] for a in range(3)]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(p.n, min(NVORO, p.n), replace=False)
    gens = np.ascontiguousarray(p.pos[idx])

    def build_and_probe(merge_tol, label, backend="auto"):
        """Tessellate the same generators with a given de-sliver merge_tol, run
        the Petkova deposit (mass error) AND the MC point-in-polyhedron partition
        probe around the smallest cell. Returns (err, f_gap, f_over, ncell)."""
        t0 = time.time()
        vg = VoronoiGrid.from_points(gens, bounds=bounds, periodic=vper,
                                     merge_tol=merge_tol, backend=backend)
        res = interpolate(p, vg, method="petkova", values=["rho"])
        err = res.data["_mass"].sum() / Mtot - 1.0
        # MC partition probe: the collapse puts slivers at the SMALLEST-cell
        # scale, not the global spacing, so probe a box around the tiniest cell.
        vol = vg.volumes()
        csmall = vg.gen[:vg.ncell][np.argmin(vol)]
        local = vol.min() ** (1.0 / 3.0)
        hw = 6.0 * local
        smp = csmall + np.random.default_rng(1).uniform(-hw, hw, size=(40000, 3))
        cand = np.ascontiguousarray(BallTree(vg.gen).query(
            smp, k=min(80, vg.ntotal), return_distance=False).astype(np.int64))
        cnt = _count_containing(
            np.ascontiguousarray(smp[:, 0]), np.ascontiguousarray(smp[:, 1]),
            np.ascontiguousarray(smp[:, 2]), cand, vg.gen,
            vg.cf_start, vg.fv_start, vg.fvx, vg.fvy, vg.fvz, 1e-6 * local)
        f_gap = float(np.mean(cnt == 0)); f_over = float(np.mean(cnt > 1))
        print(f"  {label}: {vg.ncell} cells, petkova err {100*err:+.3f}%, "
              f"partition gaps {100*f_gap:.2f}% / overlaps {100*f_over:.2f}% "
              f"[{time.time()-t0:.0f}s]", flush=True)
        return err, f_gap, f_over, vg.ncell

    # --- 1. integrator control: same particles, axis-aligned meshes --------
    # UniformGrid + AMRGrid Petkova share the signed-tet integrator with Voronoi
    # but cannot have a broken partition -> they MUST conserve, proving the
    # integral is sound and any Voronoi error is mesh-specific.
    print("\nIntegrator control (axis-aligned meshes, same particles):", flush=True)
    err_u = interpolate(p, UniformGrid.centered(p.box, 64), method="petkova",
                        values=["rho"]).data["_mass"].sum() / Mtot - 1.0
    err_a = interpolate(p, AMRGrid.build(p, base_npx=16, max_level=3,
                        criterion=refine_smoothing_length()), method="petkova",
                        values=["rho"]).data["_mass"].sum() / Mtot - 1.0
    print(f"  UniformGrid err {100*err_u:+.3f}%   AMRGrid err {100*err_a:+.3f}%",
          flush=True)

    # --- 2+3. per mesher backend: reproduce the bug with the de-sliver fix OFF,
    # then verify the BACKEND-AWARE default merge_tol restores conservation. Both
    # backends share the de-sliver fix but need DIFFERENT floors: pyvoro's slivers
    # leak ~+0.2%; multivoro's parallel solver under-cuts/overlaps core cells far
    # worse (~+2260%, 100% overlap) and needs a ~4x larger floor (see
    # `_DESLIVER_FRAC` / `multivoro-dynamic-range-defect`). The default is
    # backend-aware, so `merge_tol=None` must conserve for EACH available backend.
    backends = ["pyvoro"] + (["multivoro"] if _HAS_MULTIVORO else [])

    print("\nControl: UniformGrid + AMRGrid petkova conservation", flush=True)
    check(abs(err_u) < CONS_TOL,
          f"control: UniformGrid petkova conserves ({100*err_u:+.3f}%)")
    check(abs(err_a) < CONS_TOL,
          f"control: AMRGrid petkova conserves ({100*err_a:+.3f}%)")

    for backend in backends:
        print(f"\n=== mesher backend: {backend} ===", flush=True)
        print("Voronoi, de-sliver fix OFF (merge_tol=0) -- reproduce the bug:",
              flush=True)
        err0, gap0, over0, nc0 = build_and_probe(0.0, "merge_tol=0", backend)
        print("Voronoi, de-sliver fix ON (default backend-aware merge_tol):",
              flush=True)
        err1, gap1, over1, nc1 = build_and_probe(None, "default", backend)

        # the bug must still be reproducible with the fix off (mesh, not integral)
        check(abs(err0) > 2 * CONS_TOL and (gap0 + over0) > 1e-2,
              f"[{backend}] fix OFF reproduces the bug: err {100*err0:+.3f}% with a "
              f"broken core partition (gaps+overlaps {100*(gap0+over0):.2f}%) -> MESH")
        # the default fix must restore conservation AND a clean partition
        check(abs(err1) < CONS_TOL,
              f"[{backend}] default merge_tol restores conservation "
              f"({100*err1:+.3f}% < {100*CONS_TOL:.2f}%)")
        check((gap1 + over1) < 5e-3,
              f"[{backend}] default merge_tol restores a clean core partition "
              f"(gaps+overlaps {100*(gap1+over1):.2f}% < 0.5%)")
        print(f"  (fix merged {nc0 - nc1} coincident core generators: "
              f"{nc0} -> {nc1} cells)", flush=True)

    print("\nResults:", flush=True)
    if failures:
        print(f"\nFAILED ({len(failures)}):", flush=True)
        for m in failures:
            print(f"  - {m}", flush=True)
        sys.exit(1)
    print("\nOK -- de-sliver fix reproduced, verified, and the integrator "
          "control holds.", flush=True)


if __name__ == "__main__":
    main()
