#!/usr/bin/env python
"""Showcase the `sph_interp` interpolation across every target geometry.

Takes ONE tipsy snapshot, deposits the SAME field with the SAME method onto each
target geometry — UniformGrid, AMRGrid, VoronoiGrid, Projection2D — and renders
them side by side in one figure, next to the raw SPH particles, so the
per-geometry deposit can be eyeballed:

  particles | UniformGrid | AMRGrid (octree) | VoronoiGrid | Projection2D

A high density-contrast snapshot (default: mhdcollapse) makes the geometry
differences obvious — the AMR octree refines on the collapsing core and the
Voronoi cells shrink where the particles crowd, both tracking the structure the
uniform grid resolves bluntly.

  interp_demo.sh [SNAPSHOT] [--method sph|sph-scatter|petkova] [--field rho] ...

Each geometry's panel shares a colour scale (the 3D volume grids show a
mid-plane slab slice; Projection2D shows its column map with its own scale).
Use --per-geometry to ALSO write one full SPH-vs-grid diagnostic figure per
geometry (the `--visualise` panel set) for a closer look.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# A high-contrast snapshot, kept as a named convenience rather than a default:
# it is not present in every checkout, and a demo whose no-argument form fails
# on a missing file is a poor demo.
DEFAULT_SNAP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_cases/mhdcollapse/"
    "datafiles/"
    "mhdcollapse32_latticemu10rd360Erat0.04.00000")

# With no snapshot named the demo runs on generated data instead. These three
# are the structured families from `sph_interp.synthetic`, each calibrated
# against a real IC in test_cases/isolated_galaxy: an isolated disc with an
# open boundary, a two-disc merger, and a cosmological web. Between them they
# cover the geometries the targets behave differently on, and they need no
# data files at all.
DEFAULT_SYNTHETIC = ("disk", "merger", "cosmo")
DEFAULT_SYNTHETIC_N = 200000

# The coarse meshes are interpolated with EVERY method so the radial profile can
# contrast them; cheap at coarse resolution (the full-res showcase stays on the
# single --method the user chose). ("sph" is an alias of "sph-gather".)
COARSE_METHODS = ("sph-gather", "sph-scatter", "petkova")


def show_figure(path, enabled):
    """Pop `path` open in an image viewer (non-blocking, survives this process)."""
    import shutil
    import subprocess
    if not enabled:
        return
    for viewer in ("display", "feh", "eog", "xdg-open"):
        if shutil.which(viewer):
            try:
                subprocess.Popen([viewer, path], start_new_session=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                print(f"  displaying {path} ({viewer})")
            except Exception as e:
                print(f"  could not display ({viewer}): {e}")
            return
    print("  no image viewer found (display/feh/eog/xdg-open) — PNG written only")


def geom_cells(res, fld):
    """(centres (N,3), density (N,), volume (N,)) for any 3D target, else None."""
    from sph_interp import UniformGrid, AMRGrid, VoronoiGrid
    grid = res.target
    if isinstance(grid, UniformGrid):
        cx, cy, cz = grid.centres()
        C = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), axis=-1).reshape(-1, 3)
        return (C, np.asarray(res.data[fld]).ravel(),
                np.full(C.shape[0], grid.cell_volume))
    if isinstance(grid, (AMRGrid, VoronoiGrid)):
        return np.asarray(grid.centres()), np.asarray(res.data[fld]), grid.volumes()
    return None                                      # Projection2D: no 3D density


def report_budget(results, coarse_by_method, p, fld):
    """Print total mass (Sum rho_cell*V_cell) and total volume (Sum V_cell) for
    every grid, against the SPH particles (mass Sum m_i, volume Sum m_i/rho_i).
    `coarse_by_method` is {method: {geometry: result}} (or None) so the coarse
    budget is broken out per interpolation method."""
    sph_m = float(np.sum(p.mass))
    sph_v = float(np.sum(p.mass / np.maximum(p.rho, 1e-30)))
    box_v = float(np.prod(p.box)) if p.box is not None else float("nan")
    print("=" * 70)
    print(f"  mass / volume budget (field '{fld}')")
    print(f"  SPH particles : mass = {sph_m:.6g}   "
          f"volume = {sph_v:.6g}  (Sum m_i/rho_i)")
    print(f"  domain (box)  : volume = {box_v:.6g}")

    def _report(label, meshes):
        print(f"  {label} meshes:")
        for name in ("UniformGrid", "AMRGrid", "VoronoiGrid"):
            gc = geom_cells(meshes[name], fld)
            if gc is None:
                continue
            _, dens, vol = gc
            m = float(np.sum(dens * vol)); v = float(np.sum(vol))
            print(f"    {name:<12s} mass = {m:.6g} ({100*(m/sph_m-1):+6.2f}%)   "
                  f"volume = {v:.6g} ({100*(v/sph_v-1):+6.2f}%)")

    _report("full-resolution", results)
    if coarse_by_method:
        for method, meshes in coarse_by_method.items():
            _report(f"coarse [{method}]", meshes)
    print("=" * 70)


def shell_density(r, mass, vol, edges):
    """Volume-weighted mean density per radial shell = sum(mass)/sum(volume)."""
    m, _ = np.histogram(r, bins=edges, weights=mass)
    v, _ = np.histogram(r, bins=edges, weights=vol)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(v > 0, m / v, np.nan)
    return d


def sample_field_at_points(res, fld, pts):
    """Look up the mesh field at arbitrary 3D points (which cell contains each
    point) for any 3D target; returns (Npts,) with NaN where a point is outside
    the mesh. Lets the radial profile evaluate the ACTUAL field, so a coarse /
    single-cell mesh reads as its constant density at every radius."""
    from sph_interp import UniformGrid, AMRGrid, VoronoiGrid
    grid = res.target
    vals = np.asarray(res.data[fld])
    n = pts.shape[0]
    out = np.full(n, np.nan)
    if isinstance(grid, UniformGrid):
        npx = np.asarray(grid.npx)
        idx = np.floor((pts - grid.lo) / grid.pixwidth).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < npx), axis=1)
        ic = np.clip(idx, 0, npx - 1)
        out[inside] = vals[ic[inside, 0], ic[inside, 1], ic[inside, 2]]
        return out
    if isinstance(grid, VoronoiGrid):
        # Voronoi membership IS the nearest generator, by definition. scipy's
        # cKDTree gives the same exact answer as sklearn's BallTree did, and
        # scipy is already a hard dependency of the package while sklearn is
        # not -- its absence killed the whole profile figure.
        from scipy.spatial import cKDTree
        j = cKDTree(grid.centres()).query(pts, k=1, workers=-1)[1]
        return vals[j]
    if isinstance(grid, AMRGrid):
        nb = np.asarray(grid.base_npx)
        bidx = np.floor((pts - grid.lo) / grid.base_pw).astype(np.int64)
        inside = np.all((bidx >= 0) & (bidx < nb), axis=1)
        bc = np.clip(bidx, 0, nb - 1)
        bflat = (bc[:, 0] * nb[1] + bc[:, 1]) * nb[2] + bc[:, 2]
        clo = grid.cell_lo; chi = grid.cell_lo + grid.cell_size
        for k in range(n):                              # few leaves per base cell
            if not inside[k]:
                continue
            b = bflat[k]; P = pts[k]
            for li in range(grid.base_start[b], grid.base_start[b] + grid.base_count[b]):
                if (clo[li, 0] <= P[0] < chi[li, 0] and clo[li, 1] <= P[1] < chi[li, 1]
                        and clo[li, 2] <= P[2] < chi[li, 2]):
                    out[k] = vals[li]
                    break
        return out
    return out                                          # Projection2D: no 3D field


def sampled_radial_profile(res, fld, cc, rmid, box, periodic, rng, nsphere=600):
    """Radial profile by SAMPLING the mesh field: at each shell radius, average
    the field over `nsphere` random directions on the sphere of that radius about
    the core `cc`. Unlike binning cell centres this reads the actual field, so a
    coarse / single-cell mesh shows a flat line at its density (periodic axes are
    wrapped into the centered box for the lookup)."""
    prof = np.full(len(rmid), np.nan)
    half = 0.5 * np.asarray(box) if box is not None else None
    per = np.atleast_1d(periodic)
    for i, rr in enumerate(rmid):
        d = rng.normal(size=(nsphere, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = cc + rr * d
        if half is not None:
            for a in range(3):
                if (per[a] if per.size == 3 else per[0]):
                    pts[:, a] = (pts[:, a] + half[a]) % box[a] - half[a]
        v = sample_field_at_points(res, fld, pts)
        if np.isfinite(v).any():
            prof[i] = np.nanmean(v)
    return prof


def build_parser():
    pa = argparse.ArgumentParser(
        prog="interp_demo",
        description="Showcase sph_interp across all target geometries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("snapshot", nargs="?", default=None,
                    help="tipsy snapshot (default: none -- generate synthetic "
                         f"data instead, one showcase per "
                         f"{'/'.join(DEFAULT_SYNTHETIC)})")
    pa.add_argument("--synthetic", default=None,
                    help="generate this sph_interp.synthetic family instead of "
                         "reading a snapshot (uniform, lattice, closepacked, "
                         "glass, slab1d, slab2d, disk, merger, collapse, cosmo)")
    pa.add_argument("--synthetic-n", type=int, default=DEFAULT_SYNTHETIC_N,
                    help="particle count for --synthetic")
    pa.add_argument("--method",
                    choices=["sph", "sph-gather", "sph-scatter", "petkova"],
                    default="sph",
                    help="deposit method: sph(-gather) kernel estimate, "
                         "sph-scatter mass-conserving, or petkova exact")
    pa.add_argument("--field", default="rho", help="field to interpolate/show")
    pa.add_argument("--npx", type=int, default=128,
                    help="resolution: uniform/projection cells per axis; AMR "
                         "finest level; image raster for AMR/Voronoi slices")
    pa.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy",
                    help="slice / projection plane")
    pa.add_argument("--cmap", default="turbo",
                    help="colormap for the density / field panels. Default turbo "
                         "(no black — voids read blue, not black). Other no-black "
                         "options: viridis, plasma, cividis, Spectral_r, jet, "
                         "rainbow. (inferno/magma/gist_heat are black at the low "
                         "end.)")
    pa.add_argument("--base-npx", type=int, default=16, help="AMR base grid/axis")
    pa.add_argument("--voronoi-ncells", type=int, default=40000,
                    help="Voronoi generators (subsampled particles); 0 = use all")
    pa.add_argument("--voronoi-background", default="auto",
                    type=lambda s: s if s.lower() == "auto" else int(s),
                    help="background guard lattice per axis: auto adds 16^3 for "
                         "open/sparse-periodic domains, 0 disables, or an "
                         "integer >=2 forces N^3")
    pa.add_argument("--voronoi-backend", default="auto",
                    choices=["auto", "pyvoro", "multivoro", "arepo"],
                    help="Voronoi mesher (default auto = arepo if built else "
                         "multivoro/pyvoro). The arepo backend now meshes non-cubic "
                         "boxes natively (per-axis LONG_Y/LONG_Z); open captures "
                         "use periodic mesh closure beyond their support-complete "
                         "guard region, without wrapping the deposit.")
    pa.add_argument("--project", default="column",
                    choices=["column", "average", "rhocolumn", "slice"],
                    help="Projection2D mode")
    pa.add_argument("--mesh-zoom", type=float, default=0.25,
                    help="fraction of the box (centred) for the actual-cells "
                         "row — cells are only legible zoomed in")
    pa.add_argument("--out", default="interp_demo.png",
                    help="output figure (png/pdf)")
    pa.add_argument("--no-diagnostics", dest="diagnostics",
                    action="store_false",
                    help="omit the field-PDF and interpolated-vs-particle "
                         "rows (the interpolatetogrid diagnostic panels)")
    pa.add_argument("--no-mesh3d", dest="mesh3d", action="store_false",
                    help="skip the second figure (true 3D cells: Uniform/AMR/"
                         "Voronoi polyhedra in a sub-box)")
    pa.set_defaults(mesh3d=True)
    pa.add_argument("--no-profile", dest="profile", action="store_false",
                    help="skip the density-vs-radius profile figure")
    pa.set_defaults(profile=True)
    # Default high enough to draw the WHOLE coarse mesh, guard cells included.
    # The coarse mesh is already the mechanism that keeps the count drawable;
    # capping it a second time only hid most of it (see draw_cells_3d).
    pa.add_argument("--mesh3d-cells", type=int, default=20000,
                    help="max cells to draw in the 3D figure (per geometry; "
                         "keeps the highest-value cells, never random)")
    pa.add_argument("--mesh3d-coarse", type=int, default=12,
                    help="3D figure uses a COARSE dedicated mesh — this is its "
                         "uniform cells/axis (also sets the coarse AMR finest "
                         "level, derived from it and --mesh3d-base-npx)")
    pa.add_argument("--mesh3d-base-npx", type=int, default=6,
                    help="base grid/axis for the COARSE AMR mesh (3D figure + "
                         "profile + coarse budget); finest level is derived so "
                         "base*2^level >= --mesh3d-coarse")
    pa.add_argument("--mesh3d-voronoi", type=int, default=1500,
                    help="generators for the coarse 3D Voronoi mesh "
                         "(~matches the AMR leaf count shown)")
    pa.add_argument("--mesh3d-cull", type=float, default=None,
                    help="drop 3D cells below this value percentile (declutter "
                         "voids); default None — voids fade via value-alpha")
    pa.add_argument("--mesh3d-clip",
                    choices=["none", "half", "quarter", "octant"],
                    default="quarter",
                    help="cutaway for the 3D figure: remove the near "
                         "half/quarter/octant (centred on the densest cell) to "
                         "expose the core and central AMR refinement")
    g = pa.add_mutually_exclusive_group()
    g.add_argument("--show", dest="show", action="store_true",
                   help="pop the figures open in an X window (default: on when "
                        "$DISPLAY is set)")
    g.add_argument("--no-show", dest="show", action="store_false",
                   help="only write the PNGs, do not open a viewer")
    pa.set_defaults(show=None)                       # None -> auto from $DISPLAY
    pa.add_argument("--per-geometry", action="store_true",
                    help="also write one SPH-vs-grid diagnostic figure per "
                         "geometry (<out-stem>_<geometry>.pdf)")
    pa.add_argument("--nsmooth", type=int, default=64)
    return pa


def main(argv=None):
    args = build_parser().parse_args(argv)

    # No input named at all: showcase each default synthetic family in turn,
    # writing one figure per family. Re-invoking main() per family keeps the
    # single-source body below unchanged rather than threading a loop through
    # 400 lines of figure building.
    if args.snapshot is None and args.synthetic is None:
        base = list(argv) if argv is not None else sys.argv[1:]
        # drop any user --out (flag AND its value); we derive one per family
        clean, skip = [], False
        for a in base:
            if skip:
                skip = False
                continue
            if a == "--out":
                skip = True
                continue
            if a.startswith("--out="):
                continue
            clean.append(a)
        stem, ext = os.path.splitext(args.out)
        rc = 0
        for fam in DEFAULT_SYNTHETIC:
            print(f"\n{'=' * 70}\n  synthetic family: {fam}\n{'=' * 70}")
            rc = main(clean + ["--synthetic", fam,
                               "--out", f"{stem}_{fam}{ext}"]) or rc
        return rc

    # a 3D Voronoi tessellation needs at least a few generators; below 3 it is
    # degenerate (and pyvoro can fail). Warn and floor at 3. (0 = "use all".)
    for attr in ("voronoi_ncells", "mesh3d_voronoi"):
        v = getattr(args, attr)
        if 0 < v < 3:
            print(f"  WARNING: --{attr.replace('_', '-')}={v} is too few Voronoi "
                  f"generators; using 3.")
            setattr(args, attr, 3)

    show = args.show if args.show is not None else bool(os.environ.get("DISPLAY"))

    import matplotlib
    # When showing, prefer an interactive backend so plt.show() draws the figures
    # directly in a window, instead of writing PNGs and shelling out to an
    # external viewer. Fall back to headless Agg if none load.
    #
    # matplotlib.use() only validates the NAME -- it does not import the backend
    # until the first figure is created, so a missing GUI toolkit (e.g. GTK3
    # without PyGObject) would not fail here but blow up later in plt.subplots().
    # switch_backend() imports it now, so the try/except below is meaningful.
    interactive = False
    if show:
        cands = ["MacOSX"] if sys.platform == "darwin" else []
        cands += ["QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg"]
        env_bk = os.environ.get("MPLBACKEND")
        if env_bk:
            cands.insert(0, env_bk)
        import matplotlib.pyplot as _plt
        for bk in cands:
            try:
                _plt.switch_backend(bk)
                interactive = True
                break
            except Exception:
                continue
    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from sph_interp import (from_tipsy, interpolate, UniformGrid, AMRGrid,
                            VoronoiGrid, Projection2D, refine_smoothing_length,
                            refine_count, field_image, visualise, draw_cells,
                            draw_cells_3d, kernel_support_bounds,
                            density_pdf_panel, grid_vs_particle_panel)
    from sph_interp.targets import _resolve_backend, _backend_availability

    eff_backend = _resolve_backend(args.voronoi_backend)
    avail = _backend_availability()
    # `auto` walks arepo -> multivoro -> pyvoro; name the rungs it skipped so a
    # fallback to the slow mesher is visible rather than silent.
    if args.voronoi_backend == "auto":
        skipped = [f"{n}: {why}" for n, ok, why in avail
                   if n != eff_backend and not ok]
        backend_note = f"{eff_backend} (auto"
        backend_note += ("; tried " + ", ".join(skipped) + ")") if skipped else ")"
    else:
        backend_note = eff_backend
    # the resolved backend can still be absent (auto's last rung is unconditional)
    backend_warn = next((f"    WARNING: {n} is {why} — the VoronoiGrid build will "
                         f"fail" for n, ok, why in avail
                         if n == eff_backend and not ok), None)

    a1, a2 = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.axes]
    fld = args.field

    def interp(pp, tgt):
        """interpolate() with the demo's chosen method/field. sph-scatter is now
        implemented for every geometry (Projection2D's `slice` mode delegates to
        gather internally, since a plane sample is not an extensive projection)."""
        return interpolate(pp, tgt, method=args.method, values=[fld])

    if args.synthetic:
        from sph_interp import synthetic as _syn
        if args.synthetic not in _syn.FAMILIES:
            raise SystemExit(f"unknown --synthetic family "
                             f"{args.synthetic!r}; expected one of "
                             f"{', '.join(_syn.FAMILIES)}")
        if fld != "rho":
            raise SystemExit(f"--synthetic generates only 'rho'; "
                             f"--field {fld!r} needs a real snapshot")
        print(f"generating synthetic '{args.synthetic}' "
              f"(N={args.synthetic_n:,})  field={fld}")
        p = _syn.make(args.synthetic, args.synthetic_n)
        src_label = f"synthetic:{args.synthetic}"
    else:
        print(f"reading {args.snapshot}  field={fld}")
        p = from_tipsy(args.snapshot, fields=(fld,), nsmooth=args.nsmooth)
        src_label = os.path.basename(args.snapshot)
    print(f"  {p.n} gas particles, box {np.round(p.box, 4)}")
    box = p.box

    # --- timing: separate mesh GENERATION from INTERPOLATION for each stage --
    import time
    timings = []                             # (stage, label, seconds)

    def timed(stage, label, fn, *a, **kw):
        """Run fn(*a, **kw), record its wall time under (stage, label), return
        (result, seconds)."""
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        dt = time.perf_counter() - t0
        timings.append((stage, label, dt))
        return out, dt

    # --- build + interpolate each target geometry --------------------------
    results = {}

    def _t_line(tb, ti):
        """The per-stage timing line: total = mesh generation + interpolation."""
        print(f"    t_tot = {tb + ti:.3f}s   t_mesh = {tb:.3f}s   "
              f"t_inter = {ti:.3f}s")

    print("  UniformGrid ...")
    g, tb = timed("build", "UniformGrid", UniformGrid.centered, box, args.npx)
    print(f"    {args.npx}^3 = {args.npx ** 3} cells")
    results["UniformGrid"], ti = timed("interp", f"UniformGrid/{args.method}",
                                       interp, p, g)
    _t_line(tb, ti)

    print("  AMRGrid (octree) ...")
    import math
    amr_count = 8                            # refine while a cell holds > this many particles
    max_level = max(0, math.ceil(math.log2(max(args.npx, 1) / args.base_npx)))
    fine = args.base_npx * (2 ** max_level)
    g, tb = timed("build", "AMRGrid", AMRGrid.build, p, base_npx=args.base_npx,
                  max_level=max_level, criterion=refine_count(amr_count))
    print(f"    {g.ncell} leaf cells")
    print(f"    refine: particle-count, n_target={amr_count} "
          f"(split while cell holds > {amr_count} particles)")
    print(f"    base {args.base_npx}^3 -> finest {fine}^3 "
          f"(max_level {max_level}; effective range {args.base_npx}..{fine} per axis)")
    results["AMRGrid"], ti = timed("interp", f"AMRGrid/{args.method}", interp, p, g)
    _t_line(tb, ti)

    print(f"  VoronoiGrid ... [backend: {backend_note}]")
    if backend_warn:
        print(backend_warn)
    half = 0.5 * np.asarray(box)
    # Periodic box -> bounds ARE the period. Open Petkova capture -> the exact
    # union of every particle's compact 2h support, also containing the nominal
    # demo box. Background guards then fill the zero-source exterior.
    vper = p.periodic if any(np.atleast_1d(p.periodic)) else False
    if vper:
        bounds = [[-half[a], half[a]] for a in range(3)]
    else:
        support_lo, support_hi = kernel_support_bounds(p.pos, p.h)
        capture_lo = np.minimum(-half, support_lo)
        capture_hi = np.maximum(half, support_hi)
        bounds = [[capture_lo[a], capture_hi[a]] for a in range(3)]
        print(
            "    open capture = union(pos +/- 2h), "
            f"lo={np.round(capture_lo, 4)}, hi={np.round(capture_hi, 4)}")
    if args.voronoi_ncells and args.voronoi_ncells < p.n:
        idx = np.random.default_rng(0).choice(p.n, args.voronoi_ncells,
                                              replace=False)
        pts = p.pos[idx]
        print(f"    {args.voronoi_ncells} generators (subsampled particles)"
              f"{' periodic' if any(np.atleast_1d(vper)) else ''}")
    else:
        pts = p.pos
        print(f"    {p.n} generators (all particles)")
    g, tb = timed("build", "VoronoiGrid", VoronoiGrid.from_points, pts,
                  bounds=bounds, periodic=vper, backend=args.voronoi_backend,
                  periodic_background=args.voronoi_background)
    results["VoronoiGrid"], ti = timed("interp", f"VoronoiGrid/{args.method}",
                                       interp, p, g)
    _t_line(tb, ti)

    print(f"  Projection2D [{args.project}] ...")
    g, tb = timed("build", "Projection2D", Projection2D.centered, box, a1, a2,
                  npx=args.npx, mode=args.project)
    results["Projection2D"], ti = timed("interp", f"Projection2D/{args.method}",
                                        interp, p, g)
    _t_line(tb, ti)

    # --- assemble images; share a colour scale across the 3D volume grids ---
    order = ["UniformGrid", "AMRGrid", "VoronoiGrid", "Projection2D"]
    imgs = {}
    for name in order:
        img, extent, _ = field_image(results[name], fld, plane=(a1, a2),
                                     slab_frac=0.05, img_npx=args.npx)
        imgs[name] = (img, extent)

    vol = np.concatenate([imgs[n][0][np.isfinite(imgs[n][0])].ravel()
                          for n in ("UniformGrid", "AMRGrid", "VoronoiGrid")])
    vpos = vol[vol > 0]
    use_log = vpos.size and (vpos.max() / vpos.min() > 50)
    vnorm = (LogNorm(vmin=vpos.min(), vmax=vpos.max()) if use_log
             else Normalize(vmin=vol.min(), vmax=vol.max()))

    # --- mesh-resolution images: paint each geometry by its CELL SIZE, so the
    #     adaptivity is visible (AMR refines / Voronoi cells shrink on the core;
    #     uniform & projection are flat by construction). Reuse field_image by
    #     injecting a synthetic per-cell "cell width" field into each result.
    def cell_width_field(res):
        grid = res.target
        if isinstance(grid, UniformGrid):
            return np.full(tuple(grid.npx), float(grid.pixwidth.min()))
        if isinstance(grid, Projection2D):
            return np.full(tuple(grid.npx), float(grid.pixwidth.min()))
        return grid.volumes() ** (1.0 / 3.0)          # AMR / Voronoi: leaf width

    struct = {}
    for name in order:
        res = results[name]
        res.data["_cellw"] = cell_width_field(res)
        img, extent, _ = field_image(res, "_cellw", plane=(a1, a2),
                                     slab_frac=0.05, img_npx=args.npx)
        struct[name] = (img, extent)
    sall = np.concatenate([struct[n][0][np.isfinite(struct[n][0])].ravel()
                           for n in ("UniformGrid", "AMRGrid", "VoronoiGrid")])
    snorm = LogNorm(vmin=sall[sall > 0].min(), vmax=sall.max())

    # --- zoom window (centred) for the actual-cells row --------------------
    cen = 0.5 * (p.pos.min(0) + p.pos.max(0))
    qh = 0.5 * args.mesh_zoom * box
    zoom = (cen[a1] - qh[a1], cen[a1] + qh[a1],
            cen[a2] - qh[a2], cen[a2] + qh[a2])

    # --- the showcase figure: 3 or 5 rows x 5 cols -------------------------
    #     row 0 = interpolated field; row 1 = mesh resolution (cell width);
    #     row 2 = the ACTUAL cells (boundaries), zoomed in so they are legible.
    #     The fidelity diagnostics (field PDF, interpolated-vs-particle) go to
    #     their OWN page, <out>_diagnostics.png -- three renders is a full
    #     figure already, and the diagnostics are read differently: the renders
    #     are compared across geometries at a glance, the diagnostics are read
    #     one panel at a time.
    axlabel = "xyz"
    fig, axes = plt.subplots(3, 5, figsize=(26, 15.8))
    fig.suptitle(f"sph_interp showcase — {src_label} — "
                 f"field '{fld}', method '{args.method}', plane {args.axes}",
                 fontsize=16, fontweight="bold")
    # Empty cells read as the BOTTOM of the ramp, not as the page. A cell with
    # no material has rho = 0, which a LogNorm cannot place, so it is masked --
    # and a masked value defaults to transparent, i.e. white. That is visually
    # identical to "outside the plotted area" and makes an empty region look
    # like missing data. Painting it the ramp's low end says "nothing here".
    _base = matplotlib.colormaps[args.cmap]
    cmap = _base.with_extremes(bad=_base(0.0))
    scmap = "viridis_r"                              # small cells -> bright

    los = ({0, 1, 2} - {a1, a2}).pop()
    pos = p.pos
    zm = 0.5 * (pos[:, los].min() + pos[:, los].max())
    # A fixed 5% slab can select NOTHING. A lattice or close-packed sampling
    # puts the particles on discrete planes along the line of sight, and an
    # even number of planes (slab2d has 12 across a 0.094-thick box) leaves the
    # mid-plane in the gap between the two central ones -- 2.5% of the box is
    # narrower than half the plane spacing, so the slab lands empty and every
    # particle panel dies on an empty reduction. Widen to the nearest plane
    # whenever that happens; on a continuous distribution the nominal 5% wins
    # and nothing changes.
    _dlos = np.abs(pos[:, los] - zm)
    half_s = 0.5 * 0.05 * (pos[:, los].max() - pos[:, los].min())
    if not np.any(_dlos <= half_s):
        half_s = float(_dlos.min()) * (1.0 + 1e-9)
    sel = _dlos <= half_s
    cval = p.values[fld] if fld in p.values else p.rho

    # Rows 0-1 share the PARTICLE extent across all five columns. The particle
    # scatter autoscales to the particles while the mesh panels take their
    # limits from the grid, so on an open capture -- where the domain is padded
    # to ~100x the object -- the two columns disagreed by a factor of ~3 in
    # view size. The galaxy then looked like a dot inside a huge blank frame
    # next to a particle panel that filled its axes.
    _m1, _m2 = pos[sel, a1], pos[sel, a2]
    _pad = 0.03 * max(float(np.ptp(_m1)), float(np.ptp(_m2)))
    if _pad <= 0.0:                       # single plane / single point in-plane
        _pad = 0.03 * max(float(box[a1]), float(box[a2]))
    pext = (float(_m1.min()) - _pad, float(_m1.max()) + _pad,
            float(_m2.min()) - _pad, float(_m2.max()) + _pad)

    def _share_view(ax):
        ax.set_xlim(pext[0], pext[1])
        ax.set_ylim(pext[2], pext[3])

    # row 0, col 0: particle field
    ax = axes[0, 0]
    ax.scatter(pos[sel, a1], pos[sel, a2], c=cval[sel], s=2, cmap=cmap,
               norm=vnorm, linewidths=0)
    ax.set_title(f"SPH particles\n(slab, {sel.sum()} pts)")
    ax.set_ylabel("interpolated field", fontsize=12, fontweight="bold")
    ax.set_aspect("equal"); _share_view(ax)
    # row 1, col 0: particle smoothing length (the SPH resolution)
    ax = axes[1, 0]
    ax.scatter(pos[sel, a1], pos[sel, a2], c=p.h[sel], s=2, cmap=scmap,
               norm=LogNorm(), linewidths=0)
    ax.set_title("smoothing length h")
    ax.set_ylabel("mesh resolution (cell width)", fontsize=12, fontweight="bold")
    ax.set_aspect("equal"); _share_view(ax)
    # row 2, col 0: particles inside the zoom window (what the cells represent)
    ax = axes[2, 0]
    selz = (sel & (pos[:, a1] >= zoom[0]) & (pos[:, a1] <= zoom[1])
            & (pos[:, a2] >= zoom[2]) & (pos[:, a2] <= zoom[3]))
    ax.scatter(pos[selz, a1], pos[selz, a2], c=cval[selz], s=6, cmap=cmap,
               norm=vnorm, linewidths=0)
    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    ax.set_title(f"particles (zoom, {selz.sum()} pts)")
    ax.set_ylabel(f"actual cells (zoom x{1/args.mesh_zoom:.0f})",
                  fontsize=12, fontweight="bold")
    ax.set_xlabel(axlabel[a1]); ax.set_aspect("equal")

    for k, name in enumerate(order, start=1):
        # --- row 0: field ---
        ax = axes[0, k]
        img, extent = imgs[name]
        if name == "Projection2D":                # own scale (column units)
            d = img[np.isfinite(img)]; dp = d[d > 0]
            pn = (LogNorm(vmin=dp.min(), vmax=dp.max())
                  if dp.size and dp.max() / dp.min() > 50
                  else Normalize(vmin=d.min(), vmax=d.max()))
            im = ax.imshow(np.ma.masked_invalid(img.T), origin="lower",
                           extent=extent, cmap=cmap, norm=pn, aspect="equal",
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Projection2D\n[{args.project}] (own scale)")
        else:
            ax.imshow(np.ma.masked_invalid(img.T), origin="lower", extent=extent,
                      cmap=cmap, norm=vnorm, aspect="equal",
                      interpolation="nearest")
            extra = (f"\n{results[name].target.ncell} leaves" if name == "AMRGrid"
                     else f"\n{results[name].target.ncell} cells"
                     if name == "VoronoiGrid" else "")
            ax.set_title(f"{name}{extra}")
        _share_view(ax)
        # --- row 1: mesh resolution ---
        ax = axes[1, k]
        simg, sext = struct[name]
        flat = name in ("UniformGrid", "Projection2D")
        im = ax.imshow(np.ma.masked_invalid(simg.T), origin="lower", extent=sext,
                       cmap=scmap, norm=snorm, aspect="equal",
                       interpolation="nearest")
        ax.set_title("regular (constant cell)" if flat
                     else "adaptive (cell tracks density)")
        _share_view(ax)

        # --- row 2: the actual cells (boundaries), zoomed in ---
        ax = axes[2, k]
        cnorm = vnorm if name != "Projection2D" else None
        _, ncelltxt = draw_cells(ax, results[name], fld, plane=(a1, a2),
                                 zoom=zoom, norm=cnorm, cmap=cmap)
        ax.set_title(ncelltxt)
        ax.set_xlabel(axlabel[a1])

    # shared colorbars
    sm = plt.cm.ScalarMappable(norm=vnorm, cmap=cmap)
    fig.colorbar(sm, ax=axes[0, 1:4], fraction=0.025, pad=0.01,
                 label=f"{fld} (shared)")
    sm2 = plt.cm.ScalarMappable(norm=snorm, cmap=scmap)
    fig.colorbar(sm2, ax=axes[1, 1:4], fraction=0.025, pad=0.01,
                 label="cell width (shared)")
    for name in order:                                # tidy the injected field
        results[name].data.pop("_cellw", None)

    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote showcase -> {args.out}")
    show_figure(args.out, show and not interactive)

    # --- page 2: fidelity diagnostics, one column per geometry -------------
    # The same two panels `interpolatetogrid` writes, drawn by the same
    # functions in sph_interp.visualise, so the demo and the CLI cannot drift.
    if args.diagnostics:
        print("  building fidelity diagnostics (PDF + per-particle) ...")
        figd, axd = plt.subplots(2, len(order), figsize=(6.2 * len(order), 11.4))
        figd.suptitle(f"sph_interp fidelity — {src_label} — "
                      f"field '{fld}', method '{args.method}'",
                      fontsize=16, fontweight="bold")
        for k, name in enumerate(order):
            gsamp, parts, use_log = density_pdf_panel(
                axd[0, k], results[name], p, fld, gname=name)
            grid_vs_particle_panel(
                axd[1, k], results[name], p, fld, gsamp=gsamp, parts=parts,
                use_log=use_log, gname=name)
            axd[0, k].set_title(f"{name}\n{axd[0, k].get_title()}", fontsize=12)
        figd.text(0.5, 0.012,
                  "top: mass-weighted field PDF — SPH particles (blue), mesh "
                  "cells (orange), mesh sampled back at the particles (red); "
                  "all three coincident = the distribution survived.    "
                  "bottom: each particle's own value vs the mesh value where "
                  "it sits; r and median(grid/SPH) both 1.000 = faithful.",
                  ha="center", fontsize=11)
        figd.tight_layout(rect=[0, 0.03, 1, 0.96])
        outd = os.path.splitext(args.out)[0] + "_diagnostics.png"
        figd.savefig(outd, dpi=130, bbox_inches="tight")
        print(f"wrote diagnostics -> {outd}")
        show_figure(outd, show and not interactive)

    # --- coarse dedicated meshes (used by BOTH the 3D figure and the profile) -
    #     The interpolation meshes are far too fine for a 3D cell picture (a
    #     sub-box slices into thousands of tiny fragments). Build a coarse,
    #     COMPLETE mesh per geometry — few, large cells, like the original
    #     interpolate_petkova.py polyhedra figures.
    cc = p.pos[np.argmax(cval)]                      # densest cell: core centre
    coarse_by_method = None
    if args.mesh3d or args.profile:
        print("  building coarse dedicated meshes ...")
        ncu = args.mesh3d_coarse
        nab = max(1, args.mesh3d_base_npx)
        # coarse AMR: base nab + a few levels (finest ~ mesh3d_coarse), refined on
        # PARTICLE COUNT so it stays coarse in voids and nests on the crowded core.
        # By DEFAULT floor the depth at 3 levels so the octree nesting is visible
        # in the 3D figure; but if the user explicitly sets either coarse-AMR knob,
        # honour it EXACTLY with no floor (so --mesh3d-coarse 1 --mesh3d-base-npx 1
        # really is a single AMR cell, not 3 forced levels -> ~27 leaves).
        _d = build_parser()
        user_amr = (ncu != _d.get_default("mesh3d_coarse")
                    or args.mesh3d_base_npx != _d.get_default("mesh3d_base_npx"))
        mlc = max(0 if user_amr else 3,
                  math.ceil(math.log2(max(ncu, 1) / nab)))
        ncrit = max(32, int(p.n / (nab ** 3 * 8)))   # ~ a few-per-base-cell target
        gu, tbu = timed("build", "coarse UniformGrid", UniformGrid.centered,
                        box, ncu)
        ga, tba = timed("build", "coarse AMRGrid", AMRGrid.build, p,
                        base_npx=nab, max_level=mlc,
                        criterion=refine_count(ncrit))
        nvg = min(args.mesh3d_voronoi, p.n)
        iv = np.random.default_rng(1).choice(p.n, nvg, replace=False)
        gv, tbv = timed("build", "coarse VoronoiGrid", VoronoiGrid.from_points,
                        p.pos[iv], bounds=bounds, periodic=vper,
                        backend=args.voronoi_backend,
                        periodic_background=args.voronoi_background)
        coarse_grids = {"UniformGrid": gu, "AMRGrid": ga, "VoronoiGrid": gv}
        # Report the Voronoi count the mesh ACTUALLY has, not the count asked
        # for: a thin box gets extra generators inserted so its cells stay
        # inside the periodic unwrap limit, and printing the request there
        # (1500 for an 8112-cell mesh) misreports the figure that follows.
        nvg_built = getattr(gv, "ncentral", None) or gv.ncell
        nvg_note = "" if nvg_built == nvg else f" (asked {nvg})"
        print(f"    coarse meshes: uniform {ncu}^3, AMR base{nab}/level{mlc} "
              f"({ga.ncell} leaves), Voronoi {nvg_built} cells{nvg_note}; "
              f"methods {', '.join(COARSE_METHODS)}")
        # deposit every method onto the coarse meshes (cheap here) so the profile
        # can overlay gather / scatter / petkova; the 3D figure + budget below
        # use one representative (petkova — the exact, mass-conserving integral).
        coarse_by_method = {}
        for m in COARSE_METHODS:
            coarse_by_method[m] = {}
            for n, g in coarse_grids.items():
                res, _ = timed("interp", f"coarse {n}/{m}",
                               interpolate, p, g, method=m, values=[fld])
                coarse_by_method[m][n] = res
        tb_coarse = tbu + tba + tbv
        ti_coarse = sum(dt for st, lb, dt in timings
                        if st == "interp" and lb.startswith("coarse "))
        print(f"    coarse totals: t_tot = {tb_coarse + ti_coarse:.3f}s   "
              f"t_mesh = {tb_coarse:.3f}s   t_inter = {ti_coarse:.3f}s "
              f"({len(COARSE_METHODS)} methods x 3 geometries)")

    # --- mass / volume budget (printed text) --------------------------------
    report_budget(results, coarse_by_method, p, fld)

    # --- second figure: the TRUE 3D cells on the coarse meshes --------------
    if args.mesh3d:
        print("  rendering true 3D cells (petkova) ...")
        c3 = coarse_by_method["petkova"]            # one method suffices in 3D
        cvall = np.concatenate([np.asarray(c3[n].data[fld]).ravel()
                                for n in c3])
        cvp = cvall[cvall > 0]
        cnorm3 = (LogNorm(vmin=cvp.min(), vmax=cvp.max())
                  if cvp.size and cvp.max() / cvp.min() > 50
                  else Normalize(vmin=cvall.min(), vmax=cvall.max()))

        # cutaway centred on the densest cell -> the cut faces pass through the
        # core, exposing it (and the central AMR refinement). View is +x+y+z, so
        # remove the HIGH (near) side.
        clipdef = {"none": None,
                   "half": [(1, "high", cc[1])],
                   "quarter": [(0, "high", cc[0]), (1, "high", cc[1])],
                   "octant": [(0, "high", cc[0]), (1, "high", cc[1]),
                              (2, "high", cc[2])]}[args.mesh3d_clip]

        fig3 = plt.figure(figsize=(21, 7))
        fig3.suptitle(f"sph_interp — actual 3D cells (coarse meshes, "
                      f"{args.mesh3d_clip} cutaway) — "
                      f"{src_label}, field '{fld}', "
                      f"method 'petkova'", fontsize=15, fontweight="bold")
        for j, name in enumerate(["UniformGrid", "AMRGrid", "VoronoiGrid"], 1):
            ax3 = fig3.add_subplot(1, 3, j, projection="3d")
            _, lab = draw_cells_3d(ax3, c3[name], fld, norm=cnorm3, cmap=cmap,
                                   max_cells=args.mesh3d_cells,
                                   cull_percentile=args.mesh3d_cull,
                                   clip=clipdef)
            ax3.set_title(f"{name}\n{lab}")
        sm3 = plt.cm.ScalarMappable(norm=cnorm3, cmap=cmap)
        fig3.colorbar(sm3, ax=fig3.axes, fraction=0.015, pad=0.02, label=fld)
        out3 = os.path.splitext(args.out)[0] + "_mesh3d.png"
        fig3.savefig(out3, dpi=130, bbox_inches="tight")
        print(f"wrote 3D cells -> {out3}")
        show_figure(out3, show and not interactive)

    # --- third figure: density vs radius, full-res and coarse meshes --------
    if args.profile:
        print("  building density-vs-radius profile ...")
        # radial bins centred on the core (densest cell)
        rpart = np.linalg.norm(p.pos - cc, axis=1)
        rmax = np.percentile(rpart, 99.5)
        redges = np.linspace(0.0, rmax, 40)         # linear radial bins
        rmid = 0.5 * (redges[:-1] + redges[1:])
        # SPH particle reference: volume-weighted mean density per shell
        pvol = p.mass / np.maximum(p.rho, 1e-30)
        sph_prof = shell_density(rpart, p.mass, pvol, redges)

        gcol = {"UniformGrid": "tab:blue", "AMRGrid": "tab:green",
                "VoronoiGrid": "tab:red"}
        # each geometry keeps its colour family; on the coarse panel the three
        # methods are light->dark shades within it (sph-gather light, sph-scatter
        # mid, petkova dark), so method differences read as brightness.
        geom_shades = {
            "UniformGrid": {"sph-gather": "#9ecae1", "sph-scatter": "#3182bd",
                            "petkova": "#08306b"},
            "AMRGrid":     {"sph-gather": "#a1d99b", "sph-scatter": "#31a354",
                            "petkova": "#006d2c"},
            "VoronoiGrid": {"sph-gather": "#fc9272", "sph-scatter": "#de2d26",
                            "petkova": "#67000d"},
        }
        # 2x2 layout: top-left = full-resolution meshes (all geometries, single
        # --method); the other three = one geometry each on its COARSE mesh with
        # all three methods as light->dark shades.
        figp, axp = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)

        def _profile_context(ax):
            """SPH particle scatter + shell-binned reference (shared by every panel)."""
            sub = (np.random.default_rng(0).choice(p.n, 20000, replace=False)
                   if p.n > 20000 else np.arange(p.n))
            ax.scatter(rpart[sub], p.rho[sub], s=1, alpha=0.08, color="0.6",
                       linewidths=0, zorder=1)
            ax.plot(rmid, sph_prof, "k-", lw=2.6, label="SPH particles", zorder=5)

        # sample the mesh field on each radial shell (not bin cell centres), so a
        # coarse / single-cell mesh reads as a flat line at its density.
        # top-left: full-resolution meshes, the single chosen --method
        ax = axp[0, 0]
        _profile_context(ax)
        rng_prof = np.random.default_rng(7)
        for name in ("UniformGrid", "AMRGrid", "VoronoiGrid"):
            if geom_cells(results[name], fld) is None:
                continue
            prof = sampled_radial_profile(results[name], fld, cc, rmid,
                                          p.box, p.periodic, rng_prof)
            ax.plot(rmid, prof, lw=2.0, color=gcol[name],
                    label=f"{name} ({args.method})", zorder=4)
        ax.set_title(f"full-resolution meshes — {args.method}")

        # one coarse panel per geometry: three methods as light->dark shades
        coarse_axes = [axp[0, 1], axp[1, 0], axp[1, 1]]
        for ax, name in zip(coarse_axes,
                            ("UniformGrid", "AMRGrid", "VoronoiGrid")):
            _profile_context(ax)
            rng_prof = np.random.default_rng(7)     # same shells across methods
            for method in COARSE_METHODS:
                meshes = coarse_by_method[method]
                if geom_cells(meshes[name], fld) is None:
                    continue
                prof = sampled_radial_profile(meshes[name], fld, cc, rmid,
                                              p.box, p.periodic, rng_prof)
                ax.plot(rmid, prof, lw=2.0, color=geom_shades[name][method],
                        label=method, zorder=4)
            ax.set_title(f"coarse {name} — gather / scatter / petkova")

        for ax in axp.ravel():
            ax.set_yscale("log")                    # density spans decades
            ax.set_xlim(0, rmax)
            ax.set_xlabel("radius from core")
            ax.grid(True, which="both", ls=":", alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axp[:, 0]:
            ax.set_ylabel(f"{fld} (radial mean: mesh field-sampled, SPH shell-binned)")
        figp.suptitle(f"sph_interp — {fld} vs radius — "
                      f"{src_label}, method '{args.method}'",
                      fontsize=14, fontweight="bold")
        figp.tight_layout(rect=[0, 0, 1, 0.96])
        outp = os.path.splitext(args.out)[0] + "_profile.png"
        figp.savefig(outp, dpi=130, bbox_inches="tight")
        print(f"wrote profile -> {outp}")
        show_figure(outp, show and not interactive)

    if args.per_geometry:
        stem = os.path.splitext(args.out)[0]
        for name in order:
            out = f"{stem}_{name}.pdf"
            visualise(results[name], p, field=fld, out=out, plane=(a1, a2),
                      title=f"{name} / {args.method} — {fld}")

    # --- timing summary: mesh generation vs interpolation -------------------
    t_mesh = sum(dt for st, _, dt in timings if st == "build")
    t_inter = sum(dt for st, _, dt in timings if st == "interp")
    print("=" * 70)
    print("  timing (mesh generation vs interpolation)")
    for st, lb, dt in timings:
        print(f"    {st:<6s} {lb:<28s} {dt:8.3f}s")
    print(f"  t_tot = {t_mesh + t_inter:.3f}s   t_mesh = {t_mesh:.3f}s   "
          f"t_inter = {t_inter:.3f}s")
    print("=" * 70)

    print("done.")

    if interactive:
        print(f"  showing figures ({matplotlib.get_backend()}) — close windows to exit")
        plt.show()


if __name__ == "__main__":
    main()
