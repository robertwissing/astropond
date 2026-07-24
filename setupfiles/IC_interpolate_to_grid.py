#!/usr/bin/env python
"""Interpolate a tipsy SPH snapshot onto a grid and write it out.

A thin command-line front end for the `sph_interp` package: read a gasoline/tipsy
snapshot (-> Particles via from_tipsy), deposit the requested fields onto a chosen
TARGET geometry with a chosen METHOD, and serialize the result.

    interpolate_to_grid.sh INPUT OUTPUT [options]

Run with NO positional arguments to be prompted interactively for everything.

TARGETS (--grid):   uniform | amr | voronoi | projection
METHODS (--method): petkova (exact, mass-conserving; default) |
                    sph-scatter (AREPO-style conservative kernel scatter) |
                    sph-gather (fast kernel estimate at cell centres, NOT conserving)
OUTPUT (--output-format): npz | hdf5 | ramses | cell-table | native

Examples
  # 256^3 uniform SPH density grid -> npz
  interpolate_to_grid.sh run/snap.00100 rho.npz --grid uniform --npx 256

  # mass-conserving AMR (octree) of rho+|B|, exported to RAMSES HDF5
  interpolate_to_grid.sh run/snap.00100 mesh.h5 --grid amr --method petkova \
      --fields rho,vx,vy,vz --output-format ramses

  # 50000-cell Voronoi mesh subsampled from the particles, exact deposit
  interpolate_to_grid.sh run/snap.00100 voro.npz --grid voronoi \
      --voronoi-source subsample --ncells 50000 --method petkova

  # column-density projection map in the xy plane
  interpolate_to_grid.sh run/snap.00100 col.npz --grid projection \
      --axes xy --project column

  # divergence-aware magnetic transfer: face capture + Hodge projection
  interpolate_to_grid.sh run/snap.00100 mesh.npz --grid amr \
      --fields rho,Bx,By,Bz --magnetic-projection hodge

  # vector-potential capture (div B == 0 pointwise); stores Ax,Ay,Az too
  interpolate_to_grid.sh run/snap.00100 vecpot.npz --grid uniform --npx 128 \
      --fields Bx,By,Bz --magnetic-projection vector-potential
"""

import argparse
import os
import sys

import numpy as np

# Make sure the package next to this script is importable when invoked directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  Interactive prompts (used when no positional args are given)
# --------------------------------------------------------------------------- #

def _ask(prompt, default=None, choices=None):
    suffix = f" [{default}]" if default is not None else ""
    if choices:
        suffix = f" ({'/'.join(choices)}){suffix}"
    while True:
        ans = input(f"{prompt}{suffix}: ").strip()
        if not ans and default is not None:
            return default
        if not ans:
            print("  (required)")
            continue
        if choices and ans not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        return ans


def interactive(args):
    """Fill `args` by prompting the user. Returns the populated namespace."""
    print("=" * 64)
    print(" sph_interp : interpolate a tipsy snapshot onto a grid")
    print("=" * 64)
    args.input = _ask("Input tipsy snapshot")
    args.grid = _ask("Grid / target geometry", default="uniform",
                     choices=["uniform", "amr", "voronoi", "projection"])
    args.method = _ask("Interpolation method", default="petkova",
                       choices=["sph-gather", "sph-scatter", "petkova"])
    args.fields = _ask("Fields to interpolate (comma list)", default="rho")

    if args.grid == "uniform":
        args.npx = int(_ask("Cells per axis (npx)", default="128"))
    elif args.grid == "amr":
        args.base_npx = int(_ask("Base grid cells per axis", default="16"))
        args.npx = int(_ask("Finest resolution / cells per axis (npx)",
                            default="128"))
        args.refine = _ask("Refinement criterion", default="mass",
                           choices=["smoothing", "mass", "count"])
    elif args.grid == "voronoi":
        args.voronoi_source = _ask(
            "Voronoi generators from", default="particles",
            choices=["particles", "random", "subsample", "grid"])
        if args.voronoi_source != "particles":
            args.npx = int(_ask("Cells per axis (npx; total npx^3 cells)",
                                default="64"))
    elif args.grid == "projection":
        args.axes = _ask("Image plane", default="xy",
                         choices=["xy", "xz", "yz"])
        args.project = _ask("Projection mode", default="column",
                            choices=["column", "average", "rhocolumn", "slice"])
        args.npx = int(_ask("Pixels per axis (npx)", default="512"))

    default_fmt = {"uniform": "npz", "amr": "ramses", "voronoi": "npz",
                   "projection": "npz"}[args.grid]
    args.output_format = _ask("Output format", default=default_fmt,
                              choices=["npz", "hdf5", "ramses", "arepo",
                                       "cell-table", "native"])
    args.output = _ask("Output file")
    if _ask("Write a SPH-vs-grid diagnostic figure?", default="y",
            choices=["y", "n"]) == "y":
        args.visualise = _ask("  figure path", default=os.path.splitext(
            args.output)[0] + "_visualise.pdf")
        args.visualise_field = _ask("  field to visualise",
                                    default=args.fields.split(",")[0].strip())
        if args.grid != "projection":
            args.visualise_axes = _ask("  slice plane", default="xy",
                                       choices=["xy", "xz", "yz"])
    print("-" * 64)
    return args


# --------------------------------------------------------------------------- #
#  Building the target geometry
# --------------------------------------------------------------------------- #

def amr_max_level(args):
    """AMR refinement depth: use --max-level if given, else derive it from --npx
    (the finest target resolution) and --base-npx, so base_npx*2^level >= npx."""
    if args.max_level is not None:
        return int(args.max_level)
    import math
    return max(0, math.ceil(math.log2(max(args.npx, 1) / args.base_npx)))


def target_domain(args, p):
    """Resolve one authoritative target domain for every grid geometry."""
    if args.box is not None:
        box = np.asarray(args.box, dtype=np.float64).reshape(3)
        lo, hi = -0.5 * box, 0.5 * box
        audit = dict(mode='explicit_box', pad=0.0, lo=lo, hi=hi)
    elif any(np.atleast_1d(p.periodic)):
        box = np.asarray(p.box, dtype=np.float64).reshape(3)
        lo, hi = -0.5 * box, 0.5 * box
        audit = dict(mode='periodic_box', pad=0.0, lo=lo, hi=hi)
    else:
        from sph_interp import open_domain_bounds
        lo, hi, audit = open_domain_bounds(
            p.pos, p.mass, p.rho, p.h, mode=args.open_padding,
            nsmooth=args.nsmooth,
            constant_fraction=args.open_pad_fraction)
    args._domain_audit = audit
    return np.asarray(lo), np.asarray(hi)


def build_target(args, p):
    from sph_interp import (UniformGrid, AMRGrid, VoronoiGrid, Projection2D,
                            refine_smoothing_length, refine_mass, refine_count)
    lo, hi = target_domain(args, p)
    box = hi - lo
    audit = args._domain_audit
    detail = (f" pad={audit['pad']:.6g}/side"
              if audit.get('pad', 0.0) else '')
    print(f"  domain: {audit['mode']}{detail}, lo={np.round(lo, 4)}, "
          f"hi={np.round(hi, 4)}")

    if args.grid == "uniform":
        print(f"  target: UniformGrid {args.npx}^3 over box {np.round(box, 4)}")
        return UniformGrid(lo=lo, hi=hi, npx=args.npx)

    if args.grid == "amr":
        # Quasi-Lagrangian default, matching RAMSES poisson_refine
        # (amr/flag_utils.f90): RAMSES refines a cell when
        #     rho_cell >= m_refine(level) * mass_sph / vol_cell
        # i.e. cell_mass >= m_refine * mass_sph, where mass_sph is the gas mass of a
        # single resolution element and m_refine a per-level factor (the example
        # namelists use 1 for hydro). We take mass_sph = the mean PARTICLE mass and
        # default m_refine = 1, so a cell refines while it holds more than ~one
        # particle's worth of mass -> leaves hold ~1 particle (n_leaves ~ N_part, so
        # the mesh tracks the SPH resolution) and each leaf's deposited mass ~
        # m_particle (near-uniform). The threshold is identical at every level, so
        # refinement stops at ~1 particle per cell (constant mass per cell). The
        # near-uniform masses + quasi-Lagrangian generators also keep the AREPO
        # Voronoi density consistent (mass/V_voronoi recovers rho without the octree-
        # vs-Voronoi volume mismatch a few-particles-per-cell mesh would suffer).
        # Override the absolute threshold with --refine-value (e.g. 2*m_particle for
        # ~2 particles/cell = coarser, like RAMSES m_refine=2).
        mass_sph = float(p.mass.sum()) / max(p.n, 1)
        crit = {"smoothing": refine_smoothing_length(),
                "mass": refine_mass(args.refine_value
                                    if args.refine_value is not None
                                    else mass_sph),
                "count": refine_count(int(args.refine_value)
                                      if args.refine_value is not None
                                      else 1)}[args.refine]
        max_level = amr_max_level(args)
        fine = args.base_npx * (2 ** max_level)
        print(f"  target: AMRGrid base {args.base_npx}^3, max_level "
              f"{max_level}, refine={args.refine}")
        print(f"          adaptive: finest level resolves like {fine}^3 "
              f"(= base_npx x 2^max_level, derived from --npx {args.npx}); "
              "coarse in voids, fine only where the data warrants it")
        g = AMRGrid.build(p, base_npx=args.base_npx, max_level=max_level,
                          criterion=crit, lo=lo, hi=hi)
        dense = fine ** 3
        print(f"          -> {g.ncell} leaf cells "
              f"({dense/max(g.ncell,1):.0f}x fewer than a dense {fine}^3 grid)")
        return g

    if args.grid == "voronoi":
        # Periodic box -> bounds ARE the period (ghost images give exact periodic
        # cells, Sum volume = box volume); non-periodic -> pad past the box.
        vper = p.periodic if any(np.atleast_1d(p.periodic)) else False
        if vper:
            bounds = [[lo[a], hi[a]] for a in range(3)]
            ptag = " (periodic, ghost images)"
        else:
            pad = args.voronoi_pad
            bounds = [[lo[a] - pad, hi[a] + pad] for a in range(3)]
            ptag = ""
        if args.voronoi_source == "particles":
            print(f"  target: VoronoiGrid from {p.n} particles{ptag}")
            return VoronoiGrid.from_particles(p, bounds=bounds, periodic=vper)
        if args.voronoi_source == "grid":
            nx = args.npx
            axes = [(lo[a] + (np.arange(nx) + 0.5) * (box[a] / nx))
                    for a in range(3)]
            X, Y, Z = np.meshgrid(*axes, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            print(f"  target: VoronoiGrid from a {nx}^3 = {nx**3} regular "
                  f"lattice (cubic cells){ptag}")
            return VoronoiGrid.from_points(pts, bounds=bounds, periodic=vper)
        ncells = args.npx ** 3                     # npx per-axis-equivalent count
        rng = np.random.default_rng(args.seed)
        if args.voronoi_source == "random":
            pts = rng.uniform(lo + 1e-9, hi - 1e-9, size=(ncells, 3))
            print(f"  target: VoronoiGrid from {ncells} (={args.npx}^3) random "
                  f"generators{ptag}")
        else:  # subsample
            n = min(ncells, p.n)
            idx = rng.choice(p.n, size=n, replace=False)
            pts = p.pos[idx]
            print(f"  target: VoronoiGrid from {n} subsampled particles "
                  f"(target {args.npx}^3={ncells}){ptag}")
        return VoronoiGrid.from_points(pts, bounds=bounds, periodic=vper)

    if args.grid == "projection":
        a1, a2 = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.axes]
        print(f"  target: Projection2D {args.axes} {args.npx}^2, "
              f"mode={args.project}")
        return Projection2D(axis1=a1, axis2=a2, lo=lo[[a1, a2]],
                            hi=hi[[a1, a2]], npx=args.npx,
                            mode=args.project)

    raise ValueError(f"unknown grid {args.grid!r}")


# --------------------------------------------------------------------------- #
#  Writing the result
# --------------------------------------------------------------------------- #

def _collect(res):
    """A serialization-ready dict for any target geometry."""
    from sph_interp import UniformGrid, AMRGrid, VoronoiGrid, Projection2D
    grid = res.target
    out = {"_grid": type(grid).__name__, "_method": res.method}
    fields = {k: v for k, v in res.data.items()}

    if isinstance(grid, UniformGrid):
        cx, cy, cz = grid.centres()
        out.update(x=cx, y=cy, z=cz, lo=grid.lo, hi=grid.hi,
                   npx=np.array(grid.npx))
        out.update(fields)                       # 3D arrays as-is
    elif isinstance(grid, Projection2D):
        ex, ey = grid.centres()
        out.update(x=ex, y=ey, lo=grid.lo, hi=grid.hi,
                   axis1=grid.axis1, axis2=grid.axis2, mode=grid.mode)
        out.update(fields)                       # 2D arrays as-is
    elif isinstance(grid, AMRGrid):
        c = grid.centres()
        out.update(x=c[:, 0], y=c[:, 1], z=c[:, 2], lo=grid.lo, hi=grid.hi,
                   dx=grid.cell_size[:, 0],
                   dy=grid.cell_size[:, 1], dz=grid.cell_size[:, 2],
                   volume=grid.volumes(), level=grid.cell_level,
                   base=grid.cell_base)
        out.update(fields)                       # (Ncell,) arrays
    elif isinstance(grid, VoronoiGrid):
        c = grid.centres()
        out.update(x=c[:, 0], y=c[:, 1], z=c[:, 2], volume=grid.volumes())
        out.update(fields)                       # (Ncell,) arrays
    return out


def write_output(res, path, fmt, args, p):
    grid_name = type(res.target).__name__

    if fmt in ("ramses", "native"):
        from sph_interp import to_ramses, RAMSES_HYDRO_VARS
        rfmt = "native" if fmt == "native" else "hdf5"
        hydro = (args.hydro_fields.split(",") if args.hydro_fields
                 else [f for f in RAMSES_HYDRO_VARS if f in res.data])
        if not hydro:
            raise SystemExit(
                "ramses: none of the RAMSES hydro variables "
                f"{RAMSES_HYDRO_VARS} are in the interpolated fields "
                f"{sorted(res.data)}; pass --hydro-fields or add them to "
                "--fields (e.g. --fields rho,vx,vy,vz).")
        print(f"  writing RAMSES ({rfmt}) hydro vars {hydro} -> {path}")
        return to_ramses(res, path, hydro_fields=tuple(hydro),
                         gamma=args.gamma, boxlen=args.boxlen, fmt=rfmt)

    if fmt == "arepo":
        from sph_interp import to_arepo
        abox = (np.repeat(args.boxlen, 3) if args.boxlen is not None else p.box)
        aper = bool(np.any(np.atleast_1d(p.periodic)))
        print(f"  writing AREPO moving-mesh IC -> {path}")
        return to_arepo(res, path, tu_fac=args.tu_fac, box=abox, periodic=aper)

    if fmt == "cell-table":
        from sph_interp import to_cell_table
        table = to_cell_table(res)
        print(f"  writing cell table ({len(next(iter(table.values())))} rows) "
              f"-> {path}")
        np.savez(path, **table)
        return path

    payload = _collect(res)
    if fmt == "hdf5":
        import h5py
        print(f"  writing HDF5 -> {path}")
        with h5py.File(path, "w") as f:
            for k, v in payload.items():
                if isinstance(v, str):
                    f.attrs[k] = v
                else:
                    f.create_dataset(k, data=np.asarray(v))
        return path

    # default: npz (works for every target)
    print(f"  writing npz -> {path}")
    np.savez(path, **{k: v for k, v in payload.items()
                      if not isinstance(v, str)},
             **{f"attr_{k}": v for k, v in payload.items()
                if isinstance(v, str)})
    return path


# --------------------------------------------------------------------------- #
#  Mass-conservation report (a quick sanity line)
# --------------------------------------------------------------------------- #

def report_mass(res, p):
    from sph_interp import UniformGrid, AMRGrid, VoronoiGrid
    grid = res.target
    pm = float(p.mass.sum())
    if "_mass" in res.data:                       # petkova exposes conserved mass
        gm = float(np.asarray(res.data["_mass"]).sum())
    elif isinstance(grid, UniformGrid) and "rho" in res.data:
        gm = float(res.data["rho"].sum() * grid.cell_volume)
    elif isinstance(grid, (AMRGrid, VoronoiGrid)) and "rho" in res.data:
        gm = float((res.data["rho"] * grid.volumes()).sum())
    else:
        return
    # sph-scatter and petkova are mass-conserving; only the sph-gather estimate
    # (kernel density AT the cell centre) is approximate.
    tag = ("  [approx: sph-gather density estimate, not mass-conserving; "
           "edge/padding cells inflate sum(rho*vol)]"
           if res.method == "sph-gather" else "")
    print(f"  mass check: grid {gm:.6g} vs particles {pm:.6g}  "
          f"({100*(gm/pm - 1):+.3f}%){tag}")


# --------------------------------------------------------------------------- #
#  Magnetic components (--magnetic-projection)
# --------------------------------------------------------------------------- #

def magnetic_component_names(fields):
    """The (Bx,By,Bz) spellings from --fields, or None unless all three exist."""
    m = {f.lower(): f for f in fields}
    if all(k in m for k in ("bx", "by", "bz")):
        return (m["bx"], m["by"], m["bz"])
    return None


def conservative_cell_vector(p, target, names):
    """Exact volume-carrier deposit: cell value = sum_i V_i f_i w_ic / V_cell.

    This is the conservative m*(f/rho) Petkova deposit used by every magnetic
    mode's cell field (sum_cells content == sum_i V_i f_i to machine
    precision), as opposed to the kernel-volume-weighted average the ordinary
    field path produces.
    """
    from sph_interp import Particles, UniformGrid, interpolate
    carriers, keys = {}, []
    for nm in names:
        k = f"__mag_{nm}"
        carriers[k] = np.asarray(p.values[nm], dtype=np.float64) / p.rho
        keys.append(k)
    cp = Particles(pos=p.pos, mass=p.mass, rho=p.rho, h=p.h, values=carriers,
                   box=p.box, periodic=p.periodic)
    dep = interpolate(cp, target, method="petkova", values=keys,
                      conservative_fields=True)
    vol = (target.cell_volume if isinstance(target, UniformGrid)
           else target.volumes())
    return [np.asarray(dep.data[f"_content_{k}"], dtype=np.float64) / vol
            for k in keys]


def deposit_magnetic(args, p, target, res, bnames):
    """Fill res.data[Bx,By,Bz] (+ faces / A) per --magnetic-projection."""
    from sph_interp import UniformGrid
    mode = args.magnetic_projection
    per = bool(np.all(np.atleast_1d(p.periodic)))

    if mode == "none":
        print(f"  magnetic: conservative volume-carrier deposit of {bnames} "
              "(no projection; clean divergence in the consumer's operator "
              "with mhd_divBclean)")
        for nm, cell in zip(bnames,
                            conservative_cell_vector(p, target, bnames)):
            res.data[nm] = cell
        return

    if mode == "hodge":
        if isinstance(target, UniformGrid):
            if per:
                from sph_interp import capture_periodic_face_field
                table, audit, _cell = capture_periodic_face_field(
                    p, target, fields=bnames)
            else:
                from sph_interp import capture_open_face_field
                try:
                    table, audit, _cell = capture_open_face_field(
                        p, target, fields=bnames, background="particle-mean")
                except ValueError as err:
                    if "support" in str(err):
                        raise SystemExit(
                            f"hodge: {err}\n  (the open face capture needs a "
                            "support-complete grid: rerun with --open-padding "
                            "full_pad, or an explicit --box containing "
                            "pos +/- 2h)")
                    raise
        else:                                    # open AMRGrid
            if per:
                raise SystemExit(
                    "--magnetic-projection hodge on a PERIODIC AMR grid is "
                    "not implemented (uniform periodic, uniform open, or "
                    "open amr)")
            from sph_interp import capture_open_amr_face_field
            table, audit, _cell = capture_open_amr_face_field(
                p, target, fields=bnames, background="particle-mean",
                normal_trace="nearest", boundary_flux="balance",
                allow_clipped_support=True)
        cc = table.cell_centred()
        for k, nm in enumerate(bnames):
            res.data[nm] = cc[..., k]
        for attr, out in (("Bx_face", f"{bnames[0]}_face"),
                          ("By_face", f"{bnames[1]}_face"),
                          ("Bz_face", f"{bnames[2]}_face")):
            if hasattr(table, attr):             # uniform tables expose faces
                res.data[out] = getattr(table, attr)
        args._magnetic_audit = audit
        print(f"  magnetic: hodge face capture + solenoidal projection "
              f"({'periodic' if per else 'open'} "
              f"{'uniform' if isinstance(target, UniformGrid) else 'AMR'}; "
              "cell-centred B + face arrays where available)")
        return

    # vector-potential (uniform only; enforced at parse time)
    from sph_interp import capture_uniform_vector_potential
    if per and args.a_method == "biot-savart":
        print("  WARNING: biot-savart uses the free-space kernel; on a "
              "periodic box prefer --a-method spectral")
    table = capture_uniform_vector_potential(
        p, target, fields=bnames, a_method=args.a_method)
    cc = table.cell_centred()
    for k, nm in enumerate(bnames):
        res.data[nm] = cc[..., k]
    for k, nm in enumerate(("Ax", "Ay", "Az")):  # A as first-class channels
        res.data[nm] = table.cell_A[..., k]
    a = table.audit
    print(f"  magnetic: vector-potential ({args.a_method}) "
          f"Bbar={np.round(table.Bbar, 6)}; round-trip @particles "
          f"rel-L2 {a['roundtrip_B_rel_l2']:.3f}, "
          f"E/E0 {a['roundtrip_over_source']:.3f} "
          "(div B == 0 pointwise by construction)")


def report_magnetic(res, p, bnames):
    """Grid-vs-particle Sum(V*B) per component (the material ledger)."""
    from sph_interp import UniformGrid, AMRGrid
    grid = res.target
    if isinstance(grid, UniformGrid):
        gsum = [float(np.sum(res.data[nm]) * grid.cell_volume)
                for nm in bnames]
    elif isinstance(grid, AMRGrid):
        v = grid.volumes()
        gsum = [float(np.sum(np.asarray(res.data[nm]) * v)) for nm in bnames]
    else:
        return
    V = p.mass / p.rho
    psum = [float(np.sum(V * np.asarray(p.values[nm], np.float64)))
            for nm in bnames]
    pairs = "  ".join(f"{nm}: grid {g:.6g} vs particles {q:.6g}"
                      for nm, g, q in zip(bnames, gsum, psum))
    print(f"  magnetic integral check (Sum V*B): {pairs}")
    print("    (NB with a particle-mean background -- hodge/open -- the grid "
          "integral counts Bbar over the WHOLE box, not just the material "
          "volume, so a mismatch there is bookkeeping, not loss)")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def build_parser():
    pa = argparse.ArgumentParser(
        prog="interpolate_to_grid",
        description="Interpolate a tipsy SPH snapshot onto a grid (sph_interp). "
                    "Run with no arguments for an interactive prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("input", nargs="?", help="input tipsy snapshot")
    pa.add_argument("output", nargs="?", help="output file/prefix")

    pa.add_argument("--grid", choices=["uniform", "amr", "voronoi", "projection"],
                    default="uniform", help="target geometry")
    pa.add_argument("--method",
                    choices=["sph-gather", "sph-scatter", "petkova", "sph"],
                    default="petkova",
                    help="petkova=exact/mass-conserving (default), sph=fast/approx")
    pa.add_argument("--fields", default="rho",
                    help="comma list of fields to interpolate (rho,Bmag,vx,...)")
    pa.add_argument("--output-format",
                    choices=["npz", "hdf5", "ramses", "arepo", "cell-table",
                             "native"],
                    default=None, help="default: npz (ramses for --grid amr). "
                    "arepo=moving-mesh IC (cell centres -> generators).")

    # snapshot / domain
    pa.add_argument("--nsmooth", type=int, default=64,
                    help="neighbours for h when no smoothlength aux")
    pa.add_argument("--box", type=float, nargs="+", default=None, metavar="L",
                    help="exact target box lengths: one value for a cube or "
                         "three values LX LY LZ; overrides open padding")
    pa.add_argument("--open-padding", "--raw-padding",
                    choices=["full_pad", "rho_edge_pad", "constant_pad"],
                    default="rho_edge_pad",
                    help="non-periodic target bounds: max(2h), 2*h_edge from "
                         "the input particle count, or 0.1*max(span) per side")
    pa.add_argument("--open-pad-fraction", type=float, default=0.1,
                    help="fraction of max particle span for constant_pad")
    g = pa.add_mutually_exclusive_group()
    g.add_argument("--periodic", dest="periodic", action="store_true")
    g.add_argument("--no-periodic", "--open", dest="periodic",
                   action="store_false")
    pa.set_defaults(periodic=None)

    # resolution (single knob): cells per axis
    pa.add_argument("--npx", type=int, default=128,
                    help="cells per axis (THE resolution knob). uniform/"
                         "projection: grid/image resolution. amr: the FINEST "
                         "target resolution -> max_level is derived so "
                         "base_npx x 2^max_level >= npx (AMR is adaptive). "
                         "voronoi: lattice size for --voronoi-source grid.")
    # projection
    pa.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy",
                    help="projection image plane")
    pa.add_argument("--project",
                    choices=["column", "average", "rhocolumn", "slice"],
                    default="column", help="projection mode")
    # amr
    pa.add_argument("--base-npx", type=int, default=16, help="AMR base grid/axis")
    pa.add_argument("--max-level", type=int, default=None,
                    help="AMR max level (default: derived from --npx and "
                         "--base-npx). Set to override the npx-derived depth.")
    pa.add_argument("--refine", choices=["smoothing", "mass", "count"],
                    default="mass", help="AMR refinement criterion (default: "
                    "mass = quasi-Lagrangian, like RAMSES: refine a cell whose "
                    "gas mass exceeds --refine-value, default mean base-cell mass)")
    pa.add_argument("--refine-value", type=float, default=None,
                    help="threshold for --refine mass/count")
    # voronoi
    pa.add_argument("--voronoi-source",
                    choices=["particles", "random", "subsample", "grid"],
                    default="particles",
                    help="Voronoi generators: particles (one cell per particle, "
                         "resolution fixed by the data), or subsample / random / "
                         "grid which use --npx (npx^3 cells: subsampled particles, "
                         "random points, or a regular lattice)")
    pa.add_argument("--voronoi-pad", type=float, default=0.05,
                    help="pad cell domain past the box (keeps boundary kernels in)")
    pa.add_argument("--seed", type=int, default=0, help="RNG seed (voronoi)")

    # magnetic components (Bx,By,Bz in --fields)
    pa.add_argument("--magnetic-projection",
                    choices=["none", "hodge", "vector-potential"],
                    default="none",
                    help="treatment of the magnetic triplet Bx,By,Bz (all "
                         "three in --fields). none (default): conservative "
                         "Petkova m*(B/rho) volume-carrier deposit, NO "
                         "projection -- divergence is handled afterwards in "
                         "the consumer's own operator (mhd_divBclean). "
                         "hodge: shared-face capture + Hodge/solenoidal "
                         "projection (RT0 face table, the RAMSES-export "
                         "intermediate); uniform (periodic or open) and open "
                         "amr. vector-potential: capture A, B = "
                         "curl(interp(A)) -> div B == 0 pointwise; uniform "
                         "grids only, also stores Ax,Ay,Az cell channels.")
    pa.add_argument("--a-method", choices=["spectral", "biot-savart"],
                    default="spectral",
                    help="vector-potential A derivation: spectral Coulomb-"
                         "gauge ik x B/k^2 on the (zero-padded open) grid, or "
                         "the meshless parts-integrated Biot-Savart particle "
                         "sum Petkova-deposited to the grid")

    # visualisation
    pa.add_argument("--visualise", "--visualize", dest="visualise", nargs="?",
                    const="", default=None,
                    help="write a diagnostic figure comparing the SPH particle "
                         "field to the interpolated grid (geometry-aware slice, "
                         "density PDFs, per-particle grid-vs-SPH scatter, mass "
                         "conservation). Optionally pass a path; default is "
                         "<output>_visualise.pdf.")
    pa.add_argument("--visualise-field", default=None,
                    help="which field to visualise (default: first of --fields)")
    pa.add_argument("--visualise-axes", choices=["xy", "xz", "yz"], default="xy",
                    help="slice/projection plane for --visualise (3D targets)")

    # ramses
    pa.add_argument("--hydro-fields", default=None,
                    help="ordered RAMSES hydro vars (default rho,vx,vy,vz,P "
                         "intersected with --fields)")
    pa.add_argument("--gamma", type=float, default=5.0 / 3.0)
    pa.add_argument("--boxlen", type=float, default=None)

    # arepo
    pa.add_argument("--voronoi-particles", action="store_true",
                    help="AREPO output only: write the SPH particles DIRECTLY as "
                         "generators (1:1 copy of mass + v/u/B, no interpolation; "
                         "AREPO tessellates). The most faithful conversion. "
                         "Without this, --output-format arepo deposits onto the "
                         "chosen --grid first.")
    pa.add_argument("--tu-fac", type=float, default=None,
                    help="dTuFac for the AREPO IC: specific internal energy "
                         "u = dTuFac * T (tipsy stores temperature). Default: "
                         "read dTuFac from the run .log (read_tufac); pass a "
                         "value to override.")
    return pa


def main(argv=None):
    pa = build_parser()
    args = pa.parse_args(argv)

    # No positional args at all -> interactive mode.
    if args.input is None and args.output is None:
        args = interactive(args)
    elif args.input is None or args.output is None:
        pa.error("provide BOTH input and output, or NEITHER (interactive)")

    if args.output_format is None:
        args.output_format = "ramses" if args.grid == "amr" else "npz"

    from sph_interp import from_tipsy, interpolate

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())

    # magnetic triplet handling (--magnetic-projection)
    bnames = magnetic_component_names(fields)
    if args.magnetic_projection != "none":
        if bnames is None:
            pa.error("--magnetic-projection requires --fields to include all "
                     "of Bx,By,Bz")
        if args.grid not in ("uniform", "amr"):
            pa.error("--magnetic-projection supports --grid uniform or amr")
        if args.grid == "amr" and args.magnetic_projection == "vector-potential":
            pa.error("--magnetic-projection vector-potential is uniform-grid "
                     "only (AMR-native A capture is deferred); use --grid "
                     "uniform, or --magnetic-projection none/hodge for amr")
    # The magnetic path replaces the ordinary deposit of Bx,By,Bz whenever a
    # projection is requested, or for the default petkova method (mode 'none'
    # = the conservative volume-carrier deposit). gather/scatter with 'none'
    # keep the legacy approximate path.
    magnetic = (bnames is not None and args.grid in ("uniform", "amr")
                and (args.magnetic_projection != "none"
                     or args.method == "petkova"))

    # AREPO ICs need a temperature field (u = dTuFac*T) -> auto-include one, and
    # resolve dTuFac from the run .log unless the user gave --tu-fac.
    if args.output_format == "arepo":
        from sph_interp.export import AREPO_TEMPERATURE
        if not any(f.lower() in AREPO_TEMPERATURE for f in fields):
            fields = fields + ("temp",)
            print("  (arepo) added 'temp' to fields for internal energy")
        if args.tu_fac is None:
            # Prefer the .param unit system (authoritative for an IC we generated;
            # the IC writer stored temperature = u/dTuFac but writes no dGasConst
            # line, so the .log path defaults to a wrong 1.0). Fall back to the
            # run .log, else 1.0.
            from sph_interp.particles import read_param_tufac
            tf = read_param_tufac(args.input)
            if tf is not None:
                args.tu_fac = tf
                print(f"  (arepo) dTuFac from .param units: {args.tu_fac:g} "
                      "(override with --tu-fac)")
            else:
                from IC_analysis_general import read_tufac
                args.tu_fac = read_tufac(args.input)
                print(f"  (arepo) dTuFac from run .log: {args.tu_fac:g} "
                      "(override with --tu-fac)")

    print(f"reading {args.input}  fields={fields}")
    if args.box is not None:
        if len(args.box) == 1:
            args.box = np.repeat(args.box[0], 3)
        elif len(args.box) == 3:
            args.box = np.asarray(args.box, dtype=np.float64)
        else:
            pa.error('--box takes either one length or LX LY LZ')
        if np.any(np.asarray(args.box) <= 0.0):
            pa.error('--box lengths must be positive')
    box = None if args.box is None else np.asarray(args.box, dtype=np.float64)
    p = from_tipsy(args.input, fields=fields, nsmooth=args.nsmooth,
                   periodic=(True if args.periodic is None else args.periodic),
                   box=box)
    # An explicit CLI boundary choice overrides snapshot metadata.  With no
    # choice, from_tipsy keeps the .param periodicity as the authority.
    if args.periodic is not None:
        p.periodic = (args.periodic,) * 3
    print(f"  {p.n} gas particles, box {np.round(p.box, 4)}, "
          f"periodic={p.periodic}")

    # AREPO copy path: write the particles straight to a moving-mesh IC, no
    # tessellation / deposit on our side (AREPO builds the Voronoi mesh).
    if args.voronoi_particles:
        if args.output_format != "arepo":
            pa.error("--voronoi-particles only applies to --output-format arepo")
        from sph_interp import particles_to_arepo
        abox = (np.repeat(args.boxlen, 3) if args.boxlen is not None else p.box)
        print("  AREPO copy path: particles -> generators (no interpolation)")
        particles_to_arepo(p, args.output, tu_fac=args.tu_fac, box=abox)
        print("done.")
        return

    target = build_target(args, p)
    print(f"  depositing with method={args.method} ...")
    deposit_values = [f for f in fields if not (magnetic and f in bnames)]
    if deposit_values or not magnetic:
        res = interpolate(p, target, method=args.method, values=deposit_values)
        report_mass(res, p)
    else:                      # magnetic-only field list: nothing else to deposit
        from sph_interp import GridResult
        res = GridResult({}, None, target, args.method)
    if magnetic:
        deposit_magnetic(args, p, target, res, bnames)
        report_magnetic(res, p, bnames)

    write_output(res, args.output, args.output_format, args, p)

    if args.visualise is not None:
        from sph_interp import visualise
        vfield = args.visualise_field or fields[0]
        vpath = args.visualise or os.path.splitext(args.output)[0] + "_visualise.pdf"
        plane = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.visualise_axes]
        print(f"  visualising field '{vfield}' ({args.visualise_axes}) ...")
        visualise(res, p, field=vfield, out=vpath, plane=plane)

    print("done.")


if __name__ == "__main__":
    main()
