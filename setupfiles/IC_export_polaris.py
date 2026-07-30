"""tipsy -> sph_interp AMR octree -> POLARIS synchrotron grid (M4 of
POLARIS_EXPORT_PLAN.md §5.2).

The physics-aware driver: takes a galaxy cutout (from `_polaris_locate_galaxy.py`),
rotates it into the disc frame, builds a CUBIC power-of-two AMR octree, deposits
the hydro fields conservatively and B divergence-free, synthesises the electron
fields, and writes the POLARIS grid + a matching `POLARIS.cmd` + a sidecar JSON
recording every factor and model parameter.

Deliberately separate from `IC_interpolate_to_grid.py`: that CLI is generic and
unit-free, and the synchrotron fields need a physics vocabulary it has no business
carrying.

Cosmology: the cutout stores comoving positions and the expansion factor `a`.
Physical quantities follow  L_phys = a L_com,  rho_phys = a^-3 rho_com,
B_phys = a^-2 B_com  (flux freezing). At a = 1 these are all no-ops, which is why
the z = 0 run is the unambiguous one; for a < 1 the B convention in particular
should be confirmed against the run's own MHD implementation.

Usage:
  python IC_export_polaris.py <cut.npz> --out-dir polaris_7dwarf --base-npx 16 \
      --max-level 6 --cr-model exp
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# This script stays in setupfiles/ (it is an IC_* exporter and needs sph_interp
# and readtipsy), but the POLARIS physics modules it imports - polaris_units,
# polaris_electrons, sofia7_fir, thb_readout - live in utilities/polaris/.
# Resolve BOTH from this file's location so the script runs from any directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_POLARIS_UTILS = os.path.normpath(
    os.path.join(_HERE, '..', 'utilities', 'polaris'))
for _p in (_HERE, _POLARIS_UTILS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
#  disc frame
# --------------------------------------------------------------------------- #

def disc_frame(pos, vel, mass, rho, T, rho_pct=90.0, t_max=3.0e4):
    """Rotation matrix taking the galaxy's angular-momentum axis onto +z.

    Uses the COLD DENSE gas (above `rho_pct` in density, below `t_max`), which
    traces the disc rather than the hot halo or infalling filaments. Returns
    (R, axis): `pos @ R.T` puts the disc in the x-y plane.
    """
    sel = (rho > np.percentile(rho, rho_pct)) & (T < t_max)
    if int(sel.sum()) < 100:                    # too few: fall back to all gas
        sel = np.ones(len(mass), dtype=bool)
    L = np.cross(pos[sel], vel[sel] * mass[sel, None]).sum(axis=0)
    n = np.linalg.norm(L)
    if n == 0:
        return np.eye(3), np.array([0.0, 0.0, 1.0])
    zhat = L / n
    # any vector not parallel to zhat -> orthonormal triad
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, zhat)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    xhat = np.cross(tmp, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.stack([xhat, yhat, zhat]), zhat


# --------------------------------------------------------------------------- #
#  POLARIS command file
# --------------------------------------------------------------------------- #

CMD_TEMPLATE = """<common>
\t<nr_threads>\t{nthreads}
</common>

<task> 1
\t<cmd>\t\t\tCMD_SYNCHROTRON

{detectors}
\t<source_background nr_photons = "1">\t0 0 0 0 0 0

\t<conv_dens>\t1.0
\t<conv_len>\t1.0
\t<conv_mag>\t1.0
\t<conv_vel>\t1.0

\t<path_grid>\t"{grid}"
\t<path_out>\t"{out}"
</task>
"""


def write_cmd(path, grid_path, out_dir, wavelengths_m, distance_m, npix=511,
              nthreads=-1, sidelength=-1.0):
    """A `POLARIS.cmd` with a face-on and an edge-on detector.

    `<detector_sync>` argument order (CommandParser.cpp:1941-2035):
      wl_min wl_max nr_wl source_id rot_angle_1 rot_angle_2 [distance]
      [sidelength_x] [sidelength_y]
    Face-on is (0, 0); edge-on is a 90 deg rotation about the first angle.
    sidelength -1 = the full cube side.
    """
    wl = np.atleast_1d(np.asarray(wavelengths_m, dtype=np.float64))
    dets = []
    for label, rot1, rot2 in (("face-on", 0.0, 0.0), ("edge-on", 90.0, 0.0)):
        dets.append(
            f'\t# {label}\n'
            f'\t<detector_sync nr_pixel = "{npix}*{npix}">\t'
            f'{wl.min():.6e}\t{wl.max():.6e}\t{wl.size}\t1\t'
            f'{rot1:g}\t{rot2:g}\t{distance_m:.6e}\t'
            f'{sidelength:g}\t{sidelength:g}')
    txt = CMD_TEMPLATE.format(nthreads=nthreads, detectors="\n\n".join(dets),
                              grid=grid_path, out=out_dir.rstrip("/") + "/")
    with open(path, "w") as fh:
        fh.write(txt)
    return path


# --------------------------------------------------------------------------- #

def main():
    pa = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pa.add_argument("cut", help="cutout .npz from _polaris_locate_galaxy.py")
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--base-npx", type=int, default=16,
                    help="AMR base grid per axis; MUST be a power of two "
                         "(POLARIS needs one octree root)")
    pa.add_argument("--grade", dest="grade", action="store_true", default=True,
                    help="2:1-balance the leaf set (DEFAULT on, as the "
                         "reconstruction capture does): no leaf touches one more "
                         "than a level coarser, so a ray never crosses a "
                         "many-level jump that smears fine values over a whole "
                         "coarse cell")
    pa.add_argument("--no-grade", dest="grade", action="store_false")
    pa.add_argument("--max-level", type=int, default=12,
                    help="refinement depth. With --refine count this is only a "
                         "finite BACKSTOP for the degenerate case the criterion "
                         "cannot resolve (>part-per-cell coincident particles), "
                         "exactly as in the reconstruction capture -- the "
                         "criterion self-terminates, so set it generous and let "
                         "it decide. With --refine smoothing it IS the "
                         "resolution knob (finest cell = cube/(base_npx*2^L)).")
    pa.add_argument("--refine-factor", type=float, default=0.5,
                    help="refine until cell <= factor*h. SOFIA VII resolves the "
                         "octree at DOUBLE the simulation resolution to avoid "
                         "aliasing between the two grids -> 0.5")
    pa.add_argument("--refine", choices=["count", "mass", "smoothing"],
                    default="count",
                    help="refinement criterion. 'count' (DEFAULT) splits a cell "
                         "holding more than --part-per-cell particles, matching "
                         "the reconstruction pipeline; it keeps the mesh coarser "
                         "than the particle sampling everywhere, so particle "
                         "noise is not encoded as target structure. 'mass' is "
                         "the same rule by summed mass. 'smoothing' is the old "
                         "cell<=factor*h rule -- SOFIA VII's anti-aliasing "
                         "intent, but it leaves 86%% of leaves empty on an "
                         "evolved galaxy and blows up the THB operator memory.")
    pa.add_argument("--part-per-cell", type=int, default=8,
                    help="particles per cell for --refine count/mass "
                         "(reconstruct's default is 8)")
    pa.add_argument("--refine-min-count", type=int, default=1,
                    help="--refine smoothing only: gate the split on the cell "
                         "holding MORE than this many particles "
                         "(sph_interp.refine_occupied). 0 disables the gate.")
    pa.add_argument("--method", default="petkova",
                    choices=["petkova", "sph-gather", "sph-scatter"])
    pa.add_argument("--magnetic-projection", default="thb",
                    choices=["thb", "hodge", "none"],
                    help="'thb' (DEFAULT) = THB De Rham curl-fit: fit a spline "
                         "vector potential A and read B = curl A at the cell "
                         "centres, so div B = 0 pointwise and the field is C2 "
                         "with no C0 seams -- which is what a ray-tracer "
                         "integrating along sight lines wants. 'hodge' = the "
                         "open-AMR Hodge solve, kept as the A/B control. "
                         "'none' = raw deposit: POLARIS's polarized-RT solver "
                         "STALLS on a non-div-free field (see the plan 8b), so "
                         "it is a deposit sanity check only, never a run.")
    pa.add_argument("--thb-rtol", type=float, default=1e-6,
                    help="CG tolerance of the THB curl-fit")
    pa.add_argument("--thb-dtype", choices=["float32", "float64"],
                    default="float32",
                    help="THB operator dtype; float32 halves the (Q x nnz) "
                         "memory wall and costs ~1e-7 in the readout")
    pa.add_argument("--cr-model", default="exp",
                    choices=["exp", "pEq", "density", "constant"])
    pa.add_argument("--ionization", default="auto",
                    help="'auto' uses the run's HI/HeI/HeII auxes when present "
                         "(the simulation's own non-equilibrium state), else the "
                         "temperature ramp; or force one of "
                         "temperature/constant/saha")
    pa.add_argument("--cr-zeta", type=float, default=None,
                    help="unused by exp/pEq; kept for the energy-density path")
    pa.add_argument("--r-gal-kpc", type=float, default=None,
                    help="exp disc scale length [kpc]; default = measured from "
                         "the cold gas (NOT the paper's Milky Way 8 kpc)")
    pa.add_argument("--h-gal-kpc", type=float, default=None,
                    help="exp disc scale height [kpc]; default = measured")
    pa.add_argument("--cr-p", type=float, default=2.3)
    pa.add_argument("--cr-mask-temp", type=float, default=None)
    pa.add_argument("--cr-mask-density", type=float, default=None)
    pa.add_argument("--wavelength-cm", type=float, nargs="+", default=[6.2],
                    help="detector wavelengths [cm]; SOFIA VII uses 6.2 cm "
                         "(4.8 GHz)")
    pa.add_argument("--distance-mpc", type=float, default=10.0,
                    help="observer distance [Mpc]; SOFIA VII uses 10")
    pa.add_argument("--npix", type=int, default=511)
    pa.add_argument("--no-rotate", action="store_true",
                    help="skip the disc-frame rotation (debugging)")
    # --- FIR dust polarimetry (SOFIA VII 3.1) -- no POLARIS involved ---------
    pa.add_argument("--no-fir", action="store_true",
                    help="skip the geometric FIR dust maps")
    pa.add_argument("--fir-npix", type=int, default=512,
                    help="FIR map size; must be base_npx * 2**j (default 512)")
    pa.add_argument("--eta-dm", type=float, default=None,
                    help="dust-to-METAL mass ratio (SOFIA VII: 0.4)")
    pa.add_argument("--p0-fir", type=float, default=None,
                    help="maximum FIR polarization fraction (Planck: 0.25)")
    pa.add_argument("--t-sublimation", type=float, default=None,
                    help="dust sublimation temperature in f_cut [K] (1500)")
    pa.add_argument("--metallicity", type=float, default=0.02,
                    help="flat Z fallback when the cutout carries no 'metals'")
    args = pa.parse_args()
    import sofia7_fir as _s7
    if args.eta_dm is None:
        args.eta_dm = _s7.ETA_DUST_TO_METAL
    if args.p0_fir is None:
        args.p0_fir = _s7.P0_FIR
    if args.t_sublimation is None:
        args.t_sublimation = _s7.T_SUBLIMATION_K

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sph_interp import (Particles, AMRGrid, interpolate, to_polaris,
                            cubic_domain_bounds, refine_smoothing_length,
                            capture_open_amr_face_field)
    from polaris_units import polaris_units
    import polaris_electrons as pel

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.cut, allow_pickle=True)
    a = float(d["a"])
    kpc_unit = float(d["dKpcUnit"])
    msol_unit = float(d["dMsolUnit"])
    mu_param = (float(d["param_dMeanMolWeight"])
                if "param_dMeanMolWeight" in d.files else None)
    print(f"cutout {args.cut}")
    print(f"  a = {a:.6f} (z = {1/a - 1:.4f}), {len(d['mass']):,} gas particles, "
          f"+-{float(d['radius_kpc']):g} physical kpc")

    # ---- units: code -> SI, with the cosmological a-factors ------------------
    u = polaris_units(dMsolUnit=msol_unit, dKpcUnit=kpc_unit, mu=mu_param)
    length_m = u.length_m * a                      # comoving -> physical
    density_kgm3 = u.density_kgm3 / a ** 3
    b_tesla = u.b_tesla / a ** 2                   # flux freezing
    velocity_ms = u.velocity_ms                    # peculiar v: see the note
    print(f"  units ({u.source}): 1 code length = {length_m:.4e} m physical, "
          f"rho x {density_kgm3:.4e}, B x {b_tesla:.4e} T")
    for w in u.warnings:
        print(f"  WARNING: {w}")
    if abs(a - 1.0) > 1e-6:
        print("  WARNING: a != 1 -- the comoving->physical exponents "
              "(rho a^-3, B a^-2) should be confirmed against the run's MHD "
              "implementation before trusting absolute values.")

    # ---- disc frame ---------------------------------------------------------
    pos, mass, rho, T, h = (d["pos"], d["mass"], d["rho"], d["T"], d["h"])
    vel = np.stack([d["vx"], d["vy"], d["vz"]], axis=1)
    B = d["B"]
    if args.no_rotate:
        R, axis = np.eye(3), np.array([0.0, 0.0, 1.0])
    else:
        R, axis = disc_frame(pos, vel, mass, rho, T)
        pos = pos @ R.T
        vel = vel @ R.T
        B = B @ R.T
        print(f"  disc axis {np.round(axis, 4)} rotated onto +z")

    # ---- cubic, origin-centred domain --------------------------------------
    # Pad by the largest compact support BEFORE squaring off: the open-AMR Hodge
    # solve refuses a domain that clips any particle's 2h support (it would be
    # solving with truncated flux), and after the disc rotation the extremal
    # particles sit exactly on the bounding box.
    pad = 2.0 * float(np.max(h))
    lo0, hi0 = pos.min(axis=0) - pad, pos.max(axis=0) + pad
    lo, hi, caud = cubic_domain_bounds(lo0, hi0, about="origin")
    side_kpc = float(hi[0] - lo[0]) * kpc_unit * a
    print(f"  domain: cubic, origin-centred, side {side_kpc:.3f} physical kpc "
          f"(grew {np.round(caud['grew'] * kpc_unit * a, 3)} kpc)")

    # Metallicity rides along only for the FIR dust model (n_dust ~ rho*Z); it is
    # not a POLARIS field. Older cutouts predate it -- fall back to a flag.
    have_metals = "metals" in d.files
    Zgas = (np.asarray(d["metals"], dtype=np.float64) if have_metals
            else np.full(len(mass), args.metallicity))
    if not have_metals:
        print(f"  no 'metals' in the cutout (regenerate it) -- using a flat "
              f"Z = {args.metallicity:g} for the FIR only")

    pvals = {"rho": rho, "T": T,
             "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
             "Bx": B[:, 0], "By": B[:, 1], "Bz": B[:, 2],
             "metals": Zgas}
    p = Particles(pos=np.ascontiguousarray(pos), mass=mass, rho=rho, h=h,
                  values=pvals, box=None, periodic=False)

    # ---- AMR octree --------------------------------------------------------
    if args.base_npx & (args.base_npx - 1):
        raise SystemExit(f"--base-npx must be a power of two, got {args.base_npx}")
    # Refinement criterion. DEFAULT `count`, matching the reconstruction pipeline
    # (`glassgen/reconstruct.py:480`, `part_per_cell=8`), whose stated reason is
    # the one that matters here: "~8 keeps the table coarser than the particle
    # sampling everywhere - finer would encode particle noise as target
    # structure". A smoothing-length criterion does exactly that. It never splits
    # an EMPTY cell (`AMRGrid.build` refuses), but it happily splits cells holding
    # one or two particles, and most of the 8 children come out empty: on the
    # lowres merger, 680,940 leaves of which 86.3% are empty, median 0 particles
    # per leaf. Those leaves still receive a kernel-weighted B with no local
    # sampling to constrain it -- shot noise on a vector field, which then
    # survives the divergence-free projection as structure that was never there.
    from sph_interp import refine_count, refine_mass, refine_occupied
    if args.refine == "count":
        criterion = refine_count(args.part_per_cell)
    elif args.refine == "mass":
        criterion = refine_mass(args.part_per_cell * float(np.mean(mass)))
    else:
        criterion = refine_smoothing_length(args.refine_factor)
        if args.refine_min_count > 0:
            criterion = refine_occupied(criterion, args.refine_min_count)
    grid = AMRGrid.build(p, base_npx=args.base_npx, max_level=args.max_level,
                         criterion=criterion, lo=lo, hi=hi)
    # 2:1 balance, as the reconstruction capture does (`reconstruct.py`, grade=
    # True). Count-driven refinement leaves the thin disc's deepest cells abutting
    # level-0/1 vacuum, and any readout across a many-level jump smears the fine
    # region's values over the whole coarse width -- the glassgen "z-staircase".
    # For POLARIS this is not cosmetic: a ray crossing that jump samples one
    # coarse cell where the field varies by decades.
    if args.grade:
        from sph_interp import grade_2to1
        n0 = grid.ncell
        graded = grade_2to1(grid, p)
        if graded is not grid:
            grid = graded
            print(f"  graded 2:1: {n0:,} -> {grid.ncell:,} leaves")
        else:
            print("  graded 2:1: already balanced")
    _cc = np.asarray(grid.cell_count) if grid.cell_count is not None else None
    if _cc is not None:
        _how = (f"--refine {args.refine} --part-per-cell {args.part_per_cell}"
                if args.refine in ("count", "mass")
                else f"--refine smoothing --refine-min-count {args.refine_min_count}")
        print(f"  occupancy: {100 * (_cc == 0).mean():.1f}% empty leaves, "
              f"median {np.median(_cc):.0f} particles/leaf   ({_how})")
    finest_pc = side_kpc / (args.base_npx * 2 ** args.max_level) * 1e3
    print(f"  AMR: {grid.ncell:,} leaves, base {args.base_npx}^3, "
          f"max_level {args.max_level}, finest cell {finest_pc:.1f} pc "
          f"[{time.time() - t0:.0f}s]")

    # ---- deposit -----------------------------------------------------------
    res = interpolate(p, grid, method=args.method,
                      values=["rho", "T", "vx", "vy", "vz", "Bx", "By", "Bz",
                              "metals"])
    vol = grid.volumes()
    m_grid = float((res.data["rho"] * vol).sum())
    m_part = float(mass.sum())
    print(f"  deposit ({args.method}): mass {m_grid:.6e} vs particles "
          f"{m_part:.6e}  ({100 * (m_grid / m_part - 1):+.3f}%) "
          f"[{time.time() - t0:.0f}s]")

    # ---- divergence-free B -------------------------------------------------
    mag_info = {}
    if args.magnetic_projection == "thb":
        from thb_readout import thb_curl_readout
        Bc, mag_info = thb_curl_readout(
            grid, [res.data["Bx"], res.data["By"], res.data["Bz"]],
            positions=grid.centres(), rtol=args.thb_rtol,
            op_dtype=np.float32 if args.thb_dtype == "float32" else np.float64)
        res.data["Bx"], res.data["By"], res.data["Bz"] = (
            np.ascontiguousarray(Bc[:, 0]), np.ascontiguousarray(Bc[:, 1]),
            np.ascontiguousarray(Bc[:, 2]))
        print(f"  B: THB De Rham curl-fit -> pointwise div-free at cell centres "
              f"[{time.time() - t0:.0f}s]")
    elif args.magnetic_projection == "hodge":
        table, cap, audit = capture_open_amr_face_field(
            p, grid, fields=("Bx", "By", "Bz"), background="particle-mean")
        Bc = table.cell_centred()
        res.data["Bx"], res.data["By"], res.data["Bz"] = (
            np.ascontiguousarray(Bc[:, 0]), np.ascontiguousarray(Bc[:, 1]),
            np.ascontiguousarray(Bc[:, 2]))
        print(f"  B: open-AMR Hodge projection -> div-free cell-centred "
              f"[{time.time() - t0:.0f}s]")
    else:
        print("  B: raw deposit (NOT divergence-free)")

    Bvec = np.stack([res.data["Bx"], res.data["By"], res.data["Bz"]], axis=1)
    Bmag_T = np.linalg.norm(Bvec, axis=1) * b_tesla
    rho_si = np.asarray(res.data["rho"]) * density_kgm3
    T_grid = np.asarray(res.data["T"])
    print(f"  |B| physical: {Bmag_T.max() * 1e10:.3g}e-10 T max "
          f"({Bmag_T.max() * 1e6 * 1e4:.2f} muG), "
          f"n_H max {rho_si.max() / pel.M_H_KG / 1e6:.3g} cm^-3")

    # ---- electron fields ---------------------------------------------------
    centres = grid.centres()
    # measure the disc scales from the cold dense gas rather than adopting the
    # paper's Milky Way 8 kpc / 1 kpc (which it says to re-fit per galaxy)
    cold = (rho > np.percentile(rho, 75)) & (T < 3.0e4)
    r_cold = np.linalg.norm(pos[cold][:, :2], axis=1) * kpc_unit * a
    z_cold = np.abs(pos[cold][:, 2]) * kpc_unit * a
    r_gal = args.r_gal_kpc if args.r_gal_kpc else max(float(np.mean(r_cold)), 0.05)
    h_gal = args.h_gal_kpc if args.h_gal_kpc else max(float(np.mean(z_cold)), 0.02)
    print(f"  measured disc scales: R_gal {r_gal:.3f} kpc, h_gal {h_gal:.3f} kpc "
          f"(paper's Milky Way values are 8 / 1)")

    n_th_over = None
    ionization = args.ionization
    have_ion = all(k in d.files for k in ("HI",))
    if ionization == "auto":
        ionization = "temperature"
        if have_ion:
            # deposit the abundances on the same mesh, then form n_e there
            pa_ion = Particles(pos=np.ascontiguousarray(pos), mass=mass, rho=rho,
                               h=h, values={k: d[k] for k in
                                            ("HI", "HeI", "HeII") if k in d.files},
                               box=None, periodic=False)
            rion = interpolate(pa_ion, grid, method=args.method,
                              values=list(pa_ion.values))
            Yhe = 0.236
            n_th_over = pel.electrons_from_abundances(
                rho_si, rion.data["HI"],
                rion.data.get("HeI"), rion.data.get("HeII"), he_mass_frac=Yhe)
            print(f"  n_th from the run's OWN ionization auxes "
                  f"({sorted(k for k in ('HI','HeI','HeII') if k in d.files)}), "
                  f"Y_He={Yhe}: max {n_th_over.max() / 1e6:.3g} cm^-3")
    if n_th_over is None:
        print(f"  n_th from the '{ionization}' model (no ionization auxes)")

    ef = pel.electron_fields(
        centres * (kpc_unit * a), rho_si, T_grid, b_tesla=Bvec * b_tesla,
        mu=u.mu, cr_model=args.cr_model, ionization=ionization,
        length_kpc=1.0,                      # centres already converted to kpc
        r_gal_kpc=r_gal, h_gal_kpc=h_gal, cr_p=args.cr_p,
        cr_mask_density=args.cr_mask_density, cr_mask_temp=args.cr_mask_temp,
        n_th_override=n_th_over)
    print("  " + pel.describe(ef, f"cr_model={args.cr_model}").replace("\n", "\n  "))

    # ---- FIR dust polarimetry (SOFIA VII §3.1) -----------------------------
    # NOT a POLARIS product: this is their own analytic line-of-sight integral,
    # done here on the same mesh and the same div-free B as the synchrotron so
    # that the radio-vs-FIR comparison is genuinely apples to apples.
    if not args.no_fir:
        import sofia7_fir as s7
        n_dust = s7.dust_number_density(
            rho_si, np.asarray(res.data["metals"]), T=T_grid,
            eta_dm=args.eta_dm, t_sublim=args.t_sublimation)
        fir_out = {}
        for label, los in (("face_on", "z"), ("edge_on", "y")):
            m = s7.fir_maps(grid, n_dust, Bvec, los=los, npix=args.fir_npix,
                            p0=args.p0_fir)
            for k in ("I", "Q", "U", "P", "p", "chi"):
                fir_out[f"{label}_{k}"] = m[k].astype(np.float32)
            print("  " + s7.describe(m, label).replace("\n", "\n  "))
        fir_out["side_kpc"] = side_kpc
        fir_out["npix"] = args.fir_npix
        fir_out["p0"] = args.p0_fir
        fir_out["eta_dm"] = args.eta_dm
        fir_out["Z_mean"] = float(np.mean(np.asarray(res.data["metals"])))
        fir_path = os.path.join(args.out_dir, "fir_maps.npz")
        np.savez_compressed(fir_path, **fir_out)
        print(f"  FIR maps -> {fir_path}  [{time.time() - t0:.0f}s]")

    # ---- write the grid ----------------------------------------------------
    fields = ["rho", "Bx", "By", "Bz", "vx", "vy", "vz", "T",
              "n_th", "T_e", "n_cr", "g_min", "g_max", "cr_p"]
    data = dict(res.data)
    data["T"] = T_grid
    data.update(ef)
    res.data = data
    units = {"rho": density_kgm3, "Bx": b_tesla, "By": b_tesla, "Bz": b_tesla,
             "vx": velocity_ms, "vy": velocity_ms, "vz": velocity_ms,
             "T": 1.0, "n_th": 1.0, "T_e": 1.0, "n_cr": 1.0,
             "g_min": 1.0, "g_max": 1.0, "cr_p": 1.0}
    ids = [28, 4, 5, 6, 7, 8, 9, 3, 22, 23, 24, 25, 26, 27]
    grid_path = os.path.join(args.out_dir, "grid_sync.dat")
    meta = to_polaris(res, grid_path, fields=fields, data_ids=ids, units=units,
                      length_scale=length_m)
    print(f"  grid: {meta['ncell']:,} leaves, {meta['nrecord']:,} records, "
          f"levels {meta['level_min']}-{meta['level_max']}, "
          f"cube {meta['cube_length']:.4e} m "
          f"({meta['cube_length'] / (3.0856775814913673e19):.3f} kpc), "
          f"{os.path.getsize(grid_path) / 1e6:.1f} MB")

    # ---- command file + sidecar -------------------------------------------
    wl_m = [w * 1e-2 for w in args.wavelength_cm]
    dist_m = args.distance_mpc * 1e3 * 3.0856775814913673e19
    cmd = write_cmd(os.path.join(args.out_dir, "POLARIS.cmd"),
                    os.path.abspath(grid_path),
                    os.path.abspath(args.out_dir), wl_m, dist_m, npix=args.npix)
    side = {
        "grid": meta, "units": u.as_dict(),
        "cosmology": {"a": a, "z": 1.0 / a - 1.0,
                      "length_m_physical": length_m,
                      "density_kgm3_physical": density_kgm3,
                      "b_tesla_physical": b_tesla,
                      "exponents": "L~a, rho~a^-3, B~a^-2"},
        "cutout": {"file": os.path.abspath(args.cut),
                   "radius_kpc": float(d["radius_kpc"]),
                   "centre_code": np.asarray(d["centre"]).tolist(),
                   "n_gas": int(len(mass))},
        "frame": {"disc_axis": axis.tolist(), "rotation": R.tolist(),
                  "rotated": not args.no_rotate},
        "amr": {"base_npx": args.base_npx, "max_level": args.max_level,
                "refine_factor": args.refine_factor, "method": args.method,
                "ncell": int(grid.ncell), "finest_pc": finest_pc,
                "side_kpc": side_kpc},
        "magnetic": {"projection": args.magnetic_projection,
                     "bmax_tesla": float(Bmag_T.max()),
                     "bmax_muG": float(Bmag_T.max() * 1e10),
                     **mag_info},
        "electrons": {"cr_model": args.cr_model, "ionization": ionization,
                      "n_th_from_auxes": n_th_over is not None,
                      "r_gal_kpc": r_gal, "h_gal_kpc": h_gal,
                      "cr_p": args.cr_p,
                      "gamma_min": pel.GAMMA_MIN, "gamma_max": pel.GAMMA_MAX,
                      "n_cr_max": float(ef["n_cr"].max()),
                      "n_th_max": float(ef["n_th"].max())},
        "mass_check": {"grid": m_grid, "particles": m_part,
                       "ratio": m_grid / m_part},
        "detectors": {"wavelength_cm": args.wavelength_cm,
                      "distance_mpc": args.distance_mpc, "npix": args.npix},
    }
    spath = os.path.join(args.out_dir, "grid_sync.polaris.json")
    with open(spath, "w") as fh:
        json.dump(side, fh, indent=2, default=float)
    print(f"  cmd:     {cmd}")
    print(f"  sidecar: {spath}")
    # anchor on THIS file, not the cwd -- os.path.abspath('../utilities/...')
    # silently resolved against wherever the user happened to be standing.
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"done in {time.time() - t0:.0f}s\n\nrun it with:\n"
          f"  {os.path.join(_repo, 'utilities/POLARIS/bin/polaris')} "
          f"{os.path.abspath(cmd)}")


if __name__ == "__main__":
    main()
