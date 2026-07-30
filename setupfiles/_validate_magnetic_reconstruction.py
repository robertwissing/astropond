#!/usr/bin/env python
"""Periodic magnetic reconstruction resolution matrix.

This is the synthetic first gate for the face-field -> particle path.  Source
and target particles are independent, genuinely relaxed uniform glasses by
default.  The reference and reconstructed fields are always measured on the
same final target glass.  Jittered particles remain as an explicit fast
diagnostic, and production reconstructions can pass their actual final glass
to ``sph_interp.validate_magnetic_reconstruction``.

Usage:
  python setupfiles/_validate_magnetic_reconstruction.py --quick
  python setupfiles/_validate_magnetic_reconstruction.py \
      --particle-sides 6,8,10 --grid-npx 4,6,8,10 --json result.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "icgenerator", "glassgen_python"))

from sph_interp import (Particles, UniformGrid, capture_periodic_face_field,
                        validate_magnetic_reconstruction)
from glassgen import relax


def analytic_field(positions):
    """Smooth periodic div-free field with a nonzero mean component."""
    pos = np.asarray(positions, dtype=np.float64)
    twopi = 2.0 * np.pi
    return np.column_stack([
        0.35 + np.sin(twopi * pos[:, 1])
        + 0.20 * np.cos(twopi * pos[:, 2]),
        -0.20 + 0.70 * np.sin(twopi * pos[:, 2])
        + 0.15 * np.cos(2.0 * twopi * pos[:, 0]),
        0.10 + 0.45 * np.cos(twopi * pos[:, 0])
        + 0.10 * np.sin(2.0 * twopi * pos[:, 1]),
    ])


def jittered_particles(nside, seed, with_field):
    """Deterministic glass-like periodic points with kNN smoothing lengths."""
    from scipy.spatial import cKDTree

    spacing = 1.0 / nside
    axis = (np.arange(nside, dtype=np.float64) + 0.5) * spacing - 0.5
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    pos += rng.uniform(-0.18 * spacing, 0.18 * spacing, pos.shape)
    pos = (pos + 0.5) % 1.0 - 0.5
    n = len(pos)
    mass = np.full(n, 1.0 / n)
    rho = np.ones(n)
    k = min(32, n - 1)
    distance = cKDTree(pos + 0.5, boxsize=1.0).query(
        pos + 0.5, k=k + 1, workers=-1)[0][:, -1]
    h = 0.5 * distance
    values = {}
    if with_field:
        B = analytic_field(pos)
        values = {"Bx": B[:, 0], "By": B[:, 1], "Bz": B[:, 2]}
    return Particles(pos, mass, rho, h, values, np.ones(3), True)


def relaxed_particles(nside, seed, with_field, *, nsmooth=32,
                      max_iter=1200, icr0_stop=3e-3):
    """Build an independent periodic uniform glass from a random start."""
    rng = np.random.default_rng(seed)
    n = int(nside) ** 3
    initial = rng.uniform(-0.5, 0.5, size=(n, 3))
    mass = np.full(n, 1.0 / n)
    effective_nsmooth = min(int(nsmooth), n - 1)
    result = relax(
        initial, mass, lambda points: np.ones(len(points)), box=np.ones(3),
        nsmooth=effective_nsmooth, max_iter=int(max_iter),
        icr0_stop=float(icr0_stop))
    if not result.converged:
        raise RuntimeError(
            f"uniform glass nside={nside} did not converge: "
            f"{result.stop_reason} after {result.niter} iterations")
    values = {}
    if with_field:
        B = analytic_field(result.pos)
        values = {"Bx": B[:, 0], "By": B[:, 1], "Bz": B[:, 2]}
    particles = Particles(
        result.pos, mass, result.rho, result.h, values, np.ones(3), True)
    audit = {
        "kind": "relaxed-uniform-glass",
        "seed": int(seed),
        "nsmooth": int(effective_nsmooth),
        "niter": int(result.niter),
        "stop_reason": str(result.stop_reason),
        "rho_mean_absolute_error": float(np.mean(np.abs(result.rho - 1.0))),
        "rho_relative_std": float(result.rho.std() / result.rho.mean()),
        "particle_volume_sum": float(np.sum(mass / result.rho)),
    }
    return particles, audit


def _csv_ints(value):
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or any(item < 2 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated integers >= 2")
    return result


def _csv_reconstructions(value):
    result = tuple(item.strip().lower() for item in value.split(",")
                   if item.strip())
    allowed = {"rt0", "balsara2"}
    if not result or any(item not in allowed for item in result):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated subset of rt0,balsara2")
    return result


def run_matrix(particle_sides, grid_npx, flux_npx=20, spectrum_npx=16,
               particle_mode="glass", glass_nsmooth=32,
               glass_max_iter=1200, glass_icr0_stop=3e-3,
               reconstructions=("rt0",), balsara_limiter="mc"):
    rows = []
    for nside in particle_sides:
        if particle_mode == "glass":
            source, source_geometry = relaxed_particles(
                nside, seed=1000 + nside, with_field=True,
                nsmooth=glass_nsmooth, max_iter=glass_max_iter,
                icr0_stop=glass_icr0_stop)
            target, target_geometry = relaxed_particles(
                nside, seed=2000 + nside, with_field=False,
                nsmooth=glass_nsmooth, max_iter=glass_max_iter,
                icr0_stop=glass_icr0_stop)
        elif particle_mode == "jittered":
            source = jittered_particles(
                nside, seed=1000 + nside, with_field=True)
            target = jittered_particles(
                nside, seed=2000 + nside, with_field=False)
            source_geometry = {"kind": "jittered-lattice"}
            target_geometry = {"kind": "jittered-lattice"}
        else:
            raise ValueError("particle_mode must be 'glass' or 'jittered'")
        for npx in grid_npx:
            grid = UniformGrid.centered(np.ones(3), npx)
            table, capture_audit, _ = capture_periodic_face_field(source, grid)
            for reconstruction_name in reconstructions:
                if reconstruction_name == "rt0":
                    reconstruction = table
                elif reconstruction_name == "balsara2":
                    reconstruction = table.balsara2_reconstruction(
                        limiter=balsara_limiter)
                else:
                    raise ValueError(
                        "reconstructions must contain only rt0 or balsara2")
                report = validate_magnetic_reconstruction(
                    reconstruction, target, analytic_field,
                    flux_npx=flux_npx, spectrum_npx=spectrum_npx)
                div = report["sph_divergence"]
                capture_energy = capture_audit["magnetic_energy"]
                rows.append({
                    "particle_side": int(nside),
                    "n_particles": int(source.n),
                    "grid_npx": int(npx),
                    "reconstruction": reconstruction_name,
                    "limiter": (
                        balsara_limiter
                        if reconstruction_name == "balsara2" else None),
                    "source_geometry": source_geometry,
                    "target_geometry": target_geometry,
                    "capture": capture_audit,
                    "validation": report,
                    "summary": {
                        "B_relative_l2": report["point_error"]["relative_l2"],
                        "energy_relative_error": report["magnetic_energy"][
                            "relative_error"],
                        "source_to_raw_energy_error": (
                            capture_energy["raw_over_source"] - 1.0),
                        "source_to_face_energy_error": (
                            capture_energy["projected_over_source"] - 1.0),
                        "source_to_raw_volume_B_error": capture_audit[
                            "particle_to_raw_integral_B"]["relative_norm"],
                        "raw_to_face_volume_B_error": capture_audit[
                            "raw_to_projected_integral_B"]["relative_norm"],
                        "imprint_volume_B_error": report["volume_integral_B"][
                            "relative_norm"],
                        "flux_relative_l2": report["resolved_fluxes"][
                            "relative_l2"],
                        "spectrum_relative_l1": report["spectra"][
                            "relative_l1"],
                        "reference_div_mean": div["reference"][
                            "normalized_mean"],
                        "reconstructed_div_mean": div["reconstructed"][
                            "normalized_mean"],
                        "div_mean_ratio": div["comparison"][
                            "normalized_mean"]["ratio"],
                    },
                })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-sides", type=_csv_ints, default=(6, 8),
                        help="comma-separated source/target lattice sides")
    parser.add_argument("--grid-npx", type=_csv_ints, default=(4, 6, 8),
                        help="comma-separated uniform-grid resolutions")
    parser.add_argument("--flux-npx", type=int, default=20)
    parser.add_argument("--spectrum-npx", type=int, default=16)
    parser.add_argument("--reconstructions", type=_csv_reconstructions,
                        default=("rt0",),
                        help="comma-separated point readouts: rt0,balsara2")
    parser.add_argument("--balsara-limiter",
                        choices=("mc", "minmod", "central"), default="mc")
    parser.add_argument("--particle-mode", choices=("glass", "jittered"),
                        default="glass",
                        help="independent relaxed glasses (default) or the "
                             "old jittered-lattice diagnostic")
    parser.add_argument("--glass-nsmooth", type=int, default=32)
    parser.add_argument("--glass-max-iter", type=int, default=1200)
    parser.add_argument("--glass-icr0-stop", type=float, default=3e-3)
    parser.add_argument("--quick", action="store_true",
                        help="small two-point smoke matrix")
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args(argv)
    if args.quick:
        args.particle_sides = (5,)
        args.grid_npx = (4, 6)
        args.flux_npx = min(args.flux_npx, 12)
        args.spectrum_npx = min(args.spectrum_npx, 12)

    rows = run_matrix(args.particle_sides, args.grid_npx,
                      flux_npx=args.flux_npx,
                      spectrum_npx=args.spectrum_npx,
                      particle_mode=args.particle_mode,
                      glass_nsmooth=args.glass_nsmooth,
                      glass_max_iter=args.glass_max_iter,
                      glass_icr0_stop=args.glass_icr0_stop,
                      reconstructions=args.reconstructions,
                      balsara_limiter=args.balsara_limiter)
    print("particles grid readout    relL2(B)   dE source->raw/face   dE target->part "
          " |d(VB)| raw/face/part       SPHdiv(ref->recon)")
    for row in rows:
        summary = row["summary"]
        print(f"{row['n_particles']:9d} {row['grid_npx']:4d} "
              f"{row['reconstruction']:10s} "
              f"{summary['B_relative_l2']:9.3e} "
              f"{summary['source_to_raw_energy_error']:+8.2e}/"
              f"{summary['source_to_face_energy_error']:+8.2e} "
              f"{summary['energy_relative_error']:+8.2e} "
              f"{summary['source_to_raw_volume_B_error']:.1e}/"
              f"{summary['raw_to_face_volume_B_error']:.1e}/"
              f"{summary['imprint_volume_B_error']:.1e} "
              f"{summary['reference_div_mean']:.3e}->"
              f"{summary['reconstructed_div_mean']:.3e}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump({"schema": "magnetic-reconstruction-validation-v1",
                       "rows": rows}, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
