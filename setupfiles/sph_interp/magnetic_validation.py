"""Validation tools for face-field -> particle magnetic reconstruction.

The face-field constraint and the particle SPH divergence are different
operators.  This module compares an analytic/reference field and a reconstructed
field at *the same target particle positions*, so particle disorder and the SPH
operator's own truncation error are present in both arms.

The validator is intentionally independent of how the target particles were
made.  A synthetic jittered glass can be used for a quick resolution matrix,
while a production reconstruction can pass its actual final ``Particles``.
"""

import numpy as np

from .magnetic import sph_divergence_audit
from .particles import Particles


def _plain_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _relative(delta, reference):
    scale = abs(float(reference))
    if scale == 0.0:
        return None
    return float(delta) / scale


def _field_query(field):
    """Return one point evaluator for a uniform or AMR magnetic field."""
    if hasattr(field, "query"):
        return field.query, field
    if hasattr(field, "hdiv_reconstruction"):
        reconstruction = field.hdiv_reconstruction()
        return reconstruction.query, reconstruction
    raise TypeError(
        "magnetic field must provide query() or hdiv_reconstruction()")


def _domain(field):
    """Return ``(lo, box)`` for uniform tables and AMR reconstructions."""
    if hasattr(field, "lo") and hasattr(field, "box"):
        return (np.asarray(field.lo, dtype=np.float64),
                np.asarray(field.box, dtype=np.float64))
    grid = getattr(field, "grid", None)
    if grid is None:
        topology = getattr(field, "topology", None)
        grid = getattr(topology, "grid", None)
    if grid is None:
        raise TypeError("magnetic field does not expose a rectangular domain")
    return (np.asarray(grid.lo, dtype=np.float64),
            np.asarray(grid.box, dtype=np.float64))


def _reference_values(reference, positions):
    values = reference(positions) if callable(reference) else reference
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(positions), 3):
        raise ValueError(
            "reference magnetic field must evaluate to shape (N,3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("reference magnetic field contains non-finite values")
    return values


def _summarize_divergence(audit):
    normalized = np.asarray(audit["normalized"], dtype=np.float64)
    return {
        "operator": audit["operator"],
        "divB_rms": float(audit["divB_rms"]),
        "divB_max": float(audit["divB_max"]),
        "normalized_mean": float(audit["normalized_mean"]),
        "normalized_rms": float(audit["normalized_rms"]),
        "normalized_max": float(audit["normalized_max"]),
        "normalized_percentiles": {
            str(q): float(np.percentile(normalized, q))
            for q in (50, 90, 95, 99)
        },
    }


def _divergence_comparison(reference, reconstructed):
    keys = ("divB_rms", "divB_max", "normalized_mean", "normalized_rms",
            "normalized_max")
    out = {}
    for key in keys:
        initial = float(reference[key])
        final = float(reconstructed[key])
        out[key] = {
            "reference": initial,
            "reconstructed": final,
            "delta": final - initial,
            "ratio": (final / initial if initial != 0.0 else None),
        }
    return out


def _point_error(reference, reconstructed):
    error = reconstructed - reference
    error_norm = np.linalg.norm(error, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    numerator = float(np.sum(error * error))
    denominator = float(np.sum(reference * reference))
    return {
        "component_bias": np.mean(error, axis=0).tolist(),
        "component_rms": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "vector_rms": float(np.sqrt(np.mean(error_norm * error_norm))),
        "vector_mae": float(np.mean(error_norm)),
        "vector_max": float(np.max(error_norm)),
        "relative_l2": (float(np.sqrt(numerator / denominator))
                        if denominator != 0.0 else None),
        "relative_point_percentiles": {
            str(q): float(np.percentile(
                error_norm / np.maximum(reference_norm,
                                        np.finfo(np.float64).tiny), q))
            for q in (50, 90, 95, 99)
        },
    }


def _vector_budget(reference, reconstructed):
    reference = np.asarray(reference, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    delta = reconstructed - reference
    component_relative = np.full(3, np.nan, dtype=np.float64)
    nonzero = reference != 0.0
    component_relative[nonzero] = delta[nonzero] / reference[nonzero]
    scale = max(float(np.linalg.norm(reference)),
                np.finfo(np.float64).tiny)
    return {
        "reference": reference.tolist(),
        "reconstructed": reconstructed.tolist(),
        "delta": delta.tolist(),
        "component_relative": [
            _plain_float(value) for value in component_relative],
        "relative_norm": float(np.linalg.norm(delta) / scale),
    }


def _energy_and_mean(particles, reference, reconstructed, table):
    volume = particles.mass / particles.rho
    reference_energy = float(0.5 * np.sum(
        volume * np.sum(reference * reference, axis=1)))
    reconstructed_energy = float(0.5 * np.sum(
        volume * np.sum(reconstructed * reconstructed, axis=1)))
    total_volume = float(np.sum(volume))
    reference_mean = np.sum(volume[:, None] * reference, axis=0) / total_volume
    reconstructed_mean = (np.sum(volume[:, None] * reconstructed, axis=0)
                          / total_volume)
    reference_integral = np.sum(volume[:, None] * reference, axis=0)
    reconstructed_integral = np.sum(
        volume[:, None] * reconstructed, axis=0)
    out = {
        "particle_volume": total_volume,
        "reference": reference_energy,
        "reconstructed": reconstructed_energy,
        "delta": reconstructed_energy - reference_energy,
        "relative_error": _relative(
            reconstructed_energy - reference_energy, reference_energy),
        "mean_field_reference": reference_mean.tolist(),
        "mean_field_reconstructed": reconstructed_mean.tolist(),
        "mean_field_error": (reconstructed_mean - reference_mean).tolist(),
        "volume_integral_B": _vector_budget(
            reference_integral, reconstructed_integral),
    }
    if hasattr(table, "magnetic_energy"):
        out["face_table"] = float(table.magnetic_energy())
    return out


def _plane_points(lo, box, axis, fraction, npx):
    transverse = [component for component in range(3) if component != axis]
    centres = [(np.arange(npx, dtype=np.float64) + 0.5) / npx
               for _ in transverse]
    u, v = np.meshgrid(*centres, indexing="ij")
    points = np.empty((npx * npx, 3), dtype=np.float64)
    points[:, axis] = lo[axis] + float(fraction) * box[axis]
    points[:, transverse[0]] = (lo[transverse[0]]
                                + u.ravel() * box[transverse[0]])
    points[:, transverse[1]] = (lo[transverse[1]]
                                + v.ravel() * box[transverse[1]])
    area = box[transverse[0]] * box[transverse[1]]
    return points, float(area)


def _resolved_fluxes(query, reference, lo, box, npx, planes):
    reference_flux = np.empty((3, len(planes)), dtype=np.float64)
    reconstructed_flux = np.empty_like(reference_flux)
    for axis in range(3):
        for pi, fraction in enumerate(planes):
            points, area = _plane_points(lo, box, axis, fraction, npx)
            bref = _reference_values(reference, points)
            brec = np.asarray(query(points), dtype=np.float64)
            reference_flux[axis, pi] = area * np.mean(bref[:, axis])
            reconstructed_flux[axis, pi] = area * np.mean(brec[:, axis])
    error = reconstructed_flux - reference_flux
    denominator = float(np.sum(reference_flux * reference_flux))
    return {
        "quadrature_npx": int(npx),
        "plane_fractions": [float(value) for value in planes],
        "reference": reference_flux.tolist(),
        "reconstructed": reconstructed_flux.tolist(),
        "error": error.tolist(),
        "absolute_rms": float(np.sqrt(np.mean(error * error))),
        "relative_l2": (float(np.sqrt(np.sum(error * error) / denominator))
                        if denominator != 0.0 else None),
    }


def _sample_cube(evaluator, lo, box, npx):
    axes = [lo[axis] + (np.arange(npx) + 0.5) * box[axis] / npx
            for axis in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return np.asarray(evaluator(points), dtype=np.float64).reshape(
        npx, npx, npx, 3)


def _isotropic_power(field):
    npx = field.shape[0]
    coefficient = np.fft.fftn(field, axes=(0, 1, 2)) / npx ** 3
    power = 0.5 * np.sum(np.abs(coefficient) ** 2, axis=-1)
    frequency = np.fft.fftfreq(npx) * npx
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    shell = np.rint(np.sqrt(kx * kx + ky * ky + kz * kz)).astype(np.int64)
    maximum = int(shell.max())
    shell_power = np.bincount(shell.ravel(), weights=power.ravel(),
                              minlength=maximum + 1)
    shell_modes = np.bincount(shell.ravel(), minlength=maximum + 1)
    return shell_power, shell_modes


def _spectra(query, reference, lo, box, npx):
    reference_cube = _sample_cube(
        lambda points: _reference_values(reference, points), lo, box, npx)
    reconstructed_cube = _sample_cube(query, lo, box, npx)
    reference_power, modes = _isotropic_power(reference_cube)
    reconstructed_power, _ = _isotropic_power(reconstructed_cube)
    error = reconstructed_power - reference_power
    denominator = float(np.sum(reference_power * reference_power))
    return {
        "sample_npx": int(npx),
        "k_shell": list(range(len(reference_power))),
        "modes": modes.tolist(),
        "reference_power": reference_power.tolist(),
        "reconstructed_power": reconstructed_power.tolist(),
        "power_error": error.tolist(),
        "total_reference_power": float(reference_power.sum()),
        "total_reconstructed_power": float(reconstructed_power.sum()),
        "relative_l1": (float(np.sum(np.abs(error))
                              / np.sum(np.abs(reference_power)))
                        if np.any(reference_power) else None),
        "relative_l2": (float(np.sqrt(np.sum(error * error) / denominator))
                        if denominator != 0.0 else None),
    }


def validate_magnetic_reconstruction(
        table, target_particles, reference_field, *, flux_npx=24,
        spectrum_npx=16, flux_planes=(0.0, 0.25, 0.5, 0.75),
        return_fields=False):
    """Compare reference and reconstructed B on identical target particles.

    Parameters
    ----------
    table
        A uniform face table, an AMR face table, or an already-built magnetic
        reconstruction exposing ``query``.
    target_particles : Particles
        The final particle geometry.  Its mass, rho, and h define both arms of
        the particle-volume energy and SPH-divergence comparison.
    reference_field : callable or (N,3) array
        A callable is required when plane-flux or spectral diagnostics are
        enabled.  An array is sufficient for particle-only comparison.
    flux_npx, spectrum_npx
        Set either to zero/None to disable that diagnostic.

    Returns
    -------
    dict, or (dict, reference_B, reconstructed_B)
        The report is JSON-serializable.  Raw particle fields are returned only
        when ``return_fields=True``.
    """
    if not isinstance(target_particles, Particles):
        raise TypeError("target_particles must be sph_interp.Particles")
    if any(target_particles.periodic) and not all(target_particles.periodic):
        raise NotImplementedError(
            "magnetic validation currently requires all-periodic or all-open particles")

    query, reconstruction = _field_query(table)
    reference_B = _reference_values(reference_field, target_particles.pos)
    reconstructed_B = np.asarray(query(target_particles.pos), dtype=np.float64)
    if reconstructed_B.shape != reference_B.shape:
        raise ValueError("reconstructed magnetic field has the wrong shape")
    if not np.all(np.isfinite(reconstructed_B)):
        raise ValueError("reconstructed magnetic field contains non-finite values")

    periodic = all(target_particles.periodic)
    reference_div_raw = sph_divergence_audit(
        target_particles.pos, target_particles.mass, target_particles.rho,
        target_particles.h, reference_B, box=target_particles.box,
        periodic=periodic)
    reconstructed_div_raw = sph_divergence_audit(
        target_particles.pos, target_particles.mass, target_particles.rho,
        target_particles.h, reconstructed_B, box=target_particles.box,
        periodic=periodic)
    reference_div = _summarize_divergence(reference_div_raw)
    reconstructed_div = _summarize_divergence(reconstructed_div_raw)

    report = {
        "n_particles": int(target_particles.n),
        "periodic": bool(periodic),
        "reconstruction": type(reconstruction).__name__,
        "point_error": _point_error(reference_B, reconstructed_B),
        "magnetic_energy": _energy_and_mean(
            target_particles, reference_B, reconstructed_B, table),
        "sph_divergence": {
            "reference": reference_div,
            "reconstructed": reconstructed_div,
            "comparison": _divergence_comparison(reference_div,
                                                   reconstructed_div),
        },
    }
    report["volume_integral_B"] = report["magnetic_energy"].pop(
        "volume_integral_B")

    if flux_npx or spectrum_npx:
        if not callable(reference_field):
            raise ValueError(
                "reference_field must be callable for flux/spectral diagnostics")
        lo, box = _domain(reconstruction)
        report["domain"] = {"lo": lo.tolist(), "box": box.tolist()}
        if flux_npx:
            report["resolved_fluxes"] = _resolved_fluxes(
                query, reference_field, lo, box, int(flux_npx),
                tuple(flux_planes))
        if spectrum_npx:
            report["spectra"] = _spectra(
                query, reference_field, lo, box, int(spectrum_npx))

    if return_fields:
        return report, reference_B, reconstructed_B
    return report
