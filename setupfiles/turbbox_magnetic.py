"""Mean-field checks for the retained published-forcing turbulence campaign."""

import contextlib
import io

import numpy as np

import IC_analysis_turbbox as analysis
import turbbox_ensemble as ensemble


def particle_magnetic_moments(gas, field):
    """Approximate volume moments; m/rho is not an exact volume partition.

    Keep both the unnormalised integral and the normalised mean so a volume
    deficit cannot be mistaken for exact conservation of the periodic flux.
    """
    gas = np.asarray(gas, dtype=float)
    field = np.asarray(field, dtype=float)
    if gas.ndim != 2 or gas.shape[1] < 8 or field.shape != (len(gas), 3):
        raise ValueError("need gas columns through density and one field vector per particle")
    mass, rho = gas[:, 0], gas[:, 7]
    if (np.any(~np.isfinite(field)) or np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(rho)) or np.any(mass <= 0) or np.any(rho <= 0)
            or len(gas) == 0):
        raise ValueError("magnetic moments require finite fields and positive mass/density")
    volume = mass / rho
    volume_sum = float(volume.sum())
    integral = np.sum(volume[:, None] * field, axis=0)
    mean = integral / volume_sum
    energy = float(.5 * np.sum(volume * np.sum(field * field, axis=1)))
    mean_energy = float(.5 * volume_sum * np.dot(mean, mean))
    return {
        "volume_sum": volume_sum,
        "integral_B": integral.tolist(),
        "mean_B": mean.tolist(),
        "mean_B_magnitude": float(np.linalg.norm(mean)),
        "magnetic_energy": energy,
        "mean_energy_fraction": mean_energy / energy if energy > 0 else 0.,
    }


def magnetic_mean_figure(results, cache, source_archive=None):
    """Compare cached grid zero modes with independently read particle fields."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    initial_field = float(np.sqrt(2) * 1e-5)
    metrics = {
        "nominal_initial_mean_B_magnitude": initial_field,
        "grid_definition": "|mean(B)| and |mean(B)|^2 / mean(|B|^2) on a unit volume grid",
        "particle_definition": "sum((m/rho)*B)/sum(m/rho); approximate quadrature, not exact net flux",
        "interpretation": "mean-field growth is a dynamo-comparison limitation; no causal operator identified",
        "resolutions": {},
    }
    for i, result in enumerate(results):
        native = result["native"]
        records = list(result["records"])
        final_grid = ensemble.cached_grid(result["rows"][-1], 2*native, cache,
                                           source_archive=source_archive)
        if not any(np.isclose(r["time_crossings"], final_grid["time_crossings"],
                              rtol=0, atol=1e-8) for r in records):
            records.append(final_grid)
        records.sort(key=lambda r: float(r["time_crossings"]))
        samples = []
        for grid in records:
            tau = float(grid["time_crossings"])
            row = min(result["rows"], key=lambda r: abs(r["time"]/.05-tau))
            if not np.isclose(row["time"]/.05, tau, rtol=0, atol=1e-8):
                raise ValueError("magnetic mean-field sample has no matching particle snapshot")
            with contextlib.redirect_stdout(io.StringIO()):
                gas, _dark, _stars, header, _time = analysis.tip.readtipsy(row["filename"])
                ngas = int(header[2])
                field = analysis.read_vector_aux(row["filename"], "BField", ngas)
            particle = particle_magnetic_moments(gas[:ngas], field)
            if not np.isclose(particle["magnetic_energy"], row["emag"], rtol=1e-12, atol=0):
                raise ValueError("particle magnetic energy differs from the retained history")
            zero = float(grid["magnetic_zero_mode_power"])
            total = float(grid["magnetic_full_grid_power"])
            kinetic = float(grid["kinetic_full_grid_power"])
            if not (0 <= zero <= total and total > 0 and kinetic > 0):
                raise ValueError("invalid grid mean-field energy decomposition")
            samples.append({
                "time_crossings": tau, "snapshot": row["filename"],
                "grid_mean_B_magnitude": float(np.sqrt(2 * zero)),
                "grid_mean_energy_fraction": zero / total,
                "grid_total_magnetic_over_kinetic": total / kinetic,
                "grid_fluctuating_magnetic_over_kinetic": (total - zero) / kinetic,
                "particle": particle,
            })
        final_grids = []
        for factor in (1, 2, 4):
            grid = ensemble.cached_grid(result["rows"][-1], factor*native, cache,
                                        source_archive=source_archive)
            final_grids.append({
                "grid": factor*native,
                "mean_B_magnitude": float(np.sqrt(2*grid["magnetic_zero_mode_power"])),
                "mean_energy_fraction": float(grid["magnetic_zero_mode_power"] /
                                              grid["magnetic_full_grid_power"]),
            })
        start, end = result["late_window"]
        late = [s for s in samples if start-1e-8 <= s["time_crossings"] <= end+1e-8]
        metrics["resolutions"][str(native)] = {
            "samples": samples, "late_window_crossings": [start, end],
            "late_mean_grid_mean_energy_fraction": float(np.mean(
                [s["grid_mean_energy_fraction"] for s in late])),
            "final_grid_refinement": final_grids,
            "final_2x_to_4x_mean_B_relative_change":
                final_grids[2]["mean_B_magnitude"] / final_grids[1]["mean_B_magnitude"] - 1,
        }
        t = [s["time_crossings"] for s in samples]
        color, label = f"C{i}", rf"${native}^3$"
        axes[0, 0].semilogy(t, [s["grid_mean_B_magnitude"] for s in samples],
                            "o-", color=color, ms=3, label=label+": grid")
        axes[0, 0].semilogy(t, [s["particle"]["mean_B_magnitude"] for s in samples],
                            "--", color=color, label=label+": particle estimate")
        axes[0, 1].plot(t, [s["grid_mean_energy_fraction"] for s in samples],
                        "o-", color=color, ms=3, label=label)
        for key, style, desc in (("grid_total_magnetic_over_kinetic", "-", "total B"),
                                 ("grid_fluctuating_magnetic_over_kinetic", "--", "B fluctuations")):
            axes[1, 0].semilogy(t, [s[key] for s in samples], style, color=color,
                                label=label+": "+desc)
        axes[1, 1].plot((1, 2, 4), [g["mean_B_magnitude"] for g in final_grids],
                        "o-", color=color, label=label+": grid")
        axes[1, 1].axhline(samples[-1]["particle"]["mean_B_magnitude"], color=color,
                           ls="--", label=label+": particle estimate")
    axes[0, 0].axhline(initial_field, color="0.4", ls=":", label="nominal initial field")
    axes[0, 0].set(title="Mean field grows far above the seed", ylabel=r"$|\langle B\rangle|$")
    axes[0, 1].set(title="Mean-field energy fraction",
                   ylabel=r"$|\langle B\rangle|^2/\langle |B|^2\rangle$", ylim=(0, 1))
    axes[1, 0].set(title="Magnetic energy: total and fluctuations",
                   ylabel=r"$E_B/E_K$ on the same grid")
    axes[1, 1].set(title="Final mean field versus grid resolution",
                   xlabel="grid / native linear resolution", ylabel=r"$|\langle B\rangle|$", xticks=(1, 2, 4))
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel(r"$t/t_c$")
    for ax in axes.ravel():
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8)
    return fig, metrics
