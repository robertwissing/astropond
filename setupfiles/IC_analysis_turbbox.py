#!/usr/bin/env python3
"""Diagnostics for the OU-driven periodic turbulence box.

The analysis follows the diagnostics used by Price & Federrath (2010) and
Tricco, Price & Federrath (2016): Mach/energy histories, particle-wise
Alfvenic Mach and Alfven-speed histories, density maps and log-density PDFs,
kinetic and density-weighted velocity spectra, Helmholtz mode fractions, and
(for MHD) magnetic spectra, magnetic PDFs, and B-rho correlations.

Volume statistics use the testsuite ``sph_interp`` Petkova framework;
visual slices and z-column renders use its default SPH rendering backend.
The default statistics mesh has twice the native linear particle resolution;
use ``--grid-res`` to select its size. ``--render`` produces rho/|B| slices
at z=0.5 and z-column integrals at 1, 2, 4 and 8 crossing times.

The requested Mach number is a forcing calibration, not a feedback target.
All reported Mach numbers are measured after subtracting the mass-weighted
bulk velocity.

Examples
--------
python IC_analysis_turbbox.py RUN --save analysis/turb --report analysis/turb.pdf
python IC_analysis_turbbox.py SOL_RUN COMP_RUN --labels sol comp --grid-res 64
python IC_analysis_turbbox.py RUN --render --save analysis/turb_render
python IC_analysis_turbbox.py RUN --render --render-times 1 2 4 8 --render-z 0.5
"""

import json
import math
import os
import re
import sys
import textwrap

import numpy as np
import readtipsy as tip

from IC_analysis_framework import Param, standalone_cli
from IC_analysis_general import (
    find_files,
    read_log_series,
    read_vector_aux,
    setup_rcparams,
)
from sph_interp import UniformGrid, from_tipsy, interpolate


# Mirrors IC_setup_turbbox.B0_RMS: the fixed published seed strength, as an
# rms |B|. Both seed structures are normalised to it, so the initial magnetic
# energy 0.5*B0^2 below is correct for either.
SEED_B_RMS = math.sqrt(2.0) * 1.0e-5

PRICE2010 = "https://arxiv.org/abs/1004.1446"
TRICCO2016 = "https://arxiv.org/abs/1605.08662"
PHANTOM_PAPER = "https://arxiv.org/abs/1702.03930"


def forcing_name(value):
    """Solenoidal weight zeta in [0,1]; mirrors IC_setup_turbbox."""
    key = str(value).strip().lower()
    words = {"solenoidal": 1.0, "sol": 1.0, "s": 1.0,
             "compressive": 0.0, "comp": 0.0, "c": 0.0}
    if key in words:
        zeta = words[key]
    else:
        try:
            zeta = float(key)
        except ValueError:
            raise ValueError("forcing must be a solenoidal weight in [0,1] "
                             "(1=solenoidal, 0=compressive)") from None
    if not (math.isfinite(zeta) and 0.0 <= zeta <= 1.0):
        raise ValueError("forcing (solenoidal weight) must lie in [0,1]")
    return zeta


def forcing_label(zeta):
    """Human label for a projection weight."""
    if zeta == 1.0:
        return "solenoidal"
    if zeta == 0.0:
        return "compressive"
    return "mixed zeta=%g" % zeta


def compressive_fraction(zeta):
    """Injected compressive power fraction, ((1-z)^2)/((1-z)^2+2 z^2)."""
    zeta = float(zeta)
    return (1.0 - zeta) ** 2 / ((1.0 - zeta) ** 2 + 2.0 * zeta ** 2)


def lognormal_b(zeta):
    """Density-PDF b for a projection weight.

    Anchored on the published b=1/3 (solenoidal), ~0.4 (natural mixture,
    zeta=0.5) and 1 (compressive), monotonically interpolated in the
    compressive fraction. This is an interpolation between reference values,
    not a derived relation; the plotted curve is a reference, not a
    prediction.
    """
    return 1.0 / 3.0 + (2.0 / 3.0) * compressive_fraction(zeta) ** 2.0959


def case_physics(value):
    """(mhd, seed field) for a case number; mirrors IC_setup_turbbox."""
    cases = {"0": (0, "uniform"), "1": (1, "uniform"), "2": (1, "sin")}
    try:
        return cases[str(value).strip().lower()]
    except KeyError:
        raise ValueError("case (-c) must be 0 (hydro), 1 (MHD uniform Bz) or "
                         "2 (MHD zero-net-flux sin Bz)") from None


def case_id(value):
    """Validate and normalise a case number."""
    case_physics(value)
    return int(float(str(value).strip()))


def field_name(value):
    """Canonical seed-field structure accepted by setup and analysis."""
    key = str(value).strip().lower()
    if key in ("uniform", "u", "net", "netflux", "0", "0.0"):
        return "uniform"
    if key in ("sin", "sine", "nonetflux", "nnf", "1", "1.0"):
        return "sin"
    raise ValueError("seed field must be uniform/net/0 or sin/nonetflux/1")


def binary_flag(value):
    """Parse the shell-facing 0/1 MHD selector without accepting 1.7."""
    f = float(value)
    if f not in (0.0, 1.0):
        raise ValueError("MHD selector must be 0 or 1")
    return int(f)


_NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def _forcing_from_runname(run_name):
    match = re.search(r"M[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
                      r"(sol|comp|z[0-9.]+)(?:hydro|mhd)", run_name,
                      re.IGNORECASE)
    if match is None:
        return None
    tag = match.group(1).lower()
    if tag == "sol":
        return 1.0
    if tag == "comp":
        return 0.0
    return float(tag[1:])


def _mhd_from_runname(run_name):
    match = re.search(r"M[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
                      r"(?:sol|comp|z[0-9.]+)(hydro|mhd)", run_name,
                      re.IGNORECASE)
    if match is None:
        return None
    return 1 if match.group(1).lower() == "mhd" else 0


def _seed_from_runname(run_name):
    """0 for the published sequence, otherwise the generated OU seed."""
    if re.search(r"seed\d+paper", run_name, re.IGNORECASE):
        return 0
    match = re.search(r"seed(\d+)", run_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _case_from_runname(run_name):
    """Recover the case number from the physics and seed-field tags."""
    mhd = _mhd_from_runname(run_name)
    if mhd is None:
        return None
    if not mhd:
        return 0
    seed = re.search(r"B(uniform|sin)tc", run_name, re.IGNORECASE)
    if seed is None:
        return 1            # legacy names carried a numeric B amplitude
    return 1 if seed.group(1).lower() == "uniform" else 2


PARAM_SPEC = [
    Param("a", "mach", 10.0, float, "nominal rms sonic Mach number",
          re.compile(r"M" + _NUM + r"(?:sol|comp|z[0-9.]+)")),
    Param("b", "forcing", 1.0, forcing_name,
          "OU projection: solenoidal weight zeta in [0,1] (1=sol, 0=comp)",
          _forcing_from_runname),
    Param("c", "case", 1, case_id,
          "0=hydro, 1=MHD uniform net-flux Bz, 2=MHD zero-net-flux sin Bz",
          _case_from_runname),
    Param("d", "crossings", 100.0, float, "requested duration in crossings",
          re.compile(r"tc" + _NUM + r"seed")),
    Param("e", "seed", 0, lambda x: int(float(x)),
          "forcing realisation: 0 = published file, N>=1 = generated OU seed N",
          _seed_from_runname),
]


def crossing_time(mach, sound_speed=1.0, box_length=1.0):
    """t_cross=L/(2 M c_s), the Tricco et al. convention."""
    mach = float(mach)
    if not np.isfinite(mach) or mach <= 0.0:
        raise ValueError("mach must be finite and positive")
    return float(box_length) / (2.0 * mach * float(sound_speed))


def density_weighted_velocity(rho, velocity, power):
    """Return ``rho**power * velocity`` with an explicit vector-first shape."""
    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,) + rho.shape:
        raise ValueError("velocity must have shape (3,) + rho.shape")
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("density must be finite and non-negative")
    return np.power(rho, float(power))[None, ...] * velocity


def _snapshot_time(raw):
    return float(np.asarray(raw).ravel()[0])


def snapshot_diagnostics(filename, sound_speed=1.0):
    """Mass-weighted, bulk-subtracted one-snapshot turbulence diagnostics.

    For equal-mass SPH particles, the mass-weighted rms values below are the
    particle-wise rms values used for Phantom in Tricco et al. (2016), Fig. 5.
    In particular, ``alfven_mach_rms`` is rms of the *local* ``|dv|/v_A`` and
    is not ``vrms/va_rms``.  The latter is retained separately to make the
    distinction auditable.
    """
    tgdata, _td, _ts, header, raw_time = tip.readtipsy(filename)
    ngas = int(header[2])
    gas = np.asarray(tgdata[:ngas], dtype=float)
    if ngas == 0:
        raise ValueError(f"no gas particles in {filename}")
    mass = gas[:, 0]
    vel = gas[:, 4:7]
    rho = gas[:, 7]
    mtot = float(np.sum(mass))
    if not mtot > 0.0:
        raise ValueError(f"non-positive gas mass in {filename}")
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError(f"non-positive/non-finite gas density in {filename}")
    bulk = np.sum(mass[:, None] * vel, axis=0) / mtot
    dv = vel - bulk
    v2 = np.sum(dv * dv, axis=1)
    vrms = math.sqrt(float(np.sum(mass * v2) / mtot))
    ekin = 0.5 * float(np.sum(mass * v2))
    B = read_vector_aux(filename, "BField", ngas)
    if B is None:
        emag = float("nan")
        brms = float("nan")
        va_rms = float("nan")
        alfven_mach_rms = float("nan")
        alfven_mach_rms_raw = float("nan")
        vrms_over_va_rms = float("nan")
    else:
        B = np.asarray(B, dtype=float)
        b2 = np.sum(B * B, axis=1)
        volume = mass / rho
        emag = 0.5 * float(np.sum(volume * b2))
        vtot = float(np.sum(volume))
        brms = math.sqrt(float(np.sum(volume * b2) / vtot))
        va2 = b2 / rho
        va_rms = math.sqrt(float(np.sum(mass * va2) / mtot))
        moving_without_field = (va2 <= 0.0) & (v2 > 0.0)
        raw_v2 = np.sum(vel * vel, axis=1)
        raw_moving_without_field = (va2 <= 0.0) & (raw_v2 > 0.0)
        if np.any(moving_without_field):
            alfven_mach_rms = float("inf")
        else:
            local_ma2 = np.divide(v2, va2, out=np.zeros_like(v2),
                                  where=va2 > 0.0)
            alfven_mach_rms = math.sqrt(
                float(np.sum(mass * local_ma2) / mtot))
        if np.any(raw_moving_without_field):
            alfven_mach_rms_raw = float("inf")
        else:
            local_ma2_raw = np.divide(raw_v2, va2,
                                      out=np.zeros_like(raw_v2),
                                      where=va2 > 0.0)
            alfven_mach_rms_raw = math.sqrt(
                float(np.sum(mass * local_ma2_raw) / mtot))
        vrms_over_va_rms = (vrms / va_rms if va_rms > 0.0
                            else float("inf"))
    return {
        "time": _snapshot_time(raw_time),
        "npart": ngas,
        "mass": mtot,
        "bulk_vx": float(bulk[0]),
        "bulk_vy": float(bulk[1]),
        "bulk_vz": float(bulk[2]),
        "bulk_speed": float(np.linalg.norm(bulk)),
        "vrms": vrms,
        "mach": vrms / float(sound_speed),
        "ekin": ekin,
        "emag": emag,
        "brms": brms,
        "va_rms": va_rms,
        "va_rms_over_cs": va_rms / float(sound_speed),
        "alfven_mach_rms": alfven_mach_rms,
        "alfven_mach_rms_raw": alfven_mach_rms_raw,
        "vrms_over_va_rms": vrms_over_va_rms,
        "emag_over_ekin": (emag / ekin if np.isfinite(emag) and ekin > 0.0
                            else float("nan")),
        "rho_mean_particle": float(np.mean(rho)),
        "rho_max": float(np.max(rho)),
    }


def snapshot_series(input_arg, sound_speed=1.0):
    """Ordered snapshot diagnostics for one run/prefix."""
    try:
        files = find_files(input_arg)
    except FileNotFoundError:
        return [], {}
    rows = []
    for filename in files:
        try:
            row = snapshot_diagnostics(filename, sound_speed=sound_speed)
        except Exception as error:
            print(f"turbbox: failed to read {filename}: {error}", file=sys.stderr)
            continue
        row["filename"] = filename
        rows.append(row)
    rows.sort(key=lambda item: item["time"])
    if not rows:
        return [], {}
    keys = [key for key in rows[0] if key != "filename"]
    table = {key: np.asarray([row[key] for row in rows], dtype=float)
             for key in keys}
    table["files"] = [row["filename"] for row in rows]
    return rows, table


def _late_window(table, tcross, explicit_start=None):
    """Descriptive late-time selection; it never asserts stationarity."""
    if not table or "time" not in table or not len(table["time"]):
        return np.zeros(0, dtype=bool), float("nan"), False
    tau = np.asarray(table["time"]) / tcross
    end = float(np.max(tau))
    if explicit_start is not None:
        start = float(explicit_start)
    elif end >= 20.0:
        start = end - 10.0
    else:
        start = 0.5 * end
    mask = tau >= start
    # Five crossings is a minimal decorrelation span, not a convergence proof.
    plausible = bool(end - start >= 5.0 and end >= 5.0)
    return mask, start, plausible


def history_figure(inputs, labels, params, args, series_by_input):
    """Mach and kinetic/magnetic energy histories in crossing-time units."""
    import matplotlib.pyplot as plt

    setup_rcparams()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    tcross = crossing_time(params["mach"])
    cmap = plt.get_cmap("tab10")
    for irun, (inp, label) in enumerate(zip(inputs, labels)):
        color = cmap(irun % 10)
        _rows, tab = series_by_input[inp]
        if tab:
            tau = tab["time"] / tcross
            axes[0].plot(tau, tab["mach"], "o-", color=color, ms=3,
                         label=f"{label}: mass-weighted snapshots")
            espec = tab["ekin"] / tab["mass"]
            ekeep = np.isfinite(espec) & (espec > 0.0)
            axes[1].semilogy(tau[ekeep], espec[ekeep], "o-", color=color,
                             ms=3, label=f"{label}: $E_{{kin}}/M$")
            if np.any(np.isfinite(tab["emag"])) and np.nanmax(tab["emag"]) > 0:
                emspec = tab["emag"] / tab["mass"]
                emkeep = np.isfinite(emspec) & (emspec > 0.0)
                axes[1].semilogy(tau[emkeep], emspec[emkeep], "s--",
                                 color=color, ms=3,
                                 label=f"{label}: $E_{{mag}}/M$")

        # The dense code history is useful, but differs from the headline
        # measure: it is particle-count weighted and does not remove bulk flow.
        log = read_log_series(inp, ["dTime", "Ekin", "Emag", "rmsmach"])
        if log is not None and len(log["dTime"]):
            axes[0].plot(log["dTime"] / tcross, log["rmsmach"], ":",
                         color=color, alpha=0.55,
                         label=f"{label}: Gasoline log diagnostic")
            mass0 = tab["mass"][0] if tab else 1.0
            log_espec = log["Ekin"] / mass0
            log_keep = np.isfinite(log_espec) & (log_espec > 0.0)
            axes[1].semilogy(log["dTime"][log_keep] / tcross,
                             log_espec[log_keep], ":", color=color, alpha=0.45)

    axes[0].axhline(params["mach"], color="k", ls="--", lw=1.3,
                    label=f"nominal input $\mathcal{{M}}={params['mach']:g}$")
    axes[0].set_ylabel(r"rms sonic Mach number")
    axes[0].set_title("OU turbulence: achieved Mach number and energy")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel(r"$t/t_{cross}$")
    axes[1].set_ylabel("specific energy [code units]")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        handles, labs = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def _window_statistics(time_crossings, values, width=20.0, blocks=4,
                       logarithmic=False):
    """Descriptive late-window trend and block scatter for one time series."""
    x = np.asarray(time_crossings, dtype=float)
    y = np.asarray(values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if logarithmic:
        good &= y > 0.0
    x, y = x[good], y[good]
    if not len(x):
        return {}
    order = np.argsort(x)
    x, y = x[order], y[order]
    end = float(x[-1])
    start = max(float(x[0]), end - float(width))
    keep = x >= start
    xw, yw = x[keep], y[keep]
    span = float(xw[-1] - xw[0]) if len(xw) > 1 else 0.0
    mean = float(np.mean(yw))
    std = float(np.std(yw))
    slope = float("nan")
    relative_trend = float("nan")
    if len(xw) >= 2 and span > 0.0:
        fit_y = np.log(yw) if logarithmic else yw
        slope = float(np.polyfit(xw, fit_y, 1)[0])
        if logarithmic:
            relative_trend = float(np.expm1(slope * span))
        elif mean != 0.0:
            relative_trend = float(slope * span / mean)

    block_means = []
    if span > 0.0 and blocks > 0:
        edges = np.linspace(float(xw[0]), float(xw[-1]), int(blocks) + 1)
        for iblock in range(int(blocks)):
            if iblock + 1 == int(blocks):
                in_block = (xw >= edges[iblock]) & (xw <= edges[iblock + 1])
            else:
                in_block = (xw >= edges[iblock]) & (xw < edges[iblock + 1])
            if np.any(in_block):
                block_means.append(float(np.mean(yw[in_block])))
    block_cv = float("nan")
    first_last = float("nan")
    if len(block_means) >= 2:
        block_mean = float(np.mean(block_means))
        if block_mean != 0.0:
            block_cv = float(np.std(block_means, ddof=1) / abs(block_mean))
            first_last = float((block_means[-1] - block_means[0])
                               / abs(block_mean))
    return {
        "window_start_crossings": start,
        "window_end_crossings": end,
        "window_span_crossings": span,
        "samples": int(len(xw)),
        "mean": mean,
        "standard_deviation": std,
        "slope_per_crossing": slope,
        "relative_trend_across_window": relative_trend,
        "block_means": block_means,
        "block_coefficient_of_variation": block_cv,
        "first_to_last_block_relative_change": first_last,
    }


def convergence_diagnostics(table, tcross, target_mach, initial_B0,
                            window=20.0):
    """Quantify late Mach/energy stationarity and a magnetic-energy plateau."""
    if not table:
        return {}
    tau = np.asarray(table["time"], dtype=float) / float(tcross)
    ekin_specific = table["ekin"] / table["mass"]
    emag_specific = table["emag"] / table["mass"]
    mach = _window_statistics(tau, table["mach"], width=window)
    ekin = _window_statistics(tau, ekin_specific, width=window)
    emag = _window_statistics(tau, emag_specific, width=window,
                              logarithmic=True)
    ratio = _window_statistics(tau, table["emag_over_ekin"], width=window,
                               logarithmic=True)

    def finite_metric(record, key, default=float("nan")):
        value = record.get(key, default)
        return float(value) if np.isfinite(value) else default

    mach_bias = (finite_metric(mach, "mean") / float(target_mach) - 1.0
                 if target_mach > 0.0 else float("nan"))
    span_mach = finite_metric(mach, "window_span_crossings", 0.0)
    mach_stationary = bool(
        span_mach >= 19.0
        and abs(finite_metric(mach, "relative_trend_across_window")) <= 0.10
        and finite_metric(mach, "block_coefficient_of_variation") <= 0.10)
    ekin_stationary = bool(
        finite_metric(ekin, "window_span_crossings", 0.0) >= 19.0
        and abs(finite_metric(ekin, "relative_trend_across_window")) <= 0.15
        and finite_metric(ekin, "block_coefficient_of_variation") <= 0.15)
    magnetic_plateau = bool(
        finite_metric(emag, "window_span_crossings", 0.0) >= 19.0
        and abs(finite_metric(emag, "relative_trend_across_window")) <= 0.25
        and finite_metric(emag, "block_coefficient_of_variation") <= 0.20)
    initial_emag = 0.5 * float(initial_B0) ** 2
    amplification = (finite_metric(emag, "mean") / initial_emag
                     if initial_emag > 0.0 else float("nan"))
    return {
        "late_window_crossings": float(window),
        "mach": mach,
        "kinetic_energy": ekin,
        "magnetic_energy": emag,
        "magnetic_to_kinetic_ratio": ratio,
        "mach_relative_bias_from_driving_input": mach_bias,
        "mach_within_10_percent_of_driving_input": bool(
            np.isfinite(mach_bias) and abs(mach_bias) <= 0.10),
        "mach_stationary_by_declared_window_test": mach_stationary,
        "kinetic_energy_stationary_by_declared_window_test": ekin_stationary,
        "magnetic_energy_plateau_by_declared_window_test": magnetic_plateau,
        "initial_specific_magnetic_energy_from_B0": initial_emag,
        "late_magnetic_amplification_factor": amplification,
    }


def energy_bookkeeping_diagnostics(input_arg):
    """Check the printed Gasoline energy identity, not physical conservation."""
    names = ("dTime", "Etot", "Ekin", "Epot", "Eth", "Emag", "EClean")
    log = read_log_series(input_arg, names)
    if log is None or not len(log["dTime"]):
        return {}
    component_sum = (log["Ekin"] + log["Epot"] + log["Eth"]
                     + log["Emag"] + log["EClean"])
    residual = log["Etot"] - component_sum
    dynamic_scale = (np.abs(log["Ekin"]) + np.abs(log["Epot"])
                     + np.abs(log["Emag"]) + np.abs(log["EClean"]))
    dynamic_good = dynamic_scale > 0.0
    max_dynamic_relative = (float(np.max(np.abs(residual[dynamic_good])
                                         / dynamic_scale[dynamic_good]))
                            if np.any(dynamic_good) else float("nan"))
    etot_scale = np.maximum(np.abs(log["Etot"]), 1.0)
    eth0 = float(log["Eth"][0])
    eth_relative_range = (float(np.ptp(log["Eth"]) / abs(eth0))
                          if eth0 != 0.0 else float("nan"))
    return {
        "samples": int(len(residual)),
        "max_absolute_printed_closure_residual": float(np.max(np.abs(residual))),
        "max_closure_residual_over_total_energy": float(
            np.max(np.abs(residual) / etot_scale)),
        "max_closure_residual_over_nonthermal_energy": max_dynamic_relative,
        "thermal_energy_relative_range": eth_relative_range,
        "interpretation": ("Etot=sum of printed components to log precision; "
                           "this is bookkeeping closure, not conservation in "
                           "an externally driven isothermal system"),
    }


def alfven_history_figure(inputs, labels, params, series_by_input):
    """Tricco et al. (2016) Fig. 5-style particle-wise rms histories."""
    import matplotlib.pyplot as plt

    setup_rcparams()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    tcross = crossing_time(params["mach"])
    cmap = plt.get_cmap("tab10")
    for irun, (inp, label) in enumerate(zip(inputs, labels)):
        _rows, tab = series_by_input[inp]
        if not tab or "alfven_mach_rms" not in tab:
            continue
        color = cmap(irun % 10)
        tau = tab["time"] / tcross
        ma = np.asarray(tab["alfven_mach_rms"])
        va = np.asarray(tab["va_rms_over_cs"])
        keep_ma = np.isfinite(ma) & (ma > 0.0)
        keep_va = np.isfinite(va) & (va > 0.0)
        ax.semilogy(tau[keep_ma], ma[keep_ma], color=color, lw=1.5,
                    label=rf"{label}: rms local $M_A$")
        ax.semilogy(tau[keep_va], va[keep_va], color=color, lw=1.5, ls="--",
                    label=rf"{label}: rms $v_A/c_s$")
    ax.axhline(20.0, color="0.35", ls=":", lw=1.0,
               label=r"TPF16 saturated rms $M_A\sim20$")
    ax.axhline(2.0, color="0.55", ls="-.", lw=1.0,
               label=r"TPF16 saturated rms $v_A/c_s\sim2$")
    ax.set_xlabel(r"$t/t_{cross}$")
    ax.set_ylabel("particle-wise rms")
    ax.set_title("Alfvenic evolution (Tricco et al. 2016, Fig. 5 definition)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    return fig


def energy_growth_figure(inputs, labels, params, series_by_input):
    """Tricco et al. Fig. 2-style kinetic/magnetic growth and saturation."""
    import matplotlib.pyplot as plt

    setup_rcparams()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    tcross = crossing_time(params["mach"])
    initial_emag = 0.5 * SEED_B_RMS ** 2
    cmap = plt.get_cmap("tab10")
    for irun, (inp, label) in enumerate(zip(inputs, labels)):
        _rows, tab = series_by_input[inp]
        if not tab:
            continue
        color = cmap(irun % 10)
        tau = tab["time"] / tcross
        ekin = tab["ekin"] / tab["mass"]
        emag = tab["emag"] / tab["mass"]
        ek = np.isfinite(ekin) & (ekin > 0.0)
        em = np.isfinite(emag) & (emag > 0.0)
        axes[0].semilogy(tau[ek], ekin[ek], color=color, lw=1.5,
                         label=rf"{label}: $E_{{kin}}/M$")
        axes[0].semilogy(np.r_[0.0, tau[em]], np.r_[initial_emag, emag[em]],
                         color=color, lw=1.5, ls="--",
                         label=rf"{label}: $E_{{mag}}/M$")
        ratio = np.asarray(tab["emag_over_ekin"])
        good_ratio = np.isfinite(ratio) & (ratio > 0.0)
        axes[1].semilogy(tau[good_ratio], ratio[good_ratio], color=color,
                         lw=1.5, label=label)
    axes[0].set_ylabel("specific energy [code units]")
    axes[0].set_title("Driven kinetic and magnetic energy")
    axes[0].legend(fontsize=8.5, ncol=2)
    axes[1].axhspan(0.02, 0.04, color="0.65", alpha=0.25,
                    label="TPF16 saturated range (2--4%)")
    axes[1].set_xlabel(r"$t/t_{cross}$")
    axes[1].set_ylabel(r"$E_{mag}/E_{kin}$")
    axes[1].set_title("Magnetic-to-kinetic energy ratio")
    axes[1].legend(fontsize=8.5, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def infer_grid_resolution(npart):
    """Twice the nearest cubic native resolution (Tricco et al. 2016)."""
    native = max(2, int(round(float(npart) ** (1.0 / 3.0))))
    return 2 * native


def petkova_grid(filename, mhd, grid_res=None, nsmooth=64):
    """Deposit the requested fields on a periodic unit grid using Petkova."""
    fields = ["rho", "vx", "vy", "vz"]
    if mhd:
        fields.extend(["Bx", "By", "Bz"])
    p = from_tipsy(filename, fields=tuple(fields), nsmooth=int(nsmooth),
                   periodic=True, box=np.ones(3))
    ngrid = infer_grid_resolution(p.n) if grid_res is None else int(grid_res)
    if ngrid < 4:
        raise ValueError("grid resolution must be at least 4")
    estimated_gib = len(fields) * ngrid ** 3 * 8.0 / 2.0 ** 30
    print(f"turbbox: Petkova {ngrid}^3 grid for {os.path.basename(filename)} "
          f"({p.n} particles; field payload ~{estimated_gib:.2f} GiB)")
    target = UniformGrid.centered(np.ones(3), npx=ngrid)
    result = interpolate(p, target, method="petkova", values=fields)
    particle_mass = float(np.sum(p.mass))
    grid_mass = float(np.sum(result.data["_mass"]))
    mass_error = grid_mass / particle_mass - 1.0
    print(f"turbbox: Petkova mass closure {mass_error:+.3e}")
    return result, p, mass_error


def _rfft_geometry(shape, box=(1.0, 1.0, 1.0)):
    """Mode-number geometry and Hermitian weights for an rfftn."""
    nx, ny, nz = (int(v) for v in shape)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.rfftfreq(nz) * nz
    k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2
          + kz[None, None, :] ** 2)
    weight = np.ones(kz.size, dtype=float)
    if kz.size > 1:
        weight[1:] = 2.0
        if nz % 2 == 0:
            weight[-1] = 1.0
    return kx, ky, kz, k2, weight[None, None, :]


def isotropic_vector_spectrum(vector, box=(1.0, 1.0, 1.0)):
    """Shell-integrated total/solenoidal/compressive vector power.

    Parseval normalization makes ``sum_k E(k)`` equal
    ``0.5*mean(|v|^2)`` apart from the removed k=0 mode.  Shells are centered
    on integer dimensionless mode number kL/(2 pi).
    """
    arr = np.asarray(vector, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("vector must have shape (3,nx,ny,nz)")
    arr = arr - np.mean(arr, axis=(1, 2, 3), keepdims=True)
    fft = [np.fft.rfftn(arr[j]) for j in range(3)]
    kx, ky, kz, k2, hweight = _rfft_geometry(arr.shape[1:], box=box)
    norm = float(np.prod(arr.shape[1:])) ** 2
    total_mode = (np.abs(fft[0]) ** 2 + np.abs(fft[1]) ** 2
                  + np.abs(fft[2]) ** 2)
    dot = (kx[:, None, None] * fft[0] + ky[None, :, None] * fft[1]
           + kz[None, None, :] * fft[2])
    comp_mode = np.zeros_like(total_mode)
    nz = k2 > 0
    comp_mode[nz] = np.abs(dot[nz]) ** 2 / k2[nz]
    sol_mode = np.maximum(total_mode - comp_mode, 0.0)
    shell = np.rint(np.sqrt(k2)).astype(np.int64)
    nmax = min(arr.shape[1:]) // 2

    def collapse(mode):
        weighted = 0.5 * mode * hweight / norm
        return np.bincount(shell.ravel(), weights=weighted.ravel(),
                           minlength=nmax + 1)[:nmax + 1]

    etot = collapse(total_mode)
    esol = collapse(sol_mode)
    ecomp = collapse(comp_mode)
    kval = np.arange(nmax + 1, dtype=float)
    keep = kval >= 1.0
    return {"k": kval[keep], "total": etot[keep],
            "solenoidal": esol[keep], "compressive": ecomp[keep]}


def _spectral_slope(k, energy, kmin=4.0, kmax=None):
    k = np.asarray(k, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if kmax is None:
        kmax = max(kmin, float(np.max(k)) / 4.0)
    good = ((k >= kmin) & (k <= kmax) & np.isfinite(energy)
            & (energy > 0.0))
    if np.count_nonzero(good) < 3:
        return float("nan"), int(np.count_nonzero(good))
    slope = np.polyfit(np.log(k[good]), np.log(energy[good]), 1)[0]
    return float(slope), int(np.count_nonzero(good))


def _guide(ax, k, energy, exponent, label, style="--"):
    resolved_max = float(np.max(k)) / 4.0
    resolved = ((k >= 4.0) & (k <= resolved_max)
                & np.isfinite(energy) & (energy > 0.0))
    if np.count_nonzero(resolved) < 3:
        return
    good = np.flatnonzero((k >= 3.0) & np.isfinite(energy) & (energy > 0.0))
    if not len(good):
        return
    anchor = good[min(1, len(good) - 1)]
    guide = energy[anchor] * (k / k[anchor]) ** exponent
    ax.loglog(k, guide, style, lw=1.1, color="0.35", label=label)


def magnetic_dynamo_slope(b_spec, kmin=2.0):
    """Sub-peak magnetic slope, fitted from kmin to the measured spectral peak.

    Kazantsev k^(3/2) is the INCOMPRESSIBLE, kinematic, sub-viscous solution
    for a delta-correlated velocity field at Pm >~ 1.  A supersonic box
    satisfies none of that: the velocity field is Burgers-like, the density
    contrast is large, and SPH gives an uncontrolled Pm ~ 1.  The guide drawn
    with this slope is a subsonic reference, NOT an expectation at Mach 10,
    and it only appears when the magnetic peak has separated from the forcing
    scale far enough to leave a sub-peak band.  This is a measurement, not a
    test; magnetic_minus_kinetic_slope is the regime-agnostic diagnostic."""
    k = np.asarray(b_spec["k"], dtype=float)
    energy = np.asarray(b_spec["total"], dtype=float)
    good = np.isfinite(energy) & (energy > 0.0)
    result = {"magnetic_spectral_peak_k": float("nan"),
              "magnetic_spectral_slope_below_peak": float("nan"),
              "magnetic_slope_bins": 0}
    if np.count_nonzero(good) < 3:
        return result
    result["magnetic_spectral_peak_k"] = float(
        k[good][int(np.argmax(energy[good]))])
    band = good & (k >= kmin) & (k <= result["magnetic_spectral_peak_k"])
    if np.count_nonzero(band) < 3:
        return result
    result["magnetic_spectral_slope_below_peak"] = float(
        np.polyfit(np.log(k[band]), np.log(energy[band]), 1)[0])
    result["magnetic_slope_bins"] = int(np.count_nonzero(band))
    return result


def spectra_figure(grid, mhd):
    """Price/Tricco-style spectra and Helmholtz content."""
    import matplotlib.pyplot as plt

    rho = np.asarray(grid.data["rho"], dtype=float)
    velocity = np.stack([grid.data["vx"], grid.data["vy"], grid.data["vz"]])
    v_spec = isotropic_vector_spectrum(velocity)
    density_velocity = density_weighted_velocity(rho, velocity, 1.0 / 3.0)
    rv_spec = isotropic_vector_spectrum(density_velocity)
    b_spec = None
    kinetic_spec = None
    dynamo = {}
    if mhd and all(name in grid.data for name in ("Bx", "By", "Bz")):
        bfield = np.stack([grid.data["Bx"], grid.data["By"], grid.data["Bz"]])
        if np.any(np.abs(bfield) > 0.0):
            b_spec = isotropic_vector_spectrum(bfield)
            kinetic_spec = isotropic_vector_spectrum(
                density_weighted_velocity(rho, velocity, 0.5))

    setup_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    k = v_spec["k"]
    axes[0, 0].loglog(k, v_spec["total"], "o-", ms=3, label=r"$E_v(k)$")
    axes[0, 0].loglog(k, rv_spec["total"], "s-", ms=3,
                      label=r"$E_{\rho^{1/3}v}(k)$")
    _guide(axes[0, 0], k, v_spec["total"], -2.0, r"$k^{-2}$")
    _guide(axes[0, 0], k, rv_spec["total"], -5.0 / 3.0, r"$k^{-5/3}$", ":")
    axes[0, 0].set_title("Velocity spectra (Price & Federrath 2010)")
    axes[0, 0].set_ylabel("shell-integrated power")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].loglog(k, v_spec["solenoidal"], "-", label="solenoidal")
    axes[0, 1].loglog(k, v_spec["compressive"], "--", label="compressive")
    axes[0, 1].loglog(k, v_spec["total"], ":", color="k", label="total")
    axes[0, 1].set_title("Velocity Helmholtz components")
    axes[0, 1].legend(fontsize=9)

    denom = v_spec["solenoidal"] + v_spec["compressive"]
    frac = np.divide(v_spec["compressive"], denom,
                     out=np.full_like(denom, np.nan), where=denom > 0)
    axes[1, 0].semilogx(k, frac, "o-", ms=3)
    axes[1, 0].axhline(1.0 / 3.0, color="0.4", ls="--", lw=1.0,
                       label="isotropic 1:2 mode count")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylabel(r"$E_{comp}/(E_{comp}+E_{sol})$")
    axes[1, 0].set_title("Compressive kinetic-mode fraction")
    axes[1, 0].legend(fontsize=9)

    if b_spec is not None:
        axes[1, 1].loglog(kinetic_spec["k"], kinetic_spec["total"],
                          label=r"$P(\rho^{1/2}v)$: kinetic")
        axes[1, 1].loglog(b_spec["k"], b_spec["total"],
                          label=r"$P(B)$: magnetic")
        dynamo = magnetic_dynamo_slope(b_spec)
        peak_k = dynamo["magnetic_spectral_peak_k"]
        title = "TPF16 kinetic and magnetic spectra"
        if dynamo["magnetic_slope_bins"] >= 3:
            kb = np.asarray(b_spec["k"], dtype=float)
            band = (kb >= 2.0) & (kb <= peak_k)
            anchor = int(np.flatnonzero(band)[0])
            guide = (np.asarray(b_spec["total"])[anchor]
                     * (kb[band] / kb[anchor]) ** 1.5)
            axes[1, 1].loglog(kb[band], guide, "--", lw=1.1, color="0.35",
                              label=r"Kazantsev $k^{3/2}$")
            title += ("\nsub-peak $P(B)\\propto k^{%.2f}$ to $k_{peak}=%g$"
                      % (dynamo["magnetic_spectral_slope_below_peak"], peak_k))
        axes[1, 1].set_title(title, fontsize=10)
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].loglog(k, np.maximum(k ** 2 * v_spec["total"], 1e-300),
                          label=r"$k^2 E_v$")
        axes[1, 1].loglog(k, np.maximum(k ** (5.0 / 3.0) * rv_spec["total"],
                                       1e-300),
                          label=r"$k^{5/3}E_{\rho^{1/3}v}$")
        axes[1, 1].set_title("Compensated spectra")
        axes[1, 1].legend(fontsize=9)

    for ax in axes.ravel():
        ax.set_xlabel(r"mode $kL/(2\pi)$")
        ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    slope_v, nfit_v = _spectral_slope(k, v_spec["total"])
    slope_rv, nfit_rv = _spectral_slope(k, rv_spec["total"])
    positive = denom > 0
    comp_fraction = (float(np.sum(v_spec["compressive"][positive])
                           / np.sum(denom[positive])) if np.any(positive)
                     else float("nan"))
    metrics = {
        "velocity_spectral_slope": slope_v,
        "velocity_slope_bins": nfit_v,
        "density_weighted_velocity_spectral_slope": slope_rv,
        "density_weighted_velocity_slope_bins": nfit_rv,
        "compressive_mode_fraction": comp_fraction,
        "mhd_kinetic_spectrum_field": ("rho^(1/2) v"
                                       if kinetic_spec is not None else None),
    }
    metrics.update(dynamo)
    if b_spec is not None:
        # Same band as the velocity slopes, so the two are directly
        # comparable: a magnetic slope shallower than the kinetic one is the
        # small-scale amplification signature that survives at resolutions
        # too low to expose a Kazantsev sub-peak range.
        slope_b, nfit_b = _spectral_slope(b_spec["k"], b_spec["total"])
        slope_kin, _n = _spectral_slope(kinetic_spec["k"],
                                        kinetic_spec["total"])
        metrics.update(magnetic_spectral_slope=slope_b,
                       magnetic_slope_fit_bins=nfit_b,
                       mhd_kinetic_spectral_slope=slope_kin,
                       magnetic_minus_kinetic_slope=slope_b - slope_kin)
    return fig, metrics


def _safe_log10(a, floor=None):
    a = np.asarray(a, dtype=float)
    positive = a[np.isfinite(a) & (a > 0.0)]
    if floor is None:
        floor = float(np.min(positive)) if len(positive) else 1e-300
    return np.log10(np.maximum(a, floor))


def maps_figure(grid, mhd, time_crossings):
    """Density/velocity/B maps in the visual style of the reference studies."""
    import matplotlib.pyplot as plt

    rho = np.asarray(grid.data["rho"], dtype=float)
    vx, vy, vz = (np.asarray(grid.data[k], dtype=float)
                  for k in ("vx", "vy", "vz"))
    n = rho.shape[0]
    mid = n // 2
    dz = float(grid.target.pixwidth[2])
    sigma = np.sum(rho, axis=2) * dz
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    extent = [grid.target.lo[0], grid.target.hi[0],
              grid.target.lo[1], grid.target.hi[1]]

    setup_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    panels = [
        (_safe_log10(sigma).T, r"$\log_{10}\Sigma$", "column density"),
        (_safe_log10(rho[:, :, mid]).T, r"$\log_{10}\rho$", "mid-plane density"),
        (_safe_log10(speed[:, :, mid]).T, r"$\log_{10}(|v|/c_s)$",
         "mid-plane speed"),
    ]
    if mhd and all(k in grid.data for k in ("Bx", "By", "Bz")):
        bmag = np.sqrt(grid.data["Bx"] ** 2 + grid.data["By"] ** 2
                       + grid.data["Bz"] ** 2)
        panels.append((_safe_log10(bmag[:, :, mid]).T, r"$\log_{10}|B|$",
                       "mid-plane magnetic field"))
    else:
        dx = float(grid.target.pixwidth[0])
        dy = float(grid.target.pixwidth[1])
        dvdx = lambda a, axis, spacing: (
            np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2 * spacing)
        wx = dvdx(vz, 1, dy) - dvdx(vy, 2, dz)
        wy = dvdx(vx, 2, dz) - dvdx(vz, 0, dx)
        wz = dvdx(vy, 0, dx) - dvdx(vx, 1, dy)
        vort = np.sqrt(wx * wx + wy * wy + wz * wz)
        panels.append((_safe_log10(vort[:, :, mid]).T,
                       r"$\log_{10}|\nabla\times v|$", "mid-plane vorticity"))

    for ax, (image, cbar_label, title) in zip(axes.ravel(), panels):
        artist = ax.imshow(image, origin="lower", extent=extent,
                           interpolation="nearest", cmap="magma", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        fig.colorbar(artist, ax=ax, label=cbar_label, shrink=0.86)
    fig.suptitle(f"OU turbulence at $t/t_{{cross}}={time_crossings:.3g}$")
    fig.tight_layout()
    return fig


def _normal_pdf(x, mean, sigma):
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * sigma)


def statistics_figure(grid, mhd, forcing, achieved_mach):
    """Volume/mass density PDFs and MHD B statistics."""
    import matplotlib.pyplot as plt

    rho = np.asarray(grid.data["rho"], dtype=float)
    rho_bar = float(np.mean(rho))
    s = np.log(np.maximum(rho / rho_bar, 1e-300)).ravel()
    finite = np.isfinite(s)
    s = s[finite]
    sigma_s = float(np.std(s))
    mean_s = float(np.mean(s))
    b_theory = lognormal_b(forcing)
    sigma_theory = math.sqrt(math.log1p((b_theory * achieved_mach) ** 2))
    lo, hi = np.percentile(s, [0.2, 99.8]) if len(s) else (-1.0, 1.0)
    if not hi > lo:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, 61)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volume_pdf, _ = np.histogram(s, bins=edges, density=True)
    mass_weights = np.exp(s)
    mass_pdf, _ = np.histogram(s, bins=edges, weights=mass_weights,
                               density=True)

    setup_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    axes[0, 0].plot(centers, volume_pdf, drawstyle="steps-mid",
                    label="volume weighted")
    axes[0, 0].plot(centers, _normal_pdf(centers, mean_s, sigma_s), "--",
                    label=rf"Gaussian fit $\sigma_s={sigma_s:.3g}$")
    axes[0, 0].plot(
        centers, _normal_pdf(centers, -0.5 * sigma_theory ** 2, sigma_theory),
        ":", label=rf"interpolated $b={b_theory:.3g}$ ($\zeta={forcing:g}$)")
    axes[0, 0].set_ylabel("volume PDF")
    axes[0, 0].set_title("Log-density PDF")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(centers, mass_pdf, drawstyle="steps-mid", color="tab:orange",
                    label="mass weighted")
    axes[0, 1].plot(centers,
                    _normal_pdf(centers, mean_s + sigma_s ** 2, sigma_s),
                    "--", color="0.35", label="shifted Gaussian guide")
    axes[0, 1].set_ylabel("mass PDF")
    axes[0, 1].set_title("Mass-weighted log-density PDF")
    axes[0, 1].legend(fontsize=9)

    magnetic_metrics = {"brho_log_slope": float("nan"),
                        "magnetic_log_variance": float("nan")}
    if mhd and all(k in grid.data for k in ("Bx", "By", "Bz")):
        b2 = (np.asarray(grid.data["Bx"]) ** 2
              + np.asarray(grid.data["By"]) ** 2
              + np.asarray(grid.data["Bz"]) ** 2)
        flat_rho = rho.ravel()
        flat_b = np.sqrt(b2.ravel())
        good = (flat_rho > 0.0) & (flat_b > 0.0) & np.isfinite(flat_b)
        lr = np.log10(flat_rho[good])
        lb = np.log10(flat_b[good])
        # Deterministic thinning keeps a high-resolution hexbin manageable.
        if len(lr) > 250000:
            idx = np.linspace(0, len(lr) - 1, 250000).astype(int)
            lr_plot, lb_plot = lr[idx], lb[idx]
        else:
            lr_plot, lb_plot = lr, lb
        hb = axes[1, 0].hexbin(lr_plot, lb_plot, gridsize=55, mincnt=1,
                               bins="log", cmap="viridis")
        fig.colorbar(hb, ax=axes[1, 0], label=r"$\log_{10}N_{cell}$")
        redges = np.linspace(np.min(lr), np.max(lr), 25)
        rc, bc = [], []
        for j in range(len(redges) - 1):
            mask = (lr >= redges[j]) & (lr < redges[j + 1])
            if np.count_nonzero(mask) >= 8:
                rc.append(0.5 * (redges[j] + redges[j + 1]))
                bc.append(float(np.median(lb[mask])))
        if len(rc) >= 2:
            axes[1, 0].plot(rc, bc, "w-", lw=2.0, label="binned median")
            magnetic_metrics["brho_log_slope"] = float(
                np.polyfit(np.asarray(rc), np.asarray(bc), 1)[0])
        axes[1, 0].set_xlabel(r"$\log_{10}\rho$")
        axes[1, 0].set_ylabel(r"$\log_{10}|B|$")
        axes[1, 0].set_title("Magnetic amplification: B-rho")
        axes[1, 0].legend(fontsize=9)

        positive_b2 = b2[b2 > 0.0]
        q = np.log(np.maximum(positive_b2 / np.mean(positive_b2), 1e-300))
        magnetic_metrics["magnetic_log_variance"] = float(np.var(q))
        axes[1, 1].hist(q.ravel(), bins=60, density=True, histtype="step",
                        color="tab:purple")
        axes[1, 1].set_xlabel(r"$\ln(B^2/\langle B^2\rangle_V)$")
        axes[1, 1].set_ylabel("volume PDF")
        axes[1, 1].set_title("Magnetic-energy-density PDF")
    else:
        for ax in axes[1, :]:
            ax.axis("off")
        axes[1, 0].text(0.5, 0.5,
                        "Hydrodynamic run\n(no magnetic diagnostics)",
                        transform=axes[1, 0].transAxes, ha="center", va="center")

    for ax in axes[0, :]:
        ax.set_xlabel(r"$s=\ln(\rho/\langle\rho\rangle_V)$")
        ax.grid(alpha=0.2)
    fig.suptitle("Turbulent density and magnetic statistics")
    fig.tight_layout()
    metrics = {
        "log_density_mean": mean_s,
        "log_density_sigma": sigma_s,
        "log_density_sigma_stationary_model": sigma_theory,
        "stationary_model_b": b_theory,
        **magnetic_metrics,
    }
    return fig, metrics


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "run"


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _save_figure(fig, path, dpi=160):
    _ensure_parent(path)
    fig.savefig(path, bbox_inches="tight", dpi=int(dpi))
    print(f"saved {path}")


def _report_cover(params, args, summaries):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    title = "OU-driven SPH turbulence-box report"
    body = f"""
Generated by IC_analysis_turbbox.py

Configuration
  nominal Mach number: {params['mach']:g}
  command-line forcing default: {forcing_label(params['forcing'])}
  command-line physics default: {'MHD' if params['mhd'] else 'hydrodynamics'}
  per-run forcing/physics tags override these defaults in mixed reports
  case: {params['case']} -> {'MHD' if params['mhd'] else 'hydro'}, seed field {params['field']} (fixed rms |B| = {SEED_B_RMS:g})
  requested duration: {params['crossings']:g} crossing times
  forcing realisation: {'published file' if params['seed'] == 0 else f"generated, OU seed {params['seed']}"}
  crossing/correlation time: {crossing_time(params['mach']):.8g}

Forcing contract
  X(t+dt) = exp(-dt/T) X(t) + sqrt(1-exp(-2dt/T)) xi
  P^zeta_ij = zeta delta_ij + (1-2 zeta) k_i k_j/k^2
  zeta=1 is purely solenoidal; zeta=0 is purely compressive.
  Modes span 1 <= kL/(2 pi) <= 3 with the published parabolic envelope.
  StEnergy = 8 (Mach/10)^3 for L=c_s=1; this is an input calibration,
  not a closed-loop controller. Achieved Mach is always measured.

Analysis contract
  All particle-to-grid products use the testsuite Petkova exact kernel-volume
  integration. Density deposition is mass conserving. Velocity and magnetic
  fields are Petkova kernel-volume-weighted cell averages. The default mesh is
  twice the inferred native particle resolution, following Tricco et al.

Interpretation guard
  A short pilot demonstrates execution and diagnostics only. Spectral slopes,
  density-PDF width, and dynamo growth require a resolution and stationary-time
  study before physical interpretation.

Primary references
  Price & Federrath (2010): {PRICE2010}
  Tricco, Price & Federrath (2016): {TRICCO2016}
  Phantom methods paper: {PHANTOM_PAPER}
"""
    fig.text(0.08, 0.94, title, fontsize=19, weight="bold", va="top")
    fig.text(0.08, 0.90, body.strip(), fontsize=10.3, family="monospace",
             va="top", linespacing=1.32)
    if summaries:
        y = 0.21
        fig.text(0.08, y, "Measured run summaries", fontsize=13, weight="bold")
        y -= 0.025
        for summary in summaries:
            physics = "MHD" if summary["mhd"] else "hydro"
            line = (f"{summary['label']} [{forcing_label(summary['forcing'])}, {physics}]: "
                    f"t/tcross={summary['time_crossings']:.3g}, "
                    f"Mach={summary['mach_snapshot']:.4g}, "
                    f"late mean={summary['mach_late_mean']:.4g}, "
                    f"Petkova mass err={summary.get('petkova_mass_relative_error', float('nan')):+.2e}; "
                    f"{summary['late_window_status']}")
            wrapped = textwrap.wrap(line, width=105)
            fig.text(0.08, y, "\n".join(wrapped), fontsize=9.2, va="top")
            y -= 0.020 * len(wrapped) + 0.012
    return fig


def _report_validation(validation):
    """Render structured, measured acceptance evidence on one PDF page."""
    import matplotlib.pyplot as plt

    if not isinstance(validation, dict):
        raise ValueError("validation JSON must contain an object")
    provenance = validation.get("provenance", [])
    checks = validation.get("checks", [])
    limitations = validation.get("limitations", [])
    if not isinstance(provenance, list) or not isinstance(checks, list) \
            or not isinstance(limitations, list):
        raise ValueError(
            "validation provenance/checks/limitations must be arrays")

    lines = []

    def append_wrapped(text, prefix="", width=102):
        wrapped = textwrap.wrap(str(text), width=max(20, width - len(prefix))) \
            or [""]
        lines.append(prefix + wrapped[0])
        continuation = " " * len(prefix)
        lines.extend(continuation + line for line in wrapped[1:])

    lines.append("PROVENANCE")
    for item in provenance:
        append_wrapped(item, "  - ")
    lines.append("")
    lines.append("MEASURED ACCEPTANCE CHECKS")
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("each validation check must be an object")
        name = item.get("name", "unnamed check")
        status = item.get("status", "UNSPECIFIED")
        append_wrapped(f"[{status}] {name}", "  - ")
        evidence = item.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            raise ValueError("validation check evidence must be a string or array")
        for measurement in evidence:
            append_wrapped(measurement, "      ", width=98)
    lines.append("")
    lines.append("LIMITATIONS / INTERPRETATION")
    for item in limitations:
        append_wrapped(item, "  - ")

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95,
             validation.get("title", "Measured implementation validation"),
             fontsize=17, weight="bold", va="top")
    fig.text(0.08, 0.91, "\n".join(lines), fontsize=8.0,
             family="monospace", va="top", linespacing=1.24)
    return fig


def _load_validation(path):
    with open(path, encoding="utf-8") as stream:
        validation = json.load(stream)
    if not isinstance(validation, dict):
        raise ValueError("validation JSON must contain an object")
    return validation


def write_report(path, params, args, summaries, figures, validation=None):
    """Write a self-contained multipage PDF from already-generated figures."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    _ensure_parent(path)
    cover = _report_cover(params, args, summaries)
    validation_page = (_report_validation(validation)
                       if validation is not None else None)
    with PdfPages(path) as pdf:
        pdf.savefig(cover, bbox_inches="tight")
        if validation_page is not None:
            pdf.savefig(validation_page, bbox_inches="tight")
        for _title, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
        info = pdf.infodict()
        info["Title"] = "OU-driven SPH turbulence-box report"
        info["Author"] = "IC_analysis_turbbox.py"
        info["Subject"] = "Price & Federrath (2010); Tricco et al. (2016) diagnostics"
    plt.close(cover)
    if validation_page is not None:
        plt.close(validation_page)
    print(f"saved {path}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reference_table(input_arg, params, _args):
    """Compact normalized history for future regression baselines."""
    _rows, table = snapshot_series(input_arg)
    if not table:
        return None
    tcross = crossing_time(params["mach"])
    return {
        "x": table["time"] / tcross,
        "mach": table["mach"],
        "ekin_specific": table["ekin"] / table["mass"],
        "emag_specific": np.nan_to_num(table["emag"] / table["mass"], nan=0.0),
        "alfven_mach_rms": np.nan_to_num(table["alfven_mach_rms"], nan=0.0,
                                            posinf=0.0, neginf=0.0),
        "va_rms_over_cs": np.nan_to_num(table["va_rms_over_cs"], nan=0.0,
                                          posinf=0.0, neginf=0.0),
    }


def add_arguments(parser):
    from turbbox_render import DEFAULT_RENDER_TIMES
    parser.set_defaults(render_times=list(DEFAULT_RENDER_TIMES))
    for action in parser._actions:
        if action.dest == "render":
            action.help = "render only rho/|B| slice and z-column time sequences"
        elif action.dest == "render_times":
            action.help = "slice/column times in crossing units (default: 1 2 4 8); warn and skip unavailable times"
    parser.add_argument("--campaign-output", metavar="DIR",
                        help="audited 32^3/64^3 published-forcing ensemble, figures and PDF")
    parser.add_argument("--history-cache", metavar="DIR",
                        help="reuse checked history_N.json/npz files in campaign mode")
    parser.add_argument("--ensemble-cache", metavar="DIR",
                        help="reuse/store campaign Petkova products in DIR/cache")
    parser.add_argument("--cache-source", metavar="DIR",
                        help="archived analysis sources for verifying older grid cache provenance")
    parser.add_argument("--ensemble-window", type=float, nargs=2, default=(80,100),
                        metavar=("START","END"), help="campaign averaging window in crossing times")
    parser.add_argument("--ensemble-samples", type=int, default=11,
                        help="uniformly selected campaign samples (default 11)")
    parser.add_argument("--comparison-run", metavar="DIR",
                        help="completed 128^3 one-crossing run to include in campaign maps")
    parser.add_argument("--paper-reference", metavar="DIR",
                        help="digitized PHANTOM 128^3 reference panels for the controlled view comparison")
    parser.add_argument("--paper-render", action="store_true",
                        help="render only the shared SPH rho/|B| slice sequence")
    parser.add_argument("--render-time-crossings", type=float, default=None,
                        help="single slice time overriding --render-times (crossing units)")
    parser.add_argument("--render-z", type=float, default=.5,
                        help="literal periodic slice plane (default 0.5)")
    parser.add_argument(
        "--grid-res", type=int, default=None, metavar="N",
        help="Petkova grid cells per axis (default 2x native linear resolution)")
    parser.add_argument(
        "--grid-nsmooth", type=int, default=64, metavar="N",
        help="neighbours used to reconstruct h if no smoothlength aux exists")
    parser.add_argument(
        "--snapshot-index", type=int, default=-1,
        help="snapshot index for gridded diagnostics (default -1, final)")
    parser.add_argument(
        "--stationary-after", type=float, default=None, metavar="TCROSS",
        help="start of descriptive late-time average in crossing-time units")
    parser.add_argument(
        "--no-grid", action="store_true",
        help="make only history/report pages; skip all Petkova products")
    parser.add_argument(
        "--report", help="write a multipage PDF report to this path")
    parser.add_argument(
        "--metrics-json", help="write measured scalar diagnostics as JSON")
    parser.add_argument(
        "--validation-json", metavar="PATH",
        help="include structured measured acceptance evidence as a report page")
    parser.add_argument("--dpi", type=int, default=160,
                        help="PNG resolution when --save is used (default 160)")


def _plot(inputs, labels, params, args):
    import matplotlib.pyplot as plt

    if args.campaign_output:
        if args.render or args.paper_render or args.no_grid:
            raise ValueError("campaign mode includes its paper comparison; "
                             "use --render separately for time sequences")
        from turbbox_report import run_campaign
        return run_campaign(inputs,args)
    if args.render or args.paper_render:
        from turbbox_render import render_sequence
        from turbbox_ensemble import write_json
        slice_times = ([args.render_time_crossings] if args.render_time_crossings is not None
                       else args.render_times)
        products = [("paper_midplane", "slice", slice_times)]
        if args.render:
            products.append(("zcolumn", "column", args.render_times))
        figures, metadata = [], {}
        nsmooth = args.render_nsmooth if args.render_nsmooth is not None else 64
        for slug, project, times in products:
            fig, record = render_sequence(
                inputs, labels=args.labels, times_crossings=times,
                resolution=args.render_res, zslice=args.render_z, nsmooth=nsmooth,
                project=project, tcross=crossing_time(params["mach"]))
            metadata[slug] = record
            if fig is None:
                continue
            figures.append(fig)
            if args.save:
                path = f"{args.save}_{slug}.png"
                _ensure_parent(path)
                with open(path,"xb") as stream:
                    fig.savefig(stream,format="png",dpi=args.dpi,bbox_inches="tight")
                write_json(f"{args.save}_{slug}.json",record)
        if args.metrics_json:
            write_json(args.metrics_json,metadata)
        if not figures:
            return
        if args.report:
            from matplotlib.backends.backend_pdf import PdfPages
            _ensure_parent(args.report)
            with open(args.report,"xb") as stream, PdfPages(stream) as pdf:
                for fig in figures:
                    pdf.savefig(fig,bbox_inches="tight")
        if not args.save and not args.report:
            plt.show()
        else:
            for fig in figures:
                plt.close(fig)
        return

    if params["mach"] <= 0.0:
        raise ValueError("mach must be positive")
    if params["crossings"] <= 0.0:
        raise ValueError("crossings must be positive")
    forcing = forcing_name(params["forcing"])
    params["forcing"] = forcing
    params["mhd"], params["field"] = case_physics(params["case"])
    tcross = crossing_time(params["mach"])
    series_by_input = {inp: snapshot_series(inp) for inp in inputs}
    if not any(table for _rows, table in series_by_input.values()):
        raise FileNotFoundError("no readable turbulence-box snapshots")

    figures = []
    history = history_figure(inputs, labels, params, args, series_by_input)
    figures.append(("history", history))
    if args.save:
        _save_figure(history, f"{args.save}_history.png", args.dpi)

    energy_growth = energy_growth_figure(
        inputs, labels, params, series_by_input)
    figures.append(("energy growth", energy_growth))
    if args.save:
        _save_figure(energy_growth, f"{args.save}_energy_growth.png", args.dpi)

    if any(table and np.any(np.isfinite(table.get("va_rms_over_cs", [])))
           for _rows, table in series_by_input.values()):
        alfven_history = alfven_history_figure(
            inputs, labels, params, series_by_input)
        figures.append(("Alfvenic history", alfven_history))
        if args.save:
            _save_figure(alfven_history,
                         f"{args.save}_alfven_history.png", args.dpi)

    summaries = []
    for inp, label in zip(inputs, labels):
        rows, table = series_by_input[inp]
        if not rows:
            print(f"turbbox: no readable snapshots for {inp}", file=sys.stderr)
            continue
        try:
            selected = rows[args.snapshot_index]
        except IndexError as error:
            raise IndexError(
                f"snapshot index {args.snapshot_index} is outside 0..{len(rows)-1} "
                f"for {inp}") from error
        late, late_start, stationary_span = _late_window(
            table, tcross, args.stationary_after)
        late_mach = float(np.mean(table["mach"][late])) if np.any(late) else float("nan")
        time_crossings = selected["time"] / tcross
        tagged_forcing = _forcing_from_runname(inp)
        run_forcing = tagged_forcing if tagged_forcing is not None else forcing
        tagged_case = _case_from_runname(inp)
        run_mhd = bool(case_physics(tagged_case)[0] if tagged_case is not None
                       else params["mhd"])
        status = ("late window spans >=5 crossings; stationarity still requires "
                  "trend/resolution checks" if stationary_span else
                  "short/non-stationary diagnostic window")
        summary = {
            "label": label,
            "input": inp,
            "snapshot": selected["filename"],
            "time": selected["time"],
            "time_crossings": time_crossings,
            "npart": selected["npart"],
            "target_mach": float(params["mach"]),
            "mach_snapshot": selected["mach"],
            "mach_late_mean": late_mach,
            "late_window_start_crossings": late_start,
            "late_window_status": status,
            "bulk_speed": selected["bulk_speed"],
            "ekin_specific": selected["ekin"] / selected["mass"],
            "emag_specific": selected["emag"] / selected["mass"],
            "emag_over_ekin": selected["emag_over_ekin"],
            "va_rms_over_cs": selected["va_rms_over_cs"],
            "alfven_mach_rms": selected["alfven_mach_rms"],
            "alfven_mach_rms_raw": selected["alfven_mach_rms_raw"],
            "vrms_over_va_rms": selected["vrms_over_va_rms"],
            "forcing": run_forcing,
            "mhd": int(run_mhd),
        }
        summary["temporal_convergence"] = convergence_diagnostics(
            table, tcross, params["mach"], SEED_B_RMS)
        summary["energy_bookkeeping"] = energy_bookkeeping_diagnostics(inp)
        print(f"turbbox[{label}]: final/selected Mach={selected['mach']:.6g}; "
              f"late mean={late_mach:.6g} from t/tcross>={late_start:.3g}; {status}")
        temporal = summary["temporal_convergence"]
        if temporal:
            print(
                f"turbbox[{label}]: late-window stationarity "
                f"Mach={temporal['mach_stationary_by_declared_window_test']} "
                f"Ekin={temporal['kinetic_energy_stationary_by_declared_window_test']} "
                f"Emag plateau={temporal['magnetic_energy_plateau_by_declared_window_test']}; "
                f"Mach target match={temporal['mach_within_10_percent_of_driving_input']}")

        if not args.no_grid:
            grid, _particles, mass_error = petkova_grid(
                selected["filename"], run_mhd,
                grid_res=args.grid_res, nsmooth=args.grid_nsmooth)
            summary["grid_resolution"] = int(grid.data["rho"].shape[0])
            summary["petkova_mass_relative_error"] = mass_error
            run_slug = _slug(label)

            fmap = maps_figure(grid, run_mhd, time_crossings)
            figures.append((f"{label}: maps", fmap))
            if args.save:
                _save_figure(fmap, f"{args.save}_{run_slug}_maps.png", args.dpi)

            fspec, spectrum_metrics = spectra_figure(grid, run_mhd)
            summary.update(spectrum_metrics)
            figures.append((f"{label}: spectra", fspec))
            if args.save:
                _save_figure(fspec, f"{args.save}_{run_slug}_spectra.png", args.dpi)

            fstat, statistic_metrics = statistics_figure(
                grid, run_mhd, run_forcing, selected["mach"])
            summary.update(statistic_metrics)
            figures.append((f"{label}: PDFs", fstat))
            if args.save:
                _save_figure(fstat, f"{args.save}_{run_slug}_statistics.png", args.dpi)

        summaries.append(summary)

    if args.metrics_json:
        _ensure_parent(args.metrics_json)
        with open(args.metrics_json, "w", encoding="utf-8") as stream:
            json.dump(_json_safe({"parameters": params, "runs": summaries}),
                      stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"saved {args.metrics_json}")

    if args.report:
        validation = (_load_validation(args.validation_json)
                      if args.validation_json else None)
        write_report(args.report, params, args, summaries, figures,
                     validation=validation)

    if not args.save and not args.report:
        plt.show()
    else:
        for _name, fig in figures:
            plt.close(fig)


def main():
    standalone_cli(
        "turbbox", PARAM_SPEC, _plot,
        reference_table=reference_table,
        reg_keys=("mach", "ekin_specific", "emag_specific"),
        add_time=False,
        add_arguments=add_arguments,
        description=("OU turbulence-box diagnostics following Price & Federrath "
                     "(2010) and Tricco et al. (2016); Petkova statistics and shared SPH visual slices."),
    )


if __name__ == "__main__":
    main()
