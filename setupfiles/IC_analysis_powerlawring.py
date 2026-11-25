#!/usr/bin/env python3
"""Orbit and magnetic-field analysis for ``IC_setup_powerlawring``.

The hydro diagnostic compares the measured azimuthal speed with the exact
general-beta pressure-balanced speed.  In the flat-density science domain it
reduces to the softened circular speed. Magnetic mode 1 checks the steady
toroidal field: current-free ``B_phi=C/R`` in an annulus or its regularized
full-disk form.
Mode 2 checks the annulus-only kinematic winding solution

    B_phi(t) = (beta - 1) Omega B_R t,    B_R = C/R.

Mode 3 checks a uniform field normal to the orbital plane, another steady ideal
solution.  For spherical runs, positions and vectors are rotated back into the
initially tilted orbital plane before any cylindrical diagnostic is evaluated.
Cylindrical (``geometry=2``) runs have no vertical force and therefore no
vertical stratification; their expected rotation law is the same radial
equilibrium evaluated on the constant-density slab.
The optional full disk retains the central particles and compares its toroidal
field with the smooth ``C R/(R^2+epsilon^2)`` initial profile.
"""

import sys

import numpy as np

from IC_analysis_framework import Param, run_series, select_at_time, standalone_cli
from IC_analysis_general import (
    finish_figure_with_legend,
    loaddata,
    read_vector_aux,
    setup_rcparams,
)
from IC_setup_powerlawring import (
    DOMAIN_DISK,
    FORCE_BOUNDED_KEPLER,
    GEOMETRY_SPHERICAL,
    R_OUTER,
    circular_speed,
    domain_inner_radius,
    EQUILIBRIUM_SCALE_HEIGHT,
    equilibrium_azimuthal_speed,
    field_amplitude_from_plasma_beta,
    hopkins_scale_height,
    rotate_x,
    toroidal_field_strength,
    validated_geometry,
    validated_force_form,
)


DEFAULT_TIME = 2.0 * np.pi
PARAM_SPEC = [
    Param("a", "beta", -0.5, float, "circular-speed logarithmic slope", "beta"),
    Param("b", "geometry", 3, int,
          "2=cylindrical slab, 3=spherical equilibrium", "geom"),
    Param("c", "field_mode", 0, int,
          "0=hydro, 1=toroidal C/R, 2=radial winding, 3=vertical, "
          "4=constant toroidal (center-regular on a disk)", "fm"),
    Param("d", "plasma_beta", 1.0e6, float,
          "plasma beta 2P/B^2 at R=1 (sets the magnetic amplitude)", "pb"),
    Param("e", "softening", 0.0, float, "external-potential softening", "eps"),
    Param("f", "tilt", 0.0, float, "orbital-plane tilt in degrees", "tilt"),
    Param("g", "domain_mode", 0, int, "0=annulus, 1=full disk", "dom"),
    Param("h", "scale_height", EQUILIBRIUM_SCALE_HEIGHT, float,
          "barotrope scale height (sets the orbital Mach number)", "H"),
    Param("i", "force_form", 0, int,
          "0=power law (free edges), 1=bounded Kepler (Hopkins ring)",
          "form"),
]


def winding_rate(radius, beta, field_amp, softening=0.0):
    """Exact initial ``dB_phi/dt`` for ``B_R=field_amp/R``."""
    radius = np.asarray(radius, dtype=float)
    omega = circular_speed(radius, beta, softening) / radius
    return (beta - 1.0) * omega * field_amp / radius


def _scale_height(params):
    """The run's barotrope scale height, resolving the Hopkins override.

    ``create()`` ignores ``-h`` for the Hopkins ring and substitutes the
    published temperature, so the analyzer has to do the same or it would
    reconstruct the run with whatever ``-h`` happened to be on the command
    line.  Nothing in the ring's own diagnostics depends on H -- its density
    is flat, so the pressure term in v_phi vanishes and the test is hydro --
    but resolving it here keeps that a property of the code rather than a
    coincidence.
    """
    if validated_force_form(params.get("force_form", 0)) == FORCE_BOUNDED_KEPLER:
        return hopkins_scale_height()
    return params["scale_height"]


def _field_amp(params):
    """Amplitude ``C`` implied by the run's plasma beta.

    The analyzer must reproduce the IC's conversion exactly, so it calls the
    IC's own function rather than restating the algebra.
    """
    return field_amplitude_from_plasma_beta(
        params["plasma_beta"], params["field_mode"], params["softening"],
        params.get("domain_mode", 0), _scale_height(params))


def _weighted_mean(values, weights):
    denominator = np.sum(weights)
    return float(np.sum(weights * values) / denominator) if denominator > 0 else np.nan


def _validate_params(params):
    domain_mode = params.get("domain_mode", 0)
    domain_inner_radius(domain_mode)
    if domain_mode == DOMAIN_DISK and params["softening"] <= 0.0:
        raise ValueError("a full disk requires positive softening")
    if domain_mode == DOMAIN_DISK and params["field_mode"] == 2:
        raise ValueError("the C/R radial winding field is annulus-only")
    validated_geometry(params["geometry"])
    validated_force_form(params.get("force_form", 0))


def _state(snapshot, params):
    gas, _dark, _star, _header, time, _n, ngas, _nd, _ns, _h = loaddata(snapshot)
    position = gas[:, 1:4].astype(float)
    velocity = gas[:, 4:7].astype(float)
    if params["geometry"] == GEOMETRY_SPHERICAL:
        position = rotate_x(position, -params["tilt"])
        velocity = rotate_x(velocity, -params["tilt"])

    x, y = position[:, 0], position[:, 1]
    radius = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        vphi = np.where(radius > 0.0,
                        (x * velocity[:, 1] - y * velocity[:, 0]) / radius,
                        0.0)
        vr = np.where(radius > 0.0,
                      (x * velocity[:, 0] + y * velocity[:, 1]) / radius,
                      0.0)
    state = {
        "time": float(np.asarray(time).ravel()[0]),
        "radius": radius,
        "height": position[:, 2],
        "vphi": vphi,
        "vr": vr,
        "mass": gas[:, 0].astype(float),
        "rho": gas[:, 7].astype(float),
    }

    if params["field_mode"]:
        magnetic = read_vector_aux(snapshot, "BField", int(ngas))
        if magnetic is None:
            raise FileNotFoundError("missing BField auxiliary data")
        if params["geometry"] == GEOMETRY_SPHERICAL:
            magnetic = rotate_x(magnetic, -params["tilt"])
        with np.errstate(invalid="ignore", divide="ignore"):
            state["bphi"] = np.where(
                radius > 0.0,
                (x * magnetic[:, 1] - y * magnetic[:, 0]) / radius, 0.0)
            state["br"] = np.where(
                radius > 0.0,
                (x * magnetic[:, 0] + y * magnetic[:, 1]) / radius, 0.0)
            state["bnormal"] = magnetic[:, 2]
    return state


def _interior_mask(state, params):
    # Suppress free radial edges. A full disk has no inner boundary: retain the
    # center because its resolved MHD/dissipation behavior is the objective.
    domain_mode = params.get("domain_mode", 0)
    # The Hopkins ring needs no special case: it is the same sharp-edged
    # R=0.5..2 ring as the free-edge arm, with the same two free surfaces, so
    # it gets the same 10% edge trim.
    inner_radius = domain_inner_radius(domain_mode)
    width = R_OUTER - inner_radius
    inner_cut = (0.0 if domain_mode == DOMAIN_DISK
                 else inner_radius + 0.1 * width)
    return ((state["radius"] >= inner_cut) &
            (state["radius"] <= R_OUTER - 0.1 * width))


def _snapshot_row(_gas, _time, params, snapshot):
    state = _state(snapshot, params)
    mask = _interior_mask(state, params)
    if not np.any(mask):
        return None
    radius = state["radius"][mask]
    weights = np.where(state["rho"][mask] > 0.0,
                       state["mass"][mask] / state["rho"][mask], 0.0)
    expected_vphi = equilibrium_azimuthal_speed(
        radius, state["height"][mask], params["field_mode"],
        _field_amp(params), params["softening"],
        params.get("domain_mode", 0), params["beta"], params["geometry"],
        _scale_height(params), params.get("force_form", 0))
    row = {
        "x": state["time"],
        "L1_vphi": _weighted_mean(np.abs(state["vphi"][mask] - expected_vphi),
                                    weights),
        "rms_vr": np.sqrt(_weighted_mean(state["vr"][mask] ** 2, weights)),
        "plane_rms": np.sqrt(_weighted_mean(state["height"][mask] ** 2,
                                              weights)),
        "Lz_raw": float(np.sum(state["mass"] * state["radius"] *
                                state["vphi"])),
    }

    mode = params["field_mode"]
    if mode:
        if mode in (1, 4):
            # Mode 4 is the same toroidal geometry with a constant magnitude,
            # so it shares the profile function and this whole branch.
            field_scale = toroidal_field_strength(
                radius, _field_amp(params), params["softening"],
                params.get("domain_mode", 0), radial_taper=False,
                force_form=params.get("force_form", 0), field_mode=mode)
            expected_br = np.zeros_like(radius)
            expected_bphi = field_scale
            expected_bnormal = np.zeros_like(radius)
        elif mode == 2:
            field_scale = _field_amp(params) / radius
            expected_br = field_scale
            expected_bphi = winding_rate(
                radius, params["beta"], _field_amp(params),
                params["softening"]) * state["time"]
            expected_bnormal = np.zeros_like(radius)
        else:
            field_scale = np.full_like(radius, _field_amp(params))
            expected_br = np.zeros_like(radius)
            expected_bphi = np.zeros_like(radius)
            expected_bnormal = field_scale
        normalization = max(float(np.sqrt(_weighted_mean(field_scale ** 2,
                                                          weights))),
                            np.finfo(float).tiny)
        row["Br_error"] = np.sqrt(_weighted_mean(
            (state["br"][mask] - expected_br) ** 2, weights)) / normalization
        row["Bphi_error"] = np.sqrt(_weighted_mean(
            (state["bphi"][mask] - expected_bphi) ** 2, weights)) / normalization
        row["Bnormal_error"] = np.sqrt(_weighted_mean(
            (state["bnormal"][mask] - expected_bnormal) ** 2,
            weights)) / normalization

        denominator = _weighted_mean(expected_bphi ** 2, weights)
        if denominator > np.finfo(float).tiny:
            row["winding_ratio"] = (_weighted_mean(
                state["bphi"][mask] * expected_bphi, weights) / denominator)
        else:
            row["winding_ratio"] = 0.0
    return row


def diagnostic_series(input_arg, params):
    _validate_params(params)
    table = run_series(input_arg, _snapshot_row, params)
    if table is None:
        return None
    lz0 = table["Lz_raw"][0]
    table["Lz"] = table["Lz_raw"] / lz0 if lz0 != 0.0 else np.full_like(
        table["Lz_raw"], np.nan)
    del table["Lz_raw"]
    return table


def plot_series(inputs, labels, params, save=None, ref_table=None):
    import matplotlib.pyplot as plt

    setup_rcparams()
    figure, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    ax_orbit, ax_lz, ax_field = axes
    any_data = False
    for input_arg, label in zip(inputs, labels):
        table = diagnostic_series(input_arg, params)
        if table is None:
            continue
        line, = ax_orbit.plot(table["x"], table["L1_vphi"], "o-",
                              label=f"{label}: L1(vphi)")
        ax_orbit.plot(table["x"], table["rms_vr"], "s--",
                      color=line.get_color(), label=f"{label}: RMS(vr)")
        ax_lz.plot(table["x"], table["Lz"], "o-", color=line.get_color(),
                   label=label)
        if params["field_mode"]:
            ax_field.plot(table["x"], table["Bphi_error"], "o-",
                          color=line.get_color(), label=f"{label}: Bphi error")
            ax_field.plot(table["x"], table["Br_error"], "s--",
                          color=line.get_color(), label=f"{label}: Br error")
            ax_field.plot(table["x"], table["Bnormal_error"], "^:",
                          color=line.get_color(),
                          label=f"{label}: Bnormal error")
        else:
            ax_field.plot(table["x"], table["plane_rms"], "o-",
                          color=line.get_color(), label=label)
        print("powerlawring[%s]: final L1(vphi)=%.6g RMS(vr)=%.6g "
              "Lz/Lz0=%.6g" %
              (label, table["L1_vphi"][-1], table["rms_vr"][-1],
               table["Lz"][-1]))
        any_data = True

    if ref_table is not None and "x" in ref_table:
        for axis, keys in ((ax_orbit, ("L1_vphi", "rms_vr")),
                           (ax_lz, ("Lz",)),
                           (ax_field, ("Bphi_error", "Br_error",
                                       "Bnormal_error", "plane_rms"))):
            for key in keys:
                if key in ref_table:
                    axis.plot(ref_table["x"], ref_table[key], ":",
                              color="0.35", lw=1.5,
                              label="reference " + key)
    ax_lz.axhline(1.0, color="k", ls="--", lw=1.2, label="ideal")
    ax_orbit.set_ylabel("normalized velocity error")
    ax_lz.set_ylabel(r"$L_{plane}/L_{plane,0}$")
    ax_field.set_ylabel("normalized B error" if params["field_mode"] else
                        "RMS height")
    ax_field.set_xlabel("t")
    domain_name = ("full disk" if params.get("domain_mode", 0) == DOMAIN_DISK
                   else "annulus")
    ax_orbit.set_title("External power-law " + domain_name)
    figure.tight_layout()
    if not any_data:
        print("powerlawring: no series to plot", file=sys.stderr)
    finish_figure_with_legend(
        figure, axes,
        save=(f"{save}_series.png" if save else None))


def plot_profiles(inputs, labels, params, time_requested, save=None):
    import matplotlib.pyplot as plt

    setup_rcparams()
    magnetic = bool(params["field_mode"])
    figure, axes = plt.subplots(2 if magnetic else 1, 2,
                                figsize=(10, 8 if magnetic else 4.8),
                                squeeze=False)
    ax_vphi, ax_vr = axes[0]
    ax_bphi, ax_br = axes[1] if magnetic else (None, None)
    for input_arg, label in zip(inputs, labels):
        selected = select_at_time(input_arg, time_requested)
        if selected is None:
            continue
        snapshot, actual_time = selected
        state = _state(snapshot, params)
        mask = _interior_mask(state, params)
        radius = state["radius"][mask]
        ax_vphi.scatter(radius, state["vphi"][mask], s=3, alpha=0.35,
                        rasterized=True, label=f"{label}, t={actual_time:g}")
        ax_vr.scatter(radius, state["vr"][mask], s=3, alpha=0.35,
                      rasterized=True)
        if magnetic:
            ax_bphi.scatter(radius, state["bphi"][mask], s=3, alpha=0.35,
                            rasterized=True)
            second_field = (state["bnormal"][mask]
                            if params["field_mode"] == 3 else state["br"][mask])
            ax_br.scatter(radius, second_field, s=3, alpha=0.35,
                          rasterized=True)

    # Exactly what every cylindrical arm emits, Hopkins ring included.
    grid_inner = domain_inner_radius(params.get("domain_mode", 0))
    grid_outer = R_OUTER
    radius_grid = np.linspace(grid_inner, grid_outer, 400)
    expected_speed = equilibrium_azimuthal_speed(
        radius_grid, np.zeros_like(radius_grid), params["field_mode"],
        _field_amp(params), params["softening"],
        params.get("domain_mode", 0), params["beta"], params["geometry"],
        _scale_height(params), params.get("force_form", 0))
    ax_vphi.plot(radius_grid, expected_speed, "k-", label="analytic")
    ax_vr.axhline(0.0, color="k", lw=1.2)
    if magnetic:
        if params["field_mode"] in (1, 4):
            scale = toroidal_field_strength(
                radius_grid, _field_amp(params), params["softening"],
                params.get("domain_mode", 0), radial_taper=False,
                force_form=params.get("force_form", 0),
                field_mode=params["field_mode"])
            expected_bphi, expected_br = scale, np.zeros_like(scale)
        elif params["field_mode"] == 2:
            scale = _field_amp(params) / radius_grid
            expected_bphi = winding_rate(
                radius_grid, params["beta"], _field_amp(params),
                params["softening"]) * time_requested
            expected_br = scale
        else:
            scale = np.full_like(radius_grid, _field_amp(params))
            expected_bphi = np.zeros_like(scale)
            expected_br = np.full_like(scale, _field_amp(params))
        ax_bphi.plot(radius_grid, expected_bphi, "k-", label="analytic")
        ax_br.plot(radius_grid, expected_br, "k-")
        ax_bphi.set_ylabel(r"$B_\phi$")
        ax_br.set_ylabel(r"$B_n$" if params["field_mode"] == 3 else r"$B_R$")
        ax_bphi.set_xlabel("R")
        ax_br.set_xlabel("R")
    ax_vphi.set_ylabel(r"$v_\phi$")
    ax_vr.set_ylabel(r"$v_R$")
    ax_vphi.set_xlabel("R")
    ax_vr.set_xlabel("R")
    figure.tight_layout()
    finish_figure_with_legend(
        figure, axes.ravel(),
        save=(f"{save}_profile.png" if save else None))


def reference_table(input_arg, params, _args):
    return diagnostic_series(input_arg, params)


def _plot(inputs, labels, params, args):
    _validate_params(params)
    print("[powerlawring] beta=%g geometry=%d field_mode=%d "
          "plasma_beta=%g (C=%.4g) eps=%g tilt=%g domain=%d H=%g force=%d" %
          (params["beta"], params["geometry"], params["field_mode"],
           params["plasma_beta"], _field_amp(params), params["softening"],
           params["tilt"], params["domain_mode"], params["scale_height"],
           params["force_form"]))
    plot_profiles(inputs, labels, params, args.time, save=args.save)
    plot_series(inputs, labels, params, save=args.save,
                ref_table=getattr(args, "ref_table", None))


def main():
    standalone_cli(
        "powerlawring", PARAM_SPEC, _plot,
        reference_table=reference_table,
        description="External power-law disk/annulus and magnetic analysis.",
        reg_keys=("L1_vphi", "rms_vr", "Lz", "Bphi_error", "Br_error",
                  "Bnormal_error"),
        time_default=DEFAULT_TIME,
        time_help="profile time (default one unit-radius orbit, 2*pi)")


if __name__ == "__main__":
    main()
