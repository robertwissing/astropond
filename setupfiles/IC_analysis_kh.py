#!/usr/bin/env python3
"""
Analysis for the Kelvin-Helmholtz (kh) testcase.

Extracts, per snapshot, the linear KH mode amplitude M (sin/cos projection of
the y-velocity with exponential weighting about the shear layers) and the mean
kinetic energy of the most energetic 5% of particles, versus time. The
hydrodynamic finite-layer growth line is overlaid on the mode-amplitude panel.
For a uniform magnetic field parallel to the perturbation wavevector, that rate
is reduced by the ideal incompressible vortex-sheet suppression factor (and is
labelled as an estimate, since this setup has finite-width, compressible shear
layers). A transverse field has no tension term for this two-dimensional mode.

Parameters mirror runtest.sh setup_kh() flag-for-flag with the same defaults, so
you declare the run the same way you launched it, e.g.

    python IC_analysis_kh.py test_cases/kh/kh128_..._GASOLINE -a 2.0 -b 0.4899

Reference (regression-baseline) workflow is handled by IC_analysis_framework:
  --save-reference  writes '<runname>_reference.json' next to the run;
  a normal run auto-discovers that sibling, overlays it, and prints residuals.

Usage:
    python IC_analysis_kh.py <dir-or-prefix> [-a..-h ...] [--save out] [--save-reference]
"""

import numpy as np

from IC_analysis_framework import (
    Param, Metric, RenderSpec, StreamSpec, run_cli,
    aux_component, aux_magnitude, aux_divB_error,
    thermal_pressure, entropy_function,
)


# -a..-h match runtest.sh setup_kh() (runtest.sh:344-367) with the same defaults.
PARAM_SPEC = [
    Param("a", "rhodiff", 2.0, float, "density contrast", "rhodiff"),
    Param("b", "mach", np.sqrt(6) / 5, float, "mach number of flow", "mach"),
    Param("c", "smooth", 1, float, "smooth density disc", "smooth"),
    Param("d", "B0", 0.0, float, "initial magnetic field strength", "B"),
    Param("e", "Bdir", 1, float,
          "field direction x=1 y=2 z=3", "Bdir"),
]

# Plotted quantities returned by analyze().
METRICS = [
    Metric("M",       "Linear mode amplitude", "t",    "log", "o"),
    Metric("Ekinmax", "Max kinetic energy",    "time", "log", "o"),
]

# Adiabatic index of the kh IC (IC_setup_kh.gamma), needed for the pressure /
# entropy renders (u = dTuFac * T is handled by the framework factories).
GAMMA = 5.0 / 3.0

# Field maps for --render: density, thermal pressure and entropy always;
# magnetic field only when the run is magnetised (B0>0, the -d flag). All in
# the x-y plane (tgdata cols 1, 2). Pressure P=(gamma-1) rho u should stay
# near-uniform (~przero) -- structure is numerical surface-tension error; the
# entropic function A=P/rho^gamma is conserved along adiabatic flow, so it
# traces the shear-layer MIXING (layer contrast rhodiff^gamma). Both are
# line-of-sight averages, linear scale. Density is also an average, not an
# integrated column: a column has different units and depends on the thin
# numerical z extent. Every map uses the central unit square so n_x<32 runs
# (whose IC box is deliberately taller in y) still tile at the same panel size.
# The B fields are read from the "BField" aux file (vector per gas particle;
# component files BFieldx/y/z), smoothed with the same line-of-sight average:
# |B| (positive, log, inferno) plus the signed components Bx,By,Bz (linear,
# zero-centered diverging map). divBerr = |div B| * h / |B| (DivB aux,
# smoothlength aux for h) is the
# dimensionless divergence-cleanliness diagnostic (log, inferno). render_spec is
# a function of the run params so the B maps appear only for magnetised runs.
KH_RENDER_EXTENT = (-0.5, 0.5, -0.5, 0.5)


def render_specs(params):
    specs = [
        RenderSpec(axis1=1, axis2=2, quantity=7, label="density",
                   log=True, project="average", extent=KH_RENDER_EXTENT),
        RenderSpec(1, 2, thermal_pressure(GAMMA), "pressure",
                   log=False, project="average", cmap="viridis",
                   extent=KH_RENDER_EXTENT),
        RenderSpec(1, 2, entropy_function(GAMMA), "entropy",
                   log=False, project="average", cmap="magma",
                   extent=KH_RENDER_EXTENT),
    ]
    if params.get("B0", 0.0) > 0.0:
        bdiv = dict(log=False, project="average", cmap="RdBu_r",
                    symmetric=True, extent=KH_RENDER_EXTENT)
        specs += [
            RenderSpec(1, 2, aux_magnitude("BField"), "|B|",
                       log=True, project="average", extent=KH_RENDER_EXTENT),
            RenderSpec(1, 2, aux_component("BField", 0), "Bx", **bdiv),
            RenderSpec(1, 2, aux_component("BField", 1), "By", **bdiv),
            RenderSpec(1, 2, aux_component("BField", 2), "Bz", **bdiv),
            RenderSpec(1, 2, aux_divB_error("DivB", "BField"), "divBerr",
                       log=True, project="average",
                       clim=lambda params, proj: (0.001, 0.1),
                       extent=KH_RENDER_EXTENT),
        ]
    return specs


# Streamlines for --render (framework streamline_panels, generalized from the
# currentsheet streamline render): the in-plane velocity (vx, vy -- the rolled-up
# KH billows) always, and the in-plane magnetic field (Bx, By from the BField
# aux -- field lines stretched along the billows) for magnetised runs.
def stream_specs(params):
    specs = [StreamSpec(1, 2, 4, 5, "velocity")]
    if params.get("B0", 0.0) > 0.0:
        specs.append(StreamSpec(1, 2, aux_component("BField", 0),
                                aux_component("BField", 1), "B"))
    return specs

# Geometric constants of the KH IC (cap-packed shear layers at y = +-boundval,
# fundamental mode 4*pi). These are fixed by the setup geometry, not the -a..-h
# physical parameters, so they stay as documented constants.
MODE = 4 * np.pi
BOUNDVAL = 0.25
HYDRO_GROWTH_RATE = np.pi
GROWTH_ANCHOR_TIME = 0.35

# Synthetic anchor at t=0 matching the IC perturbation amplitude, so the
# measured growth curve starts where the analytic line does.
SEED = {"x": 0.0, "M": 1e-2, "Ekinmax": 1e-4}


def analyze_kh(tgdata):
    """Mode amplitude M and top-5% mean kinetic energy from one snapshot."""
    x = tgdata[:, 1]
    y = tgdata[:, 2]
    vy = tgdata[:, 5]
    Ekin = tgdata[:, 7] * vy ** 2 / 2.0
    h = 1.3 * (tgdata[:, 0] / tgdata[:, 7]) ** (1. / 3.)
    hvy = h * vy
    lower = (y < 0.0)
    upper = (y >= 0.0)

    s = np.zeros_like(vy)
    s[lower] = hvy[lower] * np.sin(MODE * x[lower]) * np.exp(-MODE * np.abs(y[lower] + BOUNDVAL))
    s[upper] = hvy[upper] * np.sin(MODE * x[upper]) * np.exp(-MODE * np.abs(-y[upper] + BOUNDVAL))
    c = np.zeros_like(vy)
    c[lower] = hvy[lower] * np.cos(MODE * x[lower]) * np.exp(-MODE * np.abs(y[lower] + BOUNDVAL))
    c[upper] = hvy[upper] * np.cos(MODE * x[upper]) * np.exp(-MODE * np.abs(-y[upper] + BOUNDVAL))
    d = np.zeros_like(vy)
    d[lower] = h[lower] * np.exp(-MODE * np.abs(y[lower] + BOUNDVAL))
    d[upper] = h[upper] * np.exp(-MODE * np.abs(-y[upper] + BOUNDVAL))

    M = 2 * np.sqrt((np.sum(s) / np.sum(d)) ** 2 + (np.sum(c) / np.sum(d)) ** 2)

    threshold = np.percentile(Ekin, 95)
    return M, Ekin[Ekin >= threshold].mean()


def analyze(tgdata, time, params, snap):
    """Framework callback: one row of metrics for a snapshot."""
    M, Ekinmax = analyze_kh(tgdata)
    return {"x": float(np.asarray(time).ravel()[0]), "M": M, "Ekinmax": Ekinmax}


def growth_prediction(params):
    """Return ``(gamma, Bcrit, label)`` for the plotted KH growth estimate.

    The measured smooth/compressible hydro benchmark uses ``gamma_HD = pi``.
    For a uniform field parallel to the mode wavevector (Bdir=1), multiply that
    calibrated finite-layer rate by the classical ideal, incompressible
    vortex-sheet suppression factor

        sqrt(1 - B_parallel^2 / Bcrit^2),

    where ``Bcrit^2 = rho1*rho2*DeltaU^2 / (2*(rho1+rho2))`` in the suite's
    mu0=1 magnetic units. This preserves the established hydro limit while
    supplying the physically meaningful magnetic stability threshold. It is an
    estimate rather than an eigenvalue solution for the finite-width layer.
    """
    rho1 = 1.0
    rho2 = float(params.get("rhodiff", 2.0))
    delta_u = 1.0                 # IC velocities are -0.5 and +0.5
    bcrit = np.sqrt(rho1 * rho2 * delta_u ** 2 /
                    (2.0 * (rho1 + rho2)))
    b0 = abs(float(params.get("B0", 0.0)))
    bdir = int(round(float(params.get("Bdir", 1))))
    if b0 <= 0.0:
        return HYDRO_GROWTH_RATE, bcrit, \
            rf"hydrodynamic growth prediction: $\gamma={HYDRO_GROWTH_RATE:.3g}$"
    if bdir != 1:
        return HYDRO_GROWTH_RATE, bcrit, \
            rf"transverse-field estimate ($k\!\cdot\!B=0$): $\gamma={HYDRO_GROWTH_RATE:.3g}$"
    suppression = np.sqrt(max(0.0, 1.0 - (b0 / bcrit) ** 2))
    gamma = HYDRO_GROWTH_RATE * suppression
    if gamma == 0.0:
        label = rf"parallel-field vortex-sheet estimate: stable ($B_{{crit}}={bcrit:.3g}$)"
    else:
        label = (rf"parallel-field vortex-sheet estimate: $\gamma={gamma:.3g}$, "
                 rf"$B_{{crit}}={bcrit:.3g}$")
    return gamma, bcrit, label


def analytic(metric_key, params):
    """Finite-layer hydro growth with an ideal-MHD tension correction."""
    if metric_key != "M":
        return None
    gamma, _bcrit, label = growth_prediction(params)
    t = np.linspace(0.65, 1.0, 200)
    return t, 0.01 * np.exp(gamma * (t - GROWTH_ANCHOR_TIME)), label


def main():
    run_cli(
        test="kh",
        param_spec=PARAM_SPEC,
        analyze=analyze,
        analytic=analytic,
        metrics=METRICS,
        description="Kelvin-Helmholtz mode-amplitude analysis vs time.",
        seed=SEED,
        render_spec=render_specs,
        stream_spec=stream_specs,
        show_legend=False,
    )


if __name__ == "__main__":
    main()
