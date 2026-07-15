#!/usr/bin/env python3
"""
Analysis for the 3D Ponomarenko screw-flow kinematic dynamo (IC_setup_ponomarenko),
from Teyssier, Fromang & Dormy (2006, JCP 218, 44). A helical "screw" flow
confined to a cylinder of radius R0=1 (solid-body rotation Omega=0.609711 plus
axial drift Vz=0.792624, quiescent outside) on a periodic box of side L=8.055
amplifies an m=1 helical seed field -- the classic Ponomarenko dynamo.

This script produces:
  * magnetic energy vs time on a LOG axis, E_mag/E_mag(0) (with E_kin/E_mag(0)) --
    a dynamo shows exponential growth E_mag ~ exp(2 sigma t); the kinematic
    growth rate sigma is fitted over the exponential window and printed/annotated;
  * the divergence error h|div B|/|B| vs time (mean and max) -- divergence
    robustness for a growing field;
  * field-map renders (--render): the magnetic energy density 1/2|B|^2 in the x-y
    plane (RAMSES compares the B^2/2 isosurface of the screw mode).

NOTE: in RAMSES this is a KINEMATIC dynamo (frozen velocity + explicit Ohmic
resistivity eta_mag=2.5e-3). The SPMHD codes here evolve the full MHD system, and
the screw flow is NOT a force-balanced equilibrium (the r=R0 shear layer is not
supported), so a dynamical run relaxes the flow rather than holding it fixed --
treat the growth here as a qualitative test of the induction solver and (artificial,
via dMHDAlphaB) magnetic dissipation rather than a quantitative match to Teyssier
et al. (2006).

It also supports the regression-baseline workflow (--save-reference / --reg-tol).

Usage:
    python IC_analysis_ponomarenko.py <run-dir> [-a B0] [--save out]
    python IC_analysis_ponomarenko.py <run-dir> --render --render-n 4 --save out
    python IC_analysis_ponomarenko.py <run-dir> --save-reference        # bless
    python IC_analysis_ponomarenko.py <run-dir> --reference --reg-tol 1e-6  # gate
"""

import sys

import numpy as np
import readtipsy as tip

from IC_analysis_framework import (
    Param, standalone_cli,
    RenderSpec, render_panels, aux_divB_error, magnetic_pressure,
)
from IC_analysis_general import (
    find_files, read_vector_aux, read_tufac, read_log_series,
    setup_rcparams, finish_figure_with_legend,
)

GAMMA = 5.0 / 3.0               # IC_setup_ponomarenko.gamma
BOXLEN = 8.055                 # periodic box side
REG_NPTS = 101                 # points on the regression time grid


# --------------------------------------------------------------------------- #
#  runtest.sh setup_ponomarenko() parameter order (B0). Carried for CLI/label
#  parity; the analysis normalises the energy by E_mag(0).
# --------------------------------------------------------------------------- #
PARAM_SPEC = [
    Param("a", "B0", 0.1, float, "seed vector-potential amplitude"),
]


# --------------------------------------------------------------------------- #
#  Per-snapshot diagnostics (snapshot fallback for the '.log' series)
# --------------------------------------------------------------------------- #

def _snapshot(fn):
    """(time, tgdata, ngas) for a snapshot."""
    tgdata, _td, _ts, hdr, time = tip.readtipsy(fn)
    return float(np.asarray(time).ravel()[0]), tgdata, int(hdr[2])


def divberr_of(fn):
    """(time, mean, max) of the divergence error h|div B|/|B| for snapshot fn.

    Mean is volume-weighted (V_i = m_i/rho_i); NaN if the DivB/BField aux is
    absent."""
    t, tgdata, ngas = _snapshot(fn)
    err = np.asarray(aux_divB_error()(fn, tgdata, ngas), float)
    if not np.any(err > 0):
        return t, np.nan, np.nan
    vol = tgdata[:, 0] / tgdata[:, 7]
    return t, float(np.sum(vol * err) / np.sum(vol)), float(np.max(err))


def energies_of(fn, tufac):
    """(time, E_mag, E_kin, E_therm) for snapshot fn.

    E_mag = sum 1/2 |B|^2 V (NaN if BField aux absent); E_kin = sum 1/2 m|v|^2;
    E_therm = sum m u with u = tufac*T (col 8 is temperature)."""
    t, tgdata, ngas = _snapshot(fn)
    m, rho = tgdata[:, 0], tgdata[:, 7]
    e_kin = float(np.sum(0.5 * m * np.sum(tgdata[:, 4:7]**2, axis=1)))
    e_therm = float(tufac * np.sum(m * tgdata[:, 8]))
    B = read_vector_aux(fn, "BField", ngas)
    if B is None:
        return t, np.nan, e_kin, e_therm
    b2 = np.sum(np.asarray(B, float)**2, axis=1)
    e_mag = float(np.sum(0.5 * b2 * (m / rho)))
    return t, e_mag, e_kin, e_therm


def _snapshot_rows(inp):
    """Sorted (t, divb_mean, divb_max, Emag, Ekin, Etherm) per snapshot, or None."""
    try:
        files = find_files(inp)
    except FileNotFoundError:
        return None
    tufac = read_tufac(inp)
    rows = []
    for fn in files:
        try:
            t, dmean, dmax = divberr_of(fn)
            _t, em, ek, et = energies_of(fn, tufac)
        except Exception as e:
            print(f"  failed on {fn}: {e}", file=sys.stderr)
            continue
        rows.append((t, dmean, dmax, em, ek, et))
    if not rows:
        return None
    a = np.array(rows, float)
    return a[np.argsort(a[:, 0])]


def series(inp):
    """Time-series diagnostics for one run as a dict, or None.

    Prefers the dense gasoline '.log' (every timestep) for the divergence error
    (divBerrAvg/Max) and energies (Emag/Ekin/Eth), falling back to the snapshots.
    Keys: 'divberr'=(t,mean,max), 'energy'=(t,Emag,Ekin,Etherm), 'src'={...}."""
    log = read_log_series(
        inp, ["dTime", "divBerrAvg", "divBerrMax", "Emag", "Ekin", "Eth"])
    snap = _snapshot_rows(inp)
    if log is None and snap is None:
        print(f"ponomarenko: no snapshots/.log for '{inp}'", file=sys.stderr)
        return None
    out = {"src": {}}
    if log is not None and np.any(log["divBerrAvg"] > 0):
        out["divberr"] = (log["dTime"], log["divBerrAvg"], log["divBerrMax"])
        out["src"]["divberr"] = "log"
    elif snap is not None:
        out["divberr"] = (snap[:, 0], snap[:, 1], snap[:, 2])
        out["src"]["divberr"] = "snapshots"
    if log is not None and np.any(log["Emag"] > 0):
        out["energy"] = (log["dTime"], log["Emag"], log["Ekin"], log["Eth"])
        out["src"]["energy"] = "log"
    elif snap is not None:
        out["energy"] = (snap[:, 0], snap[:, 3], snap[:, 4], snap[:, 5])
        out["src"]["energy"] = "snapshots"
    return out


def growth_rate(t, emag):
    """Kinematic growth rate sigma from a least-squares fit of ln(E_mag) vs t
    (E_mag ~ exp(2 sigma t)). Fitted over the finite, positive, growing span (up
    to the peak, before saturation/decay). NaN if < 2 usable points. Returns
    (sigma, tfit, efit) where (tfit, efit) is the fitted span."""
    t = np.asarray(t, float)
    e = np.asarray(emag, float)
    good = np.isfinite(t) & np.isfinite(e) & (e > 0)
    if good.sum() < 2:
        return np.nan, None, None
    tg, eg = t[good], e[good]
    ipk = int(np.argmax(eg))
    if ipk >= 1:
        tg, eg = tg[:ipk + 1], eg[:ipk + 1]
    if len(tg) < 2:
        return np.nan, tg, eg
    slope = np.polyfit(tg, np.log(eg), 1)[0]
    return 0.5 * slope, tg, eg


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #

def plot_energy_vs_time(runs, labels, save=None, ref_table=None):
    """Magnetic energy vs time on a LOG axis, normalised by E_mag(0): the
    Ponomarenko dynamo grows exponentially, E_mag ~ exp(2 sigma t). The fitted
    kinematic growth rate sigma is annotated; E_kin tracks the (relaxing) screw
    flow.

    When a reference is supplied its blessed E_mag / E_kin are overlaid (grey)."""
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(8, 6))
    if ref_table is not None and "Emag" in ref_table:
        ax.semilogy(ref_table["x"], ref_table["Emag"], "--", color="0.5", lw=1.4,
                    zorder=5, label="reference")
        if "Ekin" in ref_table:
            ax.semilogy(ref_table["x"], ref_table["Ekin"], ":", color="0.5",
                        lw=1.0, alpha=0.7, zorder=5)
    for s, lab in zip(runs, labels):
        if s is None or "energy" not in s:
            continue
        t, emag, ekin, _eth = s["energy"]
        e0 = emag[0]
        if not np.isfinite(e0) or e0 == 0:
            print(f"ponomarenko[{lab}]: no/zero initial E_mag; skipping.",
                  file=sys.stderr)
            continue
        style = "-" if s["src"].get("energy") == "log" else "o-"
        pre = f"{lab} " if len(runs) > 1 else ""
        line, = ax.semilogy(t, emag/e0, style, ms=3, label=f"{pre}$E_{{mag}}$")
        c = line.get_color()
        ax.semilogy(t, ekin/np.where(e0 != 0, e0, 1.0), ":", lw=1.2, color=c,
                    alpha=0.7, label=f"{pre}$E_{{kin}}$")
        sigma, tfit, efit = growth_rate(t, emag)
        if np.isfinite(sigma) and tfit is not None and len(tfit) >= 2:
            ax.semilogy(tfit, (efit[0]/e0) * np.exp(2*sigma*(tfit - tfit[0])),
                        "--", lw=1.4, color=c, alpha=0.9,
                        label=f"{pre}fit $\\sigma={sigma:.3g}$")
        print(f"ponomarenko[{lab}]: E_mag/E_mag0 {emag[0]/e0:.4g}->{emag[-1]/e0:.4g}"
              f"  growth rate sigma={sigma:.4g}"
              f"  ({len(t)} pts from {s['src'].get('energy')})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E / E_{mag}(0)$")
    ax.set_title(r"Ponomarenko dynamo: magnetic-energy growth ($E_{mag}\sim e^{2\sigma t}$)")
    fig.tight_layout()
    finish_figure_with_legend(fig, ax, save=(f"{save}_energy.png" if save else None))


def plot_divberr_vs_time(runs, labels, save=None, ref_table=None):
    """Mean and max of h|div B|/|B| vs time (divergence-handling robustness)."""
    import matplotlib.pyplot as plt
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(8, 6))
    if ref_table is not None and "divBerrMean" in ref_table:
        ax.semilogy(ref_table["x"], ref_table["divBerrMean"], "--", color="0.5",
                    lw=1.4, zorder=5, label="reference")
        if "divBerrMax" in ref_table:
            ax.semilogy(ref_table["x"], ref_table["divBerrMax"], ":", color="0.5",
                        lw=1.4, zorder=5)
    for s, lab in zip(runs, labels):
        if s is None or "divberr" not in s:
            continue
        t, mean, mx = s["divberr"]
        if np.all(np.isnan(mean)):
            print(f"ponomarenko[{lab}]: no DivB/BField aux; skipping.",
                  file=sys.stderr)
            continue
        src = s["src"].get("divberr", "snapshots")
        mstyle, xstyle = ("-", "--") if src == "log" else ("o-", "s--")
        line, = ax.semilogy(t, mean, mstyle, ms=3, label=f"{lab} mean")
        ax.semilogy(t, mx, xstyle, ms=3, color=line.get_color(), label=f"{lab} max")
        print(f"ponomarenko[{lab}]: divBerr mean {mean[0]:.3g}->{mean[-1]:.3g}  "
              f"max {mx[0]:.3g}->{mx[-1]:.3g}  ({len(t)} pts from {src})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$h\,|\nabla\!\cdot\!B|/|B|$")
    ax.set_title("Ponomarenko dynamo: divergence error vs time")
    fig.tight_layout()
    finish_figure_with_legend(fig, ax,
                              save=(f"{save}_divberr.png" if save else None))


# --------------------------------------------------------------------------- #
#  Field-map renders
# --------------------------------------------------------------------------- #

def render_specs():
    # B^2/2 in the x-y plane (the screw mode; RAMSES uses the B^2/2 isosurface).
    return [
        RenderSpec(0, 1, magnetic_pressure(), r"$\frac{1}{2}|B|^2$", True,
                   "average", None, "inferno", False),
    ]


def do_render(inputs, labels, args, save):
    """Render the magnetic energy density at the requested times."""
    entries = list(zip(labels, inputs))
    half = 0.5 * BOXLEN
    extent = args.render_extent if args.render_extent else [-half, half, -half, half]
    for sp, slug in zip(render_specs(), ("magpressure",)):
        render_panels(entries, sp, times=args.render_times, n=args.render_n,
                      res=args.render_res,
                      save=(f"{save}_{slug}" if save else None),
                      backend=args.render_backend, project=args.render_project,
                      slice_frac=args.render_slice_frac, extent=extent,
                      aspect=args.render_aspect)


# --------------------------------------------------------------------------- #
#  Regression baseline (CI/CD)
# --------------------------------------------------------------------------- #

def reference_table(inp):
    """Regression table on a fixed time grid: {x: t, Emag, Ekin, Eth,
    divBerrMean, divBerrMax} with the energies normalised by E_mag(0) (so the
    blessed curve is the amplification factor). None if no usable series."""
    s = series(inp)
    if s is None or "energy" not in s:
        return None
    t, emag, ekin, eth = s["energy"]
    e0 = emag[0] if (np.isfinite(emag[0]) and emag[0] != 0) else 1.0
    tg = np.linspace(float(t[0]), float(t[-1]), REG_NPTS)
    tbl = {"x": tg,
           "Emag": np.interp(tg, t, emag / e0),
           "Ekin": np.interp(tg, t, ekin / e0),
           "Eth": np.interp(tg, t, eth / e0)}
    if "divberr" in s:
        td, dm, dx = s["divberr"]
        tbl["divBerrMean"] = np.interp(tg, td, dm)
        tbl["divBerrMax"] = np.interp(tg, td, dx)
    return tbl


def _plot(inputs, labels, params, args):
    print(f"[ponomarenko] B0={params['B0']:g}  boxlen={BOXLEN:g}")
    runs = [series(inp) for inp in inputs]
    if all(s is None for s in runs):
        print("ponomarenko: no data to plot; exiting.", file=sys.stderr)
        sys.exit(1)
    ref = getattr(args, "ref_table", None)
    plot_energy_vs_time(runs, labels, save=args.save, ref_table=ref)
    plot_divberr_vs_time(runs, labels, save=args.save, ref_table=ref)
    if args.render:
        do_render(inputs, labels, args, save=args.save)


def main():
    standalone_cli(
        "ponomarenko", PARAM_SPEC, _plot,
        reference_table=lambda inp, params, args: reference_table(inp),
        description="Ponomarenko screw-flow kinematic-dynamo analysis "
                    "(Teyssier et al. 2006).",
        reg_keys=("Emag", "Ekin", "Eth", "divBerrMean", "divBerrMax"),
        add_time=False)


if __name__ == "__main__":
    main()
