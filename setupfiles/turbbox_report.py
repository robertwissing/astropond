#!/usr/bin/env python3
"""Render the completed published-forcing campaign from retained measurements."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import IC_analysis_turbbox as analysis
import turbbox_ensemble as ensemble
from IC_analysis_general import read_log_series
from IC_analysis_framework import styled, _color_scale_from_pool


# Published-forcing campaign: zeta=1 solenoidal, case 1 (MHD, uniform Bz),
# realisation 0 (the published sequence). mhd/field are kept denormalised
# because the figure helpers read them directly.
PARAMS = {"mach": 10, "forcing": 1.0, "case": 1, "mhd": 1,
          "field": "uniform", "crossings": 100, "seed": 0}
NOTICE = ("Published forcing; Gasoline fixed alpha=0.5 / O_64 kernel. "
          "Development resolutions; no spatial convergence claim.")


def stamp(fig, detail):
    fig.text(0.5, 0.005, detail + "\n" + NOTICE,
             ha="center", va="bottom", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.055, 1, 1))


def save(fig, path):
    path = Path(path)
    if not path.exists():
        fig.canvas.draw()
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, bbox_inches="tight")
        print(f"saved {path}", flush=True)


def load_histories(history_cache, requested_inputs=None):
    inputs, labels, series, metrics = [], [], {}, {}
    for native in (32, 64):
        rows = json.loads((history_cache/f"history_{native}.json").read_text())
        inp = str(Path(rows[0]["filename"]).parent.resolve())
        with np.load(history_cache/f"history_{native}.npz", allow_pickle=False) as data:
            table = {key: data[key] for key in data.files}
        if requested_inputs is not None and Path(inp).resolve() not in {
                Path(p).resolve() for p in requested_inputs}:
            raise ValueError(f"history cache belongs to a different run: {inp}")
        if (len(rows) != 400 or any(Path(row["filename"]).parent.resolve() != Path(inp).resolve()
                                   for row in rows)
                or any(len(values) != len(rows) for values in table.values())):
            raise ValueError("incomplete/mismatched particle history cache")
        for key, values in table.items():
            if key == "files":
                if list(values) != [r["filename"] for r in rows]:
                    raise ValueError("history filenames disagree with records")
                continue
            if not np.allclose(values, [r[key] for r in rows], rtol=0, atol=0, equal_nan=True):
                raise ValueError(f"history table and records disagree: {key}")
        inputs.append(inp)
        labels.append(rf"${native}^3$ particles")
        series[inp] = rows, table
        metrics[str(native)] = analysis.convergence_diagnostics(table, .05, 10, analysis.SEED_B_RMS)
        metrics[str(native)]["bookkeeping"] = analysis.energy_bookkeeping_diagnostics(inp)
        late = table["time"]/.05 >= 80-1e-8
        for key in ("bulk_speed", "alfven_mach_rms", "va_rms_over_cs", "vrms_over_va_rms"):
            values = table[key]
            metrics[str(native)][key] = {
                "late_mean": float(values[late].mean()),
                "late_std": float(values[late].std()), "max": float(values.max())}
    return inputs, labels, series, metrics


def quality_figure(inputs, labels, series):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    metrics = {}
    for i, (inp, label) in enumerate(zip(inputs, labels)):
        _rows, tab = series[inp]
        color = f"C{i}"
        axes[0].semilogy(tab["time"]/.05, tab["bulk_speed"], color=color,
                         label=label + ": float32 particles")
        log = read_log_series(inp, ("dTime", "Llinx", "Lliny", "Llinz",
                                    "divBerrAvg", "divBerrMax"))
        bulk = np.sqrt(sum(log[k]**2 for k in ("Llinx", "Lliny", "Llinz"))) / tab["mass"][0]
        axes[0].semilogy(log["dTime"]/.05, np.maximum(bulk, 1e-20),
                         color=color, ls="--", alpha=.75,
                         label=label + ": double-precision log")
        for key, style, desc in (("divBerrAvg", "-", "mean"), ("divBerrMax", ":", "maximum")):
            axes[1].semilogy(log["dTime"]/.05, log[key], color=color, ls=style,
                             label=label + ": " + desc)
        late = log["dTime"]/.05 >= 80-1e-8
        metrics[str(round(tab["npart"][-1]**(1/3)))] = {
            "late_divBerr_mean": float(log["divBerrAvg"][late].mean()),
            "late_divBerr_max": float(log["divBerrMax"][late].max()),
            "maximum_log_bulk_speed": float(bulk.max())}
    axes[0].set_ylabel(r"$|\langle v\rangle_m|/c_s$")
    axes[0].set_title("Bulk motion remains negligible; particle storage sets a precision floor")
    axes[1].set_ylabel(r"logged $h|\nabla\cdot B|/|B|$")
    axes[1].set_ylim(1e-4, 1e3)
    axes[1].set_title("Magnetic divergence diagnostics: cleaning enabled, residual errors remain")
    axes[1].set_xlabel(r"$t/t_c$")
    for ax in axes:
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8, ncol=2)
    stamp(fig, "400 particle snapshots and 10,001 log samples per resolution; t = 0–100 tc")
    return fig, metrics


def history_figures(output, history_cache, requested_inputs=None):
    inputs, labels, series, metrics = load_histories(history_cache, requested_inputs)
    figures = [
        ("energy_growth", analysis.energy_growth_figure(inputs, labels, PARAMS, series)),
        ("alfven_history", analysis.alfven_history_figure(inputs, labels, PARAMS, series)),
        ("mach_history", analysis.history_figure(inputs, labels, PARAMS,
                              SimpleNamespace(stationary_after=80), series)),
    ]
    for name, fig in figures:
        stamp(fig, "400 snapshots per resolution; all 100 crossing times; late assessment: 80–100 tc")
        save(fig, output/f"{name}.png")
    quality, qm = quality_figure(inputs, labels, series)
    figures.append(("numerical_quality", quality))
    save(quality, output/"numerical_quality.png")
    for native, values in qm.items():
        metrics[native]["numerical_quality"] = values
    target = output/"particle_metrics.json"
    if not target.exists():
        ensemble.write_json(target, metrics)
    return figures, metrics


def midplane_figure(output, step=100, column=False):
    """Compatibility helper using the corrected shared paper-coordinate render."""
    from turbbox_render import paper_midplane
    inputs = []
    for native in (32, 64):
        path = output/"cache"/f"n{native}_step{step:05d}_grid{2*native}.npz"
        with np.load(path, allow_pickle=False) as data:
            identity = json.loads(str(data["identity"]))
            snapshots = [p for p in identity["sources"] if Path(p).suffix[1:].isdigit()]
            if len(snapshots) != 1:
                raise ValueError("ambiguous snapshot in map cache")
            inputs.append(str(Path(snapshots[0]).parent))
    fig,_metadata = paper_midplane(inputs,time_crossings=step/100,
                                   project="column" if column else "slice")
    return fig


def load_ensembles(output, samples=11, window=(80,100)):
    results = []
    for native in (32,64):
        records = []
        for path in sorted((output/"cache").glob(f"n{native}_step*_grid{2*native}.npz")):
            with np.load(path, allow_pickle=False) as data:
                records.append({key: data[key] for key in data.files if key != "identity"})
        records.sort(key=lambda r: float(r["time_crossings"]))
        tau = np.array([float(r["time_crossings"]) for r in records])
        late = []
        for target in np.linspace(*window, samples):
            i = np.argmin(abs(tau-target))
            if not np.isclose(tau[i], target, atol=0.13):
                raise ValueError(f"missing {native}^3 ensemble sample at {target} tc")
            late.append(records[i])
        results.append({"native": native, "records": records, "late": late})
    return results


def band(ax, x, mean, std, color, label, style="-", log=True):
    draw = ax.loglog if log else ax.plot
    draw(x, mean, color=color, ls=style, label=label)
    ax.fill_between(x, np.maximum(mean-std, 1e-30) if log else np.maximum(mean-std, 0),
                    mean+std, color=color, alpha=.14)


def ensemble_figures(output, results):
    figures = []
    start, end = results[0].get("late_window", (80,100))
    samples = len(results[0]["late"])
    window_label = f"{start:g}–{end:g} tc"
    detail = (f"32^3 / 64^3 particles; 64^3 / 128^3 grids; {samples} snapshots, "
              f"{window_label}; bands: sample standard deviation")
    fig, axes = plt.subplots(2,2,figsize=(12,9))
    for i, result in enumerate(results):
        records = result["late"]
        k = records[0]["k"]
        label = rf"${result['native']}^3$"
        for ax, fields in ((axes[0,0], (("v_total", r"$v$", "-"), ("rho13v_total", r"$\rho^{1/3}v$", "--"))),
                            (axes[0,1], (("kinetic_total", r"$\rho^{1/2}v$", "-"), ("magnetic_total", r"$B$", "--"))),
                            (axes[1,0], (("density_total", r"$\rho$", "-"),))):
            for key, desc, style in fields:
                mean, std = ensemble.aggregate(records, key)
                band(ax,k,mean,std,f"C{i}",label+": "+desc,style)
        fractions = [r["v_compressive"]/np.maximum(r["v_total"],1e-300) for r in records]
        band(axes[1,1],k,np.mean(fractions,axis=0),np.std(fractions,axis=0,ddof=1),
             f"C{i}",label,log=False)
    for ax, title in zip(axes.ravel(), ("Velocity and density-weighted velocity", "Kinetic and magnetic energy spectra",
                                       "Density fluctuations", "Velocity compressive fraction")):
        ax.set_title(title)
        ax.set_xlabel(r"mode $kL/(2\pi)$")
        ax.set_ylabel("shell-integrated half-power" if ax is not axes[1,1] else r"$P_{comp}/P_{total}$")
        ax.grid(alpha=.2,which="both")
        ax.legend(fontsize=8)
    axes[1,1].set_xscale("log")
    axes[1,1].set_ylim(0,1)
    stamp(fig,detail+"; no inertial-range fits")
    figures.append(("spectra_late",fig))

    fig,axes = plt.subplots(1,2,figsize=(12,5))
    x = .5*(ensemble.RHO_EDGES[1:]+ensemble.RHO_EDGES[:-1])
    for i,result in enumerate(results):
        for key,desc,style in (("rho_pdf","volume","-"),("rho_mass_pdf","mass","--")):
            mean,std=ensemble.aggregate(result["late"],key)
            for ax in axes:
                band(ax,x,mean,std,f"C{i}",rf"${result['native']}^3$: "+desc,style,False)
    for ax in axes:
        ax.set_xlim(-8,6)
        ax.set_xlabel(r"$s=\ln(\rho/\rho_0)$")
        ax.set_ylabel("PDF per unit s")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Fixed-bin density PDFs")
    axes[1].set_title("Logarithmic PDF axis exposes tails")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-6,2)
    stamp(fig,detail+"; Delta s=0.1; fixed volume grid, not an adaptive PDF mesh")
    figures.append(("density_pdfs_late",fig))

    fig,axes=plt.subplots(2,2,figsize=(12,9))
    bx=.5*(ensemble.B2_EDGES[1:]+ensemble.B2_EDGES[:-1])
    for i,result in enumerate(results):
        records=result["records"]
        for j,target in enumerate((1,4,8,12,20)):
            record=min(records,key=lambda r:abs(float(r["time_crossings"])-target))
            if not np.isclose(record["time_crossings"],target):
                raise ValueError(f"missing evolution time {target}")
            color=plt.get_cmap("viridis")(j/5)
            axes[i,0].loglog(record["k"],record["magnetic_total"],color=color,lw=1.2,label=f"{target} tc")
            axes[i,1].plot(bx,record["b2_pdf"],color=color,lw=1.2,label=f"{target} tc")
        for ax,key,x,log in ((axes[i,0],"magnetic_total",records[0]["k"],True),
                             (axes[i,1],"b2_pdf",bx,False)):
            mean,std=ensemble.aggregate(result["late"],key)
            band(ax,x,mean,std,"C3",window_label+" mean",log=log)
            ax.grid(alpha=.2,which="both")
            ax.legend(fontsize=8,ncol=2)
        axes[i,0].set_title(rf"${result['native']}^3$: magnetic spectral growth")
        axes[i,0].set_xlabel(r"mode $kL/(2\pi)$")
        axes[i,0].set_ylabel(r"$P(B)$")
        axes[i,1].set_title(rf"${result['native']}^3$: absolute magnetic-strength PDF")
        axes[i,1].set_xlim(-15,5)
        axes[i,1].set_xlabel(r"$\log_{10}(B^2)$ [code units]")
        axes[i,1].set_ylabel(r"PDF per unit $\log_{10}(B^2)$")
    stamp(fig,detail+"; early curves are individual snapshots; spectra show shell-integrated half-power")
    figures.append(("magnetic_evolution",fig))

    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,result in zip(axes,results):
        records=result["records"]
        times=[float(r["time_crossings"]) for r in records]
        for mode in (5,10,15):
            j=int(np.argmin(abs(records[0]["k"]-mode)))
            ax.semilogy(times,[r["magnetic_total"][j] for r in records],"o-",ms=3,label=f"k={mode}")
        ax.set_title(rf"${result['native']}^3$: selected magnetic shells")
        ax.set_xlabel(r"$t/t_c$")
        ax.set_ylabel(r"shell $P(B)$")
        ax.grid(alpha=.2,which="both")
        ax.legend(fontsize=9)
    stamp(fig,f"{len(results[0]['records'])} gridded times per resolution; markers denote measured snapshots, lines connect samples")
    figures.append(("magnetic_modes",fig))
    for name,fig in figures:
        save(fig,output/f"{name}.png")
    return figures


def interpolation_figure(results, cache, source_archive=None):
    """Compare 1x/2x/4x reconstruction at the same final particle snapshot."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    measurements = {}
    for irun, result in enumerate(results):
        native = result["native"]
        row = result["rows"][-1]
        grids = [ensemble.cached_grid(row, factor*native, cache,
                                      source_archive=source_archive) for factor in (1,2,4)]
        truth = grids[-1]
        measurements[str(native)] = []
        for key, desc in (("grid_kinetic_over_particle", "kinetic"),
                          ("grid_magnetic_over_particle", "magnetic")):
            axes[irun,0].plot((1,2,4), [g[key] for g in grids], "o-", label=desc)
        axes[irun,0].axhline(1, color="0.5", ls=":")
        axes[irun,0].set(xlabel="grid / native linear resolution", ylabel="grid / particle energy",
                         title=rf"${native}^3$: energy recovery", ylim=(.75,1.02), xticks=(1,2,4))
        for factor, grid in zip((1,2,4), grids):
            entry = {"grid":factor*native, "mass_relative_error":float(grid["grid_mass_relative_error"]),
                     "kinetic_over_particle":float(grid["grid_kinetic_over_particle"]),
                     "magnetic_over_particle":float(grid["grid_magnetic_over_particle"])}
            for column, key in enumerate(("kinetic_total", "magnetic_total"), start=1):
                k = grid["k"]
                ref = truth[key][:len(k)]
                common = k <= native/2
                entry[key+"_low_k_L1_relative_to_4x"] = float(
                    np.sum(abs(grid[key][common]-ref[common]))/np.sum(ref[common]))
                if factor < 4:
                    axes[irun,column].semilogx(k[common],grid[key][common]/ref[common],
                                              label=f"{factor}x / 4x")
            entry["rho_pdf_L1_relative_to_4x"] = float(np.sum(
                abs(grid["rho_pdf"]-truth["rho_pdf"])*np.diff(ensemble.RHO_EDGES)))
            measurements[str(native)].append(entry)
        for column, title in ((1,"kinetic"),(2,"magnetic")):
            axes[irun,column].axhline(1,color="0.5",ls=":")
            axes[irun,column].set(xlabel=r"mode $kL/(2\pi)$",ylabel="shell power / 4x grid",
                                  title=rf"${native}^3$: {title} spectral stability",ylim=(.4,1.1))
    for ax in axes.ravel():
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    stamp(fig,"Same final snapshot at 100 tc; spectra compared through native Nyquist. "
          "A grid plateau does not remove SPH smoothing or establish particle convergence.")
    return fig, measurements


def ensemble_metrics(results):
    metrics = {}
    for result in results:
        summary = {"window_crossings":result["late_window"],
                   "samples":len(result["late"]),
                   "times_crossings":[float(r["time_crossings"]) for r in result["late"]]}
        for key in ("grid_kinetic_over_particle", "grid_magnetic_over_particle",
                    "grid_mass_relative_error", "volume_compressive_fraction",
                    "rho_pdf_outside", "rho_mass_pdf_outside", "b2_pdf_outside"):
            values = np.array([r[key] for r in result["late"]])
            summary[key] = {"mean":float(values.mean()),"std":float(values.std(ddof=1)),
                            "min":float(values.min()),"max":float(values.max())}
        summary["energy_accounting"] = {}
        for prefix in ("kinetic", "magnetic"):
            summary["energy_accounting"][prefix] = {
                part:float(np.mean([r[prefix+"_"+part] for r in result["late"]]))
                for part in ("full_grid_power","zero_mode_power","outside_sphere_power")}
            summary["energy_accounting"][prefix]["plotted_shell_power"] = float(np.mean(
                [np.sum(r[prefix+"_total"]) for r in result["late"]]))
        metrics[str(result["native"])] = summary
    return metrics


def text_page(title, paragraphs):
    import textwrap
    fig = plt.figure(figsize=(8.3,11.7))
    fig.text(.075,.94,title,fontsize=17,weight="bold",va="top")
    body = "\n\n".join(textwrap.fill(p,width=92) for p in paragraphs)
    fig.text(.075,.87,body,fontsize=10,va="top",linespacing=1.5)
    return fig


def campaign_report(path, metrics, figures, evidence=None):
    from matplotlib.backends.backend_pdf import PdfPages
    paragraphs = [
        "Published-forcing Gasoline MHD comparison: completed 32^3 and 64^3 "
        "particle runs through 100 crossing times, plus the requested 128^3 "
        "one-crossing morphology comparator when supplied.",
        "The two long runs pass the declared late temporal plateau tests. "
        "Their different saturated magnetic fractions do not establish spatial "
        "convergence or agreement with Tricco, Price & Federrath (2016).",
    ]
    for n in ("32","64"):
        p = metrics["particles"][n]
        paragraphs.append(f"{n}^3: late Mach {p['mach']['mean']:.4f}; "
                          f"late magnetic/kinetic ratio {p['magnetic_to_kinetic_ratio']['mean']:.4f}. "
                          f"Particle bulk speed stays below {p['bulk_speed']['max']:.3g}.")
    if "128" in metrics.get("early_particles",{}):
        p = metrics["early_particles"]["128"]
        paragraphs.append(f"128^3 at one crossing time: Mach {p['mach']:.5f}, "
                          f"magnetic/kinetic ratio {p['emag_over_ekin']:.3g}. "
                          "This early snapshot does not measure a dynamo plateau.")
    if metrics.get("magnetic_mean"):
        paragraphs.append("The reconstructed box-mean magnetic field also grows strongly. "
                          "The total magnetic plateau therefore includes a substantial mean-field "
                          "component and is not, by itself, a validated small-scale dynamo result.")
    paragraphs += [
        "Particle histories contain 400 snapshots per long run. Gridded statistics "
        "use the explicitly recorded temporal window and samples. Bands show "
        "sample standard deviations, not independent-sample confidence intervals.",
        "References: Price & Federrath (2010), arxiv.org/abs/1004.1446; "
        "Tricco, Price & Federrath (2016), arxiv.org/abs/1605.08662; "
        "Price, Federrath & Tricco forcing data, doi.org/10.5281/zenodo.13162225 "
        "(CC BY 4.0). PHANTOM source: github.com/danieljprice/phantom.",
    ]
    morphology = metrics.get("reference_comparison")
    morphology_result = (
        "The viewing error explains the main apparent Figure-1 morphology "
        "discrepancy. Against the digitized PHANTOM 128^3 panels, log-density "
        f"correlation improves from {morphology['rho']['earlier_view']['log_field_Pearson_r']:.3f} "
        f"to {morphology['rho']['corrected_view']['log_field_Pearson_r']:.3f}; log-|B| correlation "
        f"improves from {morphology['Bmag']['earlier_view']['log_field_Pearson_r']:.3f} "
        f"to {morphology['Bmag']['corrected_view']['log_field_Pearson_r']:.3f}. "
        "Both controls use the same Gasoline snapshot and density interpolant. "
        "No fitted translation or rotation is used. These are correlations of "
        "colour-limited digitized images, not error norms against raw paper data."
        if morphology else
        "The corrected coordinate convention has not been quantified against "
        "the paper's panels in this execution; use --paper-reference to add that control.")
    pages = [text_page("Turbulence box: measured comparison",paragraphs),
             text_page("Realised numerical and analysis methods",[
        "Periodic unit cube; rho0=cs=1; initially static gas; Bz=sqrt(2)*10^-5; "
        "tc=0.05. Published forcing MD5: "+ensemble.FORCING_MD5+". "
        "Pure solenoidal forcing, energy 8, update interval 0.005, correlation time 0.05.",
        "Every run is audited against its log, completion record, forcing checksum "
        "and immutable build manifest. Fixed alpha=0.5, beta=4, RECONSTRUCTION, "
        "optimized O_64 simulation kernel, 64 neighbours, Tricco resistivity, "
        "hyperbolic cleaning and gas bulk correction are the accepted Gasoline "
        "settings. These differ from the paper's PHANTOM discretisation and viscosity switch.",
        "Mach and kinetic energy use the mass-weighted gas bulk frame. Local "
        "Alfvenic Mach is rms(|v|/vA), separately from vrms/vA,rms. Magnetic energy "
        "on particles is sum((m/rho)*B^2/2). The isothermal, externally forced box "
        "is an open energy system; no total-energy conservation claim is made.",
        "Spectra and volume PDFs use the existing periodic Petkova cubic-spline "
        "kernel-volume integration, with smoothing lengths inferred from m/rho. "
        "This reconstruction kernel differs from the O_64 simulation kernel. "
        "Kinetic spectra use sqrt(rho)*v, alongside v and rho^(1/3)*v. The report "
        "accounts separately for the zero mode and unplotted Fourier corner modes.",
        "Density PDFs use fixed Delta ln(rho/rho0)=0.1 bins. Magnetic PDFs retain "
        "absolute log10(B^2), including growth in amplitude. Probability outside "
        "the fixed bins is recorded. This uniform-grid PDF is not the paper's "
        "adaptive mesh PDF; high-density tails may remain unresolved.",
        "Interpolation tests compare 1x, 2x and 4x native grids at 100 tc. The "
        "2x-to-4x change in integrated energy is small, while missing particle "
        "kinetic and magnetic power remains measurable. Do not transfer the "
        "paper's 0.1-percent energy recovery claim to this reconstruction.",
    ]), text_page("Morphology and limits of the comparison",[
        "The paper-style slice uses literal z=0.5, equivalent to stored z=-0.5, "
        "and periodic x/y coordinates in [0,1). No half-box translation is applied "
        "to the forcing origin. This supersedes the earlier delivered centred "
        "z=0 slice relabelled as paper z=0.5.",
        "The shared SPH renderer uses normalized volume interpolation of stored "
        "particle density and |B|. Density is not replaced by a mass-kernel sum, "
        "so corrected particle density definitions are retained. "
        "Colours and fixed limits follow Figure 1: log10 rho in [-2,2] and "
        "log10 |B| in [-6,-3]. Display pixels do not add particle resolution.",
        "The 128^3 morphology comparator is only one crossing time. It cannot "
        "test dynamo saturation. The paper's smallest native resolution is 128^3; "
        "the long 32^3 and 64^3 runs here are development resolutions.",
        morphology_result,
        "Remaining differences in shock width, small structures and magnetic "
        "amplitude may reflect the numerical equations, "
        "kernel, viscosity/reconstruction, magnetic dissipation/cleaning, and "
        "time integration. No driver change is justified by appearance alone.",
        "The late logged mean h|div B|/|B| is about 0.09. Residual divergence "
        "and the large resolution dependence of magnetic saturation remain "
        "material limitations of the dynamo comparison. No inertial-range "
        "slope is inferred from these resolutions.",
    ])]
    if metrics.get("magnetic_mean"):
        mean_fields = metrics["magnetic_mean"]["resolutions"]
        measured = []
        for n in ("32", "64"):
            values = mean_fields[n]
            final = values["samples"][-1]
            measured.append(
                f"{n}^3 at 100 tc: grid |mean(B)|={final['grid_mean_B_magnitude']:.5f}, "
                f"with {100*final['grid_mean_energy_fraction']:.2f}% of grid magnetic energy "
                f"in the zero mode. The independent particle estimate is "
                f"{final['particle']['mean_B_magnitude']:.5f}, with sum(m/rho)="
                f"{final['particle']['volume_sum']:.5f}. The grid mean changes by "
                f"{100*values['final_2x_to_4x_mean_B_relative_change']:.4f}% from 2x to 4x.")
        pages.append(text_page("Box-mean magnetic field: an unresolved limitation",[
            "The nominal initial mean field has magnitude sqrt(2)*1e-5. The Fourier "
            "zero mode measures the spatial mean of the reconstructed vector field; "
            "it is not the mean of |B|. It grows by many orders of magnitude in both long runs.",
            *measured,
            "Particle estimates use sum((m/rho)*B)/sum(m/rho). Their volume weights "
            "do not sum to the unit box volume. They are approximate cross-checks "
            "and must not be presented as exact "
            "net-flux integrals. The grid and particle estimates agree on the strong growth.",
            "The figure separates mean and fluctuating magnetic energy on the same grid. "
            "The plotted fluctuation energy includes all nonzero Fourier modes. This is "
            "a diagnostic decomposition; no mean field was removed from the simulation.",
            "The responsible numerical operator has not been isolated. Trace the "
            "induction, resistivity and cleaning contributions to mean-field evolution "
            "in short controlled tests before treating the plateau as a paper-equivalent "
            "small-scale dynamo or changing the magnetic update.",
        ]))
    if evidence:
        pages.append(text_page("Independent forcing checks",[
            "The attached evidence JSON records the direct PHANTOM/Gasoline "
            "raw-force probe and the existing all-record and disabled-forcing checks.",
            "The extracted public PHANTOM st_calcAccel routine reads the original "
            "Fortran records in double precision. At the measured records and "
            "positions its acceleration matches the Gasoline evaluator exactly "
            "(maximum absolute error: "+str(evidence["phantom_evaluator"]["raw_acceleration_max_absolute_error"])+").",
            "PHANTOM forceit advances only when t > tprev + dtfreq. Gasoline "
            "uses an integer clock with roundoff snapping at exact boundaries. "
            "On the recorded synthetic 101-time base-step schedule the record "
            "indices differ at "+str(len(evidence["phantom_evaluator"]["clock_record_differences"]))+
            " boundary queries. This is an offline timing comparison, not the "
            "authors' unavailable 2016 timestep history or evidence that it "
            "alone caused the remaining morphology discrepancy.",
            "Full-record byte comparison and endian/invalid-file guards pass. "
            "Moving-gas zero-energy/zero-amplitude disabled forcing is "
            "byte-identical to forcing-disabled builds in hydro and MHD. "
            "Exact forcing replay does not imply full integrator checkpoint identity.",
        ]))
    with Path(path).open("xb") as stream, PdfPages(stream) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        for _name,fig in figures:
            pdf.savefig(fig,bbox_inches="tight")
        pdf.infodict().update(Title="Published-forcing Gasoline turbulence comparison",
                             Author="IC_analysis_turbbox.py",Subject="Measured validation and resolution limitations")


def run_campaign(inputs, args):
    """Main-analysis entry point; reuse only source-verified ensemble products."""
    import contextlib
    import io
    from turbbox_render import paper_midplane, paper_reference_comparison
    output = Path(args.campaign_output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"campaign_metrics.json").exists():
        raise FileExistsError("choose a new campaign output directory; existing reports are retained")
    by_native = {json.loads((Path(inp)/"launch.json").read_text())["resolution"]:
                 str(Path(inp).resolve()) for inp in inputs}
    if len(inputs) != 2 or set(by_native) != {32,64}:
        raise ValueError("campaign mode requires the completed 32^3 and 64^3 reference runs")
    inputs = [by_native[n] for n in (32,64)]
    contracts = {str(n):ensemble.audit_contract(inp) for n,inp in zip((32,64),inputs)}
    history_cache = Path(args.history_cache) if args.history_cache else output/"histories"
    if not args.history_cache:
        history_cache.mkdir(exist_ok=True)
        for native,inp in zip((32,64),inputs):
            with contextlib.redirect_stdout(io.StringIO()):
                rows, table = analysis.snapshot_series(inp)
            ensemble.write_json(history_cache/f"history_{native}.json",rows)
            with (history_cache/f"history_{native}.npz").open("xb") as stream:
                np.savez_compressed(stream,**table)
    _inputs,_labels,series,_metrics = load_histories(history_cache,inputs)
    figures, particle_metrics = history_figures(output,history_cache,inputs)
    cache_root = Path(args.ensemble_cache) if args.ensemble_cache else output
    cache_root.mkdir(parents=True,exist_ok=True)
    results = []
    for inp in inputs:
        rows,_table = series[inp]
        result = ensemble.collect(inp,rows,cache_root,args.ensemble_samples,
                                  *args.ensemble_window,source_archive=args.cache_source)
        result["records"] = result["evolution"]
        results.append(result)
    figures += ensemble_figures(output,results)
    fig,interpolation = interpolation_figure(results,cache_root/"cache",args.cache_source)
    save(fig,output/"interpolation_convergence.png")
    figures.append(("interpolation convergence",fig))
    from turbbox_magnetic import magnetic_mean_figure
    fig,magnetic_mean = magnetic_mean_figure(results,cache_root/"cache",args.cache_source)
    stamp(fig,f"{len(magnetic_mean['resolutions']['32']['samples'])} measured times per resolution; "
          "2x Petkova grids. Particle means use approximate m/rho volume weights.")
    save(fig,output/"magnetic_mean_field.png")
    figures.append(("Mean magnetic field and fluctuation energy",fig))
    render_inputs = list(inputs)
    if args.comparison_run:
        contracts["128"] = ensemble.audit_contract(args.comparison_run,crossings=1)
        render_inputs.append(args.comparison_run)
    fig,maps = paper_midplane(render_inputs,time_crossings=1,resolution=args.render_res)
    save(fig,output/"figure1_rho_B_tc1.png")
    figures.append(("Figure 1 comparison",fig))
    reference_comparison = None
    if args.paper_reference:
        if not args.comparison_run:
            raise ValueError("--paper-reference requires --comparison-run")
        fig,reference_comparison = paper_reference_comparison(args.comparison_run,args.paper_reference)
        save(fig,output/"paper128_origin_comparison.png")
        figures.append(("Controlled Figure 1 viewing comparison",fig))
    early_particles = {}
    for record in maps:
        with contextlib.redirect_stdout(io.StringIO()):
            row = analysis.snapshot_diagnostics(record["snapshot"])
        if any(not np.isfinite(value) for value in row.values()):
            raise ValueError("non-finite early comparator diagnostics")
        early_particles[str(round(row["npart"]**(1/3)))] = row
    evidence = json.loads(Path(args.validation_json).read_text()) if args.validation_json else None
    metrics = {"contracts":contracts,"particles":particle_metrics,
               "ensembles":ensemble_metrics(results),"interpolation":interpolation,
               "magnetic_mean":magnetic_mean,
               "maps":maps,"early_particles":early_particles,"validation":evidence,
               "reference_comparison":reference_comparison,
               "source_sha256":{p.name:ensemble.digest(p) for p in
                   [Path(__file__),Path(analysis.__file__),Path(ensemble.__file__),
                    Path(__file__).with_name("turbbox_render.py"),
                    Path(__file__).with_name("turbbox_magnetic.py"),
                    Path(__file__).with_name("IC_analysis_render.py")]},
               "history_cache":str(history_cache.resolve()),
               "ensemble_cache":str(cache_root.resolve()),
               "cache_source_archive":args.cache_source}
    ensemble.write_json(output/"campaign_metrics.json",metrics)
    if args.metrics_json:
        ensemble.write_json(args.metrics_json,metrics)
    report = Path(args.report) if args.report else output/"comparison_report.pdf"
    report.parent.mkdir(parents=True,exist_ok=True)
    campaign_report(report,metrics,figures,evidence)
    for _name,fig in figures:
        plt.close(fig)
    print(f"Completed published-forcing comparison: {report}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-cache", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    history_figures(args.output, args.history_cache)
    plt.close("all")


if __name__ == "__main__":
    main()
