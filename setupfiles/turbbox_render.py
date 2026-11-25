"""TPF16-style field renders using the shared analysis rendering machinery."""

import sys

import numpy as np

from IC_analysis_framework import (
    _cached_readtipsy, _grid_image, _quantity_values, _color_scale_from_pool,
    aux_magnitude, read_box_periods, select_at_time, styled, parse_resolution,
)
from IC_analysis_general import setup_rcparams

DEFAULT_RENDER_TIMES = (1., 2., 4., 8.)


class RenderTimeUnavailable(ValueError):
    """A requested render time has no sufficiently close snapshot."""


def paper_colormaps():
    """Heat/blue ramps matched to the colour bars in TPF16 Figure 1.

    Anchors sampled from the public author PDF's colour bars, not its data
    panels. Density limits are log10 rho=[-2,2]; field limits log10 |B|=[-6,-3].
    """
    from matplotlib.colors import LinearSegmentedColormap
    density = ["#000000", "#180000", "#2a0000", "#3c0000", "#4f0000",
               "#610000", "#730000", "#850000", "#980000", "#a90000",
               "#bc1200", "#ce2a00", "#df4100", "#f25a00", "#ff7200",
               "#ff890b", "#ffa13e", "#ffba70", "#ffd19f", "#ffe9d1", "#ffffff"]
    magnetic = ["#000000", "#000017", "#000027", "#000039", "#00004a",
                "#00005b", "#00006c", "#00007d", "#000f8f", "#00229f",
                "#0037b0", "#004bc2", "#005fd2", "#0074e4", "#0088f5",
                "#049bff", "#37b0ff", "#6bc4ff", "#9cd8ff", "#d0ecff", "#ffffff"]
    return (LinearSegmentedColormap.from_list("tpf16_density", density),
            LinearSegmentedColormap.from_list("tpf16_magnetic", magnetic))


def _snapshot_fields(inp, time_crossings, resolution, zslice, nsmooth,
                     project, tcross=.05):
    """Sample rho and particle |B| through the shared rendering backend."""
    chosen = select_at_time(inp, time_crossings*tcross, test="turbbox",
                            warn_outside=False)
    if chosen is None:
        raise RenderTimeUnavailable(f"{inp}: no snapshot for {time_crossings:g} crossing times")
    filename, actual_time = chosen
    actual_crossings = float(np.asarray(actual_time).item())/tcross
    if abs(actual_crossings-time_crossings) > .126:
        raise RenderTimeUnavailable(f"{inp}: requested {time_crossings:g} crossing times, "
                                    f"nearest snapshot is at {actual_crossings:g}")
    gas, _dark, _stars, header, _time = _cached_readtipsy(filename)
    ngas = int(header[2])
    periods = read_box_periods(inp)
    if periods is None or any(not np.isclose(v, 1) for v in periods.values()):
        raise ValueError("paper render requires logged unit periodic box")
    fields = []
    for key, quantity in (("rho", 7), ("Bmag", aux_magnitude("BField"))):
        spec = styled(key, 1, 2, quantity, label=key, project=project)
        q = _quantity_values(spec.quantity, filename, gas, ngas)
        field = _grid_image(filename, gas, ngas, spec, q, 0, 1, 0, 1,
                            resolution, project, zslice, periods, nsmooth=nsmooth)
        if field is None or np.any(~np.isfinite(field)):
            raise ValueError("shared SPH renderer failed")
        fields.append(field)
    metadata = {"run": str(inp), "snapshot": filename,
                "requested_time_crossings": float(time_crossings),
                "time_crossings": actual_crossings, "crossing_time": tcross,
                "npart": ngas, "pixel_resolution": resolution,
                "project": project, "zslice": zslice if project == "slice" else None,
                "xy_extent": [0, 1, 0, 1],
                "coordinate_transform": "periodic wrap only; no half-box translation",
                "density_interpolation": ("normalized SPH interpolation of stored particle density"
                                          if project == "slice" else
                                          f"SPH {project} projection of stored particle density"),
                "magnetic_quantity": "particle |B|; magnitude taken before interpolation/integration",
                "smoothing_length": f"inferred from particle mass/density with nsmooth={nsmooth}",
                "render_backend": "grid",
                "render_method": f"shared SPH {project}"}
    if project == "column":
        metadata["column_axis"] = "z"
        metadata["integral_over_xy"] = {
            key: float(np.mean(field)) for key, field in zip(("rho", "Bmag"), fields)}
    return fields, metadata


def paper_midplane(inputs, labels=None, time_crossings=1.0, resolution=512,
                   zslice=0.5, nsmooth=64, project="slice", tcross=.05):
    """Shared SPH slice at literal phase coordinate z=0.5, periodically wrapped.

    No translation of the Fourier origin: x/y are displayed in [0,1), and
    z=0.5 is equivalent to z=-0.5 in the stored [-0.5,0.5) cube. The box-centre
    z=0 render is a different slice of the same forcing realisation.
    """
    if not inputs or resolution < 4 or project not in ("slice", "column", "average"):
        raise ValueError("need inputs, resolution>=4 and a supported render projection")
    labels = labels or [rf"${parse_resolution(inp)}^3$" for inp in inputs]
    if len(labels) != len(inputs) or not np.isfinite(zslice):
        raise ValueError("need one label per input and a finite slice coordinate")
    panels = [[], []]
    metadata = []
    for inp in inputs:
        fields, record = _snapshot_fields(inp, time_crossings, resolution,
                                           zslice, nsmooth, project, tcross)
        for i, field in enumerate(fields):
            panels[i].append(field)
        metadata.append(record)
    return paper_field_figure(panels,labels,time_crossings,resolution,zslice,project), metadata


def render_sequence(inputs, labels=None, times_crossings=DEFAULT_RENDER_TIMES,
                    resolution=512, zslice=.5, nsmooth=64, project="slice", tcross=.05):
    """Simulation-only rho/|B| sequence; warn and skip unavailable snapshots.

    Times are columns and runs are rows. Omit times unavailable in every run;
    missing snapshots within a retained time column are labelled explicitly.
    Return (None, metadata) when no requested snapshot is available.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext

    times = np.asarray(times_crossings, dtype=float)
    if (not inputs or resolution < 4 or project not in ("slice", "column")
            or times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times))
            or np.any(times < 0) or not np.isfinite(tcross) or tcross <= 0
            or not np.isfinite(zslice)):
        raise ValueError("need inputs, finite nonnegative times, positive crossing time, "
                         "resolution>=4 and slice/column projection")
    labels = labels or [rf"${parse_resolution(inp)}^3$" for inp in inputs]
    if len(labels) != len(inputs):
        raise ValueError("need one label per input")
    panels = [[], []]
    records = []
    skipped = []
    for inp in inputs:
        run_fields = [[], []]
        for tau in times:
            print(f"turbbox: {project} rho/|B| at t/tc={tau:g} for {inp}", flush=True)
            try:
                fields, record = _snapshot_fields(inp, float(tau), resolution,
                                                   zslice, nsmooth, project, tcross)
            except RenderTimeUnavailable as error:
                print(f"turbbox: warning: {error}; skipping this time.",
                      file=sys.stderr, flush=True)
                skipped.append({"run": str(inp), "requested_time_crossings": float(tau),
                                "reason": str(error)})
                for row in run_fields:
                    row.append(None)
                continue
            for i, field in enumerate(fields):
                run_fields[i].append(field)
            records.append(record)
        for i in range(2):
            panels[i].append(run_fields[i])

    available = [j for j in range(len(times))
                 if any(row[j] is not None for row in panels[0])]
    metadata = {"project": project, "render_backend": "grid",
                "requested_times_crossings": times.tolist(),
                "rendered_times_crossings": times[available].tolist(),
                "crossing_time": tcross, "zslice": zslice if project == "slice" else None,
                "panels": records, "skipped_panels": skipped, "colour_limits": {}}
    if not available:
        return None, metadata
    times = times[available]
    panels = [[[row[j] for j in available] for row in block] for block in panels]
    nruns, ntimes = len(inputs), len(times)
    density_cmap, b_cmap = paper_colormaps()
    quantities = ((r"$\rho$", r"$|B|$") if project == "slice" else
                  (r"$\int \rho\,\mathrm{d}z$", r"$\int |B|\,\mathrm{d}z$"))
    setup_rcparams()
    fig = plt.figure(figsize=(3.7*ntimes+1.2, 7.4*nruns+2.0))
    heights = [1.]*nruns + [.055, .17] + [1.]*nruns + [.055]
    layout = fig.add_gridspec(len(heights), ntimes, height_ratios=heights,
                             hspace=.08, wspace=.025)
    colour_limits = metadata["colour_limits"]
    for i, (key, cmap, quantity) in enumerate(zip(("rho", "Bmag"),
                                                 (density_cmap, b_cmap), quantities)):
        pool = [a[a > 0] for row in panels[i] for a in row
                if a is not None and np.any(a > 0)]
        spec = styled(key, 1, 2, 7, label=quantity, cmap=cmap, project=project)
        norm, lo, hi, use_log, colourmap = _color_scale_from_pool(pool, spec, {}, project)
        colour_limits[key] = {"min": float(norm.vmin if use_log else lo),
                              "max": float(norm.vmax if use_log else hi),
                              "logarithmic": bool(use_log),
                              "selection": "pooled 1st/99th percentiles across all times and runs"}
        offset = i*(nruns+2)
        for r in range(nruns):
            for j, tau in enumerate(times):
                ax = fig.add_subplot(layout[offset+r, j])
                field = panels[i][r][j]
                if field is None:
                    ax.set_facecolor("0.94")
                    ax.set_aspect("equal")
                    ax.text(.5, .5, "Snapshot\nunavailable", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color="0.4")
                else:
                    im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1),
                                   norm=norm, vmin=lo, vmax=hi, cmap=colourmap,
                                   interpolation="nearest", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(rf"$t/t_c={tau:g}$", fontsize=13, pad=5)
                if j == 0:
                    ax.set_ylabel(labels[r], fontsize=12)
        cax = fig.add_subplot(layout[offset+nruns, :])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          format=LogFormatterMathtext() if use_log else None)
        cb.set_label(quantity + (" (logarithmic colour scale)" if use_log else ""),
                     fontsize=12, labelpad=3)
    view = (rf"Slices at $z={zslice:g}$" if project == "slice" else
            "Integrals along z" + ("\n" if ntimes < 3 else " ") + "through the full periodic box")
    fig.suptitle(view, fontsize=16, y=.995)
    fig.text(.5, .012, "Times in crossing units. Periodic x/y in [0,1).\n"
             "Shared colour limits across times and runs.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, top=.955, bottom=.07)
    return fig, metadata


def paper_field_figure(panels, labels, time_crossings=1., resolution=512,
                       zslice=.5, project="slice", brands=None, footer=None):
    """Common Figure-1 layout for measured fields and digitized reference panels."""
    import matplotlib.pyplot as plt
    density_cmap,b_cmap = paper_colormaps()
    specs = (
        styled("rho",1,2,7,label=r"$\log_{10}\rho$",cmap=density_cmap,
               clim=lambda _p,_proj:(1e-2,1e2)),
        styled("Bmag",1,2,7,label=r"$\log_{10}|B|$",cmap=b_cmap,
               clim=lambda _p,_proj:(1e-6,1e-3)),
    )
    brands = brands or ["Gasoline"]*len(labels)
    setup_rcparams()
    fig = plt.figure(figsize=(5*len(labels),10.5))
    layout = fig.add_gridspec(5,len(labels),height_ratios=(1,.055,.14,1,.055),
                             hspace=.12,wspace=.012)
    for i,spec in enumerate(specs):
        positive=[a[a>0] for a in panels[i]]
        norm,_lo,_hi,_log,cmap=_color_scale_from_pool(positive,spec,{},project)
        for j,(field,label) in enumerate(zip(panels[i],labels)):
            ax=fig.add_subplot(layout[3*i,j])
            im=ax.imshow(field,origin="lower",extent=(0,1,0,1),norm=norm,cmap=cmap,
                         interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(label,fontsize=13,pad=5)
            ax.text(.025,.025,brands[j],color="white",fontsize=10,transform=ax.transAxes)
        cax=fig.add_subplot(layout[3*i+1,:])
        cb=fig.colorbar(im,cax=cax,orientation="horizontal")
        exponents=range(-2,3) if i==0 else range(-6,-2)
        cb.set_ticks([10.**p for p in exponents])
        cb.set_ticklabels([str(p) for p in exponents])
        cb.set_label(spec.label,fontsize=12,labelpad=2)
    location=(rf"$z={zslice:g}$" if project=="slice" else "z column")
    fig.suptitle(rf"$t/t_c={time_crossings:g}$, "+location+"; TPF16 Figure 1 colour scales",fontsize=15,y=.99)
    fig.text(.5,.008,footer or ("Native particle resolution is labelled above each panel. "+
             ("Shared SPH midplane render" if project=="slice" else "Shared SPH column render")+
             f"; {resolution}² display pixels. Periodic [0,1) coordinates preserve the forcing origin."),
             ha="center",fontsize=8)
    fig.subplots_adjust(top=.94,bottom=.11)
    return fig


def paper_reference_comparison(inp, reference_directory):
    """Compare the literal view to digitized TPF16 PHANTOM 128^3 panels.

    Both views use the same normalized interpolation of stored particle density;
    only the plane and coordinate convention change. Correlations compare
    colour-limited log fields, not the authors' unavailable raw snapshots.
    """
    from pathlib import Path
    import json
    from PIL import Image
    from scipy.spatial import cKDTree
    from turbbox_ensemble import digest
    reference_directory = Path(reference_directory)
    provenance = json.loads((reference_directory/"provenance.json").read_text())
    filename,actual_time = select_at_time(inp,.05,test="turbbox")
    gas,_dark,_stars,header,_time = _cached_readtipsy(filename)
    ngas = int(header[2])
    if ngas != 128**3 or not np.isclose(actual_time,.05):
        raise ValueError("direct paper comparison requires the 128^3 one-crossing snapshot")
    periods = read_box_periods(inp)
    panels,metrics = [],{}
    for key,quantity,cmap,limits,reference_name in (
        ("rho",7,paper_colormaps()[0],(-2,2),"phantom128_density.png"),
        ("Bmag",aux_magnitude("BField"),paper_colormaps()[1],(-6,-3),"phantom128_bmag.png"),
    ):
        reference_path = reference_directory/reference_name
        if digest(reference_path) != provenance["panels"][reference_name]:
            raise ValueError("reference panel checksum differs from its provenance")
        rgb = np.asarray(Image.open(reference_path).convert("RGB"),dtype=float)/255
        if rgb.shape != (600,600,3):
            raise ValueError("expected the 600^2 unlabelled Figure-1 data panel")
        levels = np.linspace(0,1,4096)
        distances,indices = cKDTree(cmap(levels)[:,:3]).query(rgb.reshape(-1,3))
        # Source raster origin is top-left; shared field arrays are origin-lower.
        log_reference = np.flipud((limits[0]+levels[indices]*(limits[1]-limits[0])).reshape(600,600))
        spec = styled(key,1,2,quantity,label=key)
        q = _quantity_values(quantity,filename,gas,ngas)
        current = _grid_image(filename,gas,ngas,spec,q,0,1,0,1,600,"slice",.5,periods,nsmooth=64)
        earlier = _grid_image(filename,gas,ngas,spec,q,-.5,.5,-.5,.5,600,"slice",0,periods,nsmooth=64)
        panels.append([10**log_reference,earlier,current])
        measured = {"reference_sha256":digest(reference_path),
                    "palette_rgb_rms_error":float(np.sqrt(np.mean(distances**2)))}
        for name,field in (("earlier_view",earlier),("corrected_view",current)):
            clipped = np.clip(np.log10(np.maximum(field,10.**limits[0])),*limits)
            measured[name] = {
                "log_field_Pearson_r":float(np.corrcoef(log_reference.ravel(),clipped.ravel())[0,1]),
                "log_field_RMSE_dex":float(np.sqrt(np.mean((log_reference-clipped)**2))),
            }
        metrics[key] = measured
    metrics.update({"snapshot":filename,"reference":"TPF16 Figure 1, PHANTOM 128^3",
                    "reference_provenance":provenance,
                    "method":"nearest-colour digitization; 600^2 pixels; fixed paper colour limits; no fitted shifts/rotations",
                    "interpretation":"render morphology only, not raw-data error norms or dynamo convergence"})
    fig = paper_field_figure(panels,[r"PHANTOM $128^3$: paper",r"Gasoline $128^3$: earlier view",
                                    r"Gasoline $128^3$: corrected view"],resolution=600,
              brands=["TPF16 Fig. 1","z=0, translated x/y","z=0.5, periodic x/y"],
              footer="Reference panels digitized from Tricco, Price & Federrath (2016), Fig. 1 (arxiv.org/abs/1605.08662).\n"
                     "Same 128³ Gasoline snapshot and density interpolant in both controls; only the viewing convention changes.")
    return fig,metrics
