"""Mass-weighted density PDF: particle truth vs Voronoi cells, de-sliver fix
ON vs OFF, for the collapse and the galaxy merger."""
import numpy as np, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from sph_interp import from_tipsy, VoronoiGrid, interpolate

COLLAPSE = ('../test_cases/mhdcollapse/'
            'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE/'
            'mhdcollapse32_N64latticemu10rd360Erat0.045facQ1GASOLINE.00800')
MERGER = ('/mn/stornext/d17/extragalactic/personal/robertwi/Projects/ICGEN/sims/'
          'OPTKERNELSIMS/testing_ground/isolated_galaxy_merger/PGALMHDJ8MERGb0/'
          'PGALMHDJ8changaMERGb0.00000')
NGEN = 40000
DEPMAX = 120000          # cap deposited particles (merger is 1.1M; big-h halo
                         # particles make the petkova Voronoi deposit expensive)


def mw_pdf(rho, mass, edges):
    m = rho > 0
    lr = np.log10(rho[m]); w = mass[m]
    h, _ = np.histogram(lr, bins=edges, weights=w)
    h = h / h.sum()
    return 0.5 * (edges[:-1] + edges[1:]), h


def cell_pdf(p, gens, bounds, periodic, merge_tol, edges):
    t0 = time.time()
    vg = VoronoiGrid.from_points(gens, bounds=bounds, periodic=periodic,
                                 merge_tol=merge_tol)
    res = interpolate(p, vg, method="petkova", values=["rho"])
    rho = res.data["rho"]; vol = vg.volumes()
    err = res.data["_mass"].sum() / p.mass.sum() - 1.0
    c, h = mw_pdf(rho, rho * vol, edges)
    tol_s = "auto(median-NN)" if merge_tol is None else ("%.3g" % merge_tol)
    print("    merge_tol=%s -> ncell=%d (merged %d)  petkova err %+.3f%%  "
          "[%.0fs]" % (tol_s, vg.ncell, len(gens) - vg.ncell, 100*err,
          time.time()-t0), flush=True)
    return c, h, vg.ncell


def run(path, periodic, label, off_tol):
    print(label, flush=True)
    p = from_tipsy(path, fields=("rho",), periodic=periodic)
    print("  N=%d  rho [%.2g, %.2g]" % (p.n, p.rho.min(), p.rho.max()), flush=True)
    # deposit-particle subsample (keep PDF apples-to-apples vs the same set)
    rng = np.random.default_rng(0)
    if p.n > DEPMAX:
        di = rng.choice(p.n, DEPMAX, replace=False)
        p = replace(p, pos=np.ascontiguousarray(p.pos[di]), mass=p.mass[di],
                    rho=p.rho[di], h=p.h[di],
                    values={k: v[di] for k, v in p.values.items()})
    # generators
    gi = np.random.default_rng(1).choice(p.n, min(NGEN, p.n), replace=False)
    gens = np.ascontiguousarray(p.pos[gi])
    if periodic:
        half = 0.5 * np.asarray(p.box)
        bounds = [[-half[a], half[a]] for a in range(3)]
    else:
        plo = gens.min(0); phi = gens.max(0); pad = 0.05 * (phi - plo)
        bounds = [[plo[a]-pad[a], phi[a]+pad[a]] for a in range(3)]
    lo, hi = np.log10(p.rho.min()), np.log10(p.rho.max())
    edges = np.linspace(lo - 0.2, hi + 0.2, 60)
    out = {}
    out["truth"] = mw_pdf(p.rho, p.mass, edges)
    out["off"] = cell_pdf(p, gens, bounds, periodic, off_tol, edges)
    out["on"] = cell_pdf(p, gens, bounds, periodic, None, edges)   # median-NN
    return out, off_tol


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panels = [
    (axes[0], COLLAPSE, True, "COLLAPSE (rho~1e8)", 0.0,
     "no fix (merge_tol=0)"),
    (axes[1], MERGER, False, "GALAXY MERGER (rho~1e6, exact dups)", 0.0,
     "no fix (merge_tol=0; exact dups auto-removed)"),
]
for ax, path, per, title, off_tol, off_label in panels:
    out, ot = run(path, per, title, off_tol)
    c, h = out["truth"]; ax.plot(c, h, "k-", lw=2.5, label="particles (truth)")
    c, h, n0 = out["off"]; ax.plot(c, h, "r--", lw=1.8, label=off_label + f"  [{n0} cells]")
    c, h, n1 = out["on"]; ax.plot(c, h, "b-", lw=1.8, label=f"de-sliver fix (median-NN)  [{n1} cells]")
    ax.set_title(title); ax.set_xlabel(r"$\log_{10}\rho$")
    ax.set_ylabel("mass fraction / bin"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Mass-weighted density PDF: de-sliver fix ON vs OFF", fontsize=13)
fig.tight_layout()
OUT = "../desliver_density_pdf.png"
fig.savefig(OUT, dpi=110)
print("wrote", OUT, flush=True)
