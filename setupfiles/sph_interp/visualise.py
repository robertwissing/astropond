"""Visual diagnostics for `sph_interp` — see the interpolated grid and judge how
faithfully it reproduces the SPH particle field.

Two entry points:

  field_image(res, field, ...)        -> a 2D slab image of ANY target geometry
                                          (UniformGrid / AMRGrid / VoronoiGrid /
                                          Projection2D), so every geometry can be
                                          eyeballed the same way.
  visualise(res, particles, field,    -> a multi-panel figure comparing the SPH
            out=...)                     particles to the interpolated grid:
                                          particle truth, interpolated grid,
                                          weighted field distributions, and a per-particle
                                          grid-vs-SPH scatter + mass-conservation
                                          line. Saved to PDF/PNG.

Geometry-aware rendering of the SAME plane for each target:
  UniformGrid   slab-averaged slice (LOS cells within the slab averaged);
  AMRGrid       leaf mosaic (each leaf intersecting the slab painted as its box,
                fine cells over coarse) -> the octree refinement is visible;
  VoronoiGrid   nearest-generator raster at the slice plane -> the true Voronoi
                cells; markers fall back to a generator scatter if asked;
  Projection2D  the 2D map as-is.
"""

import numpy as np

# Importing this module must not pin a backend: matplotlib already falls back to
# a non-interactive backend (Agg) on its own when there is no display, so forcing
# it here only clobbered an interactive backend (GTK3Agg/Qt5Agg/...) chosen by an
# analysis script for an X session, silently losing figures at plt.show().
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, to_rgba


# --------------------------------------------------------------------------- #
#  Geometry-aware 2D image of an interpolated field
# --------------------------------------------------------------------------- #

def _norm_for(values, log=None, robust=False):
    """A field norm; ``robust`` trims display-only extreme tails."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Normalize()
    pos = v[v > 0]
    if log is None:
        log = pos.size > 0 and (pos.max() / max(pos.min(), pos.max() * 1e-12) > 50)
    if log and pos.size:
        if robust and pos.size > 10:
            vmin, vmax = np.quantile(pos, [0.01, 0.995])
        else:
            vmin, vmax = pos.min(), pos.max()
        if vmin == vmax:
            vmin, vmax = vmin / 1.01, vmax * 1.01
        return LogNorm(vmin=vmin, vmax=vmax)
    if robust and v.size > 10:
        vmin, vmax = np.quantile(v, [0.01, 0.995])
    else:
        vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return Normalize(vmin=float(vmin), vmax=float(vmax))


def _shared_norm(groups, log=None, robust=True):
    """One norm spanning robust limits from every non-empty value group.

    Taking limits per group before combining gives the particle render and grid
    render equal influence even when one contains 100,000 points and the other
    only a few hundred visible pixels.
    """
    finite = []
    for values in groups:
        if values is None:
            continue
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            finite.append(v)
    if not finite:
        return Normalize()
    all_values = np.concatenate(finite)
    positive = all_values[all_values > 0.0]
    if log is None:
        log = (positive.size > 0
               and positive.max()
               / max(positive.min(), positive.max() * 1.0e-12) > 50.0)
    limits = []
    for values in finite:
        values = values[values > 0.0] if log else values
        if not values.size:
            continue
        if robust and values.size > 10:
            limits.append(np.quantile(values, [0.01, 0.995]))
        else:
            limits.append(np.array([values.min(), values.max()]))
    if not limits:
        return Normalize()
    vmin = min(pair[0] for pair in limits)
    vmax = max(pair[1] for pair in limits)
    if vmin == vmax:
        vmin, vmax = ((vmin / 1.01, vmax * 1.01) if log
                      else (vmin - 0.5, vmax + 0.5))
    return (LogNorm(vmin=vmin, vmax=vmax) if log
            else Normalize(vmin=vmin, vmax=vmax))


def _weighted_quantile(values, weights, quantiles):
    """Weighted quantiles of finite 1-D values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not good.any():
        return np.quantile(values[np.isfinite(values)], quantiles)
    order = np.argsort(values[good])
    v = values[good][order]
    w = weights[good][order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(quantiles, cdf, v, left=v[0], right=v[-1])


def _central_extent(particles, a1, a2, domain_extent, tail=1.0e-3):
    """Square view containing the central ``1-2*tail`` fraction of gas mass."""
    pos = np.asarray(particles.pos)
    mass = np.asarray(particles.mass)
    q1 = _weighted_quantile(pos[:, a1], mass, [tail, 1.0 - tail])
    q2 = _weighted_quantile(pos[:, a2], mass, [tail, 1.0 - tail])
    centre = np.array([0.5 * q1.sum(), 0.5 * q2.sum()])
    half = 0.55 * max(q1[1] - q1[0], q2[1] - q2[0])
    if not np.isfinite(half) or half <= 0.0:
        return list(domain_extent)
    view = [centre[0] - half, centre[0] + half,
            centre[1] - half, centre[1] + half]
    return [max(view[0], domain_extent[0]),
            min(view[1], domain_extent[1]),
            max(view[2], domain_extent[2]),
            min(view[3], domain_extent[3])]


def _image_values_in_extent(img, domain_extent, view_extent):
    """Finite raster values whose pixel centres lie inside ``view_extent``."""
    n1, n2 = img.shape
    x = np.linspace(domain_extent[0], domain_extent[1], n1, endpoint=False)
    y = np.linspace(domain_extent[2], domain_extent[3], n2, endpoint=False)
    x += 0.5 * (domain_extent[1] - domain_extent[0]) / n1
    y += 0.5 * (domain_extent[3] - domain_extent[2]) / n2
    i = (x >= view_extent[0]) & (x <= view_extent[1])
    j = (y >= view_extent[2]) & (y <= view_extent[3])
    return img[np.ix_(i, j)] if i.any() and j.any() else img


def field_image(res, field, plane=(0, 1), slab_frac=0.05, img_npx=None,
                zmid=None):
    """Rasterize `res.data[field]` onto a 2D image in the `plane` (a1, a2).

    Returns (image2D, extent, (a1, a2)) where image2D is indexed [col=a1, row=a2]
    and `extent = [lo_a1, hi_a1, lo_a2, hi_a2]` (display with image.T,
    origin='lower'). For 3D targets a slab of half-thickness 0.5*slab_frac*box_los
    centred on `zmid` (default mid-plane) is collapsed onto the plane.
    """
    from .targets import UniformGrid, AMRGrid, VoronoiGrid, Projection2D
    grid = res.target
    a1, a2 = plane
    if field not in res.data:
        raise KeyError(f"field {field!r} not in result ({sorted(res.data)})")
    data = np.asarray(res.data[field])

    if isinstance(grid, Projection2D):
        extent = [grid.lo[0], grid.hi[0], grid.lo[1], grid.hi[1]]
        return data, extent, (grid.axis1, grid.axis2)

    if isinstance(grid, UniformGrid):
        los = ({0, 1, 2} - {a1, a2}).pop()
        centres = grid.centres()
        zc = centres[los]
        zm = 0.5 * (grid.lo[los] + grid.hi[los]) if zmid is None else zmid
        half = 0.5 * slab_frac * (grid.hi[los] - grid.lo[los])
        sel = np.abs(zc - zm) <= max(half, grid.pixwidth[los])
        if not sel.any():
            sel[np.argmin(np.abs(zc - zm))] = True
        img = data.take(np.where(sel)[0], axis=los).mean(axis=los)
        # img is now indexed by the two remaining axes in ascending order
        rem = sorted({0, 1, 2} - {los})
        if (a1, a2) != tuple(rem):           # caller wants a2 first
            img = img.T
        extent = [grid.lo[a1], grid.hi[a1], grid.lo[a2], grid.hi[a2]]
        return img, extent, (a1, a2)

    # AMR / Voronoi: rasterize onto an image grid in the plane.
    lo = grid.lo
    hi = grid.hi
    los = ({0, 1, 2} - {a1, a2}).pop()
    zm = 0.5 * (lo[los] + hi[los]) if zmid is None else zmid
    npx = img_npx or 512
    e1 = np.linspace(lo[a1], hi[a1], npx + 1)
    e2 = np.linspace(lo[a2], hi[a2], npx + 1)
    img = np.full((npx, npx), np.nan)

    if isinstance(grid, AMRGrid):
        clo = grid.cell_lo
        csz = grid.cell_size
        chi = clo + csz
        inslab = (clo[:, los] <= zm) & (chi[:, los] >= zm)
        order = np.argsort(grid.cell_level[inslab])      # coarse first
        idx = np.where(inslab)[0][order]
        pw1 = (hi[a1] - lo[a1]) / npx
        pw2 = (hi[a2] - lo[a2]) / npx
        for c in idx:
            i0 = int(np.floor((clo[c, a1] - lo[a1]) / pw1))
            i1 = int(np.ceil((chi[c, a1] - lo[a1]) / pw1))
            j0 = int(np.floor((clo[c, a2] - lo[a2]) / pw2))
            j1 = int(np.ceil((chi[c, a2] - lo[a2]) / pw2))
            img[max(i0, 0):min(i1, npx), max(j0, 0):min(j1, npx)] = data[c]
        extent = [lo[a1], hi[a1], lo[a2], hi[a2]]
        return img, extent, (a1, a2)

    if isinstance(grid, VoronoiGrid):
        from scipy.spatial import cKDTree
        c1 = 0.5 * (e1[:-1] + e1[1:])
        c2 = 0.5 * (e2[:-1] + e2[1:])
        G1, G2 = np.meshgrid(c1, c2, indexing="ij")
        pts = np.empty((G1.size, 3))
        pts[:, a1] = G1.ravel()
        pts[:, a2] = G2.ravel()
        pts[:, los] = zm
        _, nn = cKDTree(grid.centres()).query(pts)   # central cells only
        img = data[nn].reshape(npx, npx)
        extent = [lo[a1], hi[a1], lo[a2], hi[a2]]
        return img, extent, (a1, a2)

    raise NotImplementedError(f"no image for target {type(grid).__name__}")


# --------------------------------------------------------------------------- #
#  Draw the ACTUAL cells (mesh boundaries), filled by the field value
# --------------------------------------------------------------------------- #

def draw_cells(ax, res, field, plane=(0, 1), zoom=None, norm=None,
               cmap="inferno", edgecolor="0.25", lw=0.3, slab=None):
    """Draw the real cell boundaries of `res.target` in the `plane`, each cell
    filled by `res.data[field]`, onto Matplotlib axis `ax`.

    This shows the MESH itself (not a raster): UniformGrid -> regular squares;
    AMRGrid -> the octree leaf boxes that cross the slice plane; VoronoiGrid ->
    the Voronoi polygons (2D tessellation of the generators within a thin slab,
    the faithful slice-plane depiction); Projection2D -> the pixel grid.

    zoom : (x0, x1, y0, y1) in world coords to crop to — cells are only legible
           zoomed in; default is the central quarter of the domain.
    slab : LOS half-thickness for the Voronoi generator selection (default a few
           mean spacings). Ignored by the other geometries.
    """
    from matplotlib.collections import PatchCollection, PolyCollection
    from matplotlib.patches import Rectangle
    from .targets import UniformGrid, AMRGrid, VoronoiGrid, Projection2D
    grid = res.target
    a1, a2 = (grid.axis1, grid.axis2) if isinstance(grid, Projection2D) else plane
    data = np.asarray(res.data[field])
    if norm is None:
        norm = _norm_for(data)

    # default zoom: central quarter of the plane
    if isinstance(grid, Projection2D):
        lo = np.array([grid.lo[0], grid.lo[1]]); hi = np.array([grid.hi[0], grid.hi[1]])
    else:
        lo = np.array([grid.lo[a1], grid.lo[a2]]); hi = np.array([grid.hi[a1], grid.hi[a2]])
    if zoom is None:
        c = 0.5 * (lo + hi); q = 0.25 * (hi - lo)
        zoom = (c[0] - q[0], c[0] + q[0], c[1] - q[1], c[1] + q[1])
    zx0, zx1, zy0, zy1 = zoom

    if isinstance(grid, (UniformGrid, Projection2D)):
        if isinstance(grid, Projection2D):
            e1, e2 = grid.edges(); img = data
        else:
            ed = grid.edges(); e1, e2 = ed[a1], ed[a2]
            img, _, _ = field_image(res, field, plane=(a1, a2), slab_frac=0.05)
        ax.pcolormesh(e1, e2, np.ma.masked_invalid(img.T), cmap=cmap, norm=norm,
                      edgecolors=edgecolor, linewidth=lw, shading="flat")
        ncell = "regular pixels"

    elif isinstance(grid, AMRGrid):
        los = ({0, 1, 2} - {a1, a2}).pop()
        zm = 0.5 * (grid.lo[los] + grid.hi[los])
        clo, csz = grid.cell_lo, grid.cell_size
        chi = clo + csz
        inslab = ((clo[:, los] <= zm) & (chi[:, los] >= zm)
                  & (chi[:, a1] >= zx0) & (clo[:, a1] <= zx1)
                  & (chi[:, a2] >= zy0) & (clo[:, a2] <= zy1))
        idx = np.where(inslab)[0]
        rects = [Rectangle((clo[c, a1], clo[c, a2]), csz[c, a1], csz[c, a2])
                 for c in idx]
        pc = PatchCollection(rects, cmap=cmap, norm=norm,
                             edgecolor=edgecolor, linewidth=lw)
        pc.set_array(data[idx])
        ax.add_collection(pc)
        ncell = f"{idx.size} leaf cells in slice"

    elif isinstance(grid, VoronoiGrid):
        from scipy.spatial import Voronoi
        los = ({0, 1, 2} - {a1, a2}).pop()
        gen = grid.centres()                         # central cells only
        zm = 0.5 * (grid.lo[los] + grid.hi[los])
        if slab is None:
            slab = 2.0 * (np.prod(grid.hi - grid.lo) / grid.ncell) ** (1.0 / 3.0)
        sel = (np.abs(gen[:, los] - zm) <= slab)
        # keep a margin around the zoom so boundary polygons close
        mx = 0.15 * (zx1 - zx0); my = 0.15 * (zy1 - zy0)
        sel &= ((gen[:, a1] >= zx0 - mx) & (gen[:, a1] <= zx1 + mx)
                & (gen[:, a2] >= zy0 - my) & (gen[:, a2] <= zy1 + my))
        idx = np.where(sel)[0]
        pts2 = np.column_stack([gen[idx, a1], gen[idx, a2]])
        vor = Voronoi(pts2)
        polys, vals = [], []
        for k, reg_i in enumerate(vor.point_region):
            reg = vor.regions[reg_i]
            if not reg or -1 in reg:                 # skip open (boundary) cells
                continue
            polys.append(vor.vertices[reg])
            vals.append(data[idx[k]])
        pcoll = PolyCollection(polys, cmap=cmap, norm=norm,
                               edgecolor=edgecolor, linewidth=lw)
        pcoll.set_array(np.asarray(vals))
        ax.add_collection(pcoll)
        ncell = f"{len(polys)} Voronoi cells in slab"
    else:
        raise NotImplementedError(type(grid).__name__)

    ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1)
    ax.set_aspect("equal")
    return norm, ncell


def _cube_faces(lo, size):
    """The 6 quad faces of an axis-aligned box, as a list of (4,3) vertex arrays."""
    x0, y0, z0 = lo
    x1, y1, z1 = lo + size
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6),
         (1, 2, 6, 5), (0, 3, 7, 4)]
    return [v[list(q)] for q in f]


def draw_cells_3d(ax, res, field, subbox=None, norm=None, cmap="inferno",
                  alpha=0.55, alpha_min=0.04, edgecolor="k", lw=0.3,
                  max_cells=1500, cull_percentile=None, value_alpha=True,
                  clip=None):
    """Render the ACTUAL 3D cells of `res.target` as polyhedra/boxes on a 3D axis
    `ax`, each face coloured by `res.data[field]`.

    The faithful counterpart to `draw_cells`: VoronoiGrid draws the true polyhedra
    from the stored face/vertex CSR (not a 2D slab approximation); AMRGrid and
    UniformGrid draw their leaf/cell cubes.

    This is only legible for a SMALL number of LARGE cells — feed it a COARSE
    mesh (a few hundred cells), not the interpolation mesh. Decluttering, like
    the original Voronoi-density figure:
      value_alpha       per-cell opacity scales `alpha_min`..`alpha` with the
                        normalised field value, so voids fade out;
      cull_percentile   drop cells whose value is below this percentile;
      max_cells         if still too many, keep the HIGHEST-value cells (a
                        principled cull — never a random subsample, which would
                        punch holes in the tessellation).
    subbox (xlo,xhi,...,zhi) crops to a region; default = the whole domain.
    clip : a CUTAWAY — list of (axis, 'low'|'high', pos). A cell is removed only
           if its centre is on the named side of EVERY plane (AND), so one entry
           cuts a half, two a quarter wedge, three the near octant — exposing the
           core / central refinement behind the cut faces.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .targets import UniformGrid, AMRGrid, VoronoiGrid, Projection2D
    grid = res.target
    if isinstance(grid, Projection2D):
        raise NotImplementedError("Projection2D is a 2D target — use draw_cells")
    data = np.asarray(res.data[field])
    if norm is None:
        norm = _norm_for(data)
    cm = plt.get_cmap(cmap)

    # per-cell value + geometry (flatten UniformGrid to (Ncell,) C-order)
    if isinstance(grid, UniformGrid):
        cx, cy, cz = grid.centres()
        C = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), axis=-1)
        centres = C.reshape(-1, 3)
        cellval = data.ravel()
    else:
        centres = np.asarray(grid.centres())
        cellval = data
    lo = np.asarray(grid.lo); hi = np.asarray(grid.hi)
    if subbox is None:
        subbox = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    bx = np.array(subbox).reshape(3, 2)
    idx = np.where(np.all((centres >= bx[:, 0]) & (centres <= bx[:, 1]),
                          axis=1))[0]

    if clip:                                              # cutaway: drop the wedge
        cen = centres[idx]
        rm = np.ones(idx.size, dtype=bool)
        for ax_, side, pos in clip:
            rm &= (cen[:, ax_] > pos) if side == "high" else (cen[:, ax_] < pos)
        idx = idx[~rm]

    v_in = cellval[idx]
    if cull_percentile is not None and idx.size:          # drop the void cells
        keep = v_in >= np.percentile(v_in, cull_percentile)
        idx, v_in = idx[keep], v_in[keep]
    if idx.size > max_cells:                               # keep densest cells
        # SAY SO. This keeps the densest `max_cells` and silently drops the
        # rest, which are by construction the LOW-density outer cells -- so a
        # capped plot shows a mesh that stops at some density contour and
        # reads as "there are no cells out there". On an open capture the
        # background guard alone adds 16^3 = 4096 cells, so a coarse Voronoi
        # asked for 1500 generators arrives here with ~5600 and loses 73% of
        # them without a word.
        print(f"  draw_cells_3d: drawing the {max_cells} densest of "
              f"{idx.size} cells ({100.0 * (1 - max_cells / idx.size):.0f}% "
              f"not shown); raise max_cells to see the full mesh",
              flush=True)
        top = np.argsort(v_in)[-max_cells:]
        idx, v_in = idx[top], v_in[top]

    # per-cell opacity (voids fade); repeated per face below
    a01 = np.clip(np.asarray(norm(v_in)), 0.0, 1.0)
    cell_alpha = (alpha_min + (alpha - alpha_min) * a01 if value_alpha
                  else np.full(idx.size, alpha))

    faces, facevals, facealpha = [], [], []
    if isinstance(grid, VoronoiGrid):
        for c, ci in enumerate(idx):
            for f in range(grid.cf_start[ci], grid.cf_start[ci + 1]):
                s, e = grid.fv_start[f], grid.fv_start[f + 1]
                faces.append(np.column_stack([grid.fvx[s:e], grid.fvy[s:e],
                                              grid.fvz[s:e]]))
                facevals.append(cellval[ci]); facealpha.append(cell_alpha[c])
        label = f"{idx.size} Voronoi polyhedra"
    else:
        if isinstance(grid, UniformGrid):
            i, j, k = np.unravel_index(idx, grid.npx)
            pw = grid.pixwidth
            cell_lo = np.column_stack([grid.lo[a] + ijk * pw[a]
                                       for a, ijk in enumerate((i, j, k))])
            cell_sz = np.tile(pw, (idx.size, 1))
        else:                                             # AMRGrid
            cell_lo, cell_sz = grid.cell_lo[idx], grid.cell_size[idx]
        for c in range(idx.size):
            for face in _cube_faces(cell_lo[c], cell_sz[c]):
                faces.append(face)
                facevals.append(v_in[c]); facealpha.append(cell_alpha[c])
        label = f"{idx.size} cells"

    fa = np.asarray(facealpha)
    colors = cm(norm(np.asarray(facevals)))
    colors[:, 3] = fa
    # fade the edges with the cell too, else void cells draw a solid black
    # wireframe that swamps the structure.
    ergba = np.array(to_rgba(edgecolor))
    ecolors = np.tile(ergba, (fa.size, 1))
    ecolors[:, 3] = np.clip(fa * 1.1, 0.0, 1.0)
    pc = Poly3DCollection(faces, facecolors=colors, edgecolors=ecolors,
                          linewidths=lw)
    ax.add_collection3d(pc)
    ax.set_xlim(subbox[0], subbox[1]); ax.set_ylim(subbox[2], subbox[3])
    ax.set_zlim(subbox[4], subbox[5])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    return norm, label


# --------------------------------------------------------------------------- #
#  Sample the interpolated grid back at the particle positions
# --------------------------------------------------------------------------- #

def sample_at_particles(res, particles, field):
    """Interpolated value at each particle position, or None for Projection2D.

    UniformGrid -> exact cell lookup; VoronoiGrid -> nearest generator (which IS
    the containing cell, by definition); AMRGrid -> exact containing leaf.
    """
    from .targets import UniformGrid, AMRGrid, VoronoiGrid, Projection2D
    grid = res.target
    data = np.asarray(res.data[field])
    pos = np.asarray(particles.pos, dtype=np.float64)

    if isinstance(grid, Projection2D):
        return None                          # a LOS integral, no per-particle value

    if isinstance(grid, UniformGrid):
        idx = []
        for a in range(3):
            i = np.floor((pos[:, a] - grid.lo[a]) / grid.pixwidth[a]).astype(int)
            idx.append(np.clip(i, 0, grid.npx[a] - 1))
        return data[idx[0], idx[1], idx[2]]

    from scipy.spatial import cKDTree
    if isinstance(grid, VoronoiGrid):
        _, nn = cKDTree(grid.centres()).query(pos)   # central cells only
        return data[nn]
    if isinstance(grid, AMRGrid):
        return data[_amr_containing_cells(grid, pos)]
    return None


def _amr_containing_cells(grid, positions):
    """Leaf index containing each position in a space-filling AMR grid."""
    pos = np.asarray(positions, dtype=np.float64)
    box = np.asarray(grid.hi) - np.asarray(grid.lo)
    query = pos.copy()
    for axis, periodic in enumerate(grid.periodic):
        if periodic:
            query[:, axis] = ((query[:, axis] - grid.lo[axis]) % box[axis]
                              + grid.lo[axis])
    query = np.minimum(np.maximum(query, grid.lo),
                       np.nextafter(grid.hi, grid.lo))

    levels = np.asarray(grid.cell_level, dtype=np.int64)
    centres = grid.centres()
    found = np.full(len(query), -1, dtype=np.int64)
    for level in range(int(levels.max()) + 1):
        cells = np.where(levels == level)[0]
        if cells.size == 0:
            continue
        dims = np.asarray(grid.base_npx, dtype=np.int64) * (2 ** level)
        width = box / dims
        cijk = np.floor((centres[cells] - grid.lo) / width).astype(np.int64)
        cijk = np.minimum(np.maximum(cijk, 0), dims - 1)
        ckeys = (cijk[:, 0] * dims[1] + cijk[:, 1]) * dims[2] + cijk[:, 2]
        order = np.argsort(ckeys)
        ckeys, cells = ckeys[order], cells[order]

        unresolved = np.where(found < 0)[0]
        pijk = np.floor((query[unresolved] - grid.lo) / width).astype(np.int64)
        pijk = np.minimum(np.maximum(pijk, 0), dims - 1)
        pkeys = (pijk[:, 0] * dims[1] + pijk[:, 1]) * dims[2] + pijk[:, 2]
        where = np.searchsorted(ckeys, pkeys)
        safe = np.minimum(where, len(ckeys) - 1)
        hit = (where < len(ckeys)) & (ckeys[safe] == pkeys)
        found[unresolved[hit]] = cells[safe[hit]]
    if np.any(found < 0):
        raise RuntimeError("AMR leaf lookup found an uncovered particle")
    return found


def _cell_volumes(grid, field_shape):
    """Per-cell volume array matching the flattened field, or None (2D)."""
    from .targets import UniformGrid, AMRGrid, VoronoiGrid
    if isinstance(grid, UniformGrid):
        return np.full(int(np.prod(grid.npx)), grid.cell_volume)
    if isinstance(grid, (AMRGrid, VoronoiGrid)):
        return grid.volumes()
    return None


def _mass_check(res, particles):
    """(grid_mass, particle_mass) for a 'rho' field, or None."""
    from .targets import UniformGrid, AMRGrid, VoronoiGrid
    grid = res.target
    pm = float(np.asarray(particles.mass).sum())
    if "_mass" in res.data:
        return float(np.asarray(res.data["_mass"]).sum()), pm
    if "rho" not in res.data:
        return None
    if isinstance(grid, UniformGrid):
        return float(res.data["rho"].sum() * grid.cell_volume), pm
    if isinstance(grid, (AMRGrid, VoronoiGrid)):
        return float((res.data["rho"] * grid.volumes()).sum()), pm
    return None


def resolution_audit(grid, particles):
    """Characteristic finest grid scale versus the particle smoothing scales.

    ``dx_min`` is the minimum volume-equivalent cell width (area-equivalent
    pixel width for a projection).  ``h_p05`` is a robust small-h scale: 95% of
    particles have smoothing lengths at least this large.
    """
    from .targets import AMRGrid, Projection2D, UniformGrid, VoronoiGrid

    if isinstance(grid, UniformGrid):
        dx = np.array([grid.cell_volume**(1.0 / 3.0)])
    elif isinstance(grid, (AMRGrid, VoronoiGrid)):
        volume = np.asarray(grid.volumes(), dtype=np.float64)
        dx = np.cbrt(volume)
    elif isinstance(grid, Projection2D):
        dx = np.array([np.sqrt(np.prod(grid.pixwidth))])
    else:
        return None

    dx = dx[np.isfinite(dx) & (dx > 0.0)]
    h = np.asarray(particles.h, dtype=np.float64).ravel()
    h = h[np.isfinite(h) & (h > 0.0)]
    if not dx.size or not h.size:
        return None
    dx_min = float(dx.min())
    h_min = float(h.min())
    h_p05 = float(np.quantile(h, 0.05))
    return {
        "dx_min": dx_min,
        "h_min": h_min,
        "h_p05": h_p05,
        "dx_over_h_min": dx_min / h_min,
        "dx_over_h_p05": dx_min / h_p05,
    }


def resolution_text(grid, particles):
    """Compact human-readable form of :func:`resolution_audit`."""
    audit = resolution_audit(grid, particles)
    if audit is None:
        return None
    return (
        f"resolution: dx_min={audit['dx_min']:.4g} (cell-equivalent), "
        f"h_min={audit['h_min']:.4g}, h_p05={audit['h_p05']:.4g}; "
        f"dx_min/h_min={audit['dx_over_h_min']:.3g}, "
        f"dx_min/h_p05={audit['dx_over_h_p05']:.3g}"
    )


def _pdf_curve(values, weights, edges, log=False):
    """Weighted probability density per linear unit or per log10 interval."""
    values = np.asarray(values, dtype=np.float64).ravel()
    good = np.isfinite(values)
    if log:
        good &= values > 0.0
    if weights is None:
        w = None
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        good &= np.isfinite(weights) & (weights >= 0.0)
        w = weights[good]
    counts, _ = np.histogram(values[good], bins=edges, weights=w)
    # Normalise by the full valid population, not only samples inside `edges`.
    # This matters when a diagnostic deliberately crops an uninteresting tail:
    # probability outside the visible range must not be redistributed into it.
    total = float(good.sum()) if w is None else float(w.sum())
    widths = np.diff(np.log10(edges) if log else edges)
    return (np.zeros_like(counts, dtype=np.float64) if total == 0.0
            else counts / (total * widths))


# --------------------------------------------------------------------------- #
#  The full diagnostic figure
# --------------------------------------------------------------------------- #

def density_pdf_panel(ax, res, particles, field="rho", log=None, gname=None):
    """Material-weighted field PDF: SPH particles vs mesh cells vs the mesh
    sampled back at the particle positions.

    Extracted from `visualise` so the CLI diagnostic and the demo draw the
    SAME panel rather than two implementations that can drift apart.

    Returns (gsamp, parts, use_log) — the caller needs them for
    `grid_vs_particle_panel`, and recomputing `sample_at_particles` there would
    double the cost of the most expensive part of the diagnostic.
    """
    from .targets import Projection2D
    grid = res.target
    gname = gname or type(grid).__name__
    is_proj = isinstance(grid, Projection2D)
    pvals = (np.asarray(particles.values[field])
             if field in particles.values else None)

    gsamp = None if is_proj else sample_at_particles(res, particles, field)
    cells = np.asarray(res.data[field]).ravel()
    cvol = _cell_volumes(grid, cells.shape)
    if is_proj:
        cwt, parts, pwt = None, None, None
        wlab = "area-weighted map"
    elif "_mass" in res.data:
        cwt = np.asarray(res.data["_mass"]).ravel()
        parts, pwt = pvals.ravel() if pvals is not None else None, particles.mass
        wlab = "mass-weighted"
    elif "rho" in res.data and cvol is not None:
        cwt = np.asarray(res.data["rho"]).ravel() * cvol
        parts, pwt = pvals.ravel() if pvals is not None else None, particles.mass
        wlab = "mass-weighted"
    else:
        cwt = cvol
        parts = pvals.ravel() if pvals is not None else None
        pwt = (particles.mass / particles.rho
               if parts is not None and cvol is not None else None)
        wlab = "volume-weighted" if cvol is not None else "unweighted"

    if log is None:                       # decide from the data, as _shared_norm would
        pos_only = cells[np.isfinite(cells) & (cells > 0)]
        log = bool(pos_only.size and
                   pos_only.max() / max(pos_only.min(), 1e-300) > 50.0)
    use_log = bool(log)

    finite_cells = cells[np.isfinite(cells) & ((cells > 0.0) if use_log else True)]
    finite_parts = (np.array([], dtype=float) if parts is None else
                    parts[np.isfinite(parts) & ((parts > 0.0) if use_log else True)])
    combined = np.concatenate([finite_cells, finite_parts])
    if combined.size == 0:
        combined = np.array([1.0, 10.0]) if use_log else np.array([0.0, 1.0])
    # Start just below the minimum SPH particle value.  Very low background
    # grid cells can otherwise consume most of a logarithmic PDF axis even
    # though they are outside the density range represented by particles.
    floor_source = finite_parts if finite_parts.size else combined
    particle_floor = float(floor_source.min())
    hi_pdf = float(combined.max())
    if use_log:
        lo_pdf = particle_floor / 10.0**0.05
    else:
        span = max(hi_pdf - particle_floor, abs(particle_floor), 1.0)
        lo_pdf = particle_floor - 0.02 * span
    if lo_pdf == hi_pdf:
        lo_pdf, hi_pdf = ((lo_pdf / 1.01, hi_pdf * 1.01) if use_log
                          else (lo_pdf - 0.5, hi_pdf + 0.5))
    elif hi_pdf < lo_pdf:
        hi_pdf = (lo_pdf * 1.01 if use_log else lo_pdf + 1.0)
    if use_log:
        bins = np.logspace(np.log10(lo_pdf), np.log10(hi_pdf), 60)
        ax.set_xscale("log")
    else:
        bins = np.linspace(lo_pdf, hi_pdf, 60)

    if parts is not None:
        pp = _pdf_curve(parts, pwt, bins, log=use_log)
        ax.stairs(pp, bins, lw=2, color="tab:blue", label="SPH particles")
    if gsamp is not None:
        sp = _pdf_curve(gsamp, particles.mass, bins, log=use_log)
        ax.stairs(sp, bins, lw=1.6, ls="--", color="tab:red",
                  label="grid sampled at particles")
    gp = _pdf_curve(cells, cwt, bins, log=use_log)
    ax.stairs(gp, bins, fill=True, alpha=0.4, color="tab:orange",
              label=f"{gname} cells")
    unit = f"dP/dlog10({field})" if use_log else f"dP/d{field}"
    ax.set_title(f"{field} {wlab} distribution")
    ax.set_xlabel(field); ax.set_ylabel(unit)
    ax.set_xlim(left=lo_pdf)
    ax.legend(fontsize=9)
    return gsamp, parts, use_log


def grid_vs_particle_panel(ax, res, particles, field="rho", gsamp=None,
                           parts=None, use_log=True, gname=None,
                           extra_lines=()):
    """Interpolated value at each particle against that particle's own value,
    with the 1:1 line, Pearson r and the median ratio.

    Returns (r, median_ratio), both NaN when the target cannot be sampled
    per-particle (Projection2D is a line-of-sight integral, so there is no
    per-particle counterpart to compare against).
    """
    from .targets import Projection2D
    grid = res.target
    gname = gname or type(grid).__name__
    is_proj = isinstance(grid, Projection2D)

    if gsamp is not None and parts is not None:
        m = np.isfinite(gsamp) & np.isfinite(parts)
        x = parts[m]; y = gsamp[m]
        if m.sum() > 50000:                  # subsample for a readable scatter
            sub = np.random.default_rng(0).choice(m.sum(), 50000, replace=False)
            x, y = x[sub], y[sub]
        ax.scatter(x, y, s=2, alpha=0.2, color="k", linewidths=0)
        good = (x > 0) & (y > 0) if use_log else np.ones(x.shape, bool)
        if use_log:
            ax.set_xscale("log"); ax.set_yscale("log")
        if good.any():
            lim_lo = min(x[good].min(), y[good].min())
            lim_hi = max(x[good].max(), y[good].max())
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r-", lw=1.5,
                    label="1:1")
            xx = x[good]; yy = y[good]
            if use_log:
                xx, yy = np.log10(xx), np.log10(yy)
            varying = (xx.size > 1 and np.std(xx) > 0.0
                       and np.std(yy) > 0.0)
            r = np.corrcoef(xx, yy)[0, 1] if varying else float("nan")
            ratio = good & (x != 0.0)
            med = (np.median(y[ratio] / x[ratio])
                   if ratio.any() else float("nan"))
        else:
            r = med = float("nan")
        ax.set_title(f"grid sampled at particles vs SPH\n"
                     f"r={r:.4f}, median(grid/SPH)={med:.3f}")
        ax.set_xlabel(f"SPH particle {field}")
        ax.set_ylabel(f"interpolated {field}")
        ax.legend(fontsize=9)
        return r, med

    ax.axis("off")
    lines = [f"target  : {gname}",
             f"method  : {res.method}",
             f"field   : {field}"]
    lines += list(extra_lines)
    if is_proj:
        lines.append(f"mode    : {grid.mode}")
        lines.append("(projection is a LOS integral —")
        lines.append(" no per-particle 1:1 comparison)")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=12, transform=ax.transAxes)
    return float("nan"), float("nan")


def visualise(res, particles, field="rho", out=None, plane=(0, 1),
              slab_frac=0.05, img_npx=None, title=None, log=None,
              cmap="turbo", center="mass", view="central"):
    """Compare the SPH particles to the interpolated grid; save to `out`.

    Panels (3D targets): particle truth (slab scatter) | interpolated grid
    (geometry-aware slice) | field PDFs (particles vs grid cells) | per-particle
    grid-vs-SPH scatter with the 1:1 line + a mass-conservation line.
    For Projection2D the per-particle scatter is replaced by an info panel.

    Returns the Matplotlib Figure.
    """
    from .targets import Projection2D, UniformGrid
    grid = res.target
    gname = type(grid).__name__
    is_proj = isinstance(grid, Projection2D)
    a1, a2 = (grid.axis1, grid.axis2) if is_proj else plane
    axlabel = "xyz"

    if is_proj:
        los = None
        zmid = None
        slice_half = None
    else:
        los = ({0, 1, 2} - {a1, a2}).pop()
        if center == "mass":
            zmid = float(np.average(particles.pos[:, los],
                                    weights=particles.mass))
        elif center == "domain":
            zmid = float(0.5 * (grid.lo[los] + grid.hi[los]))
        elif center == "origin":
            zmid = 0.0
        else:
            raise ValueError("center must be 'mass', 'domain', or 'origin'")
        half = 0.5 * slab_frac * (grid.hi[los] - grid.lo[los])
        slice_half = (max(half, grid.pixwidth[los])
                      if isinstance(grid, UniformGrid) else 0.0)

    img, extent, (pa1, pa2) = field_image(
        res, field, plane=(a1, a2), slab_frac=slab_frac, img_npx=img_npx,
        zmid=zmid)
    if view == "central" and not is_proj:
        plot_extent = _central_extent(particles, a1, a2, extent)
    elif view == "domain" or is_proj:
        plot_extent = list(extent)
    else:
        raise ValueError("view must be 'central' or 'domain'")
    pvals = np.asarray(particles.values[field]) if field in particles.values \
        else (np.asarray(particles.rho) if field == "rho" else None)

    norm_values = _image_values_in_extent(img, extent, plot_extent)
    base_cmap = plt.get_cmap(cmap)
    plot_cmap = base_cmap.with_extremes(bad=base_cmap(0.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(title or f"{gname} / {res.method} — field '{field}'",
                 fontsize=14, fontweight="bold")

    # --- panel A: particle truth (slab scatter) -----------------------------
    ax = axes[0, 0]
    pos = np.asarray(particles.pos)
    if is_proj:
        sel = np.ones(particles.n, dtype=bool)
        slabtxt = "all particles (projected)"
    else:
        support = 2.0 * np.asarray(particles.h)
        sel = np.abs(pos[:, los] - zmid) <= slice_half + support
        slabtxt = (f"kernels intersect {axlabel[los]}={zmid:.3g} "
                   f"({sel.sum()} pts)")
    sel &= ((pos[:, a1] >= plot_extent[0])
            & (pos[:, a1] <= plot_extent[1])
            & (pos[:, a2] >= plot_extent[2])
            & (pos[:, a2] <= plot_extent[3]))
    if not is_proj:
        slabtxt = (f"kernels intersect {axlabel[los]}={zmid:.3g} "
                   f"({sel.sum()} pts)")
    take = np.where(sel)[0]
    if take.size > 100000:
        take = np.random.default_rng(0).choice(take, 100000, replace=False)
        slabtxt += ", showing 100000"
    csc = (pvals[take] if pvals is not None else None)
    norm = _shared_norm((norm_values, csc), log=log, robust=True)
    sc = ax.scatter(pos[take, a1], pos[take, a2], c=csc, s=2,
                    cmap=plot_cmap, norm=norm, linewidths=0)
    ax.set_title(f"SPH particles — {slabtxt}")
    ax.set_xlabel(axlabel[a1]); ax.set_ylabel(axlabel[a2])
    ax.set_aspect("equal")
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])
    if csc is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=field)

    # --- panel B: interpolated grid (geometry-aware) ------------------------
    ax = axes[0, 1]
    im = ax.imshow(np.ma.masked_invalid(img.T), origin="lower", extent=extent,
                   cmap=plot_cmap, norm=norm, aspect="equal",
                   interpolation="nearest")
    modetxt = (f" [{grid.mode}]" if is_proj else
               (" slab-slice" if isinstance(grid, UniformGrid)
                else " plane-slice"))
    ax.set_title(f"{gname} interpolation{modetxt}")
    ax.set_xlabel(axlabel[a1]); ax.set_ylabel(axlabel[a2])
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=field)

    # --- panel C: material-weighted field PDFs (particles vs grid cells) -----
    gsamp, parts, use_log = density_pdf_panel(
        axes[1, 0], res, particles, field,
        log=isinstance(norm, LogNorm), gname=gname)

    # --- panel D: per-particle grid-vs-SPH scatter or info ------------------
    mass = _mass_check(res, particles)
    grid_vs_particle_panel(
        axes[1, 1], res, particles, field, gsamp=gsamp, parts=parts,
        use_log=use_log, gname=gname,
        extra_lines=(f"plane   : {axlabel[a1]}{axlabel[a2]}",))

    restext = resolution_text(grid, particles)
    if restext is not None:
        fig.text(0.5, 0.025, restext, ha="center", fontsize=9)
    if mass is not None:
        gm, pm = mass
        rel_mass = gm / pm - 1.0
        if "_mass" in res.data:
            tag = "conserved" if abs(rel_mass) <= 1.0e-5 else "FAILED"
        else:
            tag = "approx (SPH)"
        fig.text(0.5, 0.005,
                 f"mass: grid {gm:.6g} vs particles {pm:.6g}  "
                 f"({100*rel_mass:+.3f}%, {tag})",
                 ha="center", fontsize=11,
                 color=("crimson" if tag == "FAILED" else "black"))

    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    if out:
        fig.savefig(out, dpi=130)
        print(f"  wrote visualisation -> {out}")
    return fig
