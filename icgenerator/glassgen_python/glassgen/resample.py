"""Native particle split / merge for multi-resolution glass relaxation.

Ports the geometry of SPH_methods.split_particless / merge_my_neighbor
(setupfiles/SPH_methods.py) but operates directly on glassgen's (pos, mass,
aux-dict) arrays and uses glassgen's own NeighborList for the
nearest-neighbour directions - no dependency on the SPH 21-column format.

split() doubles the particle count per pass (binary 1->2, nn_ort placement:
children sit at parent +/- ortho*dr*dist where ortho is perpendicular to the
nearest-neighbour direction). A target multiplier of 2^p is reached by
repeating the binary pass p times, rebuilding neighbours each pass so each
split follows the *current* local geometry. An adaptive separation floor
(SPH_methods.py:871-888) skips neighbours closer than the relaxed-glass
minimum spacing so children are never placed on top of a just-split sibling.

merge() is the inverse (mutual-nearest pairing). Both are seed operators:
rho and h are placeholders, re-derived by the next relax().

Per-field rules (rule table; extend by adding field names):
  mass        : split -> halve     ; merge -> sum
  pos         : split -> +/- offset ; merge -> mass-weighted centroid
  HALVE_FIELDS: split -> halve     ; merge -> sum      (extensive: u, B, momenti)
  everything else (vel, rho, h, soft, metals, spin, tform, phi, ...):
                split -> inherit   ; merge -> mass-weighted mean
  (vel as a mass-weighted mean is exactly momentum conservation.)
"""
import numpy as np
from numba import njit, prange

from .kernels import W0, wzero_wc2
from .neighbors import NeighborList
from .sph_core import density_loop

# seed (a fixed phase offset) for the deterministic per-particle split-plane
# rotation hash; change it to draw a different reproducible orientation set.
_ROT_SEED = 0.6180339887

# Fields that are halved on split / summed on merge (extensive per particle),
# matching SPH_methods.split_particless (u/2, B/2, momenti/2 at lines 948-967).
HALVE_FIELDS = frozenset({'u', 'B', 'momenti'})


@njit(cache=True, parallel=True)
def _nn_dirs(pos, idx, h, box, periodic, nsmooth, dist_floor):
    """For each particle find, among its kNN candidates, the primary split
    neighbour (smallest min-image distance whose d/h >= dist_floor, i.e. at
    least the relaxed-glass spacing) and a secondary neighbour (the 2nd-nearest
    that also clears the floor) used to orient the orthogonal split plane.
    Falling back to the nearest non-primary candidate only when a single
    neighbour clears the floor.

    Returns vnn (N,3) min-image vector to the primary, dr (N,) its length, and
    vort (N,3) vector to the secondary (the ort_neigh_vector of SPH_methods).
    Falls back to the farthest candidate when none clears the floor."""
    n = pos.shape[0]
    k = idx.shape[1]
    bx, by, bz = box[0], box[1], box[2]
    vnn = np.zeros((n, 3))
    vort = np.zeros((n, 3))
    dr = np.zeros(n)
    for i in prange(n):
        floor = dist_floor * h[i]
        # primary + secondary: the two nearest neighbours clearing the floor
        # (>= relaxed-glass spacing) -> the old SPH split's rank-k / rank-(k+1)
        # pair; track farthest overall as the primary fallback.
        bp = -1
        bpd = 1e30
        bp2 = -1
        bp2d = 1e30
        bfar = -1
        bfard = -1.0
        for c in range(k):
            j = idx[i, c]
            if j < 0:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= 0.0:
                continue
            if d > bfard:
                bfard = d
                bfar = c
            if d >= floor:
                if d < bpd:
                    bp2d = bpd      # demote old primary to the secondary slot
                    bp2 = bp
                    bpd = d
                    bp = c
                elif d < bp2d:
                    bp2d = d
                    bp2 = c
        if bp < 0:
            bp = bfar  # nothing cleared the floor: take the farthest
        # secondary = smallest d distinct from the chosen primary
        bs = -1
        bsd = 1e30
        for c in range(k):
            if c == bp:
                continue
            j = idx[i, c]
            if j < 0:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= 0.0:
                continue
            if d < bsd:
                bsd = d
                bs = c
        if bp2 >= 0:
            bs = bp2  # prefer the 2nd floor-cleared neighbour (old rank k+1)
        if bp >= 0:
            j = idx[i, bp]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            vnn[i, 0] = dx
            vnn[i, 1] = dy
            vnn[i, 2] = dz
            dr[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
        if bs >= 0:
            j = idx[i, bs]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            vort[i, 0] = dx
            vort[i, 1] = dy
            vort[i, 2] = dz
    return vnn, dr, vort


@njit(cache=True, parallel=True)
def _place_children(pos, vnn, dr, vort, box, periodic, dist, rotamp, out):
    """Fill out (2N,3): out[2i]/out[2i+1] = pos[i] +/- ortho*dr*dist, ortho a
    unit vector perpendicular to the nearest-neighbour direction (cross with
    the secondary direction; arbitrary-axis fallback when parallel), then
    rotated about that direction by a per-particle continuous pseudo-random
    angle to decorrelate neighbouring splits (avoids sibling-daughter
    coincidences).
    Children wrapped into the box (single image: offset = dist*dr <= ~0.4 h
    << L/2)."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        d = dr[i]
        if d <= 0.0:
            # no usable neighbour: tiny deterministic split along x
            ox, oy, oz = 1.0, 0.0, 0.0
            disp = 1e-6
        else:
            ux, uy, uz = vnn[i, 0] / d, vnn[i, 1] / d, vnn[i, 2] / d
            # ortho = u x vort
            ox = uy * vort[i, 2] - uz * vort[i, 1]
            oy = uz * vort[i, 0] - ux * vort[i, 2]
            oz = ux * vort[i, 1] - uy * vort[i, 0]
            nrm = np.sqrt(ox * ox + oy * oy + oz * oz)
            if nrm < 1e-8:
                # parallel: cross u with the smallest-magnitude axis
                ax, ay, az = abs(ux), abs(uy), abs(uz)
                if ax < ay and ax < az:
                    tx, ty, tz = 1.0, 0.0, 0.0
                elif ay < az:
                    tx, ty, tz = 0.0, 1.0, 0.0
                else:
                    tx, ty, tz = 0.0, 0.0, 1.0
                ox = uy * tz - uz * ty
                oy = uz * tx - ux * tz
                oz = ux * ty - uy * tx
                nrm = np.sqrt(ox * ox + oy * oy + oz * oz)
            ox, oy, oz = ox / nrm, oy / nrm, oz / nrm
            # decorrelate the split-plane azimuth: rotate ortho about u by a
            # per-particle CONTINUOUS pseudo-random angle (deterministic hash of
            # position -> reproducible, parallel-safe). Without this, adjacent
            # sibling parents in a regular glass region get identical
            # (vnn, vort) -> identical ortho -> their inner daughters can land on
            # the exact same point (coincidence degeneracy fatal for a
            # density/neighbour-based h). A continuous angle drops the residual
            # collision rate to ~2^-52; discrete (e.g. 45 deg) buckets leave a
            # 1/nbucket chance two siblings share a bucket and still collide.
            hsh = np.sin(pos[i, 0] * 12.9898 + pos[i, 1] * 78.233
                         + pos[i, 2] * 37.719 + _ROT_SEED) * 43758.5453
            # continuous uniform in [-rotamp, +rotamp]: rotamp=pi is the full
            # circle (azimuth fully re-randomized); a small rotamp keeps the
            # secondary neighbour's azimuth up to a decorrelating wiggle
            theta = rotamp * (2.0 * (hsh - np.floor(hsh)) - 1.0)
            ct, st = np.cos(theta), np.sin(theta)
            # cross = u x ortho: unit, perpendicular to both u and ortho, so
            # ortho*cos + cross*sin is ortho rotated about u (stays unit, _|_ u)
            cxx = uy * oz - uz * oy
            cyy = uz * ox - ux * oz
            czz = ux * oy - uy * ox
            ox, oy, oz = (ox * ct + cxx * st,
                          oy * ct + cyy * st,
                          oz * ct + czz * st)
            disp = d * dist
        x1 = pos[i, 0] + ox * disp
        y1 = pos[i, 1] + oy * disp
        z1 = pos[i, 2] + oz * disp
        x2 = pos[i, 0] - ox * disp
        y2 = pos[i, 1] - oy * disp
        z2 = pos[i, 2] - oz * disp
        if periodic:
            if x1 > hbx:
                x1 -= bx
            elif x1 < -hbx:
                x1 += bx
            if y1 > hby:
                y1 -= by
            elif y1 < -hby:
                y1 += by
            if z1 > hbz:
                z1 -= bz
            elif z1 < -hbz:
                z1 += bz
            if x2 > hbx:
                x2 -= bx
            elif x2 < -hbx:
                x2 += bx
            if y2 > hby:
                y2 -= by
            elif y2 < -hby:
                y2 += by
            if z2 > hbz:
                z2 -= bz
            elif z2 < -hbz:
                z2 += bz
        out[2 * i, 0] = x1
        out[2 * i, 1] = y1
        out[2 * i, 2] = z1
        out[2 * i + 1, 0] = x2
        out[2 * i + 1, 1] = y2
        out[2 * i + 1, 2] = z2


@njit(cache=True, parallel=True)
def _place_children_axis(pos, axes, dr, box, periodic, dist, out):
    """Fill out (2N,3): children at pos +/- axes * (dist * dr) (axes unit),
    wrapped into the box. The separation scale matches _place_children; only
    the orientation source differs."""
    n = pos.shape[0]
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    for i in prange(n):
        d = dr[i]
        if d <= 0.0:
            ox, oy, oz = 1.0, 0.0, 0.0
            disp = 1e-6
        else:
            ox, oy, oz = axes[i, 0], axes[i, 1], axes[i, 2]
            disp = d * dist
        x1 = pos[i, 0] + ox * disp
        y1 = pos[i, 1] + oy * disp
        z1 = pos[i, 2] + oz * disp
        x2 = pos[i, 0] - ox * disp
        y2 = pos[i, 1] - oy * disp
        z2 = pos[i, 2] - oz * disp
        if periodic:
            if x1 > hbx:
                x1 -= bx
            elif x1 < -hbx:
                x1 += bx
            if y1 > hby:
                y1 -= by
            elif y1 < -hby:
                y1 += by
            if z1 > hbz:
                z1 -= bz
            elif z1 < -hbz:
                z1 += bz
            if x2 > hbx:
                x2 -= bx
            elif x2 < -hbx:
                x2 += bx
            if y2 > hby:
                y2 -= by
            elif y2 < -hby:
                y2 += by
            if z2 > hbz:
                z2 -= bz
            elif z2 < -hbz:
                z2 += bz
        out[2 * i, 0] = x1
        out[2 * i, 1] = y1
        out[2 * i, 2] = z1
        out[2 * i + 1, 0] = x2
        out[2 * i + 1, 1] = y2
        out[2 * i + 1, 2] = z2


@njit(cache=True, inline='always')
def _pair_min_d2(i, ex, ey, ez, si, pos, edir, sarr, dist, idx, kc, box,
                 periodic):
    """Min squared distance from parent i's two children (direction e, offset
    si) to the children of every OTHER parent in i's candidate row, with the
    neighbours' children derived from their current edir/sarr. Min-image.
    Returns 1e30 when the row holds no other parent."""
    bx, by, bz = box[0], box[1], box[2]
    hbx, hby, hbz = 0.5 * bx, 0.5 * by, 0.5 * bz
    mind = 1e30
    for c in range(kc):
        j = idx[i, c]
        if j < 0 or j == i:
            continue
        sj = sarr[j] * dist
        if sj < 0.0:
            sj = 0.0
        for jc in range(2):
            fj = sj if jc == 0 else -sj
            ox = pos[j, 0] + fj * edir[j, 0]
            oy = pos[j, 1] + fj * edir[j, 1]
            oz = pos[j, 2] + fj * edir[j, 2]
            for sgn in range(2):
                f = si if sgn == 0 else -si
                dxx = pos[i, 0] + f * ex - ox
                dyy = pos[i, 1] + f * ey - oy
                dzz = pos[i, 2] + f * ez - oz
                if periodic:
                    if dxx > hbx:
                        dxx -= bx
                    elif dxx < -hbx:
                        dxx += bx
                    if dyy > hby:
                        dyy -= by
                    elif dyy < -hby:
                        dyy += by
                    if dzz > hbz:
                        dzz -= bz
                    elif dzz < -hbz:
                        dzz += bz
                d2 = dxx * dxx + dyy * dyy + dzz * dzz
                if d2 < mind:
                    mind = d2
    return mind


@njit(cache=True, inline='always')
def _best_azimuth(i, pos, vnn, edir, sarr, dist, idx, kc, box, periodic, naz):
    """Scan naz azimuths in the plane perpendicular to parent i's nearest
    neighbour and return (ex,ey,ez, improved): the direction maximizing the
    min distance to the neighbours' CURRENT children (edir), keeping the
    current direction unless a strictly better azimuth exists."""
    vx, vy, vz = vnn[i, 0], vnn[i, 1], vnn[i, 2]
    vn = np.sqrt(vx * vx + vy * vy + vz * vz)
    si = sarr[i] * dist
    cex, cey, cez = edir[i, 0], edir[i, 1], edir[i, 2]
    if vn <= 0.0 or si <= 0.0:
        return cex, cey, cez, False
    ux, uy, uz = vx / vn, vy / vn, vz / vn
    # in-plane orthonormal basis (b1, b2), same construction as coord
    ax, ay, az = abs(ux), abs(uy), abs(uz)
    if ax < ay and ax < az:
        tx, ty, tz = 1.0, 0.0, 0.0
    elif ay < az:
        tx, ty, tz = 0.0, 1.0, 0.0
    else:
        tx, ty, tz = 0.0, 0.0, 1.0
    b1x = uy * tz - uz * ty
    b1y = uz * tx - ux * tz
    b1z = ux * ty - uy * tx
    nb1 = np.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
    b1x, b1y, b1z = b1x / nb1, b1y / nb1, b1z / nb1
    b2x = uy * b1z - uz * b1y
    b2y = uz * b1x - ux * b1z
    b2z = ux * b1y - uy * b1x
    # current direction is the incumbent: a scanned azimuth must strictly
    # beat it (small relative margin so ties/rounding never cause churn)
    bestd = _pair_min_d2(i, cex, cey, cez, si, pos, edir, sarr, dist, idx,
                         kc, box, periodic)
    if bestd >= 1e30:
        return cex, cey, cez, False   # isolated: nothing to optimize against
    bestd *= 1.0 + 1e-9
    bex, bey, bez = cex, cey, cez
    improved = False
    for a in range(naz):
        th = 2.0 * np.pi * a / naz
        ct, st = np.cos(th), np.sin(th)
        ex = ct * b1x + st * b2x
        ey = ct * b1y + st * b2y
        ez = ct * b1z + st * b2z
        d2 = _pair_min_d2(i, ex, ey, ez, si, pos, edir, sarr, dist, idx,
                          kc, box, periodic)
        if d2 > bestd:
            bestd = d2
            bex, bey, bez = ex, ey, ez
            improved = True
    return bex, bey, bez, improved


@njit(cache=True, parallel=True)
def _azimuth_sweeps_jac(pos, vnn, edir, sarr, idx, kc, box, periodic, dist,
                        naz, nsweeps, pkeep):
    """Jacobi azimuth relaxation: every parent re-optimizes against the
    PREVIOUS sweep's configuration, so the sweep is embarrassingly parallel
    (prange) and still deterministic. pkeep=1 is plain Jacobi - neighbouring
    parents react to each other simultaneously and OSCILLATE (verified: move
    counts stay ~N, quality stalls at the initializer's). pkeep<1 is the
    randomized fix: per sweep only a deterministic pseudo-random pkeep
    fraction of parents may move (per-(i,sweep) hash lottery), which breaks
    the simultaneous-swap deadlock like red-black coloring but without
    needing a graph coloring."""
    n = pos.shape[0]
    moves = np.zeros(nsweeps, np.int64)
    enew = np.empty_like(edir)
    # ACTIVE-SET: parent i's objective depends only on the children of the
    # parents in idx[i], so once i's scan fails to improve it cannot gain a
    # move until a candidate neighbour moves. Scans skipped this way are
    # provably no-ops -> the trajectory (and result) is IDENTICAL to the
    # full rescan, but converged regions drop out of the per-sweep cost.
    # Lottery-blocked parents were never scanned, so they stay active.
    active = np.ones(n, np.bool_)
    moved = np.zeros(n, np.bool_)
    for sw in range(nsweeps):
        nm = 0
        for i in prange(n):
            moved[i] = False
            enew[i, 0] = edir[i, 0]
            enew[i, 1] = edir[i, 1]
            enew[i, 2] = edir[i, 2]
            if not active[i]:
                continue
            if pkeep < 1.0:
                hsh = np.sin(i * 12.9898 + sw * 78.233
                             + _ROT_SEED) * 43758.5453
                if hsh - np.floor(hsh) >= pkeep:
                    continue
            ex, ey, ez, imp = _best_azimuth(i, pos, vnn, edir, sarr, dist,
                                            idx, kc, box, periodic, naz)
            if imp:
                enew[i, 0] = ex
                enew[i, 1] = ey
                enew[i, 2] = ez
                moved[i] = True
                nm += 1
            # scanned parents are at their local optimum either way: only a
            # neighbour's move can reactivate them (propagation below)
            active[i] = False
        for i in prange(n):
            if not active[i]:
                for c in range(kc):
                    j = idx[i, c]
                    if j >= 0 and j != i and moved[j]:
                        active[i] = True
                        break
        edir[:, :] = enew
        moves[sw] = nm
        if nm == 0:
            break
    return moves


# diagnostics stash: per-sweep move counts of the last jr split pass
# (bench-only introspection; not part of the API)
_LAST_SWEEP_MOVES = None


def _interleave(arr, factor=2):
    """Repeat each row `factor` times contiguously: rows 0,0,1,1,... for the
    children of each parent (matches _place_children's 2i/2i+1 layout)."""
    return np.repeat(arr, factor, axis=0)


def _split_once(pos, mass, aux, box, nsmooth, margin, dist, halve,
                orient='random', scale='nn'):
    """One binary split pass (N -> 2N)."""
    n = len(pos)
    periodic = box is not None
    boxarr = np.ones(3) if box is None else np.asarray(box, dtype=np.float64)
    nl = NeighborList(boxarr if periodic else None, nsmooth, margin=margin,
                      build_transpose=False)
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    idx, h, _, _ = nl.update(pos)
    # relaxed-glass minimum spacing / h (SPH_methods nearest_distanceoverhmin)
    dist_floor = 1.0 / (0.5 * nsmooth) ** (1.0 / 3.0)
    vnn, dr, vort = _nn_dirs(pos, idx, h, boxarr, periodic, nsmooth,
                             dist_floor)

    if scale == 'vol':
        rho = np.empty(n)
        density_loop(pos, np.ascontiguousarray(mass, dtype=np.float64), h,
                     idx, boxarr, periodic, rho, W0 * wzero_wc2(nsmooth))
        vol = mass / rho
    if scale == 'nn':
        # offset length = distance to the (floor-cleared) nearest neighbour:
        # fully local, but the MIN over the candidates is a noisy statistic
        sarr = dr
    elif scale == 'vol':
        # offset length = (m/rho)^(1/3), the local mean spacing: same average
        # as 'nn' on a uniform glass but far smoother per particle
        sarr = np.cbrt(vol)
    elif scale == 'h':
        # offset length from the smoothing length: the ball of radius 2h
        # holds nsmooth particles (counting self), so the implied local mean
        # spacing is 2h * (4pi/(3 nsmooth))^(1/3). A fixed-count radius is a
        # smooth order statistic but averages the density over the whole
        # ball (~2.5 dx), where rho kernel-weights the same neighbours
        # toward the centre - h is the LESS local of the two.
        sarr = 2.0 * h * (4.0 * np.pi / (3.0 * nsmooth)) ** (1.0 / 3.0)
    else:
        raise ValueError(f"unknown scale {scale!r}")

    pos2 = np.empty((2 * n, 3))
    if orient == 'random':
        _place_children(pos, vnn, sarr, vort, boxarr, periodic, dist, np.pi,
                        pos2)
    elif orient == 'k1jitter':
        # the ORIGINAL nn_ort azimuth (from the secondary neighbour), kept up
        # to a small decorrelating wiggle (+/- ~20 deg) instead of the full
        # re-randomization
        _place_children(pos, vnn, sarr, vort, boxarr, periodic, dist, 0.35,
                        pos2)
    elif orient[:2] == 'jr' and orient[2:].isdigit():
        # iterative azimuth relaxation: start from the DETERMINISTIC k1jitter
        # placement (azimuth from the secondary neighbour, +/- ~20 deg
        # decorrelating wiggle - parallel, no ordering), then N randomized-
        # Jacobi sweeps where every parent re-optimizes its azimuth against
        # the complete child configuration. Positions stay parent +/-
        # dist*sarr in the perpendicular plane, so this moves no centroid
        # and adds no density noise - it only removes cousin crowding.
        nsweeps = int(orient[2:])
        _place_children(pos, vnn, sarr, vort, boxarr, periodic, dist, 0.35,
                        pos2)
        # recover the unit offset directions from the initial placement
        # (children are wrapped: undo via min-image)
        dvec = pos2[0::2] - pos
        if periodic:
            dvec -= boxarr * np.round(dvec / boxarr)
        dlen = np.sqrt((dvec * dvec).sum(axis=1))
        good = dlen > 0.0
        edir = np.zeros((n, 3))
        edir[good] = dvec[good] / dlen[good, None]
        edir[~good, 0] = 1.0
        edir = np.ascontiguousarray(edir)
        if nsweeps > 0:
            moves = _azimuth_sweeps_jac(pos, vnn, edir, sarr, idx,
                                        idx.shape[1], boxarr, periodic,
                                        dist, 64, nsweeps, 0.5)
            global _LAST_SWEEP_MOVES
            _LAST_SWEEP_MOVES = moves
        _place_children_axis(pos, edir, sarr, boxarr, periodic, dist, pos2)
    else:
        raise ValueError(f"unknown orient {orient!r}")
    mass2 = _interleave(mass) * 0.5
    aux2 = {}
    for name, a in aux.items():
        a = np.asarray(a)
        rep = _interleave(a)
        if name in halve:
            rep = rep * 0.5
        aux2[name] = rep
    return pos2, mass2, aux2


def split(pos, mass, aux=None, box=None, factor=8, nsmooth=64, margin=8,
          dist=0.4, halve_fields=None, orient='random', scale='nn'):
    """Split every particle, multiplying the count by `factor` (a power of 2).

    orient selects the dumbbell-axis algorithm (all perpendicular to the
    nearest floor-cleared neighbour; 2026-07-06 benchmark in
    _bench_split.py, history in the split-quality notes):
      'random'  - deterministic pseudo-random azimuth about the perpendicular
                  plane (the original SPH_methods nn_ort port; baseline).
      'k1jitter'- the original nn_ort azimuth (set by the secondary
                  neighbour), decorrelated only by a small +/- ~20 deg wiggle
                  instead of the full re-randomization.
      'jrN'     - iterative azimuth relaxation (up to N sweeps, early-stops):
                  start from the k1jitter placement, then randomized-Jacobi
                  sweeps (parallel, deterministic; 50% per-(parent,sweep)
                  hash lottery breaks the simultaneous-swap deadlock; an
                  active-set skips provably converged parents) in which every
                  parent re-optimizes its azimuth (64-point scan) to maximize
                  its children's min distance to the neighbours' children.
                  vs k1jitter: ~17% less density noise and min pair spacing
                  0.48 -> 0.83 dx at dist=0.34 (the spacing optimum). Sweep
                  count to full convergence grows ~log N (~24 at 4k, ~68 at
                  33k parents): N=96 is a safe cap, late sweeps are ~free.

    scale selects the per-particle offset length L (children at +/- dist*L):
      'nn'  - distance to the floor-cleared nearest neighbour (original; fully
              local but the min over candidates is a noisy statistic).
      'vol' - (m/rho)^(1/3), the local mean spacing (same average on a uniform
              glass, far smoother per particle).
      'h'   - 2h*(4pi/(3 nsmooth))^(1/3), the spacing implied by the
              fixed-count smoothing radius (2h holds nsmooth particles):
              smooth like 'vol' but averages over the whole ball, less local.

    pos : (N,3). mass : (N,). aux : dict {name: (N,) or (N,3)} of extra fields
    (vel, rho, u, eps, metals, B, spin, momenti, tform, phi, ...); names in
    halve_fields are halved (extensive), the rest inherited. halve_fields :
    override the default HALVE_FIELDS set (e.g. add B-field aux keys). box :
    (3,) periodic lengths or None (open). Returns (pos2, mass2, aux2) with
    N*factor rows. Total mass is conserved (each child carries mass/2^p). The
    SPH smoothing length h is NOT stored/derived here - it is a runtime quantity
    (density / nSmooth k-nearest); the caller scales the softening eps."""
    if aux is None:
        aux = {}
    halve = HALVE_FIELDS if halve_fields is None else frozenset(halve_fields)
    factor = int(factor)
    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError(f"factor must be a power of 2, got {factor}")
    passes = factor.bit_length() - 1
    pos = np.array(pos, dtype=np.float64)
    mass = np.array(mass, dtype=np.float64)
    aux = {k: np.array(v) for k, v in aux.items()}
    for _ in range(passes):
        pos, mass, aux = _split_once(pos, mass, aux, box, nsmooth, margin,
                                     dist, halve, orient=orient, scale=scale)
    return pos, mass, aux


@njit(cache=True)
def _merge_pairs(pos, mass, idx, h, box, periodic):
    """Mutual-nearest pairing within h_i+h_j. Returns a (N,) partner array:
    partner[i] = j (>i) if (i,j) merge, -1 otherwise; each particle in at most
    one pair. Serial (greedy claim) for determinism."""
    n = pos.shape[0]
    k = idx.shape[1]
    bx, by, bz = box[0], box[1], box[2]
    nn = np.full(n, -1, np.int64)
    nnd = np.full(n, 1e30)
    for i in range(n):
        for c in range(k):
            j = idx[i, c]
            if j < 0:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d > 0.0 and d < nnd[i]:
                nnd[i] = d
                nn[i] = j
    partner = np.full(n, -1, np.int64)
    claimed = np.zeros(n, np.bool_)
    for i in range(n):
        if claimed[i]:
            continue
        j = nn[i]
        if j < 0 or claimed[j]:
            continue
        if nn[j] == i and nnd[i] <= h[i] + h[j]:
            a, b = (i, j) if i < j else (j, i)
            partner[a] = b
            claimed[i] = True
            claimed[j] = True
    return partner


@njit(cache=True, parallel=True)
def _nearest(pos, idx, box, periodic):
    """Index of each particle's single nearest neighbour (excluding self),
    scanning its small candidate row. -1 if the row holds no other particle."""
    n, k = idx.shape
    bx, by, bz = box[0], box[1], box[2]
    nn = np.full(n, -1, np.int64)
    for i in prange(n):
        best = -1
        bestd = 1e30
        for c in range(k):
            j = idx[i, c]
            if j < 0 or j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            if periodic:
                if dx > 0.5 * bx:
                    dx -= bx
                elif dx < -0.5 * bx:
                    dx += bx
                if dy > 0.5 * by:
                    dy -= by
                elif dy < -0.5 * by:
                    dy += by
                if dz > 0.5 * bz:
                    dz -= bz
                elif dz < -0.5 * bz:
                    dz += bz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < bestd:
                bestd = d2
                best = j
        nn[i] = best
    return nn


def _combine(pos, mass, aux, pa, pb, keep, halve, boxarr, periodic):
    """Combine paired particles (leaders `pa` with partners `pb`) and carry the
    unpaired `keep` through. Mass summed, pos = mass-weighted centroid (min-
    image), halve-fields summed, the rest mass-weighted mean. Shared by merge()
    and coarsen()."""
    m_a, m_b = mass[pa], mass[pb]
    m_sum = m_a + m_b
    off = pos[pb] - pos[pa]
    if periodic:
        off -= boxarr * np.rint(off / boxarr)
    cen = pos[pa] + (m_b / m_sum)[:, None] * off
    if periodic:
        cen -= boxarr * np.rint(cen / boxarr)
    new_pos = np.vstack([pos[keep], cen]) if len(keep) else cen
    new_mass = np.concatenate([mass[keep], m_sum])
    new_aux = {}
    for name, a in aux.items():
        if name in halve:
            merged = a[pa] + a[pb]
        else:
            w = (m_a / m_sum) if a.ndim == 1 else (m_a / m_sum)[:, None]
            wb = (m_b / m_sum) if a.ndim == 1 else (m_b / m_sum)[:, None]
            merged = w * a[pa] + wb * a[pb]
        new_aux[name] = np.concatenate([a[keep], merged]) if len(keep) else merged
    return new_pos, new_mass, new_aux


def _match_mass_distribution(mass_in, mass_out):
    """Post-merge mass renormalisation. Mutual-nearest pairing over several
    passes leaves a spread of merge multiplicities - unpaired parents keep
    their 1x mass while re-merged particles reach 3x, 4x, ... - so the merged
    set's mass distribution no longer looks like the input's. Physically the
    coarse set should carry the SAME mass distribution as the input, just
    n_in/n_out heavier per particle (a uniform-mass input must stay uniform).

    Rank-preserving remap: the merged masses, in rank order, are replaced by
    n_out quantiles of the input mass distribution scaled by n_in/n_out
    (heavier merged particles stay the heavier ones where the input had a
    spread), then rescaled so total mass is conserved exactly."""
    n_in, n_out = len(mass_in), len(mass_out)
    if n_out >= n_in:
        return mass_out
    q = (np.arange(n_out) + 0.5) / n_out
    # sort + interp == np.quantile(mass_in, q) with method='linear', but
    # np.quantile hands ALL n_out quantile points to introselect partition
    # at once, which degenerates to minutes-of-CPU at ~1e6 particles (numpy
    # 1.x pathological case); one O(n log n) sort is milliseconds
    srt = np.sort(mass_in)
    target = np.interp(q * (n_in - 1), np.arange(n_in), srt) * (n_in / n_out)
    out = np.empty(n_out)
    out[np.argsort(mass_out, kind='stable')] = target
    out *= mass_in.sum() / out.sum()
    return out


def merge(pos, mass, aux=None, box=None, target_factor=2, nsmooth=64,
          margin=8, halve_fields=None, keep_mass_distribution=True):
    """Merge particles toward count/target_factor (power of 2) by mutual-
    nearest pairing. Inverse of split: mass summed, pos mass-weighted centroid,
    halve_fields summed, the rest mass-weighted mean (vel -> momentum). Particles
    without a mutual partner are carried through unchanged; unless
    keep_mass_distribution=False, the final masses are renormalised to the
    input mass distribution scaled by the achieved count ratio (see
    _match_mass_distribution), so a uniform-mass input stays uniform and total
    mass is conserved exactly (at the cost of exact momentum conservation,
    which then holds only statistically)."""
    if aux is None:
        aux = {}
    halve = HALVE_FIELDS if halve_fields is None else frozenset(halve_fields)
    target_factor = int(target_factor)
    if target_factor < 1 or (target_factor & (target_factor - 1)) != 0:
        raise ValueError(f"target_factor must be a power of 2, got "
                         f"{target_factor}")
    passes = target_factor.bit_length() - 1
    periodic = box is not None
    boxarr = np.ones(3) if box is None else np.asarray(box, dtype=np.float64)
    pos = np.array(pos, dtype=np.float64)
    mass = np.array(mass, dtype=np.float64)
    aux = {k: np.array(v, dtype=np.float64) for k, v in aux.items()}
    mass_in = mass.copy()
    for _ in range(passes):
        n = len(pos)
        nl = NeighborList(boxarr if periodic else None, nsmooth,
                          margin=margin, build_transpose=False)
        idx, h, _, _ = nl.update(pos)
        partner = _merge_pairs(pos, mass, idx, h, boxarr, periodic)
        pa = np.where(partner >= 0)[0]      # pair leaders (a < b)
        pb = partner[pa]                    # their partners
        involved = np.zeros(n, dtype=bool)
        involved[pa] = True
        involved[pb] = True
        keep = np.where(~involved)[0]       # unpaired: carry through
        pos, mass, aux = _combine(pos, mass, aux, pa, pb, keep, halve,
                                  boxarr, periodic)
    if keep_mass_distribution:
        mass = _match_mass_distribution(mass_in, mass)
    return pos, mass, aux


def coarsen(pos, mass, aux=None, box=None, factor=8, k=2, margin=4,
            halve_fields=None, keep_mass_distribution=True, progress=None):
    """Coarsen the particle set to EXACTLY count//factor by repeated mutual-
    nearest pairing, rebuilding the (k-nearest) neighbours each pass until the
    target is reached - the inverse-by-count of split(): ``split(coarsen(x, F),
    F)`` restores the original count (when count is divisible by F).

    This is the iterate-to-target form of SPH_methods.merge_particles (k=2,
    parallel mutual-nearest test, rebuild each round) but WITHOUT its h_i+h_j
    distance cap - the coarse set is only a relaxation seed, so merging the
    nearest pair regardless of separation is fine and converges faster. The
    final pass merges only the `need` CLOSEST mutual pairs so the target count
    is hit exactly. Mass is conserved; rho/h are placeholders, re-derived by the
    next relax(). Unless keep_mass_distribution=False, the final masses are
    renormalised to the input mass distribution scaled by the count ratio
    (see _match_mass_distribution), so a uniform-mass input stays uniform and
    total mass is conserved exactly.

    Intended for the progressive coarse seed: coarsen the full-res random pre-IC
    down (factor = the cascade's total split factor), relax, then split back to
    the original count - no separate low-res file, count tied 1:1 to the input.

    progress : optional callable(pass_no, n_before, n_after, target) invoked
    after every pairing pass - each pass rebuilds the kNN list, so on ~1e6
    particles a pass takes seconds to minutes and callers may want to report
    liveness (the CLI does)."""
    if aux is None:
        aux = {}
    halve = HALVE_FIELDS if halve_fields is None else frozenset(halve_fields)
    factor = int(factor)
    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError(f"factor must be a power of 2, got {factor}")
    periodic = box is not None
    boxarr = np.ones(3) if box is None else np.asarray(box, dtype=np.float64)
    pos = np.array(pos, dtype=np.float64)
    mass = np.array(mass, dtype=np.float64)
    aux = {kk: np.array(v, dtype=np.float64) for kk, v in aux.items()}
    mass_in = mass.copy()
    target = max(1, len(pos) // factor)
    npass = 0
    while len(pos) > target:
        n = len(pos)
        nl = NeighborList(boxarr if periodic else None, k, margin=margin,
                          build_transpose=False)
        idx, _, _, _ = nl.update(pos)
        nn = _nearest(pos, idx, boxarr, periodic)
        # mutual-nearest leaders: nn[i] valid, i < nn[i], and nn[nn[i]] == i
        ii = np.arange(n)
        ok = nn >= 0
        back = np.where(ok, nn[np.where(ok, nn, 0)], -1)   # nn[nn[i]] (guarded)
        leader = ok & (ii < nn) & (back == ii)
        pa = np.where(leader)[0]
        pb = nn[pa]
        if len(pa) == 0:
            break                            # no mutual pairs left (degenerate)
        need = n - target                    # pairs to reach target exactly
        if len(pa) > need:
            # final pass: keep only the `need` closest pairs (most redundant)
            off = pos[pb] - pos[pa]
            if periodic:
                off -= boxarr * np.rint(off / boxarr)
            d2 = (off * off).sum(1)
            sel = np.argsort(d2)[:need]
            pa, pb = pa[sel], pb[sel]
        involved = np.zeros(n, dtype=bool)
        involved[pa] = True
        involved[pb] = True
        keep = np.where(~involved)[0]
        pos, mass, aux = _combine(pos, mass, aux, pa, pb, keep, halve,
                                  boxarr, periodic)
        npass += 1
        if progress is not None:
            progress(npass, n, len(pos), target)
    if keep_mass_distribution:
        mass = _match_mass_distribution(mass_in, mass)
    return pos, mass, aux
