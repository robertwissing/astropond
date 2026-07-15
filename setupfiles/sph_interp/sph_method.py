"""Regular SPH interpolation: deposit particle fields onto a target grid.

For each particle the field value is smeared onto nearby cells weighted by the
cubic-spline kernel evaluated at the cell centre, times the SPH volume element
m/rho, then normalized by the summed kernel weight (Shepard normalization). This
is the cheap/approximate method (cf. the Petkova exact integral).

Currently implements the UniformGrid target.

Notes vs the old `grid_interpolate.interpolate3D`:
  * operates on EXPLICIT arrays (pos/mass/rho/h/values), no hardcoded columns;
  * interpolates only the requested value channels (not all ~21 columns);
  * the periodic index range is kept raw and each index wrapped inside the loop
    (fixes the `i_max % npix` range-collapse bug);
  * parallel deposit uses per-thread buffers (no scatter-add data race);
  * the numba kernel is module-level (compiled/cached once, not per call).
"""

import numpy as np
import numba
from numba import njit, prange

from .kernels import wkernel, col_profile, COL_TABLE, RADKERNEL, RADKERNEL2
from .targets import (UniformGrid, Projection2D, AMRGrid, VoronoiGrid,
                      generator_buckets)


@njit(parallel=True, cache=True)
def _deposit_uniform(x, y, z, h, weight, vals,
                     nx, ny, nz, lox, loy, loz, pwx, pwy, pwz,
                     perx, pery, perz, hmin, out_local, norm_local):
    """Fill per-thread accumulation buffers (out_local, norm_local).

    out_local  : (nthreads, nx, ny, nz, K)
    norm_local : (nthreads, nx, ny, nz)
    """
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        wi = weight[i]
        hi = h[i]
        if wi <= 0.0 or hi <= 0.0:
            continue
        if hi < hmin:
            hi = hmin
        t = numba.get_thread_id()
        inv_hi2 = 1.0 / (hi * hi)
        kr = RADKERNEL * hi
        wnorm = wi / (np.pi * hi * hi * hi)   # weight * cnorm3D(hi)
        xi = x[i]; yi = y[i]; zi = z[i]

        ixmin = int(np.floor((xi - kr - lox) / pwx))
        ixmax = int(np.floor((xi + kr - lox) / pwx)) + 1
        jymin = int(np.floor((yi - kr - loy) / pwy))
        jymax = int(np.floor((yi + kr - loy) / pwy)) + 1
        kzmin = int(np.floor((zi - kr - loz) / pwz))
        kzmax = int(np.floor((zi + kr - loz) / pwz)) + 1
        if not perx:
            ixmin = max(ixmin, 0); ixmax = min(ixmax, nx)
        if not pery:
            jymin = max(jymin, 0); jymax = min(jymax, ny)
        if not perz:
            kzmin = max(kzmin, 0); kzmax = min(kzmax, nz)
        if ixmin >= ixmax or jymin >= jymax or kzmin >= kzmax:
            continue

        for ix in range(ixmin, ixmax):
            iw = ix % nx if perx else ix
            cx = lox + (ix + 0.5) * pwx
            dx = cx - xi
            dx2 = dx * dx * inv_hi2
            for jy in range(jymin, jymax):
                jw = jy % ny if pery else jy
                cy = loy + (jy + 0.5) * pwy
                dy = cy - yi
                dyz = dx2 + dy * dy * inv_hi2
                for kz in range(kzmin, kzmax):
                    cz = loz + (kz + 0.5) * pwz
                    dz = cz - zi
                    q2 = dyz + dz * dz * inv_hi2
                    if q2 < RADKERNEL2:
                        w = wnorm * wkernel(q2)
                        norm_local[t, iw, jw, kz % nz if perz else kz] += w
                        kw = kz % nz if perz else kz
                        for p in range(K):
                            out_local[t, iw, jw, kw, p] += w * vals[i, p]


def interpolate_uniform_sph(particles, grid, values=None, hmin=None,
                            normalise=True):
    """SPH-interpolate `particles` onto a `UniformGrid`.

    particles : Particles
    grid      : UniformGrid
    values    : list of field names (default: all in particles.values)
    hmin      : minimum smoothing length (default: 0.5 * max cell width), which
                guards against particles narrower than a cell falling between
                grid points.
    normalise : Shepard-normalize by the summed kernel weight (recommended).

    Returns (data: dict[name -> (nx,ny,nz) array], norm: (nx,ny,nz) array).
    """
    if not isinstance(grid, UniformGrid):
        raise TypeError("interpolate_uniform_sph requires a UniformGrid target")

    vals, names = particles.value_array(values)
    nx, ny, nz = grid.npx
    lo = grid.lo
    pw = grid.pixwidth
    perx, pery, perz = particles.periodic
    if hmin is None:
        hmin = 0.5 * float(np.max(pw))

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    out_local = np.zeros((nthreads, nx, ny, nz, K), dtype=np.float64)
    norm_local = np.zeros((nthreads, nx, ny, nz), dtype=np.float64)

    pos = particles.pos
    _deposit_uniform(np.ascontiguousarray(pos[:, 0]),
                     np.ascontiguousarray(pos[:, 1]),
                     np.ascontiguousarray(pos[:, 2]),
                     particles.h, particles.mass / particles.rho, vals,
                     nx, ny, nz, lo[0], lo[1], lo[2], pw[0], pw[1], pw[2],
                     perx, pery, perz, float(hmin), out_local, norm_local)

    out = out_local.sum(axis=0)
    norm = norm_local.sum(axis=0)
    if normalise:
        mask = norm > 0.0
        for p in range(K):
            out[..., p][mask] /= norm[mask]

    data = {name: out[..., p] for p, name in enumerate(names)}
    return data, norm


# --------------------------------------------------------------------------- #
#  AMR target: deposit onto octree leaf cells (per-leaf, not a dense array).
#  For each particle, loop the BASE cells its kernel overlaps (O(1) index math
#  on the base grid), then that base cell's few leaves; the kernel is evaluated
#  at each leaf centre. Periodic axes use a ghost shift so absolute leaf coords
#  see the nearest particle image.
# --------------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _deposit_amr(x, y, z, h, weight, vals, cell_lo, cell_size,
                 base_start, base_count, blo0, blo1, blo2, bpw0, bpw1, bpw2,
                 nbx, nby, nbz, perx, pery, perz, boxx, boxy, boxz,
                 out_local, norm_local):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        wi = weight[i]
        hi0 = h[i]
        if wi <= 0.0 or hi0 <= 0.0:
            continue
        t = numba.get_thread_id()
        xi = x[i]; yi = y[i]; zi = z[i]
        kr = RADKERNEL * hi0
        ibx0 = int(np.floor((xi - kr - blo0) / bpw0))
        ibx1 = int(np.floor((xi + kr - blo0) / bpw0)) + 1
        iby0 = int(np.floor((yi - kr - blo1) / bpw1))
        iby1 = int(np.floor((yi + kr - blo1) / bpw1)) + 1
        ibz0 = int(np.floor((zi - kr - blo2) / bpw2))
        ibz1 = int(np.floor((zi + kr - blo2) / bpw2)) + 1
        if not perx:
            if ibx0 < 0: ibx0 = 0
            if ibx1 > nbx: ibx1 = nbx
        if not pery:
            if iby0 < 0: iby0 = 0
            if iby1 > nby: iby1 = nby
        if not perz:
            if ibz0 < 0: ibz0 = 0
            if ibz1 > nbz: ibz1 = nbz
        for ibx in range(ibx0, ibx1):
            if perx:
                kx = ibx // nbx; iwx = ibx - kx * nbx; sx = kx * boxx
            else:
                iwx = ibx; sx = 0.0
            for iby in range(iby0, iby1):
                if pery:
                    ky = iby // nby; iwy = iby - ky * nby; sy = ky * boxy
                else:
                    iwy = iby; sy = 0.0
                for ibz in range(ibz0, ibz1):
                    if perz:
                        kz = ibz // nbz; iwz = ibz - kz * nbz; sz = kz * boxz
                    else:
                        iwz = ibz; sz = 0.0
                    bflat = (iwx * nby + iwy) * nbz + iwz
                    s0 = base_start[bflat]
                    n0 = base_count[bflat]
                    for li in range(s0, s0 + n0):
                        szx = cell_size[li, 0]
                        szy = cell_size[li, 1]
                        szz = cell_size[li, 2]
                        cx = cell_lo[li, 0] + 0.5 * szx + sx
                        cy = cell_lo[li, 1] + 0.5 * szy + sy
                        cz = cell_lo[li, 2] + 0.5 * szz + sz
                        # cell-aware minimum h: ensure the kernel reaches a leaf
                        # at least half a cell wide (else a sub-cell particle
                        # would miss the centre and lose its mass).
                        he = hi0
                        half = 0.5 * szx
                        if 0.5 * szy > half: half = 0.5 * szy
                        if 0.5 * szz > half: half = 0.5 * szz
                        if he < half: he = half
                        inv = 1.0 / (he * he)
                        dx = cx - xi; dy = cy - yi; dz = cz - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            w = wi / (np.pi * he * he * he) * wkernel(q2)
                            norm_local[t, li] += w
                            for p in range(K):
                                out_local[t, li, p] += w * vals[i, p]


def interpolate_amr_sph(particles, grid, values=None, normalise=True):
    """SPH-interpolate `particles` onto an `AMRGrid` (per-leaf-cell values).

    Returns (data: dict[name -> (Ncell,)], norm: (Ncell,)). Each field is the
    Shepard-normalized kernel average at the leaf centre, exactly as the
    UniformGrid SPH path but on adaptive cells.
    """
    if not isinstance(grid, AMRGrid):
        raise TypeError("interpolate_amr_sph requires an AMRGrid target")

    vals, names = particles.value_array(values)
    Ncell = grid.ncell
    nbx, nby, nbz = grid.base_npx
    blo = grid.lo
    bpw = grid.base_pw
    box = grid.box
    perx, pery, perz = grid.periodic

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    out_local = np.zeros((nthreads, Ncell, K), dtype=np.float64)
    norm_local = np.zeros((nthreads, Ncell), dtype=np.float64)

    pos = particles.pos
    _deposit_amr(np.ascontiguousarray(pos[:, 0]),
                 np.ascontiguousarray(pos[:, 1]),
                 np.ascontiguousarray(pos[:, 2]),
                 particles.h, particles.mass / particles.rho, vals,
                 grid.cell_lo, grid.cell_size, grid.base_start, grid.base_count,
                 blo[0], blo[1], blo[2], bpw[0], bpw[1], bpw[2],
                 nbx, nby, nbz, perx, pery, perz, box[0], box[1], box[2],
                 out_local, norm_local)

    out = out_local.sum(axis=0)
    norm = norm_local.sum(axis=0)
    if normalise:
        mask = norm > 0.0
        for p in range(K):
            out[mask, p] /= norm[mask]

    data = {name: out[:, p] for p, name in enumerate(names)}
    return data, norm


# --------------------------------------------------------------------------- #
#  CONSERVATIVE SPH SCATTER (AREPO ADDBACKGROUNDGRID style).
#
#  The plain SPH method above is a GATHER (evaluate the kernel density estimate
#  AT each cell centre) and is NOT mass-conserving. This is the conservative
#  SCATTER AREPO uses to turn SPH ICs into mesh ICs (src/add_backgroundgrid/):
#  each particle distributes its mass to nearby cells with a per-particle
#  normalization, so the shares sum to exactly 1 and total mass/momentum/energy
#  are conserved to machine precision.
#
#    Weight_i = sum_j  w(r_ij,h_i) * V_cell_j           (per-particle normaliser)
#    share_ij = w(r_ij,h_i) * V_cell_j / Weight_i        (sum_j share_ij = 1)
#    Mass_j  += m_i * share_ij ;  (m_i A_i) accumulated likewise
#  -> rho_j = Mass_j / V_cell_j ;  intensive A_j = sum_i(m_i A_i share)/Mass_j
#     (mass-weighted average).  Both passes use IDENTICAL weights over the SAME
#  (particle, cell) pairs, so conservation is exact regardless of the kernel
#  reach -- a particle's whole mass always lands on the cells it is normalised
#  over (never lost off the searched range). One prange over particles does both
#  passes (Weight_i depends only on particle i's own cells), so no global
#  per-particle buffer is needed.
# --------------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _deposit_amr_scatter(x, y, z, h, mass, vals, cell_lo, cell_size,
                         base_start, base_count, blo0, blo1, blo2,
                         bpw0, bpw1, bpw2, nbx, nby, nbz,
                         perx, pery, perz, boxx, boxy, boxz,
                         mass_local, massA_local):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        hi0 = h[i]
        mi = mass[i]
        if hi0 <= 0.0 or mi <= 0.0:
            continue
        t = numba.get_thread_id()
        xi = x[i]; yi = y[i]; zi = z[i]
        kr = RADKERNEL * hi0
        ibx0 = int(np.floor((xi - kr - blo0) / bpw0))
        ibx1 = int(np.floor((xi + kr - blo0) / bpw0)) + 1
        iby0 = int(np.floor((yi - kr - blo1) / bpw1))
        iby1 = int(np.floor((yi + kr - blo1) / bpw1)) + 1
        ibz0 = int(np.floor((zi - kr - blo2) / bpw2))
        ibz1 = int(np.floor((zi + kr - blo2) / bpw2)) + 1
        if not perx:
            if ibx0 < 0: ibx0 = 0
            if ibx1 > nbx: ibx1 = nbx
        if not pery:
            if iby0 < 0: iby0 = 0
            if iby1 > nby: iby1 = nby
        if not perz:
            if ibz0 < 0: ibz0 = 0
            if ibz1 > nbz: ibz1 = nbz

        # PASS 1: per-particle normaliser Weight_i = sum_j w_ij * V_cell_j, using a
        # cell-aware kernel width (he >= half the cell) so a sub-cell particle still
        # reaches the cell centre. The SAME he/w is recomputed identically in pass 2.
        Wi = 0.0
        for ibx in range(ibx0, ibx1):
            if perx:
                kx = ibx // nbx; iwx = ibx - kx * nbx; sx = kx * boxx
            else:
                iwx = ibx; sx = 0.0
            for iby in range(iby0, iby1):
                if pery:
                    ky = iby // nby; iwy = iby - ky * nby; sy = ky * boxy
                else:
                    iwy = iby; sy = 0.0
                for ibz in range(ibz0, ibz1):
                    if perz:
                        kz = ibz // nbz; iwz = ibz - kz * nbz; sz = kz * boxz
                    else:
                        iwz = ibz; sz = 0.0
                    bflat = (iwx * nby + iwy) * nbz + iwz
                    s0 = base_start[bflat]; n0 = base_count[bflat]
                    for li in range(s0, s0 + n0):
                        szx = cell_size[li, 0]; szy = cell_size[li, 1]
                        szz = cell_size[li, 2]
                        cx = cell_lo[li, 0] + 0.5 * szx + sx
                        cy = cell_lo[li, 1] + 0.5 * szy + sy
                        cz = cell_lo[li, 2] + 0.5 * szz + sz
                        he = hi0
                        half = 0.5 * szx
                        if 0.5 * szy > half: half = 0.5 * szy
                        if 0.5 * szz > half: half = 0.5 * szz
                        if he < half: he = half
                        inv = 1.0 / (he * he)
                        dx = cx - xi; dy = cy - yi; dz = cz - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            Vc = szx * szy * szz
                            Wi += wkernel(q2) / (he * he * he) * Vc
        if Wi <= 0.0:
            continue

        # PASS 2: distribute m_i (and m_i*A_i) with share_ij = w_ij V_j / Weight_i.
        for ibx in range(ibx0, ibx1):
            if perx:
                kx = ibx // nbx; iwx = ibx - kx * nbx; sx = kx * boxx
            else:
                iwx = ibx; sx = 0.0
            for iby in range(iby0, iby1):
                if pery:
                    ky = iby // nby; iwy = iby - ky * nby; sy = ky * boxy
                else:
                    iwy = iby; sy = 0.0
                for ibz in range(ibz0, ibz1):
                    if perz:
                        kz = ibz // nbz; iwz = ibz - kz * nbz; sz = kz * boxz
                    else:
                        iwz = ibz; sz = 0.0
                    bflat = (iwx * nby + iwy) * nbz + iwz
                    s0 = base_start[bflat]; n0 = base_count[bflat]
                    for li in range(s0, s0 + n0):
                        szx = cell_size[li, 0]; szy = cell_size[li, 1]
                        szz = cell_size[li, 2]
                        cx = cell_lo[li, 0] + 0.5 * szx + sx
                        cy = cell_lo[li, 1] + 0.5 * szy + sy
                        cz = cell_lo[li, 2] + 0.5 * szz + sz
                        he = hi0
                        half = 0.5 * szx
                        if 0.5 * szy > half: half = 0.5 * szy
                        if 0.5 * szz > half: half = 0.5 * szz
                        if he < half: he = half
                        inv = 1.0 / (he * he)
                        dx = cx - xi; dy = cy - yi; dz = cz - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            Vc = szx * szy * szz
                            mw = mi * (wkernel(q2) / (he * he * he) * Vc / Wi)
                            mass_local[t, li] += mw
                            for p in range(K):
                                massA_local[t, li, p] += mw * vals[i, p]


def interpolate_amr_scatter(particles, grid, values=None):
    """Conservative SPH scatter of `particles` onto an `AMRGrid` (AREPO style).

    Returns (data: dict[name -> (Ncell,)], mass: (Ncell,)). The density field
    (`rho`/`density`) is the CONSERVED cell mass / cell volume; every other field
    is the mass-weighted cell average. Total deposited mass = sum(particle mass)
    to machine precision (cf. the non-conserving `interpolate_amr_sph` gather)."""
    if not isinstance(grid, AMRGrid):
        raise TypeError("interpolate_amr_scatter requires an AMRGrid target")

    vals, names = particles.value_array(values)
    Ncell = grid.ncell
    nbx, nby, nbz = grid.base_npx
    blo = grid.lo
    bpw = grid.base_pw
    box = grid.box
    perx, pery, perz = grid.periodic

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    mass_local = np.zeros((nthreads, Ncell), dtype=np.float64)
    massA_local = np.zeros((nthreads, Ncell, K), dtype=np.float64)

    pos = particles.pos
    _deposit_amr_scatter(np.ascontiguousarray(pos[:, 0]),
                         np.ascontiguousarray(pos[:, 1]),
                         np.ascontiguousarray(pos[:, 2]),
                         particles.h, particles.mass, vals,
                         grid.cell_lo, grid.cell_size,
                         grid.base_start, grid.base_count,
                         blo[0], blo[1], blo[2], bpw[0], bpw[1], bpw[2],
                         nbx, nby, nbz, perx, pery, perz,
                         box[0], box[1], box[2], mass_local, massA_local)

    M = mass_local.sum(axis=0)
    MA = massA_local.sum(axis=0)
    V = grid.volumes()
    return _scatter_to_fields(names, M, MA, V), M


def _scatter_to_fields(names, M, MA, V):
    """Build the per-cell field dict from accumulated mass M and m*A sums MA.

    Density (`rho`/`density`) = M / V_cell (the conserved quantity); every other
    field = mass-weighted average MA / M (0 where the cell got no mass)."""
    nz = M > 0.0
    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = M / V
        else:
            a = np.zeros_like(M)
            a[nz] = MA[nz, p] / M[nz]
            data[name] = a
    return data


@njit(parallel=True, cache=True)
def _deposit_uniform_scatter(x, y, z, h, mass, vals, nx, ny, nz,
                             lox, loy, loz, pwx, pwy, pwz,
                             perx, pery, perz, hmin, mass_local, massA_local):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        hi = h[i]
        mi = mass[i]
        if hi <= 0.0 or mi <= 0.0:
            continue
        if hi < hmin:
            hi = hmin
        t = numba.get_thread_id()
        inv = 1.0 / (hi * hi)
        kr = RADKERNEL * hi
        xi = x[i]; yi = y[i]; zi = z[i]
        ixmin = int(np.floor((xi - kr - lox) / pwx))
        ixmax = int(np.floor((xi + kr - lox) / pwx)) + 1
        jymin = int(np.floor((yi - kr - loy) / pwy))
        jymax = int(np.floor((yi + kr - loy) / pwy)) + 1
        kzmin = int(np.floor((zi - kr - loz) / pwz))
        kzmax = int(np.floor((zi + kr - loz) / pwz)) + 1
        if not perx:
            ixmin = max(ixmin, 0); ixmax = min(ixmax, nx)
        if not pery:
            jymin = max(jymin, 0); jymax = min(jymax, ny)
        if not perz:
            kzmin = max(kzmin, 0); kzmax = min(kzmax, nz)
        if ixmin >= ixmax or jymin >= jymax or kzmin >= kzmax:
            continue
        # PASS 1: Weight_i = sum_cells w (cell volume is constant -> cancels).
        Wi = 0.0
        for ix in range(ixmin, ixmax):
            cx = lox + (ix + 0.5) * pwx; dx = cx - xi; dx2 = dx * dx * inv
            for jy in range(jymin, jymax):
                cy = loy + (jy + 0.5) * pwy; dy = cy - yi
                dyz = dx2 + dy * dy * inv
                for kz in range(kzmin, kzmax):
                    cz = loz + (kz + 0.5) * pwz; dz = cz - zi
                    q2 = dyz + dz * dz * inv
                    if q2 < RADKERNEL2:
                        Wi += wkernel(q2)
        if Wi <= 0.0:
            continue
        # PASS 2: deposit m_i * (w / Wi).
        for ix in range(ixmin, ixmax):
            iw = ix % nx if perx else ix
            cx = lox + (ix + 0.5) * pwx; dx = cx - xi; dx2 = dx * dx * inv
            for jy in range(jymin, jymax):
                jw = jy % ny if pery else jy
                cy = loy + (jy + 0.5) * pwy; dy = cy - yi
                dyz = dx2 + dy * dy * inv
                for kz in range(kzmin, kzmax):
                    cz = loz + (kz + 0.5) * pwz; dz = cz - zi
                    q2 = dyz + dz * dz * inv
                    if q2 < RADKERNEL2:
                        kw = kz % nz if perz else kz
                        mw = mi * (wkernel(q2) / Wi)
                        mass_local[t, iw, jw, kw] += mw
                        for p in range(K):
                            massA_local[t, iw, jw, kw, p] += mw * vals[i, p]


def interpolate_uniform_scatter(particles, grid, values=None, hmin=None):
    """Conservative SPH scatter onto a `UniformGrid` (AREPO style; cf. AMR).

    Returns (data: dict[name -> (nx,ny,nz)], mass: (nx,ny,nz)). Density is the
    conserved cell mass / cell volume; other fields are mass-weighted averages."""
    if not isinstance(grid, UniformGrid):
        raise TypeError("interpolate_uniform_scatter requires a UniformGrid target")

    vals, names = particles.value_array(values)
    nx, ny, nz = grid.npx
    lo = grid.lo
    pw = grid.pixwidth
    perx, pery, perz = particles.periodic
    if hmin is None:
        hmin = 0.5 * float(np.max(pw))

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    mass_local = np.zeros((nthreads, nx, ny, nz), dtype=np.float64)
    massA_local = np.zeros((nthreads, nx, ny, nz, K), dtype=np.float64)

    pos = particles.pos
    _deposit_uniform_scatter(np.ascontiguousarray(pos[:, 0]),
                             np.ascontiguousarray(pos[:, 1]),
                             np.ascontiguousarray(pos[:, 2]),
                             particles.h, particles.mass, vals,
                             nx, ny, nz, lo[0], lo[1], lo[2],
                             pw[0], pw[1], pw[2], perx, pery, perz,
                             float(hmin), mass_local, massA_local)

    M = mass_local.sum(axis=0)
    MA = massA_local.sum(axis=0)
    Vcell = grid.cell_volume
    nzm = M > 0.0
    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = M / Vcell
        else:
            a = np.zeros_like(M)
            a[nzm] = MA[..., p][nzm] / M[nzm]
            data[name] = a
    return data, M


# --------------------------------------------------------------------------- #
#  Voronoi target: deposit onto arbitrary polyhedral cells. The SPH method
#  evaluates the kernel at the cell GENERATOR (the cell centre), exactly like
#  the UniformGrid path but with the cell centres irregular. Neighbour finding
#  is FUSED into the deposit as a uniform generator-bucket walk
#  (`targets.generator_buckets`) — materializing per-particle tree candidate
#  lists (cKDTree Python lists + CSR) cost more than the whole deposit on
#  ~1e6-particle snapshots (cf. the petkova Voronoi deposit).
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _bucket_walk_bounds(xi, yi, zi, ext, nb, blo, bw):
    """Bucket index cube [ix0..ix1]x[iy0..iy1]x[iz0..iz1] covering the sphere
    of radius `ext` around the particle, clamped to the grid."""
    ix0 = int((xi - ext - blo[0]) / bw[0]); ix1 = int((xi + ext - blo[0]) / bw[0])
    iy0 = int((yi - ext - blo[1]) / bw[1]); iy1 = int((yi + ext - blo[1]) / bw[1])
    iz0 = int((zi - ext - blo[2]) / bw[2]); iz1 = int((zi + ext - blo[2]) / bw[2])
    if ix0 < 0: ix0 = 0
    if iy0 < 0: iy0 = 0
    if iz0 < 0: iz0 = 0
    if ix1 > nb - 1: ix1 = nb - 1
    if iy1 > nb - 1: iy1 = nb - 1
    if iz1 > nb - 1: iz1 = nb - 1
    return ix0, ix1, iy0, iy1, iz0, iz1


@njit(cache=True)
def _axdist(u, lo, w):
    """Distance from coordinate u to the interval [lo, lo+w] (0 inside)."""
    if u < lo:
        return lo - u
    if u > lo + w:
        return u - lo - w
    return 0.0


@njit(parallel=True, cache=True)
def _deposit_voronoi_sph(x, y, z, h, weight, vals, genx, geny, genz,
                         nb, blo, bw, bstart, border, hmin,
                         out_local, norm_local):
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        wi = weight[i]
        hi = h[i]
        if wi <= 0.0 or hi <= 0.0:
            continue
        if hi < hmin:
            hi = hmin
        t = numba.get_thread_id()
        inv = 1.0 / (hi * hi)
        wnorm = wi / (np.pi * hi * hi * hi)
        xi = x[i]; yi = y[i]; zi = z[i]
        ext = RADKERNEL * hi
        ext2 = ext * ext
        ix0, ix1, iy0, iy1, iz0, iz1 = _bucket_walk_bounds(
            xi, yi, zi, ext, nb, blo, bw)
        for ix in range(ix0, ix1 + 1):
            dxb = _axdist(xi, blo[0] + ix * bw[0], bw[0])
            for iy in range(iy0, iy1 + 1):
                dyb = _axdist(yi, blo[1] + iy * bw[1], bw[1])
                dxy2 = dxb * dxb + dyb * dyb
                for iz in range(iz0, iz1 + 1):
                    b = (ix * nb + iy) * nb + iz
                    s0 = bstart[b]; s1 = bstart[b + 1]
                    if s1 == s0:
                        continue
                    dzb = _axdist(zi, blo[2] + iz * bw[2], bw[2])
                    if dxy2 + dzb * dzb >= ext2:
                        continue
                    for jj in range(s0, s1):
                        c = border[jj]
                        dx = genx[c] - xi; dy = geny[c] - yi; dz = genz[c] - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            w = wnorm * wkernel(q2)
                            norm_local[t, c] += w
                            for p in range(K):
                                out_local[t, c, p] += w * vals[i, p]


def interpolate_voronoi_sph(particles, grid, values=None, hmin=None,
                            normalise=True):
    """SPH-interpolate `particles` onto a `VoronoiGrid` (per-cell values).

    The kernel is evaluated at each cell's generator; fields are the Shepard-
    normalized kernel average there (cf. the UniformGrid SPH path, approximate /
    not strictly mass-conserving). `hmin` defaults to half the median cell size
    (volume**(1/3)) so a particle smaller than the local cell spacing still
    reaches the nearest generator instead of falling between cells.

    Returns (data: dict[name -> (Ncell,)], norm: (Ncell,)).
    """
    if not isinstance(grid, VoronoiGrid):
        raise TypeError("interpolate_voronoi_sph requires a VoronoiGrid target")

    vals, names = particles.value_array(values)
    Ntot = grid.ntotal                # deposit over ALL cells (incl. ghosts)
    if hmin is None:
        hmin = 0.5 * float(np.median(grid.volumes() ** (1.0 / 3.0)))
    NB = 64
    blo, bw, bstart, border, _ = generator_buckets(grid.gen, nb=NB)

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    out_local = np.zeros((nthreads, Ntot, K), dtype=np.float64)
    norm_local = np.zeros((nthreads, Ntot), dtype=np.float64)

    pos = particles.pos
    _deposit_voronoi_sph(np.ascontiguousarray(pos[:, 0]),
                         np.ascontiguousarray(pos[:, 1]),
                         np.ascontiguousarray(pos[:, 2]),
                         particles.h, particles.mass / particles.rho, vals,
                         np.ascontiguousarray(grid.gen[:, 0]),
                         np.ascontiguousarray(grid.gen[:, 1]),
                         np.ascontiguousarray(grid.gen[:, 2]),
                         NB, blo, bw, bstart, border, float(hmin),
                         out_local, norm_local)

    # fold periodic ghost-cell deposits onto the real (central) cells, then
    # Shepard-normalize (numerator and denominator both folded first).
    out = grid.fold_to_central(out_local.sum(axis=0))
    norm = grid.fold_to_central(norm_local.sum(axis=0))
    if normalise:
        mask = norm > 0.0
        for p in range(K):
            out[mask, p] /= norm[mask]

    data = {name: out[:, p] for p, name in enumerate(names)}
    return data, norm


@njit(parallel=True, cache=True)
def _deposit_voronoi_scatter(x, y, z, h, mass, vals, genx, geny, genz, vol,
                             nb, blo, bw, bstart, border, nearest, hmin,
                             mass_local, massA_local):
    """Conservative SPH scatter onto Voronoi cells (cf. `_deposit_amr_scatter`),
    but the neighbours come from the fused generator-bucket walk and the cell
    volume varies per generator. Two walks with IDENTICAL weights over the SAME
    cells:
        Weight_i = sum_j w(q_ij) * V_cell_j ;  share_ij = w V_j / Weight_i
    so each particle's mass shares sum to 1 exactly (mass-conserving). A particle
    whose kernel reaches no generator (Weight_i==0; happens on a coarse/subsampled
    mesh where the nearest generator is farther than the floored kernel) dumps its
    whole mass into the cell that CONTAINS it — its nearest generator — so mass is
    still conserved rather than dropped."""
    npart = x.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        hi = h[i]; mi = mass[i]
        if hi <= 0.0 or mi <= 0.0:
            continue
        if hi < hmin:
            hi = hmin
        t = numba.get_thread_id()
        inv = 1.0 / (hi * hi)
        xi = x[i]; yi = y[i]; zi = z[i]
        ext = RADKERNEL * hi
        ext2 = ext * ext
        ix0, ix1, iy0, iy1, iz0, iz1 = _bucket_walk_bounds(
            xi, yi, zi, ext, nb, blo, bw)
        # WALK 1: per-particle normaliser Weight_i = sum_j w_ij * V_j.
        Wi = 0.0
        for ix in range(ix0, ix1 + 1):
            dxb = _axdist(xi, blo[0] + ix * bw[0], bw[0])
            for iy in range(iy0, iy1 + 1):
                dyb = _axdist(yi, blo[1] + iy * bw[1], bw[1])
                dxy2 = dxb * dxb + dyb * dyb
                for iz in range(iz0, iz1 + 1):
                    b = (ix * nb + iy) * nb + iz
                    s0 = bstart[b]; s1 = bstart[b + 1]
                    if s1 == s0:
                        continue
                    dzb = _axdist(zi, blo[2] + iz * bw[2], bw[2])
                    if dxy2 + dzb * dzb >= ext2:
                        continue
                    for jj in range(s0, s1):
                        c = border[jj]
                        dx = genx[c] - xi; dy = geny[c] - yi; dz = genz[c] - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            Wi += wkernel(q2) * vol[c]
        if Wi <= 0.0:
            c = nearest[i]                       # whole mass -> containing cell
            mass_local[t, c] += mi
            for p in range(K):
                massA_local[t, c, p] += mi * vals[i, p]
            continue
        # WALK 2: distribute m_i (and m_i*A_i) with share_ij = w_ij V_j / Weight_i.
        for ix in range(ix0, ix1 + 1):
            dxb = _axdist(xi, blo[0] + ix * bw[0], bw[0])
            for iy in range(iy0, iy1 + 1):
                dyb = _axdist(yi, blo[1] + iy * bw[1], bw[1])
                dxy2 = dxb * dxb + dyb * dyb
                for iz in range(iz0, iz1 + 1):
                    b = (ix * nb + iy) * nb + iz
                    s0 = bstart[b]; s1 = bstart[b + 1]
                    if s1 == s0:
                        continue
                    dzb = _axdist(zi, blo[2] + iz * bw[2], bw[2])
                    if dxy2 + dzb * dzb >= ext2:
                        continue
                    for jj in range(s0, s1):
                        c = border[jj]
                        dx = genx[c] - xi; dy = geny[c] - yi; dz = genz[c] - zi
                        q2 = (dx * dx + dy * dy + dz * dz) * inv
                        if q2 < RADKERNEL2:
                            mw = mi * (wkernel(q2) * vol[c] / Wi)
                            mass_local[t, c] += mw
                            for p in range(K):
                                massA_local[t, c, p] += mw * vals[i, p]


def interpolate_voronoi_scatter(particles, grid, values=None, hmin=None):
    """Conservative SPH scatter of `particles` onto a `VoronoiGrid` (AREPO style).

    Returns (data: dict[name -> (Ncell,)], mass: (Ncell,)). Density (`rho`/
    `density`) is the CONSERVED cell mass / cell volume; every other field is the
    mass-weighted cell average. Total deposited mass = sum(particle mass) to
    machine precision (cf. the non-conserving `interpolate_voronoi_sph` gather).
    `hmin` mirrors the gather path (half the median cell size) so a sub-cell
    particle still reaches the nearest generator instead of falling between cells.
    """
    if not isinstance(grid, VoronoiGrid):
        raise TypeError("interpolate_voronoi_scatter requires a VoronoiGrid target")

    vals, names = particles.value_array(values)
    Ntot = grid.ntotal                # deposit over ALL cells (incl. ghosts)
    if hmin is None:
        hmin = 0.5 * float(np.median(grid.volumes() ** (1.0 / 3.0)))
    NB = 64
    blo, bw, bstart, border, _ = generator_buckets(grid.gen, nb=NB)
    # nearest generator = the cell each particle sits in (Voronoi membership);
    # used as the conservative fallback when the kernel reaches no generator.
    from scipy.spatial import cKDTree
    nearest = cKDTree(grid.gen).query(
        particles.pos, k=1, workers=-1)[1].ravel().astype(np.int64)

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    mass_local = np.zeros((nthreads, Ntot), dtype=np.float64)
    massA_local = np.zeros((nthreads, Ntot, K), dtype=np.float64)

    pos = particles.pos
    _deposit_voronoi_scatter(np.ascontiguousarray(pos[:, 0]),
                             np.ascontiguousarray(pos[:, 1]),
                             np.ascontiguousarray(pos[:, 2]),
                             particles.h, particles.mass, vals,
                             np.ascontiguousarray(grid.gen[:, 0]),
                             np.ascontiguousarray(grid.gen[:, 1]),
                             np.ascontiguousarray(grid.gen[:, 2]),
                             grid.volume, NB, blo, bw, bstart, border,
                             nearest, float(hmin), mass_local, massA_local)

    # fold periodic ghost-cell deposits onto the real cells FIRST (relocating mass
    # to its physical parent cell), then form the fields -> total mass conserved.
    M = grid.fold_to_central(mass_local.sum(axis=0))
    MA = grid.fold_to_central(massA_local.sum(axis=0))
    V = grid.volumes()
    return _scatter_to_fields(names, M, MA, V), M


# --------------------------------------------------------------------------- #
#  2D projection (direct, no 3D cube): column / average / rhocolumn / slice
# --------------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _deposit_proj(a1, a2, los, h, mass, volw, vals, is_slice, zslice,
                  perlos, boxlos, nx, ny, lo0, lo1, pw0, pw1, per0, per1,
                  col_table, a_vol, w_vol, a_mass, w_mass):
    """Deposit onto a 2D pixel grid. `is_slice`: 3D kernel at zslice; else the
    column-integrated kernel. Accumulators are per-thread (nthreads, nx, ny[, K])."""
    npart = a1.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        mi = mass[i]
        hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        dz2 = 0.0
        if is_slice:
            dz = zslice - los[i]
            if perlos:
                dz -= boxlos * np.round(dz / boxlos)   # minimum image
            if abs(dz) >= RADKERNEL * hi:
                continue
            dz2 = dz * dz
        t = numba.get_thread_id()
        inv_h2 = 1.0 / (hi * hi)
        cn = 1.0 / (np.pi * hi * hi * hi) if is_slice else 1.0 / (np.pi * hi * hi)
        kr = RADKERNEL * hi
        vi = volw[i]
        p1 = a1[i]; p2 = a2[i]
        imin = int(np.floor((p1 - kr - lo0) / pw0))
        imax = int(np.floor((p1 + kr - lo0) / pw0)) + 1
        jmin = int(np.floor((p2 - kr - lo1) / pw1))
        jmax = int(np.floor((p2 + kr - lo1) / pw1)) + 1
        if not per0:
            imin = max(imin, 0); imax = min(imax, nx)
        if not per1:
            jmin = max(jmin, 0); jmax = min(jmax, ny)
        for ix in range(imin, imax):
            iw = ix % nx if per0 else ix
            cx = lo0 + (ix + 0.5) * pw0
            dx = cx - p1
            dx2 = dx * dx
            for iy in range(jmin, jmax):
                cy = lo1 + (iy + 0.5) * pw1
                dy = cy - p2
                r2 = dx2 + dy * dy
                if is_slice:
                    q2 = (r2 + dz2) * inv_h2
                    if q2 >= RADKERNEL2:
                        continue
                    wgt = cn * wkernel(q2)
                else:
                    s = np.sqrt(r2) / hi
                    if s >= RADKERNEL:
                        continue
                    wgt = cn * col_profile(s, col_table)
                jw = iy % ny if per1 else iy
                mw = mi * wgt
                vw = vi * wgt
                w_mass[t, iw, jw] += mw
                w_vol[t, iw, jw] += vw
                for p in range(K):
                    a = vals[i, p]
                    a_vol[t, iw, jw, p] += vw * a
                    a_mass[t, iw, jw, p] += mw * a


def interpolate_projection2d(particles, target, values=None):
    """SPH 2D projection of `particles` onto a `Projection2D` target.

    Returns data: dict[name -> (nx,ny)]. Per mode:
      column     density-> surface density (sum m K); other-> column integral int A dz
      average    kernel-weighted LOS mean  A = sum(m/rho)A K / sum(m/rho)K
      rhocolumn  mass-weighted LOS mean    A = sum(m A K) / sum(m K)
      slice      density-> 3D density at plane; other-> field value at plane (avg form)
    """
    if not isinstance(target, Projection2D):
        raise TypeError("interpolate_projection2d requires a Projection2D target")

    vals, names = particles.value_array(values)
    nx, ny = target.npx
    lo = target.lo
    pw = target.pixwidth
    a1c = np.ascontiguousarray(particles.pos[:, target.axis1])
    a2c = np.ascontiguousarray(particles.pos[:, target.axis2])
    los = np.ascontiguousarray(particles.pos[:, target.los_axis])
    per0 = particles.periodic[target.axis1]
    per1 = particles.periodic[target.axis2]
    perlos = particles.periodic[target.los_axis]
    boxlos = float(particles.box[target.los_axis]) if (perlos and particles.box is not None) else 0.0
    if perlos and boxlos <= 0.0:
        perlos = False
    is_slice = (target.mode == "slice")

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    a_vol = np.zeros((nthreads, nx, ny, K))
    w_vol = np.zeros((nthreads, nx, ny))
    a_mass = np.zeros((nthreads, nx, ny, K))
    w_mass = np.zeros((nthreads, nx, ny))

    _deposit_proj(a1c, a2c, los, particles.h, particles.mass,
                  particles.mass / particles.rho, vals, is_slice,
                  float(target.zslice), perlos, boxlos, nx, ny,
                  lo[0], lo[1], pw[0], pw[1], per0, per1, COL_TABLE,
                  a_vol, w_vol, a_mass, w_mass)

    a_vol = a_vol.sum(axis=0); w_vol = w_vol.sum(axis=0)
    a_mass = a_mass.sum(axis=0); w_mass = w_mass.sum(axis=0)
    mv = w_vol > 0.0
    mm = w_mass > 0.0

    data = {}
    for p, name in enumerate(names):
        is_dens = name.lower() in ("rho", "density")
        if target.mode == "column":
            data[name] = w_mass.copy() if is_dens else a_vol[..., p]
        elif target.mode == "average":
            g = np.zeros((nx, ny)); g[mv] = a_vol[..., p][mv] / w_vol[mv]
            data[name] = g
        elif target.mode == "rhocolumn":
            g = np.zeros((nx, ny)); g[mm] = a_mass[..., p][mm] / w_mass[mm]
            data[name] = g
        else:  # slice
            if is_dens:
                data[name] = w_mass.copy()
            else:
                g = np.zeros((nx, ny)); g[mv] = a_vol[..., p][mv] / w_vol[mv]
                data[name] = g
    return data


# --------------------------------------------------------------------------- #
#  CONSERVATIVE 2D PROJECTION SCATTER.
#
#  The extensive quantity a 2D projection conserves is the COLUMN (projected)
#  mass -> surface density. Each particle spreads its mass over the pixels its
#  column kernel covers with a per-particle normalization, so the pixel shares
#  sum to exactly 1 and Sum(Sigma_pixel * pixel_area) = Sum(m_i) to machine
#  precision (cf. the non-conserving gather). Density -> conserved surface
#  density M/area; every other field -> mass-weighted column average, exactly
#  mirroring the 3D scatter semantics (density = M/V, others = MA/M).
#
#  A `slice` is a plane SAMPLE (3D kernel at one z), not an extensive column
#  integral, so a mass-conserving scatter is undefined there -> the dispatcher
#  routes slice mode to the gather sample instead.
# --------------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _deposit_proj_scatter(a1, a2, h, mass, vals, nx, ny, lo0, lo1, pw0, pw1,
                          per0, per1, col_table, mass_local, massA_local):
    npart = a1.shape[0]
    K = vals.shape[1]
    for i in prange(npart):
        mi = mass[i]; hi = h[i]
        if mi <= 0.0 or hi <= 0.0:
            continue
        t = numba.get_thread_id()
        kr = RADKERNEL * hi
        p1 = a1[i]; p2 = a2[i]
        imin = int(np.floor((p1 - kr - lo0) / pw0))
        imax = int(np.floor((p1 + kr - lo0) / pw0)) + 1
        jmin = int(np.floor((p2 - kr - lo1) / pw1))
        jmax = int(np.floor((p2 + kr - lo1) / pw1)) + 1
        if not per0:
            imin = max(imin, 0); imax = min(imax, nx)
        if not per1:
            jmin = max(jmin, 0); jmax = min(jmax, ny)
        if imin >= imax or jmin >= jmax:
            continue
        # PASS 1: per-particle normaliser Weight_i = sum_pixels col_profile.
        Wi = 0.0
        for ix in range(imin, imax):
            cx = lo0 + (ix + 0.5) * pw0; dx = cx - p1; dx2 = dx * dx
            for iy in range(jmin, jmax):
                cy = lo1 + (iy + 0.5) * pw1; dy = cy - p2
                s = np.sqrt(dx2 + dy * dy) / hi
                if s < RADKERNEL:
                    Wi += col_profile(s, col_table)
        if Wi <= 0.0:
            continue
        # PASS 2: distribute m_i (and m_i*A_i) with share = col_profile / Weight_i.
        for ix in range(imin, imax):
            iw = ix % nx if per0 else ix
            cx = lo0 + (ix + 0.5) * pw0; dx = cx - p1; dx2 = dx * dx
            for iy in range(jmin, jmax):
                jw = iy % ny if per1 else iy
                cy = lo1 + (iy + 0.5) * pw1; dy = cy - p2
                s = np.sqrt(dx2 + dy * dy) / hi
                if s < RADKERNEL:
                    mw = mi * (col_profile(s, col_table) / Wi)
                    mass_local[t, iw, jw] += mw
                    for p in range(K):
                        massA_local[t, iw, jw, p] += mw * vals[i, p]


def interpolate_projection2d_scatter(particles, target, values=None):
    """Conservative SPH scatter (surface density) onto a `Projection2D` target.

    Returns data: dict[name -> (nx,ny)]. Density (`rho`/`density`) is the
    CONSERVED projected surface density (mass / pixel area, so Sum over pixels of
    Sigma*area = total mass); every other field is the mass-weighted column
    average. `slice` mode delegates to the gather sample (a plane sample is not an
    extensive projection, so conservative scatter is undefined there)."""
    if not isinstance(target, Projection2D):
        raise TypeError("interpolate_projection2d_scatter requires a Projection2D target")
    if target.mode == "slice":
        return interpolate_projection2d(particles, target, values=values)

    vals, names = particles.value_array(values)
    nx, ny = target.npx
    lo = target.lo
    pw = target.pixwidth
    a1c = np.ascontiguousarray(particles.pos[:, target.axis1])
    a2c = np.ascontiguousarray(particles.pos[:, target.axis2])
    per0 = particles.periodic[target.axis1]
    per1 = particles.periodic[target.axis2]

    nthreads = numba.get_num_threads()
    K = vals.shape[1]
    mass_local = np.zeros((nthreads, nx, ny))
    massA_local = np.zeros((nthreads, nx, ny, K))

    _deposit_proj_scatter(a1c, a2c, particles.h, particles.mass, vals,
                          nx, ny, lo[0], lo[1], pw[0], pw[1], per0, per1,
                          COL_TABLE, mass_local, massA_local)

    M = mass_local.sum(axis=0)           # (nx,ny) projected mass per pixel
    MA = massA_local.sum(axis=0)
    area = float(pw[0] * pw[1])
    nz = M > 0.0
    data = {}
    for p, name in enumerate(names):
        if name.lower() in ("rho", "density"):
            data[name] = M / area        # conserved surface density
        else:
            g = np.zeros((nx, ny)); g[nz] = MA[..., p][nz] / M[nz]
            data[name] = g
    return data
