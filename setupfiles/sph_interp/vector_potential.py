"""Vector-potential (A) capture on a uniform grid, with B = curl(interp(A)).

The magnetic-projection `vector-potential` mode (FIELD_RECONSTRUCTION_PLAN.md,
"Magnetic-projection CLI surface", 2026-07-22).  The guarantee: reading B as the
analytic curl of a differentiable A interpolant makes ``div B == 0`` POINTWISE at
arbitrary query positions -- a property no face/RT0 representation retains off
its lattice.

Two PRODUCTION A-derivations, both producing a cell-centred grid A and read back
identically through :func:`curl_interp`:

- ``spectral``: deposit B on the grid, split off the volume mean ``Bbar``, and
  solve the Coulomb-gauge potential of the fluctuation ``Ahat = i k x Bhat/k^2``.
  ``k . Ahat == 0`` is automatic (the cross product kills the longitudinal part),
  so ``div A == 0`` is a property, not an assumption.  Open boxes use a >=2x
  zero-pad isolated-Poisson embedding.  The mean is restored analytically as
  ``A_bar(r) = 1/2 Bbar x r`` (whose curl is exactly ``Bbar``).

- ``biot-savart``: compute A ON THE PARTICLES from particle B via the
  parts-integrated free-space sum
  ``A_a = 1/(4 pi) sum_b V_b B_b x (r_a - r_b)/|r_a - r_b|^3`` (V_b = m_b/rho_b),
  then Petkova-deposit that particle A to the grid as an ordinary conservative
  field.  Natural for open/AMR geometry (no periodic box); softened at small
  separation.  Direct O(N^2) here -- adequate for the validation matrices; a
  tree summation is the scaling follow-up for full galaxy source counts.

Regardless of method, the meshfree Biot-Savart sum also serves as the grid-side
round-trip cross-check.
"""

import numpy as np

from .particles import Particles
from .targets import UniformGrid
from .interpolate import interpolate


# --------------------------------------------------------------------------
# Spectral B -> A on a uniform grid
# --------------------------------------------------------------------------

def _mean_split(cell_B):
    """Volume mean (equal cells) and the zero-mean fluctuation."""
    Bbar = cell_B.reshape(-1, 3).mean(axis=0)
    return Bbar, cell_B - Bbar


def _spectral_A_periodic(cell_Bp, pixwidth):
    """Coulomb-gauge A of a zero-mean periodic cell field.

    cell_Bp : (nx,ny,nz,3) zero-mean B fluctuation.
    Returns A : (nx,ny,nz,3), with div A == 0 and curl A == cell_Bp to spectral
    accuracy.  A itself is periodic and zero-mean.
    """
    dims = cell_Bp.shape[:3]
    Bhat = np.fft.fftn(cell_Bp, axes=(0, 1, 2))
    kx = 2.0 * np.pi * np.fft.fftfreq(dims[0], d=pixwidth[0])
    ky = 2.0 * np.pi * np.fft.fftfreq(dims[1], d=pixwidth[1])
    kz = 2.0 * np.pi * np.fft.fftfreq(dims[2], d=pixwidth[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = 1.0                       # avoid /0; the k=0 mode is set to 0
    # Ahat = i k x Bhat / k^2
    Bx, By, Bz = Bhat[..., 0], Bhat[..., 1], Bhat[..., 2]
    Ax = 1j * (KY * Bz - KZ * By) / k2
    Ay = 1j * (KZ * Bx - KX * Bz) / k2
    Az = 1j * (KX * By - KY * Bx) / k2
    for arr in (Ax, Ay, Az):
        arr[0, 0, 0] = 0.0
    A = np.stack([np.fft.ifftn(Ax, axes=(0, 1, 2)).real,
                  np.fft.ifftn(Ay, axes=(0, 1, 2)).real,
                  np.fft.ifftn(Az, axes=(0, 1, 2)).real], axis=-1)
    return A


def _spectral_A_open(cell_Bp, pixwidth, pad=2):
    """Isolated (free-space) A via a zero-padded periodic solve, then crop.

    Embeds the domain in a grid ``pad`` times larger per axis, zeroes B outside
    the physical region, solves the periodic problem (whose periodic images are
    far enough not to contaminate the interior), and returns A on the original
    cells.  The pad field is NOT itself divergence free, so its mean is removed
    before the solve; the caller restores the physical mean analytically.
    """
    dims = np.asarray(cell_Bp.shape[:3])
    big = tuple(int(pad) * dims)
    padded = np.zeros(big + (3,), dtype=np.float64)
    padded[:dims[0], :dims[1], :dims[2], :] = cell_Bp
    padded -= padded.reshape(-1, 3).mean(axis=0)
    A_big = _spectral_A_periodic(padded, pixwidth)
    return np.ascontiguousarray(A_big[:dims[0], :dims[1], :dims[2], :])


def bfield_to_vector_potential(cell_B, pixwidth, periodic=True, pad=2,
                               mean=None, return_mean=True):
    """Grid B -> grid A (the standalone B->A primitive).

    cell_B  : (nx,ny,nz,3) cell-centred magnetic field (SPH-deposited OR a
              directly sampled analytic field).
    pixwidth: (3,) cell edge lengths.
    periodic: periodic spectral solve; else the zero-pad isolated embedding.
    mean    : the uniform background to split off before the transverse solve
              and restore analytically at readout.  ``None`` uses the equal-cell
              grid mean (the correct choice for a box-filling periodic field); a
              (3,) vector supplies an externally computed mean (e.g. the
              particle-volume-weighted mean, matching the Hodge ``particle-mean``
              decomposition for a sparse open box).

    Returns (A, Bbar) with ``return_mean`` (default), else A only.  The physical
    field is ``B = curl(A) + Bbar``; the uniform ``Bbar`` is carried separately
    and reinstated analytically at readout as ``1/2 Bbar x r``.
    """
    cell_B = np.ascontiguousarray(cell_B, dtype=np.float64)
    if cell_B.ndim != 4 or cell_B.shape[-1] != 3:
        raise ValueError("cell_B must have shape (nx,ny,nz,3)")
    pixwidth = np.asarray(pixwidth, dtype=np.float64).reshape(3)
    if mean is None:
        Bbar, Bp = _mean_split(cell_B)
    else:
        Bbar = np.asarray(mean, dtype=np.float64).reshape(3)
        Bp = cell_B - Bbar
    if periodic:
        A = _spectral_A_periodic(Bp, pixwidth)
    else:
        A = _spectral_A_open(Bp, pixwidth, pad=pad)
    return (A, Bbar) if return_mean else A


# --------------------------------------------------------------------------
# Meshless Biot-Savart particle A
# --------------------------------------------------------------------------

try:
    import numba
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _biot_savart_numba(pos, VB, eps2, out):
        n = pos.shape[0]
        inv4pi = 1.0 / (4.0 * np.pi)
        for a in prange(n):
            xa = pos[a, 0]; ya = pos[a, 1]; za = pos[a, 2]
            sx = 0.0; sy = 0.0; sz = 0.0
            for b in range(n):
                dx = xa - pos[b, 0]
                dy = ya - pos[b, 1]
                dz = za - pos[b, 2]
                r2 = dx * dx + dy * dy + dz * dz + eps2
                inv = 1.0 / (r2 * np.sqrt(r2))
                # VB_b x d
                sx += (VB[b, 1] * dz - VB[b, 2] * dy) * inv
                sy += (VB[b, 2] * dx - VB[b, 0] * dz) * inv
                sz += (VB[b, 0] * dy - VB[b, 1] * dx) * inv
            out[a, 0] = inv4pi * sx
            out[a, 1] = inv4pi * sy
            out[a, 2] = inv4pi * sz
    _HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    _HAVE_NUMBA = False


def particle_vector_potential(pos, B, vol, softening, chunk=2048,
                              targets=None):
    """Free-space vector potential from every particle's B.

    A(r_a) = 1/(4 pi) sum_b V_b B_b x (r_a - r_b) / (|r_a - r_b|^2 + eps^2)^(3/2).

    pos      : (N,3) source positions.
    B        : (N,3) source magnetic field.
    vol      : (N,) source volume V_b = m_b/rho_b.
    softening: eps (Plummer); typically a small fraction of the mean spacing.
    targets  : optional (M,3) evaluation points (default: the sources).

    Parallel numba direct sum when available (O(N*M)); else a chunked numpy
    fallback.  This is the open/free-space Coulomb-gauge A; on a periodic
    fixture it will not equal the periodic spectral A cell-for-cell, but its
    curl reproduces the same B.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64).reshape(-1)
    n = len(pos)
    if pos.shape != (n, 3) or B.shape != (n, 3) or vol.shape != (n,):
        raise ValueError("pos/B must be (N,3) and vol (N,)")
    eps2 = float(softening) ** 2
    VB = np.ascontiguousarray(vol[:, None] * B)         # (N,3) V_b B_b
    tgt = pos if targets is None else np.ascontiguousarray(targets,
                                                           dtype=np.float64)
    if _HAVE_NUMBA and targets is None:
        A = np.zeros((n, 3), dtype=np.float64)
        _biot_savart_numba(pos, VB, eps2, A)
        return A
    m = len(tgt)
    A = np.zeros((m, 3), dtype=np.float64)
    inv4pi = 1.0 / (4.0 * np.pi)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d = tgt[s:e, None, :] - pos[None, :, :]           # (c,N,3) r_a - r_b
        r2 = np.einsum("cnk,cnk->cn", d, d) + eps2
        inv = r2 ** -1.5                                   # (c,N)
        cross = np.cross(VB[None, :, :], d)                # (c,N,3)
        A[s:e] = inv4pi * np.einsum("cn,cnk->ck", inv, cross)
    return A


# --------------------------------------------------------------------------
# C1 tricubic curl readout: B = curl(interp(A))
# --------------------------------------------------------------------------

def _catmull_rom(t):
    """Catmull-Rom weights and derivatives for the four samples at -1,0,1,2.

    t in [0,1) is the fractional offset past the lower-left sample (index 0).
    Returns (w, dw) each (4, ...) for points [-1,0,1,2]; dw is d/dt.
    """
    t2 = t * t
    t3 = t2 * t
    w = np.stack([
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ], axis=0)
    dw = np.stack([
        -0.5 + 2.0 * t - 1.5 * t2,
        -5.0 * t + 4.5 * t2,
        0.5 + 4.0 * t - 4.5 * t2,
        -1.0 * t + 1.5 * t2,
    ], axis=0)
    return w, dw


def _wrap_or_clamp(idx, n, periodic):
    if periodic:
        return np.mod(idx, n)
    return np.clip(idx, 0, n - 1)


def curl_interp(cell_A, grid, positions, Bbar=None, periodic=True):
    """B = curl of the C1 tricubic interpolant of cell-centred A, at positions.

    cell_A   : (nx,ny,nz,3) grid vector potential (the fluctuation part).
    grid     : UniformGrid whose cells hold cell_A at their centres.
    positions: (M,3) query points.
    Bbar     : optional (3,) uniform field added exactly (curl of 1/2 Bbar x r).

    div B == 0 pointwise by construction (div curl == 0 for the differentiable
    interpolant).  Returns B (M,3).
    """
    cell_A = np.ascontiguousarray(cell_A, dtype=np.float64)
    dims = cell_A.shape[:3]
    pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    pw = grid.pixwidth
    lo = grid.lo
    if np.isscalar(periodic):
        periodic = (bool(periodic),) * 3
    # cell-centre coordinate of the query, in cell units
    g = (pos - (lo + 0.5 * pw)) / pw            # (M,3), integer at cell centres
    base = np.floor(g).astype(np.int64)          # lower sample (index 0 of 4)
    frac = g - base                              # (M,3) in [0,1)
    wx, dwx = _catmull_rom(frac[:, 0])           # (4,M)
    wy, dwy = _catmull_rom(frac[:, 1])
    wz, dwz = _catmull_rom(frac[:, 2])
    m = len(pos)
    # Accumulate A and its Jacobian dA_c/dx_d
    A_val = np.zeros((m, 3))
    dAdx = np.zeros((m, 3))                       # dA_c/dx for c=0,1,2
    dAdy = np.zeros((m, 3))
    dAdz = np.zeros((m, 3))
    off = (-1, 0, 1, 2)
    for a, ox in enumerate(off):
        ix = _wrap_or_clamp(base[:, 0] + ox, dims[0], periodic[0])
        for b, oy in enumerate(off):
            iy = _wrap_or_clamp(base[:, 1] + oy, dims[1], periodic[1])
            wxy = wx[a] * wy[b]
            dwx_wy = dwx[a] * wy[b]
            wx_dwy = wx[a] * dwy[b]
            for c, oz in enumerate(off):
                iz = _wrap_or_clamp(base[:, 2] + oz, dims[2], periodic[2])
                Acell = cell_A[ix, iy, iz, :]        # (M,3)
                w = wxy * wz[c]
                A_val += w[:, None] * Acell
                dAdx += (dwx_wy * wz[c])[:, None] * Acell
                dAdy += (wx_dwy * wz[c])[:, None] * Acell
                dAdz += (wxy * dwz[c])[:, None] * Acell
    dAdx /= pw[0]
    dAdy /= pw[1]
    dAdz /= pw[2]
    # curl A = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy)
    Bx = dAdy[:, 2] - dAdz[:, 1]
    By = dAdz[:, 0] - dAdx[:, 2]
    Bz = dAdx[:, 1] - dAdy[:, 0]
    B = np.stack([Bx, By, Bz], axis=1)
    if Bbar is not None:
        B = B + np.asarray(Bbar, dtype=np.float64).reshape(1, 3)
    return B


# --------------------------------------------------------------------------
# Table object + capture
# --------------------------------------------------------------------------

class VectorPotentialTable:
    """Grid A + uniform mean, read back as B = curl(interp(A)) + Bbar."""

    _SCHEMA = "sph_interp-vector-potential-1"

    def __init__(self, grid, cell_A, Bbar, periodic=True, a_method="spectral",
                 audit=None):
        self.grid = grid
        self.cell_A = np.ascontiguousarray(cell_A, dtype=np.float64)
        self.Bbar = np.asarray(Bbar, dtype=np.float64).reshape(3)
        if np.isscalar(periodic):
            periodic = (bool(periodic),) * 3
        self.periodic = tuple(bool(p) for p in periodic)
        self.a_method = a_method
        self.audit = audit or {}

    def query(self, positions):
        return curl_interp(self.cell_A, self.grid, positions,
                           Bbar=self.Bbar, periodic=self.periodic)

    def cell_centred(self):
        """B = curl(interp(A)) sampled at the cell centres (diagnostic)."""
        centres = self.grid.centres()
        gx, gy, gz = np.meshgrid(*centres, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        B = self.query(pts)
        return B.reshape(self.grid.npx + (3,))

    def magnetic_energy(self):
        B = self.cell_centred()
        return float(0.5 * self.grid.cell_volume * np.sum(B ** 2))

    def save(self, path):
        np.savez(path, schema=self._SCHEMA, cell_A=self.cell_A,
                 Bbar=self.Bbar, lo=self.grid.lo, hi=self.grid.hi,
                 npx=np.asarray(self.grid.npx), periodic=np.asarray(self.periodic),
                 a_method=self.a_method)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        if str(z["schema"]) != cls._SCHEMA:
            raise ValueError(f"unexpected schema {z['schema']!r}")
        grid = UniformGrid(lo=z["lo"], hi=z["hi"], npx=tuple(int(x) for x in z["npx"]))
        return cls(grid, z["cell_A"], z["Bbar"], periodic=tuple(z["periodic"]),
                   a_method=str(z["a_method"]))


def _deposit_cell_B(particles, grid, fields):
    """Conservative Petkova cell-average B via the m*(B/rho) volume carrier."""
    carriers = {}
    names = []
    for axis, field in zip("xyz", fields):
        comp = np.asarray(particles.values[field], dtype=np.float64)
        key = f"__A_B{axis}_over_rho"
        carriers[key] = comp / particles.rho
        names.append(key)
    cp = Particles(pos=particles.pos, mass=particles.mass, rho=particles.rho,
                   h=particles.h, values=carriers, box=particles.box,
                   periodic=particles.periodic)
    dep = interpolate(cp, grid, method="petkova", values=names,
                      conservative_fields=True)
    cellB = np.stack([np.asarray(dep.data[f"_content_{k}"], dtype=np.float64)
                      / grid.cell_volume for k in names], axis=-1)
    return cellB


def capture_uniform_vector_potential(particles, grid, fields=("Bx", "By", "Bz"),
                                     a_method="spectral", pad=2, softening=None,
                                     mean="particle"):
    """Capture A on a uniform grid from SPH particle B.

    a_method='spectral' : deposit B (petkova volume carrier) -> spectral A.
    a_method='biot-savart': meshless particle A -> petkova-deposit A to the grid.

    Returns a :class:`VectorPotentialTable` (read back with ``.query``).  The
    audit records the mean field, the round-trip B error at the source-particle
    positions, and the magnetic-energy budget through the stages.
    """
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if not isinstance(grid, UniformGrid):
        raise TypeError("vector-potential capture requires a UniformGrid")
    if len(fields) != 3:
        raise ValueError("fields must be the three magnetic components")
    periodic = tuple(bool(p) for p in (particles.periodic
                     if particles.periodic is not None else (False, False, False)))
    source_B = np.stack([np.asarray(particles.values[f], dtype=np.float64)
                         for f in fields], axis=1)
    vol = particles.mass / particles.rho

    if a_method == "spectral":
        cellB = _deposit_cell_B(particles, grid, fields)
        # 'particle' -> particle-volume-weighted mean (the box-threading part of
        # a sparse open field; matches the Hodge particle-mean decomposition).
        if isinstance(mean, str) and mean == "particle":
            mean_vec = np.sum(vol[:, None] * source_B, axis=0) / float(vol.sum())
        elif isinstance(mean, str) and mean == "cell":
            mean_vec = None
        else:
            mean_vec = np.asarray(mean, dtype=np.float64).reshape(3)
        cell_A, Bbar = bfield_to_vector_potential(
            cellB, grid.pixwidth, periodic=all(periodic), pad=pad, mean=mean_vec)
    elif a_method == "biot-savart":
        if softening is None:
            softening = 0.5 * float(np.mean(grid.pixwidth))
        pA = particle_vector_potential(particles.pos, source_B, vol, softening)
        # Petkova-deposit particle A (conserved volume carrier m*A/rho / V).
        carriers, names = {}, []
        for axis in range(3):
            key = f"__A_comp{axis}_over_rho"
            carriers[key] = pA[:, axis] / particles.rho
            names.append(key)
        cp = Particles(pos=particles.pos, mass=particles.mass, rho=particles.rho,
                       h=particles.h, values=carriers, box=particles.box,
                       periodic=particles.periodic)
        dep = interpolate(cp, grid, method="petkova", values=names,
                          conservative_fields=True)
        cell_A = np.stack([np.asarray(dep.data[f"_content_{k}"], dtype=np.float64)
                           / grid.cell_volume for k in names], axis=-1)
        # A already contains any uniform part; no separate analytic mean.
        Bbar = np.zeros(3)
    else:
        raise ValueError("a_method must be 'spectral' or 'biot-savart'")

    table = VectorPotentialTable(grid, cell_A, Bbar, periodic=periodic,
                                 a_method=a_method)

    # Round-trip audit at the source positions.
    B_rt = table.query(particles.pos)
    diff = B_rt - source_B
    denom = np.sqrt(np.sum(source_B ** 2))
    rel_l2 = float(np.sqrt(np.sum(diff ** 2)) / denom) if denom > 0 else None
    e_src = float(0.5 * np.sum(vol * np.sum(source_B ** 2, axis=1)))
    e_rt = float(0.5 * np.sum(vol * np.sum(B_rt ** 2, axis=1)))
    table.audit = dict(
        schema=table._SCHEMA, a_method=a_method,
        mean_field=table.Bbar.tolist(),
        roundtrip_B_rel_l2=rel_l2,
        source_energy=e_src, roundtrip_energy=e_rt,
        roundtrip_over_source=(e_rt / e_src if e_src > 0 else None),
    )
    return table
