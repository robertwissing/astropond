"""FFT coarse-grid correction of a relaxed glass in a periodic box.

Plain gradient descent removes an error of wavelength L in ~(L/h)^2 iterations, so a relaxed glass
keeps long-wavelength density offsets whose force per particle is below the particle noise. This
removes them in one step, like a multigrid coarse-grid correction: the relax force is smoothed over
R (shorter scales are left to the relaxer), inverted to the pseudo-pressure change that balances
it, and that change is made with a mass-conserving displacement.

At equilibrium under pP = (rho/rho0)^2 (pP ~ 1): grad ln pP = rho a, a = the relax acceleration.
  1. F = rho a on a mesh (sph_interp sph-scatter deposit: conservative, kernel of h floored at half a
     cell, so it stays smooth when cells are smaller than h; Gaussian-smoothed over R); lap delta = div F
     (the curl part cannot be balanced by any pseudo-pressure and is dropped).
  2. Eulerian density change rho d ln rho = rho (delta + c) / 2, c such that its mean is 0 (mass).
  3. Mass flux j = rho u = grad phi, lap phi = -rho (delta + c) / 2, read back to the particles
     trilinearly; u = j / max(rho_i, rho_R) (rho_i = their own SPH density, when given). Both divisors
     agree where the mesh resolves the density. Where it does not (a dense object smaller than R),
     j / rho_R translates the object against its fixed target (the edge error it makes is invisible to
     the mesh and feeds back: areablob diverged), and j / rho_i over-moves the light particles beside
     it (they carry flux the mesh assigned to the smeared object: mhdcollapse stalled). The larger
     divisor = the least motion.
"""
import numba
import numpy as np


def _cic(pos, box, nc):
    """Trilinear (cloud-in-cell) weights of the 8 cells around each position (readback)."""
    g = (pos / box + 0.5) * nc - 0.5          # mesh coordinates, cell centres at integers
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    corners = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = np.ravel_multi_index(((i0[:, 0] + dx) % nc[0], (i0[:, 1] + dy) % nc[1],
                                            (i0[:, 2] + dz) % nc[2]), nc)
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                corners.append((idx, w))
    return corners


def coarse_correction(pos, mass, acc, box, R, h, rho=None, serial=False, strain=False):
    """Displacement u (N,3) removing the long-wavelength (> R) residual of the relax force acc.
    pos centred on 0 or not (wrapped here), box (3,) periodic lengths, h (N,) the particles' smoothing
    lengths (deposit kernel, support 2h), rho (N,) the particle densities (None = divide by the mesh
    rho_R alone). serial: deposit on one numba thread, so the result is bitwise reproducible (the
    parallel deposit sums each cell in a run-dependent order, ~1e-15 differences that the relax
    amplifies into a different run); costs 0.01-0.05 s per call. strain: also return info['strain'],
    the largest |d u_a / d x_b| of the mesh velocity j / rho_R (9 more FFTs). Returns (u, info)."""
    from .reconstruct import _import_sph_interp
    si = _import_sph_interp()
    box = np.asarray(box, float)
    nc = np.maximum(4, np.ceil(2.0 * box / R)).astype(int)    # cell <= R/2
    vcell = box.prod() / int(nc.prod())
    pos = pos - box * np.rint(pos / box)
    # Particles needs a rho; sph-scatter does not use it (it deposits mass with kernel h)
    p = si.Particles(pos=pos, mass=mass, rho=np.ones(len(mass)), h=h, box=box, periodic=True,
                     values={'ax': acc[:, 0], 'ay': acc[:, 1], 'az': acc[:, 2]})
    nt = numba.get_num_threads()
    if serial:
        numba.set_num_threads(1)
    try:
        res = si.interpolate(p, si.UniformGrid.centered(box, tuple(int(n) for n in nc)), method='sph-scatter')
    finally:
        numba.set_num_threads(nt)
    # res.norm = deposited mass per cell; res.data = mass-weighted mean a per cell
    F = [res.data[k] * res.norm / vcell for k in ('ax', 'ay', 'az')]
    cor = _cic(pos, box, nc)

    k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(nc[d], box[d] / nc[d]) for d in range(3)], indexing='ij')
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    G = np.exp(-0.5 * k2 * R * R)
    k2[0, 0, 0] = 1.0
    divF = sum(1j * k[d] * np.fft.fftn(F[d]) for d in range(3)) * G
    delta = np.real(np.fft.ifftn(-divF / k2))
    rhoR = np.real(np.fft.ifftn(np.fft.fftn(res.norm / vcell) * G))
    c = -(rhoR * delta).mean() / rhoR.mean()
    sh = np.fft.fftn(0.5 * rhoR * (delta + c))
    sh[0, 0, 0] = 0.0
    phih = sh / k2
    u = np.empty((len(pos), 3))
    for d in range(3):
        jd = np.real(np.fft.ifftn(1j * k[d] * phih))
        if rho is None:
            jd = jd / rhoR
        jd = jd.ravel()
        u[:, d] = sum(w * jd[i] for i, w in cor)
    if rho is not None:
        rr = rhoR.ravel()
        u /= np.maximum(rho, sum(w * rr[i] for i, w in cor))[:, None]
    dl = 0.5 * (delta + c)
    info = dict(nc=nc, dlnrho=dl, dlnrho_rms=float(np.sqrt((rhoR * dl * dl).mean() / rhoR.mean())))
    if strain:
        info['strain'] = 0.0
        for d in range(3):
            ud = np.fft.fftn(np.real(np.fft.ifftn(1j * k[d] * phih)) / rhoR)
            for e in range(3):
                info['strain'] = max(info['strain'], float(np.abs(np.real(np.fft.ifftn(1j * k[e] * ud))).max()))
    return u, info
