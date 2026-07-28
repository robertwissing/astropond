"""Synthetic particle distributions modelled after the real IC families.

Built for `_bench_scaling.py`: a smoke test and a scaling harness need inputs
that reproduce the REGIMES the interpolators actually meet, because cost is
driven by the (particle, cell) pair count and that is a property of the h
distribution, not of N alone. A Poisson box and an evolved galaxy differ by
orders of magnitude in dynamic range and behave nothing alike.

CALIBRATION. The structured families are tuned against every IC in
`test_cases/isolated_galaxy`, measured with `describe()`:

  IC (real)        N       h_dyn    rho_dyn   occ_64   modelled by
  isolated_disc    800k    3.5e3    4.4e10    0.028    disk
  lowres_merger    518k    6.2e3    8.4e14    0.150    merger
  highres_merger   8.1M    2.6e5    1.1e16    0.097    merger (deeper)
  dwarf_cosmo      6.0M    3.1e4    1.5e14    0.155    cosmo
  agora_cosmo      14.8M   9.0e3    7.0e11    0.282    cosmo

The defaults are TUNED to those numbers, checked at matched N (both h_dyn and
occ_64 depend on particle count, so a comparison at different N is worthless):

  family    h_dyn vs real   occ_64 vs real
  disk           1.20x           1.26x
  merger         0.99x           1.17x
  cosmo          1.03x           0.95x

One deliberate mismatch: `rho_dyn` for `merger` comes out ~3e-4 of the real
snapshot's. Here rho is DERIVED from h, so rho_dyn == h_dyn^3 identically,
while the real merger's stored rho spans further than its own h implies
(8.4e14 against h_dyn^3 = 2.3e11) -- a multiphase ISM carries density
structure the smoothing length has already averaged over. Matching it would
mean abandoning the equal-mass, h-consistent construction that makes every
family comparable, so it is left as a known limitation. For interpolation
COST, which is what this bench measures, h is the variable that matters.

The single most important number there is `occ_64`: on a 64^3 grid over the
box, between 72% and 97% of cells hold NO particle. Interpolation cost on real
data is dominated by a thin, extremely inhomogeneous occupied fraction, and a
synthetic that fills its box uniformly does not test that at all.

Ten families, each `make_<name>(n, seed=...) -> Particles`:

  uniform   Poisson points, periodic unit box. Flat density, narrow h. The
            control: no dynamic range, no structure.
  lattice   cubic lattice, periodic. Every h identical and every particle
            commensurate with a power-of-two grid -- the petkova stress case
            (see PLAN_PETKOVA_FIX: a biased near-field integral shows up as
            stripes here and nowhere else).
  closepacked  ABC/hcp lattice, periodic -- the geometry `IC_distribute`'s
            `setcloseddist` produces, and what the shocktube and Orszag-Tang
            setups actually use by default.
  glass     relaxed quasi-uniform packing, periodic. Uniform like the lattice
            but with no symmetry to cancel errors; the production IC style.
  slab1d    128:12:12 box with an 8:1 density jump -- the shocktube geometry.
  slab2d    128:128:12 box, uniform -- the Orszag-Tang geometry. Thin boxes
            have the worst history here (the AREPO per-axis LONG_Y/Z work and
            the +340% petkova error both came from them), so they are their
            own families rather than a variant.
  disk      bulge + exponential disk + sech^2 vertical profile + a sparse
            diffuse halo, OPEN boundary in a box much larger than the disk.
            Models the isolated galaxy: a thin slab in a nearly empty box.
  merger    two inclined discs at a separation, OPEN unit box. Models the
            merger snapshots -- two concentrations rather than one, which is
            what breaks a single-centre assumption in any spatial partition.
  collapse  isothermal (rho ~ r^-2) core plus a uniform background, periodic.
            The idealised extreme-contrast case, tuned to ~1e9 in density.
  cosmo     halo population + connecting filaments + diffuse background,
            periodic unit box. Models the cosmological runs: most of the
            volume empty, structure on every scale, h over four decades.

All generators are seeded and deterministic, and every family has a box that
depends only on its parameters, NOT on the sampled particles. That matters for
benchmarking: the AREPO Voronoi backend compiles a per-aspect-ratio binary, so
a box that drifted with N would rebuild it on every point of a scaling sweep
and charge the rebuild to the mesh time.

Masses are equal within a family (the SPH convention), so the density profile
comes out of the SAMPLING, not out of a per-particle weight, and `rho` is
measured from the neighbour distance the same way for every family.
"""

import os
import tempfile

import numpy as np

from .particles import Particles

__all__ = ["FAMILIES", "make", "make_uniform", "make_lattice",
           "make_closepacked", "make_glass", "make_slab1d", "make_slab2d",
           "make_disk", "make_merger", "make_collapse", "make_cosmo",
           "describe", "occupancy"]

FAMILIES = ("uniform", "lattice", "closepacked", "glass", "slab1d", "slab2d",
            "disk", "merger", "collapse", "cosmo")

_KNN = 32                      # neighbours for the h estimate
_GLASS_CACHE = os.environ.get(
    "SPH_INTERP_GLASS_CACHE",
    os.path.join(tempfile.gettempdir(), "sph_interp_glass_cache"))


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _knn_h(pos, box, periodic, k=_KNN):
    """h = d_k / 2 (the 2h-compact convention used everywhere in this package).

    Periodic families get a genuinely periodic kNN (cKDTree `boxsize`), which
    matters more than it looks: without it the particles within d_k of a face
    report an inflated h, and since cost scales as h^3 that inflates the
    measured run time at exactly the boundary cells the scaling test is
    trying to hold fixed.
    """
    from scipy.spatial import cKDTree
    pos = np.asarray(pos, dtype=np.float64)
    per = np.atleast_1d(periodic)
    if per.size == 1:
        per = np.repeat(per, 3)
    kk = min(int(k), max(1, pos.shape[0] - 1))
    if per.all():
        box = np.asarray(box, dtype=np.float64).reshape(3)
        # cKDTree wants coordinates in [0, L); ours are centred on 0.
        shifted = (pos + 0.5 * box) % box
        tree = cKDTree(shifted, boxsize=box)
        d, _ = tree.query(shifted, k=kk + 1, workers=-1)
    else:
        tree = cKDTree(pos)
        d, _ = tree.query(pos, k=kk + 1, workers=-1)
    return 0.5 * d[:, -1]


def _finalize(pos, box, periodic, k=_KNN, mtot=1.0):
    """Equal masses, h from kNN, rho consistent with (m, h, k)."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    n = pos.shape[0]
    box = np.asarray(box, dtype=np.float64).reshape(3)
    h = _knn_h(pos, box, periodic, k)
    mass = np.full(n, float(mtot) / n)
    rho = mass * k / (4.0 * np.pi / 3.0 * (2.0 * h) ** 3)
    return Particles(pos=pos, mass=mass, rho=rho, h=h,
                     values={"rho": rho}, box=box, periodic=periodic)


def _sample_inside(n, draw, halfbox, rng, max_rounds=64):
    """Draw exactly `n` points inside the half-box, resampling rejects.

    Truncating (rather than letting the box follow the sample) is what keeps
    the box independent of N -- see the module docstring on the AREPO binary.
    """
    out = []
    got = 0
    for _ in range(max_rounds):
        want = max(1024, int(1.4 * (n - got)))
        cand = draw(want, rng)
        ok = np.all(np.abs(cand) < halfbox, axis=1)
        cand = cand[ok]
        # SHUFFLE before truncating. A draw function that concatenates
        # components (merger: disc then bulge) returns them in blocks, and
        # slicing the first `n` would silently drop whichever component came
        # last -- which is how the merger lost its entire bulge, and with it
        # h_min and 2.5 decades of dynamic range.
        if cand.shape[0] > 1:
            rng.shuffle(cand, axis=0)
        if cand.size:
            out.append(cand[: n - got])
            got += out[-1].shape[0]
        if got >= n:
            break
    if got < n:                       # pathological parameters: pad centrally
        out.append(rng.uniform(-0.1, 0.1, (n - got, 3)) * halfbox)
    return np.concatenate(out)[:n]


def _isotropic(r, rng):
    """Turn radii into isotropic 3D positions."""
    cost = rng.uniform(-1.0, 1.0, r.size)
    sint = np.sqrt(np.maximum(0.0, 1.0 - cost ** 2))
    phi = rng.uniform(0.0, 2.0 * np.pi, r.size)
    return np.stack([r * sint * np.cos(phi),
                     r * sint * np.sin(phi),
                     r * cost], axis=1)


def _plummer(n, a, rng, rmax_mult=25.0):
    """Plummer sphere of scale `a`, truncated at `rmax_mult * a`.

    CORED on purpose. A Gamma- or power-law-sampled "bulge" has a 1/r number
    density cusp, which makes h_min scale-free: it keeps shrinking as particles
    are added and stops responding to the scale length, so the family's dynamic
    range cannot be calibrated. A Plummer core pins h_min to `a` and the local
    particle count, which is what makes `h_dyn` a tunable property here.
    """
    u = rng.random(int(n))
    r = a / np.sqrt(np.maximum(u ** (-2.0 / 3.0) - 1.0, 1e-12))
    return _isotropic(np.minimum(r, rmax_mult * a), rng)


def _shell(n, r_in, r_out, rng, flatten=1.0):
    """Isothermal (n ~ r^-2) shell between r_in and r_out.

    The inner cutoff is the point: it keeps the diffuse component from piling
    up at the galaxy centre, so h_max (set by the sparse outer edge) and h_min
    (set by the bulge) can be tuned independently.
    """
    r = r_in + (r_out - r_in) * rng.random(int(n))
    pos = _isotropic(r, rng)
    pos[:, 2] *= flatten
    return pos


def _rotate(pos, incl, pa=0.0):
    """Incline a disc by `incl` about x, then rotate by `pa` about z."""
    ci, si = np.cos(incl), np.sin(incl)
    rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    cp, sp = np.cos(pa), np.sin(pa)
    rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    return pos @ rx.T @ rz.T


# --------------------------------------------------------------------------
# uniform / lattice / glass -- the quasi-uniform periodic trio
# --------------------------------------------------------------------------

def make_uniform(n, seed=42, box=1.0):
    """Poisson points in a periodic cube centred on the origin."""
    box3 = np.full(3, float(box))
    rng = np.random.default_rng(seed)
    pos = (rng.random((int(n), 3)) - 0.5) * box3
    return _finalize(pos, box3, True)


def make_lattice(n, seed=42, box=1.0):
    """Cubic lattice, periodic. N is rounded to the nearest perfect cube."""
    box3 = np.full(3, float(box))
    nside = max(2, int(round(float(n) ** (1.0 / 3.0))))
    c = (np.arange(nside) + 0.5) / nside - 0.5
    g = np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)
    return _finalize(g * box3, box3, True)


def _closepacked_points(box, delta):
    """ABC-stacked close-packed lattice filling `box`, centred on the origin.

    Geometry mirrors `IC_distribute.setcloseddist`, which is what the shocktube
    and Orszag-Tang setups actually use (`distri=0`): nearest-neighbour spacing
    `delta` along x, hex row spacing delta*sqrt(3)/2, layer spacing
    delta*sqrt(6)/3, with ABC layer offsets. Row and layer counts are forced
    even / multiple-of-three so the tiling closes periodically, and the spacings
    are then recomputed to fit the box exactly.

    Worth having as its own family: a close-packed lattice has a different
    neighbour count and a different symmetry from the cubic one, so the error
    cancellations that make `lattice` a petkova stress case do not carry over.
    """
    box = np.asarray(box, dtype=np.float64).reshape(3)
    dx_req = float(delta)
    dy_req = dx_req * np.sqrt(3.0) / 2.0
    dz_req = dx_req * np.sqrt(6.0) / 3.0

    nx = max(1, int(round(box[0] / dx_req)))
    ny = max(2, 2 * int(round((box[1] / dy_req) / 2.0)))
    nz = max(3, 3 * int(round((box[2] / dz_req) / 3.0)))
    dx, dy, dz = box[0] / nx, box[1] / ny, box[2] / nz

    k = np.arange(nx)
    out = np.empty((nz, ny, nx, 3), dtype=np.float64)
    for m in range(nz):
        jz = m % 3
        x_layer = 0.0 if jz != 1 else 0.5 * dx
        y_layer = (0.0, dy / 3.0, 2.0 * dy / 3.0)[jz]
        z = 0.5 * dz + m * dz
        for l in range(ny):
            x_row = 0.5 * dx if (l % 2 == 0) else 0.0
            out[m, l, :, 0] = 0.5 * dx + x_layer + x_row + k * dx
            out[m, l, :, 1] = 0.5 * dy + l * dy + y_layer
            out[m, l, :, 2] = z
    pos = out.reshape(-1, 3)
    pos = pos % box - 0.5 * box                     # wrap, then centre
    return pos


def _delta_for(box, n):
    """Close-packed spacing that puts ~n particles in `box`."""
    box = np.asarray(box, dtype=np.float64).reshape(3)
    cell = np.sqrt(3.0) / 2.0 * np.sqrt(6.0) / 3.0   # dy*dz per dx^2
    return float((np.prod(box) / (cell * max(1, int(n)))) ** (1.0 / 3.0))


def make_closepacked(n, seed=42, box=1.0):
    """Close-packed (ABC/hcp) lattice in a periodic cube."""
    box3 = np.full(3, float(box))
    return _finalize(_closepacked_points(box3, _delta_for(box3, n)), box3, True)


def _glass_lloyd(pos, box, iters=60, seed=42):
    """Kernel-repulsion relaxation: the standard way to beat Poisson noise out
    of a point set without introducing lattice symmetry.

    Each particle is pushed off its `k` nearest neighbours along the pair
    separation with a weight that falls off linearly to the neighbour radius,
    with the step capped at a fraction of the mean spacing so the set cannot
    explode. Converges to a quasi-uniform packing in a few tens of sweeps,
    which is all a bench input needs.
    """
    from scipy.spatial import cKDTree
    pos = np.array(pos, dtype=np.float64, copy=True)
    n = pos.shape[0]
    box = np.asarray(box, dtype=np.float64).reshape(3)
    spacing = float(np.prod(box) / n) ** (1.0 / 3.0)
    k = 16
    step = 0.20 * spacing
    for it in range(iters):
        shifted = (pos + 0.5 * box) % box
        tree = cKDTree(shifted, boxsize=box)
        d, idx = tree.query(shifted, k=k + 1, workers=-1)
        d, idx = d[:, 1:], idx[:, 1:]                 # drop self
        sep = shifted[:, None, :] - shifted[idx]      # minimum image
        sep -= box * np.round(sep / box)
        r = np.maximum(d, 1e-12)[:, :, None]
        w = np.maximum(0.0, 1.0 - d / (2.0 * spacing))[:, :, None]
        push = (sep / r * w).sum(axis=1)
        norm = np.maximum(np.linalg.norm(push, axis=1, keepdims=True), 1e-12)
        push = push / norm * np.minimum(norm, 1.0) * step * (1.0 - it / iters)
        pos = pos + push
        pos = (pos + 0.5 * box) % box - 0.5 * box
    return pos


def _glass_glassgen(n, seed, box, max_iter):
    """Authentic glass from the in-tree glassgen relaxer, if it imports."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..",
                                        "icgenerator", "glassgen_python"))
    if root not in sys.path:
        sys.path.insert(0, root)
    import glassgen

    rng = np.random.default_rng(seed)
    box3 = np.asarray(box, dtype=np.float64).reshape(3)
    pos0 = (rng.random((int(n), 3)) - 0.5) * box3
    mass = np.full(int(n), 1.0 / int(n))
    dens = glassgen.CallableDensity(lambda x: np.ones(len(x)))
    res = glassgen.relax(pos0, mass, dens, box=box3, nsmooth=_KNN,
                         max_iter=max_iter, verbose=False)
    return np.asarray(res.pos, dtype=np.float64)


def make_glass(n, seed=42, box=1.0, max_iter=400, backend="auto", cache=True):
    """Relaxed quasi-uniform packing in a periodic cube.

    `backend`: "glassgen" (the in-tree relaxer), "lloyd" (the local
    kernel-repulsion fallback) or "auto" (glassgen, falling back to lloyd).
    Relaxation is the expensive part of the whole bench, so results are cached
    on disk by (n, seed, box, backend).
    """
    n = int(n)
    box3 = np.full(3, float(box))
    tag = f"glass_n{n}_s{seed}_b{float(box):g}_{backend}.npz"
    path = os.path.join(_GLASS_CACHE, tag)
    if cache and os.path.exists(path):
        return _finalize(np.load(path)["pos"], box3, True)

    pos = None
    if backend in ("auto", "glassgen"):
        try:
            pos = _glass_glassgen(n, seed, box3, max_iter)
        except Exception as exc:                     # noqa: BLE001
            if backend == "glassgen":
                raise
            print(f"  [synthetic] glassgen unavailable ({type(exc).__name__}: "
                  f"{exc}); falling back to the lloyd relaxer")
    if pos is None:
        rng = np.random.default_rng(seed)
        pos0 = (rng.random((n, 3)) - 0.5) * box3
        pos = _glass_lloyd(pos0, box3, seed=seed)

    if cache:
        os.makedirs(_GLASS_CACHE, exist_ok=True)
        np.savez_compressed(path, pos=pos)
    return _finalize(pos, box3, True)


# --------------------------------------------------------------------------
# slab1d / slab2d -- the thin-box test geometries
# --------------------------------------------------------------------------

# Aspect ratios in units of the base particle spacing, from the real setups:
# the shocktube runs a long-x tube thin in y and z, Orszag-Tang a square sheet
# thin only in z (`IC_setup_shocktube.py`, `IC_setup_orzag.py`).
SLAB1D_ASPECT = (128.0, 12.0, 12.0)
SLAB2D_ASPECT = (128.0, 128.0, 12.0)


def make_slab1d(n, seed=42, rho_ratio=8.0, distribution="closepacked",
                aspect=SLAB1D_ASPECT):
    """Shocktube geometry: a long, thin periodic box with a density jump.

    Box aspect 128:12:12, i.e. quasi-1D. Two states of different density meet
    at x = 0, sampled at the spacing each state implies (spacing ratio
    rho_ratio^(1/3); Sod's 8:1 in density is 2:1 in spacing), so the
    DISCONTINUITY IS IN THE PARTICLE DISTRIBUTION, not painted on afterwards.

    This is the geometry with the worst history in this package: the AREPO
    Voronoi backend's per-axis LONG_Y/LONG_Z handling exists because a thin box
    meshed through the isotropic ghost path gave a +340% petkova error, and the
    mixed-periodicity ghost bug (plan task 3) reached ~10^4x in volume on
    exactly a thin slab. A scaling bench that only ever sees cubes would miss
    the entire class.
    """
    aspect = np.asarray(aspect, dtype=np.float64)
    box = aspect / aspect[0]                       # normalise: x extent = 1
    nleft = int(round(n * rho_ratio / (1.0 + rho_ratio)))
    nright = max(1, n - nleft)

    # each half is its own close-packed block at its own spacing
    halfbox = np.array([0.5 * box[0], box[1], box[2]])
    dl = _delta_for(halfbox, nleft)
    dr = _delta_for(halfbox, nright)
    left = _closepacked_points(halfbox, dl)
    right = _closepacked_points(halfbox, dr)
    if distribution == "random":
        rng = np.random.default_rng(seed)
        left = (rng.random(left.shape) - 0.5) * halfbox
        right = (rng.random(right.shape) - 0.5) * halfbox
    left[:, 0] -= 0.25 * box[0]
    right[:, 0] += 0.25 * box[0]

    pos = np.concatenate([left, right])
    pos = (pos + 0.5 * box) % box - 0.5 * box
    return _finalize(pos, box, True)


def make_slab2d(n, seed=42, distribution="closepacked", aspect=SLAB2D_ASPECT):
    """Orszag-Tang geometry: a square sheet, thin in z, uniform density.

    Box aspect 128:128:12, i.e. quasi-2D. Orszag-Tang carries its structure in
    the velocity and magnetic fields rather than in the particle positions, so
    the distribution stays uniform -- which makes this the clean control for
    the thin-box AXIS EFFECT on its own, with none of the density dynamic
    range the disk and collapse families add.
    """
    aspect = np.asarray(aspect, dtype=np.float64)
    box = aspect / aspect[0]
    if distribution == "random":
        rng = np.random.default_rng(seed)
        pos = (rng.random((int(n), 3)) - 0.5) * box
    else:
        pos = _closepacked_points(box, _delta_for(box, n))
    return _finalize(pos, box, True)


# --------------------------------------------------------------------------
# disk -- the isolated-galaxy analogue, open boundary
# --------------------------------------------------------------------------

def _disc_component(n, rd, z0, rng, kind="exp"):
    """Exponential surface density with a sech^2 vertical profile.

    Sigma ~ exp(-R/rd) gives a radial pdf ~ R exp(-R/rd), i.e. Gamma(2, rd),
    sampled directly rather than by rejection. sech^2(z/z0) is sampled by its
    exact inverse CDF, z = z0 * artanh(u).
    """
    r = rng.gamma(shape=2.0, scale=rd, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    z = z0 * np.arctanh(rng.uniform(-1.0, 1.0, n))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def make_disk(n, seed=42, rd=1.0, z0=0.12, bulge_frac=0.10, bulge_scale=0.010,
              halo_frac=0.07, halo_reach=20.0, r_box=55.0, z_box=18.0,
              periodic=False):
    """Bulge + exponential disc + sech^2 vertical profile + diffuse halo, in a
    box much larger than the galaxy, with an OPEN boundary.

    Calibrated against `test_cases/isolated_galaxy/datafiles` (h_dyn 3.5e3,
    rho_dyn 4.4e10, 2.8% of a 64^3 grid occupied). The three components each
    buy one feature of that measurement:

    * the BULGE sets h_min -- a dense, tiny-scale-length concentration is what
      produces the smallest smoothing lengths in a real galaxy;
    * the DISC sets the bulk of the mass and the thin-slab aspect ratio, which
      is what leaves a uniform grid mostly empty;
    * the HALO sets h_max. It is deliberately sparse: a handful of isolated
      particles at large radius have enormous h, and since petkova's cost goes
      as h^3 those few particles dominate the pair count. On the real galaxy
      h ran to 34.5 against a median of 0.067.

    OPEN boundary by default: the galaxy does not fill its box, so wrapping it
    would be physically wrong and would also bypass the open-wall handling
    this package has separate code for.
    """
    n = int(n)
    rng = np.random.default_rng(seed)
    halfbox = np.array([r_box, r_box, z_box], dtype=np.float64)
    box = 2.0 * halfbox

    nb = int(round(bulge_frac * n))
    nh = int(round(halo_frac * n))
    nd = n - nb - nh

    disc = _sample_inside(
        nd, lambda m, g: _disc_component(m, rd, z0, g), halfbox, rng)
    bulge = _sample_inside(
        nb, lambda m, g: _plummer(m, bulge_scale, g), halfbox, rng)
    # halo: isothermal shell from just outside the disc out to `halo_reach`,
    # flattened, deliberately under-sampled -- this sets h_max
    halo = _sample_inside(
        nh, lambda m, g: _shell(m, 2.0 * rd, halo_reach, g, flatten=0.45),
        halfbox, rng)

    pos = np.concatenate([disc, bulge, halo])
    rng.shuffle(pos)
    if periodic:
        pos = (pos + halfbox) % box - halfbox
    return _finalize(pos, box, periodic)


# --------------------------------------------------------------------------
# merger -- two concentrations in one periodic box
# --------------------------------------------------------------------------

def make_merger(n, seed=42, sep=0.35, rd=0.12, z0=0.018, bulge_frac=0.12,
                bulge_scale=1.9e-4, envelope_frac=0.06, envelope_reach=0.28,
                incl=(0.0, 1.05), box=1.0, periodic=False):
    """Two inclined discs at a separation, plus a shared diffuse envelope, in
    a unit box with an OPEN boundary.

    Open, like `disk`: an isolated merger is not a periodic system, and the
    unit box the real snapshots report is a tipsy units convention, not a
    physical period. Particles beyond the box are RESAMPLED rather than
    wrapped -- the exponential disc has a long tail, and wrapping it would
    place disc material on the far side of the box, which is exactly the
    artefact the open boundary exists to avoid.

    Models `lowres_isolated_merger` / `highres_isolated_merger` (h_dyn 6e3 and
    2.6e5, 15% and 10% of a 64^3 grid occupied). The feature that separates
    this from `disk` is having TWO centres: any method that partitions space
    around a single concentration -- an octree base grid, a Voronoi generator
    subsample -- behaves differently with two, and the merger snapshots are
    the case the POLARIS export actually runs on.

    The discs are given different inclinations because a pair of coplanar
    discs is an unrepresentatively easy case for an axis-aligned octree.
    """
    n = int(n)
    rng = np.random.default_rng(seed)
    box3 = np.full(3, float(box))
    halfbox = 0.5 * box3

    ne = int(round(envelope_frac * n))
    ngal = n - ne
    n1 = ngal // 2
    n2 = ngal - n1

    def one(m, g, incl_i, pa, centre):
        nb = int(round(bulge_frac * m))
        disc = _disc_component(m - nb, rd, z0, g)
        bulge = _plummer(nb, bulge_scale, g)
        return _rotate(np.concatenate([disc, bulge]), incl_i, pa) + centre

    c1 = np.array([-0.5 * sep, 0.0, 0.0])
    c2 = np.array([+0.5 * sep, 0.0, 0.0])
    g1 = _sample_inside(n1, lambda m, g: one(m, g, incl[0], 0.0, c1),
                        halfbox, rng)
    g2 = _sample_inside(n2, lambda m, g: one(m, g, incl[1], 0.7, c2),
                        halfbox, rng)
    env = _sample_inside(
        ne, lambda m, g: _shell(m, 1.5 * rd, envelope_reach, g, flatten=0.8),
        halfbox, rng)

    pos = np.concatenate([g1, g2, env])
    rng.shuffle(pos)
    if periodic:
        pos = (pos + halfbox) % box3 - halfbox
    return _finalize(pos, box3, periodic)


# --------------------------------------------------------------------------
# collapse -- decades of density in one periodic box
# --------------------------------------------------------------------------

def make_collapse(n, seed=42, box=1.0, core_frac=0.85, r_max=0.25,
                  r_min=6.0e-6):
    """Isothermal (rho ~ r^-2) core plus a uniform background, periodic.

    For rho ~ r^-2 the enclosed mass is linear in r, so r = r_max * U samples
    the profile exactly with no rejection step, and the density contrast is
    set by the radius ratio alone. At the defaults that is ~1e9 in rho and
    ~1e3 in h -- deliberately in the range the real snapshots showed rather
    than the mild contrast a naive Plummer sphere would give.

    The point of this family is the h distribution: a core with tiny h next to
    a background with h three decades larger, all in one deposit. That is the
    load-imbalance regime `_balanced_order` exists for.
    """
    n = int(n)
    box3 = np.full(3, float(box))
    rng = np.random.default_rng(seed)
    nc = int(round(core_frac * n))
    nb = n - nc

    r = np.maximum(r_min, r_max * rng.random(nc))
    core = _isotropic(r, rng)
    bg = (rng.random((nb, 3)) - 0.5) * box3

    pos = np.concatenate([core, bg])
    pos = (pos + 0.5 * box3) % box3 - 0.5 * box3
    return _finalize(pos, box3, True)


# --------------------------------------------------------------------------
# cosmo -- halo population + filaments + background
# --------------------------------------------------------------------------

def make_cosmo(n, seed=42, box=1.0, nhalo=40, halo_frac=0.75,
               filament_frac=0.23, bg_frac=0.02, alpha=1.9,
               r_halo_max=0.055, r_halo_min=1.0e-7, fil_radius=0.006,
               halo_conc=1.20):
    """Cosmological-web analogue: a halo population, filaments connecting
    them, and a diffuse background, in a periodic unit box.

    Models `7dwarfcosmo` and `agoracosmomhd` (h_dyn 3.1e4 and 9.0e3, rho_dyn
    1.5e14 and 7.0e11, 16% and 28% of a 64^3 grid occupied). None of the other
    families resemble these: they have structure on EVERY scale at once, with
    most of the volume near-empty but the occupied fraction spread through the
    box rather than concentrated in one place.

    Halo masses follow a power law dN/dM ~ M^-alpha, so a few halos hold most
    of the particles and set h_min while the rest populate the web. Filaments
    are cylinders of particles joining each halo to a near neighbour -- they
    are what lifts the occupied fraction above what isolated halos alone give,
    and they are the reason the cosmological runs occupy 2-10x more cells than
    the merger at comparable N.
    """
    n = int(n)
    box3 = np.full(3, float(box))
    rng = np.random.default_rng(seed)

    centres = (rng.random((nhalo, 3)) - 0.5) * box3
    # power-law masses: w ~ U^(-1/(alpha-1)), normalised to a particle share
    w = rng.random(nhalo) ** (-1.0 / (alpha - 1.0))
    w = np.minimum(w, 60.0)
    w = w / w.sum()

    nh_tot = int(round(halo_frac * n))
    nf_tot = int(round(filament_frac * n))
    nb_tot = n - nh_tot - nf_tot

    # --- halos: isothermal, radius scaling with mass
    chunks = []
    counts = np.maximum(8, (w * nh_tot).astype(int))
    for c, cnt, ww in zip(centres, counts, w):
        rh = r_halo_max * (ww * nhalo) ** (1.0 / 3.0)
        # `halo_conc` > 1 steepens the profile inside the isothermal r = rh*U
        # law. It is the knob that sets h_min: the smallest radius a halo of
        # `cnt` particles reaches is ~rh/cnt^halo_conc, and h_min follows it.
        # The real cosmological runs need it -- a plain r^-2 halo cannot reach
        # their 3e4 dynamic range at any affordable particle count.
        r = np.maximum(r_halo_min, rh * rng.random(int(cnt)) ** halo_conc)
        chunks.append(_isotropic(r, rng) + c)

    # --- filaments: each halo to a near neighbour, cylinder cross-section
    if nf_tot > 0:
        d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        partner = np.argmin(d2, axis=1)
        per_fil = max(4, nf_tot // nhalo)
        for i, j in enumerate(partner):
            a, b = centres[i], centres[j]
            seg = b - a
            seg -= box3 * np.round(seg / box3)         # minimum image
            t = rng.random(per_fil)[:, None]
            axis = a + t * seg
            # isotropic scatter about the axis, tapered towards the ends
            taper = (0.35 + 0.65 * np.sin(np.pi * t))
            off = rng.normal(0.0, fil_radius, (per_fil, 3)) * taper
            chunks.append(axis + off)

    chunks.append((rng.random((max(0, nb_tot), 3)) - 0.5) * box3)

    pos = np.concatenate(chunks)
    if pos.shape[0] > n:                 # trim the power-law overshoot
        pos = pos[rng.permutation(pos.shape[0])[:n]]
    elif pos.shape[0] < n:
        extra = (rng.random((n - pos.shape[0], 3)) - 0.5) * box3
        pos = np.concatenate([pos, extra])
    pos = (pos + 0.5 * box3) % box3 - 0.5 * box3
    return _finalize(pos, box3, True)


# --------------------------------------------------------------------------
# dispatch + reporting
# --------------------------------------------------------------------------

_MAKERS = {
    "uniform": make_uniform,
    "lattice": make_lattice,
    "closepacked": make_closepacked,
    "glass": make_glass,
    "slab1d": make_slab1d,
    "slab2d": make_slab2d,
    "disk": make_disk,
    "merger": make_merger,
    "collapse": make_collapse,
    "cosmo": make_cosmo,
}


def make(family, n, seed=42, **kw):
    """`make("disk", 100000)` -> Particles. See FAMILIES for the names."""
    try:
        fn = _MAKERS[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r}; "
                         f"expected one of {FAMILIES}") from None
    return fn(n, seed=seed, **kw)


def occupancy(p, npx=64):
    """Fraction of an `npx`^3 grid over the box that holds >= 1 particle.

    The headline number for how much of a uniform grid is wasted on vacuum --
    between 0.028 and 0.282 on the real ICs.
    """
    pos = np.asarray(p.pos, dtype=np.float64)
    box = np.asarray(p.box, dtype=np.float64).reshape(3)
    idx = np.floor((pos + 0.5 * box) / box * npx).astype(np.int64)
    idx = np.clip(idx, 0, npx - 1)
    flat = (idx[:, 0] * npx + idx[:, 1]) * npx + idx[:, 2]
    return float(np.unique(flat).size) / float(npx ** 3)


def describe(p, npx=64):
    """The properties that actually predict interpolation cost."""
    h = np.asarray(p.h, dtype=np.float64)
    rho = np.asarray(p.rho, dtype=np.float64)
    box = np.asarray(p.box, dtype=np.float64)
    return {
        "n": int(p.n),
        "box": [float(b) for b in box],
        "periodic": [bool(b) for b in np.atleast_1d(p.periodic)],
        "h_min": float(h.min()), "h_med": float(np.median(h)),
        "h_max": float(h.max()), "h_dyn": float(h.max() / h.min()),
        "rho_min": float(rho.min()), "rho_med": float(np.median(rho)),
        "rho_max": float(rho.max()), "rho_dyn": float(rho.max() / rho.min()),
        "occupancy_64": occupancy(p, npx),
        # sum (4h / cell)^3 at a nominal 64^3 grid: the pair-count proxy that
        # petkova's run time tracks linearly (plan task 14).
        "pair_proxy_64": float(np.sum((4.0 * h / (box.min() / 64.0)) ** 3)),
    }
