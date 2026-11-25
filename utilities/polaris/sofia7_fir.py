"""Far-infrared polarized dust emission, the SOFIA VII way.

**POLARIS is not involved anywhere in this module.** Martin-Alvarez et al.
(SALSA VII, arXiv:2311.06356) use POLARIS only for the radio synchrotron; their
FIR polarimetry is an analytic line-of-sight integral, their §3.1 Eqs. 4-6, with
no radiative transfer, no grain model, no dust temperature and no wavelength.
This module reproduces exactly that, which is why it is NOT called
`polaris_fir.py` -- in this repo the `polaris_` prefix means "feeds or reads the
POLARIS binary". (POLARIS *can* do real dust polarimetry with RAT alignment; this
module does not use it.)

Their RAMSES run tracks no dust, so dust is assigned in post-processing as a
fixed fraction of the metals:

    n_dust = rho_gas * Z * eta_D/M / m_dust * f_cut(T)
    f_cut(T) = min(1, exp[1 - T / 1500 K])            (sublimation)

and the Stokes maps are the Fiege-Pudritz / Planck-XX geometric formalism:

    I = \\int n_dust ( 1 - p0 [ (Bx^2 + By^2)/B^2 - 2/3 ] ) dl
    Q = \\int p0 n_dust (By^2 - Bx^2)/B^2 dl
    U = \\int p0 n_dust (2 Bx By)/B^2 dl

with x,y the plane of sky and p0 = 0.25 (Planck's maximum observed polarization
fraction) standing in for all grain-alignment physics. Two properties of that
form are worth stating because they are what the tests check:

  * `B^2` in the denominator is the FULL 3-D field, so a field along the line of
    sight gives Q = U = 0 identically -- the geometric depolarization is built in
    rather than bolted on;
  * the polarization is PERPENDICULAR to B, as thermal dust emission must be:
    B along the plane-of-sky x axis gives Q = -p0 n_dust, U = 0, i.e. a
    polarization angle of 90 deg. The `-2/3` in I keeps the emission normalised
    when averaged over field orientations.

**The sign of U is a frame handedness convention, and it is NOT free.** Taking
the paper's printed `U = +2 Bx By / B^2` together with `psi = 0.5 atan2(U, Q)`
in an ordinary right-handed frame gives `psi = 90 deg - a` for a field at angle
`a` -- a REFLECTION of the field direction, not a rotation of it. It happens to
agree with "perpendicular to B" at a = 0 and a = 90 deg and disagrees everywhere
else (at 45 deg it returns 45 deg, i.e. PARALLEL to B). That is the signature of
a left-handed plane-of-sky frame, which is what astronomers actually use: position
angle runs North through East, and maps are drawn with East to the LEFT. The
paper's form is therefore correct in its own frame and wrong in ours.

Since our grid axes are right-handed and the figures are ours, `u_sign` defaults
to **-1**, which makes the polarization genuinely perpendicular to B_perp at every
angle in our frame. Pass `u_sign=+1` to reproduce the paper's expression
literally. `I` and `Q` are unaffected by handedness -- only `U` flips.

The maps are surface-brightness-like in ARBITRARY units: the model has no
emissivity and no wavelength, so only ratios (the polarization fraction) and
morphology are physical. That is equally true of the source paper.

Units: `rho_si` in kg/m^3 and `m_dust_kg` in kg give `n_dust` in m^-3, and the
projection multiplies by a path length in metres, so the maps are m^-2. B may be
in any unit -- it only ever enters as a ratio to |B|.
"""

import numpy as np

# --- SOFIA VII §3.1 defaults --------------------------------------------------
#: dust-to-METAL mass ratio (not dust-to-gas). Their eta_D/M.
ETA_DUST_TO_METAL = 0.4
#: their single grain: a = 0.1 um, rho_grain = 3 g/cm^3 -> m = 1.26e-14 g.
M_DUST_KG = 1.26e-17
#: maximum polarization fraction, from Planck. Stands in for grain alignment.
P0_FIR = 0.25
#: dust sublimation temperature in f_cut [K]. NOTE POLARIS's own silicate
#: catalogue declares 1200 K; do not apply both cutoffs with POLARIS dust RT.
T_SUBLIMATION_K = 1500.0

#: cyclic, right-handed (plane-of-sky-1, plane-of-sky-2, line-of-sight) triples.
_LOS_AXES = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}
#: sign of U. -1 = our right-handed grid frame, in which the polarization comes
#: out perpendicular to B at every angle. +1 = the paper's printed form, which is
#: the same physics in a left-handed (East-left) sky frame. See the module note.
U_SIGN_RIGHT_HANDED = -1.0
U_SIGN_PAPER = +1.0


# --------------------------------------------------------------------------- #
#  The dust field
# --------------------------------------------------------------------------- #

def dust_cutoff(T, t_sublim=T_SUBLIMATION_K):
    """f_cut(T) = min(1, exp[1 - T/T_sub]) -- removes dust from hot gas.

    Equals 1 for T <= T_sub and falls exponentially above it, so it is a soft
    sublimation cutoff rather than a step.
    """
    T = np.asarray(T, dtype=np.float64)
    if t_sublim <= 0.0:
        raise ValueError(f"t_sublim must be positive, got {t_sublim}")
    return np.minimum(1.0, np.exp(1.0 - T / float(t_sublim)))


def dust_number_density(rho_si, metallicity, T=None,
                        eta_dm=ETA_DUST_TO_METAL, m_dust_kg=M_DUST_KG,
                        t_sublim=T_SUBLIMATION_K):
    """n_dust [m^-3] = rho * Z * eta_D/M / m_dust * f_cut(T).

    `metallicity` is the gas metal MASS FRACTION (tipsy gas column 10), not a
    metallicity relative to solar. `T` may be None to skip the sublimation cut.
    """
    rho_si = np.asarray(rho_si, dtype=np.float64)
    Z = np.asarray(metallicity, dtype=np.float64)
    if m_dust_kg <= 0.0:
        raise ValueError(f"m_dust_kg must be positive, got {m_dust_kg}")
    if eta_dm < 0.0:
        raise ValueError(f"eta_dm must be non-negative, got {eta_dm}")
    if np.any(rho_si < 0.0) or np.any(Z < 0.0):
        raise ValueError("rho and metallicity must be non-negative")
    n = rho_si * Z * float(eta_dm) / float(m_dust_kg)
    if T is not None:
        n = n * dust_cutoff(T, t_sublim)
    return n


# --------------------------------------------------------------------------- #
#  The per-cell Stokes integrands (SOFIA VII Eqs. 4-6)
# --------------------------------------------------------------------------- #

def fir_integrands(n_dust, B, los="z", p0=P0_FIR,
                   u_sign=U_SIGN_RIGHT_HANDED):
    """Per-cell (i, q, u) integrands -- the bracketed parts of Eqs. 4-6.

    `B` is (N,3) in ANY unit (it enters only as a ratio to |B|). Cells with
    |B| = 0 contribute unpolarized emission with the isotropic `2/3` value, which
    is the only continuous choice: there is no orientation to polarize along.

    `u_sign` selects the plane-of-sky handedness; the default makes the
    polarization perpendicular to B in our right-handed grid frame. See the
    module docstring -- this is a real choice, not a cosmetic one.

    Returns three (N,) arrays; multiply by a path length and sum to get I, Q, U.
    """
    n_dust = np.asarray(n_dust, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError(f"B must be (N,3), got {B.shape}")
    if B.shape[0] != n_dust.shape[0]:
        raise ValueError(f"n_dust has {n_dust.shape[0]} rows, B has {B.shape[0]}")
    if los not in _LOS_AXES:
        raise ValueError(f"los must be one of {sorted(_LOS_AXES)}, got {los!r}")

    a1, a2, _ = _LOS_AXES[los]
    b1, b2 = B[:, a1], B[:, a2]
    b2sum = np.sum(B * B, axis=1)
    # |B| = 0 -> no preferred direction: fall back to the isotropic average, for
    # which the bracket in I vanishes and Q = U = 0.
    ok = b2sum > 0.0
    inv = np.zeros_like(b2sum)
    np.divide(1.0, b2sum, out=inv, where=ok)

    perp = (b1 * b1 + b2 * b2) * inv          # (Bx^2+By^2)/B^2, 0 where |B|=0
    perp = np.where(ok, perp, 2.0 / 3.0)      # keeps the I bracket at zero
    i = n_dust * (1.0 - p0 * (perp - 2.0 / 3.0))
    q = p0 * n_dust * (b2 * b2 - b1 * b1) * inv
    u = float(u_sign) * p0 * n_dust * (2.0 * b1 * b2) * inv
    return i, q, u


# --------------------------------------------------------------------------- #
#  Conservative projection of an AMR leaf field
# --------------------------------------------------------------------------- #

def project_amr(grid, values, los="z", npix=512):
    """Line-of-sight integral of a per-leaf quantity -> an (npix, npix) map.

    EXACT, not interpolated. The leaves are axis-aligned cubes that tile a cubic
    domain, and the view is down a principal axis, so every leaf projects to an
    exact square in the image plane. The map value at a pixel is the pixel-area
    average of `\\int values dl`, accumulated one refinement level at a time:

      * a leaf COARSER than a pixel covers an integer block of pixels, and every
        one of them receives `value * cell_size` (the column through the leaf is
        uniform);
      * a leaf FINER than a pixel covers 1/k^2 of it, so it contributes
        `value * cell_size / k^2`.

    Both branches are exact because `npix` is required to be `base_npx * 2**j`,
    which makes every level commensurate with the pixel grid. Summing the levels
    then reconstructs the full column, since the leaves are space-filling.

    `values` may be (N,) or (N, M) -- the M variant projects M fields in one
    pass, returning (npix, npix, M).
    """
    lo = np.asarray(grid.lo, dtype=np.float64)
    hi = np.asarray(grid.hi, dtype=np.float64)
    side = hi - lo
    if not np.allclose(side, side[0], rtol=1e-9):
        raise ValueError(f"project_amr needs a CUBIC domain, got side {side}")
    base = tuple(int(b) for b in grid.base_npx)
    if len(set(base)) != 1:
        raise ValueError(f"base_npx must be equal on all axes, got {base}")
    nb = base[0]
    if nb & (nb - 1):
        raise ValueError(f"base_npx must be a power of two, got {nb}")
    npix = int(npix)
    if npix < nb or (npix // nb) * nb != npix or (npix // nb) & (npix // nb - 1):
        raise ValueError(f"npix must be base_npx * 2**j (base_npx={nb}), got {npix}")

    vals = np.asarray(values, dtype=np.float64)
    squeeze = vals.ndim == 1
    if squeeze:
        vals = vals[:, None]
    if vals.shape[0] != grid.cell_lo.shape[0]:
        raise ValueError(f"values has {vals.shape[0]} rows, grid has "
                         f"{grid.cell_lo.shape[0]} leaves")
    nf = vals.shape[1]

    a1, a2, _ = _LOS_AXES[los]
    L = float(side[0])
    out = np.zeros((npix, npix, nf), dtype=np.float64)
    lvl = np.asarray(grid.cell_level).astype(np.int64)

    for level in np.unique(lvl):
        m = lvl == level
        n_l = nb * (1 << int(level))                 # cells per axis at this level
        s = L / n_l                                  # cell edge == path length
        i = np.floor((grid.cell_lo[m, a1] - lo[a1]) / s + 0.5).astype(np.int64)
        j = np.floor((grid.cell_lo[m, a2] - lo[a2]) / s + 0.5).astype(np.int64)
        np.clip(i, 0, n_l - 1, out=i)
        np.clip(j, 0, n_l - 1, out=j)
        contrib = vals[m] * s

        if n_l <= npix:
            k = npix // n_l
            acc = np.zeros((n_l, n_l, nf), dtype=np.float64)
            np.add.at(acc, (i, j), contrib)
            out += np.repeat(np.repeat(acc, k, axis=0), k, axis=1)
        else:
            k = n_l // npix
            np.add.at(out, (i // k, j // k), contrib / (k * k))

    return out[:, :, 0] if squeeze else out


# --------------------------------------------------------------------------- #
#  The maps
# --------------------------------------------------------------------------- #

def fir_maps(grid, n_dust, B, los="z", npix=512, p0=P0_FIR,
             u_sign=U_SIGN_RIGHT_HANDED):
    """Full FIR Stokes maps for one viewing axis.

    Returns a dict with `I`, `Q`, `U`, `P` (= sqrt(Q^2+U^2)), `p` (= P/I, the
    polarization fraction), `psi` (the polarization angle [rad], measured in the
    plane of sky from the first axis) and `chi` (the inferred B orientation,
    psi + 90 deg, wrapped into [-pi/2, pi/2)) -- plus `los`, `npix` and `p0`.

    `chi` is what to draw as segments on a figure: it is the plane-of-sky field
    direction, and with the default `u_sign` it really is that in the grid frame.
    """
    i, q, u = fir_integrands(n_dust, B, los=los, p0=p0, u_sign=u_sign)
    maps = project_amr(grid, np.stack([i, q, u], axis=1), los=los, npix=npix)
    I, Q, U = maps[:, :, 0], maps[:, :, 1], maps[:, :, 2]
    P = np.hypot(Q, U)
    p = np.zeros_like(I)
    np.divide(P, I, out=p, where=I > 0.0)
    psi = 0.5 * np.arctan2(U, Q)
    chi = (psi + 0.5 * np.pi + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return {"I": I, "Q": Q, "U": U, "P": P, "p": p, "psi": psi, "chi": chi,
            "los": los, "npix": int(npix), "p0": float(p0)}


def describe(maps, label=""):
    """One-line-per-quantity summary, matching `polaris_electrons.describe`.

    The polarization median is quoted over pixels with **I > 0**, and the pixel
    count is printed with it. Which mask is used matters a lot here -- medianing
    over `p > 0` instead silently drops every unpolarized pixel and reported
    22.3% where the I > 0 median is 14.9% on the same map -- so it is stated
    rather than left to the reader.
    """
    head = f"FIR maps {label}".rstrip()
    out = [f"{head}  (los={maps['los']}, {maps['npix']}^2, p0={maps['p0']:g})"]
    I = np.asarray(maps["I"], dtype=np.float64)
    m = I > 0
    n = int(m.sum())
    out.append(f"  I      total {float(I.sum()):.4e}   peak {float(I.max()):.4e}"
               f" [m^-2]   {n:,} lit pixels")
    p = np.asarray(maps["p"], dtype=np.float64)
    med = float(np.median(p[m])) if n else float("nan")
    out.append(f"  P/I    median {med * 100:.2f}% (over I>0)"
               f"   peak {float(p.max()) * 100:.2f}%"
               f"   (ceiling {maps['p0'] / (1.0 - maps['p0'] / 3.0) * 100:.2f}%)")
    return "\n".join(out)
