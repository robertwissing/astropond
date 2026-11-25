"""Thermal- and cosmic-ray-electron fields for the POLARIS export.

Ideal-MHD gasoline has no electrons and no cosmic rays, but `CMD_SYNCHROTRON`
cannot run without them, and POLARIS has NO command-file defaults for these
fields -- every accessor returns 0 when the id is absent
(`GridBasic.cpp:3937-4000`), so a missing or mis-scaled `n_cr` is a silently
black map rather than an error. This module synthesises them following
Martin-Alvarez et al., SOFIA/SALSA VII (arXiv:2311.06356), which adopts the CR1
model of Reissl et al. (2019); the explicit equations come from the companion
paper arXiv:2409.08437.

Everything here works in and returns **SI** (m^-3, K), because the grid is
written in SI with every POLARIS `conv_*` at 1.0. The one exception is internal:
the pEq model's algebra is cgs (B in Gauss, energies in erg), converted at the
boundary.

Two things worth knowing before changing anything:

* POLARIS's `n_cr` is the TOTAL electron number density integrated over
  [g_min, g_max], not the power-law prefactor N0 -- `Gamma_I_p`
  (`Synchrotron.cpp:583`) carries the 1/(g_min^(1-p) - g_max^(1-p)) that converts
  back to N0. Both models below therefore produce a total number density, and
  `n_cr_from_energy_density` is the general converter for anything specified as
  an energy density.
* The paper's "gamma in [4, 300]" is a KINETIC energy range in units of m_e c^2;
  the Lorentz factors are 5 and 301. POLARIS's spectrum is a power law in gamma,
  so 5 and 301 are what go into ids 25/26 (`GAMMA_MIN`/`GAMMA_MAX` below).
"""

import numpy as np

# --- physical constants -------------------------------------------------------
M_H_KG = 1.6735575e-27              # hydrogen atom mass [kg]
E_E_ERG = 8.1871057769e-7           # m_e c^2 [erg]
E_P_ERG = 1.503277615e-3            # m_p c^2 [erg]  (the pEq spectral break)
M_E_C2_J = 8.1871057769e-14         # m_e c^2 [J]
TESLA_TO_GAUSS = 1.0e4
PERCC_TO_PERM3 = 1.0e6              # cm^-3 -> m^-3

# --- SOFIA VII / CR1 defaults -------------------------------------------------
#: Lorentz-factor bounds (ids 25, 26). See the module note: the paper's [4, 300]
#: is the KINETIC energy in units of m_e c^2, so the Lorentz factors are 5, 301.
GAMMA_MIN = 5.0
GAMMA_MAX = 301.0
#: CR electron spectral index N(gamma) ~ gamma^-p (id 27). 2.2-3.0 spans
#: starburst to aged-loss spectra.
CR_P = 2.3
#: CR1 exponential-disc normalisation [cm^-3], calibrated to the CR electron
#: density at the Milky Way centre and the solar-neighbourhood value (Sun+ 2008).
N_CR0_PERCC = 1.74e-4
#: CR1 disc scales [kpc]. The paper states these are "selected to adjust to the
#: studied galaxy dimensions" -- MEASURE them on our galaxy, do not adopt blindly.
R_GAL_KPC = 8.0
H_GAL_KPC = 1.0
#: pEq proton/electron ratio (SNR shock fronts + Galactic CR data).
K0_PROTON_ELECTRON = 100.0
#: thermal-ionization ramp bounds [K].
T_IONIZED_LO = 1.0e4
T_IONIZED_HI = 2.0e4

IONIZATION_MODES = ("temperature", "constant", "saha", "table")
CR_MODELS = ("exp", "pEq", "density", "constant")


# --------------------------------------------------------------------------- #
#  Thermal electrons -- POLARIS id 22 (n_th) and id 23 (T_e)
# --------------------------------------------------------------------------- #

def hydrogen_number_density(rho_si, mu=1.0):
    """n_H = rho / (mu m_H) [m^-3] from an SI mass density."""
    rho_si = np.asarray(rho_si, dtype=np.float64)
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    return rho_si / (float(mu) * M_H_KG)


def ionization_fraction(T, mode="temperature", xe_const=1.0,
                        t_lo=T_IONIZED_LO, t_hi=T_IONIZED_HI, table=None,
                        n_ref_percc=1.0):
    """Electron fraction x_e in [0, 1].

    SOFIA VII interpolates the gas number density through the CLOUDY-based
    WARPFIELD-EMP tables of Pellegrini et al. (2020); we do not have that table,
    so the default leans on what gasoline DOES give us -- a real gas temperature,
    which is what those tables interpolate around.

    ``temperature`` (default)
        x_e = 0 below `t_lo` (1e4 K), 1 above `t_hi` (2e4 K), linear in log T
        between. The cold-phase mask.
    ``constant``
        x_e = `xe_const` everywhere.
    ``saha``
        hydrogen-only Saha equilibrium at the local T, for when the ramp is too
        crude. Evaluated at the FIXED reference density `n_ref_percc` [cm^-3], so
        it is a pure function of T -- an honest T-only proxy, not a real
        density-dependent Saha (which would need n_H per cell; pass a
        precomputed 2-D table via ``mode="table"`` if that matters).
    ``table``
        hook for a CLOUDY / WARPFIELD-EMP table: pass ``table=(T_nodes, xe_nodes)``
        and it is interpolated in log T.
    """
    T = np.asarray(T, dtype=np.float64)
    if mode not in IONIZATION_MODES:
        raise ValueError(f"unknown ionization mode {mode!r}; "
                         f"choose {IONIZATION_MODES}")
    if mode == "constant":
        if not 0.0 <= xe_const <= 1.0:
            raise ValueError(f"xe_const must be in [0,1], got {xe_const}")
        return np.full(T.shape, float(xe_const))
    if mode == "temperature":
        if not t_hi > t_lo > 0.0:
            raise ValueError(f"need t_hi > t_lo > 0, got {t_lo}, {t_hi}")
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = ((np.log10(np.maximum(T, 1e-30)) - np.log10(t_lo))
                    / (np.log10(t_hi) - np.log10(t_lo)))
        return np.clip(np.where(np.isfinite(frac), frac, 0.0), 0.0, 1.0)
    if mode == "table":
        if table is None:
            raise ValueError("ionization mode 'table' needs table=(T, x_e)")
        Tn, xn = (np.asarray(a, dtype=np.float64) for a in table)
        if Tn.ndim != 1 or Tn.shape != xn.shape or Tn.size < 2:
            raise ValueError("table must be two matching 1-D arrays of length >= 2")
        return np.clip(np.interp(np.log10(np.maximum(T, 1e-30)),
                                 np.log10(Tn), xn), 0.0, 1.0)
    # 'saha': hydrogen-only, x^2/(1-x) = a with a = K(T)/n_ref,
    #   K(T) = 2.41e21 T^1.5 exp(-13.6 eV / kT)   [cm^-3]
    # Solve the quadratic in the CONJUGATE form x = 2a/(a + sqrt(a^2+4a)) rather
    # than (-a + sqrt(a^2+4a))/2: the latter is catastrophic cancellation once
    # a >> 1 (a ~ 1e24 at 1e6 K collapses to exactly 0 in float64, i.e. a fully
    # ionized cell reported as neutral).
    if not n_ref_percc > 0.0:
        raise ValueError(f"n_ref_percc must be positive, got {n_ref_percc}")
    kT_eV = 8.617333262e-5 * np.maximum(T, 1e-30)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        K = (2.41e21 * np.power(np.maximum(T, 1e-30), 1.5)
             * np.exp(-13.6 / kT_eV))
        a = K / float(n_ref_percc)
        x = np.where(a > 0.0, 2.0 * a / (a + np.sqrt(a * a + 4.0 * a)), 0.0)
    return np.clip(np.where(np.isfinite(x), x, 1.0), 0.0, 1.0)


def electrons_from_abundances(rho_si, HI, HeI=None, HeII=None,
                              he_mass_frac=0.236):
    """n_e [m^-3] from gasoline's own non-equilibrium ionization auxes.

    STRICTLY BETTER than any `ionization_fraction` model when the auxes exist,
    and closest in spirit to SOFIA VII (which interpolates CLOUDY/WARPFIELD-EMP
    tables): this is the ionization state the simulation itself evolved
    (`bIonNonEqm = 1`).

    gasoline writes HI / HeI / HeII as MASS fractions of the TOTAL gas mass (the
    HI aux tops out at the hydrogen mass fraction X_H = 1 - `he_mass_frac`, which
    is how you can tell). Free electrons then come from ionized H plus singly and
    doubly ionized He:

        n_e = rho/m_H * [ (X_H - HI) + HeII/4 + 2*(Y - HeI - HeII)/4 ]

    with Y = `he_mass_frac` (gasoline's `dMassFracHelium`). Missing He auxes are
    treated as fully neutral helium, which only removes electrons.
    """
    rho_si = np.asarray(rho_si, dtype=np.float64)
    HI = np.asarray(HI, dtype=np.float64)
    Y = float(he_mass_frac)
    X_H = 1.0 - Y
    if not 0.0 <= Y < 1.0:
        raise ValueError(f"he_mass_frac must be in [0,1), got {Y}")
    HeI = np.zeros_like(rho_si) if HeI is None else np.asarray(HeI, np.float64)
    HeII = np.zeros_like(rho_si) if HeII is None else np.asarray(HeII, np.float64)
    h_ionized = np.clip(X_H - HI, 0.0, X_H)                 # HII mass fraction
    he_iii = np.clip(Y - HeI - HeII, 0.0, Y)                # doubly ionized He
    per_mh = h_ionized + 0.25 * HeII + 0.5 * he_iii         # electrons per m_H
    return rho_si / M_H_KG * per_mh


def thermal_electrons(rho_si, T, mu=1.0, mode="temperature", xe_const=1.0,
                      t_lo=T_IONIZED_LO, t_hi=T_IONIZED_HI, table=None):
    """n_th [m^-3] = x_e(T) * n_H(rho) -- POLARIS id 22.

    SCOPE NOTE: `n_th` drives Faraday rotation/conversion only, never CR emission
    (`CheckSynchrotron`: n_th alone -> "only Faraday RM"). Stokes I is robust to
    this modelling choice; **RM results are not and must carry the caveat.**
    """
    xe = ionization_fraction(T, mode=mode, xe_const=xe_const, t_lo=t_lo,
                             t_hi=t_hi, table=table)
    return hydrogen_number_density(rho_si, mu=mu) * xe


# --------------------------------------------------------------------------- #
#  CR electrons -- POLARIS ids 24-27
# --------------------------------------------------------------------------- #

def cr_exp(x, y, z, n_cr0_percc=N_CR0_PERCC, r_gal_kpc=R_GAL_KPC,
           h_gal_kpc=H_GAL_KPC, length_kpc=1.0, centre=(0.0, 0.0, 0.0)):
    """CR1 / SOFIA VII default: an analytic exponential disc [m^-3].

        n_CR(r, z) = n_CR0 * exp(-r/R_gal) * cosh(z/h_gal)^-2

    `x, y, z` are cell centres in the grid's own length unit; `length_kpc`
    converts that unit to kpc (so `r_gal_kpc`/`h_gal_kpc` stay physical), and
    `centre` is the galaxy centre in the same grid units.

    This is a smooth analytic disc, INDEPENDENT of the local simulated gas --
    SOFIA VII flags that as a known limitation. r and z are galactocentric, so
    the disc centre and orientation must be established first (the caller is
    expected to have rotated the grid coordinates into the disc frame).
    """
    x = np.asarray(x, dtype=np.float64) - float(centre[0])
    y = np.asarray(y, dtype=np.float64) - float(centre[1])
    z = np.asarray(z, dtype=np.float64) - float(centre[2])
    if not (r_gal_kpc > 0.0 and h_gal_kpc > 0.0):
        raise ValueError(f"R_gal and h_gal must be positive, got "
                         f"{r_gal_kpc}, {h_gal_kpc}")
    r_kpc = np.sqrt(x * x + y * y) * float(length_kpc)
    z_kpc = z * float(length_kpc)
    return (float(n_cr0_percc) * PERCC_TO_PERM3
            * np.exp(-r_kpc / float(r_gal_kpc))
            / np.cosh(z_kpc / float(h_gal_kpc)) ** 2)


def cr_peq(b_tesla, p=CR_P, k0=K0_PROTON_ELECTRON, gamma_min=GAMMA_MIN,
           gamma_max=GAMMA_MAX):
    """Proton-equipartition CR electron density [m^-3] -- the companion paper's
    Eq. 8, re-derived from their Eqs. 2-7:

        n_CR,pEq = B^2 (p-2) E_p^(p-2) (E_min^(1-p) - E_max^(1-p))
                   / [4 pi K0 p (p-1)]

    cgs throughout (B in Gauss, energies in erg, eps_B = B^2/8pi); `b_tesla` is
    converted on the way in and the result on the way out. `E_min`/`E_max` are
    the KINETIC bounds (gamma_min - 1) m_e c^2 and (gamma_max - 1) m_e c^2, i.e.
    the paper's 4 E_e and 300 E_e for the default gamma range.

    Derivation chain (the E0 pivot cancels identically -- a unit test):
      eps_B = B^2/8pi = eps_p = n_p0 E0^2 (E0/E_p)^(p-2) (1/2 + 1/(p-2))
      n_e0 = n_p0 / K0
      n_CR,e = n_e0 E0^p (E_min^(1-p) - E_max^(1-p)) / (p-1)

    Unlike `cr_exp` this ties the CR electrons to the SIMULATED field, which is
    the more interesting model for our purposes (and the subject of the companion
    paper) -- but note it scales as B^2, so it inherits the capture's magnetic
    energy loss quadratically.
    """
    p = float(p)
    if not p > 2.0:
        raise ValueError(f"pEq needs p > 2 (the energy integral diverges "
                         f"otherwise), got {p}")
    if not k0 > 0.0:
        raise ValueError(f"K0 must be positive, got {k0}")
    if not gamma_max > gamma_min > 1.0:
        raise ValueError(f"need gamma_max > gamma_min > 1, got "
                         f"{gamma_min}, {gamma_max}")
    b_gauss = np.abs(np.asarray(b_tesla, dtype=np.float64)) * TESLA_TO_GAUSS
    e_min = (float(gamma_min) - 1.0) * E_E_ERG
    e_max = (float(gamma_max) - 1.0) * E_E_ERG
    spec = e_min ** (1.0 - p) - e_max ** (1.0 - p)
    n_percc = (b_gauss ** 2 * (p - 2.0) * E_P_ERG ** (p - 2.0) * spec
               / (4.0 * np.pi * float(k0) * p * (p - 1.0)))
    return n_percc * PERCC_TO_PERM3


def n_cr_from_energy_density(u_cr_si, p=CR_P, gamma_min=GAMMA_MIN,
                            gamma_max=GAMMA_MAX):
    """Total CR electron number density [m^-3] from an energy density [J/m^3].

    The general converter for any model specified as an energy density (e.g. a
    plain zeta * B^2/2mu0 equipartition), since POLARIS wants a total number
    density:

        n_cr = (u_cr / m_e c^2) * (p-2)/(p-1)
               * (g_min^(1-p) - g_max^(1-p)) / (g_min^(2-p) - g_max^(2-p))
    """
    p = float(p)
    if not p > 2.0:
        raise ValueError(f"need p > 2, got {p}")
    u = np.asarray(u_cr_si, dtype=np.float64)
    gmin, gmax = float(gamma_min), float(gamma_max)
    num = gmin ** (1.0 - p) - gmax ** (1.0 - p)
    den = gmin ** (2.0 - p) - gmax ** (2.0 - p)
    return (u / M_E_C2_J) * (p - 2.0) / (p - 1.0) * (num / den)


def cr_mask(n_cr, rho_si=None, T=None, min_density=None, max_temp=None,
            min_number_density=None, mu=1.0):
    """Zero `n_cr` outside the gas the model is meant to describe.

    The `exp` disc in particular is analytic and will happily emit in the halo.
    `min_density` [kg/m^3] / `min_number_density` [m^-3] keep only gas denser
    than the threshold; `max_temp` [K] drops hot phases. Empty octree cells must
    carry a genuine zero, not a floor -- so this multiplies rather than clips.
    """
    n = np.array(n_cr, dtype=np.float64, copy=True)
    keep = np.ones(n.shape, dtype=bool)
    if min_density is not None:
        if rho_si is None:
            raise ValueError("cr_mask(min_density=...) needs rho_si")
        keep &= np.asarray(rho_si, dtype=np.float64) >= float(min_density)
    if min_number_density is not None:
        if rho_si is None:
            raise ValueError("cr_mask(min_number_density=...) needs rho_si")
        keep &= (hydrogen_number_density(rho_si, mu=mu)
                 >= float(min_number_density))
    if max_temp is not None:
        if T is None:
            raise ValueError("cr_mask(max_temp=...) needs T")
        keep &= np.asarray(T, dtype=np.float64) <= float(max_temp)
    n[~keep] = 0.0
    return n


def electron_fields(centres, rho_si, T, b_tesla=None, mu=1.0,
                    cr_model="exp", ionization="temperature",
                    xe_const=1.0, t_lo=T_IONIZED_LO, t_hi=T_IONIZED_HI,
                    ion_table=None,
                    n_cr0_percc=N_CR0_PERCC, r_gal_kpc=R_GAL_KPC,
                    h_gal_kpc=H_GAL_KPC, length_kpc=1.0, centre=(0.0, 0.0, 0.0),
                    cr_p=CR_P, k0=K0_PROTON_ELECTRON,
                    gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX,
                    cr_frac=None, cr_const=None,
                    cr_mask_density=None, cr_mask_temp=None,
                    cr_mask_number_density=None, n_th_override=None):
    """All six POLARIS electron fields for a cell list, in SI.

    Returns a dict ready to merge into the export's field table:
    ``n_th`` (22), ``T_e`` (23), ``n_cr`` (24), ``g_min`` (25), ``g_max`` (26),
    ``cr_p`` (27). g_min/g_max/cr_p are written as full per-cell arrays because
    POLARIS treats them as per-cell fields, even though they are constant here.

    centres : (Ncell, 3) cell centres in the grid's length unit, already in the
              disc frame (the `exp` model's r, z are galactocentric).
    b_tesla : (Ncell, 3) or (Ncell,) |B| in Tesla -- required by `cr_model="pEq"`.

    All fields are asserted finite and non-negative before returning: POLARIS
    would ray-trace a NaN silently.
    """
    if cr_model not in CR_MODELS:
        raise ValueError(f"unknown cr_model {cr_model!r}; choose {CR_MODELS}")
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("centres must be (Ncell, 3)")
    rho_si = np.asarray(rho_si, dtype=np.float64).reshape(-1)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    n = centres.shape[0]
    if rho_si.size != n or T.size != n:
        raise ValueError(f"rho_si/T must have one value per cell ({n})")

    if n_th_override is not None:
        # the run's own ionization auxes (see electrons_from_abundances) beat any
        # model here, so the driver passes n_e straight through when it has them
        n_th = np.asarray(n_th_override, dtype=np.float64).reshape(-1)
        if n_th.size != n:
            raise ValueError(f"n_th_override must have one value per cell ({n})")
    else:
        n_th = thermal_electrons(rho_si, T, mu=mu, mode=ionization,
                                 xe_const=xe_const, t_lo=t_lo, t_hi=t_hi,
                                 table=ion_table)

    if cr_model == "exp":
        n_cr = cr_exp(centres[:, 0], centres[:, 1], centres[:, 2],
                      n_cr0_percc=n_cr0_percc, r_gal_kpc=r_gal_kpc,
                      h_gal_kpc=h_gal_kpc, length_kpc=length_kpc, centre=centre)
    elif cr_model == "pEq":
        if b_tesla is None:
            raise ValueError("cr_model='pEq' needs b_tesla (it scales as B^2)")
        b = np.asarray(b_tesla, dtype=np.float64)
        bmag = np.linalg.norm(b, axis=1) if b.ndim == 2 else np.abs(b.reshape(-1))
        if bmag.size != n:
            raise ValueError(f"b_tesla must have one value/vector per cell ({n})")
        n_cr = cr_peq(bmag, p=cr_p, k0=k0, gamma_min=gamma_min,
                      gamma_max=gamma_max)
    elif cr_model == "density":
        if cr_frac is None:
            raise ValueError("cr_model='density' needs cr_frac")
        n_cr = float(cr_frac) * hydrogen_number_density(rho_si, mu=mu)
    else:                                    # 'constant'
        if cr_const is None:
            raise ValueError("cr_model='constant' needs cr_const [m^-3]")
        n_cr = np.full(n, float(cr_const))

    n_cr = cr_mask(n_cr, rho_si=rho_si, T=T, min_density=cr_mask_density,
                   max_temp=cr_mask_temp,
                   min_number_density=cr_mask_number_density, mu=mu)

    out = {
        "n_th": n_th,
        "T_e": T.copy(),                     # id 23 = the gas temperature
        "n_cr": n_cr,
        "g_min": np.full(n, float(gamma_min)),
        "g_max": np.full(n, float(gamma_max)),
        "cr_p": np.full(n, float(cr_p)),
    }
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"electron field {k!r} has non-finite values; "
                             "POLARIS would ray-trace these silently")
        if np.any(v < 0.0):
            raise ValueError(f"electron field {k!r} has negative values")
    return out


def describe(fields, model_info=""):
    """One-line-per-field summary for the run log / sidecar."""
    lines = [f"electron fields{' (' + model_info + ')' if model_info else ''}:"]
    for k in ("n_th", "T_e", "n_cr", "g_min", "g_max", "cr_p"):
        v = fields[k]
        nz = int(np.count_nonzero(v))
        lines.append(f"  {k:>6}: min {v.min():.4e} max {v.max():.4e} "
                     f"nonzero {nz}/{v.size}")
    return "\n".join(lines)
