"""gasoline/tipsy code units -> SI, for the POLARIS export.

POLARIS applies the command file's `conv_dens`/`conv_len`/`conv_mag`/`conv_vel`
to whatever is in the grid file -- and `conv_dens` multiplies the gas density,
`n_th` AND `n_cr` with a single factor (`GridBasic.cpp:302,442,465`). Three
quantities, one knob, so the only self-consistent choice is to write everything
already in SI and leave every `conv_*` at 1.0. That puts the whole unit
derivation here, in Python, where it can be tested -- which is the point of this
module.

The gasoline unit system is G = 1, fixed entirely by `dMsolUnit` (M, in Msol) and
`dKpcUnit` (L, in kpc). This module does NOT re-derive it: `code_units_from` in
`IC_analysis_general` is the single source of truth for the cgs factors, and
everything below is the cgs->SI step plus the POLARIS field mapping.

The magnetic unit is the trap. The snapshot `BField` is in the internal
ideal-MHD Alfven unit (`1 code B = v_cgs * sqrt(4 pi rho_cgs)` Gauss), NOT Gauss,
even when the IC was specified in Gauss. Two independent implementations in this
suite agree on it -- `code_units`' `b_gauss` and `readtipsy`'s
`CodetoGauss = sqrt(4 pi M / L) / t` are algebraically identical -- and
`_test_polaris_units.py` pins that agreement.

Usage:

    u = polaris_units(snap="run/galaxy.00100")        # from the sibling .param
    u = polaris_units(dMsolUnit=1e10, dKpcUnit=1.0)   # or explicitly
    to_polaris(res, path, fields=["rho", "Bx", "By", "Bz"],
               units=u.field_units(["rho", "Bx", "By", "Bz"]),
               length_scale=u.length_m)
"""

import math
from dataclasses import dataclass, field as _dc_field

import os as _os, sys as _sys  # noqa: E402
# sph_interp / readtipsy / IC_* live in setupfiles/; this file sits in
# utilities/polaris/, so put setupfiles on the path BEFORE importing them
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'setupfiles')))

from IC_analysis_general import code_units_from, read_log_param

#: cgs -> SI
CM_TO_M = 1.0e-2
GCC_TO_KGM3 = 1.0e3                     # g/cm^3 -> kg/m^3
CMS_TO_MS = 1.0e-2                      # cm/s   -> m/s
GAUSS_TO_TESLA = 1.0e-4                 # Gauss  -> Tesla (POLARIS wants Tesla)

#: hydrogen atom mass [kg]. Used only for mass density -> number density; the
#: suite's dGasConst path uses a rounder 1.67e-24 g, a 0.2% difference that does
#: not enter here (the tipsy thermal column is already a temperature in K).
M_H_KG = 1.6735575e-27

#: field name -> the `PolarisUnits` factor that converts it from code units.
#: `gas_dens`/`n_gas` are POLARIS's NUMBER density (id 0): the factor turns a code
#: MASS density into m^-3, so feed it the same rho field. Temperatures are already
#: in K in tipsy, and the electron/CR fields are built directly in SI by the
#: physics layer, so both are 1.0.
POLARIS_FIELD_UNITS = {
    "rho": "density_kgm3", "gas_mdens": "density_kgm3",
    "gas_dens": "number_density_m3", "n_gas": "number_density_m3",
    "T": "one", "gas_temp": "one", "T_e": "one",
    "vx": "velocity_ms", "vy": "velocity_ms", "vz": "velocity_ms",
    "Bx": "b_tesla", "By": "b_tesla", "Bz": "b_tesla",
    "mx": "b_tesla", "my": "b_tesla", "mz": "b_tesla",
    "n_th": "one", "n_cr": "one",
    "g_min": "one", "g_max": "one", "cr_p": "one",
    "cr_gamma_min": "one", "cr_gamma_max": "one",
}


@dataclass(frozen=True)
class PolarisUnits:
    """Multiplicative code -> SI factors, plus the provenance for the sidecar."""
    dMsolUnit: float
    dKpcUnit: float
    mu: float                            # dMeanMolWeight
    gamma: float                         # dConstGamma
    source: str                          # where the unit values came from
    length_m: float
    density_kgm3: float
    number_density_m3: float             # code MASS density -> n_total [m^-3]
    velocity_ms: float
    b_tesla: float
    time_s: float
    energy_per_mass_si: float            # erg/g -> J/kg
    b_gauss: float                       # the intermediate, for cross-checks
    one: float = 1.0                     # identity, for already-SI fields
    warnings: tuple = _dc_field(default_factory=tuple)

    def field_units(self, fields):
        """The `units=` dict for `to_polaris`, for exactly `fields`.

        Raises on a field with no known unit rather than silently writing it
        unscaled -- an unscaled density is a wrong map, not a warning.
        """
        out = {}
        unknown = []
        for f in fields:
            attr = POLARIS_FIELD_UNITS.get(f)
            if attr is None:
                unknown.append(f)
                continue
            out[f] = float(getattr(self, attr))
        if unknown:
            raise KeyError(
                f"no code->SI factor known for field(s) {unknown}; add them to "
                f"POLARIS_FIELD_UNITS (known: {sorted(POLARIS_FIELD_UNITS)})")
        return out

    def as_dict(self):
        """Flat dict for the `<out>.polaris.json` sidecar."""
        d = {k: getattr(self, k) for k in
             ("dMsolUnit", "dKpcUnit", "mu", "gamma", "source", "length_m",
              "density_kgm3", "number_density_m3", "velocity_ms", "b_tesla",
              "time_s", "energy_per_mass_si", "b_gauss")}
        d["warnings"] = list(self.warnings)
        d["conv_note"] = ("grid written in SI; every POLARIS conv_* must be 1.0 "
                          "(conv_dens is shared by gas_dens, n_th and n_cr)")
        return d

    def describe(self):
        return (f"units from {self.source}: dMsolUnit={self.dMsolUnit:g}, "
                f"dKpcUnit={self.dKpcUnit:g}, mu={self.mu:g}, "
                f"gamma={self.gamma:g}\n"
                f"  1 code length   = {self.length_m:.6e} m "
                f"({self.dKpcUnit:g} kpc)\n"
                f"  1 code density  = {self.density_kgm3:.6e} kg/m^3 "
                f"-> n = {self.number_density_m3:.6e} m^-3 (mu={self.mu:g})\n"
                f"  1 code velocity = {self.velocity_ms:.6e} m/s\n"
                f"  1 code B        = {self.b_gauss:.6e} G "
                f"= {self.b_tesla:.6e} T   <- Alfven unit, NOT Gauss")


def read_unit_params(snap):
    """Unit parameters for a snapshot: `.param` first, then the run `.log`.

    Returns (values, source) where `values` holds whichever of dMsolUnit,
    dKpcUnit, dMeanMolWeight, dConstGamma were found (missing ones absent).
    """
    import os
    from sph_interp.particles import _param_dict

    found, src = {}, []
    vals = _param_dict(snap)
    if vals:
        for key in ("dMsolUnit", "dKpcUnit", "dMeanMolWeight", "dConstGamma",
                    "bComove"):
            try:
                found[key] = float(vals[key])
            except (KeyError, ValueError, TypeError):
                pass
        if found:
            import re
            src.append(os.path.basename(re.sub(r"\.\d+$", "", snap) + ".param"))
    if ("dMsolUnit" not in found or "dKpcUnit" not in found
            or "bComove" not in found):
        run = os.path.dirname(snap) or "."
        for key in ("dMsolUnit", "dKpcUnit", "dMeanMolWeight", "dConstGamma",
                    "bComove"):
            if key in found:
                continue
            v = read_log_param(run, key)
            if v is not None:
                found[key] = float(v)
                if ".log" not in "".join(src):
                    src.append(".log")
    return found, (" + ".join(src) if src else "not found")


def scale_factor(snap, header_time):
    """The scale factor `a` for a snapshot, from the tipsy header time.

    **The tipsy header time is only a scale factor when the run is COMOVING.**
    In an isolated run (`bComove = 0`) it is a TIME in code units, and reading it
    as `a` silently rescales every length -- by 150x on the highres merger, whose
    header time is 150.251. So this consults `bComove` and returns exactly 1.0
    for a non-comoving run.

    Returns (a, comoving, source_note).
    """
    params, src = read_unit_params(snap)
    bc = params.get("bComove")
    if bc is None:
        # No bComove anywhere. A header time of exactly 1 or a plausible scale
        # factor in (0,1] is ambiguous; anything above 1 cannot be a scale factor.
        t = float(header_time)
        if t > 1.0 + 1e-9:
            return 1.0, False, (f"bComove not found; header time {t:g} > 1 "
                                "cannot be a scale factor -> treating as "
                                "NON-comoving (a = 1)")
        return t, True, (f"bComove not found; assuming COMOVING with a = {t:g} "
                         f"(from {src}). Verify -- if this run is isolated, "
                         "every length is wrong by 1/a.")
    if int(bc) == 0:
        return 1.0, False, f"bComove = 0 ({src}): isolated run, a = 1"
    return float(header_time), True, f"bComove = 1 ({src}): comoving, a = header time"


def polaris_units(dMsolUnit=None, dKpcUnit=None, mu=None, gamma=None,
                  snap=None):
    """Code -> SI factors for the POLARIS export.

    dMsolUnit, dKpcUnit : the gasoline G=1 unit pair. Explicit values win;
        otherwise they are read from `snap`'s sibling `.param`, then the run
        `.log`. There is no default -- nothing on disk records them for the
        shipped isolated_galaxy snapshot, and guessing would silently rescale
        every map.
    mu    : dMeanMolWeight, used only for mass density -> number density.
        Defaults to the `.param` value, else 1.0 WITH a recorded warning: the
        suite's readers hard-code 1.0 and a different mu silently scales both the
        derived temperature and `n_H`.
    gamma : dConstGamma, carried for the sidecar (no POLARIS field needs it).
    """
    found, source = ({}, "explicit")
    if snap is not None and (dMsolUnit is None or dKpcUnit is None
                            or mu is None or gamma is None):
        found, source = read_unit_params(snap)

    def pick(name, explicit):
        return float(explicit) if explicit is not None else found.get(name)

    msol = pick("dMsolUnit", dMsolUnit)
    kpc = pick("dKpcUnit", dKpcUnit)
    if msol is None or kpc is None:
        raise ValueError(
            "dMsolUnit / dKpcUnit are required and were not found"
            + (f" in {source}" if snap is not None else "")
            + ". Nothing on disk records them for the shipped isolated_galaxy "
              "snapshot; pass them explicitly -- "
              "every SI factor is a no-op without them.")
    if dMsolUnit is not None and dKpcUnit is not None and snap is None:
        source = "explicit"

    warnings = []
    mu_v = pick("dMeanMolWeight", mu)
    if mu_v is None or mu_v <= 0.0:
        mu_v = 1.0
        warnings.append(
            "dMeanMolWeight not found; using mu = 1.0 (the value the suite's "
            "readers hard-code). A different mu scales n_H, and the tipsy "
            "temperature column, by that factor.")
    gamma_v = pick("dConstGamma", gamma)
    if gamma_v is None or gamma_v <= 1.0:
        gamma_v = 5.0 / 3.0
        warnings.append("dConstGamma not found; recording gamma = 5/3 "
                        "(unused by any POLARIS field).")

    cu = code_units_from(msol, kpc)                  # the one derivation
    density_kgm3 = cu.density_gcc * GCC_TO_KGM3
    return PolarisUnits(
        dMsolUnit=float(msol), dKpcUnit=float(kpc), mu=float(mu_v),
        gamma=float(gamma_v), source=source,
        length_m=cu.length_cm * CM_TO_M,
        density_kgm3=density_kgm3,
        number_density_m3=density_kgm3 / (float(mu_v) * M_H_KG),
        velocity_ms=(cu.velocity_kms * 1.0e5) * CMS_TO_MS,
        b_tesla=cu.b_gauss * GAUSS_TO_TESLA,
        time_s=cu.time_s,
        energy_per_mass_si=cu.erg_per_g * 1.0e-4,    # erg/g = 1e-4 J/kg
        b_gauss=cu.b_gauss,
        warnings=tuple(warnings),
    )


def readtipsy_b_factor(dMsolUnit, dKpcUnit):
    """`readtipsy.convertfromcodeunits`' CodetoGauss, recomputed independently.

    Kept here as the cross-check target for `_test_polaris_units.py`: it uses
    that function's OWN rounder constants, so agreement with `b_gauss` to ~1e-3
    is the algebraic identity `sqrt(4 pi M / L) / t == v * sqrt(4 pi rho)` being
    confirmed by two independent code paths, not a tautology.
    """
    MSOLG, KPCCM, GCGS = 1.99e33, 3.085678e21, 6.67e-8
    rho = (dMsolUnit * MSOLG) / (dKpcUnit * KPCCM) ** 3
    t = math.sqrt(1.0 / (rho * GCGS))
    gauss_to_code = math.sqrt(((1.0 / MSOLG) / dMsolUnit)
                              / (((1.0 / KPCCM) / dKpcUnit) * 4.0 * math.pi)) * t
    return 1.0 / gauss_to_code


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pa.add_argument("snap", nargs="?", help="tipsy snapshot (reads its .param)")
    pa.add_argument("--msol", type=float, help="dMsolUnit override")
    pa.add_argument("--kpc", type=float, help="dKpcUnit override")
    pa.add_argument("--mu", type=float, help="dMeanMolWeight override")
    a = pa.parse_args()
    u = polaris_units(dMsolUnit=a.msol, dKpcUnit=a.kpc, mu=a.mu, snap=a.snap)
    print(u.describe())
    for w in u.warnings:
        print(f"  WARNING: {w}")
