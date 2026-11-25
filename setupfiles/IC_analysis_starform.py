#!/usr/bin/env python3
"""
Analysis for the star-formation box (IC_setup_starform), covering both its
RATE and HANDOFF arms.

RATE arm (`coldfrac` = 0).  Every gas particle is eligible and `dStarEff` = 1,
so the acceptance test reduces to `p > rand()` with
p = 1 - exp(-dCStar dt / tdyn) and the per-step count is Binomial(N, p).  The
star-formation density smooth is gas-only, so removing gas genuinely lowers rho
and the surviving fraction follows the CLOSED recursion

    f_0 = 1,   p_k = 1 - exp(-dCStar dt sqrt(4 pi rho_0 f_k)),
    f_{k+1} = f_k (1 - p_k)

-- not a fitted exponential.  That recursion is the analytic overlay, and
`zfgas` reports the deviation in units of the propagated binomial sigma, so
"inside a 3-sigma band at every step" is a number, not an eyeball.

HANDOFF arm (`coldfrac` > 0).  A sparse cold subset in a warm box converts in
one step.  The conserved quantity is the FLUX `sum (m/rho) B`: `sfBFieldBlend`
is a volume-weighted average of the physical field, so it conserves sum V B
identically and deposits the magnetic energy that the averaging destroys as
heat.  `Emag` = sum 1/2 |B|^2 (m/rho) is therefore NOT conserved by design, and
under `B/sqrt(rho)` it merely happens to come back as well.  Both are reported,
neither is gated; the residual on the flux quantifies the model's limitation.

Everything the recipe actually ran with is read from the run `.log`, never from
the `.param`: `master.c` snaps `dDeltaStarForm` to the largest `dDelta * 2^-k`
strictly below the request, so the requested value is not the effective one.
The script prints the effective interval, `iStarFormRung` and the preprocessor
macros it found, and warns when any of them invalidates the analytic curve.

Produces the usual metric-vs-time figures, plus a one-off `.starlog` report:
every event's `rhoForm` against the box mean, its `TForm` against the
temperature the IC encodes, and (for the divv arm) whether `rForm` lies in the
compressing half of the box.

Usage:
    python IC_analysis_starform.py <run-dir> [-a rho0 ... -k nstep] [--save out]
    python IC_analysis_starform.py <run-dir> --render --render-n 4 --save out
    python IC_analysis_starform.py <run-dir> --save-reference      # bless
    python IC_analysis_starform.py <run-dir> --reference --reg-tol 1e-6
"""

import os
import re
import sys

import numpy as np

from IC_analysis_framework import (
    Param, Metric, RenderSpec, run_cli, styled,
    _cached_readtipsy, aux_magnitude, aux_component,
)
from IC_analysis_general import (
    read_bfield, read_log_param, read_tufac, _find_logs,
)
from IC_setup_starform import (
    temperature_from_u, tdyn, formation_probability, BOXLEN, case_spec,
)
import IC_starlog as starlog


# -a..-k mirror runtest.sh setup_starform() flag-for-flag, same defaults; the
# `runname` tags mirror its runname_extra so an analysis run by hand recovers
# them from the folder name.
PARAM_SPEC = [
    # Mirrors runtest.sh setup_starform(): -a picks the case, -b/-c optionally
    # override the two dials a rate measurement sweeps (0 = the case's value).
    # Everything else is a property of the case and is read from its table, so
    # the analysis cannot disagree with the IC about what was run.
    Param("a", "case", 1, int,
          "which case (table in IC_setup_starform.CASES)"),
    Param("b", "cstar", 0.0, float, "dCStar override (0 = the case's)", "cst"),
    Param("c", "pstep", 0.0, float,
          "per-step formation probability override (0 = the case's)", "pst"),
    # The two ARMS. They select the BUILD, not the IC, so they change no
    # analytic curve -- but runtest.sh -r forwards every flag it was given, and
    # a spec that stops at -c makes `--save-reference` die on `-d`/`-e` for
    # exactly the runs that need a reference most. The run-name rules mirror
    # IC_setup_starform.case_runname_tag().
    Param("d", "bvar", "sqrtrho", str,
          "evolved magnetic variable W = B rho^-n (sqrtrho|b|rho)",
          {"bvB": "b", "bvRho": "rho"}),
    Param("e", "sfb", 1, int, "SFBFIELD compiled in (1) or not (0)",
          {"nosfb": 0}),
]

METRICS = [
    Metric("fgas", r"$N_{gas}/N_{gas,0}$", "t", "linear", "o"),
    Metric("zfgas", r"$(f - f_{\rm analytic})\,/\,\sigma$", "t", "linear", "o"),
    Metric("Nstar", r"$N_\star$", "t", "linear", "s"),
    Metric("Mstar", r"$M_\star / M_{gas,0}$", "t", "linear", "s", norm="Mgas"),
    Metric("rho_gas", r"$\langle\rho\rangle_m / \rho_0$", "t", "linear", "^",
           norm="rho_gas"),
    Metric("Emag", r"$\sum \frac{1}{2}|B|^2 V$", "t", "linear", "d"),
    Metric("fluxmag", r"$|\sum V_i B_i|$", "t", "linear", "v"),
    Metric("Etot", r"$E_{tot}/E_{tot,0}$", "t", "linear", "x", norm="Etot"),
]

# Where analyze() parks what it read from the .log so analytic() -- which is
# handed only (metric_key, params) -- can build the same curve. run_cli always
# runs the whole series before plotting, so the entry is present by then; it is
# also saved into the blessed reference as provenance.
_LOG_KEY = "_sflog"

_DELTASF_RE = re.compile(
    r"dDeltaStarForm \(set\):\s*([-\d.eE+]+),\s*effectively:\s*([-\d.eE+]+)")
_MACRO_RE = re.compile(r"^#\s*Preprocessor macros:(.*)$", re.MULTILINE)

_SFLOG_CACHE = {}


def sf_log_params(run):
    """Star-formation settings as the RUN saw them, from its '.log'.

    Returns a dict with `dt_sf` (the EFFECTIVE `dDeltaStarForm` in code units),
    `dDelta`, `iStarFormRung`, `dCStar`, `dStarEff`, `dTempMax`, `dPhysDenMin`
    (code units -- master.c converts it before logging), `tufac` and `macros`.
    Missing entries come back None so the caller can fall back and say so.
    """
    key = os.path.normpath(run)
    if key in _SFLOG_CACHE:
        return _SFLOG_CACHE[key]
    text = ""
    for log in _find_logs(run):
        try:
            with open(log) as fh:
                text = fh.read()
            break
        except OSError:
            continue
    out = {k: read_log_param(run, k) for k in
           ("dCStar", "dStarEff", "dTempMax", "dPhysDenMin", "dDelta",
            "iStarFormRung", "bStarForm", "dInitStarMass",
            "dMinGasMass")}
    m = _DELTASF_RE.search(text)
    out["dt_sf"] = float(m.group(2)) if m else None
    out["dDeltaStarForm_set"] = float(m.group(1)) if m else None
    mm = _MACRO_RE.search(text)
    out["macros"] = mm.group(1).split() if mm else []
    out["tufac"] = read_tufac(run)
    _SFLOG_CACHE[key] = out
    return out


def depletion_series(n0, nsteps, cstar, dt, rho0):
    """The analytic gas-depletion recursion and its binomial 1-sigma envelope.

    f_{k+1} = f_k exp(-a sqrt(f_k)) with a = dCStar dt sqrt(4 pi rho_0), i.e.
    dCStar dt / tdyn evaluated at the CURRENT gas density rho_0 f_k.

    The spread is propagated linearly: one step draws Binomial(N_k, p_k), which
    contributes f_k p_k (1-p_k) / n0 to the variance of the fraction, and the
    variance already accumulated is carried through the map's derivative
    g'(f) = exp(-a sqrt f) (1 - a sqrt f / 2).  Good to the accuracy the 3-sigma
    acceptance band needs; it is not a substitute for an ensemble.
    """
    a = float(cstar) * float(dt) * np.sqrt(4.0 * np.pi * float(rho0))
    f, var = 1.0, 0.0
    fs, sigs = [f], [0.0]
    for _ in range(int(nsteps)):
        root = np.sqrt(f)
        decay = np.exp(-a * root)
        p = 1.0 - decay
        gprime = decay * (1.0 - 0.5 * a * root)
        var = gprime * gprime * var + (f * p * (1.0 - p) / max(n0, 1))
        f = f * decay
        fs.append(f)
        sigs.append(np.sqrt(max(var, 0.0)))
    return np.array(fs), np.array(sigs)


def _energies(tgdata, sdata, snap, ngas, tufac):
    """(E_kin, E_therm, E_mag, |sum V B|, sum V |B|) for one snapshot.

    E_kin includes the stars: a converted particle keeps its velocity, so
    leaving it out would make E_tot fall by construction at every event.
    """
    m, rho = tgdata[:, 0], tgdata[:, 7]
    e_kin = float(np.sum(0.5 * m * np.sum(tgdata[:, 4:7] ** 2, axis=1)))
    if len(sdata):
        e_kin += float(np.sum(0.5 * sdata[:, 0]
                              * np.sum(sdata[:, 4:7] ** 2, axis=1)))
    e_th = float(tufac * np.sum(m * tgdata[:, 8]))
    B = read_bfield(snap, ngas)
    if B is None:
        return e_kin, e_th, np.nan, np.nan, np.nan
    B = np.asarray(B, dtype=float)
    vol = m / rho
    e_mag = float(np.sum(0.5 * np.sum(B ** 2, axis=1) * vol))
    flux = np.sum(vol[:, None] * B, axis=0)
    absflux = float(np.sum(vol * np.sqrt(np.sum(B ** 2, axis=1))))
    return e_kin, e_th, e_mag, float(np.linalg.norm(flux)), absflux


def report_starlog(run, params, sf, tgdata, ngas):
    """One-off `.starlog` report: what every formation event actually recorded.

    Checks the three things the starlog is the only witness to -- the density
    the criterion used, the temperature it evaluated, and where in the box the
    events happened -- against what the IC and the run parameters imply.
    """
    # What temperature the GAS actually carries, from the snapshot's own thermal
    # column through the same relation the criterion uses. This is the
    # end-to-end check: the tipsy column is u/dTuFac, and the
    # criterion's T is CoolCodeEnergyToTemperature(u), two different
    # conventions that have to compose back to the IC's requested T.
    tgas = temperature_from_u(np.asarray(tgdata[:, 8], dtype=float)
                              * float(sf["tufac"]))
    print("[starform] gas T from the snapshot: %.4g .. %.4g K "
          "(IC asked for cold %.4g K, dTempMax %s)"
          % (tgas.min(), tgas.max(), float(case_spec(params.get("case", 1))["T0"]),
             sf.get("dTempMax")))

    try:
        events = starlog.read_starlog(run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[starform] no usable .starlog ({exc})")
        return
    print(f"[starform] {starlog.summarize(events, limit=5)}")
    if not len(events):
        return

    rho_mean = float(np.sum(tgdata[:, 0] * tgdata[:, 7]) / np.sum(tgdata[:, 0]))
    rho_form = np.asarray(events["rhoForm"], dtype=float)
    print("[starform] rhoForm/rho_mean(t=0): %.4f .. %.4f (mean %.4f)"
          % (rho_form.min() / rho_mean, rho_form.max() / rho_mean,
             rho_form.mean() / rho_mean))
    dpdm = sf.get("dPhysDenMin")
    if dpdm:
        print("[starform] rhoForm/dPhysDenMin: %.3g .. %.3g (all must exceed 1)"
              % (rho_form.min() / dpdm, rho_form.max() / dpdm))

    tform = np.asarray(events["TForm"], dtype=float)
    if np.all(tform == 0.0):
        print("[starform] TForm is 0 for every event -- a NOCOOLING build has "
              "no temperature to record (its criterion uses a hard 10 K).")
    else:
        expect = float(case_spec(params.get("case", 1))["T0"])
        print("[starform] TForm %.4g .. %.4g K (IC cold gas: %.4g K)"
              % (tform.min(), tform.max(), expect))

    _c = case_spec(params.get("case", 1))
    if int(_c["vmode"]) == 1 and float(_c["ampl"]) != 0:
        # vx = -ampl sin(2 pi x / L) converges for |x| < L/4.
        x = np.asarray(events["rFormx"], dtype=float)
        inside = np.count_nonzero(np.abs(x) < 0.25 * BOXLEN)
        print("[starform] %d/%d events in the compressing half |x| < L/4 "
              "(the divv criterion admits no others)" % (inside, len(x)))


def analyze(tgdata, time, params, snap):
    """Framework callback: one row of metrics per snapshot."""
    t = float(np.asarray(time).ravel()[0])
    run = os.path.dirname(os.path.normpath(snap)) or "."
    sf = sf_log_params(run)

    _g, _d, sdata, hdr, _t = _cached_readtipsy(snap)
    ngas = int(hdr[2])
    nstar = int(hdr[4])
    sdata = np.asarray(sdata, dtype=float)
    # Every star is exactly one spent gas particle -- the dInitStarMass clamp
    # converts a WHOLE particle (not dStarEff, which is 0.5 here and inert), so
    # N_gas + N_star is the initial gas count and the depletion fraction needs
    # no cross-snapshot state.
    n0 = ngas + nstar
    m = tgdata[:, 0]
    fgas = ngas / n0 if n0 else np.nan

    e_kin, e_th, e_mag, fluxmag, absflux = _energies(
        tgdata, sdata, snap, ngas, sf["tufac"])

    row = {
        "x": t,
        "fgas": fgas,
        "Nstar": float(nstar),
        "Mstar": float(sdata[:, 0].sum()) if len(sdata) else 0.0,
        "Mgas": float(m.sum()),
        "rho_gas": float(np.sum(m * tgdata[:, 7]) / np.sum(m)),
        "Ekin": e_kin,
        "Eth": e_th,
        "Emag": e_mag,
        "fluxmag": fluxmag,
        "fluxres": (fluxmag / absflux) if absflux else np.nan,
        "Etot": e_kin + e_th + (e_mag if np.isfinite(e_mag) else 0.0),
    }

    # Effective star-formation cadence: the log's value, or the setup's own
    # derivation if the log lacks it (and say which was used).
    dt_sf = sf.get("dt_sf")
    cstar = sf.get("dCStar")
    spec = case_spec(params.get("case", 1))
    rho0 = spec["rho0"]
    if cstar is None:
        cstar = float(params.get("cstar") or spec["cstar"])
    if dt_sf is None:
        pstep = float(params.get("pstep") or spec["pstep"])
        dt_sf = -np.log(1.0 - pstep) * float(tdyn(rho0)) / cstar

    # z-score against the recursion: k is read from the snapshot time rather
    # than assumed, so a run with iOutInterval > 1 still lands on the right step.
    k = int(round(t / dt_sf)) if dt_sf > 0 else 0
    fpred, sig = depletion_series(n0, max(k, 1), cstar, dt_sf, rho0)
    if k < len(fpred) and sig[k] > 0:
        row["zfgas"] = (fgas - fpred[k]) / sig[k]
    else:
        row["zfgas"] = 0.0

    # Per-run settings, keyed by run: comparing several runs in one figure must
    # score each against ITS OWN cadence. The analytic overlay can only be one
    # curve, so it uses the FIRST run's -- the same one the reference is blessed
    # from -- while spanning every run's time range.
    store = params.setdefault(_LOG_KEY, {})
    eff = store.get(run)
    if eff is None:
        eff = {"dt_sf": float(dt_sf), "cstar": float(cstar), "rho0": rho0,
               "n0": int(n0), "run": run,
               "from_log": sf.get("dt_sf") is not None}
        store[run] = eff
        store.setdefault("_ref", eff)
        _announce(run, eff, sf, n0)
        report_starlog(run, params, sf, tgdata, ngas)
    store["_ref"]["tmax"] = max(store["_ref"].get("tmax", 0.0), t)
    return row


def _announce(run, eff, sf, n0):
    """Print -- once per run -- the settings the analytic curve depends on, and
    warn about anything that would invalidate it."""
    macros = sf.get("macros") or []
    interesting = [x for x in macros
                   if x in ("STARFORM", "SFBFIELD", "DIVVOFF", "MHD",
                            "NOCOOLING", "COOLING_METAL", "DTADJUST",
                            "CORRPARTITION")]
    print("[starform] %s" % run)
    print("[starform] macros: %s" % (" ".join(interesting) or "(none found)"))
    for need in ("STARFORM", "SFBFIELD"):
        if macros and need not in macros:
            print(f"[starform] WARNING: the run was not built with {need}.",
                  file=sys.stderr)
    if "CORRPARTITION" in macros:
        print("[starform] WARNING: CORRPARTITION skips the star-formation "
              "density re-smooth (master.c), so the depletion recursion's "
              "falling rho does not apply.", file=sys.stderr)
    rung = sf.get("iStarFormRung")
    if rung is not None and rung != 0:
        print("[starform] WARNING: iStarFormRung = %g, so star formation does "
              "NOT fire once per base step; the analytic cadence is void."
              % rung, file=sys.stderr)
    print("[starform] dCStar=%g  dt_sf=%.6g (%s)  dDelta=%s  dStarEff=%s  "
          "N_gas(0)=%d" %
          (eff["cstar"], eff["dt_sf"],
           "from .log" if eff["from_log"] else "DERIVED, no .log value",
           sf.get("dDelta"), sf.get("dStarEff"), n0))
    p0 = float(formation_probability(eff["cstar"], eff["dt_sf"], eff["rho0"]))
    print("[starform] p(t=0) = %.6g per particle per step; tdyn = %.6g"
          % (p0, float(tdyn(eff["rho0"]))))
    # Whole-particle conversion is what reaches the handoff AND what makes the
    # analytic probability exactly dMProb. It comes from dInitStarMass
    # being clamped down to the particle mass, so check THAT, not dStarEff.
    m0, ism = sf.get("dMinGasMass"), sf.get("dInitStarMass")
    if not ism:
        print("[starform] WARNING: dInitStarMass = %s, so the star mass is "
              "dStarEff x m_gas, a FRACTION. A partial conversion never "
              "reaches the SFBFIELD handoff and breaks the analytic "
              "probability." % ism, file=sys.stderr)
    elif m0 and ism < 2.0 * m0:
        print("[starform] WARNING: dInitStarMass = %g is not comfortably above "
              "the particle mass (dMinGasMass = %g); whole-particle conversion "
              "is not guaranteed." % (ism, m0), file=sys.stderr)


def analytic(metric_key, params):
    """The depletion recursion (and what follows from it) versus time."""
    eff = (params.get(_LOG_KEY) or {}).get("_ref")
    if eff is None:                      # no snapshot processed -> nothing to say
        return None
    dt = eff["dt_sf"]
    nsteps = max(1, int(round(eff.get("tmax", 0.0) / dt))) if dt > 0 else 1
    f, sig = depletion_series(eff["n0"], nsteps, eff["cstar"], dt, eff["rho0"])
    t = np.arange(len(f)) * dt
    if metric_key == "fgas":
        return t, f
    if metric_key == "zfgas":
        return t, np.zeros_like(t)
    if metric_key == "Nstar":
        return t, eff["n0"] * (1.0 - f)
    if metric_key == "Mstar":
        # Equal-mass particles: the mass fraction IS the number fraction.
        return t, 1.0 - f
    if metric_key == "rho_gas":
        # The gas-only density smooth follows the surviving gas;
        # plotted normalised to its own t=0 value, so this is the ratio.
        return t, f
    return None


def render_specs(params):
    """Gas density always; once the box is magnetised, |B| and the SIGNED
    components that carry the physics of the case.

    |B| alone is not enough for the handoff cases. Under the winding
    (`field = 2`, `B = B0(0, cos kx, sin kx)`) |B| is constant by construction,
    so a |B| map shows only where the blend SHORTENED the field and nothing
    about the rotation that shortened it; `B_y` is the component whose coherent
    sum `sum V B_y` is the ledger's flux observable, and `B_z` is its quadrature
    partner, so the pair is the winding itself. Under the reversal
    (`field = 1`) `B_y` is the component that changes sign across the sheet.

    NOTE these are the CARTESIAN components, not a toroidal/poloidal split. The
    suite's toroidal convention (`B_phi = (x B_y - y B_x)/R`, mhdcollapse and
    powerlawring) needs a symmetry axis; this box is periodic with `B_x = 0`
    and no axis, so `B_phi` would be an artefact of the origin, not a field.
    """
    extent = [-0.5 * BOXLEN, 0.5 * BOXLEN, -0.5 * BOXLEN, 0.5 * BOXLEN]
    specs = [RenderSpec(axis1=1, axis2=2, quantity=7, label=r"$\rho$",
                        log=False, project="slice", clim=None, cmap="viridis",
                        extent=extent, slug="rho")]
    spec = case_spec(params.get("case", 1))
    if float(spec["B0"]) > 0.0:
        specs.append(RenderSpec(axis1=1, axis2=2,
                                quantity=aux_magnitude("BField"), label=r"$|B|$",
                                log=False, project="slice", clim=None,
                                cmap="inferno", extent=extent, slug="Bmag"))
        if int(spec["field"]) in (1, 2):
            specs.append(styled("By", 1, 2, aux_component("BField", 1),
                                label=r"$B_y$", project="slice", extent=extent))
        if int(spec["field"]) == 2:
            specs.append(styled("Bz", 1, 2, aux_component("BField", 2),
                                label=r"$B_z$", project="slice", extent=extent))
    return specs


def main():
    run_cli(
        test="starform",
        param_spec=PARAM_SPEC,
        analyze=analyze,
        analytic=analytic,
        metrics=METRICS,
        description="Star-formation box: SF rate vs the closed depletion "
                    "recursion, plus the SFBFIELD magnetic-handoff budget.",
        render_spec=render_specs,
    )


if __name__ == "__main__":
    main()
