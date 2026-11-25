#!/usr/bin/env python3
"""SFBFIELD handoff ledger: what the magnetic handoff moved, against what it owed.

Standalone, like `IC_starlog.py` -- the framework's `analyze()` sees one
snapshot of gas at a time, and this is a cross-run, cross-arm comparison.

WHAT IT TESTS

`sfBFieldBlend` (smoothfcn.c:7492) is a VOLUME-WEIGHTED AVERAGE of the physical
field,

    B_q <- (V_q B_q + dV B_d) / (V_q + dV),     sum_q dV_q = V_d

so with the donor's volume split across its neighbours it conserves `sum V B`
identically and releases the magnetic energy that averaging destroys,

    Q_mix = sum_q (1/2) V_q (dV_q/V_new,q) |B_q - B_d|^2

as heat.  The handoff is therefore modelled on FLUX; the energy is accounted,
not conserved.  `postDistSFBField` is a separate step that re-expresses the
blended field at the particle's actual SPH volume.

UPDATED 2026-09-14.  That rescale used to be sqrt(1 + x) whatever the evolved
variable -- the energy convention of `mhdCorrPartitionFac` -- and then the two
steps agreed for one variable only.  With W = B rho^-n and x = dVolSF/V a
recipient ended the next density evaluation holding

    flux delivered / owed   = (1 + x)^(1/2 - n)
    energy delivered / owed = (1 + x)^(1 - 2n)

-- both exactly 1 at n = 1/2 and only there.  The rescale is now (1 + x)^n,
which cancels the (1 + x)^-n that the stored W picks up at the next density
evaluation, so the blend's own answer is delivered for EVERY n and the handoff
is flux-conserving throughout.  What survives is the algorithm's own caveat that
"exact sum(V*B) conservation requires the SPH volume to match the donated-volume
model": the recipients' SPH volume response is not the donated volume, leaving

    flux delivered / owed = ((1 + x_sph) / (1 + x_model))^(1 - n)

which is exactly 1 at n = 1, where V B = m W carries no density at all.
`replay()` transcribes the current source; a prediction made with this file
before 2026-09-14 is not comparable with a run of it.

WHY THIS NEEDS THE WINDING IC (field = 2)

Under a uniform field none of the above is observable.  B_q == B_d makes the
blend the identity map and Q_mix identically zero, and `sum V B` and
`sum (1/2) B^2 V` are then the SAME statement, so no measurement can separate a
flux-conserving handoff from an energy-conserving one.  `winding_b` holds |B|
fixed and rotates its direction on the kernel scale, which makes the blend do
real work with a closed-form expected result.

METHOD

The prediction is not a fit and not a first-order expansion: the algorithm is
re-run in Python on the run's OWN initial conditions -- the same kernel, the
same neighbour weights, the same three steps -- and the result is compared with
the snapshot.  Two independent references, so a disagreement localises:

  * the A/B twin (`-e 0`, SFBFIELD off, same stars, same deletion) isolates the
    DEPOSIT from the deletion, by subtraction rather than by modelling;
  * the Python replica predicts both arms from the IC alone.

Usage:
    IC_handoff_ledger.py <rundir> [<rundir> ...]
    IC_handoff_ledger.py --pair <sfb-on-rundir> <sfb-off-rundir>
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import IC_starlog as starlog
import readtipsy as tip
from IC_analysis_general import read_bfield, read_log_param
from IC_setup_starform import BOXLEN, _opt_coeffs, _w_opt, handoff_ratios


# --------------------------------------------------------------------------- #
#  SPH state on the IC: the h and rho the run itself would have had
# --------------------------------------------------------------------------- #
def h_from_density(mass, rho, nsmooth=64):
    """gasoline's OTHER h convention: h = (m*hfact3/rho)^(1/3).

    `pkdUpdatefBall` (pkd.c) sets fBall2 = 4h^2 from the DENSITY rather than
    from a neighbour count, with hfact3 = 3*nSmooth/(32 pi) (master.c). It is
    the h of a uniform sphere of radius 2h holding nSmooth particles, so it
    agrees with `sph_state`'s d_nSmooth/2 only where the distribution is locally
    uniform. On the suite's 32^3 glass the two differ by a mean ratio 1.0065
    with a 1.58 per cent spread (range 0.943 - 1.063), which is enough to flip
    kernel membership for particles near 2h.

    WHICH ONE IS LIVE DEPENDS ON THE BUILD. Without CORRPARTITION the density
    smooth sets fBall from its own neighbour search and `sph_state` is right --
    verified against a frozen no-star-formation control to a median 1.3e-09 in
    rho. Under CORRPARTITION, master.c runs msrUpdatefBall in the density pass,
    so h comes from here instead; use this, not sph_state's h, for anything that
    has to reproduce the code's neighbour sets in such a build.
    """
    hfact3 = 3.0 * float(nsmooth) / (32.0 * np.pi)
    return (np.asarray(mass, dtype=float) * hfact3
            / np.asarray(rho, dtype=float)) ** (1.0 / 3.0)


def sph_state(pos, mass, nsmooth=64, boxlen=BOXLEN):
    """(h, rho, tree) on a periodic box, with gasoline's own conventions.

    `BALL2` is the distance to the nSmooth-th neighbour squared and the kernel's
    support is `r < 2h`, so h = d_nSmooth / 2 -- which is also what
    `bDoSphhOutput` writes (`outtype.c`: sqrt(fBall2*0.25)).  The density sum
    carries fNorm = 1/(pi h^3) and the `wzero` self-term correction, both from
    master.c via glassgen.kernels.

    This is the NON-CORRPARTITION convention. Under CORRPARTITION h is set from
    the density by pkdUpdatefBall instead -- see `h_from_density`.
    """
    from scipy.spatial import cKDTree
    from IC_setup_starform import _kernel_module
    wzero_opt = _kernel_module().wzero_opt

    norm, coeffs, _ = _opt_coeffs(nsmooth)
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float)
    # cKDTree's periodic boxsize wants coordinates in [0, L).
    tree = cKDTree(np.mod(pos + 0.5 * boxlen, boxlen), boxsize=boxlen)
    dist, _idx = tree.query(np.mod(pos + 0.5 * boxlen, boxlen), k=nsmooth)
    h = dist[:, -1] * 0.5

    rho = np.empty(len(pos))
    wzero = wzero_opt(nsmooth)
    for i in range(len(pos)):
        nb = tree.query_ball_point(np.mod(pos[i] + 0.5 * boxlen, boxlen),
                                   2.0 * h[i])
        nb = np.asarray(nb, dtype=int)
        d = _pdist(pos[nb], pos[i], boxlen)
        w = _w_opt((d / h[i]) ** 2, norm, coeffs)
        # the self term is the i == i entry, corrected by wzero
        w[nb == i] = _w_opt(np.zeros(1), norm, coeffs)[0] * wzero
        rho[i] = np.sum(mass[nb] * w) / (np.pi * h[i] ** 3)
    return h, rho, tree


def _pdist(points, centre, boxlen=BOXLEN):
    """Minimum-image distances from `centre` to `points`."""
    d = np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)
    d -= boxlen * np.round(d / boxlen)
    return np.sqrt(np.sum(d * d, axis=-1))


# --------------------------------------------------------------------------- #
#  The algorithm, re-run in Python on the IC
# --------------------------------------------------------------------------- #
def replay(pos, mass, B, h, rho, donors, n, sfb, nsmooth=64, boxlen=BOXLEN):
    """Replicate DistSFBField + postDistSFBField + the next density evaluation.

    Returns a dict of the post-event totals over the SURVIVING gas, so it can be
    compared with a snapshot directly.  `n` is the power in W = B rho^-n; `sfb`
    selects whether the handoff runs at all (the A/B twin).

    Deliberately a transcription of smoothfcn.c rather than of the algebra: the
    exponent law is what the algebra gives, but a transcription also catches a
    weight normalisation or a support test being wrong.
    """
    norm, coeffs, _ = _opt_coeffs(nsmooth)
    from scipy.spatial import cKDTree
    wrapped = np.mod(pos + 0.5 * boxlen, boxlen)
    tree = cKDTree(wrapped, boxsize=boxlen)

    alive = np.ones(len(pos), dtype=bool)
    alive[donors] = False
    V = mass / rho
    Bnew = np.array(B, dtype=float)
    dVolSF = np.zeros(len(pos))
    Qmix = np.zeros(len(pos))
    rho_after = np.array(rho, dtype=float)

    for d in np.asarray(donors, dtype=int):
        nb = np.asarray(tree.query_ball_point(wrapped[d], 2.0 * h[d]), dtype=int)
        r = _pdist(pos[nb], pos[d], boxlen)
        # sfBFieldWeight: gas, not a donor, not deleted, and strictly inside 2h
        ok = alive[nb] & (r < 2.0 * h[d]) & (nb != d)
        nb, r = nb[ok], r[ok]
        rs = _w_opt((r / h[d]) ** 2, norm, coeffs)
        rs = np.where(rs > 0, rs, 0.0)
        rstot = rs.sum()
        if not rstot > 0:
            continue
        # The deletion itself: the neighbours' density loses the donor's
        # contribution. This happens whether or not SFBFIELD is on, and it is
        # evaluated in the RECIPIENT's kernel -- rho_q = sum_j m_j W(r/h_q) --
        # not the donor's, which is what sets the handoff weights above.
        rs_q = _w_opt((r / h[nb]) ** 2, norm, coeffs)
        rho_after[nb] -= mass[d] * np.where(rs_q > 0, rs_q, 0.0) \
            / (np.pi * h[nb] ** 3)
        if not sfb:
            continue
        dV = (rs / rstot) * V[d]
        Vq = V[nb] + dVolSF[nb]
        Vnew = Vq + dV
        dB2 = np.sum((Bnew[nb] - B[d]) ** 2, axis=1)
        Bnew[nb] = ((Vq[:, None] * Bnew[nb] + dV[:, None] * B[d])
                    / Vnew[:, None])
        Qmix[nb] += 0.5 * Vq * (dV / Vnew) * dB2
        dVolSF[nb] += dV

    if sfb:
        # postDistSFBField: re-express the blended field at the volume the
        # blend's own model implies, by the EVOLVED VARIABLE's own power.
        # The stored W is carried through the next density change unchanged, so
        # the derived B = W rho^n picks up (1 + x)^-n on its own; multiplying by
        # (1 + x)^n here cancels exactly that and leaves the blended field over
        # the enlarged volume, which is what conserves sum V B.
        #   n = 0 -> 1 (a density change moves nothing)
        #   n = 1/2 -> sqrt(1 + x)
        #   n = 1 -> 1 + x
        # Mirrors smoothfcn.c postDistSFBField.
        got = dVolSF > 0
        Bnew[got] *= ((1.0 + dVolSF[got] / V[got]) ** float(n))[:, None]

    # The next density evaluation. The evolved variable W = B rho^-n is what
    # the code stores, so the physical field moves with rho^n.
    # The algorithm's own caveat (smoothfcn.c:7594): "exact sum(V*B)
    # conservation requires the SPH volume to match the donated-volume model".
    # dVolSF is split by the DONOR's kernel; the volume the recipients actually
    # gain comes from their OWN density estimate losing the donor. Those are
    # different sums, and the ratio below is exactly how far apart they are --
    # which is the size of the flux residual that survives even at n = 1/2.
    dV_sph = mass / rho_after - mass / rho
    Vd_total = float(np.sum(mass[donors] / rho[donors]))
    scale = (rho_after[alive] / rho[alive]) ** float(n)
    Bfin = Bnew[alive] * scale[:, None]
    Vfin = mass[alive] / rho_after[alive]
    return {
        "N_gas": int(alive.sum()),
        # Per-particle, for the distribution question: the totals below say
        # whether the right AMOUNT of flux survived, these say whether it went
        # to the right PARTICLES. `alive` indexes the IC; `Bfin`/`Vfin` are the
        # surviving rows in IC order, which is the order a snapshot stores them.
        "alive": alive,
        "Bfin": Bfin,
        "Vfin": Vfin,
        "E_mag": float(np.sum(0.5 * np.sum(Bfin ** 2, axis=1) * Vfin)),
        "flux": np.sum(Vfin[:, None] * Bfin, axis=0),
        "absflux": float(np.sum(Vfin * np.sqrt(np.sum(Bfin ** 2, axis=1)))),
        "Q_mix": float(Qmix.sum()),
        "x_mean": float(np.mean(dVolSF[dVolSF > 0] / V[dVolSF > 0]))
        if np.any(dVolSF > 0) else 0.0,
        "vol_response": float(np.sum(dV_sph[alive]) / Vd_total),
    }


def initial_totals(mass, B, rho):
    V = mass / rho
    return {
        "N_gas": len(mass),
        "E_mag": float(np.sum(0.5 * np.sum(B ** 2, axis=1) * V)),
        "flux": np.sum(V[:, None] * B, axis=0),
        "absflux": float(np.sum(V * np.sqrt(np.sum(B ** 2, axis=1)))),
    }


# --------------------------------------------------------------------------- #
#  Measurement
# --------------------------------------------------------------------------- #
def snapshot_totals(snap):
    """(N_gas, E_mag, sum V B, sum V |B|, E_th-without-tufac, m, rho) of a dump."""
    gdata, _d, sdata, hdr, _t = tip.readtipsy(snap)
    gdata = np.asarray(gdata, dtype=float)
    ngas = int(hdr[2])
    m, rho = gdata[:, 0], gdata[:, 7]
    V = m / rho
    B = read_bfield(snap, ngas)
    B = np.asarray(B, dtype=float)[:ngas]
    return {
        "N_gas": ngas,
        "N_star": int(hdr[4]),
        "E_mag": float(np.sum(0.5 * np.sum(B ** 2, axis=1) * V)),
        "flux": np.sum(V[:, None] * B, axis=0),
        "absflux": float(np.sum(V * np.sqrt(np.sum(B ** 2, axis=1)))),
        "mu": float(np.sum(m * gdata[:, 8])),   # thermal column x mass
        "m": m, "rho": rho, "B": B,
        "pos": gdata[:, 1:4],
    }


def bvar_n(run):
    """The power n in W = B rho^-n, read from the run's own macro line."""
    macros = []
    for log in sorted(glob.glob(os.path.join(run, "*.log"))):
        with open(log) as fh:
            for line in fh:
                if line.startswith("# Preprocessor macros:"):
                    macros = line.split(":", 1)[1].split()
                    break
        if macros:
            break
    for tok in macros:
        if tok.startswith("MHD_B_VAR="):
            val = tok.split("=", 1)[1]
            if val == "B":
                return 0.0, val, ("SFBFIELD" in macros)
            if val == "B/rho":
                return 1.0, val, ("SFBFIELD" in macros)
            return 0.5, val, ("SFBFIELD" in macros)
    return 0.5, "B/sqrt(rho) (default)", ("SFBFIELD" in macros)


def find_snap(run, index):
    hits = sorted(glob.glob(os.path.join(run, "*.%05d" % index)))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
def ledger(run, ic=None, nsmooth=64, verbose=True):
    """Everything the handoff did in one run, measured and predicted."""
    n, label, has_sfb = bvar_n(run)
    events = starlog.read_starlog(run)
    after = find_snap(run, 1)
    if after is None:
        raise SystemExit("%s: no .00001 snapshot" % run)

    if ic is None:
        ic = default_ic(run)
    gdata, _d, _s, hdr, _t = tip.readtipsy(ic)
    gdata = np.asarray(gdata, dtype=float)
    nic = int(hdr[2])
    pos = gdata[:, 1:4]
    mass = gdata[:, 0]
    Bic = np.asarray(read_bfield(ic, nic), dtype=float)[:nic]

    h, rho, _tree = sph_state(pos, mass, nsmooth)
    donors = np.asarray(sorted(events["iOrdGas"].astype(int)))

    # NOT a check on the reconstruction. The starlog's rhoForm is recorded at
    # the formation moment INSIDE the step; a dump's fDensity is the value at
    # the end of it, and the two differ by ~6e-04 median / 1e-02 max. Measured
    # on case 9 against a frozen (ACCZERO) no-star-formation control,
    # which gives the code's own density on the untouched IC configuration:
    #
    #   sph_state vs the code's dump   median -1.3e-09   max 5.8e-08
    #   sph_state vs starlog rhoForm   median +5.9e-04   max 1.1e-02
    #   the code's dump vs rhoForm     median +5.9e-04   max 1.1e-02
    #
    # i.e. the code disagrees with its own starlog by exactly the amount this
    # number reports, and sph_state sits on the dump to float32 precision. Read
    # this as a starlog-vs-dump convention difference.
    rho_chk = np.abs(rho[donors] / np.asarray(events["rhoForm"])[
        np.argsort(np.asarray(events["iOrdGas"]).astype(int))] - 1.0)

    pred = replay(pos, mass, Bic, h, rho, donors, n, has_sfb, nsmooth)
    pred0 = replay(pos, mass, Bic, h, rho, donors, n, False, nsmooth)
    init = initial_totals(mass, Bic, rho)
    meas = snapshot_totals(after)

    if verbose:
        print("=" * 78)
        print("%s" % os.path.basename(os.path.normpath(run)))
        print("  MHD_B_VAR = %-18s n = %.1f    SFBFIELD %s"
              % (label, n, "ON" if has_sfb else "OFF"))
        print("  donors %d   <x = dVolSF/V> = %.5f   h = %.5f +/- %.5f"
              % (len(donors), pred["x_mean"], h.mean(), h.std()))
        print("  starlog rhoForm vs this tool's rho (a CONVENTION difference,"
              " not a\n    reconstruction error -- the code's own dump differs"
              " from its starlog by\n    the same amount): median %.2e  max %.2e"
              % (np.median(rho_chk), rho_chk.max()))
        print("  SPH volume response / donated volume = %.5f  "
              "(1 = the model the blend assumes)" % pred["vol_response"])
        fr, er = handoff_ratios(n, pred["x_mean"])
        print("  delivered/owed at that x:  flux %.5f   energy %.5f"
              % (fr, er))
        print("     ^ SUPERSEDED: that is the law from before 2026-09-14, when")
        print("       postDistSFBField rescaled by sqrt(1+x) at every n. The")
        print("       rescale is now (1+x)^n, so the blend's own answer is")
        print("       delivered for every n and the only surviving error is the")
        print("       volume model's: ((1+x_sph)/(1+x_model))^(1-n), == 1 at n=1.")
        print("  N_gas  predicted %6d   measured %6d"
              % (pred["N_gas"], meas["N_gas"]))
        _row("E_mag", init["E_mag"], pred["E_mag"], meas["E_mag"])
        _row("sum V|B|", init["absflux"], pred["absflux"], meas["absflux"])
        for j, ax in enumerate("xyz"):
            _row("sum V B_%s" % ax, init["flux"][j], pred["flux"][j],
                 meas["flux"][j], denom=init["absflux"])
        if has_sfb:
            print("  Q_mix predicted %.6e  = %.4f x the donors' own E_mag"
                  % (pred["Q_mix"],
                     pred["Q_mix"] / (0.5 * np.sum(
                         np.sum(Bic[donors] ** 2, axis=1)
                         * mass[donors] / rho[donors]))))
        print("  handoff deposit, predicted (sfb on - sfb off):")
        print("      E_mag   %+.1f ppm of E_mag(0)"
              % (1e6 * (pred["E_mag"] - pred0["E_mag"]) / init["E_mag"]))
        print("      sum V|B| %+.1f ppm of sum V|B|(0)"
              % (1e6 * (pred["absflux"] - pred0["absflux"]) / init["absflux"]))
    return dict(run=run, n=n, label=label, sfb=has_sfb, pred=pred,
                pred_nosfb=pred0, init=init, meas=meas, donors=donors,
                x_mean=pred["x_mean"], rho_err=float(rho_chk.max()))


def _row(name, init, pred, meas, denom=None):
    d = denom if denom is not None else (abs(init) if init else 1.0)
    print("  %-10s init %+.6e  pred %+.6e (%+8.1f ppm)  meas %+.6e (%+8.1f ppm)"
          % (name, init, pred, 1e6 * (pred - init) / d,
             meas, 1e6 * (meas - init) / d))


def default_ic(run):
    """The shared IC of this run's case, in the suite's datafiles/ directory."""
    base = os.path.basename(os.path.normpath(run))
    # "…case10bvRhoSep6GASOLINE-CZ" -> "10": the LEADING digits only, or the
    # 6 in "Sep6" joins the case number.
    tail = base.split("case", 1)[1] if "case" in base else ""
    case = ""
    for ch in tail:
        if not ch.isdigit():
            break
        case += ch
    root = os.path.join(os.path.dirname(os.path.normpath(run)), "datafiles")
    hits = sorted(glob.glob(os.path.join(root, "*case%s-I.00000" % case)))
    if not hits:
        raise SystemExit("no IC found for case %r under %s" % (case, root))
    return hits[0]


def compare(on, off, tufac=1.0):
    """The measured deposit: the A/B difference, which subtracts hydro drift."""
    print("=" * 78)
    print("DEPOSIT  (SFBFIELD on) - (SFBFIELD off), same stars, same deletion")
    print("  %s" % os.path.basename(os.path.normpath(on["run"])))
    print("  %s" % os.path.basename(os.path.normpath(off["run"])))
    assert on["meas"]["N_gas"] == off["meas"]["N_gas"], \
        "the arms deleted different particles; they are not a twin"
    i0 = on["init"]
    for key, norm in (("E_mag", i0["E_mag"]), ("absflux", i0["absflux"])):
        dm = on["meas"][key] - off["meas"][key]
        dp = on["pred"][key] - off["pred"][key]
        print("  %-9s measured %+9.1f ppm    predicted %+9.1f ppm"
              % (key, 1e6 * dm / norm, 1e6 * dp / norm))
    dmu = (on["meas"]["mu"] - off["meas"]["mu"]) * tufac
    print("  Q_mix     measured %+.6e (thermal)   predicted %+.6e"
          % (dmu, on["pred"]["Q_mix"]))
    if on["pred"]["Q_mix"]:
        print("            ratio measured/predicted = %.4f"
              % (dmu / on["pred"]["Q_mix"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--ic", default=None, help="IC file (default: the case's)")
    ap.add_argument("--nsmooth", type=int, default=64)
    ap.add_argument("--pair", action="store_true",
                    help="treat the runs as (SFBFIELD on, SFBFIELD off) twins")
    args = ap.parse_args()

    out = [ledger(r, args.ic, args.nsmooth) for r in args.runs]
    if args.pair or (len(out) == 2 and out[0]["sfb"] != out[1]["sfb"]):
        on = [o for o in out if o["sfb"]]
        off = [o for o in out if not o["sfb"]]
        if on and off:
            tufac = read_log_param(on[0]["run"], "dTuFac", 1.0) or 1.0
            compare(on[0], off[0], tufac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
