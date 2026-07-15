#!/usr/bin/env python
"""Convert a gasoline/tipsy IC into a Phantom full-dump IC (+ a `.in` run file).

Phantom is a particle (SPH) code, so the IC is a 1:1 copy of the SPH particles
into Phantom's tagged Fortran dump format (see
sph_interp/_phantom/phantom_dump.py). No interpolation. The dump writer is
self-validated against Phantom's own readPhantomDump.read_dump.

    ICconvert_tipsytophantom.py <input> <output_prefix> [--in-out P] [--tu-fac F]

  input          tipsy snapshot (e.g. datafiles/run.00000).
  output_prefix  Phantom dump path; written as '<output_prefix>_00000'.

Reads the gasoline `.param` beside the input for u = dTuFac*T, gamma, and the run
times that go into the `.in`. Equal-mass particles are assumed (Phantom's
massoftype scheme); a warning is printed if the tipsy masses are not uniform.
NOTE: hydro only (no MHD block yet) -- validate an MHD dump against a Phantom
built with MHD=yes before using it for the magnetic tests.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp import from_tipsy
from sph_interp._phantom.phantom_dump import write_phantom_fulldump


def _find_param(tipsy_input):
    base = tipsy_input.rsplit(".", 1)[0] if "." in os.path.basename(tipsy_input) \
        else tipsy_input
    cand = base + ".param"
    return cand if os.path.exists(cand) else None


def _param_dict(path):
    vals = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.split("#")[0]
                if "=" in line:
                    k, _, v = line.partition("=")
                    vals[k.strip()] = v.strip()
    return vals


def _f(vals, key, default=None):
    try:
        return float(vals[key])
    except (KeyError, ValueError):
        return default


def write_infile(in_path, prefix, gas, gamma):
    """A minimal Phantom `.in`: dumpfile + run times from the gasoline .param.

    Deliberately minimal: Phantom rejects an incomplete `.in`, rewrites a complete
    one for its compile options, and exits (the suite runs a two-pass handshake +
    phantom_patch_infile.py to re-assert these values). We only write keys Phantom
    actually accepts in the `.in` for our builds -- `gamma` is NOT one of them
    (Phantom warns "unknown variable gamma" and reads gamma from the dump header
    instead), so it is intentionally omitted here.
    """
    dDelta = _f(gas, "dDelta", 0.01)
    nSteps = _f(gas, "nSteps", 0) or 0
    iOut = _f(gas, "iOutInterval", 1) or 1
    tmax = (nSteps or 1) * dDelta
    dtmax = iOut * dDelta
    lines = [
        "# Phantom input file generated from a gasoline .param by",
        "#   ICconvert_tipsytophantom.py",
        "",
        "            dumpfile =  %s_00000   ! dump file to start from" % os.path.basename(prefix),
        "",
        "# options controlling run time and output",
        "                tmax =      %.8g   ! end time" % tmax,
        "               dtmax =      %.8g   ! time between dumps" % dtmax,
        # Force a FULL dump every snapshot (not Phantom's default nfulldump=10).
        # Small (intermediate) dumps store positions as real4 and omit velocities,
        # which Phantom's readPhantomDump mis-parses (x read at half length, no vx);
        # the reverse converter + analysis need full double-precision dumps.
        "           nfulldump =          1   ! full dump every dump (no small dumps)",
        "",
        "# equation of state",
        "                ieos =          2   ! adiabatic eos",
    ]
    with open(in_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    pa = argparse.ArgumentParser(prog="ICconvert_tipsytophantom")
    pa.add_argument("input")
    pa.add_argument("output_prefix")
    pa.add_argument("--in-out", default=None, help="also write a Phantom .in here")
    pa.add_argument("--tu-fac", type=float, default=None)
    pa.add_argument("--hfact", type=float, default=1.2)
    args = pa.parse_args(argv)

    param = _find_param(args.input)
    gas = _param_dict(param)
    gamma = _f(gas, "dConstGamma", 5.0 / 3.0)

    # Self-gravity (bDoGravity=1, e.g. evrard/polytrope): Phantom computes its
    # code-unit gravitational constant from the dump's units as
    # G_code = gg * umass * utime^2 / udist^3 (physcon.gg=6.672041e-8, units.f90).
    # The gasoline IC is written in a G=1 code-unit system (dKpcUnit/dMsolUnit fix
    # the length/mass units; the time unit is chosen so G=1). To reproduce that in
    # Phantom we must set utime = sqrt(udist^3 / (gg*umass)) -- NOT the default 1,
    # which would give G_code = gg (~6.7e-8) and gravity ~1e7x too weak (no
    # collapse). Using gasoline's physical udist/umass makes utime EQUAL gasoline's
    # own time unit, so velocities/times round-trip unchanged. Hydro/MHD tests keep
    # unit=1 (no gravity -> G_code irrelevant). Phantom physcon constants:
    _GG, _SOLARM, _KPC = 6.672041e-8, 1.9891e33, 3.086e21
    grav = int(_f(gas, "bDoGravity", 0.0)) == 1
    isperiodic = int(_f(gas, "bPeriodic", 1.0)) != 0
    if grav:
        umass = _f(gas, "dMsolUnit", 1.0) * _SOLARM
        udist = _f(gas, "dKpcUnit", 1.0) * _KPC
        utime = (udist ** 3 / (_GG * umass)) ** 0.5
        print("Self-gravity IC: udist=%.6g cm umass=%.6g g utime=%.6g s "
              "-> G_code=%.6g" % (udist, umass, utime,
                                  _GG * umass * utime ** 2 / udist ** 3))
    else:
        udist = umass = utime = 1.0

    # dTuFac for u = dTuFac * T. The IC .param has no dTuFac/dGasConst line, so
    # compute it from the gasoline unit system (dKpcUnit, dMsolUnit, dMeanMolWeight,
    # dConstGamma) via read_param_tufac -- the SAME path the AREPO/GIZMO converters
    # use. Falling back to u=T (dTuFac=1) leaves the gas ~1000x too cold: e.g. the KH
    # IC then has cs~0.03 vs shear v~0.5 (Mach ~16 instead of ~0.3), so the supersonic
    # streams shock-collapse instead of forming KH billows. Order: CLI override ->
    # .param units -> .log dTuFac/dGasConst -> 1.0.
    tu_fac = args.tu_fac
    if tu_fac is None:
        try:
            from sph_interp.particles import read_param_tufac
            tu_fac = read_param_tufac(args.input)
        except Exception:
            tu_fac = None
    if tu_fac is None:
        try:
            from IC_analysis_general import read_tufac
            tu_fac = read_tufac(args.input, 1.0)
        except Exception:
            tu_fac = 1.0

    p = from_tipsy(args.input, fields=("rho", "vx", "vy", "vz", "T"),
                   periodic=isperiodic)
    vals = p.values
    n = p.pos.shape[0]
    vel = np.zeros((n, 3))
    for k, c in enumerate(("vx", "vy", "vz")):
        if c in vals:
            vel[:, k] = vals[c]
    # u = dTuFac * T (gasoline thermal state is a temperature)
    T = vals.get("T", vals.get("temp"))
    if T is None:
        raise KeyError("tipsy IC has no temperature field for u = dTuFac*T")
    u = np.asarray(T, float) * tu_fac

    mass = np.asarray(p.mass, float)
    massoftype = float(mass.mean())
    if mass.std() > 1e-6 * abs(massoftype):
        print("WARNING: tipsy masses are not uniform (std/mean=%.2e); Phantom's "
              "massoftype scheme assumes equal mass -- using the mean."
              % (mass.std() / abs(massoftype)))

    # Smoothing length: compute h EXACTLY as Phantom defines it, h = hfact*(m/rho)^(1/3),
    # rather than reuse the tipsy/getsmooth2 estimate (a different nsmooth-based
    # convention, ~3% off). This makes the IC self-consistent with Phantom's density-h
    # relation so its first density/h iteration converges from the start; a mismatched
    # h otherwise drives a runaway neighbour search (kdtree getneigh) on startup.
    rho = np.asarray(p.rho, float)
    h = args.hfact * (massoftype / rho) ** (1.0 / 3.0)

    # MHD: write an MHD dump (narraylengths=4, Bxyz block) ONLY if the IC carries a
    # GENUINELY nonzero B (has_nonzero_b) -- writealltipsyaux writes an all-zero
    # BField aux for EVERY IC (incl. hydro evrard), so file presence alone isn't
    # MHD. This also keeps hydro/grav dumps in hydro format (narraylengths=2) so the
    # non-MHD phantom / phantom_grav binaries read them without choking on a stray
    # magnetic block. (Same auto-detect the GIZMO/OpenGadget3 converters use.)
    Bxyz = None
    from ICconvert_common import has_nonzero_b
    if has_nonzero_b(args.input):
        try:
            from IC_analysis_general import read_vector_aux
            Bxyz = read_vector_aux(args.input, "BField", n)
        except Exception:
            Bxyz = None

    # Isolated (bPeriodic=0) IC: OMIT the periodic box so the header carries no
    # boundary positions -- a non-periodic Phantom build (the self-gravity binary;
    # periodic + gravity would need Ewald) expects that. Periodic tests pass box.
    box = p.box if isperiodic else None
    out = args.output_prefix + "_00000"
    write_phantom_fulldump(out, p.pos, vel, u, h,
                           massoftype, time=0.0, gamma=gamma, hfact=args.hfact,
                           udist=udist, umass=umass, utime=utime,
                           box=box, Bxyz=Bxyz)
    print("Wrote Phantom dump -> %s (%d gas particles, massoftype=%.6g, "
          "u=dTuFac*T with dTuFac=%.6g, %s)"
          % (out, n, massoftype, tu_fac,
             "MHD: Bxyz written" if Bxyz is not None else "hydro (no B)"))
    if args.in_out:
        write_infile(args.in_out, args.output_prefix, gas, gamma)
        print("Wrote Phantom .in -> %s" % args.in_out)


if __name__ == "__main__":
    main()
