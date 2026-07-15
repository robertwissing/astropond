#!/usr/bin/env python
"""Convert a gasoline/tipsy IC into a RAMSES GridIC + namelist (code=4).

RAMSES has no file-based hydro IC, so we deposit the SPH field onto a fine uniform
grid with our petkova interpolation (sph_interp.gridic.write_grid_ic), write the
portable GridIC binary `ic_gridic.bin` into the RAMSES run dir, and generate a
namelist whose `&INIT_PARAMS condinit_kind='gridic'` makes the testsuite condinit
patch read it. The SAME file drives a uniform run (levelmin=levelmax) or an AMR run
(levelmin<levelmax; coarse cells volume-average the fine GridIC cells).

    ICconvert_tipsytoramses.py <input> <rundir> --levelmax L [--levelmin l]
                               [--grid uniform|amr] [--method petkova|sph]
                               [--boxsize Lx]

  input    tipsy snapshot (datafiles/run.00000).
  rundir   RAMSES run dir; writes <rundir>/ic_gridic.bin and <rundir>/run.nml.

Run times/gamma come from the gasoline .param beside the input. Cubic boxes only
(first cut). MHD: a genuinely NONZERO BField aux switches to the MHD GridIC
(6 div-free face-B components) + an hlld MHD namelist for ramses3d_mhd (built
with the suite_mhd condinit patch); an all-zero BField aux (e.g. sedov) stays
on the hydro path.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp.gridic import write_grid_ic, read_grid_ic
from ICconvert_common import has_nonzero_b as _has_nonzero_b


def _find_param(tipsy_input):
    base = tipsy_input.rsplit(".", 1)[0] if "." in os.path.basename(tipsy_input) \
        else tipsy_input
    cand = base + ".param"
    return cand if os.path.exists(cand) else None


def _param(path):
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


def write_namelist(path, gas, boxlen, levelmin, levelmax, gamma, amr,
                   mass_sph=None, m_refine=8.0, mhd=False, np_ranks=1):
    """Write a RAMSES namelist for a condinit='gridic' hydro run.

    AMR refinement is QUASI-LAGRANGIAN (mass-based, RAMSES's standard
    strategy): a cell refines when it holds more than `m_refine * mass_sph`
    of gas, so the grid resolution follows the mass exactly like SPH's does
    -- the SPH-comparable mode. `mass_sph` should be the mean BASE-cell mass
    (Mtot / 2**(3*levelmin)); with m_refine=8 a cell splits when its children
    would each carry ~the base-cell mean."""
    dDelta = _f(gas, "dDelta", 0.01)
    nSteps = int(_f(gas, "nSteps", 0) or 0)
    iOut = int(_f(gas, "iOutInterval", 1) or 1)
    tend = (nSteps or 1) * dDelta
    dtout = iOut * dDelta
    # explicit output times 0, dtout, 2*dtout, ... tend
    nout = max(1, int(round(tend / dtout))) if dtout > 0 else 1
    touts = [min(i * dtout, tend) for i in range(1, nout + 1)]
    tout_str = ",".join("%.8g" % t for t in touts)

    # hlld (riemann2d for the 2D corner fluxes of the CT induction update) is
    # the accurate MHD choice; the hydro solver only knows the string 'hllc'.
    riemann = "riemann='hlld'\nriemann2d='hlld'" if mhd else "riemann='hllc'"

    # ngridmax is a PER-RANK allocation (uold etc. scale with it): keep the
    # serial-validated 2M total but split it across MPI ranks, with a floor
    # generous enough for boundary/load imbalance on small test grids.
    ngridmax = max(200000, -(-2000000 // max(1, np_ranks)))

    refine = ""
    if amr:
        nlev = max(levelmax, 1)
        mref_str = ",".join("%.8g" % m_refine for _ in range(nlev))
        refine = (
            "&REFINE_PARAMS\n"
            "interpol_var=1\n"
            "interpol_type=0\n"
            f"mass_sph={mass_sph:.10g}\n"
            f"m_refine={mref_str}\n"
            "/\n")
    # nremap: load-balance every 10 coarse steps — a no-op in serial (ncpu=1)
    # but required for MPI AMR runs (RAMSESNP>1), where refinement would
    # otherwise pile new grids onto the initial decomposition forever.
    nml = f"""&RUN_PARAMS
hydro=.true.
nrestart=0
nremap=10
ncontrol=1
verbose=.false.
/

&AMR_PARAMS
levelmin={levelmin}
levelmax={levelmax}
ngridmax={ngridmax}
nexpand=1
boxlen={boxlen:.8g}
/

&INIT_PARAMS
condinit_kind='gridic'
nregion=1
region_type='square'
/

&OUTPUT_PARAMS
noutput={len(touts)}
tout={tout_str}
/

&HYDRO_PARAMS
gamma={gamma:.10g}
courant_factor=0.8
scheme='muscl'
slope_type=1
{riemann}
/

{refine}"""
    with open(path, "w") as f:
        f.write(nml)


def main(argv=None):
    pa = argparse.ArgumentParser(prog="ICconvert_tipsytoramses")
    pa.add_argument("input")
    pa.add_argument("rundir")
    pa.add_argument("--levelmax", type=int, required=True,
                    help="finest level RAMSES may refine to")
    pa.add_argument("--levelmin", type=int, default=None,
                    help="base level; the GridIC is deposited at npx=2**levelmin "
                         "(= the SPH per-axis resolution for a matched "
                         "comparison). Default = levelmax (uniform run); "
                         "levelmax-2 for --grid amr")
    pa.add_argument("--grid", choices=["uniform", "amr"], default="uniform")
    pa.add_argument("--method", choices=["petkova", "sph"], default="petkova")
    pa.add_argument("--m-refine", type=float, default=8.0,
                    help="quasi-Lagrangian refinement: refine a cell holding "
                         "more than m_refine * (mean base-cell mass) of gas")
    pa.add_argument("--boxsize", type=float, default=None,
                    help="cubic box side (default: dxPeriod)")
    pa.add_argument("--np", type=int, default=1, dest="np_ranks",
                    help="MPI ranks the run will use; scales the per-rank "
                         "ngridmax in the namelist")
    args = pa.parse_args(argv)

    gas = _param(_find_param(args.input))
    gamma = _f(gas, "dConstGamma", 5.0 / 3.0)
    boxlen = args.boxsize if args.boxsize is not None else _f(gas, "dxPeriod", 1.0)

    # RAMSES's base grid is a 2^levelmin CUBE: a non-cubic suite box (kh/orzag
    # slabs) cannot be represented — depositing it into a cubic GridIC would
    # silently produce wrong ICs. Fail loudly instead (see GRID_CODES_PLAN.md).
    dy = _f(gas, "dyPeriod", boxlen)
    dz = _f(gas, "dzPeriod", boxlen)
    if abs(dy - boxlen) > 1e-12 * boxlen or abs(dz - boxlen) > 1e-12 * boxlen:
        sys.exit(f"ERROR: non-cubic periodic box ({boxlen} x {dy} x {dz}) — "
                 f"RAMSES needs a cubic box (2^levelmin base grid). This test "
                 f"is unsupported for code=4 (first cut is cubic tests only).")

    levelmax = args.levelmax
    levelmin = args.levelmin
    if levelmin is None:
        levelmin = levelmax if args.grid == "uniform" else max(1, levelmax - 2)
    amr = (args.grid == "amr") and (levelmin < levelmax)

    # The GridIC is deposited at the BASE resolution (npx = 2**levelmin): for a
    # matched SPH-vs-grid comparison levelmin is set so 2**levelmin = the SPH
    # per-axis resolution -- the coarse regions have the same element count in
    # both codes, and AMR refinement (quasi-Lagrangian, mass-based) adds dynamic
    # range ABOVE the SPH IC resolution during the run, like SPH's Lagrangian
    # adaptivity. Finer-than-base cells at t=0 subsample the GridIC piecewise-
    # constant (the condinit volume-average degenerates to a single-cell lookup),
    # which adds no information but stays conservative.
    npx = 2 ** levelmin
    mhd = _has_nonzero_b(args.input)
    os.makedirs(args.rundir, exist_ok=True)

    import numpy as np
    box = np.array([boxlen, boxlen, boxlen])
    ic_path = os.path.join(args.rundir, "ic_gridic.bin")
    write_grid_ic(args.input, ic_path, npx=npx, method=args.method, mhd=mhd,
                  box=box, gamma=gamma)
    if mhd:
        print("  MHD IC (nonzero B): GridIC carries the 6 div-free face-B "
              "components; hlld MHD namelist written — run with ramses3d_mhd.")

    # quasi-Lagrangian refinement scale: the mean base-cell mass, from the
    # GridIC just written (exact -- petkova conserves the SPH total mass).
    mass_sph = None
    if amr:
        gic = read_grid_ic(ic_path)
        cellvol = (boxlen / npx) ** 3
        mass_sph = float(gic["rho"].sum() * cellvol) / float(npx ** 3)

    nml_path = os.path.join(args.rundir, "run.nml")
    write_namelist(nml_path, gas, boxlen, levelmin, levelmax, gamma, amr,
                   mass_sph=mass_sph, m_refine=args.m_refine, mhd=mhd,
                   np_ranks=args.np_ranks)
    print("Wrote %s (levelmin=%d levelmax=%d %s%s) and %s"
          % (ic_path, levelmin, levelmax,
             "AMR" if amr else "uniform", " MHD" if mhd else "", nml_path))


if __name__ == "__main__":
    main()
