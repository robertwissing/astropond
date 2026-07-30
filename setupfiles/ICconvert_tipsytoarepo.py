#!/usr/bin/env python
"""Convert a gasoline/tipsy IC into an AREPO moving-mesh IC (HDF5).

The AREPO analog of `ICconvert_tipsytosphexa.py`. It is a thin front end over the
`sph_interp` interpolation software (`IC_interpolate_to_grid.py --output-format
arepo`): read the tipsy snapshot (-> Particles via from_tipsy, periodic box +
dTuFac from the .param/.log), turn it into AREPO generators + per-cell conserved
quantities, and write the GADGET/AREPO HDF5 IC.

    ICconvert_tipsytoarepo.py <input> <output> [mode] [method]

  input    tipsy snapshot (e.g. datafiles/run.00000).
  output   AREPO IC HDF5 path (e.g. datafiles/run_arepo.hdf5).
  mode     particles (default; 1:1 copy of the SPH particles as generators, the
           most faithful conversion) | voronoi | amr | uniform (deposit the SPH
           fields onto that mesh and use its cell centres as generators) |
           arepo-grid (pure-Python equivalent of AREPO's ADDBACKGROUNDGRID: build
           the same bggrid oct-tree generators in Python -- ~1 cell per particle in
           sampled gas, oct-tree void-fill -- then deposit the SPH fields with the
           exact petkova kernel for a CORRECT magnetic field. No AREPO binary; the
           box is enlarged to >=2x the gas extent, so it is for isolated/collapse
           ICs, not periodic boxes).
  method   sph-gather | sph-scatter | petkova (default; exact, mass-conserving)
           -- deposit modes only. sph-scatter is AREPO's conservative kernel
           scatter (matches src/add_backgroundgrid); sph-gather is the fast,
           non-conserving cell-centre estimate.

Options: --fields (default rho,vx,vy,vz,Bx,By,Bz), --tu-fac (override dTuFac),
--npx, and the other IC_interpolate_to_grid knobs are forwarded.

If --param-out is given, a fresh AREPO simulation `param.txt` is also written,
with run parameters (timestepping, box, output cadence) taken from the gasoline
`.param` beside the input (see sph_interp/_arepo/arepo_param.py).  This replaces
the old approach of sed-editing AREPO's noh_3d example file.
"""
import argparse
import os

from ICconvert_common import find_sibling_param
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from IC_interpolate_to_grid import main as interp_main
from sph_interp._arepo.arepo_param import write_run_param
from sph_interp._arepo.arepo_bggrid import regrid_amr_petkova
from sph_interp.export import scale_hdf5_bfield, MU0_CODE_TO_GAUSS




def build_argv(args, output=None):
    # arepo-grid first builds the exact 1:1 particle IC (== particles mode); the
    # ADDBACKGROUNDGRID pass then regrids it (handled in __main__).
    output = output if output is not None else args.output
    argv = [args.input, output, "--output-format", "arepo",
            "--fields", args.fields]
    if args.mode in ("particles", "arepo-grid"):
        argv += ["--voronoi-particles"]
    else:
        argv += ["--grid", args.mode, "--method", args.method]
        if args.npx is not None:
            argv += ["--npx", str(args.npx)]
        # amr: --base-npx is the COARSE (level-0) grid per axis; set it to the SPH
        # per-axis resolution so the moving mesh starts as fine as the particle IC
        # and the octree refines UP toward --npx where the data warrants. --refine
        # picks the criterion (mass = quasi-Lagrangian, like RAMSES: refine a cell
        # whose summed particle mass exceeds --refine-value); --max-level caps depth.
        if args.mode == "amr":
            if args.base_npx is not None:
                argv += ["--base-npx", str(args.base_npx)]
            argv += ["--refine", args.refine]
            if args.refine_value is not None:
                argv += ["--refine-value", str(args.refine_value)]
            if args.max_level is not None:
                argv += ["--max-level", str(args.max_level)]
    if args.tu_fac is not None:
        argv += ["--tu-fac", str(args.tu_fac)]
    return argv


def parse(argv=None):
    pa = argparse.ArgumentParser(
        prog="ICconvert_tipsytoarepo",
        description="tipsy -> AREPO moving-mesh IC (via sph_interp).")
    pa.add_argument("input", help="tipsy snapshot")
    pa.add_argument("output", help="AREPO IC HDF5 path")
    pa.add_argument("mode", nargs="?", default="particles",
                    choices=["particles", "voronoi", "amr", "uniform",
                             "arepo-grid"],
                    help="particles=1:1 copy (default), a deposit mesh "
                         "(voronoi/amr/uniform), or arepo-grid (pure-Python "
                         "bggrid-equivalent oct-tree + petkova deposit; correct B, "
                         "no AREPO binary; isolated-object framing)")
    pa.add_argument("method", nargs="?", default="petkova",
                    choices=["sph-gather", "sph-scatter", "petkova", "sph"],
                    help="deposit method (sph-scatter = AREPO-style conservative)")
    pa.add_argument("--fields", default="rho,vx,vy,vz,Bx,By,Bz")
    pa.add_argument("--tu-fac", type=float, default=None)
    pa.add_argument("--npx", type=int, default=None,
                    help="cells per axis: uniform=grid resolution; amr=FINEST "
                         "resolution (base_npx x 2^max_level >= npx)")
    pa.add_argument("--base-npx", type=int, default=None,
                    help="amr only: COARSE (level-0) grid cells per axis "
                         "(default 16). Set to the SPH resolution to match it.")
    pa.add_argument("--refine", choices=["smoothing", "mass", "count"],
                    default="mass",
                    help="amr refinement criterion (default: mass = "
                         "quasi-Lagrangian, like RAMSES)")
    pa.add_argument("--refine-value", type=float, default=None,
                    help="amr refinement threshold (mass: cell-mass cutoff, "
                         "default = mean base-cell mass; count: particles/cell)")
    pa.add_argument("--max-level", type=int, default=None,
                    help="amr max refinement depth (default: derived from "
                         "--npx and --base-npx)")
    pa.add_argument("--param-out", default=None,
                    help="also write a fresh AREPO sim param.txt here, with run "
                         "parameters taken from the gasoline .param beside INPUT")
    pa.add_argument("--output-dir", default=None,
                    help="AREPO OutputDir to write into the generated param.txt "
                         "(default: '<param-out dir>/')")
    pa.add_argument("--boxsize", type=float, default=None,
                    help="override BoxSize in the param.txt (default: dxPeriod). "
                         "For non-cubic boxes pass the X period; y/z come from "
                         "compile-time LONG_Y/LONG_Z.")
    pa.add_argument("--gridsize", type=int, default=16,
                    help="arepo-grid only: base (coarse void-floor) grid cells per "
                         "axis == AREPO GridSize. The oct-tree refines DOWN to ~1 "
                         "cell/particle where the gas is; the void stays at this "
                         "resolution. Typically a small power of two (16, 32).")
    pa.add_argument("--include-occupied", dest="include_occupied",
                    action="store_true", default=True,
                    help="arepo-grid only: keep a generator at every particle's "
                         "leaf cell (default; finer, quasi-Lagrangian mesh).")
    pa.add_argument("--no-include-occupied", dest="include_occupied",
                    action="store_false",
                    help="arepo-grid only: emit ONLY the empty-leaf centres, "
                         "reproducing AREPO's ADDBACKGROUNDGRID mesh exactly.")
    return pa.parse_args(argv)


def write_param(args):
    """Write the AREPO simulation param.txt for this conversion (if requested)."""
    if not args.param_out:
        return
    gas_param = find_sibling_param(args.input)
    if gas_param is None:
        print("WARNING: no gasoline .param found beside %r; cannot write "
              "param.txt." % args.input)
        return
    # InitCondFile = the AREPO IC path without its extension (ICFormat 3 = HDF5).
    init_cond = os.path.splitext(args.output)[0]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (os.path.dirname(os.path.abspath(args.param_out)) or ".") \
            + os.sep
    write_run_param(gas_param, args.param_out, init_cond, output_dir,
                    boxsize=args.boxsize)
    print("Wrote AREPO param.txt -> %s (run params from %s)"
          % (args.param_out, gas_param))


if __name__ == "__main__":
    args = parse()
    if args.mode == "arepo-grid":
        # Pure-Python ADDBACKGROUNDGRID equivalent (no AREPO binary): build the
        # bggrid oct-tree generators, deposit the SPH fields with the exact petkova
        # kernel (correct B, unlike AREPO's own scatter), write the AREPO IC. The
        # framing enlarges the box to >= 2x the gas extent (isolated-object mode).
        fields = [f for f in args.fields.split(",") if f]
        print("arepo-grid (pure-Python bggrid octree + petkova): %s -> %s "
              "(gridsize=%d)" % (args.input, args.output, args.gridsize))
        _, boxsize_out = regrid_amr_petkova(
            args.input, args.output, args.gridsize, fields, tu_fac=args.tu_fac,
            include_occupied=args.include_occupied)
        # petkova to_arepo writes B in CODE units -> scale to AREPO's Gauss
        # convention (the deposit path does not round-trip Gauss as the binary did).
        if scale_hdf5_bfield(args.output, MU0_CODE_TO_GAUSS):
            print("Scaled IC MagneticField by sqrt(4pi)=%.6f (mu0=1 code -> "
                  "AREPO Gauss)" % MU0_CODE_TO_GAUSS)
        # The framing enlarged the box (>= 2x gas extent); the run param MUST use the
        # box actually written, else IC and param disagree.
        if args.boxsize is not None and \
                abs(boxsize_out - args.boxsize) > 1e-9 * max(1.0, args.boxsize):
            print("  box enlarged: %g -> %g (>=2x gas extent); run param.txt will "
                  "use %g." % (args.boxsize, boxsize_out, boxsize_out))
        args.boxsize = boxsize_out
    else:
        interp_main(build_argv(args))
        # AREPO reads the IC MagneticField as GAUSS (io_fields.c: B /= sqrt(4pi);
        # internal field is Heaviside-Lorentz) -> convert the mu0=1 tipsy B to
        # Gauss so the internal field matches gasoline. No-op for hydro ICs.
        if scale_hdf5_bfield(args.output, MU0_CODE_TO_GAUSS):
            print("Scaled IC MagneticField by sqrt(4pi)=%.6f (mu0=1 code -> "
                  "AREPO Gauss)" % MU0_CODE_TO_GAUSS)
    write_param(args)
