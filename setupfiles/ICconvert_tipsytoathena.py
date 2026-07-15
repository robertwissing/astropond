#!/usr/bin/env python
"""Convert a gasoline/tipsy IC into an Athena++ GridIC + athinput (code=7).

Same strategy as RAMSES (code=4): deposit the SPH field onto a fine uniform
grid (sph_interp.gridic.write_grid_ic, petkova) and write the canonical GridIC
binary `ic_gridic.bin` into the run dir — Athena++ reads it via the suite's
`from_gridic` problem generator (codes/athena++/patches/from_gridic.cpp). This
script also generates the matching `athinput.gridic`: the mesh MUST equal the
GridIC resolution (the pgen enforces it), box centred like gasoline
([-L/2, L/2] per axis), periodic BCs, .athdf (prim) outputs on the .param
cadence.

    ICconvert_tipsytoathena.py <input> <rundir> --npx N [--method petkova|sph]
                               [--boxsize L] [--np R]

  input    tipsy snapshot (datafiles/run.00000).
  rundir   Athena++ run dir; writes <rundir>/ic_gridic.bin + <rundir>/athinput.gridic.

NON-CUBIC (slab) boxes are supported — unlike RAMSES, Athena++ is natively
per-axis. --npx N sets the X-axis cell count; the other axes get
npx_i = round(N * L_i / L_x) cells, so cells are exactly cubic when the box
aspect is rational (kh's 1x2 plane) and nearest-integer slightly anisotropic
otherwise (the thin 4*sqrt(6)/nx z-slabs; petkova deposits are exact either
way).

MHD: a genuinely NONZERO BField aux writes the 6 div-free face-B components
into the GridIC (project_solenoidal) — run with the athena_mhd binary (CT);
an all-zero aux stays hydro. Parallelism is MPI over MeshBlocks: each axis is
split into blocks whose size DIVIDES that axis' cell count (Athena++ hard
requirement), blocks >= 8 cells/axis where the axis allows it, until there
are >= --np blocks; extra ranks beyond the block count would abort Athena++,
so runtest caps -np at the block count it prints.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph_interp.gridic import write_grid_ic
from ICconvert_tipsytoramses import _find_param, _param, _f
from ICconvert_common import has_nonzero_b as _has_nonzero_b


def _axis_blocks(n):
    """Feasible meshblock sizes for an axis of n cells: divisors of n, >= 8
    (Athena++ needs mesh % block == 0); the full axis is always feasible."""
    sizes = [d for d in range(8, n) if n % d == 0]
    return sorted(set(sizes + [n]), reverse=True)


def _meshblock(npx3, np_ranks):
    """Per-axis meshblock sizes: greedily split the axis with the largest
    current block to its next smaller feasible size until #blocks >= np_ranks
    (or no axis can split further). Returns (mb1, mb2, mb3)."""
    feas = [_axis_blocks(n) for n in npx3]
    idx = [0, 0, 0]                      # index into each axis' feasible list
    def nblocks():
        return (npx3[0] // feas[0][idx[0]]) * (npx3[1] // feas[1][idx[1]]) \
            * (npx3[2] // feas[2][idx[2]])
    while nblocks() < max(1, np_ranks):
        cand = [a for a in range(3) if idx[a] + 1 < len(feas[a])]
        if not cand:
            break
        a = max(cand, key=lambda a: feas[a][idx[a]])
        idx[a] += 1
    return tuple(feas[a][idx[a]] for a in range(3))


def write_athinput(path, gas, box, npx3, gamma, mhd, np_ranks):
    dDelta = _f(gas, "dDelta", 0.01)
    nSteps = int(_f(gas, "nSteps", 0) or 0)
    iOut = int(_f(gas, "iOutInterval", 1) or 1)
    tlim = (nSteps or 1) * dDelta
    dtout = iOut * dDelta
    mb = _meshblock(npx3, np_ranks)
    nblocks = (npx3[0] // mb[0]) * (npx3[1] // mb[1]) * (npx3[2] // mb[2])
    mesh = ""
    for ax in range(3):
        half = 0.5 * box[ax]
        mesh += f"""
nx{ax + 1}        = {npx3[ax]}
x{ax + 1}min      = {-half:.8g}
x{ax + 1}max      = {half:.8g}
ix{ax + 1}_bc     = periodic
ox{ax + 1}_bc     = periodic
"""
    txt = f"""<comment>
problem   = testsuite GridIC ({'MHD' if mhd else 'hydro'})
configure = --prob=from_gridic {'-b ' if mhd else ''}-mpi -hdf5

<job>
problem_id = gridic

<output1>
file_type  = hst
dt         = {dtout:.8g}

<output2>
file_type  = hdf5
variable   = prim
dt         = {dtout:.8g}

<time>
cfl_number = 0.4
nlim       = -1
tlim       = {tlim:.8g}
integrator = vl2
xorder     = 2
ncycle_out = 100

<mesh>{mesh}
<meshblock>
nx1 = {mb[0]}
nx2 = {mb[1]}
nx3 = {mb[2]}

<hydro>
gamma = {gamma:.10g}

<problem>
input_filename = ic_gridic.bin
"""
    with open(path, "w") as f:
        f.write(txt)
    return nblocks


def main(argv=None):
    pa = argparse.ArgumentParser(prog="ICconvert_tipsytoathena")
    pa.add_argument("input")
    pa.add_argument("rundir")
    pa.add_argument("--npx", type=int, required=True,
                    help="X-axis cell count (= the SPH per-axis resolution "
                         "Nres for a matched comparison); other axes scale "
                         "by the box aspect; the athinput mesh matches")
    pa.add_argument("--method", choices=["petkova", "sph"], default="petkova")
    pa.add_argument("--boxsize", type=float, default=None,
                    help="force a CUBIC box of this side (default: the "
                         ".param dx/dy/dzPeriod, non-cubic allowed)")
    pa.add_argument("--np", type=int, default=1, dest="np_ranks",
                    help="MPI ranks the run will use; the mesh is split into "
                         "at least this many meshblocks (block >= 8/axis "
                         "where the axis allows)")
    args = pa.parse_args(argv)

    gas = _param(_find_param(args.input))
    gamma = _f(gas, "dConstGamma", 5.0 / 3.0)
    if args.boxsize is not None:
        box3 = [args.boxsize] * 3
    else:
        lx = _f(gas, "dxPeriod", 1.0)
        box3 = [lx, _f(gas, "dyPeriod", lx), _f(gas, "dzPeriod", lx)]

    # per-axis cell counts from the X resolution and the box aspect; cells are
    # exactly cubic for rational aspects, nearest-integer anisotropic else
    npx3 = tuple(max(4, int(round(args.npx * box3[ax] / box3[0])))
                 for ax in range(3))

    mhd = _has_nonzero_b(args.input)
    os.makedirs(args.rundir, exist_ok=True)

    import numpy as np
    box = np.array(box3)
    ic_path = os.path.join(args.rundir, "ic_gridic.bin")
    write_grid_ic(args.input, ic_path, npx=npx3, method=args.method,
                  mhd=mhd, box=box, gamma=gamma)

    ain_path = os.path.join(args.rundir, "athinput.gridic")
    nblocks = write_athinput(ain_path, gas, box3, npx3, gamma, mhd,
                             args.np_ranks)
    # runtest caps mpirun -np at this (ranks > meshblocks aborts Athena++)
    print("meshblocks=%d" % nblocks)
    print("Wrote %s (npx=%dx%dx%d%s, %d meshblocks) and %s"
          % ((ic_path,) + npx3 + (" MHD" if mhd else "", nblocks, ain_path)))


if __name__ == "__main__":
    main()
