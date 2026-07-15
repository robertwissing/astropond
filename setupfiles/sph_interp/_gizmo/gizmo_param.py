"""Generate a GIZMO text parameterfile from a gasoline/tipsy `.param`.

The GIZMO analog of `sph_interp/_arepo/arepo_param.py`. GIZMO is a particle code
that reads a GADGET type-0 HDF5 IC (ICFormat 3) -- the SAME file the AREPO export
writer produces -- so no IC format work is needed here; this module only writes
the run parameterfile.

Mapping gasoline `.param` -> GIZMO parameterfile (modelled on
gizmo/scripts/params.txt and the test_problems/*.params examples):
  dDelta                  -> MaxSizeTimestep
  nSteps   * dDelta       -> TimeMax
  iOutInterval * dDelta   -> TimeBetSnapshot
  nSmooth                 -> DesNumNgb (the SPH kernel neighbour number)
  dxPeriod                -> BoxSize  (x period; non-cubic y/z need GIZMO compiled
                             with BOX_LONG_Y/BOX_LONG_Z, like AREPO's LONG_*)
  dConstGamma             -> checked against GIZMO's compile-time EOS_GAMMA (5/3)

Everything else (periodic, code units = 1, no cooling/SF) is a suite-wide default
matching how the tests are configured. With SELFGRAVITY_OFF the softening
parameters are inert but are written for parser completeness. With MAGNETIC the IC
carries B in the HDF5, so the BiniX/Y/Z seed fields are 0.

Self-gravity (isolated collapse tests: evrard, ...): keyed off `bDoGravity=1` in
the gasoline .param (auto-detected, or forced via `grav=`). GIZMO's
`GravityConstantInternal` = the value of G in CODE units; 0 means "derive PHYSICAL
G from UnitLength/Mass/Velocity", which is WRONG for these G=1 code-unit tests, so
gravity runs set it to 1.0. The run must use the GIZMO_grav binary (Config_grav.sh:
no BOX_PERIODIC, no SELFGRAVITY_OFF, + ADAPTIVE_GRAVSOFT_FORGAS); Softening_Type0
is then the adaptive gas-softening floor.

The gasoline-param reader is shared with the AREPO generator.
"""
import os

from sph_interp._arepo.arepo_param import read_gasoline_param, _f, _i

#: GIZMO compile-time adiabatic index (Config EOS_GAMMA, default 5/3).
GIZMO_COMPILE_GAMMA = 5.0 / 3.0


def build_param_values(gas, init_cond_file, output_dir, boxsize=None,
                       mhd=False, grav=None, extra=None):
    """Ordered list of (key, value) GIZMO parameters from a parsed gasoline dict.

    init_cond_file : GIZMO IC path WITHOUT extension (ICFormat 3 = HDF5).
    output_dir     : GIZMO OutputDir (snapshots land here).
    boxsize        : override BoxSize (else dxPeriod); the X period.
    mhd            : include the BiniX/Y/Z seed fields (0; the IC carries B).
    grav           : self-gravity on (GravityConstantInternal=1.0, G in code units);
                     None -> auto-detect from bDoGravity in the gasoline .param.
                     Requires the GIZMO_grav binary (Config_grav.sh).
    extra          : dict of literal key->value overrides (last word wins).
    """
    dDelta = _f(gas, "dDelta")
    if dDelta is None or dDelta <= 0:
        raise ValueError("gasoline .param has no usable dDelta (base timestep)")
    nSteps = _i(gas, "nSteps", 0)
    iOut = _i(gas, "iOutInterval", nSteps or 1)
    nSmooth = _i(gas, "nSmooth", 32)
    if grav is None:
        grav = _i(gas, "bDoGravity", 0) == 1

    if boxsize is None:
        boxsize = _f(gas, "dxPeriod", 1.0)

    time_max = nSteps * dDelta if nSteps else dDelta
    time_bet_snap = iOut * dDelta if iOut else time_max

    P = [
        ("% GIZMO parameter file generated from a gasoline .param by", None),
        ("%   sph_interp/_gizmo/gizmo_param.py  (do not hand-edit; regenerated)", None),
        (None, None),
        ("InitCondFile", init_cond_file),
        ("OutputDir", output_dir),
        ("ICFormat", 3),
        ("SnapFormat", 3),
        ("RestartFile", "restart"),
        ("SnapshotFileBase", "snapshot"),
        ("OutputListOn", 0),
        ("OutputListFilename", "output_times.txt"),
        ("NumFilesPerSnapshot", 1),
        ("NumFilesWrittenInParallel", 1),
        (None, None),
        ("TimeOfFirstSnapshot", 0.0),
        ("TimeBetSnapshot", time_bet_snap),
        ("TimeLimitCPU", 90000),
        ("CpuTimeBetRestartFile", 9000),
        ("MaxMemSize", 8000),
        ("PartAllocFactor", 5.0),
        ("BufferSize", 100),
        (None, None),
        ("TimeBegin", 0.0),
        ("TimeMax", time_max),
        ("MaxSizeTimestep", dDelta),
        ("MinSizeTimestep", 1e-12),
        (None, None),
        ("UnitLength_in_cm", 1.0),
        ("UnitMass_in_g", 1.0),
        ("UnitVelocity_in_cm_per_s", 1.0),
        ("UnitMagneticField_in_gauss", 1.0),
        # G in CODE units: 1.0 for the G=M=R=1 collapse tests (NOT 0.0, which would
        # make GIZMO derive physical G from the unit system). Gravity-off runs keep 0.
        ("GravityConstantInternal", 1.0 if grav else 0.0),
        (None, None),
        ("ComovingIntegrationOn", 0),
        ("BoxSize", boxsize),
        (None, None),
        ("InitGasTemp", 0.0),
        ("MinGasTemp", 0.0),
        ("DesNumNgb", nSmooth),
        ("MaxHsml", 1e10),
        ("MinGasHsmlFractional", 0.0),
        (None, None),
    ]
    # Softening_Type0 = gas. Under ADAPTIVE_GRAVSOFT_FORGAS (the GIZMO_grav build)
    # the gas softening tracks the kernel h and Type0 is only the floor; give it the
    # setup eps (0.02) and an effectively-unclamped MaxPhysLimit so it can grow with
    # the collapsing core. Gravity-off runs leave the inert defaults.
    gas_soft = 0.02 if grav else 0.001
    gas_soft_max = 1e10 if grav else 0.0005
    for t in range(6):
        P.append(("Softening_Type%d" % t, gas_soft if t == 0 else 0.001))
    for t in range(6):
        P.append(("Softening_Type%d_MaxPhysLimit" % t,
                  gas_soft_max if t == 0 else 0.0005))
    if mhd:
        P += [(None, None),
              ("BiniX", 0.0), ("BiniY", 0.0), ("BiniZ", 0.0)]

    if extra:
        existing = {k for k, _ in P if k}
        for k, v in extra.items():
            if k in existing:
                P = [(kk, v if kk == k else vv) for kk, vv in P]
            else:
                P.append((k, v))
    return P


def _format(P):
    out = []
    for k, v in P:
        if k is None and v is None:
            out.append("")
        elif v is None:
            out.append(k)
        else:
            vs = "%.12g" % v if isinstance(v, float) else str(v)
            out.append("%-30s %s" % (k, vs))
    return "\n".join(out) + "\n"


def write_run_param(gas_param_path, out_path, init_cond_file, output_dir,
                    boxsize=None, mhd=False, grav=None, extra=None,
                    gamma_warn=True):
    """Write a GIZMO parameterfile at `out_path` from the gasoline `.param`.
    Returns the parsed gasoline dict."""
    gas = read_gasoline_param(gas_param_path)
    if gamma_warn:
        g = _f(gas, "dConstGamma")
        if g is not None and abs(g - GIZMO_COMPILE_GAMMA) > 1e-3:
            print("WARNING: gasoline dConstGamma=%.6g but GIZMO compiled with "
                  "EOS_GAMMA=%.6g (5/3). Rebuild GIZMO with EOS_GAMMA=%.6g."
                  % (g, GIZMO_COMPILE_GAMMA, g))
    P = build_param_values(gas, init_cond_file, output_dir, boxsize=boxsize,
                           mhd=mhd, grav=grav, extra=extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(_format(P))
    return gas
