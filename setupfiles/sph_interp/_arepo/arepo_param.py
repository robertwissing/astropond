"""Generate a fresh AREPO simulation `param.txt` from a gasoline/tipsy `.param`.

The old code path sed-edited AREPO's *noh_3d* example parameter file, which
carried noh's timestepping (`MaxSizeTimestep 0.01`, `CourantFac 0.3`) regardless
of the test being run.  For sedov (`dDelta=0.0002`) that let AREPO take steps 50x
too large over the critical early blast, so the solution was wrong; orzag was
likewise mis-stepped.  This module instead builds a COMPLETE param file from
scratch and fills the run-specific values from the gasoline `.param` that the
test suite already writes for every IC.

Mapping gasoline `.param` -> AREPO `param.txt`:
  dDelta                         -> MaxSizeTimestep   (gasoline base/rung-0 step)
  nSteps   * dDelta              -> TimeMax
  iOutInterval * dDelta          -> TimeBetSnapshot
  iLogInterval * dDelta          -> TimeBetStatistics
  dxPeriod                       -> BoxSize           (y/z via compile-time LONG_*)
  dConstGamma                    -> checked against AREPO's compile-time GAMMA (5/3)

DesNumNgb is left at AREPO's standard 64 and is NOT mapped from gasoline's
nSmooth: AREPO's moving-mesh finite-volume hydro takes its neighbours from the
Voronoi tessellation (cells sharing a face), so DesNumNgb does not set the hydro
resolution the way nSmooth does in SPH -- it is essentially inert here.

Everything else (no gravity, no cooling, periodic, code units) is a suite-wide
default, matching how the tests are configured.  The full key set mirrors the
one AREPO's strict parser expects for the suite's Config.sh, so AREPO never
complains about a missing parameter.
"""
import os

# AREPO's compile-time adiabatic index (Config.sh GAMMA, default 5/3).  A test
# whose dConstGamma differs needs AREPO rebuilt with -DGAMMA=... ; we warn.
AREPO_COMPILE_GAMMA = 5.0 / 3.0


def read_gasoline_param(path):
    """Parse a gasoline `.param` (`key = value`, `#` comments) into a dict of
    strings.  Repeated keys keep the FIRST occurrence (the suite concatenates a
    few IC variants into one .param; they share run parameters)."""
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and k not in vals:
                vals[k] = v
    return vals


def _f(vals, key, default=None):
    try:
        return float(vals[key])
    except (KeyError, ValueError):
        return default


def _i(vals, key, default=None):
    v = _f(vals, key, None)
    return int(v) if v is not None else default


def build_param_values(gas, init_cond_file, output_dir,
                       boxsize=None, grav=None, extra=None):
    """Build the ordered list of (key, value) AREPO parameters from a parsed
    gasoline param dict `gas`.

    init_cond_file  AREPO IC path WITHOUT extension (ICFormat 3 = HDF5).
    output_dir      AREPO OutputDir (snapshots land here).
    boxsize         override BoxSize (else dxPeriod); the X period — non-cubic
                    y/z are handled by compile-time LONG_Y/LONG_Z, not here.
    extra           dict of literal AREPO key->value overrides (last word wins).
    """
    dDelta = _f(gas, "dDelta")
    if dDelta is None or dDelta <= 0:
        raise ValueError("gasoline .param has no usable dDelta (base timestep)")
    nSteps = _i(gas, "nSteps", 0)
    iOut = _i(gas, "iOutInterval", nSteps or 1)
    iLog = _i(gas, "iLogInterval", 1)

    if boxsize is None:
        boxsize = _f(gas, "dxPeriod", 1.0)

    # Self-gravity (isolated collapse: evrard, polytrope). None -> auto-detect from
    # bDoGravity in the gasoline .param (matches the Gadget-family generators). Needs
    # the Arepo_grav binary (Config_grav.sh: SELFGRAVITY + GRAVITY_NOT_PERIODIC +
    # ADAPTIVE_HYDRO_SOFTENING). The mesh box stays periodic; gravity is open.
    if grav is None:
        grav = _i(gas, "bDoGravity", 0) == 1

    time_max = nSteps * dDelta if nSteps else dDelta
    time_bet_snap = iOut * dDelta if iOut else time_max
    time_bet_stat = max(iLog, 1) * dDelta

    P = [
        ("% AREPO parameter file generated from a gasoline .param by", None),
        ("%   sph_interp/_arepo/arepo_param.py  (do not hand-edit; regenerated)", None),
        (None, None),
        ("InitCondFile", init_cond_file),
        ("ICFormat", 3),
        (None, None),
        ("OutputDir", output_dir),
        ("SnapshotFileBase", "snap"),
        ("SnapFormat", 3),
        ("NumFilesPerSnapshot", 1),
        ("NumFilesWrittenInParallel", 1),
        (None, None),
        ("ResubmitOn", 0),
        ("ResubmitCommand", "my-scriptfile"),
        ("OutputListFilename", "ol"),
        ("OutputListOn", 0),
        (None, None),
        ("CoolingOn", 0),
        ("StarformationOn", 0),
        (None, None),
        ("Omega0", 0.0),
        ("OmegaBaryon", 0.0),
        ("OmegaLambda", 0.0),
        ("HubbleParam", 1.0),
        (None, None),
        ("BoxSize", boxsize),
        ("PeriodicBoundariesOn", 1),
        ("ComovingIntegrationOn", 0),
        (None, None),
        ("MaxMemSize", 8000),
        (None, None),
        ("TimeOfFirstSnapshot", 0.0),
        ("CpuTimeBetRestartFile", 9000),
        ("TimeLimitCPU", 90000),
        (None, None),
        ("TimeBetStatistics", time_bet_stat),
        ("TimeBegin", 0.0),
        ("TimeMax", time_max),
        ("TimeBetSnapshot", time_bet_snap),
        (None, None),
        ("UnitVelocity_in_cm_per_s", 1.0),
        ("UnitLength_in_cm", 1.0),
        ("UnitMass_in_g", 1.0),
        # G in CODE units: 1.0 for the G=M=R=1 collapse tests (0.0 = derive physical
        # G from the unit system, wrong here). Gravity-off box tests keep 0.0.
        ("GravityConstantInternal", 1.0 if grav else 0.0),
        (None, None),
        ("ErrTolIntAccuracy", 0.1),
        ("ErrTolTheta", 0.1),
        ("ErrTolForceAcc", 0.1),
        (None, None),
        ("MaxSizeTimestep", dDelta),
        # MinSizeTimestep is AREPO's local-timestep FLOOR (a safety catch), not
        # gasoline's iMaxRung rung floor -- the two do not map cleanly, so keep a
        # fixed small value.
        ("MinSizeTimestep", 1e-7),
        # CourantFac is AREPO's own (its Courant factor is NOT defined the same
        # way as gasoline's dEtaCourant), so keep AREPO's standard value.
        ("CourantFac", 0.3),
        (None, None),
        ("LimitUBelowThisDensity", 0.0),
        ("LimitUBelowCertainDensityToThisValue", 0.0),
        ("DesNumNgb", 64),
        ("MaxNumNgbDeviation", 2),
        (None, None),
        ("MultipleDomains", 2),
        ("TopNodeFactor", 4),
        ("ActivePartFracForNewDomainDecomp", 0.1),
        (None, None),
        ("TypeOfTimestepCriterion", 0),
        ("TypeOfOpeningCriterion", 1),
        ("GasSoftFactor", 0.01),
        (None, None),
    ]
    for t in range(6):
        P.append(("SofteningComovingType%d" % t, 0.1))
    P.append((None, None))
    for t in range(6):
        P.append(("SofteningMaxPhysType%d" % t, 0.1))
    P.append((None, None))
    P.append(("SofteningTypeOfPartType0", 0))
    for t in range(1, 6):
        P.append(("SofteningTypeOfPartType%d" % t, 1))
    if grav:
        # ADAPTIVE_HYDRO_SOFTENING (Config_grav) REQUIRES these: the gas softening =
        # GasSoftFactor * cell_radius, binned into NSOFTTYPES_HYDRO logarithmic steps
        # spaced by AdaptiveHydroSofteningSpacing and floored at MinimumComovingHydro-
        # Softening. Standard AREPO values; AREPO aborts if they are absent.
        P += [
            (None, None),
            ("MinimumComovingHydroSoftening", 1e-4),
            ("AdaptiveHydroSofteningSpacing", 1.2),
        ]
    P += [
        (None, None),
        ("InitGasTemp", 0.0),
        ("MinGasTemp", 0.0),
        ("MinEgySpec", 0.0),
        ("MinimumDensityOnStartUp", 0.0),
        (None, None),
        ("CellShapingSpeed", 0.5),
        ("CellMaxAngleFactor", 2.25),
    ]

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
        elif v is None:                       # bare comment line
            out.append(k)
        else:
            if isinstance(v, float):
                vs = "%.12g" % v
            else:
                vs = str(v)
            out.append("%-37s %s" % (k, vs))
    return "\n".join(out) + "\n"


def write_mesh_param(out_path, boxsize, maxmem, init_cond_file="snap",
                     output_dir="./output/"):
    """Write a complete AREPO `param.txt` for a mode-14 mesh-only dump (used by
    the Voronoi mesh backend in `arepo_mesh.py`).  Timestepping is irrelevant for
    mode 14 -- AREPO builds the mesh and exits -- so dummy run parameters are
    supplied; only `BoxSize` and `MaxMemSize` actually matter."""
    gas = {"dDelta": "1.0", "nSteps": "2", "iOutInterval": "1",
           "iLogInterval": "1", "dxPeriod": str(boxsize)}
    P = build_param_values(gas, init_cond_file, output_dir, boxsize=boxsize,
                           extra={"MaxMemSize": int(maxmem)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        f.write(_format(P))


def write_bggrid_param(out_path, boxsize, gridsize, maxmem,
                       init_cond_file="snap", output_dir="./output/"):
    """Write a complete AREPO `param.txt` for an ADDBACKGROUNDGRID regrid pass
    (used by the `arepo-grid` converter mode via `arepo_bggrid.py`).  Like the
    mode-14 mesh dump this does NOT simulate -- AREPO reads the particle IC at
    `init_cond_file`, regrids it, writes `<output_dir>/snap_000` and exits -- so
    the timestepping values are dummies; only `BoxSize`, `GridSize`, `MaxMemSize`
    and the IC/output paths matter.  `GridSize` is the target finest background
    resolution per axis (<= ADDBACKGROUNDGRIDMAX = 256); AREPO refines down to
    ~1 cell per particle where particles exist and leaves voids coarse."""
    gas = {"dDelta": "1.0", "nSteps": "2", "iOutInterval": "1",
           "iLogInterval": "1", "dxPeriod": str(boxsize)}
    P = build_param_values(gas, init_cond_file, output_dir, boxsize=boxsize,
                           extra={"MaxMemSize": int(maxmem),
                                  "GridSize": int(gridsize)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        f.write(_format(P))


def write_run_param(gas_param_path, out_path, init_cond_file, output_dir,
                    boxsize=None, grav=None, extra=None, gamma_warn=True):
    """Write a fresh AREPO `param.txt` at `out_path` from the gasoline `.param`
    at `gas_param_path`.  Returns the parsed gasoline dict (handy for the caller
    to reuse, e.g. for the LONG_Y/LONG_Z build decision)."""
    gas = read_gasoline_param(gas_param_path)
    if gamma_warn:
        g = _f(gas, "dConstGamma")
        if g is not None and abs(g - AREPO_COMPILE_GAMMA) > 1e-3:
            print("WARNING: gasoline dConstGamma=%.6g but AREPO was compiled "
                  "with GAMMA=%.6g (5/3). Rebuild AREPO with -DGAMMA=%.6g for a "
                  "correct EOS." % (g, AREPO_COMPILE_GAMMA, g))
    P = build_param_values(gas, init_cond_file, output_dir,
                           boxsize=boxsize, grav=grav, extra=extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        f.write(_format(P))
    return gas
