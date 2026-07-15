"""Generate an OpenGadget3 text parameterfile from a gasoline/tipsy `.param`.

The OpenGadget3 analog of `sph_interp/_gizmo/gizmo_param.py`. OpenGadget3 is a
Gadget-family particle code that reads a GADGET type-0 HDF5 IC (ICFormat 3) -- the
SAME file the AREPO/GIZMO export writer produces -- so no IC format work is needed
here; this module only writes the run parameterfile.

The key set and default values are modelled on the project's own known-good MHD
reference `tests/MHDTest/paramfile_wc6.par` (a GADGET-2-style parameterfile read by
OpenGadget3's `setup_parameter` framework). Mapping gasoline `.param` ->
OpenGadget3:
  dDelta                  -> MaxSizeTimestep
  nSteps   * dDelta       -> TimeMax
  iOutInterval * dDelta   -> TimeBetSnapshot
  nSmooth                 -> DesNumNgb  (SPH kernel neighbour number)
  dxPeriod                -> BoxSize    (x period; non-cubic y/z need OpenGadget3
                             compiled with LONG_Y/LONG_Z, like GIZMO's BOX_LONG_*)

Self-gravity is neutralised via GravityConstantInternal=0 (the suite's box tests
are gravity-free, matching paramfile_wc6.par). Units are code units = 1. With
`mhd=True` the magnetic block is emitted; those tags require OpenGadget3 built with
MAGNETIC + ARTIFICIAL_MAGNETIC_DISSIPATION + DIVBCLEANING_TRICCO +
ARTIFICIAL_CONDUCTIVITY (see codes/opengadget3/Config_mhd.sh -- keep in sync).

Output goes to OutputDir as `<base>_NNN.hdf5` (SnapFormat 3); OutputDir is
concatenated to the filename WITHOUT a separator by OpenGadget3, so it must end in
'/'. The gasoline-param reader is shared with the AREPO/GIZMO generators.
"""
import os

from sph_interp._arepo.arepo_param import read_gasoline_param, _f, _i

#: OpenGadget3 compile-time adiabatic index (default 5/3 for these tests).
OPENGADGET3_COMPILE_GAMMA = 5.0 / 3.0


def build_param_values(gas, init_cond_file, output_dir, boxsize=None,
                       mhd=False, grav=None, extra=None):
    """Ordered list of (key, value) OpenGadget3 parameters from a parsed gasoline
    dict. `key=None,value=None` -> blank line; `value=None` -> bare comment line.

    init_cond_file : IC path WITHOUT extension (ICFormat 3 -> OpenGadget3 appends
                     '.hdf5' itself).
    output_dir     : OutputDir (snapshots land here; MUST end in '/').
    boxsize        : override BoxSize (else dxPeriod); the X period.
    mhd            : include the magnetic block (needs the MHD Config flags).
    extra          : dict of literal key->value overrides (last word wins).
    """
    dDelta = _f(gas, "dDelta")
    if dDelta is None or dDelta <= 0:
        raise ValueError("gasoline .param has no usable dDelta (base timestep)")
    nSteps = _i(gas, "nSteps", 0)
    iOut = _i(gas, "iOutInterval", nSteps or 1)
    nSmooth = _i(gas, "nSmooth", 64)
    # Self-gravity (isolated collapse tests: evrard, polytrope). None -> auto-detect
    # from bDoGravity in the gasoline .param (matches gizmo_param.py). Requires the
    # OpenGadget3_grav binary (Config_grav.sh: gravity on, non-periodic, adaptive
    # gas softening). Isolated -> PeriodicBoundariesOn=0 (Config_grav has no PERIODIC).
    if grav is None:
        grav = _i(gas, "bDoGravity", 0) == 1

    if boxsize is None:
        boxsize = _f(gas, "dxPeriod", 1.0)
    if not output_dir.endswith(os.sep):
        output_dir = output_dir + os.sep

    time_max = nSteps * dDelta if nSteps else dDelta
    time_bet_snap = iOut * dDelta if iOut else time_max

    P = [
        ("% OpenGadget3 parameterfile generated from a gasoline .param by", None),
        ("%   sph_interp/_opengadget3/opengadget3_param.py (regenerated; do not edit)", None),
        (None, None),
        ("InitCondFile", init_cond_file),
        ("OutputDir", output_dir),
        ("SnapshotFileBase", "snap"),
        ("ICFormat", 3),
        ("SnapFormat", 3),
        ("NumFilesPerSnapshot", 1),
        ("NumFilesWrittenInParallel", 1),
        (None, None),
        ("EnergyFile", "energy.txt"),
        ("InfoFile", "info.txt"),
        ("TimingsFile", "timings.txt"),
        ("CpuFile", "cpu.txt"),
        ("TimebinFile", "timebin.txt"),
        ("RestartFile", "restart"),
        (None, None),
        ("TimeLimitCPU", 86400),
        ("CpuTimeBetRestartFile", 40000),
        ("ComovingIntegrationOn", 0),
        ("CoolingOn", 0),
        ("StarformationOn", 0),
        ("MaxMemSize", 8000),
        (None, None),
        ("TimeBegin", 0.0),
        ("TimeMax", time_max),
        ("Omega0", 0.0),
        ("OmegaLambda", 0.0),
        ("OmegaBaryon", 0.0),
        ("HubbleParam", 0.7),
        ("BoxSize", boxsize),
        # Isolated self-gravity runs are non-periodic (Config_grav has no PERIODIC);
        # box tests are periodic. Must match the compiled binary or OpenGadget3 aborts.
        ("PeriodicBoundariesOn", 0 if grav else 1),
        (None, None),
        ("OutputListOn", 0),
        ("OutputListFilename", "output_times.txt"),
        ("TimeBetSnapshot", time_bet_snap),
        ("TimeOfFirstSnapshot", 0.0),
        ("TimeBetStatistics", time_bet_snap),
        (None, None),
        ("TypeOfTimestepCriterion", 0),
        ("ErrTolIntAccuracy", 0.1),
        ("MaxSizeTimestep", dDelta),
        ("MinSizeTimestep", 1e-8),
        (None, None),
        ("ErrTolTheta", 0.0),
        ("ErrTolForceAcc", 0.05),
        ("MaxRMSDisplacementFac", 0.25),
        (None, None),
        ("% --- SPH ---", None),
        ("DesNumNgb", nSmooth),
        ("ArtBulkViscConst", 1.0),
        ("CourantFac", 0.15),
        ("ArtCondConstant", 0.2),
        (None, None),
        ("PartAllocFactor", 3.0),
        ("BufferSize", 100),
        (None, None),
        ("% --- units (code units = 1) ---", None),
        ("UnitLength_in_cm", 1.0),
        ("UnitMass_in_g", 1.0),
        ("UnitVelocity_in_cm_per_s", 1.0),
        # G in CODE units: 1.0 for the G=M=R=1 collapse tests (NOT 0.0, which under a
        # gravity build would derive PHYSICAL G from the unit system). Gravity-off box
        # tests keep 0.0 (with the NOGRAVITY build gravity is off regardless).
        ("GravityConstantInternal", 1.0 if grav else 0.0),
        (None, None),
        ("MinGasHsmlFractional", 0.0),
    ]
    # SofteningGas is the gas gravitational softening. Under ADAPTIVE_GRAVSOFT_FORGAS
    # (+ _HSML, the OpenGadget3_grav build) it tracks the SPH kernel h and this is the
    # floor; give it the evrard setup eps (0.02) and an effectively-unclamped MaxPhys
    # so it can grow with the collapsing core. Gravity-off runs keep the inert 0.001.
    gas_soft = 0.02 if grav else 0.001
    gas_soft_max = 1e10 if grav else 0.001
    for s in ("Gas", "Halo", "Disk", "Bulge", "Stars", "Bndry"):
        P.append(("Softening" + s, gas_soft if s == "Gas" else 0.001))
    for s in ("Gas", "Halo", "Disk", "Bulge", "Stars", "Bndry"):
        P.append(("Softening" + s + "MaxPhys", gas_soft_max if s == "Gas" else 0.001))

    if mhd:
        # Only ArtificialMagneticDissipationConstant is emitted: it is a NECESSARY
        # parameter (must be present; override always allowed). The divB-cleaning
        # tuning params are left at their compiled defaults on purpose --
        # DivBcleaningParabolicSigma is a DEVELOPER-level parameter (overriding it
        # needs EXPERT_LEVEL=42, the "I know what I'm doing" level), and
        # DivBOverCleanFac's default is already 2; emitting either trips
        # OpenGadget3's "trying to overwrite default value" guard at EXPERT_LEVEL=1.
        P += [
            (None, None),
            ("% --- SPMHD (needs MAGNETIC + ARTIFICIAL_MAGNETIC_DISSIPATION +", None),
            ("%       DIVBCLEANING_TRICCO; see codes/opengadget3/Config_mhd.sh) ---", None),
            ("ArtificialMagneticDissipationConstant", 0.1),
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
        elif v is None:
            out.append(k)
        else:
            vs = "%.12g" % v if isinstance(v, float) else str(v)
            out.append("%-30s %s" % (k, vs))
    return "\n".join(out) + "\n"


def write_run_param(gas_param_path, out_path, init_cond_file, output_dir,
                    boxsize=None, mhd=False, grav=None, extra=None, gamma_warn=True):
    """Write an OpenGadget3 parameterfile at `out_path` from the gasoline `.param`.
    Returns the parsed gasoline dict."""
    gas = read_gasoline_param(gas_param_path)
    if gamma_warn:
        g = _f(gas, "dConstGamma")
        if g is not None and abs(g - OPENGADGET3_COMPILE_GAMMA) > 1e-3:
            print("WARNING: gasoline dConstGamma=%.6g but OpenGadget3 compiled with "
                  "gamma=%.6g (5/3). Rebuild OpenGadget3 to match."
                  % (g, OPENGADGET3_COMPILE_GAMMA))
    P = build_param_values(gas, init_cond_file, output_dir, boxsize=boxsize,
                           mhd=mhd, grav=grav, extra=extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(_format(P))
    return gas
