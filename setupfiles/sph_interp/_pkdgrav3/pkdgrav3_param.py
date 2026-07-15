"""Generate a PKDGRAV3 `.par` parameterfile from a gasoline/tipsy `.param`.

PKDGRAV3 reads tipsy ICs natively (achInFile) and writes tipsy snapshots
(`{name}.{step:05d}`), so this is the ONLY pkdgrav3-specific artefact: the run
parameterfile. The `.par` is Python syntax (key = value), optionally importing
helpers; for the SPH hydro model it needs `from PKDGRAV import HYDRO_MODEL`.

Mapping gasoline `.param` -> pkdgrav3 `.par`:
  (tipsy IC path)         -> achInFile
  <runname>               -> achOutName  (+ achOutPath '<dir>/{name}.{step:05d}{type}')
  dDelta                  -> dDelta
  nSteps                  -> nSteps
  iOutInterval            -> iOutInterval
  nSmooth                 -> nSmooth (SPH target neighbour number)
  dxPeriod                -> dPeriod (periodic box side)

UNIT CAVEAT: the suite runs in code units (G implicit, all unit scales = 1).
pkdgrav3's SPH EOS is driven by dKpcUnit/dMsolUnit (physical units); a faithful
code-unit run may need those reconciled or the builtin-ideal-gas EOS. The
generated .par sets the builtin ideal-gas SPH path and leaves the unit scales at
1 -- VALIDATE against a built pkdgrav3 before trusting hydro results. Gravity-only
/ cosmology tests do not hit this caveat.
"""
import os

from sph_interp._arepo.arepo_param import read_gasoline_param, _f, _i


def build_par_lines(gas, infile, outname, outpath, boxsize=None,
                    hydro=True, extra=None):
    """Ordered list of '.par' lines (str) from a parsed gasoline dict.

    infile   : tipsy IC path (read natively; no conversion).
    outname  : achOutName (base name for snapshots/log).
    outpath  : achOutPath template, e.g. '<rundir>/{name}.{step:05d}{type}'.
    boxsize  : periodic box side (else dxPeriod).
    hydro    : True -> SPH hydro model (HYDRO_MODEL.SPH) + builtin ideal gas.
    """
    dDelta = _f(gas, "dDelta")
    if dDelta is None or dDelta <= 0:
        raise ValueError("gasoline .param has no usable dDelta (base timestep)")
    nSteps = _i(gas, "nSteps", 0)
    iOut = _i(gas, "iOutInterval", nSteps or 1)
    nSmooth = _i(gas, "nSmooth", 64)
    if boxsize is None:
        boxsize = _f(gas, "dxPeriod", 1.0)

    L = []
    L.append("# PKDGRAV3 parameter file generated from a gasoline .param by")
    L.append("#   sph_interp/_pkdgrav3/pkdgrav3_param.py  (do not hand-edit)")
    if hydro:
        L.append("from PKDGRAV import HYDRO_MODEL")
    L.append("")
    L.append('achInFile       = "%s"' % infile)
    L.append('achOutName      = "%s"' % outname)
    L.append('achOutPath      = "%s"' % outpath)
    L.append("")
    L.append("# Non-cosmological periodic box, code units.")
    L.append("bComove         = False")
    L.append("bPeriodic       = True")
    L.append("bEwald          = False")
    L.append("dPeriod         = %.12g" % boxsize)
    L.append("")
    L.append("iStartStep      = 0")
    L.append("nSteps          = %d" % nSteps)
    L.append("dDelta          = %.12g" % dDelta)
    L.append("iOutInterval    = %d" % iOut)
    L.append("")
    if hydro:
        L.append("# SPH hydro (builtin ideal-gas EOS; see UNIT CAVEAT in module).")
        L.append("hydro_model       = HYDRO_MODEL.SPH")
        L.append("nSmooth           = %d" % nSmooth)
        L.append("bGasBuiltinIdeal  = True")
        L.append("dConstGamma       = %.12g" % (_f(gas, "dConstGamma", 5.0 / 3.0)))
        L.append("dMsolUnit         = 1.0")
        L.append("dKpcUnit          = 1.0")
        L.append("")
    L.append("# I/O / performance")
    L.append("bOverwrite      = True")
    L.append("bMemUnordered   = True")
    L.append("bDoDensity      = False")
    L.append("bVDetails       = True")

    if extra:
        for k, v in extra.items():
            L.append("%-15s = %s" % (k, v))
    return L


def write_run_param(gas_param_path, out_path, infile, outname, outpath,
                    boxsize=None, hydro=True, extra=None):
    """Write a pkdgrav3 `.par` at `out_path` from the gasoline `.param`. Returns
    the parsed gasoline dict."""
    gas = read_gasoline_param(gas_param_path)
    lines = build_par_lines(gas, infile, outname, outpath, boxsize=boxsize,
                            hydro=hydro, extra=extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return gas
