#!/usr/bin/env python3
"""Periodic isothermal turbulence box for the Gasoline OU driver.

The reference normalization is Tricco, Price & Federrath (2016): a unit
periodic cube with rho=1, c_s=1, initially zero velocity, forcing in
1 <= kL/(2*pi) <= 3 with a parabolic envelope, and (for MHD) a seed field
of rms strength B0=sqrt(2)*1e-5 (plasma beta 1e10 for mu0=1).

The seed AMPLITUDE is fixed at that published value; only its STRUCTURE is a
test parameter.  'uniform' is Tricco's net-flux Bz=B0.  'sin' is the
zero-net-flux Bz=sqrt(2)*B0*sin(2*pi*x/L), whose amplitude is chosen so both
structures carry the same volume-averaged B^2: identical initial magnetic
energy, plasma beta and amplification baseline, so the structure is the only
difference.  A periodic box conserves <B>, so uniform keeps an undecayable
mean field while sin stays zero-mean and can decay -- the sharper dynamo
test.
Their Mach-10 calculation used st_decay=t_cross=0.05 and st_energy=8.

For another requested Mach number M, the setup keeps the dimensionless
problem similar by taking t_corr=L/(2 M c_s) and

    st_energy = 8 (M/10)^3 (c_s/1)^3 (1/L).

This makes the OU acceleration scale as M^2 and the velocity impulse over one
correlation time scale as M.  It is a reference-calibrated *nominal* Mach
input, not a feedback controller; IC_analysis_turbbox reports the achieved
Mach number from the snapshots/log.
"""

import math

import numpy as np

import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip


# Published Tricco, Price & Federrath (2016) seed strength, as an rms |B|.
B0_RMS = math.sqrt(2.0) * 1.0e-5


def driving_source(value, mach, forcing, seed, crossings):
    """Select the published pattern for the standard Mach-10, seed-1 case."""
    key = str(value).strip().lower()
    if key not in ("auto", "paper", "generated"):
        raise ValueError("driving must be auto, paper, or generated")
    compatible = (float(mach) == 10.0
                  and forcing_weight(forcing)[1] == 1.0 and float(seed) == 1.)
    if key == "auto":
        key = "paper" if compatible else "generated"
    if key == "paper":
        if not compatible:
            raise ValueError(
                "the published forcing file fixes its own physics in its "
                "header (energy=8 i.e. Mach 10, weight=1 i.e. solenoidal), so "
                "-e 0 requires -a 10 and -b 1; use -e N>=1 to generate another "
                "OU realisation at your own Mach and projection weight")
        if float(crossings) > 100.0:
            raise ValueError("the published forcing file ends at 100 crossings "
                             "(header endTime=5.0, decay=0.05): use -d <= 100 "
                             "or -e N>=1 for a generated sequence")
    return key


def forcing_weight(value):
    """Return (canonical_name, zeta) for a forcing selector.

    zeta is StSolWeight, the driver's projection weight: 1 purely solenoidal,
    0 purely compressive, anything between a mixture.  The INJECTED
    compressive power fraction is (1-zeta)^2/((1-zeta)^2 + 2 zeta^2) -- the
    same denominator the driver normalises with -- so zeta=0.5 gives the
    natural 1:2 compressive:solenoidal mixture, not a half-and-half one.
    """
    key = str(value).strip().lower()
    words = {"solenoidal": 1.0, "sol": 1.0, "s": 1.0,
             "compressive": 0.0, "comp": 0.0, "c": 0.0}
    if key in words:
        zeta = words[key]
    else:
        try:
            zeta = float(key)
        except ValueError:
            raise ValueError(
                "forcing must be a solenoidal weight in [0,1] (1=solenoidal, "
                "0=compressive, e.g. 0.5 for the natural mixture), or the "
                "words 'solenoidal'/'sol' or 'compressive'/'comp'") from None
    if not (math.isfinite(zeta) and 0.0 <= zeta <= 1.0):
        raise ValueError("forcing (solenoidal weight) must lie in [0,1]")
    if zeta == 1.0:
        return ("solenoidal", 1.0)
    if zeta == 0.0:
        return ("compressive", 0.0)
    return ("mixed%g" % zeta, zeta)


def forcing_tag(value):
    """Short run-name token. Endpoints keep sol/comp so existing run names,
    baselines and run-name parsers are unaffected; mixtures get z<zeta>."""
    zeta = forcing_weight(value)[1]
    if zeta == 1.0:
        return "sol"
    if zeta == 0.0:
        return "comp"
    return "z%g" % zeta


def compressive_fraction(zeta):
    """Injected compressive power fraction for a projection weight zeta."""
    zeta = float(zeta)
    return (1.0 - zeta) ** 2 / ((1.0 - zeta) ** 2 + 2.0 * zeta ** 2)


def case_physics(value):
    """Map a case number to (mhd, seed-field structure).

    0 = hydrodynamics, no field
    1 = MHD, uniform net-flux Bz = B0            (Tricco et al. 2016)
    2 = MHD, zero-net-flux Bz = sqrt(2) B0 sin(2 pi x/L), same <B^2>
    """
    key = str(value).strip().lower()
    cases = {"0": (0, "uniform"), "1": (1, "uniform"), "2": (1, "sin")}
    try:
        return cases[key]
    except KeyError:
        raise ValueError(
            "case (-c) must be 0 (hydro), 1 (MHD, uniform net-flux Bz) or "
            "2 (MHD, zero-net-flux sin Bz)") from None


def case_tag(value):
    """Run-name tokens (physics, seed) for a case number."""
    mhd, field = case_physics(value)
    return ("mhd" if mhd else "hydro"), field


def field_structure(value):
    """Canonical seed-field structure. The amplitude is not selectable."""
    key = str(value).strip().lower()
    aliases = {
        "uniform": "uniform", "u": "uniform", "net": "uniform",
        "netflux": "uniform", "0": "uniform", "0.0": "uniform",
        "sin": "sin", "sine": "sin", "nonetflux": "sin", "nnf": "sin",
        "1": "sin", "1.0": "sin",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "seed field must be 'uniform'/'net'/0 (net-flux Bz=B0) or "
            "'sin'/'nonetflux'/1 (zero-net-flux Bz ~ sin(2 pi x/L))") from exc


class setup_turbbox(object):
    # Uniform glass target metadata used by IC_createsetup/glassgen.
    dICdensRsmooth = 0.1
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0
    dICdensinner = 1.0
    dICdensouter = 1.0

    inflow = 0
    rhoit = 0
    npart = 0
    ngas = 0
    ndark = 0
    nstar = 0
    shape = 0
    ndim = 3
    time = 0
    cosmo = 0
    ns = 64
    periodic = 1
    grav = 0
    dmsolunit = 1.0
    dkpcunit = 1.0
    molweight = 1.0
    adi = 0
    gamma = 5.0 / 3.0
    # Gasoline's runtime-isothermal path freezes u but evaluates the generic
    # ideal-gas sound-speed expression.  gamma -> 1 gives P=rho*c_s^2 and the
    # correct isothermal characteristic speed without the specialised,
    # dimensional BAROTROPIC compile-time EOS in this fork.
    isothermal_gamma = 1.0000000001
    gamma_eos = isothermal_gamma

    rcylmin2 = 0
    rmin2 = 0
    rcylmax2 = 10**22
    rmax2 = 10**22
    rhozero = 1.0

    def __init__(self):
        # Instance-owned arrays: setup objects can safely be constructed more
        # than once in an interactive/test process.
        for name in ("mass", "x", "y", "z", "vx", "vy", "vz", "rho",
                     "u", "h", "P", "Bx", "By", "Bz", "Bpsi", "soft",
                     "tform", "metals", "pot", "totvol"):
            setattr(self, name, [])
        self.dxbound = self.dybound = self.dzbound = []

    def create(self, nx, distri=0, vm=0,
               entry="datafiles/turbbox64_preglass.00000",
               mach=10.0, forcing="solenoidal", mhd=1,
               field="uniform", crossings=100.0, seed=1,
               driving="auto"):
        nx = int(nx)
        mach = float(mach)
        mhd = int(float(mhd))
        field = field_structure(field)
        crossings = float(crossings)
        seed = int(float(seed))
        if nx < 2:
            raise ValueError("nx must be at least 2")
        if not math.isfinite(mach) or mach <= 0.0:
            raise ValueError("mach must be finite and positive")
        if mhd not in (0, 1):
            raise ValueError("mhd must be 0 (hydro) or 1 (MHD)")
        if not math.isfinite(crossings) or crossings <= 0.0:
            raise ValueError("crossings must be finite and positive")
        forcing_name, zeta = forcing_weight(forcing)
        driving = driving_source(driving, mach, forcing, seed, crossings)

        boxlen = 1.0
        cs = 1.0
        half = 0.5 * boxlen
        deltax = boxlen / nx
        distribute.setbound(self, -half, half, -half, half, -half, half)

        if distri == 0:
            distribute.setcloseddist(
                self, -half, half, -half, half, -half, half, deltax)
        elif distri == 1:
            distribute.setrandomdist(
                self, -half, half, -half, half, -half, half, deltax)
        else:
            tgdata, _td, _ts, _header, _time = tip.readtipsy(entry)
            self.x = tgdata[:, 1]
            self.y = tgdata[:, 2]
            self.z = tgdata[:, 3]
            self.rho = tgdata[:, 7]

        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.npart = len(self.x)
        self.ngas = self.npart
        if self.npart == 0:
            raise ValueError("particle distribution is empty")

        total_mass = self.rhozero * boxlen**3
        self.mass = np.full(self.npart, total_mass / self.npart)
        if float(vm) != 0.0:
            if self.npart % 2:
                raise ValueError("variable-mass pairing needs an even particle count")
            # Retain the suite's legacy paired perturbation while conserving
            # total mass exactly. Production paper comparisons use vm=0.
            rng = np.random.default_rng(seed)
            for i in range(0, self.npart, 2):
                dm = 0.25 * self.mass[i] * rng.random()
                self.mass[i] += dm
                self.mass[i + 1] -= dm

        self.h = smth.getsmooth(self, 200)
        self.rho = np.full(self.npart, self.rhozero)
        self.vx = np.zeros(self.npart)
        self.vy = np.zeros(self.npart)
        self.vz = np.zeros(self.npart)

        # convertfromcodeunits uses gamma_eos to turn this code-unit specific
        # energy into the Tipsy temperature field. The runtime reconstruction
        # recovers P/rho=(gamma_iso-1)u=c_s^2.
        self.u = np.full(
            self.npart, cs**2 / (self.isothermal_gamma - 1.0))
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        # Bz=f(x) has dBz/dz=0, so either seed is exactly divergence free
        # without a vector potential. The sine amplitude sqrt(2)*B0_RMS makes
        # <B^2>=B0_RMS^2 for both, matching uniform's initial energy.
        if not mhd:
            self.Bz = np.zeros(self.npart)
        elif field == "uniform":
            self.Bz = np.full(self.npart, B0_RMS)
        else:
            self.Bz = (math.sqrt(2.0) * B0_RMS
                       * np.sin(2.0 * math.pi * self.x / boxlen))

        # Published large-scale forcing and Mach-10 calibration.
        self.turbbox_mach = mach
        self.turbbox_forcing = forcing_name
        self.turbbox_mhd = mhd
        self.turbbox_field = field if mhd else "none"
        self.turbbox_B0 = B0_RMS if mhd else 0.0
        self.turbbox_crossings = crossings
        self.turbbox_cs = cs
        self.turbbox_tcross = boxlen / (2.0 * mach * cs)
        self.turbbox_seed = seed
        self.StDecay = self.turbbox_tcross
        self.StEnergy = 8.0 * (mach / 10.0)**3 * cs**3 / boxlen
        self.StDtFreq = 0.1 * self.StDecay
        self.StKmin = 2.0 * math.pi / boxlen
        self.StKmax = 6.0 * math.pi / boxlen
        self.StSolWeight = zeta
        self.StAmplFac = 1.0
        self.StScaleHeight = 1.0e30
        self.StStartTime = 0.0
        self.StSpectForm = 1
        self.StSeed = seed
        self.StCorrectBulkMotion = 1
        self.turbbox_driving = driving
        self.StForcingFile = "forcing/forcing.dat" if driving == "paper" else ""
        if driving == "paper":
            # Exact values in the published file header, DOI 13162225.
            self.StKmin, self.StKmax = 6.28, 18.90
            self.StDtFreq = 0.005

        # One hundred top-level steps per crossing time; the Courant condition
        # may select smaller particle steps. Four snapshots per crossing give
        # useful histories without the 1000-output burden of the paper runs.
        self.deltastep = self.turbbox_tcross / 100.0
        self.nsteps = int(math.ceil(crossings * 100.0))
        self.freqout = 25

        print("Turbulence box (Price & Federrath 2010; Tricco et al. 2016)")
        print(" nx=%d N=%d M_target=%g forcing=%s mhd=%d seed=%s B0_rms=%g" %
              (nx, self.npart, mach, forcing_name, mhd, self.turbbox_field,
               self.turbbox_B0))
        print(" t_cross=t_corr=%g StEnergy=%g StDtFreq=%g crossings=%g" %
              (self.StDecay, self.StEnergy, self.StDtFreq, crossings))

    def getrho(self, _x):
        return self.rhozero

    def getrhoi(self, _i):
        return self.rhozero
