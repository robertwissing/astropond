"""
FDM linear waves in a periodic slab (fuzzy dark matter, ChaNGa --enable-fdm).

case 0 (free quantum dispersion, gravity off): a standing density wave at rest,
    rho(x, 0) = rho0 (1 + A cos(k x)),  k = 2 pi n / L,  v = 0.
The linearised Madelung equations give delta(t) = A cos(omega t) with
    omega = (hbar / 2m) k^2,
the free Schroedinger dispersion. The run covers two periods. The particle-scale
parameter of the test is k h (h the smoothing length): the SPH estimators are
accurate for k h << 1 and the Nyquist end of the spectrum lies at k h ~ 4.

Geometry as IC_setup_wave: x = [-L/2, L/2] holds n wavelengths, y and z are thin
(24 close-packed rows each, x the thickness factor -d: use 2 for 400 neighbours, whose kernel
is 9 spacings wide at the mean density and wider where the density is low), all periodic. The particles are dark matter: the
pre-IC and glass are relaxed as gas in the wave profile, and IC_createsetup
writes the final IC as dark particles (dark_only). The boson mass is fixed
(fdm_mass_eV); hbar/m enters only as a unit choice here.

Cases 1-3 have self-gravity, and ChaNGa's Ewald sum needs a cubic box: a periodic
cube of side L (nx per side) cut from one shared uniform glass (or lattice). The
mode is imprinted as a Zel'dovich displacement x = x0 - (A/k) sin(k x0), which
gives delta = A cos(k x0) to first order, at rest. Softening 1 x the mean
interparticle spacing (SOFT), dTheta 0.3 (tree errors on a near-uniform particle set).

case 1 (Jeans, static box): linearised, delta'' = -omega^2 delta with
    omega^2 = (hbar/2m)^2 k^4 - 4 pi G rho0
(the Ewald sum drops the mean density: the Jeans swindle). rho0 puts the Jeans
wavenumber at k_J = 2 pi NJ / L, NJ = 1.5 by default (-d): mode n = 1 grows as A cosh(gamma t),
gamma^2 = -omega^2 = 4.06 omega_free^2 (run to gamma t = 2.5); n = 2 oscillates at
omega = 0.827 omega_free, n = 3 at 0.968 omega_free (two periods). A scan over n
measures the fraction f of the quantum pressure the code delivers at each k h
(IC_analysis_fdmwave). The scan and the f fit follow the Jeans instability test
of Ida Martine Brynjulfsen: box 1000 kpc, rho0 1 Msun kpc^-3, rho = 1 + 0.1 cos kx,
n = 1..9. Only k/k_J and k h enter, so her runs are this case with k_J at NJ = 4.408:
    for n in 1 2 3 4 5 6 7 8 9; do ./runtest.sh -a 1 -b $n -c 0.1 -d 4.408 fdmwave 400 32 2 1; done
(n = 1-4 grow, 5-9 oscillate). Her runs all end at t = 2.48 / sqrt(4 pi G rho0); here
each mode runs to gamma t = 2.5 or for two periods.
case 2 (comoving Jeans crossing): EdS (Omega_m 1, H0 = sqrt(8 pi / 3) code,
comoving mean density 1), a = 0.1 -> 1. The comoving Jeans wavenumber grows as
k_J = (16 pi G rho_bar a)^(1/4) / sqrt(hbar/m); the box (L ~ 83 kpc comoving) is
chosen so that mode n = 1 crosses it at a = 0.2: it oscillates first, then
grows; n = 2 oscillates throughout. Reference: the linear ODE
    delta'' + 2 H delta' = (4 pi G rho_bar a^-3 - (hbar/m)^2 k^4 / (4 a^4)) delta
from delta(a_i) = A, delta'(a_i) = 0 (comoving_growth).
case 3 (null test): the uniform glass (no mode, -b/-c ignored), rho0 = RHO0, so
the box (L = 1 kpc) is 4x smaller than the Jeans length: nothing should happen.
Run for NULL_PERIODS periods of the box's n = 1 free mode; the analysis reports
the noise-driven velocities, density noise and energy.
case 7 (comoving growing mode, Schive 2025 review, arXiv:2509.23231, Sec. 4.2.1, Fig. 11): the box and
cosmology of case 2, but the mode starts on the growing solution
    delta(a) = A f(xi),  f = (3 cos xi + 3 xi sin xi - xi^2 cos xi) / xi^2,  xi = (hbar/m) k^2 / (H0 a^1/2)
(xi^2 = 6 at k = k_J; f solves the ODE of case 2): Zel'dovich displacement A f(xi_i) and the velocity of
d delta/dt (case 2 starts at rest, a mix of f and the second solution). a = A7_INIT -> A7_END, so mode n
covers xi from xi_n(A7_INIT) down to xi_n(A7_END), xi_n ~ n^2: n = 1 4.47 -> 0.78, 2 17.9 -> 3.10, 3 40 -> 7.0,
4 72 -> 12.4; the runs of n = 1..3 (4) together trace Fig. 11 (xi = 0.8..50). Default A = 0.01 (runtest.sh):
max delta = A f(0.78) = 0.055.

Cases 4 and 5 are exact non-linear solutions of the free Schroedinger equation (gravity off) in the
slab of case 0, after Hopkins (2019, MNRAS 489, 2367; Secs. 3.5 and 3.4). nu = hbar/m.
case 4 (two interfering plane waves, Psi = a1 exp(i k1 x - i w1 t) + a2 exp(i k2 x - i w2 t), w = nu k^2/2,
k1 = K, k2 = 2K, K = 2 pi n / L):
    rho = rho0 (1 + A cos(theta)),  theta = K x - Omega t,  Omega = (3/2) nu K^2,  A = 2 r / (1 + r^2), r = a2/a1,
    u   = nu K (1 + 2 r^2 + 3 r cos(theta)) / (1 + r^2 + 2 r cos(theta)).
-c is the contrast A (default 1: nodes rho = 0, and u = (3/2) nu K is uniform, so the pattern is a
stationary state drifting with the particles; A < 1: the particles stream through the pattern).
NODE_PERIODS periods 2 pi / Omega.
case 6 (three interfering plane waves, Hopkins 2019 Fig. 2 bottom: a1 = a2 = a3 = 1, k_m = m K, w_m = nu k_m^2/2 = m^2 w1):
    rho = (rho0/3) |sum_m exp(i (m K x - m^2 w1 t))|^2,  at t = 0 rho = (rho0/3) (1 + 2 cos K x)^2 and u = 2 nu K.
The pattern repeats every T1 = 2 pi / w1. At t = 0 and every T1/2 its nodes are 'discontinuous': just before and
after, u diverges there with opposite signs. -c must be 1 (equal amplitudes). Run to t = NW3_TEND / w2 (w2 = 4 w1),
the end time of the paper's figure.
case 5 (two-stream intersection: a converging Gaussian that bounces), tau = sigma0^2 / nu, f = 1 + (t/tau)^2:
    rho = rho_pk exp(-x^2 / (sigma0^2 f)) / sqrt(f),   u = x t / (tau^2 f),
run from t = -tau (converging) through the maximum compression t = 0 to t = +tau, where it must be
the mirror image (diverging). sigma0 = L / TS_BOX; the run's own time is t + tau. -b/-c ignored.
"""
import os
import numpy as np
import IC_distribute as distribute
import IC_fixdens as fixdens
import IC_smoothlength as smth
import readtipsy as tip

L_BOX = 1.0          # kpc, slab length along x
RHO0 = 1.0e7         # Msun kpc^-3 (soliton-core-like; inert while gravity is off)
NPERIODS = 2         # run length in wave periods
OUT_PER_PERIOD = 20  # outputs per period
NJ = 1.5             # case 1: Jeans wavenumber k_J = 2 pi NJ / L (modes n < NJ grow)
GROW = 2.5           # case 1: a growing mode runs to gamma t = GROW (40 outputs)
A_INIT = 0.1         # case 2: start expansion factor
A_CROSS = 0.2        # case 2: expansion factor at which mode n = 1 crosses k_J
A7_INIT = 0.06       # case 7: start expansion factor
A7_END = 2.0         # case 7: end expansion factor
NULL_PERIODS = 4     # case 3: run length in periods of the box's n = 1 free mode
NODE_PERIODS = int(os.environ.get('FDM_NODE_PERIODS', 10))  # case 4: run length in pattern periods 2 pi / Omega
NODE_OUT = 10        # case 4: outputs per period
NW3_TEND = 200.0     # case 6: run length in units of 1 / w2 (= 7.96 T1)
NW3_OUT = 200        # case 6: outputs
TS_BOX = 12.0        # case 5: slab length / sigma0
TS_OUT = 40          # case 5: outputs over t = -tau .. +tau
SOFT = 1.0           # cases 1-3: softening / mean interparticle spacing. Not smaller: sub-kernel particle noise
                     # does not feel the SPH quantum pressure and clumps gravitationally (at 0.2 the case-1 n = 2
                     # mode is destroyed within one period)


def hbar_over_m_code(mass_eV, msol=1.0, kpc=1.0):
    """hbar/m in code units (kpc^2 per code time, G = 1), with ChaNGa's constants
    (physconst.h and the dFDMHbarOverM block of ParallelGravity.cpp)."""
    hbar, erg_per_eV, c = 1.054571817e-27, 1.602176634e-12, 2.99792458e10
    msolg, gcgs, kpccm = 1.99e33, 6.67e-8, 3.085678e21
    hbar_over_m = hbar * c * c / (mass_eV * erg_per_eV)            # cm^2 s^-1
    sec_unit = np.sqrt(1.0 / (msol * msolg / (kpc * kpccm)**3 * gcgs))
    return hbar_over_m * sec_unit / (kpc * kpccm)**2


def omega_free(k, mass_eV):
    """Free quantum dispersion omega = (hbar / 2m) k^2 (code units)."""
    return 0.5 * hbar_over_m_code(mass_eV) * k**2


def node_wave(theta, A, K, nu):
    """Case 4: (rho / rho0, u) of the two-wave solution at phase theta = K x - Omega t."""
    r = (1. - np.sqrt(1. - A * A)) / A
    c = np.cos(theta)
    if A == 1.:             # r = 1: u is uniform (0/0 at the nodes otherwise)
        return 1. + c, np.full(np.shape(c), 1.5 * nu * K)
    return 1. + A * c, nu * K * (1. + 2. * r * r + 3. * r * c) / (1. + r * r + 2. * r * c)


def three_wave_modes(w1t, K, nu):
    """Case 6: the mass-weighted means <exp(-i q K x)> and <v_x exp(-i q K x)> for q = 1, 2 at w1 t (exact)."""
    e3, e5, e8 = np.exp(-3j * w1t), np.exp(-5j * w1t), np.exp(-8j * w1t)
    return (e3 + e5) / 3., e8 / 3., nu * K * (3. * e3 + 5. * e5) / 6., 2. * nu * K * e8 / 3.


def two_stream(mass_eV):
    """Case 5: (sigma0, tau)."""
    sigma0 = L_BOX / TS_BOX
    return sigma0, sigma0**2 / hbar_over_m_code(mass_eV)


def jeans_rho0(mass_eV, nj=NJ):
    """Case 1 mean density: k_J = 2 pi nj / L, rho0 = (hbar/m)^2 k_J^4 / (16 pi G)."""
    return hbar_over_m_code(mass_eV)**2 * (2. * np.pi * nj / L_BOX)**4 / (16. * np.pi)


def omega2_jeans(k, rho0, nu):
    """Case 1 dispersion omega^2 = (nu/2)^2 k^4 - 4 pi G rho0 (< 0: growth rate gamma = sqrt(-omega^2))."""
    return (0.5 * nu)**2 * k**4 - 4. * np.pi * rho0


def comoving_units(mass_eV):
    """Case 2 units (dKpcUnit = comoving box, dMsolUnit from IC_createsetup's calculatecosmoMsol(1, L), so the
    comoving mean density is 1). hbar/m in code units scales as L^-2 and the crossing expansion factor of mode n = 1,
    a_c = (2 pi)^4 nu^2 / (16 pi), as L^-4: L is set from a_c = A_CROSS. Returns (L_kpc, nu)."""
    import contextlib, io
    from IC_writeascii import calculatecosmoMsol

    def nu_of(L):
        with contextlib.redirect_stdout(io.StringIO()):
            msol = calculatecosmoMsol(1.0, L)
        return hbar_over_m_code(mass_eV, msol=msol, kpc=L)
    a_ref = (2. * np.pi)**4 * nu_of(1000.)**2 / (16. * np.pi)
    L = 1000. * (a_ref / A_CROSS)**0.25
    return L, nu_of(L)


def comoving_growth(k, nu, a_out, delta_i, a_i=A_INIT, H0=2.894405):
    """Case 2 reference: delta(a) of the linear EdS FDM equation in cosmic time t,
    delta'' + 4/(3t) delta' = (2/(3t^2) - nu^2 k^4 / (4 a^4)) delta, a = (t/t0)^(2/3),
    from delta(a_i) = delta_i, delta'(a_i) = 0. k comoving (code), nu = hbar/m (code)."""
    from scipy.integrate import solve_ivp
    t0 = 2. / (3. * H0)
    a = lambda t: (t / t0)**(2. / 3.)
    f = lambda t, y: [y[1], -4. / (3. * t) * y[1] + (2. / (3. * t**2) - nu**2 * k**4 / (4. * a(t)**4)) * y[0]]
    ti = t0 * a_i**1.5
    tt = t0 * np.atleast_1d(np.asarray(a_out, dtype=float))**1.5
    s = solve_ivp(f, [ti, max(tt.max(), ti)], [delta_i, 0.], rtol=1e-10, atol=1e-14 * abs(delta_i),
                  dense_output=True)
    return s.sol(tt)[0]


def schive_xi(k, nu, a, H0=2.894405):
    """Case 7: xi = (hbar/m) k^2 / (H0 a^1/2) (Schive 2025 Eq. 76; xi^2 = 6 at k = k_J). k comoving, nu = hbar/m (code)."""
    return nu * k**2 / (H0 * np.sqrt(a))


def schive_f(xi):
    """Case 7: the growing solution of the linear comoving FDM equation (Schive 2025 Eq. 77, Fig. 11)."""
    return (3. * np.cos(xi) + 3. * xi * np.sin(xi) - xi**2 * np.cos(xi)) / xi**2


def schive_df(xi):
    """Case 7: d f / d xi."""
    return -6. * np.cos(xi) / xi**3 - 6. * np.sin(xi) / xi**2 + 3. * np.cos(xi) / xi + np.sin(xi)


class setup_fdmwave(object):
    # FDM test: dark particles in the final IC; IC_createparamfile writes bFDM, dFDMMass_eV.
    dark_only = True
    fdm_mass_eV = 1e-22

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
    ndim = 3
    time = 0
    grav = 0
    cosmo = 0
    H0 = 2.894405        # case 2: EdS in code units (critical density 1)
    omega0 = 1.0
    omegaLambda = 0.0
    molweight = 1.0
    gamma = 5. / 3.
    ns = 64
    periodic = 1
    deltastep = 0.01
    nsteps = NPERIODS * OUT_PER_PERIOD
    freqout = 1
    dmsolunit = 1.0
    dkpcunit = 1.0
    adi = 1
    shape = 2
    rcylmin2 = 0
    rmin2 = 0
    rcylmax2 = 10**22
    rmax2 = 10**22
    glass_density_table = None
    _rho_z_grid = None

    def __init__(self):
        self.mass, self.x, self.y, self.z = [], [], [], []
        self.vx, self.vy, self.vz = [], [], []
        self.rho, self.u, self.h, self.P = [], [], [], []
        self.Bx, self.By, self.Bz = [], [], []
        self.dxbound, self.dybound, self.dzbound, self.totvol = [], [], [], []
        self.rhozero, self.ampl, self.wk = RHO0, 0.0, 0.0
        self.case = 0

    def getrho(self, x):
        if self.case == 5:      # the Gaussian at t = -tau (f = 2); its mean over the slab is rhozero
            s2 = 2. * two_stream(self.fdm_mass_eV)[0]**2
            return self.rhozero * L_BOX / np.sqrt(np.pi * s2) * np.exp(-np.asarray(x)**2 / s2)
        # floor: the lattice stretch map divides by rho, which is 0 at the nodes of case 4 with A = 1 (and case 6)
        if self.case == 6:
            return self.rhozero * np.maximum((1.0 + 2.0 * np.cos(self.wk * np.asarray(x)))**2 / 3.0, 1e-6)
        return self.rhozero * np.maximum(1.0 + self.ampl * np.cos(self.wk * np.asarray(x)), 1e-6)

    def getrhoi(self, i):
        return self.getrho(self.x[i])

    def create(self, nx, distri=0, vm=0, entry='1', case=0, mode=1, ampl=3e-2, thick=None):
        # case: see the module docstring; mode n: wavelengths in the box; ampl: density amplitude A;
        # thick: slab thickness in units of 24 rows (cases 0, 4, 5, 6; default 1); case 1 reads it as NJ.
        case, mode = int(case), int(mode)
        if case in (1, 2, 3, 7):
            nj = NJ if thick is None or case != 1 else float(thick)
            return self.create_cube(nx, distri, vm, entry, case, mode, ampl, nj)
        thick = 1 if thick is None else thick
        if case not in (0, 4, 5, 6):
            raise NotImplementedError(f'fdmwave case {case} is not implemented (0 dispersion, 1 Jeans, '
                                      '2 comoving Jeans crossing, 3 null test, 4 two-wave nodes, 5 two-stream, '
                                      '6 three-wave nodes, 7 comoving growing mode)')
        if mode < 1:
            raise ValueError('fdmwave mode number n must be >= 1')
        self.case = case
        self.ampl = 0.0 if case == 5 else float(ampl)
        if case == 4 and not 0. < self.ampl <= 1.:
            raise ValueError('fdmwave case 4: the contrast -c must be in (0, 1]')
        if case == 6 and self.ampl != 1.:
            raise ValueError('fdmwave case 6: the three amplitudes are equal, -c must be 1')
        dx = 0.5 * L_BOX
        deltax = L_BOX / nx
        fac = 12. * int(thick)
        dy = fac * deltax * np.sqrt(0.75)
        dz = fac * deltax * np.sqrt(6.) / 3.
        distribute.setbound(self, -dx, dx, -dy, dy, -dz, dz)
        self.wk = 2. * np.pi * mode / L_BOX
        omega = omega_free(self.wk, self.fdm_mass_eV)
        if case == 0:
            self.deltastep = 2. * np.pi / omega / OUT_PER_PERIOD
            print(f'FDM wave case {case}: n = {mode}, k = {self.wk:.4g} kpc^-1, A = {self.ampl:g}, '
                  f'omega = (hbar/2m) k^2 = {omega:.4g}, period {2*np.pi/omega:.4g} code units')
        elif case == 4:
            self.deltastep = 2. * np.pi / (3. * omega) / NODE_OUT
            self.nsteps = NODE_PERIODS * NODE_OUT
            print(f'FDM two-wave case 4: K = 2 pi {mode} / L, contrast A = {self.ampl:g}, Omega = (3/2)(hbar/m) K^2 = '
                  f'{3*omega:.4g}, {NODE_PERIODS} periods of {2*np.pi/(3*omega):.4g} code units')
        elif case == 6:
            self.deltastep = NW3_TEND / (4. * omega) / NW3_OUT
            self.nsteps = NW3_OUT
            print(f'FDM three-wave case 6: K = 2 pi {mode} / L, w1 = (hbar/2m) K^2 = {omega:.4g}, pattern period '
                  f'T1 = 2 pi / w1 = {2*np.pi/omega:.4g} code units, run to {NW3_TEND:g} / w2 = '
                  f'{NW3_TEND / (8. * np.pi):.2f} T1')
        else:
            sigma0, tau = two_stream(self.fdm_mass_eV)
            self.deltastep = 2. * tau / TS_OUT
            self.nsteps = TS_OUT
            print(f'FDM two-stream case 5: sigma0 = L / {TS_BOX:g} = {sigma0:.4g} kpc, tau = sigma0^2 / (hbar/m) = '
                  f'{tau:.4g} code units, run t = -tau .. +tau')

        if distri == 0:
            distribute.setcloseddist(self, -dx, dx, -dy, dy, -dz, dz, deltax, periodic=True)
        elif distri == 1:
            distribute.setrandomdist(self, -dx, dx, -dy, dy, -dz, dz, deltax)
        else:
            tgdata, tddata, tsdata, data_header, time = tip.readtipsy(entry)
            self.x = tgdata[:, 1]
            self.y = tgdata[:, 2]
            self.z = tgdata[:, 3]
            self.rho = tgdata[:, 7]
        # As IC_setup_wave: the profile grid is the stretch map's input and the
        # glass relaxer's target; a glass arrives relaxed in it.
        xgrid = np.linspace(-dx, dx, 20001)
        self._rho_z_grid = (xgrid, self.getrho(xgrid))
        self.glass_density_table = (xgrid.copy(), self._rho_z_grid[1].copy(), 'signed_x')
        if distri != 2 and not self.preglass:
            fixdens.set_density_profile(self, -dx, dx)

        self.npart = len(self.x)
        self.ngas = self.npart
        totmass = self.rhozero * self.dxbound * self.dybound * self.dzbound
        if vm != 0:
            raise NotImplementedError('Variable particle masses have not been implemented.')
        self.mass = np.full(self.npart, totmass / self.npart)
        self.h = smth.getsmooth(self, self.ns)
        self.vx = np.zeros(self.npart)
        self.vy = np.zeros(self.npart)
        self.vz = np.zeros(self.npart)
        if not self.preglass:
            xp, nu = np.asarray(self.x, dtype=float), hbar_over_m_code(self.fdm_mass_eV)
            if case == 4:
                self.vx = node_wave(self.wk * xp, self.ampl, self.wk, nu)[1]
            elif case == 6:
                self.vx = np.full(self.npart, 2. * nu * self.wk)
            elif case == 5:
                self.vx = -xp / (2. * two_stream(self.fdm_mass_eV)[1])    # x t / (tau^2 f) at t = -tau
        self.u = np.full(self.npart, 1e8)    # gas only during the glass step
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        self.Bz = np.zeros(self.npart)
        print('npart = ', self.npart, ' particle mass = ', self.mass[0])

    def create_cube(self, nx, distri, vm, entry, case, mode, ampl, nj=NJ):
        """Cases 1-3, 7: periodic cube [-L/2, L/2]^3 from the shared uniform glass, Zel'dovich-displaced mode."""
        if vm != 0:
            raise NotImplementedError('Variable particle masses have not been implemented.')
        if mode < 1:
            raise ValueError('fdmwave mode number n must be >= 1')
        if case == 1 and not (nj > 0 and mode != nj):
            raise ValueError(f'fdmwave case 1: NJ must be > 0 and differ from the mode number n (got NJ {nj})')
        hl = 0.5 * L_BOX
        distribute.setbound(self, -hl, hl, -hl, hl, -hl, hl)
        deltax = L_BOX / nx
        if distri == 0:
            distribute.setcloseddist(self, -hl, hl, -hl, hl, -hl, hl, deltax, periodic=True)
        elif distri == 1:
            distribute.setrandomdist(self, -hl, hl, -hl, hl, -hl, hl, deltax)
        else:
            self.x, self.y, self.z = (tip.readtipsy(entry)[0][:, j].astype(float) for j in (1, 2, 3))
        x0 = np.asarray(self.x, dtype=float)
        n = len(x0)
        self.npart = self.ngas = n
        self.vx, self.vy, self.vz = np.zeros(n), np.zeros(n), np.zeros(n)
        self.u = np.full(n, 1e8)             # gas only during the glass step
        self.Bx, self.By, self.Bz = np.zeros(n), np.zeros(n), np.zeros(n)
        if self.preglass:                    # uniform seed for the glass: unit density
            self.mass = np.full(n, L_BOX**3 / n)
            self.rho = np.ones(n)
            self.h = smth.getsmooth(self, self.ns)
            return
        nu = hbar_over_m_code(self.fdm_mass_eV)
        k = 2. * np.pi * mode / L_BOX
        A = 0.0 if case == 3 else float(ampl)
        d0, ddot = A, 0.0                    # IC density amplitude and its time derivative
        self.grav = 1
        self.dtheta = self.dtheta2 = 0.3
        if case == 1:
            rho0 = jeans_rho0(self.fdm_mass_eV, nj)
            w2 = omega2_jeans(k, rho0, nu)
            if w2 > 0:
                self.deltastep = 2. * np.pi / np.sqrt(w2) / OUT_PER_PERIOD
                self.nsteps = NPERIODS * OUT_PER_PERIOD
                what = f'oscillates, omega {np.sqrt(w2):.4g} = {np.sqrt(w2) / omega_free(k, self.fdm_mass_eV):.4f} omega_free'
            else:
                self.deltastep = GROW / np.sqrt(-w2) / 40
                self.nsteps = 40
                what = f'grows, gamma {np.sqrt(-w2):.4g} = {np.sqrt(-w2) / omega_free(k, self.fdm_mass_eV):.4f} omega_free'
            self.ewald = 1
            print(f'FDM Jeans case 1: rho0 {rho0:.5g}, k_J = 2 pi {nj:g} / L, n = {mode} (k/k_J {mode / nj:.3f}) {what}')
        elif case == 2:
            L_kpc, nu = comoving_units(self.fdm_mass_eV)
            rho0 = 1.0
            self.cosmo, self.a, self.dkpcunit = 1, A_INIT, L_kpc
            from IC_setup_cosmowave import eds_time
            self.nsteps = 100
            self.deltastep = (eds_time(1., self.H0) - eds_time(A_INIT, self.H0)) / self.nsteps
            kJ = lambda a: (16. * np.pi * a)**0.25 / np.sqrt(nu)
            print(f'FDM comoving Jeans case 2: box {L_kpc:.3f} kpc (comoving), hbar/m {nu:.5g} code, '
                  f'n = {mode}: k/k_J {k / kJ(A_INIT):.3f} at a = {A_INIT} -> {k / kJ(1.):.3f} at a = 1')
        elif case == 7:
            L_kpc, nu = comoving_units(self.fdm_mass_eV)
            rho0 = 1.0
            self.cosmo, self.a, self.a_end, self.dkpcunit = 1, A7_INIT, A7_END, L_kpc
            from IC_setup_cosmowave import eds_time
            # Outputs: IC_createsetup's 100 expansion factors log-spaced to a_end (.red), written at big-step ends;
            # big steps no longer than the first (shortest) output interval give each output its own step.
            T = eds_time(A7_END, self.H0) - eds_time(A7_INIT, self.H0)
            dt1 = eds_time(A7_INIT * (A7_END / A7_INIT)**(1. / 99.), self.H0) - eds_time(A7_INIT, self.H0)
            self.nsteps = int(np.ceil(T / dt1))
            self.deltastep = T / self.nsteps
            self.freqout = self.nsteps
            xi = schive_xi(k, nu, A7_INIT, self.H0)
            d0 = A * schive_f(xi)
            # d delta/dt = A f'(xi) dxi/dt, dxi/dt = -xi H / 2, H = H0 a^-3/2
            ddot = -0.5 * A * schive_df(xi) * xi * self.H0 * A7_INIT**-1.5
            print(f'FDM comoving growing mode case 7: box {L_kpc:.3f} kpc (comoving), n = {mode}: xi '
                  f'{xi:.3f} at a = {A7_INIT} -> {schive_xi(k, nu, A7_END, self.H0):.3f} at a = {A7_END} '
                  f'(k = k_J at xi = 2.449), delta_i / A = f(xi_i) = {d0 / max(A, 1e-300):.4f}')
        else:
            rho0 = RHO0
            self.deltastep = 2. * np.pi / omega_free(2. * np.pi / L_BOX, self.fdm_mass_eV) / 10
            self.nsteps = 10 * NULL_PERIODS
            self.ewald = 1
            print(f'FDM null test case 3: uniform rho0 {rho0:g}, box / Jeans length '
                  f'{L_BOX * (16. * np.pi * rho0 / nu**2)**0.25 / (2. * np.pi):.3f}')
        if case not in (2, 7):
            self.cosmo = 0
        self.x = x0 - d0 / k * np.sin(k * x0)
        self.x = (self.x + hl) % L_BOX - hl
        # continuity: d delta/dt = -d xdot/dx (comoving x; ChaNGa reads xdot and scales it to a^2 xdot itself)
        self.vx = -ddot / k * np.sin(k * x0)
        self.mass = np.full(n, rho0 * L_BOX**3 / n)
        self.rho = rho0 * (1. + d0 * np.cos(k * x0))
        self.h = np.full(n, SOFT * deltax * (len(x0) / nx**3)**(-1. / 3.))
        print(f'  n = {mode}, A = {A:g}, {n} particles, eps {self.h[0]:.4g}, {self.nsteps} steps x {self.deltastep:.4g}')
