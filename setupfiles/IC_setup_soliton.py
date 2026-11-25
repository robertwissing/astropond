"""
Setup for the solitonic core test from Nori & Baldi (2018) with a small modification to work numerically
with the glass relaxation method. A small (1e-3) constant was added to the density function so that the background
density is not exactly zero.

If choosing random pre-ICs, glass relaxation is done to fit these to the desired density profile (recommended approach).
Lattice pre-ICs are not supported (no spherical lattice stretch map): use random (1) or glass (2).

Cases (-a):
  0  the static Gaussian core above (default). Original setup: Ida Martine Brynjulfsen.
  1  free wave packet: the case-0 Gaussian (same glass and IC) with gravity off in a periodic box, run to
     t = tau = 2 sigma^2 / (hbar/m) in 20 steps. A free Gaussian packet spreads homologously,
     sigma(t) = sigma s(t), s = sqrt(1 + (t/tau)^2), its central density drops as s^-3 and its quantum energy
     turns into kinetic energy, E_kin = E_Q,0 (t/tau)^2 / (1 + (t/tau)^2) (half of it at t = tau).
  2  self-gravitating Schroedinger-Poisson ground state (soliton), r_c = 1 kpc, truncated at 4 r_c, at rest in an
     open box; it must stay in equilibrium (density oscillation period ~660 Myr). A uniform periodic glass cube is
     relaxed, a sphere cut from it and mapped radially onto the soliton's M(r).
  3  head-on collision of two case-2 solitons at x = -+D/2 r_c moving at +-v/2 v_c(r_c) (-b D, -c v). Presets:
     fast -b 10 -c 6 (passes through), slow -b 8 -c 1 (bound). Reference: the Schroedinger-Poisson solution of
     soliton_reference.py in reference_solution_highres/soliton/.
  4  multi-soliton halo: N case-2 solitons (-b N, default 4) at rest, centres drawn (seeded, halo_positions) in a
     sphere of radius R r_c (-c R, default 6) at least HALO_DMIN r_c apart; they collapse (free-fall ~140 Myr for the
     default) and merge into one halo with a soliton core, run for 1573 Myr. Reference: soliton_reference.py --halo
     (Schroedinger-Poisson, three phase realisations: all phases 0 and two seeded random sets).
  5  the case-2 soliton with baryons, each of mass -b (default 1e-3) x the soliton's: NSTAR_SHELL tracer stars per
     radius of STAR_R on circular orbits (tracer_stars), and gas from the inner GAS_NFRAC of the same glass sphere
     mapped radially onto the hydrostatic sphere of gas_profile, c_s = -c (default 0.5) x v_c(r_c); -d gas EOS
     (0 isothermal, P = rho c_s^2; 1 adiabatic, gamma 5/3, same uniform-T start, for an energy budget). The baryon
     mass is left out of the star orbits and the gas profile, so only a small -b is an equilibrium. The gas uses the
     .param nSmooth (64), the FDM particles the run's nSmoothFDM.
"""
import functools
import numpy as np
import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip
import IC_denstable as denstable
from IC_setup_fdmwave import hbar_over_m_code

RC = 1.0              # kpc, core radius of the case 2/3 soliton (rho(rc) = rho_c / 2)
RT = 4.0              # truncation radius of the particle soliton / rc
DDELTA = 1.6686e-5    # code time (7.9 Myr): base step of cases 2 and 3
HALO_SEED = 4         # case 4: RNG seed of the soliton centres
HALO_DMIN = 2.5       # case 4: minimum centre separation / rc
STAR_R = (0.5, 1.0, 2.0, 3.0)  # case 5: radii of the tracer-star orbits / rc
NSTAR_SHELL = 250     # case 5: stars per radius
STAR_SEED = 5         # case 5: RNG seed of the orbit planes and phases
GAS_NFRAC = 0.3       # case 5: gas particles per dark particle (5150 at 32^3)


def myr_per_code(msol=1.0, kpc=1.0):
    """Code time unit in Myr (G = 1), with the constants of hbar_over_m_code."""
    msolg, gcgs, kpccm = 1.99e33, 6.67e-8, 3.085678e21
    return np.sqrt(1.0 / (msol * msolg / (kpc * kpccm)**3 * gcgs)) / 3.15576e13


def _shoot(E, xmax=12.0):
    from scipy.integrate import solve_ivp

    def f(x, y):
        phi, dphi, Phi, dPhi = y
        if x < 1e-8:
            return [dphi, (2 * (Phi - E) * phi) / 3, dPhi, 4 * np.pi * phi**2 / 3]
        return [dphi, 2 * (Phi - E) * phi - 2 * dphi / x, dPhi, 4 * np.pi * phi**2 - 2 * dPhi / x]
    node = lambda x, y: y[0]
    node.terminal = True                                  # phi crosses zero: E too high
    turn = lambda x, y: y[1]
    turn.terminal = True
    turn.direction = 1                                    # phi turns up: E too low
    return solve_ivp(f, [0, xmax], [1.0, 0, 0, 0], events=[node, turn], rtol=1e-11, atol=1e-13,
                     dense_output=True)


@functools.lru_cache(maxsize=None)
def sp_groundstate(rc=RC, mass_eV=1e-22):
    """Schroedinger-Poisson ground state in code units (kpc, Msun, G = 1) with core radius rc (rho(rc) = rho_c/2).
    Dimensionless problem (hbar/m = G = 1): -1/2 lap phi + Phi phi = E phi, lap Phi = 4 pi phi^2, phi(0) = 1, shot on
    E; scaling r = a x, rho = nu^2/a^4 rho~ (nu = hbar/m). Returns dict(r, rho, M, rhoc, nu, vc = sqrt(M(<rc)/rc))."""
    from scipy.integrate import cumulative_trapezoid
    lo, hi = 0.0, 5.0
    while hi - lo > 1e-14:
        E = 0.5 * (lo + hi)
        if len(_shoot(E).t_events[0]):
            hi = E
        else:
            lo = E
    s = _shoot(lo)
    x = np.linspace(0, 0.98 * s.t[-1], 20001)             # up to the divergence of the shot
    rho = s.sol(x)[0]**2
    M = cumulative_trapezoid(4 * np.pi * x**2 * rho, x, initial=0)
    a = rc / np.interp(0.5, rho[::-1], x[::-1])
    nu = hbar_over_m_code(mass_eV)
    S = dict(r=a * x, rho=nu**2 / a**4 * rho, M=nu**2 / a * M, rhoc=nu**2 / a**4 * rho[0], nu=nu)
    S['vc'] = np.sqrt(np.interp(rc, S['r'], S['M']) / rc)
    return S


def collision_energy(D, v, S):
    """Total energy of two untruncated solitons at separation D rc and relative speed v v_c (point-mass
    interaction): 2 W/2 - M^2/D + M (v v_c / 2)^2, W = -int M(r) rho 4 pi r dr (virial: E_sol = W/2)."""
    M = S['M'][-1]
    W = -np.trapz(S['M'] * S['rho'] * 4 * np.pi * S['r'], S['r'])
    return W - M**2 / (D * RC) + M * (0.5 * v * S['vc'])**2


def halo_positions(N, R):
    """Case 4 soliton centres (N x 3, kpc): uniform in a sphere of radius R rc, at least HALO_DMIN rc apart (rejection,
    seeded with HALO_SEED, so the particle IC and the reference agree), centre of mass at the origin."""
    N = int(round(N))
    rng = np.random.default_rng(HALO_SEED)
    pos = []
    for _ in range(100000):
        p = rng.uniform(-R, R, 3) * RC
        if np.linalg.norm(p) <= R * RC and all(np.linalg.norm(p - q) >= HALO_DMIN * RC for q in pos):
            pos.append(p)
            if len(pos) == N:
                break
    if len(pos) < N:
        raise ValueError(f'cannot place {N} solitons {HALO_DMIN} rc apart in a sphere of {R} rc')
    pos = np.array(pos)
    return pos - pos.mean(0)


def gas_profile(S, cs, Mgas):
    """Case 5 gas: isothermal (P = rho cs^2) hydrostatic sphere in the soliton's potential (its own gravity left out),
    rho = rho_0 exp(-(Phi - Phi(0)) / cs^2), Phi - Phi(0) = int_0^r M / r'^2 dr', truncated at RT rc and normalised to
    the mass Mgas. Returns (r, rho, M(<r))."""
    from scipy.integrate import cumulative_trapezoid
    w = S['r'] <= RT * RC
    r, M = S['r'][w], S['M'][w]
    rho = np.exp(-cumulative_trapezoid(M / np.maximum(r, 1e-30)**2, r, initial=0) / cs**2)
    Mr = cumulative_trapezoid(4 * np.pi * r**2 * rho, r, initial=0)
    return r, rho * Mgas / Mr[-1], Mr * Mgas / Mr[-1]


def tracer_stars(S):
    """Case 5 stars: NSTAR_SHELL at each radius of STAR_R (rc), shell after shell, on circular orbits in the soliton's
    potential, v = sqrt(M(<r) / r) (baryon mass left out), isotropic random orbit planes and phases (seed STAR_SEED).
    Returns (positions, velocities)."""
    rng = np.random.default_rng(STAR_SEED)
    rad = np.repeat(np.asarray(STAR_R) * RC, NSTAR_SHELL)
    p = rng.normal(size=(len(rad), 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    t = rng.normal(size=(len(rad), 3))
    t -= (t * p).sum(1)[:, None] * p                     # random direction perpendicular to p
    t /= np.linalg.norm(t, axis=1)[:, None]
    return p * rad[:, None], t * np.sqrt(np.interp(rad, S['r'], S['M']) / rad)[:, None]


def packet_tau(sigma, mass_eV=1e-22):
    """Case 1 spreading time tau = 2 sigma^2 / (hbar/m) (code units)."""
    return 2.0 * sigma**2 / hbar_over_m_code(mass_eV)


def run_length(case, D=0.0, v=0.0):
    """(nSteps, iOutInterval) at dDelta = DDELTA: 629 Myr for an unbound collision (E > 0), else 1573 Myr."""
    if case == 3 and collision_energy(D, v, sp_groundstate()) > 0:
        return 80, 2
    return 200, 4

class setup_soliton(object):
    # FDM test: the particles are dark matter. IC_createsetup writes the final IC
    # as dark particles, and IC_createparamfile switches on the quantum pressure.
    dark_only = True
    fdm_mass_eV = 1e-22

    # Glass relaxation parameters
    dICdensprofile = 8   # Density-profile model. 8 = tabulated density.
    dICdensdir     = 4   # Coord. mapping for density profile. 1,2,3=abs(x,y,z), 4=spherical r, 5=cylindrical r, 6=x 7=y 8=z
    # as far i understand, none of the below are used when using a density table:
    dICdensRsmooth = 0.1 
    dICdensR       = 1.0 
    dICdensinner   = 1.0 # calculated depending on mass.
    dICdensouter   = 1.0

    # Geometry limits
    shape = 2 # shape=1 -> cube/box-shells, shape=2 -> sphere/cylinder
    rmin2 = 0
    rmax2 = 1e22
    rcylmin2 = 0 
    rcylmax2 = 1e22

    # Gas parameters
    adi = 1 # 0=isothermal gas, 1=adiabatic, 2=adiabatic
    gamma = 5.0 / 3.0
    molweight = 1.0

    # Code parameters
    inflow = 0 
    rhoit = 0
    npart = 0 
    ngas = 0
    ndark = 0 
    nstar = 0
    ndim = 3     
    ns = 64     
    grav = 1
    cosmo = 0 
    periodic = 0 
    time = 0
    deltastep = 1e-22
    freqout = 1
    nsteps = 50     
    dmsolunit = 1.0 
    dkpcunit = 1.0 

    def __init__(self):
        # Density profile parameters
        self.rho0 = 160  # Msun kpc^-3, background density
        self.sigma = 500 # kpc
        self.c = 1e-3    # bakcground parameter

        # Gas-particle properties arrays
        self.mass    = []
        self.x       = []
        self.y       = []
        self.z       = []
        self.vx      = []
        self.vy      = []
        self.vz      = []
        self.rho     = []
        self.h       = [] 
        self.u       = []
        self.P       = []
        self.Bx      = []
        self.By      = []
        self.Bz      = []
        self.dxbound = []
        self.dybound = []
        self.dzbound = []
        self.totvol  = []

    def getrho(self, r):
        """
        Density at some radius r.
        """
        return self.rho0 * (self.c + np.exp(-r**2 / (2 * self.sigma**2)))

    def getrhoi(self, i):
        """
        Density for some particle i at radius r from the center.
        """
        r = np.sqrt(self.x[i]**2 + self.y[i]**2 + self.z[i]**2)
        return self.rho0 * (self.c + np.exp(-r**2 / (2 * self.sigma**2)))
    
    def getrho_vec(self, r_vals):
        """
        Vectorized density calculation, computes density for a
        vector of radii r_vals.
        """
        return self.rho0 * (self.c + np.exp(-r_vals**2/(2*self.sigma**2)))


    def create(self, nx, distri=0, vm=0, entry='1', case=0, D=10.0, v=6.0, eos=0):
        """
        nx: resolution (number of elements along x)
        vm: variable SPH particle masses (0=equal-mass particles)
        distri: 0=lattice (FCC or cubic, chosen by IC_createsetup), 1=random (also the
                pre-glass IC), 2=rebuild the final IC from the relaxed glass IC in entry
        case: see the module docstring; D [rc] and v [v_c(rc)] are used by case 3; case 4 reads D as the number of
              solitons and v as the radius [rc] of the sphere holding their centres; case 5 reads D as the baryon mass
              fraction b, v as c_s / v_c(rc) and eos as the gas EOS (0 isothermal, 1 adiabatic)
        """
        case = int(case)
        if case in (2, 3, 4, 5):
            return self.create_groundstate(nx, distri, vm, entry, case, D, v, int(eos))
        if case not in (0, 1):
            raise NotImplementedError(f'soliton case {case} is not implemented (0 static core, 1 free packet, '
                                      '2 ground state, 3 collision, 4 multi-soliton halo, 5 soliton with baryons)')
        if case == 1:
            tau = packet_tau(self.sigma, self.fdm_mass_eV)
            self.grav, self.periodic = 0, 1
            self.deltastep, self.nsteps, self.freqout = tau / 20, 20, 1
            print(f'soliton case 1 (free packet): gravity off, periodic, tau = 2 sigma^2 / (hbar/m) = {tau:.5g}, '
                  f'20 steps to t = tau')
        # Set 10 000 kpc (10 Mpc) box domain boundaries
        self.xleft  = -5000
        self.xright = 5000
        ymin        = -5000
        ymax        = 5000
        zmin        = -5000
        zmax        = 5000
        distribute.setbound(self, self.xleft, self.xright, ymin, ymax, zmin, zmax)
        deltax = self.dxbound / nx 

        # Generate particle IC distribution
        if distri == 0:
            # IC_fixdens.coord_transform has no spherical mapping (igeom=3), so a
            # lattice cannot be stretched to this radial profile: relax a glass.
            raise NotImplementedError('spherical lattice stretching is not supported; '
                                      'use distrib 1 (random) or 2 (glass)')
        elif distri == 1:
            distribute.setrandomdist(self, self.xleft, self.xright, ymin, ymax, zmin, zmax, deltax)          
        else:
            print('You chose to rebuild the final IC from a glass relaxed IC file.')
            tgdatain, tddata, tsdata, data_header, time = tip.readtipsy(entry)
            self.mass = tgdatain[:,0]
            self.x = tgdatain[:,1]
            self.y = tgdatain[:,2]
            self.z = tgdatain[:,3]
            self.rho = tgdatain[:,7]
        print('The distribution was successful.')

        # set other particle quantites
        self.npart = len(self.x)
        self.ngas = self.npart

        # Radial density table: the glass relaxation's target, written with the pre-glass IC
        if self.preglass:
            ntab = 10000
            dimtab = 1
            self.Rtab = np.sqrt((self.dxbound/2)**2 + (self.dybound/2)**2 + (self.dzbound/2)**2)
            gridtab, rhotab = denstable.generate_density_table(self, n=ntab, dim=dimtab)
        
        # runtest.sh relaxes against it with --table densitytable.npz
        if self.preglass:
            out = denstable.write_density_table_npz("densitytable.npz", gridtab, rhotab, mapping='r_sph')
            print("Wrote density table to:", out)
        

        # Numerically integrated total mass in box
        total_mass = denstable.compute_total_mass(self, int(nx))

        # Assign equal particle masses
        if vm == 0: 
            self.mass = np.full(self.npart, total_mass/self.npart) 
            print('npart = ', self.npart, ' particle mass = ', self.mass[1])
        else: 
            raise NotImplementedError("Variable particle masses have not been implemented.")
        
        # Recompute smoothing lengths
        self.h = smth.getsmooth(self, self.ns)

        # set particle quantities     
        self.vx = np.zeros(self.npart)
        self.vy = np.zeros(self.npart)
        self.vz = np.zeros(self.npart)
        self.u = np.full(self.npart, 1e8) # setting initial internal energy u0 = 1e8
        # No magnetic field; the driver writes B for the gas (pre-glass) IC.
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        self.Bz = np.zeros(self.npart)

    def create_groundstate(self, nx, distri, vm, entry, case, D, v, eos=0):
        """Cases 2-5, with the settings of the runs checked against the Schroedinger-Poisson reference: a sphere
        cut from a uniform glass cube [-1/2, 1/2]^3 (nx per side), mapped radially so that M(<r')/M(<RT rc) = (r/R0)^3;
        constant softening, half the core interparticle spacing; dTheta 0.5, dDelta DDELTA."""
        if vm != 0:
            raise NotImplementedError("Variable particle masses have not been implemented.")
        if distri == 0:
            raise NotImplementedError('soliton cases 2-4 map a sphere cut from a uniform glass: use distrib 2 (or 1)')
        R0 = 0.5
        distribute.setbound(self, -R0, R0, -R0, R0, -R0, R0)
        self.dICdensprofile = 1       # the glass target is uniform (the pre-glass box is forced periodic)
        self.dICdensdir = 1
        self.dtheta = 0.5
        self.deltastep = DDELTA
        self.nsteps, self.freqout = run_length(case, D, v)
        if distri == 1:
            distribute.setrandomdist(self, -R0, R0, -R0, R0, -R0, R0, self.dxbound / nx)
            x = np.column_stack((self.x, self.y, self.z)).astype(float)
        else:
            x = tip.readtipsy(entry)[0][:, 1:4].astype(float)
        u = 1e8                              # gas only in the pre-glass seed (case 5: the baryon gas)
        if getattr(self, 'preglass', False):
            vel = np.zeros_like(x)
            m, rho, h = 1.0 / len(x), np.ones(len(x)), None
        else:
            S = sp_groundstate()
            r = np.linalg.norm(x, axis=1)
            x, r = x[r < R0], r[r < R0]
            xg, rg = x.copy(), r                   # case 5: the unmapped sphere, for the gas
            Mt = np.interp(RT * RC, S['r'], S['M'])
            rnew = np.interp((r / R0)**3 * Mt, S['M'], S['r'])
            x = x * (rnew / np.maximum(r, 1e-30))[:, None]
            m = Mt / len(x)
            eps = 0.5 * (m / S['rhoc'])**(1.0 / 3.0)
            rho = np.interp(rnew, S['r'], S['rho'])
            vel = np.zeros_like(x)
            print(f'soliton case {case}: rc {RC} kpc, rho_c {S["rhoc"]:.5e}, M(<{RT} rc) {Mt:.5e} '
                  f'({Mt / S["M"][-1]:.4f} of the untruncated), v_c(rc) {S["vc"]:.2f}, {len(x)} particles per '
                  f'soliton, eps {eps:.4f} kpc, {self.nsteps} steps x {DDELTA} ({self.nsteps * DDELTA * myr_per_code():.0f} Myr)')
            if case == 3:
                xr = x * np.array([-1.0, 1.0, 1.0])            # mirrored copy: symmetric about x = 0
                x = np.vstack((x - [0.5 * D * RC, 0, 0], xr + [0.5 * D * RC, 0, 0]))
                vel = np.zeros_like(x)
                vel[:len(rho), 0], vel[len(rho):, 0] = 0.5 * v * S['vc'], -0.5 * v * S['vc']
                rho = np.concatenate((rho, rho))
                E = collision_energy(D, v, S)
                print(f'  collision: D {D} rc, v_rel {v} v_c = {v * S["vc"]:.1f}, E_tot {E:.4e} '
                      f'({"unbound" if E > 0 else "bound"}), fringe spacing 2 pi hbar/m / v_rel '
                      f'{2 * np.pi * S["nu"] / (v * S["vc"]):.2f} rc')
            if case == 4:
                pos = halo_positions(D, v)
                x = np.vstack([x + c for c in pos])
                vel = np.zeros_like(x)
                rho = np.tile(rho, len(pos))
                d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
                print(f'  halo: {len(pos)} solitons in a sphere of {v} rc (seed {HALO_SEED}), separations '
                      f'{d[d > 0].min():.2f}..{d.max():.2f} rc, total mass {len(pos) * Mt:.4e}')
            if case == 5:
                # The dark particles and the stars go out through the create_dark / create_star hooks of
                # IC_createsetup (tipsy columns m, x, v, eps, phi and m, x, v, metals, tform, eps, phi); the gas
                # replaces them in the gas columns below.
                b, cs = D, v * S['vc']
                xs, vs = tracer_stars(S)
                dark = np.column_stack((np.full(len(x), m), x, vel, np.full(len(x), eps), np.zeros(len(x))))
                star = np.column_stack((np.full(len(xs), b * Mt / len(xs)), xs, vs, np.zeros((len(xs), 2)),
                                        np.full(len(xs), eps), np.zeros(len(xs))))
                self.create_dark = lambda tgdata: (dark, len(dark))
                self.create_star = lambda tgdata: (star, len(star))
                self.dark_only = False
                self.adi = 1 if eos == 1 else 0
                self.isothermal_gamma = self.gamma    # bGasIsothermal: P = (gamma - 1) rho u = rho cs^2
                Rg = R0 * GAS_NFRAC**(1.0 / 3.0)
                xg, rg = xg[rg < Rg], rg[rg < Rg]
                rt, rhot, Mgt = gas_profile(S, cs, b * Mt)
                rnew = np.interp((rg / Rg)**3 * Mgt[-1], Mgt, rt)
                x = xg * (rnew / np.maximum(rg, 1e-30))[:, None]
                vel = np.zeros_like(x)
                m = b * Mt / len(x)
                rho = np.interp(rnew, rt, rhot)
                u = cs**2 / (self.gamma - 1)
                print(f'  baryons: b {b}, {len(xs)} stars at {STAR_R} rc, {len(x)} gas particles '
                      f'({"isothermal" if eos != 1 else "adiabatic"}), c_s {v} v_c = {cs:.1f}, gas r50 '
                      f'{np.interp(0.5 * Mgt[-1], Mgt, rt):.3f} rc, gas/dark particle mass {m * len(dark) / Mt:.2e}')
            h = np.full(len(x), eps)
        self.x, self.y, self.z = x[:, 0], x[:, 1], x[:, 2]
        self.vx, self.vy, self.vz = vel[:, 0], vel[:, 1], vel[:, 2]
        self.npart = len(self.x)
        self.ngas = self.npart
        self.mass = np.full(self.npart, m)
        self.rho = rho
        self.h = smth.getsmooth(self, self.ns) if h is None else h
        self.u = np.full(self.npart, u)
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        self.Bz = np.zeros(self.npart)
        print('npart = ', self.npart, ' particle mass = ', m)
