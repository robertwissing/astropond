## Setup for the ABC-FLOW kinematic dynamo
#  Reference:
#    R. Teyssier, S. Fromang & E. Dormy, J. Comput. Phys. 218 (2006) 44
#      "Kinematic dynamos using constrained transport with high order Godunov
#       schemes and adaptive mesh refinement" (arXiv:astro-ph/0601715).
#    This is the same problem as the RAMSES tests/mhd/abc-flow case.
#
#  The Arnold-Beltrami-Childress (ABC) flow is a steady Beltrami solution of
#  the incompressible Euler equations (u parallel to curl u) on the periodic
#  unit cube [0,1]^3 with wavenumber k=2*pi and A=B=C=1:
#    u = ( C cos(k y) + A sin(k z),
#          B sin(k x) + A cos(k z),
#          B cos(k x) + C sin(k y) )   (RAMSES velana form, A=B=C=1)
#  A weak uniform seed field B = (0,0,B0) is amplified by the chaotic stretch-
#  twist-fold of the flow (a "fast dynamo").
#
#  NOTE on kinematic vs dynamical:
#    In RAMSES this runs with scheme='induction' + explicit Ohmic resistivity
#    eta_mag=1e-3: the velocity is FROZEN and only the induction equation is
#    integrated. The SPMHD codes in this suite solve the FULL MHD system, so
#    the flow evolves self-consistently. Because the ABC flow is a steady Euler
#    solution and the seed field is weak (Lorentz force negligible), the early,
#    kinematic growth phase is still reproduced before the field saturates.
#    These codes have no explicit Ohmic term, so the effective resistivity is
#    set by the artificial magnetic dissipation (dMHDAlphaB) instead of eta_mag.
import numpy as np
import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip


class setup_abcflow(object):
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
    freqout = 10
    ns = 64
    gamma = 5. / 3.
    periodic = 1
    grav = 0
    deltastep = 0.005
    nsteps = 2000
    dmsolunit = 1.0
    dkpcunit = 1.0
    molweight = 1.0
    adi = 1
    mass = []
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    rho = []
    u = []
    h = []
    P = []
    Bx = []
    By = []
    Bz = []
    Bpsi = []
    soft = []
    tform = []
    metals = []
    pot = []
    dxbound = []
    dybound = []
    dzbound = []
    totvol = []
    rcylmin2 = 0
    rmin2 = 0
    rcylmax2 = 10**22
    rmax2 = 10**22
    rhozero = 1.0

    def __init__(self):
        pass

    def create(self, nx, distri=0, vm=0,
               entry='datafiles/alfvenwave128_preglass.00000',
               B0=1.0e-3, amp=1.0):
        # B0  : uniform seed magnetic field (Bz), default 1e-3 (RAMSES C_region)
        # amp : ABC flow amplitude A=B=C, default 1.0
        boxlen = 1.0
        rhozero = 1.0
        przero = 1.0
        self.rhozero = rhozero
        self.dICdensinner = rhozero
        self.dICdensouter = rhozero
        gam1 = self.gamma - 1.
        uuzero = przero / (gam1 * rhozero)

        print('Setup for the ABC-flow kinematic dynamo (Teyssier et al. 2006)...')
        print(' box = ', boxlen, ', density = ', rhozero, ', pressure = ', przero,
              ', flow amp A=B=C = ', amp, ', seed Bz = ', B0)

        half = 0.5 * boxlen
        deltax = boxlen / nx
        distribute.setbound(self, -half, half, -half, half, -half, half)

        if distri == 0:
            distribute.setcloseddist(self, -half, half, -half, half, -half, half, deltax)
        elif distri == 1:
            distribute.setrandomdist(self, -half, half, -half, half, -half, half, deltax)
        else:
            tgdata, tddata, tsdata, data_header, time = tip.readtipsy(entry)
            self.x = tgdata[:, 1]
            self.y = tgdata[:, 2]
            self.z = tgdata[:, 3]
            self.rho = tgdata[:, 7]

        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        self.z = np.asarray(self.z)
        self.npart = len(self.x)
        self.ngas = self.npart
        totmass = self.dxbound * self.dybound * self.dzbound * rhozero
        self.mass = [totmass / self.npart] * self.npart
        if vm == 1:
            print("Varying masses -> ONLY FOR RAND + RELAXING TO GLASS")
            for i in range(0, self.npart, 2):
                dmrat = 0.25 * self.mass[i] * np.random.rand()
                self.mass[i] = self.mass[i] + dmrat
                self.mass[i + 1] = self.mass[i + 1] - dmrat

        print('npart = ', self.npart, ' particle mass = ', self.mass[0])
        self.h = smth.getsmooth(self, 200)

        # shift box-centred coords into [0,boxlen] so the field pattern matches RAMSES
        twopi = 2. * np.pi
        xa = self.x + half
        ya = self.y + half
        za = self.z + half

        self.vx = amp * (np.cos(twopi * ya) + np.sin(twopi * za))
        self.vy = amp * (np.sin(twopi * xa) + np.cos(twopi * za))
        self.vz = amp * (np.cos(twopi * xa) + np.sin(twopi * ya))

        self.u = np.full(self.npart, uuzero)

        # weak uniform seed field along z
        self.Bx = np.zeros(self.npart)
        self.By = np.zeros(self.npart)
        self.Bz = np.full(self.npart, B0)

    def getrhoi(self, i):
        return self.rhozero
