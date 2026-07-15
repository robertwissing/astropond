## Setup for the PONOMARENKO screw-flow kinematic dynamo
#  Reference:
#    R. Teyssier, S. Fromang & E. Dormy, J. Comput. Phys. 218 (2006) 44
#      "Kinematic dynamos using constrained transport with high order Godunov
#       schemes and adaptive mesh refinement" (arXiv:astro-ph/0601715).
#    This is the same problem as the RAMSES tests/mhd/ponomarenko-dynamo case.
#
#  The Ponomarenko dynamo is a smooth (here, top-hat) helical "screw" flow
#  confined to a cylinder of radius R0=1: solid-body rotation plus a uniform
#  axial drift inside, quiescent outside, on a periodic box of side L=8.055
#  (the optimal pitch box of Teyssier et al. 2006):
#    r = sqrt(x^2+y^2) < R0 :  v = ( -Omega y, +Omega x, Vz )
#                              Omega = 0.609711, Vz = 0.792624
#    r >= R0                :  v = 0
#  The seed field is the m=1 helical field B = curl(A) of the RAMSES "mag_screw"
#  vector potential, confined to r < 2:
#    A = h(r) ( x^2, x y, 0 ) sin(k z),   h(r) = B0 (2-r)/r,  k = 2 pi / L
#  giving (analytically divergence-free)
#    Bx = -h(r) x y  k cos(k z)
#    By =  h(r) x^2  k cos(k z)
#    Bz =  h(r) y      sin(k z)
#  with z measured from the box face (z in [0,L]).
#
#  NOTE on kinematic vs dynamical:
#    In RAMSES this runs with scheme='induction' + explicit Ohmic resistivity
#    eta_mag=2.5e-3: the velocity is FROZEN and only the induction equation is
#    integrated. The SPMHD codes in this suite solve the FULL MHD system. The
#    screw flow is NOT a force-balanced equilibrium (the r=R0 shear layer is not
#    supported), so a dynamical run will smooth/relax the flow rather than hold
#    it fixed -- treat the dynamo growth here as qualitative, exercising the
#    induction solver and (artificial, via dMHDAlphaB) magnetic dissipation
#    rather than quantitatively matching Teyssier et al. (2006).
import numpy as np
import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip


class setup_ponomarenko(object):
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
    deltastep = 0.02
    nsteps = 1000
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

    def create(self, nx, distri=0, vm=0,
               entry='datafiles/alfvenwave128_preglass.00000',
               B0=0.1):
        # B0 : amplitude of the seed vector potential (RAMSES A_screw uses 0.1)
        boxlen = 8.055
        R0 = 1.0          # screw-flow cylinder radius
        Rseed = 2.0       # seed-field cylinder radius
        omega = 0.609711  # solid-body rotation rate (Teyssier et al. 2006)
        vzaxial = 0.792624  # axial drift speed
        rhozero = 1.0
        przero = 1.0
        self.rhozero = rhozero
        self.dICdensinner = rhozero
        self.dICdensouter = rhozero
        gam1 = self.gamma - 1.
        uuzero = przero / (gam1 * rhozero)

        print('Setup for the Ponomarenko screw-flow dynamo (Teyssier et al. 2006)...')
        print(' box = ', boxlen, ', density = ', rhozero, ', pressure = ', przero,
              ', Omega = ', omega, ', Vz = ', vzaxial, ', seed amp = ', B0)

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

        # box-centred coordinates (x,y) and cylindrical radius
        xc = self.x
        yc = self.y
        rcyl = np.sqrt(xc**2 + yc**2)

        # --- screw velocity field: rigid rotation + axial drift inside R0 ---
        inside = rcyl < R0
        self.vx = np.where(inside, -omega * yc, 0.0)
        self.vy = np.where(inside, omega * xc, 0.0)
        self.vz = np.where(inside, vzaxial, 0.0)

        self.u = np.full(self.npart, uuzero)

        # --- m=1 helical seed field B = curl(A), confined to r < Rseed ---
        twopi = 2. * np.pi
        k = twopi / boxlen
        za = self.z + half               # z in [0,boxlen]
        sinkz = np.sin(k * za)
        coskz = np.cos(k * za)
        seed = (rcyl > 0) & (rcyl < Rseed)
        # h(r) = B0 (2-r)/r where the seed lives, 0 elsewhere
        hh = np.where(seed, B0 * (Rseed - rcyl) / np.where(rcyl > 0, rcyl, 1.0), 0.0)
        self.Bx = -hh * xc * yc * k * coskz
        self.By = hh * xc**2 * k * coskz
        self.Bz = hh * yc * sinkz

    def getrhoi(self, i):
        return self.rhozero
