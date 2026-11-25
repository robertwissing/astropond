"""
Setup for the 3D Gaussian overdensity test from Nori & Baldi (2018).
If choosing random pre-ICs, glass relaxation is done to fit these to the desired density profile (recommended approach).
Lattice pre-ICs are not supported (no spherical lattice stretch map): use random (1) or glass (2).

Original setup: Ida Martine Brynjulfsen.
"""
import numpy as np
import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip
import IC_denstable as denstable

class setup_gaussOverdens(object):
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
        self.rho0 = 167  # Msun kpc^-3, background density
        self.c = 1       # background parameter
        self.sigma = 500 # kpc

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
        return self.rho0 * (self.c + np.exp(-r**2/(2*self.sigma**2)))

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


    def create(self, nx, distri=0, vm=0, entry='1'):
        """
        nx: resolution (number of elements along x)
        vm: variable SPH particle masses (0=equal-mass particles)
        distri: 0=lattice (FCC or cubic, chosen by IC_createsetup), 1=random (also the
                pre-glass IC), 2=rebuild the final IC from the relaxed glass IC in entry
        """ 
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
