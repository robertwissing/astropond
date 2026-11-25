#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:30:46 2018

@author: robertwi
"""

## Setup for ORZAG-TANG
import numpy as np
import IC_distribute as distribute
import IC_fixdens as fixdens
import IC_smoothlength as smth
import readtipsy as tip


def flat_cosmic_time(a, H0, omega_m=1., omega_lambda=0.):
    """Cosmic time for a flat matter + Lambda cosmology.

    The EdS limit is evaluated separately so this is also safe when
    ``omega_lambda == 0``.
    """
    a = np.asarray(a, dtype=float)
    if omega_m <= 0. or omega_lambda < 0.:
        raise ValueError("flat cosmowave requires omega_m > 0 and omega_lambda >= 0")
    if abs((omega_m + omega_lambda) - 1.) > 1.e-12:
        raise ValueError("flat cosmowave requires omega_m + omega_lambda = 1")
    if omega_lambda == 0.:
        return 2.*a**1.5/(3.*H0*np.sqrt(omega_m))
    return (2./(3.*H0*np.sqrt(omega_lambda))
            * np.arcsinh(np.sqrt(omega_lambda/omega_m)*a**1.5))


def eds_time(a, H0):
    """Backward-compatible Einstein-de Sitter specialization."""
    return flat_cosmic_time(a, H0, 1., 0.)


class setup_cosmowave(object):
    dICdensRsmooth = 0.1
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0 
    dICdensinner = 1.0 # calculated depending on mass
    dICdensouter = 1.0
    rhoit=0
    ns=64
    freqout=1
    inflow=0
    npart=0
    ngas=0
    ndark=0
    nstar=0
    ndim=3
    time=0
    grav=0
    H0=2.894405
    omega0=1.0
    omegaLambda=0.0
    cosmo=1
    molweight=1.0
    gamma=5./3.
    periodic=1
    deltastep=0.0005
    grav=0
    nsteps=100
    dmsolunit=1.0
    dkpcunit=1000.0
    adi=1
    shape=2
    mass=[]
    x=[]
    y=[]
    z=[]
    vx=[]
    vy=[]
    vz=[]
    rho=[]
    u=[]
    h=[]
    P=[]
    Bx=[]
    By=[]
    Bz=[]
    dxbound=[]
    dybound=[]
    dzbound=[]
    totvol=[]
    rcylmin2= 0
    rmin2= 0
    rcylmax2= 10**22
    rmax2= 10**22
    _rho_z_grid=None
    wk=0.
    x0=0.
    xdot0=0.
    rhozero=1.
    drho=0.
    Bzero=1.
    dB=0.
    a=0.
    kappa=1.
    def __init__(self):
        pass
    def create(self,nx,distri=0,vm=0,entry='datafiles/alfvenwave128_preglass.00000',
               case=3,factor_A=1,factor_G=0,A_u=1.e-1,theta=1.e-1,
               clean_fch=1.,clean_damping=0.):
        # case selects one of the unique test setups from Berlok 2022 (MNRAS 515 3492):
        #   0 = standing Alfven wave           (paper Sec. 5.1.1, eqs. 75, 76)
        #   1 = traveling Alfven wave          (paper Sec. 5.1.2, eqs. 79, 80; n=2)
        #   2 = standing compressible wave     (paper Sec. 5.2.1/2/3, eqs. 81, 82)
        #   3 = traveling compressible gamma=4/3 (paper Sec. 5.2.1, eqs. 83, 84; n=3)
        #   4 = divergence-cleaning test in EdS (this work):
        #       A deliberately non-solenoidal Fourier mode with v=0.  In the
        #       intended comoving GLM system, B_c=a^2 B_phys and
        #       Q_c=sqrt(a*rho_c)*BClean obey
        #         dB_c/dt = -(c_h/a) grad_x Q_c,
        #         dQ_c/dt = -(c_h/a) div_x B_c - Q_c/tau.
        #       For gamma=4/3 in EdS, c_h is proportional to a^-1/2 and the
        #       coefficients are constant in s=ln(a/a_i).  With damping off,
        #         dB_c/Bc0 = A_u cos(Omega_h*s) sin(k(x-x0)),
        #         Q_c/Bc0  =-A_u sin(Omega_h*s) cos(k(x-x0)),
        #       Omega_h=fch*omega_S.  Thus the comoving
        #       amplitude and B/Q cleaning energy stay constant: there is no
        #       a^2 envelope.  Optional damping is checked against the damped
        #       two-variable solution by IC_analysis_cosmowave.py.
        #       IC: zero magnetic background, dBx=A_u*Bscale*sin(k(x-x0)),
        #       psi(a_i)=0, v=0, drho=0, with Bscale=V_S*sqrt(rho_c).
        #       The zero background is essential: a transverse B0 combined with
        #       the deliberately nonzero div(B) drives an O(A_u) tensile/Powell
        #       force and no longer isolates the cleaning subsystem.  With B0=0
        #       magnetic forces are O(A_u^2), while c_h is set by the sound speed.
        #   5 = standing compressible gamma=5/3 in LambdaCDM
        #       (paper Fig. 5, Secs. 4 and 5.2.4; numeric ODE reference)
        #   6 = standing isothermal compressible wave in EdS
        #       (paper Fig. 3, gamma=1, omega_S=2*pi)
        #   7 = standing adiabatic compressible wave in EdS
        #       (paper Fig. 4, gamma=5/3, omega_S=pi/5)
        # factor_A, factor_G scale the paper-default frequencies (case 4 is the
        # exception: it always runs MHD with zero background and c_h set by V_S):
        #   omega_A = factor_A * pi    (0 = MHD off; 1 = paper default)
        #   omega_G = factor_G * pi/2  (0 = self-gravity off; 1 = paper default)
        # omega_S is case-dependent exactly as in the paper. A_u is deliberately
        # larger than the paper's 1e-6 default so the mode remains measurable
        # above SPH noise.
        # theta is used only when factor_G != 0.
        self.drho = 0
        #self.rhozero  = 1.-self.drho-0.000005
        """
        #dimensionless amplitude Au=10**-6
        #wavenumber kL=2pi k=2pi
        #z=127 ai=1/128
        #du(ai)=dA*Va cos(kx) for alfven waves
        #du(ai)=dA*Vs cos(kx) for compressible
        #The travelling waves are initialized using the analytic solutions at
        #a = ai . The amplitudes shown in the figures have been divided by
        #the value of Au such that the velocity amplitude appears as 1 at a =
        #ai
        We use this solution to initialize a simulation with A = π and
        show the amplitude evolution of the wave in Fig. 1.
        #omega_A=pi
        #omega_A=2pi
        show δBc /Bc multiplied with a1/4 and δu/VA
        multiplied with a3/4 as a function of ln a/ai 
        
        omega_S/V_S=k/H_0
        omega_A/V_A=k/H_0
        
        V_A=B_c/sqrt(rho_c)
        V_S=sqrt(gamma*rho_c0/rho_c)
        V_G=sqrt(4piGrho_c)/k
        omega_x=kV_x/H_0
        K=sqrt(omega_A**2-1/16)
        phi=Kln(a/a_i) phi(a_i)=0   
        
        du(x,a_i)=V_A*A_u*cos(phi)*cos(kx)
        du(x,a_i)=H_0/k*omega_A*A_u*cos(kx)
        B_c(a_i)=(H_0/k)*omega_A*sqrt(rho_c)
        The critical density of the Universe in simulation units is 1.
        
        dTheta2          = 0.725
        dHubble0         = 2.894405
        dOmega0          = 1.0
        dLambda          = 0.0
        dRedTo   = 0.0
        bComove          = 1
        """
        dx = 0.5
        wavelength=2*dx
        k=2.*np.pi/wavelength
        deltax = wavelength/nx
        A_u = float(A_u)
        if A_u <= 0:
            raise ValueError("A_u must be positive")

        case = int(case)
        if case not in (0, 1, 2, 3, 4, 5, 6, 7):
            raise ValueError("cosmowave case must be one of 0, 1, 2, 3, 4, "
                             "5, 6, 7")

        clean_fch = float(clean_fch)
        clean_damping = float(clean_damping)
        if clean_fch < 0.:
            raise ValueError("clean_fch must be non-negative")
        if clean_damping < 0.:
            raise ValueError("clean_damping must be non-negative")
        self.clean_fch = clean_fch
        self.clean_damping = clean_damping

        if case == 4 and float(factor_A) == 0.:
            raise ValueError("cosmowave case 4 requires factor_A != 0 "
                             "(factor_A = 0 selects MHD off, but the "
                             "divergence-cleaning test is always magnetic)")

        omega_A = factor_A * np.pi
        omega_S = (2.*np.pi if case == 6 else
                   np.pi/5. if case in (5, 7) else np.pi)
        omega_G = factor_G * np.pi/2

        self.gamma = (1. if case == 6 else
                      5./3. if case in (5, 7) else 4./3.)
        self.adi = 0 if case == 6 else 1
        self.omega0 = 0.3 if case == 5 else 1.0
        self.omegaLambda = 0.7 if case == 5 else 0.0
        self.nsteps = 300 if case == 4 else 100

        V_A=(self.H0/k)*omega_A
        V_S=(self.H0/k)*omega_S
        V_G=(self.H0/k)*omega_G
        print("V_a ",V_A,"V_S ",V_S,"V_G ",V_G)
        
        sigma2=omega_A**2+omega_S**2-omega_G**2
        if case in (2,3) and sigma2 <= 1./16.:
            raise ValueError("compressible Berlok cases require sigma^2 > 1/16")
        sigma=np.sqrt(max(sigma2,0.))

        kappa=np.sqrt(max(sigma2-1./16.,0.))
        self.kappa=kappa
        kappa_A2=omega_A**2-1./16.
        if case in (0,1) and kappa_A2 <= 0.:
            raise ValueError("Alfven Berlok cases require omega_A^2 > 1/16")
        kappa_A=np.sqrt(max(kappa_A2,0.))
        self.kappa_A=kappa_A
        alfven = case in (0,1)
        if case == 0:
            """ standing Alfven (paper 5.1.1) """
            a = 1./128.
        elif case == 1:
            """ traveling Alfven (paper 5.1.2), n=2 """
            n = 2
            a = np.exp(-2*n*np.pi/kappa_A)
        elif case in (2, 5, 6, 7):
            """ standing compressible (paper 5.2.1/2/3) """
            a = 1./128.
        elif case == 4:
            """ divergence-cleaning test (this work) """
            a = 1./128.
        else:
            """ traveling compressible gamma=4/3 (paper 5.2.1), n=3 """
            n = 3
            a = np.exp(-2*n*np.pi/kappa)
        self.a=a
        # Gasoline writes both dDelta and nSteps, so choose dDelta explicitly
        # from the appropriate Friedmann solution to land at a=1.
        self.deltastep=(flat_cosmic_time(1., self.H0, self.omega0,
                                         self.omegaLambda)
                        - flat_cosmic_time(a, self.H0, self.omega0,
                                           self.omegaLambda))/self.nsteps
        self.grav=int(float(factor_G) != 0.)
        self.dtheta=float(theta) if self.grav else 0.6
        self.dtheta2=float(theta) if self.grav else 0.8
        if self.grav and self.dtheta <= 0.:
            raise ValueError("gravity opening angle theta must be positive")
        phi=kappa*np.log(1) #log(a/a_i)
        if self.grav:
            self.rhozero=((k*V_G)**2)/(4*np.pi)
        print("rhozero", self.rhozero)
        
        valf=V_A*a**(-0.5)
        # Case 4 has no magnetic background.  Bzero is only its injected-field
        # normalization, chosen from the sound speed so the linear cleaning
        # speed and analytic phase are independent of the perturbation itself.
        Bscale = V_S if case == 4 else V_A
        Bzero=Bscale*np.sqrt(self.rhozero)*a**(-0.5)
        # Gasoline represents an isothermal gas with bGasIsothermal=1 but still
        # requires a finite internal-energy value.  Use an epsilon-offset EOS
        # gamma only for that storage conversion; the scientific gamma remains 1.
        gamma_eos = 1.0000000001 if case == 6 else self.gamma
        # ``self.gamma`` remains the scientific gamma used by the analytic
        # setup.  Tipsy stores temperature rather than internal energy, so its
        # conversion must use the same finite EOS gamma that Gasoline reads
        # for an isothermal run.  Otherwise gamma=1 makes the generic
        # (gamma-1) conversion silently write an all-zero thermal field.
        self.gamma_eos = gamma_eos
        gam1_eos = gamma_eos - 1.
        
        print("Bzero",Bzero)
        
        """ Incompressible """
        #uzero=V_A*A_u*a**(-1)
        #cs=uzero*10
        
        """ Compressible / Alfven coordinate-velocity amplitude.

        Tipsy stores xdot=u/a for a comoving Gasoline run.  The Berlok initial
        condition is delta-u=A_u*V, hence xdot=A_u*V/a here.  The analysis must
        multiply snapshot velocities by a before comparing with delta-u.
        """
        uzero=(V_A if alfven else V_S)*A_u*a**(-1)
        cs=V_S*a**((-3*(self.gamma-1.))/2)
        print('cs',cs,'cs**2',cs**2,'Va',valf)
        print('uzero',uzero)
        
        uuzero = cs**2/(gamma_eos*gam1_eos)
        pzero=uuzero*self.rhozero*gam1_eos
        print("uuzeroth",uuzero,"pzero",pzero)
        #uuzero=vA**2  B**2/(rho*self.gamma*gam1)
        dy=dx
        dz=dx
        distribute.setbound(self,-dx,dx,-dy,dy,-dz,dz)
        self.wk = k
        self.x0 = (-dx,-dy,-dz)
        self.xdot0 = -dx
        glass_input_mass = None
        glass_input_rho = None
        if int(distri) == 0:
            distribute.setcubicdist(self,-dx,dx,-dy,dy,-dz,dz,deltax)
        elif int(distri) == 1:
            distribute.setrandomdist(self,-dx,dx,-dy,dy,-dz,dz,deltax)
        else:
            tgdata, _tddata, _tsdata, data_header, _time = tip.readtipsy(entry)
            expected = int(round(float(nx)))**3
            if int(data_header[2]) != expected:
                raise ValueError(
                    "cosmowave glass has %d gas particles, expected nx^3=%d; "
                    "the cached glass is incompatible and must be regenerated "
                    "under a new name" % (int(data_header[2]), expected))
            self.x = np.asarray(tgdata[:,1], dtype=float).copy()
            self.y = np.asarray(tgdata[:,2], dtype=float).copy()
            self.z = np.asarray(tgdata[:,3], dtype=float).copy()
            self.rho = np.asarray(tgdata[:,7], dtype=float).copy()
            glass_input_mass = np.asarray(tgdata[:,0], dtype=float)
            glass_input_rho = self.rho.copy()
        #dbrho=A_u*omega_S*np.sqrt(a)/(4*sigma**2)*4*kappa
        if case == 3:
            self.drho = self.rhozero*A_u*omega_S*np.sqrt(a)/(4*sigma**2)
        else:
            # standing waves (0,2) and traveling Alfven (1) have no density perturbation at a_i
            self.drho = 0.
        print("DRHO",self.drho,"wk",self.wk,"xdot0",self.xdot0,"kappa",kappa)
        if self.drho != 0:
            xgrid = np.linspace(-dx, dx, 20001)
            self._rho_z_grid = (xgrid, self.getrho(xgrid))
            # IC_createsetup writes this as a self-describing signed-x table;
            # runtest feeds it to glassgen so a case-3 glass is RELAXED in the
            # analytic Berlok density profile, rather than relaxing a uniform
            # glass and stretching it afterwards.
            self.glass_density_table = (
                xgrid.copy(), self._rho_z_grid[1].copy(), 'signed_x')
            if int(distri) == 2:
                # Guard against accidentally supplying an old uniform cached
                # glass.  The measured SPH density in a profile-relaxed glass
                # must contain the analytic fundamental with the right phase.
                good = np.isfinite(glass_input_rho) & (glass_input_rho > 0.)
                phase = self.wk*(np.asarray(self.x)[good] - self.xdot0)
                design = np.column_stack((np.ones(np.count_nonzero(good)),
                                          np.cos(phase), np.sin(phase)))
                volume = glass_input_mass[good]/glass_input_rho[good]
                rootw = np.sqrt(volume)
                coeff = np.linalg.lstsq(
                    design*rootw[:,None], glass_input_rho[good]*rootw,
                    rcond=None)[0]
                measured_mode = complex(coeff[1], -coeff[2])
                target_mode = complex(4.*kappa*self.drho, self.drho)
                profile_error = abs(measured_mode-target_mode)/abs(target_mode)
                print("profile-relaxed glass density mode", measured_mode,
                      "target", target_mode, "relative error", profile_error)
                if profile_error > 0.25:
                    raise ValueError(
                        "cosmowave case-3 glass was not relaxed against its "
                        "analytic density profile (fundamental-mode relative "
                        "error %.3g)" % profile_error)
            elif self.preglass:
                # The PRE-glass seed only: glassgen relaxes it against the table
                # just exported, so stretch-mapping it first is work the
                # relaxation immediately overwrites.  A standalone rand IC is
                # NOT this case and falls through to the stretch map below.
                print("PRE-GLASS DISTRIBUTION: PROFILE COMES FROM THE GLASS "
                      "RELAXATION, NOT FROM A STRETCH MAP")
            else:
                print("STRETCHING THE NON-GLASS DENSITY HERE")
                fixdens.set_density_profile(self,-dx,dx,1)
        

        #Bzero=V_A*np.sqrt(self.rhozero)*a**(-0.5)
        #c_s=V_S*a**(-0.5)
        #P_th=rho*uzero*gam1
        #P_th=rho*cs**2/(gamma)
        #Pmag=B**2/2=rho*vA**2/2          P_th/P_mag = (cs**2/vA**2)*(2/gamma)
        

        self.npart=len(self.x)
        self.ngas=self.npart
        totmass = self.rhozero*self.dxbound*self.dybound*self.dzbound
        self.mass = [totmass/self.npart]*self.npart
        self.h=smth.getsmooth(self,200)
        
        print('npart = ',self.npart,' particle mass = ',self.mass[0], 'wk',self.wk,'x0',self.xdot0)
  
        for i in range(self.npart):
            cosx1 = np.cos(self.wk*(self.x[i]-self.xdot0))
            sinx1 = np.sin(self.wk*(self.x[i]-self.xdot0))
            rhoi = self.getrhoi(i) if self.drho != 0 else self.rhozero
            self.rho[i] = rhoi
            if alfven:
                # Alfven: dv perpendicular to B0 (B0 along x-hat, dv along y-hat)
                self.vx.append(0.)
                self.vy.append(uzero*cosx1)
                self.vz.append(0.)
            elif case == 4:
                # div-cleaning: no v
                self.vx.append(0.)
                self.vy.append(0.)
                self.vz.append(0.)
            else:
                # compressible: dv parallel to k (along x-hat)
                self.vx.append(uzero*cosx1)
                self.vy.append(0.)
                self.vz.append(0.)
            if case == 0:
                # standing Alfven: uniform B0 along x-hat.
                # factor_A=0 makes Bzero=0.
                self.Bx.append(Bzero)
                self.By.append(0.)
                self.Bz.append(0.)
            elif case == 1:
                # traveling Alfven eigenmode (paper eq. 79):
                # dBc/Bc = A_u*sqrt(a_i)/(4*OmegaA) * [sin(kx-psi) - 4*kappa_A*cos(kx-psi)], psi=0 at a_i
                dBy = Bzero*A_u*np.sqrt(a)/(4.*omega_A)*(sinx1 - 4.*kappa_A*cosx1)
                self.Bx.append(Bzero)
                self.By.append(dBy)
                self.Bz.append(0.)
            elif case in (2, 5, 6, 7):
                # standing compressible: B0 along z-hat, dB(a_i)=0
                self.Bx.append(0.)
                self.By.append(0.)
                self.Bz.append(Bzero)
            elif case == 4:
                # Divergence-cleaning IC: zero background and a longitudinal
                # field dBx=A_u*Bscale*sin(k(x-x0)).  This gives
                # div(Bc)=A_u*Bscale*k*cos(k(x-x0)) at a_i with psi(a_i)=0,
                # without the O(A_u) tensile force caused by a transverse B0.
                self.Bx.append(Bzero*A_u*sinx1)
                self.By.append(0.)
                self.Bz.append(0.)
            else:
                # case 3: traveling compressible gamma=4/3 (existing behavior)
                # uses dBc/Bc = drho/rho simplifying assumption (paper eq. 37)
                self.Bx.append(0.)
                self.By.append(0.)
                self.Bz.append(Bzero+Bzero*(rhoi-self.rhozero)/self.rhozero)
            if case == 6:
                uui = uuzero
            else:
                uui=(pzero+cs**2*(rhoi-self.rhozero))/((self.gamma-1.)*rhoi)
            self.u.append(uui)

             
    def getrho(self,x):
        return (self.rhozero + self.drho*(4*self.kappa*np.cos(self.wk*(x - self.xdot0))-np.sin(self.wk*(x - self.xdot0))))

    def getrhoi(self,i):
        return (self.rhozero + self.drho*(4*self.kappa*np.cos(self.wk*(self.x[i] - self.xdot0))-np.sin(self.wk*(self.x[i] - self.xdot0))))

    def getamplitudes():
        return
    
    #------------------------------------------------------
    #+
    #  transform vectors from rotated to unrotated coords
    #+
    #------------------------------------------------------
    def transform_vec(self,xvec,sina,sinb,cosa,cosb):
         x = [0,0,0]
         x[0] = xvec[0]*cosa*cosb - xvec[1]*sinb - xvec[2]*sina*cosb
         x[1] = xvec[0]*cosa*sinb + xvec[1]*cosb - xvec[2]*sina*sinb
         x[2] = xvec[0]*sina + xvec[2]*cosa
         return x
