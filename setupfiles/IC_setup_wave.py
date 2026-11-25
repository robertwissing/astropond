#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:30:46 2018

@author: robertwi
"""

## Setup for LINEAR (M)HD WAVES
## Standard linear-wave convergence test (Stone et al. 2008, ApJS 178, 137;
## Gardiner & Stone 2008). A small-amplitude eigenmode of the linearised ideal
## (M)HD equations is initialised on a uniform background; after one wave period
## it should return unchanged, so the L1 error vs the IC measures numerical
## dissipation and the formal order of convergence.
##
## wavetype selects the eigenmode (wave propagates along x):
##   0 = sound wave        (hydro, B=0,  speed a    = 1)
##   1 = fast magnetosonic (MHD,         speed c_f  = 2)
##   2 = slow magnetosonic (MHD,         speed c_s  = 1/2)
##   3 = Alfven            (MHD,         speed c_A  = 1)
## MHD background (Stone et al. 2008): rho=1, P=1/gamma, B=(1, sqrt(2), 1/2),
## gamma=5/3 -> a=1, c_A=1, c_s=1/2, c_f=2.
##
## AMPLITUDE vs DISTRIBUTION.  The default ampl=3e-2 is a GLASS number: a
## relaxed uniform glass has a particle-disorder floor that shows up in the
## eigenmode projection at ~4.5e-3 after one period, whatever the amplitude, so
## the old 1e-4 default put the wave ~50x BELOW its own noise (measured nx=32,
## nSmooth=64: signal/noise 0.02 for all four modes).  S/N crosses 1 at
## ampl ~ 5e-3; above that:
##   ampl   1e-2    3e-2     1e-1
##   S/N     ~2      ~5-7    ~15-21      (all four modes, after one period)
##   A2/A1  0.02-0.035  0.05-0.07  0.13-0.21   (2nd harmonic = steepening)
## The compressive modes leave the linear regime as the sinusoid steepens: the
## measured A2/A1 tracks t/(2 t_shock) with t_shock = 2/((gamma+1) k dv) to
## ~20%, so one period is shock-free only for ampl < ~0.2.  3e-2 is the sweet
## spot -- S/N 5-7 with under 7% harmonic content, and it minimises L1/ampl for
## the sound mode.  The Alfven mode is an exact nonlinear solution (A2/A1 ~ 1e-4
## even at 0.1), so for it just push the amplitude up.
## On a lattice (distrib 0) the floor is 4e-5 instead, so a lattice convergence
## study can drop back to 1e-4 for a cleaner order measurement -- but see
## LATTICE_INSTABILITY.md: never run it at nSmooth 57-80.
import numpy as np
import IC_distribute as distribute
import IC_fixdens as fixdens
import IC_smoothlength as smth
import readtipsy as tip


def mhd_primitive_eigvec(rho0,Bx,By0,Bz0,gamma,P0,target):
    """Right eigenvector (real, unit norm) of the 1D ideal-MHD primitive matrix
    for the eigenvalue closest to `target`. Variable order:
    (rho, vx, vy, vz, By, Bz, P)."""
    A=np.zeros((7,7))
    A[0,1]=rho0
    A[1,4]=By0/rho0; A[1,5]=Bz0/rho0; A[1,6]=1./rho0
    A[2,4]=-Bx/rho0
    A[3,5]=-Bx/rho0
    A[4,1]=By0; A[4,2]=-Bx
    A[5,1]=Bz0; A[5,3]=-Bx
    A[6,1]=gamma*P0
    vals,vecs=np.linalg.eig(A)
    idx=np.argmin(np.abs(vals.real-target))
    R=vecs[:,idx].real
    nrm=np.linalg.norm(R)
    if nrm>0:
        R=R/nrm
    # fix the overall sign so the density (or, for incompressible modes, the
    # first non-zero) component is positive -> reproducible phase
    for c in R:
        if abs(c)>1e-12:
            if c<0: R=-R
            break
    return R, vals[idx].real


class setup_wave(object):
    dICdensRsmooth = 0.1
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0
    dICdensinner = 1.0 # calculated depending on mass
    dICdensouter = 1.0
    inflow=0
    rhoit=0
    npart=0
    ngas=0
    ndark=0
    nstar=0
    ndim=3
    time=0
    grav=0
    cosmo=0
    molweight=1.0
    gamma=5./3.
    ns=64
    periodic=1
    deltastep=0.01
    nsteps=1000
    freqout=50
    dmsolunit=1.0
    dkpcunit=1.0
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
    Bpsi=[]
    soft=[]
    tform=[]
    metals=[]
    pot=[]
    dxbound=[]
    dybound=[]
    dzbound=[]
    totvol=[]
    rcylmin2= 0
    rmin2= 0
    rcylmax2= 10**22
    rmax2= 10**22
    wk=0.
    x0=0.
    xdot0=0.
    rhozero=1.
    drho=0.
    _rho_z_grid=None
    glass_density_table=None

    def __init__(self):
        pass
    def create(self,nx,distri=0,vm=0,entry='datafiles/alfvenwave128_preglass.00000',wavetype=0,ampl=3e-2):
    #--setup parameters
    # wavetype : 0=sound 1=fast 2=slow 3=Alfven ; ampl : eigenmode amplitude
        self.rhozero = 1.
        gam1 = self.gamma - 1.
        P0 = 1.0/self.gamma            # -> sound speed a = sqrt(gamma*P/rho) = 1
        if wavetype==0:
            Bx=By0=Bz0=0.0
        else:
            Bx=1.0; By0=np.sqrt(2.); Bz0=0.5

        # target wave speed for eigenmode selection
        a2=self.gamma*P0/self.rhozero
        cA2=Bx**2/self.rhozero
        b2=(Bx**2+By0**2+Bz0**2)/self.rhozero
        cf=np.sqrt(0.5*((a2+b2)+np.sqrt((a2+b2)**2-4.*a2*cA2)))
        cs=np.sqrt(0.5*((a2+b2)-np.sqrt(max((a2+b2)**2-4.*a2*cA2,0.))))
        if   wavetype==0: target=np.sqrt(a2)
        elif wavetype==1: target=cf
        elif wavetype==2: target=cs
        else:             target=np.sqrt(cA2)

        R,cwave = mhd_primitive_eigvec(self.rhozero,Bx,By0,Bz0,self.gamma,P0,target)
        Rrho,Rvx,Rvy,Rvz,RBy,RBz,RP = R
        self.cwave = cwave

        print('Setup for linear (M)HD wave: wavetype=',wavetype,' speed=',cwave)
        print(' background rho=',self.rhozero,' P=',P0,' B=(',Bx,',',By0,',',Bz0,')')
        print(' eigenvector (rho,vx,vy,vz,By,Bz,P)=',np.round(R,5))

        dx = 0.5
        wavelength = 2*dx
        deltax = wavelength/nx
        # y,z boundaries chosen as a multiple of the close-packed particle spacing
        fac = 12.*(int((1.-np.finfo(float).eps)*2./6.) + 1)
        dy = fac*deltax*np.sqrt(0.75)
        dz = fac*deltax*np.sqrt(6.)/3.
        distribute.setbound(self,-dx,dx,-dy,dy,-dz,dz)
        self.wk = 2.*np.pi/wavelength
        self.x0 = (-dx,-dy,-dz)
        self.xdot0 = -dx
        # Density-perturbation amplitude.  The Alfven mode is exactly
        # incompressible, but np.linalg.eig returns its rho component as
        # roundoff (~1e-17) rather than 0, so snap it: the same 1e-12 the
        # eigenvector sign fix uses.  Otherwise a mode with no density
        # structure still triggers the stretch map and the glass table below.
        self.drho = ampl*Rrho if abs(Rrho) > 1.e-12 else 0.

        if distri==0:
            distribute.setcloseddist(self,-dx,dx,-dy,dy,-dz,dz,deltax,periodic=True)
        elif distri==1:
            distribute.setrandomdist(self,-dx,dx,-dy,dy,-dz,dz,deltax)
        else:
            tgdata,tddata,tsdata,data_header,time=tip.readtipsy(entry);
            self.x=tgdata[:,1]
            self.y=tgdata[:,2]
            self.z=tgdata[:,3]
            self.rho=tgdata[:,7]
        # A longitudinal mode carries a density perturbation.  Build its profile
        # grid whatever the distribution: the stretch map below needs it, and it
        # is ALSO the glass relaxer's target.
        if self.drho != 0:
            xgrid = np.linspace(-dx, dx, 20001)
            self._rho_z_grid = (xgrid, self.getrho(xgrid))
            # IC_createsetup writes this as a self-describing signed-x table;
            # runtest feeds it to glassgen so the glass is RELAXED in the
            # analytic profile.  Without it the target is the .param's
            # dICdensprofile=1 (uniform) and the density wave does not survive
            # the relaxation at all.
            self.glass_density_table = (
                xgrid.copy(), self._rho_z_grid[1].copy(), 'signed_x')
            # Realise the wave by displacing particles -- lattice/rand only: a
            # glass arrives already relaxed in the profile, and the pre-glass
            # seed is about to be, so stretching either is work the relaxation
            # overwrites.
            if distri != 2 and not self.preglass:
                fixdens.set_density_profile(self,-dx,dx)

        uuzero = P0/(gam1*self.rhozero)

        self.npart=len(self.x)
        self.ngas=self.npart
        totmass = self.rhozero*self.dxbound*self.dybound*self.dzbound
        self.mass = [totmass/self.npart]*self.npart
        if vm==1:
            print("Varying masses -> ONLY FOR RAND + RELAXING TO GLASS");
            for i in range(0,self.npart,2):
                dmrat=0.25*self.mass[i]*np.random.rand()
                self.mass[i]=self.mass[i]+dmrat
                self.mass[i+1]=self.mass[i+1]-dmrat

        # set the period so the run covers exactly ten wave crossing
        if abs(cwave)>0:
            T = 10.0*wavelength/abs(cwave)
            self.deltastep = T/self.nsteps

        self.h=smth.getsmooth(self,200)
        print('npart = ',self.npart,' particle mass = ',self.mass[0], 'wk',self.wk,'period',self.nsteps*self.deltastep)

        for i in range(self.npart):
            sinx1 = np.sin(self.wk*(self.x[i]-self.xdot0))
            rhoi  = self.rhozero + self.drho*sinx1
            self.vx.append(ampl*Rvx*sinx1)
            self.vy.append(ampl*Rvy*sinx1)
            self.vz.append(ampl*Rvz*sinx1)
            self.Bx.append(Bx)
            self.By.append(By0 + ampl*RBy*sinx1)
            self.Bz.append(Bz0 + ampl*RBz*sinx1)
            Pi = P0 + ampl*RP*sinx1
            self.u.append(Pi/(gam1*rhoi))

    def getrho(self,x):
        return (self.rhozero + self.drho*np.sin(self.wk*(x - self.xdot0)))
    def getrhoi(self,i):
        return (self.rhozero + self.drho*np.sin(self.wk*(self.x[i] - self.xdot0)))
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
