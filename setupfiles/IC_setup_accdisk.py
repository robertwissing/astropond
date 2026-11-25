## Setup for ORZAG-TANG
import numpy as np
import IC_distribute as distribute
import IC_fixdens as fixdens
import IC_smoothlength as smth
import readtipsy as tip

class setup_accdisk(object):
    dICdensRsmooth = 0.1
    dICdensprofile = 7
    dICdensdir = 5
    dICdensR = 10.0 
    dICdensinner = 360.0 # calculated depending on mass
    dICdensouter = 1E-12
    rhoit=0
    ns=58
    npart=0
    ngas=0
    ndark=0
    nstar=0
    ndim=3
    time=0
    gamma=5./3.
    inflow=0
    periodic=1
    deltastep=0.62831853071  #1/10 of an orbit
    freqout=2
    nsteps=1000 #100 orbits
    dmsolunit=1.0
    dkpcunit=4.84814E-9
    adi=0
    grav=1
    cosmo=0
    molweight=1.0
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
    dxbound=[]
    dybound=[]
    dzbound=[]
    totvol=[]
    rcylmin2= 0
    rmin2= 0
    rcylmax2= 10**22
    rmax2= 10**22
    shape=2
    rhozero = 1.0
    massdisk= 1E-6
    massrat = 1E-3
    rdisk = 10.0
    q = 0.5
    hr = 0.1
    rorb = 1.0
    onlyoneBH = 0
    
    def __init__(self):
        pass
    def create(self,nx,distri=0,vm=0,entry=' ',hr=0.1,massrat=1E-3,onlyoneBH=0,q=0.5,rorb=1.0,rdisk=10.0,rinner=0.5):
        
         gam1=self.gamma-1
         self.onlyoneBH=onlyoneBH
         self.dICdensRsmooth = hr
         self.dICdensR = rdisk;
         self.hr = hr;
         self.massrat = massrat;
         self.q = q;
         self.rorb = rorb;
         self.rdisk = rdisk;

         rinnertemp=rinner
         p=0.5
         if(p < 2.0):
            integral = (rdisk**(2 - p) - rinnertemp**(2 - p)) / (2 - p)
            self.dICdensinner = self.massdisk / (2 * np.pi * integral)
         elif(p == 2.0):
            log_term = np.log(rdisk) - np.log(rinnertemp)
            self.dICdensinner = self.massdisk / (2 * np.pi * log_term)
         else:
            integral = (rinnertemp**(2 - p) - rdisk**(2 - p)) / (p - 2)
            self.dICdensinner = self.massdisk / (2 * np.pi * integral)
         #self.masszero=1.0
         #cs0 is same as scaleheight h/r and = 1/mach vorb=r**(-q)
         #P=cs0*density*r**(-q)
         #just cube, icgenerator will do rest
         dz=1.0
         dy=self.rdisk
         dx=self.rdisk
         distribute.setbound(self,-dx,dx,-dy,dy,-dz,dz)
         deltax = self.dxbound/nx
         if distri==0:
            distribute.setcloseddist(self,-dx,dx,-dy,dy,-dz,dz,deltax)
         elif distri==1:
            distribute.setrandomdist(self,-dx,dx,-dy,dy,-dz,dz,deltax)
         else:
            tgdatain,tddata,tsdata,data_header,time=tip.readtipsy(entry);
            #mapp = np.any([tgdatain[:,7] > self.dICdensouter,tgdatain[:,1]**2 + tgdatain[:,2]**2 > self.dICdensRsmooth**2], axis=0)
            rcyl=np.sqrt(tgdatain[:,1]**2 + tgdatain[:,2]**2)
            hin = tgdatain[:,9];
            fMass = tgdatain[:,0];
            Hdisk=self.dICdensRsmooth * rcyl**(1.5-self.q);   # H = cs/Omega_K, cs = hr R^-q
            rhocentral = self.dICdensinner*rcyl**(-p) /(Hdisk*np.sqrt(2*np.pi));   # Sigma ~ R^-p
            hcentral = (fMass / rhocentral)**(1.0/3.0)
            ttres = Hdisk / hcentral
            hedge = (fMass / rhocentral*np.exp(ttres**2/2.0))**(1.0/3.0)
            z_abs = np.abs(tgdatain[:,3])
            mapp = (tgdatain[:,7] > self.dICdensouter) & \
                           (tgdatain[:,1]**2 + tgdatain[:,2]**2 > rinner**2) & \
                           (z_abs < ttres*Hdisk+hedge)
            #    (tgdatain[:,1]**2 + tgdatain[:,2]**2 < rdisk**2)
            tgdata=tgdatain[mapp,:]
            #tgdata=tgdatain
            self.x=tgdata[:,1]
            self.y=tgdata[:,2]
            self.z=tgdata[:,3]
            self.rho=tgdata[:,7]
            self.periodic=0
         self.npart=len(self.x)
         self.ngas=self.npart
         totmass = self.massdisk
         print('npart = ',self.npart,' total disk mass = ',totmass)  
         self.mass = [totmass/self.npart]*self.npart
         print(' particle mass = ',self.mass[1])
         self.dxbound = self.rdisk*4
         self.dybound =	self.rdisk*4
         self.dzbound =	self.rdisk*2
         # Pre-glass: the relaxer's target and force tables (see glass_tables). Final IC: the same force
         # table again (the glass keeps the pre-IC particle mass, which sets ttres), for v_phi.
         if self.preglass:
             self.glass_density_table, self.glass_force_table = self.glass_tables(self.mass[0], p, rinner)
             ftab = self.glass_force_table
         else:
             ftab = self.glass_tables(self.mass[0] if distri in (0, 1) else fMass[0], p, rinner)[1]
         self.vphi = self.vphi_from_table(ftab)

         for i in range(self.npart):
             rad = np.sqrt(self.x[i]**2 + self.y[i]**2+self.z[i]**2)
             vxi,vyi,vzi = self.getveli(i)
             self.vx.append(vxi)
             self.vy.append(vyi)
             self.vz.append(vzi)
             self.u.append(self.hr*rad**(-self.q)/(gam1))
             self.Bx.append(0.)
             self.By.append(0.)
             self.Bz.append(0.)
             #self.h.append(hr*0.006) ## 1/100 of secondary BH
             if self.onlyoneBH==0:
                self.h.append(hr*0.6)
             else:
                self.h.append(0.5)

    def glass_tables(self, fmass, p, rinner):
        """Target and force tables for glassgen on a uniform (R, |z|) grid over the glass box.

        rho_t is the C relaxer's dICdensprofile 7 (gasoline.gdicgen pkd.c) with its Gaussian replaced by
        the exact isothermal profile in the point-mass potential, rho_mid exp[(1/r - 1/R)/cs^2] (GM = 1,
        the Gaussian for z << R): that disk up to
        z_t = ttres H, a power-law tail from there down to the edge density at zedge, the Fermi taper
        at rdisk and the dICdensouter floor. Unlike profile 7, the table also tapers at rinner (Fermi,
        width hin = 2 x the midplane spacing there, as hedge at rdisk): without it the disk continues
        to the axis as an unresolvable sheet (rho_mid ~ R^-(p+1.5-q), H = hr R^(1.5-q)) that the crop removes anyway. ttres = H / h_mid depends on the pre-IC particle mass
        fmass, so both tables belong to one pre-IC (datafiles/<prefile>.density.npz, .force.npz).
        The force table (glassgen --force-table, relax.config GLASS_FORCEMATCH) is locally
        isothermal like gasoline's ISOR EOS: P_t = cs^2 rho_t, cs^2 = hr^2 (R + 0.001)^-2q, closure 1,
        and g = grad P_t/rho_t = cs^2 grad ln rho_t + grad cs^2. The relax has no rotation, so g_R
        is only the radial pressure gradient (the setup's rotation balances the rest of gravity);
        g_z is the point-mass vertical gravity -z/r^3 below z_t (H = cs/Omega_K = hr R^(1.5-q) sets z_t;
        Sigma ~ R^-p). Profile 7 itself is the thin-disk Gaussian (g_z = -z/R^3) and has no q (H = hr R).
        Grid: dz = H(rinner)/5, dR = 2 dz; the axis node takes the value at R = dR/2."""
        hr, Rd, floor = self.hr, self.rdisk, self.dICdensouter
        s = 1.5 - self.q          # H = hr R^s
        dz = hr*rinner**s/5.0
        dR = 2.0*dz
        Rg = np.arange(0.0, np.hypot(self.dxbound, self.dybound)/2.0 + dR, dR)
        zg = np.arange(0.0, self.dzbound/2.0 + dz, dz)
        R = np.maximum(Rg, 0.5*dR)[:, None]
        z = zg[None, :]
        C1 = self.dICdensinner/(hr*np.sqrt(2*np.pi))
        C2 = (self.dICdensinner*hr*hr/(fmass*np.sqrt(2*np.pi)))**(1./3.)
        H = hr*R**s
        mid = C1*R**(-p-s)
        ttres = C2*R**((2.0*s-p)/3.0)
        zt = ttres*H
        cs2 = hr*hr/(R + 0.001)**(2.0*self.q)
        # isothermal hydrostatic profile in the point-mass potential: cs2(R) dln rho/dz = -z/r^3, i.e.
        # rho/rho_mid = exp[(1/r - 1/R)/cs2], with 1/r - 1/R = -z^2/(r R (r + R)) (no cancellation)
        def vert(R_, z_, c2):
            r_ = np.sqrt(R_*R_ + z_*z_)
            return np.exp(-z_*z_/(r_*R_*(r_ + R_))/c2)
        rhoedgecentre = C1*Rd**(-p-s)
        zedge = hr*C2*Rd**((5.0*s-p)/3.0)
        rhoedge = max(rhoedgecentre*vert(Rd, zedge, hr*hr/(Rd + 0.001)**(2.0*self.q)), floor)
        hedge = 2.0*(fmass/rhoedgecentre)**(1./3.)
        hin = 2.0*(fmass/(C1*rinner**(-p-s)))**(1./3.)
        ttresexp = vert(R, zt, cs2)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            poly1 = np.maximum(-(np.log(rhoedge) - np.log(mid*ttresexp))/(np.log(zedge) - np.log(zt)), 0.0)
            rho = mid*np.where(z > zt, ttresexp*(z/zt)**(-poly1), vert(R, z, cs2))
            rho = rho/(1.0 + np.exp(-(Rd - R)/hedge))/(1.0 + np.exp(-(R - rinner)/hin))
        rho = np.where(np.isfinite(rho) & (rho > floor), rho, floor)
        density_table = ((Rg, zg), rho, ('r_cyl', 'z'))
        dlnR, dlnz = np.gradient(np.log(rho), dR, dz)
        dlnz[:, 0] = 0.0
        force_table = dict(ax0=Rg, ax1=zg, mapping=np.array(['r_cyl', 'z']),
                                      P=cs2*rho, rho=rho, gR=cs2*dlnR - 2.0*self.q*hr*hr/(R + 0.001)**(2.0*self.q + 1.0),
                                      gz=cs2*dlnz, closure=1.0)
        print('glass tables: (R, |z|) grid %d x %d, dR %g dz %g, hedge %g, hin %g, zedge %g'
              % (len(Rg), len(zg), dR, dz, hedge, hin, zedge))
        return density_table, force_table

    def vphi_from_table(self, ftab):
        """v_phi per particle from radial balance with the force table the glass is relaxed to:
        v_phi^2 = R (GM R/r^3) + R g_R,table(R, |z|), GM = 1 (the primary at the origin), clipped at 0.
        g_R,table = (1/rho_t) dP_t/dR; bilinear in (R, |z|) with edge clamping, as glassgen's TableDensity."""
        x, y, z = (np.asarray(a, dtype=np.float64) for a in (self.x, self.y, self.z))
        R = np.hypot(x, y)
        r = np.sqrt(R*R + z*z)
        ax0, ax1, g = ftab['ax0'], ftab['ax1'], ftab['gR']
        t0 = np.clip((R - ax0[0])/(ax0[1] - ax0[0]), 0.0, len(ax0) - 1.0)
        t1 = np.clip((np.abs(z) - ax1[0])/(ax1[1] - ax1[0]), 0.0, len(ax1) - 1.0)
        j0 = np.minimum(t0.astype(int), len(ax0) - 2)
        j1 = np.minimum(t1.astype(int), len(ax1) - 2)
        f0, f1 = t0 - j0, t1 - j1
        gR = (g[j0, j1]*(1 - f0)*(1 - f1) + g[j0, j1 + 1]*(1 - f0)*f1
              + g[j0 + 1, j1]*f0*(1 - f1) + g[j0 + 1, j1 + 1]*f0*f1)
        with np.errstate(divide='ignore', invalid='ignore'):
            v2 = np.where(r > 0.0, R*R/r**3 + R*gR, 0.0)
        return np.sqrt(np.maximum(v2, 0.0))

    def create_star(self,tgdata):
        # central black hole(s): single sink, or a binary with CM-corrected primary
        massprim = 1.0
        softprim = 0.5
        sinkprim = -1.0
        if self.onlyoneBH == 1:
            tsdata = np.array([[massprim,0.0,0.0,0.0,0.0,0.0,0.0,0.0,sinkprim,softprim,0.0]])  # one row
            return tsdata, 1
        else:
            xsec = -self.rorb
            vysec = -self.rorb**(-0.5)   # Keplerian about the primary (GM = 1)
            softsec = self.hr*0.6
            masssec = massprim*self.massrat
            tssec = np.array([masssec,xsec,0.0,0.0,0.0,vysec,0.0,0.0,0.0,softsec,0.0])
            # Set the centre-of-mass velocity by adjusting the primary
            totmass = massprim+masssec+np.sum(tgdata[:,0])
            vcmx = (0.0+np.sum(tgdata[:,0]*tgdata[:,4]))/totmass
            vcmy = (masssec*vysec+np.sum(tgdata[:,0]*tgdata[:,5]))/totmass
            vcmz = (0.0+np.sum(tgdata[:,0]*tgdata[:,6]))/totmass
            tsdata = np.array([tssec, np.array([massprim,0.0,0.0,0.0,-vcmx,-vcmy,-vcmz,0.0,sinkprim,softprim,0.0])])
            return tsdata, 2

    def getrhoi(self,i):
        return self.rhozero
    
    def getveli(self,i):
        vazi=self.vphi[i]   # radial balance with the glass's force table (vphi_from_table)

        vx,vy,vz = self.transform_coordinate(0.0,vazi,0.0,self.x[i],self.y[i],self.z[i],"cyl","cart")
        return vx,vy,vz
    
    def transform_coordinate(self,v1,v2,v3,x,y,z,fromcord,tocord):
        
        if(fromcord=="cyl"):
            r=np.sqrt(x**2+y**2)
            if(tocord=="cart"):
                ux=v1*x/r-v2*y/r
                uy=v1*y/r+v2*x/r 
                uz=v3
                
        return ux,uy,uz
