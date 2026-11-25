#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated dissipation-operator tests (`disstest`).

These were grown inside `cosmowave` as cases 8--11 and have now outgrown it:
they share none of Berlok's mode machinery, every one of them is a frozen or
single-step measurement rather than a wave, and each was bending
`case_params`, `scan_input` and the case-ID guard further out of shape.

Every case comes in a STATIC (`bComove=0`) and an EXPANDING (`bComove=1`) arm.
That pairing is the point of the suite: the static arm measures the discrete
operator with the scale factor frozen out of the problem entirely, and the
expanding arm is then compared against it, so what is being tested is the
scale-factor bookkeeping alone rather than the operator and its `a`-factors
tangled together.  It also makes a tautological configuration obvious -- if a
channel's static and expanding arms agree trivially, that channel carries no
`a`-dependence to test.

Velocity convention, which differs between the arms and is easy to get wrong:
Gasoline's `vFac` is `1/a^2` when `bComove=1` and `1` when `bComove=0`, and
tipsy stores `xdot`.  The physical peculiar velocity is therefore `a*xdot`
expanding and `xdot` static.  Both arms here write the SAME numbers, so they
describe the same physical state at `a=1` and the static arm is the natural
`a=1` reference.

Cases
    0 null    uniform rho, u, B with v=0.  Nothing evolves but the background,
              so every dissipative channel must stay at roundoff (each is built
              on a pair DIFFERENCE) while the homogeneous laws hold.
    1 uwave   u = u0(1 + A cos kx) at uniform rho.  Fixed-coefficient parabolic
              conduction; the mode decays and the decay's SHAPE carries the
              coefficient's a-dependence.
    2 bhelix  B = (0, A cos kx, A sin kx).  A Beltrami field: curl B = -k B so
              J x B vanishes identically and |B| is uniform, hence no Lorentz
              force at all and div B = 0 exactly.  Fixed-coefficient resistivity.
    3 ladder  Uniform rho and u plus a velocity field, so the FLOW-DRIVEN
              channels switch on: AV, the default resistivity and p->diff are
              all identically zero at v=0 and are unreachable by cases 0-2.
              variant 0 longitudinal (compressive), 1 transverse (shear).
"""

import numpy as np

import IC_distribute as distribute
import IC_fixdens as fixdens
import IC_smoothlength as smth
import readtipsy as tip
from IC_writeascii import calculatecosmoMsol

CASES = {0: 'null', 1: 'uwave', 2: 'bhelix', 3: 'ladder'}


def eds_time(a, H0):
    """Einstein-de Sitter cosmic time."""
    return 2.*np.asarray(a, dtype=float)**1.5/(3.*H0)


class setup_disstest(object):
    dICdensRsmooth = 0.1
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0
    dICdensinner = 1.0
    dICdensouter = 1.0
    rhoit = 0
    ns = 64
    freqout = 1
    inflow = 0
    npart = 0
    ngas = 0
    ndark = 0
    nstar = 0
    ndim = 3
    time = 0
    grav = 0
    H0 = 2.894405
    omega0 = 1.0
    omegaLambda = 0.0
    cosmo = 1
    molweight = 1.0
    gamma = 5./3.
    periodic = 1
    deltastep = 0.0005
    nsteps = 100
    dmsolunit = 1.0
    dkpcunit = 1000.0
    adi = 1
    shape = 2
    dtheta = 0.6
    dtheta2 = 0.8
    # IC_distribute.checkboundary reads these radial cuts on every particle;
    # the values disable them, since this suite is always a full periodic box.
    drho = 0.
    _rho_z_grid = None
    glass_density_table = None
    rmin2 = 0.
    rmax2 = 1.e22
    rcylmin2 = 0.
    rcylmax2 = 1.e22

    def __init__(self):
        for name in ('mass x y z vx vy vz rho u h P Bx By Bz '
                     'dxbound dybound dzbound totvol').split():
            setattr(self, name, [])

    def create(self, nx, distri=0, vm=0, entry='0',
               case=0, cosmo=1, amp=1.e-1, bfac=1., variant=0, a_init=0.,
               vamp=0., strat=0.):
        case = int(case)
        cosmo = int(cosmo)
        variant = int(variant)
        amp = float(amp)
        bfac = float(bfac)
        a_init = float(a_init)
        strat = float(strat)
        # The velocity amplitude is separable from the field-perturbation
        # amplitude so an amplitude scan can move ONE of them at a time.
        # The shock and turbulent coefficients scale with the velocity,
        # while the thing they act on (grad u, delta B) scales with `amp`;
        # driving both from one knob would confound the two.
        vamp = float(vamp) if float(vamp) > 0. else amp

        # One guard, at the single choke point.
        if case not in CASES:
            raise ValueError("disstest case must be one of %s"
                             % ", ".join("%d (%s)" % (k, v)
                                         for k, v in sorted(CASES.items())))
        if cosmo not in (0, 1):
            raise ValueError("disstest cosmo must be 0 (static, bComove=0) "
                             "or 1 (expanding, bComove=1)")
        if amp <= 0.:
            raise ValueError("amp must be positive")
        if case == 2 and bfac == 0.:
            raise ValueError("disstest case 2 (bhelix) is a magnetic mode: "
                             "bfac sets the helical amplitude and must be "
                             "nonzero")
        if variant not in (0, 1):
            raise ValueError("variant must be 0 or 1")
        if variant != 0 and case != 3:
            raise ValueError("variant is only used by case 3 (ladder)")
        if a_init != 0. and not 0. < a_init <= 1.:
            raise ValueError("a_init must lie in (0, 1]")
        if a_init != 0. and cosmo == 0:
            raise ValueError("a_init is meaningless for the static arm "
                             "(bComove=0 has no scale factor)")
        if strat != 0. and case != 1:
            raise ValueError("strat stratifies the background density for the "
                             "case 1 (uwave) conduction operator; the other "
                             "cases assume a uniform background")
        if not -1. < strat < 1.:
            raise ValueError("strat must lie in (-1, 1): |strat| >= 1 would "
                             "drive the density non-positive")

        self.cosmo = cosmo
        # IC_createsetup overrides dmsolunit with the critical-density
        # normalization for cosmo=1 only.  Left alone, the static arm would end
        # up in a DIFFERENT unit system and its code-unit rates would not be
        # comparable with the expanding arm's -- which would defeat the whole
        # point of running the pair.  Adopt the same units in both.
        self.dmsolunit = calculatecosmoMsol(1.0, self.dkpcunit)
        self.gamma = 5./3.
        self.adi = 1
        dx = 0.5
        k = 2.*np.pi/(2.*dx)
        deltax = (2.*dx)/nx
        self.wk, self.xdot0 = k, -dx

        # The static arm has no scale factor; the expanding arm starts at
        # a_init (default 1/128, giving a 128x lever arm on every a-factor).
        a = 1.0 if cosmo == 0 else (a_init if a_init != 0. else 1./128.)
        self.a = a
        self.time = a if cosmo == 1 else 0.0

        # Amplitudes are deliberately a-INDEPENDENT.  For the ladder this makes
        # the comoving template written at one a_init byte-identical to the one
        # written at another, which is what lets a rung-to-rung comparison
        # isolate the scale factor.  For the other cases it simply means the
        # static and expanding arms start from the same physical state at a=1.
        self.rhozero = 1.0
        v_scale = (self.H0/k)*np.pi          # the family's velocity unit
        cs = v_scale
        uuzero = cs*cs/(self.gamma*(self.gamma - 1.))
        bzero = bfac*v_scale*np.sqrt(self.rhozero)

        if cosmo == 1:
            self.deltastep = (eds_time(1., self.H0)
                              - eds_time(a, self.H0))/self.nsteps
        else:
            # A static run needs its own duration; one sound-crossing of the
            # box is a natural unit and keeps the two arms comparable.
            self.deltastep = (2.*dx/cs)/self.nsteps

        # The stratification knob.  `drho` is the amplitude of a COSINE in x,
        # which integrates to zero over the box, so the total mass -- hence the
        # per-particle mass below -- is untouched and `strat` is the only thing
        # differing between a stratified run and its uniform twin.
        #
        # IN PHASE with the thermal mode, deliberately.  Pair averaging is
        # EXACT for anything linear in x, so the leading residual of Brookshaw's
        # 0.5*(Igr_p + Igr_q) goes as grad^2 Igr ~ grad^2 rho.  In phase, that
        # error and the signal grad^2 u are both ~ cos(k x'), so the FRACTIONAL
        # error is spatially constant and measurable everywhere.  A sine
        # profile would put them 90 degrees apart, making the fractional error
        # ~ tan(k x') -- divergent, and zero exactly where the signal is.
        self.drho = self.rhozero*strat
        if self.drho != 0.:
            xgrid = np.linspace(-dx, dx, 20001)
            self._rho_z_grid = (xgrid, self.getrho(xgrid))
            # IC_createsetup writes this as a self-describing signed-x table;
            # runtest feeds it to glassgen, so a stratified GLASS is relaxed in
            # the analytic profile rather than being stretched afterwards.
            self.glass_density_table = (
                xgrid.copy(), self._rho_z_grid[1].copy(), 'signed_x')

        distribute.setbound(self, -dx, dx, -dx, dx, -dx, dx)
        self.x0 = (-dx, -dx, -dx)
        if int(distri) == 0:
            distribute.setcubicdist(self, -dx, dx, -dx, dx, -dx, dx, deltax)
        elif int(distri) == 1:
            distribute.setrandomdist(self, -dx, dx, -dx, dx, -dx, dx, deltax)
        else:
            tgdata, _td, _ts, header, _t = tip.readtipsy(entry)
            expected = int(round(float(nx)))**3
            if int(header[2]) != expected:
                raise ValueError(
                    "disstest glass has %d gas particles, expected nx^3=%d"
                    % (int(header[2]), expected))
            self.x = np.asarray(tgdata[:, 1], dtype=float).copy()
            self.y = np.asarray(tgdata[:, 2], dtype=float).copy()
            self.z = np.asarray(tgdata[:, 3], dtype=float).copy()
            self.rho = np.asarray(tgdata[:, 7], dtype=float).copy()

        if self.drho != 0. and int(distri) != 2 and not self.preglass:
            # Lattice/random only: stretch-map into the profile.  A glass
            # arrives already relaxed in it (glass_density_table above), so
            # stretching one here would apply the profile twice -- and the
            # pre-glass seed is relaxed against that same table next, so
            # stretching it first is work the relaxation overwrites.
            fixdens.set_density_profile(self, -dx, dx, 1)

        self.npart = len(self.x)
        self.ngas = self.npart
        total = self.rhozero*self.dxbound*self.dybound*self.dzbound
        self.mass = [total/self.npart]*self.npart
        self.h = smth.getsmooth(self, 200)
        self.rho = ([self.getrhoi(i) for i in range(self.npart)]
                    if self.drho != 0. else [self.rhozero]*self.npart)

        for i in range(self.npart):
            phase = self.wk*(self.x[i] - self.xdot0)
            cosx, sinx = np.cos(phase), np.sin(phase)

            if case == 3:
                # The flow that switches the coefficients on.
                if variant == 0:
                    self.vx.append(vamp*v_scale*sinx); self.vy.append(0.)
                else:
                    self.vx.append(0.); self.vy.append(vamp*v_scale*cosx)
                self.vz.append(0.)
            else:
                self.vx.append(0.); self.vy.append(0.); self.vz.append(0.)

            if case == 1:
                self.u.append(uuzero*(1. + amp*cosx))
            elif case == 3:
                # a gradient for the thermal channel to act on
                self.u.append(uuzero*(1. + amp*cosx))
            else:
                self.u.append(uuzero)

            if case == 2:
                # Beltrami: curl B = -k B, |B| uniform, Bx = 0 so div B = 0.
                self.Bx.append(0.)
                self.By.append(bzero*cosx)
                self.Bz.append(bzero*sinx)
            elif case == 3:
                # uniform background plus a perturbation for the resistive
                # channels to diffuse
                self.Bx.append(bzero)
                self.By.append(bzero*amp*sinx)
                self.Bz.append(0.)
            else:
                self.Bx.append(bzero); self.By.append(0.); self.Bz.append(0.)

    def getrho(self, x):
        """Background density.  Uniform unless `strat` stratifies it, in which
        case a sine in x, which both the lattice stretch map and the glass
        relaxation read through this one function."""
        x = np.asarray(x, dtype=float)
        if self.drho == 0.:
            return np.full_like(x, self.rhozero)
        return self.rhozero + self.drho*np.cos(self.wk*(x - self.xdot0))

    def getrhoi(self, i):
        """Per-particle density, called by IC_distribute while it lays out
        the particles (so it must work before `create` finishes)."""
        if self.drho == 0.:
            return self.rhozero
        return float(self.rhozero
                     + self.drho*np.cos(self.wk*(self.x[i] - self.xdot0)))
