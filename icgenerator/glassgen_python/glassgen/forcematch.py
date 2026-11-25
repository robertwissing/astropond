"""Force-matched relax mode (--force-table): the glass reproduces the target's pressure acceleration.

The pseudo-pressure relax holds pP = (rho/rho0)^rhopow and aims at rho = rho0; the pressure force a
code then sees is whatever its operator makes of that density. In this mode the relax works on the
real pressure P_i = A(x_i) rho_i^Gamma, with the target's A = P_t / rho_t^Gamma, and the force is

    f_i = [ -(1/rho) (grad P)_SPH + g(x_i) ] / P_t(x_i).

The fixed point has (1/rho) (grad P)_SPH = g at every particle. With g = grad P_t / rho_t the target
itself is the continuum solution, so the glass ends with the target's pressure acceleration in the
operator's discrete form (its density close to, not exactly, rho_t). For a target in hydrostatic
equilibrium (polytrope) that g is its gravity and the glass is hydrostatic; for one that is not
(Evrard collapse) g is only the acceleration that holds the target, not a physical force.
Gamma is the closure the IC uses for u: gamma (entropy, u from A and the glass density) or 1
(temperature, u = u_t(x)). Dividing by P_t only weights particles; with uniform P_t and g = 0 the
force is the pseudo-pressure force with rhopow = Gamma.

Table (.npz), one of
  radial:        r (radius about the box origin, ascending), P (P_t), g (radial, < 0 inward;
                 normally grad P_t / rho_t), rho (rho_t on the same r), closure (Gamma);
  axisymmetric:  ax0 (cylindrical R), ax1 (|z|), mapping ('r_cyl', 'z'), and on that grid P, rho,
                 gR (along Rhat) and gz (along +z for z > 0; g_z is odd in z), closure. Uniform axes,
                 bilinear like density.TableDensity (which evaluates each field).
"""
import numpy as np


class ForceTable:
    def __init__(self, path):
        d = np.load(path)
        P = np.asarray(d['P'], dtype=np.float64)
        rho = np.asarray(d['rho'], dtype=np.float64)
        self.gamma = float(d['closure'])
        self.Pref = float(P.max())
        self.axisym = 'ax1' in d.files
        if not self.axisym:
            self.r = np.asarray(d['r'], dtype=np.float64)
            self.P = P
            self.g = np.asarray(d['g'], dtype=np.float64)
            self.lnA = np.log(P) - self.gamma * np.log(rho)
            return
        from .density import TableDensity
        mapping = tuple(str(s) for s in d['mapping'])
        if mapping != ('r_cyl', 'z'):
            raise ValueError(f"{path}: axisymmetric force table needs mapping ('r_cyl', 'z'), "
                             f'got {mapping}')
        axes = (d['ax0'], d['ax1'])
        field = lambda v: TableDensity(axes, v, mapping)
        self.lnA = field(np.log(P) - self.gamma * np.log(rho))
        self.lnP = field(np.log(P))
        self.gR = field(d['gR'])
        self.gz = field(d['gz'])

    def pressure(self, pos, rho, pP):
        """pP <- A(x) rho^Gamma / Pref (in place)."""
        if self.axisym:
            lnA = self.lnA.rho0(pos)
        else:
            lnA = np.interp(np.sqrt((pos * pos).sum(1)), self.r, self.lnA)
        pP[:] = np.exp(lnA + self.gamma * np.log(rho)) / self.Pref

    def add_g(self, pos, icvel):
        """icvel (the force of pP) <- (icvel + g / Pref) * Pref / P_t(x) (in place)."""
        if self.axisym:
            R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            gR = self.gR.rho0(pos) / (self.Pref * np.where(R > 0.0, R, 1.0))
            icvel[:, 0] += gR * pos[:, 0]
            icvel[:, 1] += gR * pos[:, 1]
            icvel[:, 2] += np.sign(pos[:, 2]) * self.gz.rho0(pos) / self.Pref
            icvel *= (self.Pref / np.exp(self.lnP.rho0(pos)))[:, None]
            return
        r = np.sqrt((pos * pos).sum(1))
        g = np.interp(r, self.r, self.g) / self.Pref
        rs = np.where(r > 0.0, r, 1.0)
        icvel += (g / rs)[:, None] * pos
        icvel *= (self.Pref / np.interp(r, self.r, self.P))[:, None]
