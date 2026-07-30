#!/usr/bin/env python3
"""Apply a magnetic field configuration to a tipsy IC, optionally via A.

The general form of the old setBField.py: instead of nine hard-coded menu
entries, a field is specified as four ORTHOGONAL choices

    profile   x  strength   x  direction   x  kind

    profile   the dimensionless spatial shape s(r) in [0,1]-ish
              uniform | rho | R | z | sin | exp | turbulent
    strength  what sets the amplitude, and in what measure
              gauss | code | beta-th | beta-kin
    direction the unit vector field the amplitude rides on
              x | y | z | azi (phi-hat) | Rhat (cylindrical) | rhat (spherical)
    kind      whether the constructed vector IS the field, or its POTENTIAL
              field | potential

so the constructed vector is

    V_i = sign * amplitude_i * profile_i * direction_i

with ONE exception: `turbulent` is a VECTOR profile - a random solenoidal
field with a power-law spectrum, drawn in Fourier space - so it supplies the
orientation itself and --direction does not apply. Its --spectral-index
always describes B (E_B(k) ~ k^a); with --kind potential the code draws the A
whose curl has that spectrum. Being a curl-friendly random field it is the
profile where --kind potential earns the most: the curl of a random A is
solenoidal by construction, whereas a random B prescribed directly is not.

and every option of the original script is a combination, e.g. its

    option 1 (uniform field one direction)   profile=uniform direction=x
                                             strength=code kind=field
    option 4 (beta-based quadrupole)         profile=z direction=azi
                                             strength=beta-th kind=potential
    option 8 (density-varying, one direction) profile=rho direction=azi
                                             strength=code kind=field

KIND=POTENTIAL is the substantive addition. The old script wrote A into the
BField aux and left the curl to a vector-potential GASOLINE build; here the
curl is taken directly on the particles,

    B_i = (curl A)_i = sum_j m_j D_ij [gradWbar_ij x (A_j - A_i)]

in a flavour chosen from the glassgen.flavours registry (gdforce, rho2,
gdforce-own, rho2-own) - the SAME discrete operator family that
mhd_divBclean's divergence cleaning uses, so if cleaning follows it removes
the divergence in exactly the measure the curl produced it (and, with the
gdforce default, the measure the gasolinemhd consumer itself evolves).
Potential mode writes both results: B to BField and A to AField.

TWO THINGS ABOUT THE CURL WORTH KNOWING (both are properties of the meshfree
estimator, not of this script; see glassgen/flavours.py):

  * div(curl) is not identically zero on particles, so B = curl A carries a
    small divergence residual - far below a directly prescribed field, and
    removable with --clean.
  * the curl's AMPLITUDE is systematically low by the estimator's
    normalisation shortfall (~2% at nsmooth=64, ~0.03% at 128, even on a
    perfect lattice), while its DIRECTION is essentially exact.

The second point, plus the fact that A carries units of B x length, is why
--normalise rescales the final field to the strength actually requested (in
potential mode by default; see its help for the two meanings of "requested",
which differ once a non-uniform profile is in play). --kind field is written
exactly as specified, as before.

Usage (the input snapshot is required; any unspecified option is prompted
for):

    python IC_setBField.py IN.00000 -o OUT.00000 \
        --profile rho --rho-power 0.6667 --direction azi \
        --strength beta-th --value 50 --kind potential \
        --flavour gdforce --clean --target-mean 1e-3

    python IC_setBField.py IN.00000           # ask for unspecified options

Units come from a sibling .param or .log (dKpcUnit/dMsolUnit/dConstGamma/
dMeanMolWeight) when one is present, with --kpc-unit/--msol-unit/--gamma/--mu
overrides. If neither file exists, conventional defaults are offered. Output:
OUT plus the aux files, with every B representation the input carried replaced
consistently.
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTSUITE = os.path.normpath(os.path.join(HERE, '..'))
GLASSGEN = os.path.join(TESTSUITE, 'icgenerator', 'glassgen_python')
sys.path.insert(0, HERE)
sys.path.insert(0, GLASSGEN)

import readtipsy as tip                                        # noqa: E402
from glassgen.flavours import (FLAVOUR_SPECS, curl,             # noqa: E402
                               divergence, divergence_error,
                               operator_state)

# tipsy gas columns
C_MASS, C_POS, C_VEL, C_RHO, C_TEMP, C_HSM, C_MET, C_PHI = (
    0, slice(1, 4), slice(4, 7), 7, 8, 9, 10, 11)

PROFILES = ('uniform', 'rho', 'R', 'z', 'sin', 'exp', 'turbulent')
# profiles that produce a VECTOR shape and therefore consume --direction
# themselves (the amplitude x profile x direction factorisation does not
# survive a field whose orientation varies with position)
VECTOR_PROFILES = ('turbulent',)
STRENGTHS = ('gauss', 'code', 'beta-th', 'beta-kin')
DIRECTIONS = ('x', 'y', 'z', 'azi', 'Rhat', 'rhat')
KINDS = ('field', 'potential')
UNIT_FIELDS = (
    ('kpc_unit', '--kpc-unit', 'dKpcUnit (length unit in kpc)'),
    ('msol_unit', '--msol-unit', 'dMsolUnit (mass unit in Msol)'),
    ('gamma', '--gamma', 'dConstGamma (adiabatic index)'),
    ('mu', '--mu', 'dMeanMolWeight (mean molecular weight)'),
)
DEFAULT_UNITS = {
    'kpc_unit': 1.0,
    'msol_unit': 1.0,
    'gamma': 1.66667,
    'mu': 0.59259,
}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

MSOLG = 1.99e33
KBOLTZ = 1.38e-16
MHYDR = 1.67e-24
KPCCM = 3.085678e21
GCGS = 6.67e-8
MU0CGS = 4.0 * np.pi


def unit_conversions(dKpcUnit, dMsolUnit, dConstGamma, dMeanMolWeight,
                     verbose=True):
    """Code-unit conversion factors, verbatim in form from setBField.py.

    Returns a dict with GausstoCode/CodetoGauss (B), vFac, duTFac (T -> u)
    and TtouFac, plus the density unit.
    """
    gtoMsol = 1.0 / MSOLG
    cmtoKpc = 1.0 / KPCCM
    dGasConst = (dKpcUnit * KPCCM * KBOLTZ
                 / (MHYDR * GCGS * dMsolUnit * MSOLG))
    dGmPerCcUnit = (dMsolUnit * MSOLG) / (dKpcUnit * KPCCM) ** 3
    dSecUnit = np.sqrt(1.0 / (dGmPerCcUnit * GCGS))
    duTFac = (dConstGamma - 1.0) * dMeanMolWeight / dGasConst
    GausstoCode = np.sqrt((gtoMsol / dMsolUnit)
                          / ((cmtoKpc / dKpcUnit) * MU0CGS)) * dSecUnit
    u = dict(GausstoCode=GausstoCode, CodetoGauss=1.0 / GausstoCode,
             vFac=dSecUnit, duTFac=duTFac, TtouFac=1.0 / duTFac,
             dSecUnit=dSecUnit, dGmPerCcUnit=dGmPerCcUnit,
             gamma=dConstGamma, mu=dMeanMolWeight)
    if verbose:
        print(f'units: 1 code B = {u["CodetoGauss"]:.6e} G, '
              f'1 G = {GausstoCode:.6e} code; dSecUnit {dSecUnit:.6e} s; '
              f'T->u {u["TtouFac"]:.6e}; gamma {dConstGamma}, mu '
              f'{dMeanMolWeight}', flush=True)
    return u


def find_param(infile):
    """Sibling .param for a snapshot (IN.00000 -> IN.param), else any in cwd."""
    stem = re.sub(r'\.\d+$', '', infile)
    cand = stem + '.param'
    if os.path.isfile(cand):
        return cand
    here = sorted(glob.glob(os.path.join(os.path.dirname(infile) or '.',
                                         '*.param')), key=os.path.getmtime)
    return here[-1] if here else None


def find_log(infile):
    """Sibling .log for a snapshot (IN.00000 -> IN.log), else any nearby."""
    stem = re.sub(r'\.\d+$', '', infile)
    for cand in (stem + '.log', infile + '.log'):
        if os.path.isfile(cand):
            return cand
    here = sorted(glob.glob(os.path.join(os.path.dirname(infile) or '.',
                                         '*.log')), key=os.path.getmtime)
    return here[-1] if here else None


def params_from_file(path):
    """(dKpcUnit, dMsolUnit, dConstGamma, dMeanMolWeight) present in a .param."""
    from glassgen.params import read_param
    p = read_param(path)
    got = {}
    for key, name in (('dKpcUnit', 'kpc_unit'), ('dMsolUnit', 'msol_unit'),
                      ('dConstGamma', 'gamma'),
                      ('dMeanMolWeight', 'mu')):
        if key in p:
            try:
                got[name] = float(p[key])
            except ValueError:
                pass
    return got, p


def params_from_log(path):
    """Unit parameters recorded as ``name: value`` in a gasoline .log."""
    with open(path) as fh:
        text = fh.read()
    got = {}
    for key, name in (('dKpcUnit', 'kpc_unit'), ('dMsolUnit', 'msol_unit'),
                      ('dConstGamma', 'gamma'),
                      ('dMeanMolWeight', 'mu')):
        match = re.search(
            rf'\b{re.escape(key)}\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?'
            rf'|\.\d+)(?:[eE][-+]?\d+)?)',
            text)
        if match:
            got[name] = float(match.group(1))
    return got


# ---------------------------------------------------------------------------
# interactive helpers
# ---------------------------------------------------------------------------

class Asker:
    """Prompt for anything the command line did not supply.

    Refuses to block when stdin is not a terminal: a batch run that forgot a
    flag gets a clear error naming the flag instead of hanging on input().
    """

    def __init__(self, interactive=None):
        self.interactive = (sys.stdin.isatty() if interactive is None
                            else interactive)

    def _demand(self, flag):
        raise SystemExit(f'{flag} not given and stdin is not a terminal '
                         f'(add {flag} or run interactively)')

    def choice(self, value, flag, prompt, choices, descriptions=None):
        if value is not None:
            if value not in choices:
                raise SystemExit(f'{flag}: {value!r} not in {list(choices)}')
            return value
        if not self.interactive:
            self._demand(flag)
        while True:
            print(f'\n{prompt}')
            for i, c in enumerate(choices, 1):
                description = descriptions.get(c) if descriptions else None
                extra = f'  - {description}' if description else ''
                print(f'  {i}: {c}{extra}')
            raw = input(f'{flag} [1-{len(choices)} or name]: ').strip()
            if raw in choices:
                return raw
            try:
                k = int(raw)
                if 1 <= k <= len(choices):
                    return choices[k - 1]
            except ValueError:
                pass
            print('invalid choice')

    def number(self, value, flag, prompt, cast=float):
        if value is not None:
            return value
        if not self.interactive:
            self._demand(flag)
        while True:
            try:
                return cast(input(f'{prompt}: ').strip())
            except ValueError:
                print('not a number')

    def yes(self, value, flag, prompt):
        if value is not None:
            return bool(value)
        if not self.interactive:
            self._demand(flag)
        return input(f'{prompt} [y/N]: ').strip().lower().startswith('y')


# ---------------------------------------------------------------------------
# geometry: profiles and directions
# ---------------------------------------------------------------------------

def cylindrical(pos):
    """(R, phi, z) with a safe phi at R = 0."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return R, phi, z


def direction_field(name, pos):
    """(N,3) unit vector field. Degenerate points (R=0 / r=0) get the zero
    vector rather than a random direction - an honest hole beats a fake."""
    n = len(pos)
    if name in ('x', 'y', 'z'):
        d = np.zeros((n, 3))
        d[:, 'xyz'.index(name)] = 1.0
        return d
    R, phi, _ = cylindrical(pos)
    if name == 'azi':
        d = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], axis=1)
        d[R == 0.0] = 0.0
        return d
    if name == 'Rhat':
        d = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
        d[R == 0.0] = 0.0
        return d
    if name == 'rhat':
        r = np.linalg.norm(pos, axis=1)
        d = np.zeros((n, 3))
        ok = r > 0.0
        d[ok] = pos[ok] / r[ok, None]
        return d
    raise SystemExit(f'unknown direction {name!r}')


def profile_field(name, pos, rho, args, box):
    """(N,) dimensionless shape in ~[0,1], plus a text description.

    Normalising every profile by its own reference keeps the strength basis
    meaningful: the amplitude you request is the amplitude at the profile's
    peak, whatever the profile.
    """
    if name == 'uniform':
        return np.ones(len(pos)), 'uniform'
    if name == 'rho':
        ref = (args.profile_ref if args.profile_ref is not None
               else float(rho.max()))
        p = float(args.rho_power)
        return (rho / ref) ** p, f'(rho/{ref:.6g})^{p:g}'
    if name == 'R':
        R, _, _ = cylindrical(pos)
        ref = (args.profile_ref if args.profile_ref is not None
               else float(R.max()))
        return R / ref, f'R/{ref:.6g}'
    if name == 'z':
        ref = (args.profile_ref if args.profile_ref is not None
               else float(np.abs(pos[:, 2]).max()))
        return pos[:, 2] / ref, f'z/{ref:.6g}  (odd in z -> quadrupole)'
    if name == 'sin':
        axis = 'xyz'.index(args.sin_axis)
        L = float(box[axis]) if box is not None else 1.0
        k = float(args.wavenumber)
        return (np.sin(2.0 * np.pi * k * pos[:, axis] / L),
                f'sin(2 pi {k:g} {args.sin_axis}/{L:g})  (no net flux)')
    if name == 'exp':
        rs = float(args.scale_radius)
        r = np.linalg.norm(pos, axis=1)
        return np.exp(-r / rs), f'exp(-r/{rs:g})'
    if name in VECTOR_PROFILES:
        raise SystemExit(f'{name!r} is a vector profile - call '
                         'vector_profile_field, not profile_field')
    raise SystemExit(f'unknown profile {name!r}')


# ---------------------------------------------------------------------------
# vector profiles: a shape whose DIRECTION varies with position
# ---------------------------------------------------------------------------

def turbulent_domain(pos, box, pad_frac):
    """(origin, L) of the cube the random field is generated periodic on.

    Periodic input: the simulation box itself, so the field wraps exactly the
    way the particles do. Open input: the particle bounding box grown by
    `pad_frac`, which buys two things - the field's own periodicity lands in
    empty space instead of correlating opposite edges of the object, and no
    particle sits exactly on the wrap where trilinear interpolation is least
    accurate. The field is still periodic on the padded cube; that is a
    property of the FFT, not a claim about the simulation.
    """
    if box is not None:
        return np.zeros(3), np.asarray(box, dtype=np.float64)
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-30)
    pad = float(pad_frac) * span
    return lo - pad, span + 2.0 * pad


def turbulent_field(pos, args, box):
    """(N,3) random solenoidal vector field with a power-law spectrum.

    A Gaussian random field is drawn in Fourier space with random phases and
    |V(k)|^2 a power law, projected transverse to k, and inverse-transformed
    onto a uniform grid, from which it is interpolated to the particles.

    THE SPECTRAL INDEX ALWAYS DESCRIBES B, never the vector actually built.
    With `--spectral-index a`, the shell-integrated magnetic power spectrum is
    E_B(k) ~ k^a (a = -5/3 Kolmogorov, -2 Burgers, +3/2 Batchelor). Since
    E(k) = 4 pi k^2 |V(k)|^2 /2 and B = curl A brings a factor k,

        kind=field      |B(k)|^2 ~ k^(a-2)
        kind=potential  |A(k)|^2 ~ k^(a-4)

    so the same requested index gives the same B either way and the caller
    does not have to convert. This is the one profile where --kind changes
    what is drawn rather than only what is done with it.

    The transverse projection V <- V - khat (khat.V) makes the field exactly
    solenoidal ON THE GRID. For kind=potential that is a gauge choice
    (Coulomb) which costs nothing and makes |B| = k|A| exact; for kind=field
    it is the div B = 0 the field is wanted for - though only up to the
    trilinear interpolation onto the particles and, thereafter, up to the SPH
    divergence operator's own view of it. `--clean` still applies.

    k = 0 is always zeroed: a mean A is pure gauge, and a mean B cannot be
    produced by any curl. So a turbulent field carries NO net flux by
    construction, and `--profile turbulent` cannot be used to make a
    uniform-plus-fluctuation field. Superpose two runs for that.
    """
    ng = int(args.turb_ngrid)
    if ng < 4 or ng % 2:
        raise SystemExit('--turb-ngrid must be an even number >= 4')
    origin, L = turbulent_domain(pos, box, args.turb_pad)
    alpha = float(args.spectral_index)
    # the drawn vector's own |V(k)|^2 exponent (see the docstring)
    beta = alpha - (4.0 if args.kind == 'potential' else 2.0)

    # --- draw in Fourier space -------------------------------------------
    # real-space white noise -> rfftn gives a spectrum that is Hermitian by
    # construction, so the inverse transform is exactly real without any
    # symmetry bookkeeping of our own
    rng = np.random.default_rng(args.seed)
    Vk = np.fft.rfftn(rng.standard_normal((3, ng, ng, ng)), axes=(1, 2, 3))

    kx = 2.0 * np.pi * np.fft.fftfreq(ng, d=L[0] / ng)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ng, d=L[1] / ng)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(ng, d=L[2] / ng)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    kmag = np.sqrt(k2)

    # the band is quoted in units of the fundamental mode of the cube; for an
    # anisotropic box the geometric mean is the only single length that makes
    # 'one mode across the box' mean the same thing on every axis
    kf = 2.0 * np.pi / float(np.cbrt(L[0] * L[1] * L[2]))
    # The default ceiling is the ISOTROPIC Nyquist: a mode of magnitude K
    # exists in every direction only while K stays under the smallest axis
    # Nyquist pi*ng/L_a, i.e. under pi*ng/max(L). On a cube that is exactly
    # ng/2 fundamental modes (the obvious answer); on an elongated domain it
    # is less, and taking ng/2 there would ask for modes the grid cannot
    # represent along the long axis - the band would be silently clipped in
    # one direction and the field would not be isotropic.
    nyq = np.pi * ng / float(np.max(L)) / kf
    nmax = nyq if args.turb_kmax is None else float(args.turb_kmax)
    kmin = float(args.turb_kmin) * kf
    kmax = nmax * kf
    if args.turb_kmax is not None and nmax > nyq * 1.0001:
        print(f'  WARNING: --turb-kmax {nmax:g} is above the isotropic '
              f'Nyquist {nyq:.3g} of a {ng}^3 grid on this domain; the band '
              'is clipped along the long axis and the field will not be '
              'isotropic. Raise --turb-ngrid instead.')
    if kmin > kmax:
        raise SystemExit(f'--turb-kmin {args.turb_kmin:g} exceeds --turb-kmax '
                         f'{nmax:g}')
    band = (kmag >= kmin) & (kmag <= kmax) & (kmag > 0.0)

    amp = np.zeros_like(kmag)
    amp[band] = kmag[band] ** (0.5 * beta)
    Vk *= amp

    # transverse projection (and k=0 killed with it)
    with np.errstate(invalid='ignore', divide='ignore'):
        kdotV = (kx * Vk[0] + ky * Vk[1] + kz * Vk[2]) / np.where(k2 > 0, k2, 1)
    Vk[0] -= kx * kdotV
    Vk[1] -= ky * kdotV
    Vk[2] -= kz * kdotV
    Vk[:, 0, 0, 0] = 0.0

    grid = np.fft.irfftn(Vk, s=(ng, ng, ng), axes=(1, 2, 3))

    # construction check: the drawn field's own spectrum, before it meets any
    # particle. The measured slope is E_V ~ k^(beta+2); report it as the B
    # slope it implies so it can be compared with what was asked for.
    absfit = grid_spectrum_slope(grid, L, kmin, kmax)
    resfit = grid_spectrum_slope(grid, L, kmin, kmax, amp=amp)
    if absfit is not None and resfit is not None:
        # the drawn field's own E ~ k^(beta+2); report it as the B slope it
        # implies, so it can be read straight against --spectral-index
        implied = absfit[0] + (2.0 if args.kind == 'potential' else 0.0)
        print(f'  grid spectrum check: E_B ~ k^{implied:+.3f} +- '
              f'{absfit[1]:.3f} (requested {alpha:+.3f}); realised minus '
              f'requested {resfit[0]:+.3f} +- {resfit[1]:.3f}')

    # --- interpolate to the particles ------------------------------------
    V = trilinear_sample(grid, pos, origin, L)
    rms = float(np.sqrt(np.mean(np.sum(V * V, axis=1))))
    if rms <= 0.0:
        raise SystemExit(
            'the turbulent field came out identically zero - the requested '
            f'band ({args.turb_kmin:g}..{nmax:g} fundamental modes) holds no '
            f'mode on a {ng}^3 grid; widen it or raise --turb-ngrid')
    V /= rms                             # unit rms |V|, the vector analogue
    kdesc = f'{args.turb_kmin:g}..{nmax:.4g}'
    return V, (f'turbulent (E_B ~ k^{alpha:g}, |{"A" if args.kind == "potential" else "B"}(k)|^2 '
               f'~ k^{beta:g}, k {kdesc} k_f, {ng}^3 grid, seed {args.seed})')


def trilinear_sample(grid, pos, origin, L):
    """Sample a (3,ng,ng,ng) periodic grid at `pos` -> (N,3).

    Trilinear rather than nearest so the result is continuous; the grid field
    is exactly solenoidal but the interpolant is only approximately so, which
    is why the potential route (curl on the particles) is the recommended one.
    """
    ng = grid.shape[1]
    g = (pos - origin) / L * ng
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    i0 %= ng
    i1 = (i0 + 1) % ng
    out = np.zeros((len(pos), 3))
    for c in range(3):
        gc = grid[c]
        for bx in (0, 1):
            wx = f[:, 0] if bx else 1.0 - f[:, 0]
            ix = i1[:, 0] if bx else i0[:, 0]
            for by in (0, 1):
                wy = f[:, 1] if by else 1.0 - f[:, 1]
                iy = i1[:, 1] if by else i0[:, 1]
                for bz in (0, 1):
                    wz = f[:, 2] if bz else 1.0 - f[:, 2]
                    iz = i1[:, 2] if bz else i0[:, 2]
                    out[:, c] += wx * wy * wz * gc[ix, iy, iz]
    return out


_VECTOR_PROFILE_CACHE = {}


def vector_profile_field(name, pos, args, box):
    """(N,3) dimensionless vector shape (unit rms magnitude) + description.

    Memoised on the parameters that define the draw, because the interactive
    amplitude loop and --normalise both ask for the same field again and a
    Gaussian random field is expensive AND (without a fixed seed) different
    every time. `main` resolves --seed to a concrete integer before any call,
    so a rerun of the printed command reproduces the field exactly.
    """
    if name != 'turbulent':
        raise SystemExit(f'unknown vector profile {name!r}')
    key = (name, args.seed, int(args.turb_ngrid), float(args.spectral_index),
           float(args.turb_kmin),
           None if args.turb_kmax is None else float(args.turb_kmax),
           float(args.turb_pad), args.kind, len(pos),
           None if box is None else tuple(np.asarray(box, dtype=float)))
    hit = _VECTOR_PROFILE_CACHE.get(key)
    if hit is None:
        hit = turbulent_field(pos, args, box)
        _VECTOR_PROFILE_CACHE[key] = hit
    return hit


def grid_spectrum_slope(grid, L, kmin, kmax, amp=None):
    """Shell-spectrum fit of a (3,ng,ng,ng) field. -> (slope, stderr) or None.

    Measured ON THE GRID, where the draw is exact, so it checks the
    construction (the filter, the projection, the normalisation) and nothing
    else. What reaches the particles additionally passes through trilinear
    interpolation, and in potential mode through the SPH curl - neither is
    audited here.

    WITH `amp` (the requested filter, i.e. sqrt of the requested |V(k)|^2),
    the fit is of the RESIDUAL power/amp^2, whose slope is 0 for a faithful
    draw. It is the more robust of the two on a short or anisotropic band,
    because dividing mode-by-mode cancels the mode geometry exactly and needs
    no 4 pi k^2 bookkeeping at all.

    WITHOUT `amp` it is the absolute slope of E(k). Over 6 seeds that
    recovers a requested -5/3 as -1.666 +- 0.056 at 64^3 and -1.677 +- 0.025
    at 128^3, so it is unbiased and the scatter is honest single-realisation
    noise.
    """
    ng = grid.shape[1]
    Vk = np.fft.rfftn(grid, axes=(1, 2, 3))
    kx = 2.0 * np.pi * np.fft.fftfreq(ng, d=L[0] / ng)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ng, d=L[1] / ng)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(ng, d=L[2] / ng)[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    # rfft keeps only half of k-space; every plane except kz=0 (and Nyquist)
    # stands for two modes. This is a MULTIPLICITY, so it belongs in the
    # averaging weights, not multiplied into the power: the kz=0 plane is a
    # much larger fraction of the few modes at low k than of the many at high
    # k, so folding it into the values instead tilts the fitted slope by
    # ~+0.3 - a bias that survives seed averaging and looks exactly like a
    # mis-drawn spectrum.
    mult = np.full(kmag.shape, 2.0)
    mult[:, :, 0] = 1.0
    if ng % 2 == 0:
        mult[:, :, -1] = 1.0
    power = np.sum(np.abs(Vk) ** 2, axis=0)

    sel = (kmag >= kmin) & (kmag <= kmax)
    if amp is not None:
        sel = sel & (amp > 0.0)
        power = np.divide(power, amp ** 2, out=np.zeros_like(power),
                          where=amp > 0.0)
    if sel.sum() < 8:
        return None
    kk = kmag[sel]
    edges = np.geomspace(kk.min(), kk.max(), 13)
    idx = np.clip(np.digitize(kk, edges) - 1, 0, len(edges) - 2)
    pw = power[sel]
    mu = mult[sel]
    nb = len(edges) - 1
    tot = np.bincount(idx, weights=mu * pw, minlength=nb)
    cnt = np.bincount(idx, weights=mu, minlength=nb)
    ksum = np.bincount(idx, weights=mu * kk, minlength=nb)
    ok = (cnt > 0) & (tot > 0)
    if ok.sum() < 4:
        return None
    # Abscissa = the MEAN k of the modes that actually landed in the bin, not
    # the bin's geometric centre: the modes are not distributed uniformly
    # inside a shell, and on a steep spectrum that mismatch biases the fitted
    # slope by a few hundredths - enough to look like a construction error.
    kc = ksum[ok] / cnt[ok]
    # E(k) dk = shell power: 4 pi k^2 <|V|^2> dk, and <|V|^2> = tot/cnt.
    # In residual mode the 4 pi k^2 is dropped: it would add exactly 2 to a
    # slope that must come out 0.
    E = tot[ok] / cnt[ok]
    if amp is None:
        E = 4.0 * np.pi * kc ** 2 * E
    x = np.log(kc)
    y = np.log(E)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    dof = len(x) - 2
    sxx = float(np.sum((x - x.mean()) ** 2))
    stderr = (float(np.sqrt(np.sum(resid ** 2) / dof / sxx))
              if dof > 0 and sxx > 0 else float('nan'))
    return float(slope), stderr


# ---------------------------------------------------------------------------
# amplitude
# ---------------------------------------------------------------------------

def pressures(gas, units):
    """(P_thermal, P_kinetic) per gas particle, in code units.

    Thermal P = (gamma-1) rho u with u = T * TtouFac (the tipsy column is a
    temperature in the gasoline convention the old script assumed); kinetic
    P = rho v^2, the ram pressure the beta-kin basis balances against.
    """
    rho = gas[:, C_RHO].astype(np.float64)
    T = gas[:, C_TEMP].astype(np.float64)
    v2 = np.sum(gas[:, C_VEL].astype(np.float64) ** 2, axis=1)
    P = (units['gamma'] - 1.0) * rho * T * units['TtouFac']
    return P, rho * v2


def amplitude(basis, value, gas, units):
    """(N,) per-particle amplitude in CODE units, plus a description.

    Every basis is LINEAR in a single scalar knob (a field value, or
    1/sqrt(beta)), which is what lets the interactive 'happy with it?' loop
    rescale an already-computed curl instead of recomputing one per attempt.
    """
    n = len(gas)
    P, Pkin = pressures(gas, units)
    if basis == 'gauss':
        return (np.full(n, value * units['GausstoCode']),
                f'{value:g} G uniform')
    if basis == 'code':
        return np.full(n, float(value)), f'{value:g} code uniform'
    if basis == 'beta-th':
        return np.sqrt(2.0 * P / value), f'beta_th = {value:g}'
    if basis == 'beta-kin':
        return np.sqrt(2.0 * Pkin / value), f'beta_kin = {value:g}'
    raise SystemExit(f'unknown strength basis {basis!r}')


def _beta_stats(Pref, mag):
    """Per-particle beta = 2P/B^2: (min, median, max, n_zero_B).

    Particles with B = 0 have an infinite beta (unmagnetised); they are
    EXCLUDED from the percentiles and counted separately rather than being
    allowed to poison the max, which is the number you actually want to look
    at when checking that no region is over-magnetised.
    """
    zero = mag <= 0.0
    if zero.all():
        return None, None, None, int(zero.sum())
    b = 2.0 * Pref[~zero] / mag[~zero] ** 2
    return (float(b.min()), float(np.median(b)), float(b.max()),
            int(zero.sum()))


def report_field(B, gas, units, vol, label='B'):
    """Print the field's amplitude, beta, energy and net-flux diagnostics."""
    P, Pkin = pressures(gas, units)
    mag = np.linalg.norm(B, axis=1)
    c2g = units['CodetoGauss']
    rho = gas[:, C_RHO].astype(np.float64)
    mass = gas[:, C_MASS].astype(np.float64)
    rms = float(np.sqrt(np.average(mag ** 2, weights=vol)))

    print(f'  |{label}|      max {mag.max() * c2g:.4e} G  '
          f'min {mag.min() * c2g:.4e} G  rms {rms * c2g:.4e} G   '
          f'[rms volume weighted]')
    print(f'  {" " * len(label)}       ({mag.max():.4e} / {mag.min():.4e} / '
          f'{rms:.4e} code)')

    # net flux: a "no net flux" configuration must have |<B>| << Brms, and a
    # uniform one |<B>|/Brms ~ 1. Volume weighted, so it is the field's mean,
    # not the particle sampling's.
    mean_B = np.average(B, axis=0, weights=vol)
    net = float(np.linalg.norm(mean_B)) / rms if rms > 0.0 else 0.0
    print(f'  <{label}>      ({mean_B[0]:+.3e}, {mean_B[1]:+.3e}, '
          f'{mean_B[2]:+.3e}) code   |<{label}>|/rms {net:.4f}  [net flux]')

    if rms > 0.0:
        for name, Pr in (('beta_th ', P), ('beta_kin', Pkin)):
            volb = 2.0 * float(np.average(Pr, weights=vol)) / rms ** 2
            lo, med, hi, nz = _beta_stats(Pr, mag)
            extra = f'   ({nz} particles with B=0)' if nz else ''
            if lo is None:
                print(f'  {name}  vol {volb:.4g}{extra}')
            else:
                print(f'  {name}  vol {volb:9.4g}   per particle: '
                      f'min {lo:.4g}  median {med:.4g}  max {hi:.4g}{extra}')

    # energies, in code units (mu0 = 1 in this unit system, so E_mag = B^2/2
    # per unit volume). The RATIOS are the meaningful numbers: they say
    # whether the field is dynamically relevant at all.
    e_mag = float(0.5 * np.sum(vol * mag ** 2))
    e_th = float(np.sum(mass * gas[:, C_TEMP].astype(np.float64)
                        * units['TtouFac']))
    e_kin = float(0.5 * np.sum(mass * np.sum(
        gas[:, C_VEL].astype(np.float64) ** 2, axis=1)))
    frac = lambda a, b: f'{a / b:.4e}' if b > 0 else 'n/a'   # noqa: E731
    print(f'  energy    E_mag {e_mag:.4e}   E_therm {e_th:.4e}   '
          f'E_kin {e_kin:.4e}')
    print(f'            E_mag/E_therm {frac(e_mag, e_th)}   '
          f'E_mag/E_kin {frac(e_mag, e_kin)}')

    # Alfven speed: max v_A is what the MHD signal-velocity timestep keys off,
    # so a field that looks harmless in beta can still make the run crawl.
    with np.errstate(divide='ignore', invalid='ignore'):
        vA = np.where(rho > 0.0, mag / np.sqrt(np.maximum(rho, 1e-300)), 0.0)
    vrms = float(np.sqrt(np.mean(np.sum(
        gas[:, C_VEL].astype(np.float64) ** 2, axis=1))))
    print(f'  v_Alfven  max {vA.max():.4e}   median {np.median(vA):.4e} code'
          + (f'   (v_rms {vrms:.4e}, max v_A/v_rms {vA.max() / vrms:.3g})'
             if vrms > 0 else ''))
    return dict(Brms=rms, Bmax=float(mag.max()), Bmin=float(mag.min()),
                mean_B=mean_B.tolist(), net_flux_ratio=net, e_mag=e_mag,
                e_therm=e_th, e_kin=e_kin, vA_max=float(vA.max()))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def build_vector(args, gas, rho, units, box, value):
    """V = sign * amplitude * profile * direction  (the field OR the potential).

    A VECTOR profile supplies the orientation itself, so it replaces the
    (scalar profile x direction) pair rather than riding on it.
    """
    pos = gas[:, C_POS].astype(np.float64)
    amp, adesc = amplitude(args.strength, value, gas, units)
    if args.profile in VECTOR_PROFILES:
        shape, pdesc = vector_profile_field(args.profile, pos, args, box)
        V = (args.sign * amp)[:, None] * shape
        return V, f'{adesc}  x  {pdesc}   [vector profile: --direction unused]'
    prof, pdesc = profile_field(args.profile, pos, rho, args, box)
    dirn = direction_field(args.direction, pos)
    V = (args.sign * amp * prof)[:, None] * dirn
    return V, f'{adesc}  x  {pdesc}  x  {args.direction}-hat'


def normalisation_target(mode, args, gas, rho, units, vol, value, pos, box):
    """The volume-weighted rms |B| that --normalise should deliver, or None.

    Renormalisation exists because in POTENTIAL mode the constructed vector is
    A (units of B x length): the curl's raw scale is set by the geometry and
    the estimator's normalisation shortfall, not by the strength you asked
    for. Two meanings of "the strength you asked for":

    profile-rms  the rms of amplitude x profile, i.e. the requested value is
                 the amplitude AT THE PROFILE'S PEAK. Exact for a uniform
                 profile; for a decaying profile the field is weaker than the
                 requested value nearly everywhere, which is what the old
                 script did (its option 2 multiplied a beta-based amplitude by
                 a raw cylindrical radius).
    beta         the requested BETA is delivered in the volume-weighted sense,
                 2 <P>_vol / Brms^2 == value. Almost always what you want when
                 you specify a field by beta AND shape it with a profile -
                 otherwise the profile silently changes the beta you get.
    """
    if mode == 'none':
        return None
    if mode == 'beta':
        if args.strength not in ('beta-th', 'beta-kin'):
            raise SystemExit('--normalise beta needs --strength beta-th or '
                             'beta-kin (it is the beta that gets matched)')
        P, Pkin = pressures(gas, units)
        Pref = Pkin if args.strength == 'beta-kin' else P
        return float(np.sqrt(2.0 * np.average(Pref, weights=vol) / value))
    if mode == 'profile-rms':
        amp, _ = amplitude(args.strength, value, gas, units)
        if args.profile in VECTOR_PROFILES:
            shape, _ = vector_profile_field(args.profile, pos, args, box)
            return float(np.sqrt(np.average(
                amp ** 2 * np.sum(shape ** 2, axis=1), weights=vol)))
        prof, _ = profile_field(args.profile, pos, rho, args, box)
        return float(np.sqrt(np.average((amp * prof) ** 2, weights=vol)))
    raise SystemExit(f'unknown --normalise mode {mode!r}')


def default_normalise(args):
    """kind=field is written exactly as specified (the old script's contract);
    kind=potential must be rescaled, and a beta basis wants its beta met."""
    if args.kind == 'field':
        return 'none'
    return 'beta' if args.strength.startswith('beta') else 'profile-rms'


def curl_to_field(args, A, pos, mass, state, flavour):
    """B = curl A in the chosen flavour, with the normalisation report."""
    print(f'\ntaking the SPH curl of A  (flavour {flavour}, '
          f'nsmooth {args.nsmooth}) ...', flush=True)
    B, _ = curl(pos, mass, A, state=state, flavour=flavour)
    return B


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def divergence_report(pos, mass, B, state, flavour, tag):
    """Print divBErr statistics using the same definition as divbclean.

    divBErr = h|div B|/max(|B|, 1e-12 Brms). Rules of thumb for an IC: mean
    <~1e-3 is clean, ~1e-2 is the level a raw prescribed field sits at, and the
    fraction above 1e-2 says whether a few bad particles or the whole
    distribution carries the error - they need different fixes (more cleaning
    steps vs a better field construction).
    """
    d, _ = divergence(pos, mass, B, state=state, flavour=flavour)
    eps = divergence_error(B, d, state.h)
    over = float(np.mean(eps > 1e-2))
    print(f'  divBErr ({tag}, flavour {flavour}, nsmooth {state.nsmooth}):')
    print(f'    h|divB|/max(|B|,1e-12 Brms)   mean {eps.mean():.4e}  median '
          f'{np.median(eps):.4e}  p95 {np.percentile(eps, 95):.4e}  '
          f'max {eps.max():.4e}')
    print(f'    fraction with divBErr > 1e-2: {over:.2%}   '
          f'max |div B| {np.abs(d).max():.4e} code')
    return eps, dict(eps_mean=float(eps.mean()),
                     eps_median=float(np.median(eps)),
                     eps_p95=float(np.percentile(eps, 95)),
                     eps_max=float(eps.max()), eps_frac_over_1e2=over,
                     max_abs_divB=float(np.abs(d).max()))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def pad_to_all(vec, ngas, ndark, nstar):
    """Gas-only (ngas,3) -> full (N,3) with zeros for dark and star."""
    out = np.zeros((ngas + ndark + nstar, 3))
    out[:ngas] = vec
    return out


def write_snapshot(infile, out, gas, dark, star, header, time, fields,
                   copy_aux=True):
    """Write OUT + the requested vector aux files, then the sibling aux.

    `fields` maps aux label -> (N,3) array. Component twins (LABELx/y/z) are
    written alongside the packed file because some consumers read those
    INSTEAD - leaving a stale twin behind means a consumer can silently pick
    up the old field.
    """
    tip.writetipsy(gas, dark, star, out, header, time=time)
    written = []
    for label, arr in fields.items():
        tip.writetipsyVecaux(arr, label, out)
        written.append(f'{out}.{label}')
        for i, c in enumerate('xyz'):
            tip.writetipsyaux(arr[:, i], label + c, out)
            written.append(f'{out}.{label}{c}')
    if copy_aux:
        skip = tuple(fields)
        copied = 0
        for aux in glob.glob(infile + '.*'):
            lab = aux[len(infile) + 1:]
            if not lab or '/' in lab or lab.startswith(skip):
                continue
            shutil.copyfile(aux, out + '.' + lab)
            copied += 1
        if copied:
            print(f'copied {copied} sibling aux files')
    print('wrote ' + out + '\n  ' + '\n  '.join(written))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog=os.environ.get('IC_TOOL_NAME'),   # set by the bin/ wrapper
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('infile', help='input tipsy snapshot')
    p.add_argument('-o', dest='out', default=None,
                   help='output snapshot (default: _B inserted before .NNNNN)')

    g = p.add_argument_group('field specification')
    g.add_argument('--profile', choices=PROFILES,
                   help='dimensionless spatial shape')
    g.add_argument('--strength', choices=STRENGTHS,
                   help='what the amplitude is specified in')
    g.add_argument('--value', type=float,
                   help='the amplitude value (G, code, or the beta)')
    g.add_argument('--direction', choices=DIRECTIONS,
                   help='unit vector field the amplitude rides on')
    g.add_argument('--kind', choices=KINDS,
                   help="'field' = the vector IS B; 'potential' = it is A "
                        'and B is its SPH curl')
    g.add_argument('--sign', type=float, default=1.0,
                   help='+1 or -1 (default +1)')
    g.add_argument('--rho-power', type=float, default=2.0 / 3.0,
                   help='exponent of the rho profile (default 2/3, flux '
                        'freezing under isotropic compression)')
    g.add_argument('--profile-ref', type=float, default=None,
                   help='reference value the profile is normalised by '
                        '(default: its own maximum)')
    g.add_argument('--wavenumber', type=float, default=1.0,
                   help='wavenumber of the sin profile in box units')
    g.add_argument('--sin-axis', choices=('x', 'y', 'z'), default='x',
                   help='axis the sin profile varies along (default x)')
    g.add_argument('--scale-radius', type=float, default=1.0,
                   help='scale radius of the exp profile')

    g = p.add_argument_group('turbulent profile')
    g.add_argument('--spectral-index', type=float, default=-5.0 / 3.0,
                   help='slope a of the MAGNETIC shell spectrum E_B(k) ~ k^a '
                        '(default -5/3 Kolmogorov; -2 Burgers, +1.5 '
                        'Batchelor). Always describes B: what gets drawn is '
                        'converted for you, so --kind potential draws A with '
                        'two powers less')
    g.add_argument('--turb-ngrid', type=int, default=64,
                   help='FFT grid per side (even, >=4; default 64)')
    g.add_argument('--turb-kmin', type=float, default=1.0,
                   help='smallest wavenumber kept, in fundamental modes of '
                        'the domain (default 1 = one wavelength across it)')
    g.add_argument('--turb-kmax', type=float, default=None,
                   help='largest wavenumber kept, same units (default: the '
                        'isotropic Nyquist - ngrid/2 on a cubic domain, less '
                        'on an elongated one, since a mode must fit along '
                        'every axis)')
    g.add_argument('--turb-pad', type=float, default=0.1,
                   help='fractional padding of the particle bounding box on '
                        'an OPEN domain (default 0.1); ignored when periodic')
    g.add_argument('--seed', type=int, default=None,
                   help='random seed for the turbulent draw (default: drawn '
                        'from entropy and recorded in the audit json, so the '
                        'run is always reproducible after the fact)')

    g = p.add_argument_group('SPH operators')
    g.add_argument('--flavour', choices=sorted(FLAVOUR_SPECS),
                   help='operator flavour for BOTH the curl and the cleaning '
                        '(gdforce = the gasolinemhd/GDSPH convention)')
    g.add_argument('-s', '--nsmooth', type=int, default=64,
                   help='neighbour count for rho/h and the operators (64)')
    g.add_argument('--snapshot-rho-h', action='store_true',
                   help="use the snapshot's own rho and hsmooth columns "
                        'instead of recomputing the glassgen kNN estimate')
    g.add_argument('--normalise', choices=('none', 'profile-rms', 'beta'),
                   default=None,
                   help='rescale the FINAL B so the requested strength is '
                        "actually delivered. 'profile-rms': its volume rms "
                        'equals rms(amplitude x profile); '
                        "'beta': the requested beta is met in the volume "
                        "sense. Default: 'none' for --kind field (written "
                        'exactly as specified), else beta for a beta basis '
                        'and profile-rms otherwise')
    g.add_argument('--no-renormalise', dest='normalise',
                   action='store_const', const='none',
                   help='alias for --normalise none')
    dom = p.add_mutually_exclusive_group()
    dom.add_argument('--box', type=float, nargs='+', metavar='L',
                     help='periodic box lengths (one value or LX LY LZ)')
    dom.add_argument('--open', action='store_true', help='open domain')

    g = p.add_argument_group('divergence cleaning (mhd_divBclean)')
    g.add_argument('--clean', action='store_true',
                   help='run static hyperbolic cleaning after building B')
    g.add_argument('--clean-steps', type=int, default=500)
    g.add_argument('--sigma', type=float, default=1.0)
    g.add_argument('--dt', type=float, default=0.2)
    g.add_argument('--target-mean', type=float, default=0.0,
                   help='stop cleaning once mean eps <= this (0 = disabled)')
    g.add_argument('--target-max', type=float, default=0.0,
                   help='stop cleaning once max eps <= this (0 = disabled)')

    g = p.add_argument_group(
        'units (else from a sibling .param/.log, else defaults)')
    g.add_argument('--kpc-unit', type=float, default=None)
    g.add_argument('--msol-unit', type=float, default=None)
    g.add_argument('--gamma', type=float, default=None)
    g.add_argument('--mu', type=float, default=None)

    g = p.add_argument_group('output')
    g.add_argument('--center', choices=('none', 'potential-min', 'mass'),
                   default=None, help='recentre the snapshot first')
    g.add_argument('--no-copy-aux', dest='copy_aux', action='store_false')
    g.add_argument('--dry-run', action='store_true',
                   help='report everything, write nothing')
    g.add_argument('--yes', action='store_true',
                   help='accept the first amplitude (skip the happy loop)')
    return p


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    ask = Asker()

    # ---- input ----------------------------------------------------------
    infile = args.infile
    if not os.path.isfile(infile):
        raise SystemExit(f'no such snapshot: {infile}')
    out = args.out
    if out is None and ask.interactive and not args.dry_run:
        out = input('Output name (blank = auto): ').strip() or None
    if out is None:
        out = (re.sub(r'(\.\d+)$', r'_B\1', infile)
               if re.search(r'\.\d+$', infile) else infile + '_B')
    if os.path.abspath(out) == os.path.abspath(infile):
        raise SystemExit('output would overwrite the input snapshot')

    gas, dark, star, header, time = tip.readtipsy(infile)
    gas = np.asarray(gas, dtype=np.float64)
    dark = np.asarray(dark, dtype=np.float64)
    star = np.asarray(star, dtype=np.float64)
    ngas, ndark, nstar = int(header[2]), int(header[3]), int(header[4])
    if ngas == 0:
        raise SystemExit('no gas particles - nothing to magnetise')
    print(f'read {infile}: {ngas} gas, {ndark} dark, {nstar} star')

    # ---- units ----------------------------------------------------------
    pfile = find_param(infile)
    logfile = find_log(infile)
    got, praw = ({}, {})
    if logfile:
        lgot = params_from_log(logfile)
        got.update(lgot)
        print(f'parameters from {logfile}: '
              + (', '.join(f'{k}={v:g}' for k, v in lgot.items()) or 'none'))
    if pfile:
        pgot, praw = params_from_file(pfile)
        got.update(pgot)
        print(f'parameters from {pfile}: '
              + (', '.join(f'{k}={v:g}' for k, v in pgot.items()) or 'none'))

    cli_units = {name: getattr(args, name)
                 for name, _, _ in UNIT_FIELDS
                 if getattr(args, name) is not None}
    if not pfile and not logfile:
        got.update(DEFAULT_UNITS)
        print('no sibling .param or .log file was found; using default '
              'unit values:')
        for name, _, prompt in UNIT_FIELDS:
            print(f'  {prompt}: {DEFAULT_UNITS[name]:g}')
        enter_units = (ask.interactive
                       and ask.yes(None, '--units',
                                   'Would you like to input unit values?'))
        if enter_units:
            for name, flag, prompt in UNIT_FIELDS:
                if name not in cli_units:
                    got[name] = ask.number(None, flag, prompt)

    got.update(cli_units)
    for name, flag, prompt in UNIT_FIELDS:
        cli = getattr(args, name)
        if cli is None and name not in got:
            # a unit with no safe default must be supplied; gamma/mu fall back
            # rather than blocking a batch run on a conventional value
            default = DEFAULT_UNITS[name] if name in ('gamma', 'mu') else None
            if default is not None and not ask.interactive:
                print(f'  {flag} not given, using default {default:g}')
                got[name] = default
            else:
                got[name] = ask.number(None, flag, prompt)
    units = unit_conversions(got['kpc_unit'], got['msol_unit'],
                             got['gamma'], got['mu'])

    # ---- domain ---------------------------------------------------------
    if args.box is not None:
        box = (np.repeat(args.box[0], 3) if len(args.box) == 1
               else np.asarray(args.box, dtype=np.float64))
        if box.shape != (3,):
            ap.error('--box takes one length or LX LY LZ')
    elif args.open:
        box = None
    elif praw:
        from glassgen.params import box_from_param
        box = box_from_param(praw)
    else:
        box = None
    print(f'domain: {"open" if box is None else f"periodic {box}"}')

    # ---- centering ------------------------------------------------------
    center = args.center
    if center is None:
        center = ('potential-min' if ask.interactive and ask.yes(
            None, '--center',
            'Recentre the simulation on the gas potential minimum?')
            else 'none')
    if center != 'none':
        if center == 'potential-min':
            origin = gas[np.argmin(gas[:, C_PHI]), C_POS].copy()
        else:
            m = gas[:, C_MASS]
            origin = np.average(gas[:, C_POS], axis=0, weights=m)
        print(f'recentring on {center} at {origin}')
        for arr in (gas, dark, star):
            if len(arr):
                arr[:, C_POS] -= origin

    pos = np.ascontiguousarray(gas[:, C_POS])
    mass = np.ascontiguousarray(gas[:, C_MASS])

    # ---- the four axes --------------------------------------------------
    args.profile = ask.choice(
        args.profile, '--profile', 'Spatial profile of the field:', PROFILES,
        {'uniform': 'constant', 'rho': 'density powerlaw (rho/rho_ref)^p',
         'R': 'linear in cylindrical radius', 'z': 'linear in z (quadrupole)',
         'sin': 'sinusoidal, no net flux', 'exp': 'exponential in r',
         'turbulent': 'random solenoidal field, power-law spectrum '
                      '(sets its own direction)'})
    if args.profile in VECTOR_PROFILES:
        # the profile carries the orientation, so there is nothing to ask and
        # nothing for a supplied --direction to do
        if args.direction is not None:
            print(f'  NOTE: --direction {args.direction} is ignored for the '
                  f'{args.profile} profile, which is a vector field in its '
                  'own right.')
        args.direction = None
    else:
        args.direction = ask.choice(
            args.direction, '--direction',
            'Direction the vector points along:', DIRECTIONS,
            {'azi': 'phi-hat (toroidal)', 'Rhat': 'cylindrical radial',
             'rhat': 'spherical radial'})
    args.kind = ask.choice(
        args.kind, '--kind', 'Is the constructed vector the field or a '
        'potential?', KINDS,
        {'field': 'it IS B', 'potential': 'it is A; B = SPH curl A'})
    args.output = 'field' if args.kind == 'field' else 'both'
    args.strength = ask.choice(
        args.strength, '--strength', 'Base the field strength on:', STRENGTHS,
        {'gauss': 'a value in Gauss', 'code': 'a value in code units',
         'beta-th': 'thermal plasma beta 2P/B^2',
         'beta-kin': 'kinetic beta 2 rho v^2/B^2'})
    flavour = args.flavour
    if flavour is None:
        flavour = ('gdforce' if not (args.kind == 'potential' or args.clean)
                   else ask.choice(
                       None, '--flavour',
                       'SPH operator flavour - used for BOTH the curl and the '
                       'cleaning, so the IC is clean in the same measure the '
                       'field was built in:', sorted(FLAVOUR_SPECS),
                       {'gdforce': 'GDSPH geometric density average '
                                   '(gasolinemhd) - default',
                        'rho2': 'standard pP/rho^2',
                        'gdforce-own': 'GDSPH with the Phantom window rule',
                        'rho2-own': 'Tricco & Price / Phantom convention'}))
    if args.normalise is None:
        args.normalise = default_normalise(args)
    if args.profile in VECTOR_PROFILES and args.seed is None:
        # resolve it ONCE, before anything draws: the amplitude loop and
        # --normalise both re-request the field, and an unseeded rng would
        # hand back a different one each time
        args.seed = int(np.random.SeedSequence().entropy % (2 ** 32))
        print(f'\n--seed not given; using {args.seed} (recorded in the audit '
              'json - pass it back to reproduce this field)')
    print(f'\nconfiguration: profile={args.profile} strength={args.strength} '
          f'direction={args.direction or "n/a (vector profile)"} '
          f'kind={args.kind} sign={args.sign:+g}\n'
          f'               flavour={flavour} output={args.output} '
          f'normalise={args.normalise}')

    # ---- rho, h and the neighbour state --------------------------------
    if args.snapshot_rho_h:
        rho = np.ascontiguousarray(gas[:, C_RHO])
        h = np.ascontiguousarray(gas[:, C_HSM])
        print('using the snapshot rho/hsmooth columns for the operators')
    else:
        rho = h = None
    print(f'building neighbour state (nsmooth {args.nsmooth}) ...', flush=True)
    state = operator_state(pos, mass, box=box, nsmooth=args.nsmooth,
                           rho=rho, h=h)
    rho = state.rho
    vol = mass / rho

    # ---- amplitude, with the interactive accept loop --------------------
    value = args.value
    raw_stats = raw_divstats = None
    while True:
        if value is None:
            value = ask.number(
                None, '--value',
                {'gauss': 'Field strength [Gauss]', 'code': 'Field strength '
                 '[code]', 'beta-th': 'beta_th', 'beta-kin': 'beta_kin'}
                [args.strength])
        V, desc = build_vector(args, gas, rho, units, box, value)
        print(f'\nconstructed vector: {desc}')

        if args.kind == 'potential':
            if box is not None and args.profile in ('R', 'z', 'uniform') \
                    and args.direction in ('x', 'y', 'z'):
                print('  NOTE: this A is not periodic; in a periodic box the '
                      'minimum-image walk\n        will see a jump in A across '
                      'the wrap and the boundary shell will be wrong.')
            B = curl_to_field(args, V, pos, mass, state, flavour)
            print('  raw curl amplitude:')
            raw_stats = report_field(B, gas, units, vol, label='B')
            _, raw_divstats = divergence_report(
                pos, mass, B, state, flavour, 'raw curl')
        else:
            B = V

        want = normalisation_target(args.normalise, args, gas, rho, units,
                                    vol, value, pos, box)
        did_normalise = False
        if want is not None:
            raw = float(np.sqrt(np.average(np.sum(B ** 2, axis=1),
                                           weights=vol)))
            if raw > 0.0 and want > 0.0:
                B = B * (want / raw)
                did_normalise = True
                print(f'  --normalise {args.normalise}: rescaled by '
                      f'x{want / raw:.6g}')

        print('\nresulting field:')
        stats = report_field(B, gas, units, vol, label='B')
        _, divstats = divergence_report(
            pos, mass, B, state, flavour,
            'normalised field' if did_normalise else 'resulting field')
        if args.yes or not ask.interactive:
            break
        if input('Are you happy with this value? [y/N]: ').lower().startswith(
                'y'):
            break
        value = None

    audit = dict(input=infile, output=out, profile=args.profile,
                 strength=args.strength, value=value,
                 direction=args.direction, kind=args.kind, sign=args.sign,
                 flavour=flavour, nsmooth=args.nsmooth,
                 normalise=args.normalise, aux=args.output,
                 periodic=(box is None and 'open' or [float(b) for b in box]),
                 units={k: float(v) for k, v in units.items()},
                 field=stats, divergence_as_built=divstats)
    if raw_stats is not None:
        audit['field_raw'] = raw_stats
        audit['divergence_raw'] = raw_divstats
    if args.profile in VECTOR_PROFILES:
        # everything needed to redraw the identical field
        audit['turbulent'] = dict(
            seed=int(args.seed), spectral_index=float(args.spectral_index),
            ngrid=int(args.turb_ngrid), kmin=float(args.turb_kmin),
            kmax=(None if args.turb_kmax is None else float(args.turb_kmax)),
            pad=float(args.turb_pad))

    # ---- optional cleaning ----------------------------------------------
    if args.clean:
        from mhd_divBclean import clean_divergence
        print(f'\ncleaning (flavour {flavour}, sigma {args.sigma}, '
              f'dt {args.dt}, up to {args.clean_steps} steps) ...', flush=True)
        B, audit_c = clean_divergence(
            pos, mass, B, box=box, nsmooth=args.nsmooth, sigma=args.sigma,
            dt=args.dt, steps=args.clean_steps, flavour=flavour, verbose=True,
            rho=state.rho, h=state.h,
            target_mean=args.target_mean, target_max=args.target_max)
        m0, _, x0 = audit_c['eps_before']
        m1, _, x1 = audit_c['eps_after']
        print(f'  divBErr mean {m0:.4e} -> {m1:.4e}, '
              f'max {x0:.4e} -> {x1:.4e}, '
              f'E_B {audit_c["e_b_after"] / audit_c["e_b_before"] - 1.0:+.3%}')
        if (args.target_mean or args.target_max) \
                and not audit_c['target_reached']:
            print('  WARNING: divergence target not reached - raise '
                  '--clean-steps')
        print('\nfield after cleaning:')
        audit['field_cleaned'] = report_field(B, gas, units, vol, label='B')
        # re-measure with the same operator, so 'before' and 'after' in the
        # audit are the same number computed the same way
        _, audit['divergence_cleaned'] = divergence_report(
            pos, mass, B, state, flavour, 'cleaned')
        audit['clean'] = dict(
            steps=int(audit_c['steps']), sigma=args.sigma, dt=args.dt,
            target_mean=args.target_mean, target_max=args.target_max,
            target_reached=bool(audit_c['target_reached']),
            e_b_before=float(audit_c['e_b_before']),
            e_b_after=float(audit_c['e_b_after']))

    # ---- write -----------------------------------------------------------
    fields = {'BField': pad_to_all(B, ngas, ndark, nstar)}
    if args.kind == 'potential':
        fields['AField'] = pad_to_all(V, ngas, ndark, nstar)

    # Cleaning acts on B. A is NOT re-derived from the cleaned B (that would
    # need a potential solve), so say explicitly that cleaning did not reach A.
    if args.clean and args.kind == 'potential':
        print('\nWARNING: --clean modified B, but the AField aux holds the '
              f'UNCLEANED potential A.\n         A consumer that curls that A '
              f'gets the pre-cleaning divergence '
              f'({"divBErr mean %.4e" % audit_c["eps_before"][0]}), '
              f'not the cleaned '
              f'one. The cleaned field is in BField.')

    if args.dry_run:
        print('\n--dry-run: nothing written')
        return
    write_snapshot(infile, out, gas, dark, star, header, time, fields,
                   copy_aux=args.copy_aux)
    # the full configuration + diagnostics beside the IC, so a snapshot can
    # always be traced back to the field it was given and how good it was
    import json
    with open(out + '_setBField.json', 'w') as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f'  {out}_setBField.json  (configuration + diagnostics)')


if __name__ == '__main__':
    main()
