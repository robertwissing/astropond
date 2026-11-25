"""Read a gasoline .param and build the box + target density it encodes.

The C gdicgen reads the density profile it relaxes toward straight from the
.param (dICdensprofile / dICdensdir / dICdensR / ...). This module ports the
same reading so the Python backend (cli.py) and the benchmark get an identical
target without a separate density table - profiles 1/2/3 (uniform / step /
tanh). Profiles 4-7 (disk/accdisk), 8 (tabulated - the C backend reads
densitytable_xdr implicitly) and 9 (gaussian slab) are not ported and need a
density table (--table TABLE.npz; see glassgen.density).
"""
import argparse
import sys

import math

import numpy as np
from numba import njit, prange

from .density import DensityField


def expert_help(argv=None):
    """Hidden expert tier shared by the three production CLIs: returns
    `hide`, a help-text wrapper for rarely-used-but-working
    flags. Wrapped flags stay fully parseable but are omitted from --help
    unless --help-extra is on the command line; each parser also adds its own

        ap.add_argument('--help-extra', action='help', help=...)

    so `--help-extra` prints the complete surface (argparse's help action
    fires after the whole parser is built, by which point every hide() call
    already returned the real text)."""
    args = sys.argv[1:] if argv is None else list(argv)
    show = '--help-extra' in args

    def hide(text):
        return text if show else argparse.SUPPRESS

    return hide


def read_param(path):
    """Parse a gasoline .param into a {key: value-string} dict."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0]
            if '=' not in line:
                continue
            k, v = (s.strip() for s in line.split('=', 1))
            d[k] = v
    return d


def box_from_param(p):
    """Periodic box (3,) from the .param, or None for an open domain."""
    g = lambda k, dv=None: float(p[k]) if k in p else dv  # noqa: E731
    if int(float(p.get('bPeriodic', 1))) == 0:
        return None
    dp = g('dPeriod', 1.0)
    return np.array([g('dxPeriod', dp), g('dyPeriod', dp), g('dzPeriod', dp)])


def target_rho0(p):
    """Callable rho0(pos) -> (N,) : a faithful port of pkdUpdateDensity
    profiles 1/2/3 (uniform / radial step / tanh) from a .param. For uniform
    (profile 1) the absolute scale is irrelevant (only the ratio rho/rho0
    drives the relaxation); for 2/3 the inner/outer set the contrast, matching
    the C generator. Added here, not in the C generator: dICdensdir 0 is the
    square/cube of IC_setup_square.py, all of |x|, |y|, |z| within R."""
    prof = int(float(p.get('dICdensprofile', 1)))
    d = int(float(p.get('dICdensdir', 1)))
    R = float(p.get('dICdensR', 0.0))
    Rs = float(p.get('dICdensRsmooth', 1.0))
    inner = float(p.get('dICdensinner', 1.0))
    outer = float(p.get('dICdensouter', 1.0))
    if d == 0 and prof == 3:
        raise ValueError('dICdensdir 0 (square) has no smooth form: it takes dICdensprofile 2')

    def rho0(pos):
        if d == 0:
            r = np.abs(pos).max(axis=1)
        elif d < 4:
            r = np.abs(pos[:, d - 1])
        elif d == 4:
            r = np.sqrt((pos ** 2).sum(1))
        elif d == 5:
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        else:
            r = pos[:, d - 6]
        if prof >= 4:
            what = {4: 'disk', 5: 'disk', 6: 'disk', 7: 'accdisk',
                    8: 'table-driven: the C backend reads densitytable_xdr '
                       'implicitly, the Python one needs the table named',
                    9: 'gaussian slab'}.get(prof, 'unknown')
            raise NotImplementedError(
                f'density profile {prof} ({what}) is not evaluated from the '
                '.param - pass the density table the IC setup wrote '
                '(--table densitytable.npz; convert a legacy '
                'densitytable_xdr with python -m glassgen.density) or use '
                'the C backend.')
        if prof == 1:
            return np.full(len(pos), outer)
        if prof == 2:
            return np.where(r > R, outer, inner)
        # profile 3 (tanh): radial for r-directions, double-tanh slab for x/y/z
        if d > 3:
            return outer + (inner - outer) / (1.0 + np.exp(-(R - r) / Rs))
        x = pos[:, d - 1]
        return outer + (inner - outer) * 0.5 * (
            np.tanh((x + R) / Rs) - np.tanh((x - R) / Rs))

    return rho0


@njit(cache=True, inline='always')
def _profile_point(x, y, z, prof, d, R, Rs, inner, outer):
    """target_rho0 at one point (profiles 1/2/3), for the fused stencil."""
    if prof == 1:
        return outer
    c = x if (d == 1 or d == 6) else (y if (d == 2 or d == 7) else z)
    if d == 4:
        r = math.sqrt(x * x + y * y + z * z)
    elif d == 5:
        r = math.sqrt(x * x + y * y)
    elif d == 0:
        r = max(abs(x), abs(y), abs(z))
    elif d < 4:
        r = abs(c)
    else:
        r = c
    if prof == 2:
        return outer if r > R else inner
    if d > 3:
        return outer + (inner - outer) / (1.0 + math.exp(-(R - r) / Rs))
    return outer + (inner - outer) * 0.5 * (math.tanh((c + R) / Rs)
                                            - math.tanh((c - R) / Rs))


@njit(cache=True, parallel=True)
def _profile_stencil(pos, support, offsets, weights, prof, d, R, Rs, inner,
                     outer, out):
    for i in prange(pos.shape[0]):
        s = 0.0
        for q in range(offsets.shape[0]):
            s += weights[q] * _profile_point(
                pos[i, 0] + support[i] * offsets[q, 0],
                pos[i, 1] + support[i] * offsets[q, 1],
                pos[i, 2] + support[i] * offsets[q, 2],
                prof, d, R, Rs, inner, outer)
        out[i] = s


class ParamDensity(DensityField):
    """DensityField backed by a .param's dICdens* profile (profiles 1/2/3)."""

    def __init__(self, p):
        self._fn = target_rho0(p)
        self._prof = (int(float(p.get('dICdensprofile', 1))),
                      int(float(p.get('dICdensdir', 1))),
                      float(p.get('dICdensR', 0.0)),
                      float(p.get('dICdensRsmooth', 1.0)),
                      float(p.get('dICdensinner', 1.0)),
                      float(p.get('dICdensouter', 1.0)))

    def rho0(self, pos):
        return self._fn(pos)

    def rho0_stencil(self, pos, support, offsets, weights):
        """Fused kernel-matched readout sum_q w_q rho0(pos_i + support_i u_q)
        in one parallel numba pass (density.KernelMatchedDensity delegates
        here instead of building the offset-position block)."""
        if self._prof[0] >= 4:
            return self._fn(pos)   # not evaluated from the .param: raises
        out = np.empty(len(pos))
        _profile_stencil(np.ascontiguousarray(pos, dtype=np.float64),
                         np.ascontiguousarray(support, dtype=np.float64),
                         offsets, weights, *self._prof, out)
        return out


def density_from_param(p):
    """DensityField for the .param's density profile - the Python backend's
    automatic target, matching what the C gdicgen reads from the same file."""
    return ParamDensity(p)
