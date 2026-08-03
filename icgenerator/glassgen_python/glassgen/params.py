"""Read a gasoline .param and build the box + target density it encodes.

The C gdicgen reads the density profile it relaxes toward straight from the
.param (dICdensprofile / dICdensdir / dICdensR / ...). This module ports the
same reading so the Python backend (cli.py) and the benchmark get an identical
target without a separate density table - profiles 1/2/3 (uniform / step /
tanh). Profiles 4-7 (disk/accdisk), 8 (tabulated - the C backend reads
densitytable_xdr implicitly) and 9 (gaussian slab) are not ported and need a
density table (--table TABLE.npz; see glassgen.density); see PLAN.md
"Future tasks".
"""
import argparse
import sys

import numpy as np

from .density import DensityField


def expert_help(argv=None):
    """Hidden expert tier shared by the three production CLIs (CLEANUP_PLAN
    C4): returns `hide`, a help-text wrapper for rarely-used-but-working
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
    the C generator."""
    prof = int(float(p.get('dICdensprofile', 1)))
    d = int(float(p.get('dICdensdir', 1)))
    R = float(p.get('dICdensR', 0.0))
    Rs = float(p.get('dICdensRsmooth', 1.0))
    inner = float(p.get('dICdensinner', 1.0))
    outer = float(p.get('dICdensouter', 1.0))

    def rho0(pos):
        if d < 4:
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
                'the C backend; see PLAN.md "Future tasks".')
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


class ParamDensity(DensityField):
    """DensityField backed by a .param's dICdens* profile (profiles 1/2/3)."""

    def __init__(self, p):
        self._fn = target_rho0(p)

    def rho0(self, pos):
        return self._fn(pos)


def density_from_param(p):
    """DensityField for the .param's density profile - the Python backend's
    automatic target, matching what the C gdicgen reads from the same file."""
    return ParamDensity(p)
