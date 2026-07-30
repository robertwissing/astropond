"""Shared fixtures for the IC-tool tests.

The scripts under test live in `setupfiles/` and import each other (and
`glassgen`) by path, so the path setup here mirrors what the scripts do for
themselves when run as executables.
"""
import os
import sys
import warnings

import numpy as np
import pytest

SETUPFILES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSUITE = os.path.dirname(SETUPFILES)
GLASSGEN = os.path.join(TESTSUITE, 'icgenerator', 'glassgen_python')
for p in (SETUPFILES, GLASSGEN):
    if p not in sys.path:
        sys.path.insert(0, p)

# readtipsy's writers open each file with mode 'w' then 'a' without closing the
# first handle: a quirk of that shared module, not of anything tested here.
warnings.filterwarnings('ignore', category=ResourceWarning)

_BE_F8 = '>f8'
_BE_F4 = '>f4'
_BE_I4 = '>i4'

# tipsy gas columns: mass, x,y,z, vx,vy,vz, rho, temp, hsmooth, metals, phi
NGAS_COLS = 12
NDARK_COLS = 9
NSTAR_COLS = 11


def write_tipsy(path, gas, dark=None, star=None, time=0.0):
    """Minimal big-endian tipsy writer (the layout readtipsy.readtipsy reads).

    Written here rather than through readtipsy.writetipsy so a failure in the
    writer cannot mask a failure in the tool under test - the fixtures stay
    independent of the code path they are used to exercise.
    """
    gas = np.asarray(gas, dtype=float)
    dark = np.zeros((0, NDARK_COLS)) if dark is None else np.asarray(dark, float)
    star = np.zeros((0, NSTAR_COLS)) if star is None else np.asarray(star, float)
    n = len(gas) + len(dark) + len(star)
    header = np.array([n, 3, len(gas), len(dark), len(star), 0])
    with open(path, 'wb') as fh:
        np.array([float(time)]).astype(_BE_F8).tofile(fh)
        header.astype(_BE_I4).tofile(fh)
        for arr in (gas, dark, star):
            if len(arr):
                arr.astype(_BE_F4).tofile(fh)


def write_vector_aux(path, label, vec):
    """Packed '<path>.<label>' plus the x/y/z twins, as the tools expect."""
    vec = np.asarray(vec, dtype=float)
    with open(path + '.' + label, 'wb') as fh:
        np.array([len(vec)]).astype(_BE_I4).tofile(fh)
        vec.ravel().astype(_BE_F4).tofile(fh)
    for i, comp in enumerate('xyz'):
        with open(f'{path}.{label}{comp}', 'wb') as fh:
            np.array([len(vec)]).astype(_BE_I4).tofile(fh)
            vec[:, i].astype(_BE_F4).tofile(fh)


def read_aux(path, label):
    """Read '<path>.<label>' -> flat float array (count header, then f4)."""
    with open(f'{path}.{label}', 'rb') as fh:
        np.fromfile(fh, dtype=_BE_I4, count=1)
        return np.fromfile(fh, dtype=_BE_F4).astype(float)


def read_vector_aux(path, label):
    """Read a packed vector aux -> (N, 3)."""
    return read_aux(path, label).reshape(-1, 3)


def read_tipsy_gas(path):
    """Read back a snapshot written by write_tipsy -> (gas, header)."""
    with open(path, 'rb') as fh:
        np.fromfile(fh, dtype=_BE_F8, count=1)
        header = np.fromfile(fh, dtype=_BE_I4, count=6)
        ngas = int(header[2])
        gas = np.fromfile(fh, dtype=_BE_F4,
                          count=ngas * NGAS_COLS).astype(float)
    return gas.reshape(ngas, NGAS_COLS), header


def read_tipsy_all(path):
    """Read back every family -> (gas, dark, star, header)."""
    with open(path, 'rb') as fh:
        np.fromfile(fh, dtype=_BE_F8, count=1)
        header = np.fromfile(fh, dtype=_BE_I4, count=6)
        ngas, ndark, nstar = (int(header[2]), int(header[3]), int(header[4]))
        out = []
        for n, ncol in ((ngas, NGAS_COLS), (ndark, NDARK_COLS),
                        (nstar, NSTAR_COLS)):
            a = np.fromfile(fh, dtype=_BE_F4, count=n * ncol).astype(float)
            out.append(a.reshape(n, ncol))
    return (*out, header)


def centre_of_mass(gas, dark, star):
    """Mass-weighted centre over every family (columns 0 = mass, 1:4 = pos)."""
    m = np.concatenate([a[:, 0] for a in (gas, dark, star) if len(a)])
    x = np.concatenate([a[:, 1:4] for a in (gas, dark, star) if len(a)])
    return np.average(x, axis=0, weights=m)


def make_disc(n=600, seed=0, radius=1.0, thickness=0.15, with_dark=True):
    """A small exponential-ish disc of gas (+ optional dark), tipsy columns."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    phi = rng.uniform(0, 2 * np.pi, n)
    gas = np.zeros((n, NGAS_COLS))
    gas[:, 0] = 1.0 / n
    gas[:, 1] = r * np.cos(phi)
    gas[:, 2] = r * np.sin(phi)
    gas[:, 3] = rng.normal(0.0, thickness, n)
    gas[:, 4:7] = rng.normal(0.0, 0.05, (n, 3))
    gas[:, 7] = 1.0 + np.exp(-r / (0.3 * radius))     # rho
    gas[:, 8] = 1.0e4                                 # temperature
    gas[:, 9] = 0.15 * radius                         # hsmooth
    gas[:, 11] = -(1.0 / (r + 0.1))                   # phi, min at the centre
    dark = None
    if with_dark:
        m = n // 4
        dark = np.zeros((m, NDARK_COLS))
        dark[:, 0] = 2.0 / n
        dark[:, 1:4] = rng.normal(0.0, radius, (m, 3))
        dark[:, 7] = 0.05
    return gas, dark


@pytest.fixture
def disc_snapshot(tmp_path):
    """A gas+dark snapshot on disk with a BField aux. -> path (str)."""
    gas, dark = make_disc()
    path = str(tmp_path / 'disc.00000')
    write_tipsy(path, gas, dark)
    rng = np.random.default_rng(1)
    n = len(gas) + len(dark)
    B = np.zeros((n, 3))
    B[:len(gas)] = 1e-3 * rng.normal(size=(len(gas), 3))
    write_vector_aux(path, 'BField', B)
    return path


@pytest.fixture
def periodic_snapshot(tmp_path):
    """A uniform periodic box of gas in [-0.5, 0.5)^3. -> (path, box_length)."""
    rng = np.random.default_rng(4)
    n = 800
    gas = np.zeros((n, NGAS_COLS))
    gas[:, 0] = 1.0 / n
    gas[:, 1:4] = rng.uniform(-0.5, 0.5, (n, 3))
    gas[:, 7] = 1.0
    gas[:, 8] = 1.0e4
    gas[:, 9] = 0.12
    path = str(tmp_path / 'box.00000')
    write_tipsy(path, gas)
    return path, 1.0
