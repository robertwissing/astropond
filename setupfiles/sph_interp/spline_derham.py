"""Tensor-product B-spline De Rham complex on a uniform grid (periodic).

The structure-preserving ("compatible") readout for the magnetic
vector-potential mode (FIELD_RECONSTRUCTION_PLAN.md, "THB-spline De Rham
complex readout").  This module is the FOUNDATION rung (T-b on a uniform
periodic grid); the open-boundary and hierarchical-THB rungs build on it.

The idea in one line: differentiating a degree-p B-spline yields degree-(p-1)
B-splines, so a chain of spline spaces closes EXACTLY under grad/curl/div::

    S0 = (p,p,p)                         scalar potentials     (0-forms)
    S1 = {(p-1,p,p),(p,p-1,p),(p,p,p-1)} H(curl) vector fields (1-forms)
    S2 = {(p,p-1,p-1),(p-1,p,p-1),(p-1,p-1,p)} H(div) fields   (2-forms)
    S3 = (p-1,p-1,p-1)                    densities             (3-forms)

(the notation is the per-axis polynomial degree of each tensor-product
component).  We use two rungs: the captured vector potential A lives in S1,
and B = curl A lands EXACTLY in the divergence-conforming space S2, so
``div B == 0`` holds MACHINE-EXACTLY at the coefficient level (div curl == 0
is the discrete identity ``d d = 0``) and pointwise off the grid.

For uniform periodic B-splines every 1D space on ``n`` intervals has dimension
``n``; the exterior derivative in one direction is the circulant BACKWARD
DIFFERENCE ``d_j = (c_j - c_{j-1})/h`` (the exact uniform-knot spline
derivative, ``B_{i,p}' = B_{i,p-1} - B_{i+1,p-1}``).  The captured cell-centred
A is projected into S1 by per-axis FFT deconvolution of the B-spline
cell-centre sampling kernel (a Schoenberg-positive symbol, so invertible).

Reference: Buffa, Sangalli & Vazquez, isogeometric discrete differential
forms; Evans & Hughes, divergence-conforming B-splines.
"""

import numpy as np


# --------------------------------------------------------------------------
# Univariate uniform B-spline evaluation (de Boor / Cox-de Boor, unit knots)
# --------------------------------------------------------------------------

def _deboor_weights(frac, deg):
    """Nonzero degree-``deg`` uniform B-spline values at points whose local
    fractional coordinate within their knot span is ``frac`` in [0, 1).

    Returns ``N`` of shape ``(deg + 1, M)``; ``N[r]`` is basis
    ``B_{i-deg+r}`` in span ``i``.  Columns sum to 1 (partition of unity).
    Piegl & Tiller A2.2 specialised to uniform unit knots (left[j]=frac+j-1,
    right[j]=j-frac).
    """
    frac = np.ascontiguousarray(frac, dtype=np.float64).reshape(-1)
    m = frac.shape[0]
    N = np.zeros((deg + 1, m))
    N[0] = 1.0
    for j in range(1, deg + 1):
        saved = np.zeros(m)
        for r in range(j):
            right = float(r + 1) - frac          # right[r+1] = (r+1) - frac
            left = frac + float(j - r - 1)        # left[j-r]  = frac + (j-r-1)
            temp = N[r] / (right + left)
            N[r] = saved + right * temp
            saved = left * temp
        N[j] = saved
    return N


def _deboor_deriv_weights(frac, deg):
    """d/du of the ``deg+1`` covering degree-``deg`` uniform B-splines at local
    ``frac`` (scalar).  Uses ``N'_{i,p} = N_{i,p-1} - N_{i+1,p-1}`` (unit knots).
    Returns a (deg+1,) array aligned with :func:`_deboor_weights`."""
    if deg == 0:
        return np.zeros(1)
    nm = _deboor_weights(np.array([frac], dtype=np.float64), deg - 1)[:, 0]
    dN = np.zeros(deg + 1)
    for r in range(deg + 1):
        t1 = nm[r - 1] if r >= 1 else 0.0
        t2 = nm[r] if r <= deg - 1 else 0.0
        dN[r] = t1 - t2
    return dN


def _stencil_1d(u_cell, deg, n, periodic=True):
    """Weights + coefficient indices of the degree-``deg`` B-splines active at
    cell-unit coordinates ``u_cell`` (M,).  Returns ``(w, idx)`` each
    ``(deg + 1, M)``.  In span ``i`` the active coefficients are ``i-deg..i``
    (periodic wrap by ``% n``; clamped if not periodic)."""
    u = np.ascontiguousarray(u_cell, dtype=np.float64).reshape(-1)
    i = np.floor(u).astype(np.int64)
    frac = u - i
    w = _deboor_weights(frac, deg)                       # (deg+1, M)
    r = np.arange(deg + 1)[:, None]
    idx = i[None, :] - deg + r                           # (deg+1, M)
    if periodic:
        idx = np.mod(idx, n)
    else:
        idx = np.clip(idx, 0, n - 1)
    return w, idx


# --------------------------------------------------------------------------
# Cell-centre quasi-interpolation projector (periodic FFT deconvolution)
# --------------------------------------------------------------------------

def _centre_symbol(deg, n):
    """FFT of the circulant kernel that samples a degree-``deg`` periodic
    B-spline space at cell centres (x_k = k + 1/2).  Deconvolving with it maps
    cell-centre samples to spline coefficients that reproduce them exactly."""
    centres = np.arange(n) + 0.5
    w, idx = _stencil_1d(centres, deg, n, periodic=True)   # (deg+1, n)
    kern = np.zeros(n)
    for s in range(deg + 1):
        hit = idx[s] == 0
        kern[hit] += w[s][hit]
    return np.fft.fft(kern)


def _project_to_spline(samples, degs, npx):
    """Coefficients of the tensor-product spline (per-axis degree ``degs``)
    that interpolates the cell-centred ``samples`` (nx, ny, nz), periodic.

    The centre-sampling symbol of an ODD-degree B-spline space has a single
    null at the Nyquist bin of an even-``n`` axis (the half-sample
    interpolation null); we deconvolve with the pseudo-inverse there, dropping
    only that one unreliable highest-frequency grid mode (negligible for the
    smooth fields A carries).  Even degrees / odd ``n`` have no null."""
    fhat = np.fft.fftn(np.ascontiguousarray(samples, dtype=np.float64))
    for a in range(3):
        sym = _centre_symbol(degs[a], npx[a])
        eps = 1e-8 * np.max(np.abs(sym))
        inv = np.where(np.abs(sym) > eps, 1.0 / np.where(np.abs(sym) > eps,
                                                         sym, 1.0), 0.0)
        shape = [1, 1, 1]
        shape[a] = npx[a]
        fhat = fhat * inv.reshape(shape)
    return np.fft.ifftn(fhat).real


# --------------------------------------------------------------------------
# Coefficient-level exterior derivative (circulant backward difference)
# --------------------------------------------------------------------------

def _diff(coeffs, axis, h):
    """Exact uniform-knot spline derivative in coefficient space:
    ``(c_j - c_{j-1}) / h`` along ``axis`` (drops that axis' degree by one)."""
    return (coeffs - np.roll(coeffs, 1, axis=axis)) / h


def _A_component_degrees(p):
    return ((p - 1, p, p), (p, p - 1, p), (p, p, p - 1))


def _B_component_degrees(p):
    return ((p, p - 1, p - 1), (p - 1, p, p - 1), (p - 1, p - 1, p))


def curl_coeffs(Ax, Ay, Az, h):
    """B = curl A at the coefficient level (S1 -> S2).  Each output component
    is an ``n^3`` coefficient array in the divergence-conforming space."""
    Bx = _diff(Az, 1, h[1]) - _diff(Ay, 2, h[2])
    By = _diff(Ax, 2, h[2]) - _diff(Az, 0, h[0])
    Bz = _diff(Ay, 0, h[0]) - _diff(Ax, 1, h[1])
    return Bx, By, Bz


def div_coeffs(Bx, By, Bz, h):
    """div B at the coefficient level (S2 -> S3).  Identically zero for any
    ``B = curl A`` (the discrete ``div curl == 0``)."""
    return _diff(Bx, 0, h[0]) + _diff(By, 1, h[1]) + _diff(Bz, 2, h[2])


# --------------------------------------------------------------------------
# Field evaluation
# --------------------------------------------------------------------------

def _eval_scalar(coeffs, u_axes, degs, npx, periodic=True):
    """Evaluate a scalar tensor-product spline (per-axis degree ``degs``) at
    query points given as per-axis cell-unit coordinates ``u_axes`` (list of
    three (M,) arrays)."""
    w0, i0 = _stencil_1d(u_axes[0], degs[0], npx[0], periodic)
    w1, i1 = _stencil_1d(u_axes[1], degs[1], npx[1], periodic)
    w2, i2 = _stencil_1d(u_axes[2], degs[2], npx[2], periodic)
    m = u_axes[0].shape[0]
    val = np.zeros(m)
    for a in range(degs[0] + 1):
        for b in range(degs[1] + 1):
            wab = w0[a] * w1[b]
            iab0, iab1 = i0[a], i1[b]
            for c in range(degs[2] + 1):
                val += wab * w2[c] * coeffs[iab0, iab1, i2[c]]
    return val


# --------------------------------------------------------------------------
# The complex object + the readout
# --------------------------------------------------------------------------

class _GridSpec:
    """Minimal grid duck-type (lo / pixwidth / npx) for the padded open grid."""

    def __init__(self, lo, pixwidth, npx):
        self.lo = np.asarray(lo, dtype=np.float64).reshape(3)
        self._pw = np.asarray(pixwidth, dtype=np.float64).reshape(3)
        self.npx = tuple(int(n) for n in npx)

    @property
    def pixwidth(self):
        return self._pw


class SplineDeRham:
    """Periodic uniform B-spline De Rham complex on a :class:`UniformGrid`.

    ``degree`` is the top (S0) polynomial degree ``p`` (default cubic).  A is
    stored in S1, read back as ``B = curl A`` in the divergence-conforming S2
    space, so ``div B == 0`` exactly.
    """

    def __init__(self, grid, degree=3):
        if degree < 1:
            raise ValueError('degree must be >= 1')
        self.grid = grid
        self.p = int(degree)
        self.npx = tuple(int(n) for n in grid.npx)
        self.h = np.asarray(grid.pixwidth, dtype=np.float64)
        self.lo = np.asarray(grid.lo, dtype=np.float64)
        if min(self.npx) <= self.p:
            raise ValueError('need npx > degree on every axis')

    # -- projector: captured cell-centred A -> S1 coefficients --------------
    def project_A(self, cell_A):
        cell_A = np.ascontiguousarray(cell_A, dtype=np.float64)
        degA = _A_component_degrees(self.p)
        return [_project_to_spline(cell_A[..., c], degA[c], self.npx)
                for c in range(3)]

    # -- B = curl A coefficients (S2) --------------------------------------
    def curl_from_A(self, cell_A):
        Ax, Ay, Az = self.project_A(cell_A)
        return curl_coeffs(Ax, Ay, Az, self.h)

    def _u_axes(self, positions):
        pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        return [np.mod((pos[:, a] - self.lo[a]) / self.h[a], self.npx[a])
                for a in range(3)]

    def query(self, cell_A, positions, Bbar=None):
        """B = curl(spline interpolant of A) at ``positions``.  ``div B == 0``
        pointwise.  ``Bbar`` (3,) is added exactly (curl of ``1/2 Bbar x r``)."""
        Bx, By, Bz = self.curl_from_A(cell_A)
        degB = _B_component_degrees(self.p)
        u = self._u_axes(positions)
        Bxv = _eval_scalar(Bx, u, degB[0], self.npx)
        Byv = _eval_scalar(By, u, degB[1], self.npx)
        Bzv = _eval_scalar(Bz, u, degB[2], self.npx)
        B = np.stack([Bxv, Byv, Bzv], axis=1)
        if Bbar is not None:
            B = B + np.asarray(Bbar, dtype=np.float64).reshape(1, 3)
        return B

    def divergence_coeffs(self, cell_A):
        """div(curl A) at the coefficient level - machine-zero by construction
        (structure-preserving check)."""
        Bx, By, Bz = self.curl_from_A(cell_A)
        return div_coeffs(Bx, By, Bz, self.h)


def spline_derham_curl(cell_A, grid, positions, Bbar=None, degree=3,
                       periodic=True, pad=2):
    """Functional drop-in for :func:`vector_potential.curl_interp`, but the
    curl is taken in the spline De Rham complex (structure-preserving,
    higher-order, exactly divergence-free).

    ``periodic`` may be a scalar or a per-axis triple.  On a non-periodic
    (open) axis the captured A is zero-padded by a factor ``pad`` (>=2) and the
    periodic complex is run on the padded grid - the isolated/free-space
    embedding, matching :func:`bfield_to_vector_potential`'s open convention.
    curl annihilates any constant, so no mean removal is needed, and
    ``div B == 0`` still holds exactly at the interior particle positions
    (which fall in the un-padded low corner).  A must decay toward the capture
    boundary (a compact source) for the zero-pad to be accurate there."""
    per = (bool(periodic),) * 3 if np.isscalar(periodic) \
        else tuple(bool(p) for p in periodic)
    if all(per):
        return SplineDeRham(grid, degree=degree).query(
            cell_A, positions, Bbar=Bbar)

    cell_A = np.ascontiguousarray(cell_A, dtype=np.float64)
    npx = np.asarray(grid.npx, dtype=np.int64)
    factor = np.array([1 if per[a] else int(pad) for a in range(3)])
    if np.any(factor[~np.array(per)] < 2):
        raise ValueError('open-axis pad must be >= 2')
    npx_pad = npx * factor
    padded = np.zeros(tuple(npx_pad) + (3,), dtype=np.float64)
    padded[:npx[0], :npx[1], :npx[2], :] = cell_A     # data in the low corner
    gpad = _GridSpec(grid.lo, grid.pixwidth, npx_pad)
    return SplineDeRham(gpad, degree=degree).query(
        padded, positions, Bbar=Bbar)
