"""Second-order divergence-free reconstruction from shared magnetic faces.

This is the Cartesian Balsara (2001/2004) reconstruction.  The normal field on
each face is represented by a limited linear polynomial in the two transverse
coordinates.  Inside every cell the three components use the seven-term basis

``Bx = a0 + ax*x + ay*y + az*z + axx*x^2 + axy*x*y + axz*x*z``

and cyclic permutations for ``By`` and ``Bz``.  Matching both face polynomials
and imposing the four polynomial divergence constraints gives a unique field.
Consequently the reconstruction preserves every face-average normal flux and
has pointwise divergence equal to the finite-volume face divergence.
"""

from dataclasses import dataclass

import numpy as np


_EXPONENTS = (
    # Bx: 1, x, y, z, x^2, xy, xz
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
     (2, 0, 0), (1, 1, 0), (1, 0, 1)),
    # By: 1, x, y, z, xy, y^2, yz
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (0, 2, 0), (0, 1, 1)),
    # Bz: 1, x, y, z, xz, yz, z^2
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 0, 1), (0, 1, 1), (0, 0, 2)),
)


def _minmod(*values):
    """Elementwise minmod of equally-shaped arrays."""
    stack = np.stack(values, axis=0)
    positive = np.all(stack > 0.0, axis=0)
    negative = np.all(stack < 0.0, axis=0)
    magnitude = np.min(np.abs(stack), axis=0)
    return np.where(positive, magnitude,
                    np.where(negative, -magnitude, 0.0))


def _limited_slope(values, axis, spacing, periodic, limiter):
    """TVD slope of face averages along one transverse coordinate."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[axis] <= 1:
        return np.zeros_like(values)
    if limiter not in ("mc", "minmod", "central"):
        raise ValueError("Balsara2 limiter must be 'mc', 'minmod', or 'central'")
    if periodic:
        previous = np.roll(values, 1, axis=axis)
        following = np.roll(values, -1, axis=axis)
        backward = (values - previous) / spacing
        forward = (following - values) / spacing
        centred = (following - previous) / (2.0 * spacing)
        if limiter == "central":
            return centred
        if limiter == "minmod":
            return _minmod(backward, forward)
        return _minmod(centred, 2.0 * backward, 2.0 * forward)

    # There is no transverse neighbour beyond an open physical edge.  Use the
    # unique one-sided derivative there; interior points retain the selected
    # limiter.  This is linear-exact and does not alter the face average.
    out = np.zeros_like(values)
    middle = [slice(None)] * values.ndim
    lower = [slice(None)] * values.ndim
    upper = [slice(None)] * values.ndim
    middle[axis] = slice(1, -1)
    lower[axis] = slice(0, -2)
    upper[axis] = slice(2, None)
    if values.shape[axis] > 2:
        vm = values[tuple(middle)]
        vp = values[tuple(upper)]
        vn = values[tuple(lower)]
        backward = (vm - vn) / spacing
        forward = (vp - vm) / spacing
        centred = (vp - vn) / (2.0 * spacing)
        if limiter == "central":
            out[tuple(middle)] = centred
        elif limiter == "minmod":
            out[tuple(middle)] = _minmod(backward, forward)
        else:
            out[tuple(middle)] = _minmod(
                centred, 2.0 * backward, 2.0 * forward)
    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    last = [slice(None)] * values.ndim
    penultimate = [slice(None)] * values.ndim
    first[axis] = 0
    second[axis] = 1
    last[axis] = -1
    penultimate[axis] = -2
    out[tuple(first)] = ((values[tuple(second)] - values[tuple(first)])
                         / spacing)
    out[tuple(last)] = ((values[tuple(last)] - values[tuple(penultimate)])
                        / spacing)
    return out


def _cell_sides(values, axis, periodic):
    """Return the lower/upper face arrays aligned with cell indices."""
    if periodic:
        return np.roll(values, 1, axis=axis), values
    lower = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
    upper = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
    return lower, upper


def _polynomial_coefficients_from_slopes(face_arrays, slopes, spacing,
                                         periodic):
    """Build Balsara coefficients from shared face values and face slopes."""
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    sides = []
    slope_sides = []
    for normal in range(3):
        sides.append(_cell_sides(face_arrays[normal], normal, periodic))
        slope_sides.append([
            _cell_sides(slopes[normal][axis], normal, periodic)
            for axis in range(3)
        ])
    shape = sides[0][0].shape
    coefficients = np.zeros(shape + (3, 7), dtype=np.float64)

    # Normal linear terms and mean transverse slopes.
    for component in range(3):
        lower, upper = sides[component]
        coefficients[..., component, component + 1] = (
            upper - lower) / spacing[component]
        for axis in range(3):
            if axis == component:
                continue
            slope_lower, slope_upper = slope_sides[component][axis]
            coefficients[..., component, axis + 1] = 0.5 * (
                slope_lower + slope_upper)

    # Mixed terms are the normal variation of a transverse face slope.
    axy = ((slope_sides[0][1][1] - slope_sides[0][1][0]) / spacing[0])
    axz = ((slope_sides[0][2][1] - slope_sides[0][2][0]) / spacing[0])
    bxy = ((slope_sides[1][0][1] - slope_sides[1][0][0]) / spacing[1])
    byz = ((slope_sides[1][2][1] - slope_sides[1][2][0]) / spacing[1])
    cxz = ((slope_sides[2][0][1] - slope_sides[2][0][0]) / spacing[2])
    cyz = ((slope_sides[2][1][1] - slope_sides[2][1][0]) / spacing[2])
    coefficients[..., 0, 5] = axy
    coefficients[..., 0, 6] = axz
    coefficients[..., 1, 4] = bxy
    coefficients[..., 1, 6] = byz
    coefficients[..., 2, 4] = cxz
    coefficients[..., 2, 5] = cyz

    # The three non-constant divergence constraints determine the pure
    # quadratic terms.  The remaining constant is then fixed by both face
    # means (which agree because x^2 has the same value on +/- faces).
    axx = -0.5 * (bxy + cxz)
    byy = -0.5 * (axy + cyz)
    czz = -0.5 * (axz + byz)
    coefficients[..., 0, 4] = axx
    coefficients[..., 1, 5] = byy
    coefficients[..., 2, 6] = czz
    half = 0.5 * spacing
    coefficients[..., 0, 0] = (0.5 * (sides[0][0] + sides[0][1])
                                - axx * half[0] ** 2)
    coefficients[..., 1, 0] = (0.5 * (sides[1][0] + sides[1][1])
                                - byy * half[1] ** 2)
    coefficients[..., 2, 0] = (0.5 * (sides[2][0] + sides[2][1])
                                - czz * half[2] ** 2)
    return np.ascontiguousarray(coefficients)


def _polynomial_coefficients(face_arrays, spacing, periodic, limiter):
    """Vectorized Balsara coefficients for a uniform rectangular mesh."""
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    slopes = []
    for normal, face in enumerate(face_arrays):
        component_slopes = []
        for axis in range(3):
            component_slopes.append(
                np.zeros_like(face) if axis == normal else
                _limited_slope(face, axis, spacing[axis], periodic, limiter))
        slopes.append(component_slopes)
    return _polynomial_coefficients_from_slopes(
        face_arrays, slopes, spacing, periodic)


def _polynomial_energy(coefficients, spacing):
    """Exact integral of ``0.5*|B|^2`` for an array of cell polynomials."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    half = 0.5 * np.asarray(spacing, dtype=np.float64).reshape(3)

    def moment(power, extent):
        if power % 2:
            return 0.0
        return 2.0 * extent ** (power + 1) / (power + 1)

    total = 0.0
    for component, exponents in enumerate(_EXPONENTS):
        gram = np.empty((7, 7), dtype=np.float64)
        for i, left in enumerate(exponents):
            for j, right in enumerate(exponents):
                powers = tuple(left[a] + right[a] for a in range(3))
                gram[i, j] = np.prod([
                    moment(powers[a], half[a]) for a in range(3)])
        coeff = coefficients[..., component, :]
        cell_integrals = np.einsum(
            "...i,ij,...j->...", coeff, gram, coeff)
        total += float(np.sum(cell_integrals))
    return 0.5 * total


@dataclass
class Balsara2Reconstruction:
    """Second-order point evaluator built from a uniform shared-face table."""

    lo: np.ndarray
    box: np.ndarray
    npx: tuple
    periodic: bool
    coefficients: np.ndarray
    limiter: str = "mc"

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.box = np.asarray(self.box, dtype=np.float64).reshape(3)
        self.npx = tuple(int(n) for n in self.npx)
        expected = self.npx + (3, 7)
        self.coefficients = np.ascontiguousarray(
            self.coefficients, dtype=np.float64)
        if self.coefficients.shape != expected:
            raise ValueError(
                f"Balsara2 coefficients have shape {self.coefficients.shape}, "
                f"expected {expected}")
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("Balsara2 coefficients contain non-finite values")

    @classmethod
    def from_face_table(cls, table, limiter="mc"):
        from .magnetic import FaceMagneticTable, OpenFaceMagneticTable

        if isinstance(table, FaceMagneticTable):
            periodic = True
        elif isinstance(table, OpenFaceMagneticTable):
            periodic = False
        else:
            raise TypeError(
                "Balsara2 uniform reconstruction requires a FaceMagneticTable "
                "or OpenFaceMagneticTable")
        faces = (table.Bx_face, table.By_face, table.Bz_face)
        coefficients = _polynomial_coefficients(
            faces, table.pixwidth, periodic, limiter)
        return cls(table.lo, table.box, table.npx, periodic,
                   coefficients, limiter=limiter)

    @property
    def pixwidth(self):
        return self.box / np.asarray(self.npx, dtype=np.float64)

    @property
    def cell_volume(self):
        return float(np.prod(self.pixwidth))

    def _locate(self, positions):
        pos = np.asarray(positions, dtype=np.float64)
        scalar = pos.ndim == 1
        pos = np.atleast_2d(pos)
        if pos.shape[1] != 3:
            raise ValueError("positions must have shape (N,3) or (3,)")
        if not np.all(np.isfinite(pos)):
            raise ValueError("positions contain non-finite values")
        dims = np.asarray(self.npx, dtype=np.int64)
        if self.periodic:
            q = np.mod((pos - self.lo) / self.pixwidth, dims)
            index = np.floor(q).astype(np.int64)
            frac = q - index
        else:
            hi = self.lo + self.box
            tol = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, self.box)
            if np.any(pos < self.lo - tol) or np.any(pos > hi + tol):
                raise ValueError(
                    "query position lies outside the open Balsara2 reconstruction")
            q = ((np.minimum(np.maximum(pos, self.lo), hi) - self.lo)
                 / self.pixwidth)
            index = np.minimum(np.floor(q).astype(np.int64), dims - 1)
            frac = np.minimum(q - index, 1.0)
        local = (frac - 0.5) * self.pixwidth
        return scalar, index, local

    def query(self, positions):
        """Evaluate the second-order divergence-free polynomial."""
        scalar, index, local = self._locate(positions)
        ix, iy, iz = index.T
        x, y, z = local.T
        basis = np.stack([
            np.column_stack([np.ones(len(local)), x, y, z,
                             x * x, x * y, x * z]),
            np.column_stack([np.ones(len(local)), x, y, z,
                             x * y, y * y, y * z]),
            np.column_stack([np.ones(len(local)), x, y, z,
                             x * z, y * z, z * z]),
        ], axis=1)
        coeff = self.coefficients[ix, iy, iz]
        out = np.einsum("nci,nci->nc", coeff, basis)
        return out[0] if scalar else out

    def divergence_coefficients(self):
        """Cell coefficients of ``div B = d0 + dx*x + dy*y + dz*z``."""
        c = self.coefficients
        return np.stack([
            c[..., 0, 1] + c[..., 1, 2] + c[..., 2, 3],
            2.0 * c[..., 0, 4] + c[..., 1, 4] + c[..., 2, 4],
            c[..., 0, 5] + 2.0 * c[..., 1, 5] + c[..., 2, 5],
            c[..., 0, 6] + c[..., 1, 6] + 2.0 * c[..., 2, 6],
        ], axis=-1)

    def divergence(self, positions):
        scalar, index, local = self._locate(positions)
        ix, iy, iz = index.T
        d = self.divergence_coefficients()[ix, iy, iz]
        out = d[:, 0] + np.einsum("ni,ni->n", d[:, 1:], local)
        return out[0] if scalar else out

    def cell_centred(self):
        """Exact cell-volume averages of the reconstructed field."""
        c = self.coefficients
        half = 0.5 * self.pixwidth
        return np.stack([
            c[..., 0, 0] + c[..., 0, 4] * half[0] ** 2 / 3.0,
            c[..., 1, 0] + c[..., 1, 5] * half[1] ** 2 / 3.0,
            c[..., 2, 0] + c[..., 2, 6] * half[2] ** 2 / 3.0,
        ], axis=-1)

    def face_averages(self):
        """Reconstruct the six normal face averages of every cell.

        Transverse-linear and mixed terms integrate to zero on a centred
        rectangular face.  This is useful for auditing that Balsara2 changed
        only the within-cell readout and not the finite-volume face fluxes.
        """
        c = self.coefficients
        half = 0.5 * self.pixwidth
        out = np.empty(self.npx + (3, 2), dtype=np.float64)
        for component in range(3):
            normal_linear = c[..., component, component + 1]
            pure_quadratic = c[..., component, 4 + component]
            common = (c[..., component, 0]
                      + pure_quadratic * half[component] ** 2)
            out[..., component, 0] = (
                common - normal_linear * half[component])
            out[..., component, 1] = (
                common + normal_linear * half[component])
        return out

    def magnetic_energy(self):
        """Exact ``integral 0.5*|B|^2 dV`` of the cell polynomials."""
        return _polynomial_energy(self.coefficients, self.pixwidth)


def build_balsara2_reconstruction(table, limiter="mc"):
    """Public constructor for the uniform Balsara2 readout."""
    return Balsara2Reconstruction.from_face_table(table, limiter=limiter)
