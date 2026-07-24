"""Balsara2 readout on a 2:1 composite AMR magnetic face field.

Every conforming topology patch receives one limited transverse slope shared by
both adjacent leaves.  Coarse leaves that touch fine patches use the same
2x2x2 virtual subdivision as the RT0 readout.  When a larger topology patch is
split virtually, its linear face polynomial is integrated onto the four
subfaces, preserving the original patch-average flux.  A local fixed-boundary
Hodge solve then fills divergence-free internal virtual faces.
"""

from dataclasses import dataclass

import numpy as np

from .magnetic_amr import (AMRFaceMagneticTable,
                           _project_fixed_boundary_box)
from .magnetic_balsara import (
    _limited_slope, _polynomial_coefficients_from_slopes,
    _polynomial_energy)


def _patch_transverse_slopes(topology, values, limiter):
    """Estimate one shared, limited transverse slope per topology patch.

    A local two-dimensional least-squares plane is essential at coarse/fine
    edges: centres of touching patches can be offset in *both* transverse
    coordinates, so independent one-dimensional differences would mix the two
    derivatives.  The limiter rescales the plane without rotating its gradient.
    """
    if limiter not in ("mc", "minmod", "central"):
        raise ValueError("Balsara2 limiter must be 'mc', 'minmod', or 'central'")
    values = np.asarray(values, dtype=np.float64)
    slopes = np.zeros((topology.nface, 3), dtype=np.float64)
    grid = topology.grid
    max_level = int(np.max(grid.cell_level))
    fine_dims = np.asarray(grid.base_npx, dtype=np.int64) * (2 ** max_level)
    fine_width = grid.box / fine_dims

    # Integer dyadic coordinates avoid fragile floating-point equality when
    # grouping coplanar patches and matching their transverse edges.
    centre_key = np.rint(
        (topology.centre - grid.lo) / fine_width).astype(np.int64)
    patch_lo = topology.centre - 0.5 * topology.patch_size
    patch_hi = topology.centre + 0.5 * topology.patch_size
    low_key = np.rint((patch_lo - grid.lo) / fine_width).astype(np.int64)
    high_key = np.rint((patch_hi - grid.lo) / fine_width).astype(np.int64)

    groups = {}
    for face in range(topology.nface):
        normal = int(topology.axis[face])
        groups.setdefault((normal, int(centre_key[face, normal])), []).append(face)

    for (normal, _plane), group in groups.items():
        transverse = [axis for axis in range(3) if axis != normal]
        ids = np.asarray(group, dtype=np.int64)
        if np.all(values[ids] == values[ids[0]]):
            continue
        local_index = {face: index for index, face in enumerate(group)}
        m00 = np.zeros(len(group), dtype=np.float64)
        m01 = np.zeros(len(group), dtype=np.float64)
        m11 = np.zeros(len(group), dtype=np.float64)
        r0 = np.zeros(len(group), dtype=np.float64)
        r1 = np.zeros(len(group), dtype=np.float64)
        neighbour_min = values[ids].copy()
        neighbour_max = values[ids].copy()
        has_u_lo = np.zeros(len(group), dtype=bool)
        has_u_hi = np.zeros(len(group), dtype=bool)
        has_v_lo = np.zeros(len(group), dtype=bool)
        has_v_hi = np.zeros(len(group), dtype=bool)
        for direction in transverse:
            other = transverse[1] if direction == transverse[0] else transverse[0]
            ending = {}
            starting = {}
            for face in group:
                ending.setdefault(int(high_key[face, direction]), []).append(face)
                starting.setdefault(int(low_key[face, direction]), []).append(face)

            for boundary in set(ending).intersection(starting):
                for left in ending[boundary]:
                    for right in starting[boundary]:
                        overlap_key = (min(high_key[left, other],
                                           high_key[right, other])
                                       - max(low_key[left, other],
                                             low_key[right, other]))
                        if overlap_key <= 0:
                            continue
                        weight = overlap_key * fine_width[other]
                        du = (topology.centre[right, transverse[0]]
                              - topology.centre[left, transverse[0]])
                        dv = (topology.centre[right, transverse[1]]
                              - topology.centre[left, transverse[1]])
                        db = values[right] - values[left]
                        li = local_index[left]
                        ri = local_index[right]
                        for index in (li, ri):
                            m00[index] += weight * du * du
                            m01[index] += weight * du * dv
                            m11[index] += weight * dv * dv
                            r0[index] += weight * du * db
                            r1[index] += weight * dv * db
                        neighbour_min[li] = min(neighbour_min[li], values[right])
                        neighbour_max[li] = max(neighbour_max[li], values[right])
                        neighbour_min[ri] = min(neighbour_min[ri], values[left])
                        neighbour_max[ri] = max(neighbour_max[ri], values[left])
                        if du > 0.0:
                            has_u_hi[li] = True
                            has_u_lo[ri] = True
                        elif du < 0.0:
                            has_u_lo[li] = True
                            has_u_hi[ri] = True
                        if dv > 0.0:
                            has_v_hi[li] = True
                            has_v_lo[ri] = True
                        elif dv < 0.0:
                            has_v_lo[li] = True
                            has_v_hi[ri] = True

        determinant = m00 * m11 - m01 ** 2
        scale = np.maximum(m00 * m11, np.finfo(float).tiny)
        full_rank = determinant > 128.0 * np.finfo(float).eps * scale
        selected_local = np.where(full_rank)[0]
        selected = ids[selected_local]
        det = determinant[full_rank]
        gradient_u = ((m11[selected_local] * r0[selected_local]
                       - m01[selected_local] * r1[selected_local]) / det)
        gradient_v = ((m00[selected_local] * r1[selected_local]
                       - m01[selected_local] * r0[selected_local]) / det)
        slopes[selected, transverse[0]] = gradient_u
        slopes[selected, transverse[1]] = gradient_v

        if limiter != "central" and len(selected):
            surrounded = (has_u_lo[selected_local] & has_u_hi[selected_local]
                          & has_v_lo[selected_local] & has_v_hi[selected_local])
            extent = (np.abs(gradient_u)
                      * 0.5 * topology.patch_size[selected, transverse[0]]
                      + np.abs(gradient_v)
                      * 0.5 * topology.patch_size[selected, transverse[1]])
            nonzero = surrounded & (extent > 0.0)
            factor = 1.0 if limiter == "minmod" else 2.0
            alpha = np.ones(len(selected), dtype=np.float64)
            alpha[nonzero] = np.minimum(
                1.0,
                factor * np.minimum(
                    neighbour_max[selected_local[nonzero]]
                    - values[selected[nonzero]],
                    values[selected[nonzero]]
                    - neighbour_min[selected_local[nonzero]])
                / extent[nonzero])
            slopes[selected, transverse[0]] *= alpha
            slopes[selected, transverse[1]] *= alpha
    return slopes


def _subdivisions(table):
    topology = table.topology
    grid = table.grid
    result = np.ones(grid.ncell, dtype=np.int64)
    for face in range(topology.nface):
        for cell in (int(topology.minus[face]), int(topology.plus[face])):
            if cell < 0:
                continue
            normal = int(topology.axis[face])
            transverse = [axis for axis in range(3) if axis != normal]
            ratios = [grid.cell_size[cell, axis]
                      / topology.patch_size[face, axis]
                      for axis in transverse]
            result[cell] = max(result[cell], int(round(max(ratios))))
    if np.any(result > 2):
        raise ValueError("AMR Balsara2 requires a 2:1-balanced leaf mesh")
    return result


def _divergence_coefficients(coefficients):
    c = coefficients
    return np.stack([
        c[..., 0, 1] + c[..., 1, 2] + c[..., 2, 3],
        2.0 * c[..., 0, 4] + c[..., 1, 4] + c[..., 2, 4],
        c[..., 0, 5] + 2.0 * c[..., 1, 5] + c[..., 2, 5],
        c[..., 0, 6] + c[..., 1, 6] + 2.0 * c[..., 2, 6],
    ], axis=-1)


def _coefficients_from_cell_faces(face_values, face_slopes, spacing):
    """Vectorized Balsara coefficients for unrelated whole AMR leaves."""
    face_values = np.asarray(face_values, dtype=np.float64)
    face_slopes = np.asarray(face_slopes, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    coefficients = np.zeros(face_values.shape[:-2] + (3, 7), dtype=np.float64)
    for component in range(3):
        coefficients[..., component, component + 1] = (
            (face_values[..., component, 1]
             - face_values[..., component, 0]) / spacing[..., component])
        for axis in range(3):
            if axis != component:
                coefficients[..., component, axis + 1] = 0.5 * (
                    face_slopes[..., component, 0, axis]
                    + face_slopes[..., component, 1, axis])

    axy = ((face_slopes[..., 0, 1, 1] - face_slopes[..., 0, 0, 1])
           / spacing[..., 0])
    axz = ((face_slopes[..., 0, 1, 2] - face_slopes[..., 0, 0, 2])
           / spacing[..., 0])
    bxy = ((face_slopes[..., 1, 1, 0] - face_slopes[..., 1, 0, 0])
           / spacing[..., 1])
    byz = ((face_slopes[..., 1, 1, 2] - face_slopes[..., 1, 0, 2])
           / spacing[..., 1])
    cxz = ((face_slopes[..., 2, 1, 0] - face_slopes[..., 2, 0, 0])
           / spacing[..., 2])
    cyz = ((face_slopes[..., 2, 1, 1] - face_slopes[..., 2, 0, 1])
           / spacing[..., 2])
    coefficients[..., 0, 5] = axy
    coefficients[..., 0, 6] = axz
    coefficients[..., 1, 4] = bxy
    coefficients[..., 1, 6] = byz
    coefficients[..., 2, 4] = cxz
    coefficients[..., 2, 5] = cyz
    quadratic = np.stack([
        -0.5 * (bxy + cxz),
        -0.5 * (axy + cyz),
        -0.5 * (axz + byz),
    ], axis=-1)
    coefficients[..., 0, 4] = quadratic[..., 0]
    coefficients[..., 1, 5] = quadratic[..., 1]
    coefficients[..., 2, 6] = quadratic[..., 2]
    half = 0.5 * spacing
    for component in range(3):
        coefficients[..., component, 0] = (
            0.5 * (face_values[..., component, 0]
                   + face_values[..., component, 1])
            - quadratic[..., component] * half[..., component] ** 2)
    return coefficients


@dataclass
class AMRBalsara2Reconstruction:
    """Second-order, flux-preserving readout on 2:1 open AMR leaves."""

    grid: object
    subdivisions: np.ndarray
    cell_coefficients: np.ndarray
    subdivided_coefficients: dict
    limiter: str
    virtual_divergence_max: float
    patch_flux_error_max: float

    @classmethod
    def from_face_table(cls, table, limiter="mc"):
        if not isinstance(table, AMRFaceMagneticTable):
            raise TypeError("expected an AMRFaceMagneticTable")
        topology = table.topology
        grid = table.grid
        subdivisions = _subdivisions(table)
        patch_slopes = _patch_transverse_slopes(
            topology, table.values, limiter)
        subdivided_cells = np.where(subdivisions > 1)[0]
        subdivided_set = set(int(cell) for cell in subdivided_cells)
        incident = {int(cell): [] for cell in subdivided_cells}
        for face in range(topology.nface):
            for cell in (int(topology.minus[face]), int(topology.plus[face])):
                if cell in subdivided_set:
                    incident[cell].append(face)

        cell_coefficients = np.zeros((grid.ncell, 3, 7), dtype=np.float64)
        subdivided_coefficients = {}
        cell_face_values = table.cell_faces()
        whole_cells = np.where(subdivisions == 1)[0]
        whole_slopes = np.zeros((grid.ncell, 3, 2, 3), dtype=np.float64)
        for normal in range(3):
            face_ids = np.where(topology.axis == normal)[0]
            minus = topology.minus[face_ids]
            plus = topology.plus[face_ids]
            use_minus = ((minus >= 0)
                         & (subdivisions[np.maximum(minus, 0)] == 1))
            use_plus = ((plus >= 0)
                        & (subdivisions[np.maximum(plus, 0)] == 1))
            whole_slopes[minus[use_minus], normal, 1] = patch_slopes[
                face_ids[use_minus]]
            whole_slopes[plus[use_plus], normal, 0] = patch_slopes[
                face_ids[use_plus]]
        cell_coefficients[whole_cells] = _coefficients_from_cell_faces(
            cell_face_values[whole_cells], whole_slopes[whole_cells],
            grid.cell_size[whole_cells])
        del whole_slopes
        whole_divergence = _divergence_coefficients(
            cell_coefficients[whole_cells])
        max_divergence = (float(np.max(np.abs(whole_divergence)))
                          if len(whole_cells) else 0.0)
        max_flux_error = 0.0
        for cell in subdivided_cells:
            cell = int(cell)
            n = int(subdivisions[cell])
            subsize = grid.cell_size[cell] / n
            arrays = (
                np.zeros((n + 1, n, n), dtype=np.float64),
                np.zeros((n, n + 1, n), dtype=np.float64),
                np.zeros((n, n, n + 1), dtype=np.float64),
            )
            assigned = [np.zeros_like(array, dtype=bool) for array in arrays]

            def patch_index(face):
                normal = int(topology.axis[face])
                transverse = [axis for axis in range(3) if axis != normal]
                normal_index = (n if int(topology.minus[face]) == cell else 0)
                face_lo = (topology.centre[face]
                           - 0.5 * topology.patch_size[face])
                face_hi = (topology.centre[face]
                           + 0.5 * topology.patch_size[face])
                ranges = []
                for axis in transverse:
                    start = int(round((face_lo[axis] - grid.cell_lo[cell, axis])
                                      / subsize[axis]))
                    stop = int(round((face_hi[axis] - grid.cell_lo[cell, axis])
                                     / subsize[axis]))
                    ranges.append((max(0, start), min(n, stop)))
                s0 = slice(*ranges[0])
                s1 = slice(*ranges[1])
                if normal == 0:
                    index = (normal_index, s0, s1)
                elif normal == 1:
                    index = (s0, normal_index, s1)
                else:
                    index = (s0, s1, normal_index)
                return normal, transverse, ranges, index

            for face in incident[cell]:
                normal, transverse, ranges, index = patch_index(face)
                axes0 = (grid.cell_lo[cell, transverse[0]]
                         + (np.arange(*ranges[0]) + 0.5)
                         * subsize[transverse[0]])
                axes1 = (grid.cell_lo[cell, transverse[1]]
                         + (np.arange(*ranges[1]) + 0.5)
                         * subsize[transverse[1]])
                q0, q1 = np.meshgrid(axes0, axes1, indexing="ij")
                subface_values = (
                    table.values[face]
                    + patch_slopes[face, transverse[0]]
                    * (q0 - topology.centre[face, transverse[0]])
                    + patch_slopes[face, transverse[1]]
                    * (q1 - topology.centre[face, transverse[1]]))
                arrays[normal][index] = subface_values
                assigned[normal][index] = True
                patch_area = topology.area[face]
                max_flux_error = max(
                    max_flux_error,
                    abs(float(np.mean(subface_values)) - table.values[face])
                    * patch_area)

            external = (
                np.concatenate([assigned[0][0].ravel(),
                                assigned[0][-1].ravel()]),
                np.concatenate([assigned[1][:, 0].ravel(),
                                assigned[1][:, -1].ravel()]),
                np.concatenate([assigned[2][:, :, 0].ravel(),
                                assigned[2][:, :, -1].ravel()]),
            )
            if not all(np.all(mask) for mask in external):
                raise RuntimeError("AMR Balsara2 cell boundary is incomplete")

            for i in range(1, n):
                fraction = i / n
                arrays[0][i] = ((1.0 - fraction) * arrays[0][0]
                                + fraction * arrays[0][n])
                arrays[1][:, i] = ((1.0 - fraction) * arrays[1][:, 0]
                                   + fraction * arrays[1][:, n])
                arrays[2][:, :, i] = (
                    (1.0 - fraction) * arrays[2][:, :, 0]
                    + fraction * arrays[2][:, :, n])
            local_divergence = _project_fixed_boundary_box(
                *arrays, subsize)
            max_divergence = max(
                max_divergence, float(np.max(np.abs(local_divergence))))

            slopes = []
            for normal, face_array in enumerate(arrays):
                slopes.append([
                    (np.zeros_like(face_array) if axis == normal else
                     _limited_slope(face_array, axis, subsize[axis],
                                    False, limiter))
                    for axis in range(3)
                ])
            # The topology-patch slope is the shared trace.  Override locally
            # estimated slopes on every external virtual face.
            for face in incident[cell]:
                normal, transverse, _ranges, index = patch_index(face)
                for direction in transverse:
                    slopes[normal][direction][index] = patch_slopes[
                        face, direction]

            coefficients = _polynomial_coefficients_from_slopes(
                arrays, slopes, subsize, periodic=False)
            max_divergence = max(
                max_divergence,
                float(np.max(np.abs(_divergence_coefficients(coefficients)))))
            subdivided_coefficients[cell] = coefficients

        return cls(grid, subdivisions, cell_coefficients,
                   subdivided_coefficients, limiter, max_divergence,
                   max_flux_error)

    def _locate(self, positions):
        pos = np.asarray(positions, dtype=np.float64)
        hi = self.grid.hi
        tol = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, self.grid.box)
        if np.any(pos < self.grid.lo - tol) or np.any(pos > hi + tol):
            raise ValueError("query position lies outside the AMR domain")
        clipped = np.minimum(np.maximum(pos, self.grid.lo),
                             np.nextafter(hi, self.grid.lo))
        levels = np.asarray(self.grid.cell_level, dtype=np.int64)
        centres = self.grid.centres()
        found = np.full(len(pos), -1, dtype=np.int64)
        for level in range(int(levels.max()) + 1):
            cells = np.where(levels == level)[0]
            if not len(cells):
                continue
            dims = np.asarray(self.grid.base_npx, dtype=np.int64) * (2 ** level)
            ijk = np.floor((centres[cells] - self.grid.lo)
                           / (self.grid.box / dims)).astype(np.int64)
            keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
            order = np.argsort(keys)
            keys = keys[order]
            cells = cells[order]
            unresolved = np.where(found < 0)[0]
            pijk = np.floor((clipped[unresolved] - self.grid.lo)
                            / (self.grid.box / dims)).astype(np.int64)
            pijk = np.minimum(np.maximum(pijk, 0), dims - 1)
            pkeys = ((pijk[:, 0] * dims[1] + pijk[:, 1])
                     * dims[2] + pijk[:, 2])
            where = np.searchsorted(keys, pkeys)
            safe = np.minimum(where, len(keys) - 1)
            hit = (where < len(keys)) & (keys[safe] == pkeys)
            found[unresolved[hit]] = cells[safe[hit]]
        if np.any(found < 0):
            raise RuntimeError("AMR leaf lookup found an uncovered point")
        return clipped, found

    def query(self, positions):
        pos = np.asarray(positions, dtype=np.float64)
        scalar = pos.ndim == 1
        pos = np.atleast_2d(pos)
        if pos.shape[1] != 3:
            raise ValueError("positions must have shape (N,3) or (3,)")
        clipped, cells = self._locate(pos)
        out = np.empty((len(pos), 3), dtype=np.float64)
        order = np.argsort(cells, kind="stable")
        cuts = np.flatnonzero(np.diff(cells[order])) + 1
        for take in np.split(order, cuts):
            cell = int(cells[take[0]])
            n = int(self.subdivisions[cell])
            spacing = self.grid.cell_size[cell] / n
            q = (clipped[take] - self.grid.cell_lo[cell]) / spacing
            index = np.minimum(np.floor(q).astype(np.int64), n - 1)
            local = (np.minimum(q - index, 1.0) - 0.5) * spacing
            if n == 1:
                coefficients = np.broadcast_to(
                    self.cell_coefficients[cell], (len(take), 3, 7))
            else:
                ix, iy, iz = index.T
                coefficients = self.subdivided_coefficients[cell][ix, iy, iz]
            x, y, z = local.T
            basis = np.stack([
                np.column_stack([np.ones(len(take)), x, y, z,
                                 x * x, x * y, x * z]),
                np.column_stack([np.ones(len(take)), x, y, z,
                                 x * y, y * y, y * z]),
                np.column_stack([np.ones(len(take)), x, y, z,
                                 x * z, y * z, z * z]),
            ], axis=1)
            out[take] = np.einsum("nci,nci->nc", coefficients, basis)
        return out[0] if scalar else out

    def magnetic_energy(self):
        total = 0.0
        whole = np.where(self.subdivisions == 1)[0]
        if len(whole):
            for level in np.unique(self.grid.cell_level[whole]):
                cells = whole[self.grid.cell_level[whole] == level]
                total += _polynomial_energy(
                    self.cell_coefficients[cells], self.grid.cell_size[cells[0]])
        for cell, coefficients in self.subdivided_coefficients.items():
            total += _polynomial_energy(
                coefficients, self.grid.cell_size[cell]
                / self.subdivisions[cell])
        return total


def build_amr_balsara2_reconstruction(table, limiter="mc"):
    return AMRBalsara2Reconstruction.from_face_table(table, limiter=limiter)
