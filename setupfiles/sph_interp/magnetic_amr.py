"""Open AMR constrained-transport face fluxes.

The leaf mesh is converted into a conforming set of rectangular face patches.
At a coarse/fine interface the four fine rectangles are the degrees of freedom;
the coarse cell's face value is their area-weighted restriction.  A finite-
volume Hodge projection on this face graph removes cell divergence while fixing
arbitrary compatible outer-boundary normal fluxes.

For particle readout, a coarse leaf touching fine patches is virtually split
2x2x2.  Its external virtual faces inherit the shared flux patches and a local
fixed-boundary Hodge solve supplies divergence-free internal faces.  RT0 is
then evaluated inside each virtual cell.  This is globally H(div)-conforming,
but remains lowest order: tangential jumps are legal and can be noisy under a
disordered-particle SPH divergence operator.
"""

from dataclasses import dataclass

import numpy as np

from .interpolate import GridResult, interpolate
from .magnetic import _component_budget
from .particles import Particles
from .targets import AMRGrid

# `_component_budget` is shared with the uniform capture path (magnetic.py), so
# the uniform and AMR audits report the same budget keys.


@dataclass
class AMRFaceTopology:
    """Conforming rectangular face patches for an open octree leaf mesh."""

    grid: AMRGrid
    axis: np.ndarray
    minus: np.ndarray
    plus: np.ndarray
    area: np.ndarray
    distance: np.ndarray
    centre: np.ndarray
    patch_size: np.ndarray

    @property
    def nface(self):
        return len(self.axis)

    @property
    def boundary(self):
        return (self.minus < 0) | (self.plus < 0)

    @classmethod
    def build(cls, grid):
        if not isinstance(grid, AMRGrid):
            raise TypeError("AMRFaceTopology requires an AMRGrid")
        if any(grid.periodic):
            raise NotImplementedError("AMR face topology currently supports open axes")
        if grid.ncell == 0:
            raise ValueError("AMR grid has no leaf cells")
        levels = np.asarray(grid.cell_level, dtype=np.int64)
        max_level = int(levels.max())
        fine_dims = np.asarray(grid.base_npx, dtype=np.int64) * (2 ** max_level)
        centres = grid.centres()
        dims = (np.asarray(grid.base_npx, dtype=np.int64)[None, :]
                * (2 ** levels)[:, None])
        ijk = np.floor((centres - grid.lo) / (grid.box / dims)).astype(np.int64)
        ijk = np.minimum(np.maximum(ijk, 0), dims - 1)

        low_maps = [dict() for _ in range(3)]
        high_maps = [dict() for _ in range(3)]

        def insert(maps, axis, plane, level, u, v, cell):
            by_level = maps[axis].setdefault(int(plane), {})
            level_map = by_level.setdefault(int(level), {})
            key = (int(u), int(v))
            if key in level_map:
                raise RuntimeError("duplicate AMR leaf face key")
            level_map[key] = int(cell)

        for cell in range(grid.ncell):
            level = int(levels[cell])
            scale = 2 ** (max_level - level)
            for axis in range(3):
                transverse = [a for a in range(3) if a != axis]
                u, v = int(ijk[cell, transverse[0]]), int(ijk[cell, transverse[1]])
                insert(low_maps, axis, ijk[cell, axis] * scale,
                       level, u, v, cell)
                insert(high_maps, axis, (ijk[cell, axis] + 1) * scale,
                       level, u, v, cell)

        face_axis = []
        face_minus = []
        face_plus = []
        face_area = []
        face_distance = []
        face_centre = []
        face_patch_size = []
        coverage = np.zeros((grid.ncell, 3, 2), dtype=np.float64)

        def find_ancestor(level_maps, level, u, v, include_equal):
            start = level if include_equal else level - 1
            for ancestor_level in range(start, -1, -1):
                values = level_maps.get(ancestor_level)
                if values is None:
                    continue
                shift = level - ancestor_level
                cell = values.get((u >> shift, v >> shift))
                if cell is not None:
                    return cell
            return None

        def emit(axis, minus, plus, patch_cell):
            transverse = [a for a in range(3) if a != axis]
            area = float(grid.cell_size[patch_cell, transverse[0]]
                         * grid.cell_size[patch_cell, transverse[1]])
            if minus >= 0 and plus >= 0:
                distance = float(centres[plus, axis] - centres[minus, axis])
                if distance <= 0.0:
                    raise RuntimeError("invalid AMR face orientation")
            else:
                distance = 0.0
            face_axis.append(axis)
            face_minus.append(minus)
            face_plus.append(plus)
            face_area.append(area)
            face_distance.append(distance)
            centre = centres[patch_cell].copy()
            if minus >= 0:
                centre[axis] = grid.cell_lo[minus, axis] + grid.cell_size[minus, axis]
            else:
                centre[axis] = grid.lo[axis]
            if plus < 0:
                centre[axis] = grid.hi[axis]
            patch_size = grid.cell_size[patch_cell].copy()
            patch_size[axis] = 0.0
            face_centre.append(centre)
            face_patch_size.append(patch_size)
            if minus >= 0:
                coverage[minus, axis, 1] += area
            if plus >= 0:
                coverage[plus, axis, 0] += area

        for axis in range(3):
            # Physical boundaries.  Orientation is always from minus to plus.
            for level, values in low_maps[axis].get(0, {}).items():
                for (_u, _v), cell in values.items():
                    emit(axis, -1, cell, cell)
            upper_plane = int(fine_dims[axis])
            for level, values in high_maps[axis].get(upper_plane, {}).items():
                for (_u, _v), cell in values.items():
                    emit(axis, cell, -1, cell)

            planes = sorted(set(high_maps[axis]).intersection(low_maps[axis]))
            for plane in planes:
                if plane == 0 or plane == upper_plane:
                    continue
                right_levels = high_maps[axis][plane]
                left_levels = low_maps[axis][plane]
                # A right-side face is the patch when the opposite face is at
                # the same or a coarser level.
                for level, values in right_levels.items():
                    for (u, v), minus in values.items():
                        plus = find_ancestor(left_levels, level, u, v, True)
                        if plus is not None:
                            emit(axis, minus, plus, minus)
                # Otherwise the left-side face is finer. Strict-ancestor lookup
                # avoids duplicating the equal-level matches emitted above.
                for level, values in left_levels.items():
                    for (u, v), plus in values.items():
                        minus = find_ancestor(right_levels, level, u, v, False)
                        if minus is not None:
                            emit(axis, minus, plus, plus)

        expected = np.empty_like(coverage)
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            a = grid.cell_size[:, transverse[0]] * grid.cell_size[:, transverse[1]]
            expected[:, axis, 0] = a
            expected[:, axis, 1] = a
        if not np.allclose(coverage, expected, rtol=2e-13, atol=1e-14):
            missing = float(np.max(np.abs(coverage - expected)))
            raise ValueError("AMR leaves do not form a complete face-connected "
                             f"domain (maximum uncovered face area {missing:.3e})")
        return cls(grid, np.asarray(face_axis, dtype=np.int8),
                   np.asarray(face_minus, dtype=np.int64),
                   np.asarray(face_plus, dtype=np.int64),
                   np.asarray(face_area, dtype=np.float64),
                   np.asarray(face_distance, dtype=np.float64),
                   np.asarray(face_centre, dtype=np.float64),
                   np.asarray(face_patch_size, dtype=np.float64))

    def initial_values(self, cell_B):
        """Distance-weight cell-centred normal B onto every shared patch."""
        cell_B = np.asarray(cell_B, dtype=np.float64)
        if cell_B.shape != (self.grid.ncell, 3):
            raise ValueError("cell_B must have shape (ncell,3)")
        values = np.zeros(self.nface, dtype=np.float64)
        interior = ~self.boundary
        ids = np.where(interior)[0]
        m = self.minus[ids]
        p = self.plus[ids]
        a = self.axis[ids]
        dm = 0.5 * self.grid.cell_size[m, a]
        dp = 0.5 * self.grid.cell_size[p, a]
        values[ids] = ((dp * cell_B[m, a] + dm * cell_B[p, a])
                       / (dm + dp))
        return values

    def cell_divergence(self, face_values):
        values = np.asarray(face_values, dtype=np.float64)
        if values.shape != (self.nface,):
            raise ValueError("face_values must have shape (nface,)")
        flux = values * self.area
        net = np.zeros(self.grid.ncell, dtype=np.float64)
        interior_minus = self.minus >= 0
        interior_plus = self.plus >= 0
        np.add.at(net, self.minus[interior_minus], flux[interior_minus])
        np.add.at(net, self.plus[interior_plus], -flux[interior_plus])
        return net / self.grid.volumes()

    def boundary_net_flux(self, face_values):
        """Return the outward flux integrated over all boundary patches."""
        values = np.asarray(face_values, dtype=np.float64)
        if values.shape != (self.nface,):
            raise ValueError("face_values must have shape (nface,)")
        low = self.boundary & (self.minus < 0)
        high = self.boundary & (self.plus < 0)
        return float(np.sum(values[high] * self.area[high])
                     - np.sum(values[low] * self.area[low]))

    def project(self, face_values, boundary_flux="balance", return_audit=False):
        """Hodge-project interior patches while preserving boundary traces.

        The default ``boundary_flux="balance"`` applies the minimum uniform
        outward-normal correction across all boundary patches, emits a warning,
        and records it in the optional audit return.  ``"error"`` strictly
        rejects a trace with nonzero total outward flux.
        """
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.linalg import cg, spsolve

        values = np.asarray(face_values, dtype=np.float64).copy()
        if boundary_flux not in ("error", "balance"):
            raise ValueError("boundary_flux must be 'error' or 'balance'")
        if not np.all(np.isfinite(values)):
            raise ValueError("face_values contain non-finite values")
        flux_before = self.boundary_net_flux(values)
        boundary_scale = float(np.sum(
            np.abs(values[self.boundary]) * self.area[self.boundary]))
        tolerance = (2048.0 * np.finfo(np.float64).eps
                     * max(boundary_scale, np.finfo(np.float64).tiny))
        outward_correction = 0.0
        if abs(flux_before) > tolerance:
            if boundary_flux == "error":
                raise ValueError(
                    "AMR normal boundary trace has nonzero net outward flux "
                    f"{flux_before:.16e} (tolerance {tolerance:.3e})")
            outward_correction = (
                -flux_before / float(np.sum(self.area[self.boundary])))
            import warnings
            warnings.warn(
                "balanced incompatible AMR magnetic boundary trace: "
                f"net outward flux {flux_before:.16e}, uniform outward-normal "
                f"correction {outward_correction:.16e}",
                RuntimeWarning, stacklevel=2)
            low = self.boundary & (self.minus < 0)
            high = self.boundary & (self.plus < 0)
            values[low] -= outward_correction
            values[high] += outward_correction
        flux_after_balance = self.boundary_net_flux(values)
        divergence = self.cell_divergence(values)
        volumes = self.grid.volumes()
        ids = np.where(~self.boundary)[0]
        m = self.minus[ids]
        p = self.plus[ids]
        weight = self.area[ids] / self.distance[ids]
        rows = np.concatenate([m, m, p, p])
        cols = np.concatenate([m, p, p, m])
        data = np.concatenate([weight, -weight, weight, -weight])
        matrix = coo_matrix((data, (rows, cols)),
                            shape=(self.grid.ncell, self.grid.ncell)).tocsr()
        rhs = -volumes * divergence
        phi = np.zeros(self.grid.ncell, dtype=np.float64)
        if self.grid.ncell > 1:
            reduced = matrix[1:, 1:]
            reduced_rhs = rhs[1:]
            # Direct sparse factorisation is convenient for small tests but
            # prohibitively memory hungry for production galaxy AMRs.  The
            # graph Laplacian is SPD after fixing one gauge cell, so large
            # systems use diagonally preconditioned conjugate gradients.
            if self.grid.ncell < 20_000:
                phi[1:] = spsolve(reduced, reduced_rhs)
                solver = "spsolve"
                solver_iterations = None
            else:
                diagonal = reduced.diagonal()
                inverse_diagonal = np.zeros_like(diagonal)
                nonzero = diagonal != 0.0
                inverse_diagonal[nonzero] = 1.0 / diagonal[nonzero]
                iterations = [0]

                def count_iteration(_value):
                    iterations[0] += 1

                solution, info = cg(
                    reduced, reduced_rhs, M=diags(inverse_diagonal),
                    rtol=2.0e-11, atol=0.0, maxiter=10_000,
                    callback=count_iteration)
                if info != 0:
                    raise RuntimeError(
                        "AMR Hodge CG did not converge "
                        f"(info={info}, iterations={iterations[0]})")
                phi[1:] = solution
                solver = "cg-jacobi"
                solver_iterations = int(iterations[0])
        else:
            solver = "none"
            solver_iterations = None
        values[ids] -= (phi[p] - phi[m]) / self.distance[ids]
        audit = {
            "flux_policy": boundary_flux,
            "net_flux_before": flux_before,
            "net_flux_after_balance": flux_after_balance,
            "uniform_outward_correction": outward_correction,
            "compatibility_tolerance": tolerance,
            "linear_solver": solver,
            "linear_solver_iterations": solver_iterations,
        }
        if return_audit:
            return values, phi, audit
        return values, phi


@dataclass
class AMRFaceMagneticTable:
    """Divergence-free values on an :class:`AMRFaceTopology`."""

    topology: AMRFaceTopology
    values: np.ndarray

    def __post_init__(self):
        self.values = np.ascontiguousarray(self.values, dtype=np.float64)
        if self.values.shape != (self.topology.nface,):
            raise ValueError("AMR face value count does not match topology")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("AMR face values contain non-finite values")

    @property
    def grid(self):
        return self.topology.grid

    def cell_divergence(self):
        return self.topology.cell_divergence(self.values)

    def cell_faces(self):
        """Area-restricted left/right normal values for every leaf cell."""
        sums = np.zeros((self.grid.ncell, 3, 2), dtype=np.float64)
        for face in range(self.topology.nface):
            axis = int(self.topology.axis[face])
            weighted = self.values[face] * self.topology.area[face]
            minus = int(self.topology.minus[face])
            plus = int(self.topology.plus[face])
            if minus >= 0:
                sums[minus, axis, 1] += weighted
            if plus >= 0:
                sums[plus, axis, 0] += weighted
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            area = (self.grid.cell_size[:, transverse[0]]
                    * self.grid.cell_size[:, transverse[1]])
            sums[:, axis, 0] /= area
            sums[:, axis, 1] /= area
        return sums

    def cell_centred(self):
        faces = self.cell_faces()
        return 0.5 * (faces[:, :, 0] + faces[:, :, 1])

    def restriction_error(self):
        """Flux mismatch after restricting patches onto each whole cell face."""
        faces = self.cell_faces()
        maximum = 0.0
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            area = (self.grid.cell_size[:, transverse[0]]
                    * self.grid.cell_size[:, transverse[1]])
            for side in range(2):
                expected = faces[:, axis, side] * area
                actual = np.zeros(self.grid.ncell, dtype=np.float64)
                for face in range(self.topology.nface):
                    if int(self.topology.axis[face]) != axis:
                        continue
                    cell = (int(self.topology.plus[face]) if side == 0
                            else int(self.topology.minus[face]))
                    if cell >= 0:
                        actual[cell] += (self.values[face]
                                         * self.topology.area[face])
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        return maximum

    def hdiv_reconstruction(self):
        """Build a divergence-preserving virtual-cell RT0 reconstruction."""
        return AMRHDivReconstruction.from_face_table(self)

    def balsara2_reconstruction(self, limiter="mc"):
        """Build the flux-preserving second-order composite-AMR readout."""
        from .magnetic_amr_balsara import build_amr_balsara2_reconstruction
        return build_amr_balsara2_reconstruction(self, limiter=limiter)


def _project_fixed_boundary_box(Bxf, Byf, Bzf, spacing):
    """Project interior box faces while preserving all six boundary traces."""
    from scipy.fft import dctn, idctn

    nx, ny, nz = Bxf.shape[0] - 1, Byf.shape[1] - 1, Bzf.shape[2] - 1
    dx, dy, dz = np.asarray(spacing, dtype=np.float64)

    def divergence():
        return ((Bxf[1:] - Bxf[:-1]) / dx
                + (Byf[:, 1:] - Byf[:, :-1]) / dy
                + (Bzf[:, :, 1:] - Bzf[:, :, :-1]) / dz)

    if nx == ny == nz == 1:
        return divergence()
    D = divergence()
    kx = np.arange(nx, dtype=np.float64)
    ky = np.arange(ny, dtype=np.float64)
    kz = np.arange(nz, dtype=np.float64)
    lam = (-4.0 * np.sin(0.5 * np.pi * kx / nx)[:, None, None] ** 2 / dx ** 2
           -4.0 * np.sin(0.5 * np.pi * ky / ny)[None, :, None] ** 2 / dy ** 2
           -4.0 * np.sin(0.5 * np.pi * kz / nz)[None, None, :] ** 2 / dz ** 2)
    transformed = dctn(D, type=2, norm="ortho")
    lam[0, 0, 0] = 1.0
    transformed[0, 0, 0] = 0.0
    phi = idctn(transformed / lam, type=2, norm="ortho")
    Bxf[1:nx] -= (phi[1:] - phi[:-1]) / dx
    Byf[:, 1:ny] -= (phi[:, 1:] - phi[:, :-1]) / dy
    Bzf[:, :, 1:nz] -= (phi[:, :, 1:] - phi[:, :, :-1]) / dz
    return divergence()


@dataclass
class AMRHDivReconstruction:
    """Piecewise RT0 field on divergence-preserving virtual leaf subdivisions."""

    grid: AMRGrid
    subdivisions: np.ndarray
    cell_face_values: np.ndarray
    subdivided_face_fields: dict
    virtual_divergence_max: float

    @classmethod
    def from_face_table(cls, table):
        if not isinstance(table, AMRFaceMagneticTable):
            raise TypeError("expected an AMRFaceMagneticTable")
        topology = table.topology
        grid = table.grid
        subdivisions = np.ones(grid.ncell, dtype=np.int64)
        for face in range(topology.nface):
            for cell in (int(topology.minus[face]), int(topology.plus[face])):
                if cell < 0:
                    continue
                axis = int(topology.axis[face])
                transverse = [a for a in range(3) if a != axis]
                ratios = [grid.cell_size[cell, a] / topology.patch_size[face, a]
                          for a in transverse]
                subdivisions[cell] = max(subdivisions[cell],
                                         int(round(max(ratios))))
        if np.any(subdivisions > 2):
            jump = int(subdivisions.max())
            raise ValueError("AMR H(div) readout requires a 2:1-balanced leaf "
                             f"mesh; encountered a face subdivision factor {jump}")

        # Whole-leaf RT0 needs only six compact values per cell.  Materialise
        # 2x2x2 face arrays solely for coarse leaves touching finer patches;
        # keeping three NumPy arrays for every production leaf costs hundreds
        # of MB in Python-object overhead.
        cell_face_values = table.cell_faces()
        subdivided_cells = np.where(subdivisions > 1)[0]
        subdivided_set = set(int(cell) for cell in subdivided_cells)
        incident = {int(cell): [] for cell in subdivided_cells}
        for face in range(topology.nface):
            for cell in (int(topology.minus[face]), int(topology.plus[face])):
                if cell in subdivided_set:
                    incident[cell].append(face)

        fields = {}
        leaf_divergence = table.cell_divergence()
        max_divergence = float(np.max(np.abs(leaf_divergence)))
        for cell in subdivided_cells:
            cell = int(cell)
            n = int(subdivisions[cell])
            Bxf = np.zeros((n + 1, n, n), dtype=np.float64)
            Byf = np.zeros((n, n + 1, n), dtype=np.float64)
            Bzf = np.zeros((n, n, n + 1), dtype=np.float64)
            arrays = (Bxf, Byf, Bzf)
            assigned = [np.zeros_like(a, dtype=bool) for a in arrays]
            subsize = grid.cell_size[cell] / n
            for face in incident[cell]:
                axis = int(topology.axis[face])
                transverse = [a for a in range(3) if a != axis]
                normal_index = (n if int(topology.minus[face]) == cell else 0)
                patch_lo = topology.centre[face] - 0.5 * topology.patch_size[face]
                patch_hi = topology.centre[face] + 0.5 * topology.patch_size[face]
                ranges = []
                for a in transverse:
                    start = int(round((patch_lo[a] - grid.cell_lo[cell, a])
                                      / subsize[a]))
                    stop = int(round((patch_hi[a] - grid.cell_lo[cell, a])
                                     / subsize[a]))
                    ranges.append((max(0, start), min(n, stop)))
                s0 = slice(ranges[0][0], ranges[0][1])
                s1 = slice(ranges[1][0], ranges[1][1])
                if axis == 0:
                    index = (normal_index, s0, s1)
                elif axis == 1:
                    index = (s0, normal_index, s1)
                else:
                    index = (s0, s1, normal_index)
                previous = assigned[axis][index]
                if np.any(previous):
                    old = arrays[axis][index][previous]
                    if not np.allclose(old, table.values[face], rtol=0.0, atol=1e-13):
                        raise RuntimeError("conflicting AMR external subface values")
                arrays[axis][index] = table.values[face]
                assigned[axis][index] = True

            external_masks = (
                np.concatenate([assigned[0][0].ravel(), assigned[0][-1].ravel()]),
                np.concatenate([assigned[1][:, 0].ravel(), assigned[1][:, -1].ravel()]),
                np.concatenate([assigned[2][:, :, 0].ravel(),
                                assigned[2][:, :, -1].ravel()]),
            )
            if not all(np.all(mask) for mask in external_masks):
                raise RuntimeError("AMR virtual-cell boundary was not fully covered")

            for i in range(1, n):
                t = i / n
                Bxf[i] = (1.0 - t) * Bxf[0] + t * Bxf[n]
                Byf[:, i] = (1.0 - t) * Byf[:, 0] + t * Byf[:, n]
                Bzf[:, :, i] = (1.0 - t) * Bzf[:, :, 0] + t * Bzf[:, :, n]
            local_div = _project_fixed_boundary_box(
                Bxf, Byf, Bzf, grid.cell_size[cell] / n)
            max_divergence = max(max_divergence,
                                 float(np.max(np.abs(local_div))))
            fields[cell] = (Bxf, Byf, Bzf)
        return cls(grid, subdivisions, cell_face_values, fields,
                   max_divergence)

    def _locate(self, positions):
        pos = np.asarray(positions, dtype=np.float64)
        hi = self.grid.hi
        tol = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, self.grid.box)
        if np.any(pos < self.grid.lo - tol) or np.any(pos > hi + tol):
            raise ValueError("query position lies outside the AMR domain")
        clipped = np.minimum(np.maximum(pos, self.grid.lo),
                             np.nextafter(hi, self.grid.lo))
        levels = np.asarray(self.grid.cell_level, dtype=np.int64)
        all_centres = self.grid.centres()
        found = np.full(len(pos), -1, dtype=np.int64)
        for level in range(int(levels.max()) + 1):
            cells = np.where(levels == level)[0]
            if cells.size == 0:
                continue
            dims = np.asarray(self.grid.base_npx, dtype=np.int64) * (2 ** level)
            centres = all_centres[cells]
            cijk = np.floor((centres - self.grid.lo)
                            / (self.grid.box / dims)).astype(np.int64)
            cijk = np.minimum(np.maximum(cijk, 0), dims - 1)
            ckeys = (cijk[:, 0] * dims[1] + cijk[:, 1]) * dims[2] + cijk[:, 2]
            order = np.argsort(ckeys)
            ckeys = ckeys[order]
            cells = cells[order]
            unresolved = np.where(found < 0)[0]
            pijk = np.floor((clipped[unresolved] - self.grid.lo)
                            / (self.grid.box / dims)).astype(np.int64)
            pijk = np.minimum(np.maximum(pijk, 0), dims - 1)
            pkeys = (pijk[:, 0] * dims[1] + pijk[:, 1]) * dims[2] + pijk[:, 2]
            where = np.searchsorted(ckeys, pkeys)
            safe = np.minimum(where, len(ckeys) - 1)
            hit = (where < len(ckeys)) & (ckeys[safe] == pkeys)
            found[unresolved[hit]] = cells[safe[hit]]
        if np.any(found < 0):
            raise RuntimeError("AMR leaf lookup found an uncovered point")
        return clipped, found

    def query(self, positions):
        """Evaluate the globally H(div)-conforming virtual-cell RT0 field."""
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
            q = ((clipped[take] - self.grid.cell_lo[cell])
                 / (self.grid.cell_size[cell] / n))
            index = np.minimum(np.floor(q).astype(np.int64), n - 1)
            frac = np.minimum(q - index, 1.0)
            ix, iy, iz = index.T
            tx, ty, tz = frac.T
            if n == 1:
                faces = self.cell_face_values[cell]
                out[take, 0] = faces[0, 0] + tx * (
                    faces[0, 1] - faces[0, 0])
                out[take, 1] = faces[1, 0] + ty * (
                    faces[1, 1] - faces[1, 0])
                out[take, 2] = faces[2, 0] + tz * (
                    faces[2, 1] - faces[2, 0])
            else:
                Bxf, Byf, Bzf = self.subdivided_face_fields[int(cell)]
                out[take, 0] = (Bxf[ix, iy, iz]
                                + tx * (Bxf[ix + 1, iy, iz]
                                        - Bxf[ix, iy, iz]))
                out[take, 1] = (Byf[ix, iy, iz]
                                + ty * (Byf[ix, iy + 1, iz]
                                        - Byf[ix, iy, iz]))
                out[take, 2] = (Bzf[ix, iy, iz]
                                + tz * (Bzf[ix, iy, iz + 1]
                                        - Bzf[ix, iy, iz]))
        return out[0] if scalar else out

    def magnetic_energy(self):
        """Exact ``integral 0.5*|B|^2 dV`` of the virtual RT0 field."""
        total = 0.0
        whole = np.where(self.subdivisions == 1)[0]
        if whole.size:
            faces = self.cell_face_values[whole]
            mean_square = np.sum(
                (faces[:, :, 0] ** 2
                 + faces[:, :, 0] * faces[:, :, 1]
                 + faces[:, :, 1] ** 2) / 3.0,
                axis=1)
            total += float(np.sum(self.grid.volumes()[whole] * mean_square))
        for cell, arrays in self.subdivided_face_fields.items():
            n = int(self.subdivisions[cell])
            virtual_volume = float(self.grid.volumes()[cell]) / n ** 3
            local = 0.0
            for face, axis in zip(arrays, range(3)):
                left = np.take(face, np.arange(n), axis=axis)
                right = np.take(face, np.arange(1, n + 1), axis=axis)
                local += float(np.sum(
                    (left * left + left * right + right * right) / 3.0))
            total += virtual_volume * local
        return 0.5 * total


def capture_open_amr_face_field(particles, grid, fields=("Bx", "By", "Bz"),
                                background=None, normal_trace=None,
                                boundary_flux="balance",
                                allow_clipped_support=False):
    """Petkova capture followed by a fixed-normal composite-AMR Hodge solve.

    ``background="particle-mean"`` removes the particle-volume-weighted mean
    before capture/projection and restores it on every face. This exactly
    represents constant box-threading fields while leaving a general residual
    for the fixed-trace solve. ``normal_trace=None`` retains zero residual
    outer flux. ``"nearest"`` extrapolates the adjacent residual cell's normal
    component. A callable is evaluated at boundary patch centres and may return
    either full vectors or normal components; an array may likewise contain
    one value or one vector per boundary patch.

    Production magnetic captures should tile the complete compact kernel
    support. ``allow_clipped_support=True`` is an explicit diagnostic escape
    hatch for a pre-existing density/relax AMR whose domain is only the
    particle bounding box; clipping remains visible in the conservation audit.
    """
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if not isinstance(grid, AMRGrid):
        raise TypeError("open AMR magnetic capture requires an AMRGrid")
    if any(particles.periodic) or any(grid.periodic):
        raise NotImplementedError("open AMR magnetic capture requires open axes")
    if len(fields) != 3:
        raise ValueError("fields must contain the three magnetic components")
    if np.any(particles.rho <= 0.0) or not np.all(np.isfinite(particles.rho)):
        raise ValueError("magnetic B/rho carriers require finite positive rho")
    support_lo = np.min(particles.pos - 2.0 * particles.h[:, None], axis=0)
    support_hi = np.max(particles.pos + 2.0 * particles.h[:, None], axis=0)
    tol = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, grid.box)
    support_clipped = bool(
        np.any(grid.lo > support_lo + tol)
        or np.any(grid.hi < support_hi - tol))
    if support_clipped and not allow_clipped_support:
        raise ValueError("open AMR magnetic grid clips compact SPH support")

    names = []
    source = []
    for axis, field in zip("xyz", fields):
        if field not in particles.values:
            raise KeyError(f"particle magnetic component {field!r} is missing")
        component = np.asarray(particles.values[field], dtype=np.float64)
        key = f"__amr_magnetic_B{axis}_over_rho"
        names.append(key)
        source.append(component)
    source_B = np.stack(source, axis=1)
    particle_volume = particles.mass / particles.rho
    source_integral = np.sum(particle_volume[:, None] * source_B, axis=0)
    source_volume = float(particle_volume.sum())
    particle_background = np.zeros(3, dtype=np.float64)
    deposited_B = source_B
    particle_mean_background = (
        isinstance(background, str) and background == "particle-mean")
    if background is not None and not particle_mean_background:
        raise ValueError("AMR background must be None or 'particle-mean'")
    if particle_mean_background:
        particle_background = source_integral / source_volume
        deposited_B = source_B - particle_background[None, :]

    carrier_values = {
        key: component / particles.rho
        for key, component in zip(names, deposited_B.T)
    }
    volume_key = "__amr_magnetic_volume_over_mass"
    carrier_values[volume_key] = 1.0 / particles.rho
    carriers = Particles(particles.pos, particles.mass, particles.rho,
                         particles.h, carrier_values, particles.box,
                         particles.periodic)
    deposited = interpolate(carriers, grid, method="petkova",
                            values=names + [volume_key],
                            conservative_fields=True)
    volumes = grid.volumes()
    contents = [np.asarray(deposited.data[f"_content_{name}"], dtype=np.float64)
                for name in names]
    raw_residual_B = np.stack(
        [content / volumes for content in contents], axis=1)
    topology = AMRFaceTopology.build(grid)
    initial = topology.initial_values(raw_residual_B)
    if normal_trace is not None:
        boundary_ids = np.where(topology.boundary)[0]
        axes = topology.axis[boundary_ids].astype(np.int64)
        if isinstance(normal_trace, str):
            if normal_trace != "nearest":
                raise ValueError("AMR normal_trace string must be 'nearest'")
            cells = np.where(topology.minus[boundary_ids] >= 0,
                             topology.minus[boundary_ids],
                             topology.plus[boundary_ids])
            boundary_values = raw_residual_B[cells, axes]
        else:
            supplied = (normal_trace(topology.centre[boundary_ids])
                        if callable(normal_trace) else normal_trace)
            supplied = np.asarray(supplied, dtype=np.float64)
            if supplied.shape == (len(boundary_ids), 3):
                boundary_values = supplied[np.arange(len(boundary_ids)), axes]
            elif supplied.shape == (len(boundary_ids),):
                boundary_values = supplied
            else:
                raise ValueError(
                    "AMR normal_trace must provide (nboundary,) normal values "
                    "or (nboundary,3) field vectors")
        if particle_mean_background and not isinstance(normal_trace, str):
            boundary_values = boundary_values - particle_background[axes]
        if not np.all(np.isfinite(boundary_values)):
            raise ValueError("AMR normal_trace contains non-finite values")
        initial[boundary_ids] = boundary_values
    projected, _phi, boundary_audit = topology.project(
        initial, boundary_flux=boundary_flux, return_audit=True)
    projected_residual = projected.copy()
    if particle_mean_background:
        projected += particle_background[topology.axis]
    table = AMRFaceMagneticTable(topology, projected)
    faces = table.cell_faces()
    centred = table.cell_centred()
    div = table.cell_divergence()

    raw_B = raw_residual_B + particle_background
    data = {field: raw_B[:, component]
            for component, field in enumerate(fields)}
    data.update({"Bx_l": faces[:, 0, 0], "Bx_r": faces[:, 0, 1],
                 "By_l": faces[:, 1, 0], "By_r": faces[:, 1, 1],
                 "Bz_l": faces[:, 2, 0], "Bz_r": faces[:, 2, 1],
                 "Bx_c": centred[:, 0], "By_c": centred[:, 1],
                 "Bz_c": centred[:, 2], "divB": div})
    result = GridResult(data, None, grid, "petkova-B-over-rho-open-amr")

    deposited_volume = float(np.asarray(
        deposited.data[f"_content_{volume_key}"], dtype=np.float64).sum())
    residual_raw_integral = np.asarray(
        [content.sum() for content in contents])
    material_raw_integral = (
        residual_raw_integral + particle_background * deposited_volume)
    geometric_raw_integral = (
        residual_raw_integral
        + particle_background * float(np.sum(volumes)))
    projected_integral = np.sum(centred * volumes[:, None], axis=0)
    level_jump = ((topology.minus >= 0) & (topology.plus >= 0)
                  & (grid.cell_level[np.maximum(topology.minus, 0)]
                     != grid.cell_level[np.maximum(topology.plus, 0)]))
    audit = {
        "schema": "sph-interp-open-amr-face-magnetic-v1",
        "boundary_condition": (
            ("fixed-particle-mean-normal" if normal_trace is None
             else "fixed-spatial-normal-plus-particle-mean")
            if particle_mean_background else
            ("fixed-zero-normal" if normal_trace is None
             else "fixed-spatial-normal")),
        "boundary_background": particle_background.tolist(),
        "boundary_projection": boundary_audit,
        "support_lo": support_lo.tolist(),
        "support_hi": support_hi.tolist(),
        "support_clipped": support_clipped,
        "nleaf": int(grid.ncell),
        "nface_patches": int(topology.nface),
        "coarse_fine_patches": int(np.count_nonzero(level_jump)),
        "particle_to_raw_integral_B": _component_budget(
            source_integral, material_raw_integral),
        "raw_to_projected_cell_integral_B": _component_budget(
            geometric_raw_integral, projected_integral),
        "residual_raw_integral_B": residual_raw_integral.tolist(),
        "geometric_raw_integral_B": geometric_raw_integral.tolist(),
        "particle_volume": source_volume,
        "deposited_volume": deposited_volume,
        "volume_relative_error": ((deposited_volume - source_volume) / source_volume
                                  if source_volume != 0.0 else None),
        "hodge_face_correction_rms": float(np.sqrt(np.mean(
            (projected_residual - initial) ** 2))),
        "hodge_face_correction_max": float(np.max(np.abs(
            projected_residual - initial))),
        "face_divB_rms": float(np.sqrt(np.mean(div ** 2))),
        "face_divB_max": float(np.max(np.abs(div))),
        "boundary_flux_max": float(np.max(np.abs(projected[topology.boundary]))),
        "restriction_flux_error_max": table.restriction_error(),
    }
    return table, audit, result


def imprint_open_amr_face_field(table, particles, reconstruction="rt0",
                                limiter="mc"):
    """Prolong an AMR face table, sample target particles, and measure SPH DivB."""
    from .magnetic import sph_divergence_audit

    if not isinstance(table, AMRFaceMagneticTable):
        raise TypeError("table must be an AMRFaceMagneticTable")
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if any(particles.periodic):
        raise NotImplementedError("open AMR magnetic imprint requires open particles")
    if reconstruction == "rt0":
        evaluator = table.hdiv_reconstruction()
        reconstruction_label = "virtual-2to1-RT0"
    elif reconstruction == "balsara2":
        evaluator = table.balsara2_reconstruction(limiter=limiter)
        reconstruction_label = f"virtual-2to1-Balsara2-{limiter}"
    else:
        raise ValueError("reconstruction must be 'rt0' or 'balsara2'")
    B = evaluator.query(particles.pos)
    audit = sph_divergence_audit(
        particles.pos, particles.mass, particles.rho, particles.h, B,
        box=None, periodic=False)
    audit.update({
        "reconstruction": reconstruction_label,
        "virtual_cells": int(np.sum(evaluator.subdivisions ** 3)),
        "subdivided_leaves": int(np.count_nonzero(
            evaluator.subdivisions > 1)),
        "virtual_divB_max": evaluator.virtual_divergence_max,
        "sampled_B_rms": float(np.sqrt(np.mean(B ** 2))),
        "sampled_B_max": float(np.max(np.linalg.norm(B, axis=1))),
    })
    if reconstruction == "balsara2":
        audit["limiter"] = limiter
        audit["patch_flux_error_max"] = evaluator.patch_flux_error_max
    return B, audit, evaluator
