"""Conservative SPH -> divergence-free face-field -> particle transfer.

The magnetic field is an intensive quantity per unit volume, not per unit mass.
Petkova's conservative particle carrier is therefore ``B/rho``: depositing
``m * (B/rho)`` gives the exact kernel-volume integral of B.  Dividing that raw
cell content by the geometric cell volume produces a cell-average magnetic
field.  A staggered Hodge projection then makes the shared face-normal fluxes
discretely divergence-free.

``FaceMagneticTable.query`` uses the lowest-order cubical H(div) (RT0)
reconstruction.  In each cell Bx varies only with x between the two x faces,
and analogously for By and Bz.  Its pointwise divergence is exactly the finite-
volume face divergence, while the normal component is single-valued across a
shared face.  Tangential jumps are allowed; forcing all components continuous
at vertices would generally destroy the divergence constraint.

This first implementation is deliberately periodic and uniform.  Open/AMR
domains need a boundary condition plus coarse/fine flux restriction and
prolongation before the same H(div) readout can be used safely.
"""

from dataclasses import dataclass
import json

import numpy as np

from .interpolate import GridResult, interpolate
from .particles import Particles
from .solenoidal import project_solenoidal, project_solenoidal_open
from .targets import UniformGrid


_SCHEMA = "sph-interp-periodic-face-magnetic-v1"


def _component_budget(source, target):
    """JSON-friendly absolute/relative residual for a three-vector budget."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - source
    relative = np.full(3, np.nan, dtype=np.float64)
    nonzero = source != 0.0
    relative[nonzero] = delta[nonzero] / source[nonzero]
    scale = max(float(np.linalg.norm(source)), np.finfo(np.float64).tiny)
    return {
        "source": source.tolist(),
        "target": target.tolist(),
        "delta": delta.tolist(),
        "relative": [float(x) if np.isfinite(x) else None for x in relative],
        "relative_norm": float(np.linalg.norm(delta) / scale),
    }


@dataclass
class FaceMagneticTable:
    """Periodic uniform-grid face fluxes with an H(div) point evaluator.

    ``Bx_face[i,j,k]`` is the x-normal field on the right face of cell
    ``(i,j,k)``; the cell's left value is ``Bx_face[i-1,j,k]``.  The y and z
    arrays follow the same convention.  Thus every periodic face is stored
    exactly once and is shared rather than reconstructed independently.
    """

    lo: np.ndarray
    box: np.ndarray
    npx: tuple
    Bx_face: np.ndarray
    By_face: np.ndarray
    Bz_face: np.ndarray

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.box = np.asarray(self.box, dtype=np.float64).reshape(3)
        self.npx = tuple(int(n) for n in self.npx)
        if len(self.npx) != 3 or any(n <= 0 for n in self.npx):
            raise ValueError("npx must contain three positive cell counts")
        if np.any(self.box <= 0.0):
            raise ValueError("box lengths must be positive")
        shape = self.npx
        for name in ("Bx_face", "By_face", "Bz_face"):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            setattr(self, name, value)

    @classmethod
    def from_projection(cls, grid, projected):
        if not isinstance(grid, UniformGrid):
            raise TypeError("FaceMagneticTable requires a UniformGrid")
        return cls(grid.lo, grid.box, grid.npx,
                   projected["Bx_r"], projected["By_r"], projected["Bz_r"])

    @property
    def pixwidth(self):
        return self.box / np.asarray(self.npx, dtype=np.float64)

    @property
    def cell_volume(self):
        return float(np.prod(self.pixwidth))

    def cell_divergence(self):
        """Finite-volume divergence, also the RT0 pointwise divergence."""
        dx, dy, dz = self.pixwidth
        return ((self.Bx_face - np.roll(self.Bx_face, 1, axis=0)) / dx
                + (self.By_face - np.roll(self.By_face, 1, axis=1)) / dy
                + (self.Bz_face - np.roll(self.Bz_face, 1, axis=2)) / dz)

    def cell_centred(self):
        """Face-pair averages, useful for integral and correction audits."""
        return np.stack([
            0.5 * (self.Bx_face + np.roll(self.Bx_face, 1, axis=0)),
            0.5 * (self.By_face + np.roll(self.By_face, 1, axis=1)),
            0.5 * (self.Bz_face + np.roll(self.Bz_face, 1, axis=2)),
        ], axis=-1)

    def magnetic_energy(self):
        """Return ``integral 0.5*|B|^2 dV`` of the RT0 reconstruction.

        For a component linear between face traces ``a`` and ``b``, its exact
        mean square is ``(a*a + a*b + b*b)/3``.  No point quadrature is needed.
        The caller may apply its own physical unit factor (for example 1/mu0).
        """
        total = 0.0
        for face, axis in ((self.Bx_face, 0),
                           (self.By_face, 1),
                           (self.Bz_face, 2)):
            left = np.roll(face, 1, axis=axis)
            total += np.sum((left * left + left * face + face * face) / 3.0)
        return 0.5 * self.cell_volume * float(total)

    def query(self, positions):
        """Evaluate the periodic RT0 field at arbitrary positions.

        The return shape is ``(N,3)``.  Coordinates are wrapped into the table
        domain; values exactly at a face are independent of which adjacent cell
        owns the floating-point coordinate because the normal face value is
        shared.
        """
        pos = np.asarray(positions, dtype=np.float64)
        scalar = pos.ndim == 1
        pos = np.atleast_2d(pos)
        if pos.shape[1] != 3:
            raise ValueError("positions must have shape (N,3) or (3,)")
        if not np.all(np.isfinite(pos)):
            raise ValueError("positions contain non-finite values")

        dims = np.asarray(self.npx, dtype=np.int64)
        q = np.mod((pos - self.lo) / self.pixwidth, dims)
        index = np.floor(q).astype(np.int64)
        frac = q - index
        ix, iy, iz = index.T
        tx, ty, tz = frac.T

        bx_l = self.Bx_face[(ix - 1) % dims[0], iy, iz]
        bx_r = self.Bx_face[ix, iy, iz]
        by_l = self.By_face[ix, (iy - 1) % dims[1], iz]
        by_r = self.By_face[ix, iy, iz]
        bz_l = self.Bz_face[ix, iy, (iz - 1) % dims[2]]
        bz_r = self.Bz_face[ix, iy, iz]
        out = np.column_stack([
            bx_l + tx * (bx_r - bx_l),
            by_l + ty * (by_r - by_l),
            bz_l + tz * (bz_r - bz_l),
        ])
        return out[0] if scalar else out

    def balsara2_reconstruction(self, limiter="mc"):
        """Build a second-order divergence-compatible point evaluator.

        The stored shared face averages are unchanged.  Balsara2 augments
        their normal traces with limited transverse slopes before constructing
        a cell polynomial whose divergence equals ``cell_divergence()``.
        """
        from .magnetic_balsara import build_balsara2_reconstruction
        return build_balsara2_reconstruction(self, limiter=limiter)

    def save(self, path, metadata=None):
        """Save the face table without pickled Python objects."""
        meta = {} if metadata is None else dict(metadata)
        np.savez_compressed(
            path, schema=np.asarray(_SCHEMA), lo=self.lo, box=self.box,
            npx=np.asarray(self.npx, dtype=np.int64),
            Bx_face=self.Bx_face, By_face=self.By_face, Bz_face=self.Bz_face,
            metadata=np.asarray(json.dumps(meta, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"].item())
            if schema != _SCHEMA:
                raise ValueError(f"unsupported magnetic table schema {schema!r}")
            table = cls(data["lo"], data["box"], tuple(data["npx"]),
                        data["Bx_face"], data["By_face"], data["Bz_face"])
            metadata = json.loads(str(data["metadata"].item()))
        return table, metadata


@dataclass
class OpenFaceMagneticTable:
    """Open uniform face fluxes with prescribed shared domain boundaries."""

    lo: np.ndarray
    box: np.ndarray
    npx: tuple
    Bx_face: np.ndarray
    By_face: np.ndarray
    Bz_face: np.ndarray

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.box = np.asarray(self.box, dtype=np.float64).reshape(3)
        self.npx = tuple(int(n) for n in self.npx)
        nx, ny, nz = self.npx
        shapes = {"Bx_face": (nx + 1, ny, nz),
                  "By_face": (nx, ny + 1, nz),
                  "Bz_face": (nx, ny, nz + 1)}
        for name, shape in shapes.items():
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            setattr(self, name, value)

    @classmethod
    def from_projection(cls, grid, projected):
        if not isinstance(grid, UniformGrid):
            raise TypeError("OpenFaceMagneticTable requires a UniformGrid")
        return cls(grid.lo, grid.box, grid.npx,
                   projected["Bx_face"], projected["By_face"],
                   projected["Bz_face"])

    @property
    def pixwidth(self):
        return self.box / np.asarray(self.npx, dtype=np.float64)

    @property
    def cell_volume(self):
        return float(np.prod(self.pixwidth))

    def cell_divergence(self):
        dx, dy, dz = self.pixwidth
        return ((self.Bx_face[1:] - self.Bx_face[:-1]) / dx
                + (self.By_face[:, 1:] - self.By_face[:, :-1]) / dy
                + (self.Bz_face[:, :, 1:] - self.Bz_face[:, :, :-1]) / dz)

    def cell_centred(self):
        return np.stack([
            0.5 * (self.Bx_face[:-1] + self.Bx_face[1:]),
            0.5 * (self.By_face[:, :-1] + self.By_face[:, 1:]),
            0.5 * (self.Bz_face[:, :, :-1] + self.Bz_face[:, :, 1:]),
        ], axis=-1)

    def query(self, positions):
        """Evaluate the open-domain RT0 field; positions must lie inside."""
        pos = np.asarray(positions, dtype=np.float64)
        scalar = pos.ndim == 1
        pos = np.atleast_2d(pos)
        if pos.shape[1] != 3:
            raise ValueError("positions must have shape (N,3) or (3,)")
        hi = self.lo + self.box
        tol = 32.0 * np.finfo(np.float64).eps * np.maximum(1.0, self.box)
        if np.any(pos < self.lo - tol) or np.any(pos > hi + tol):
            raise ValueError("query position lies outside the open magnetic table")
        dims = np.asarray(self.npx, dtype=np.int64)
        q = (np.minimum(np.maximum(pos, self.lo), hi) - self.lo) / self.pixwidth
        index = np.minimum(np.floor(q).astype(np.int64), dims - 1)
        frac = np.minimum(q - index, 1.0)
        ix, iy, iz = index.T
        tx, ty, tz = frac.T
        out = np.column_stack([
            self.Bx_face[ix, iy, iz]
            + tx * (self.Bx_face[ix + 1, iy, iz] - self.Bx_face[ix, iy, iz]),
            self.By_face[ix, iy, iz]
            + ty * (self.By_face[ix, iy + 1, iz] - self.By_face[ix, iy, iz]),
            self.Bz_face[ix, iy, iz]
            + tz * (self.Bz_face[ix, iy, iz + 1] - self.Bz_face[ix, iy, iz]),
        ])
        return out[0] if scalar else out

    def balsara2_reconstruction(self, limiter="mc"):
        """Build a second-order divergence-compatible point evaluator."""
        from .magnetic_balsara import build_balsara2_reconstruction
        return build_balsara2_reconstruction(self, limiter=limiter)

    def magnetic_energy(self):
        total = 0.0
        for face, axis in ((self.Bx_face, 0),
                           (self.By_face, 1),
                           (self.Bz_face, 2)):
            left = np.take(face, np.arange(face.shape[axis] - 1), axis=axis)
            right = np.take(face, np.arange(1, face.shape[axis]), axis=axis)
            total += np.sum((left * left + left * right + right * right) / 3.0)
        return 0.5 * self.cell_volume * float(total)


def capture_periodic_face_field(particles, grid, fields=("Bx", "By", "Bz")):
    """Conservatively capture particle B into a periodic face-field table.

    For component k, Petkova deposits ``m * (B_k/rho) * fraction``.  The raw
    content is divided by cell volume, never by deposited mass.  The returned
    audit distinguishes exact volume-carrier conservation from changes made by
    the Hodge projection.

    Returns ``(table, audit, cell_result)``.  ``cell_result`` contains the raw
    cell averages plus all projected face/centre arrays for detailed inspection.
    """
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if not isinstance(grid, UniformGrid):
        raise TypeError("periodic magnetic capture currently requires UniformGrid")
    if not all(particles.periodic):
        raise NotImplementedError("periodic magnetic capture requires all axes periodic")
    if particles.box is None or not np.allclose(particles.box, grid.box,
                                                rtol=1e-12, atol=1e-14):
        raise ValueError("particle period and grid box must match")
    if len(fields) != 3:
        raise ValueError("fields must contain the three magnetic components")
    if np.any(~np.isfinite(particles.rho)) or np.any(particles.rho <= 0.0):
        raise ValueError("magnetic B/rho carriers require finite positive rho")

    carrier_values = {}
    carrier_names = []
    source_B = []
    for axis, field in zip("xyz", fields):
        if field not in particles.values:
            raise KeyError(f"particle magnetic component {field!r} is missing")
        component = np.asarray(particles.values[field], dtype=np.float64)
        if not np.all(np.isfinite(component)):
            raise ValueError(f"particle magnetic component {field!r} is non-finite")
        key = f"__magnetic_B{axis}_over_rho"
        carrier_names.append(key)
        carrier_values[key] = component / particles.rho
        source_B.append(component)
    volume_key = "__magnetic_volume_over_mass"
    carrier_values[volume_key] = 1.0 / particles.rho

    carrier_particles = Particles(
        pos=particles.pos, mass=particles.mass, rho=particles.rho, h=particles.h,
        values=carrier_values, box=particles.box, periodic=particles.periodic)
    deposited = interpolate(
        carrier_particles, grid, method="petkova",
        values=carrier_names + [volume_key], conservative_fields=True)

    cell_data = {}
    raw_contents = []
    for field, key in zip(fields, carrier_names):
        content = np.asarray(deposited.data[f"_content_{key}"], dtype=np.float64)
        raw_contents.append(content)
        cell_data[field] = content / grid.cell_volume
    raw_cell_B = np.stack([cell_data[name] for name in fields], axis=-1)
    result = GridResult(data=cell_data, norm=None, target=grid,
                        method="petkova-B-over-rho")
    projected = project_solenoidal(result, fields=fields, periodic=True,
                                   add_to_result=True)
    table = FaceMagneticTable.from_projection(grid, projected)

    source_B = np.stack(source_B, axis=1)
    particle_volume = particles.mass / particles.rho
    source_integral = np.sum(particle_volume[:, None] * source_B, axis=0)
    raw_integral = np.asarray([x.sum() for x in raw_contents])
    projected_integral = table.cell_centred().sum(axis=(0, 1, 2)) * grid.cell_volume
    source_volume = float(particle_volume.sum())
    raw_volume = float(np.asarray(
        deposited.data[f"_content_{volume_key}"], dtype=np.float64).sum())
    correction = table.cell_centred() - raw_cell_B
    div = table.cell_divergence()
    source_magnetic_energy = float(
        0.5 * np.sum(particle_volume * np.sum(source_B ** 2, axis=1)))
    raw_magnetic_energy = float(
        0.5 * grid.cell_volume * np.sum(raw_cell_B ** 2))
    projected_magnetic_energy = table.magnetic_energy()

    audit = {
        "schema": _SCHEMA,
        "carrier": "m*(B/rho)",
        "particle_to_raw_integral_B": _component_budget(source_integral,
                                                          raw_integral),
        "raw_to_projected_integral_B": _component_budget(raw_integral,
                                                           projected_integral),
        "particle_volume": source_volume,
        "deposited_volume": raw_volume,
        "volume_relative_error": ((raw_volume - source_volume) / source_volume
                                  if source_volume != 0.0 else None),
        "hodge_correction_rms": float(np.sqrt(np.mean(correction ** 2))),
        "hodge_correction_max": float(np.max(np.abs(correction))),
        "magnetic_energy": {
            "source_particle_volume_quadrature": source_magnetic_energy,
            "raw_cell_constant": raw_magnetic_energy,
            "projected_rt0": projected_magnetic_energy,
            "raw_over_source": (raw_magnetic_energy / source_magnetic_energy
                                if source_magnetic_energy != 0.0 else None),
            "projected_over_source": (
                projected_magnetic_energy / source_magnetic_energy
                if source_magnetic_energy != 0.0 else None),
        },
        "face_divB_rms": float(np.sqrt(np.mean(div ** 2))),
        "face_divB_max": float(np.max(np.abs(div))),
    }
    return table, audit, result


def capture_open_face_field(particles, grid, fields=("Bx", "By", "Bz"),
                            background=None, normal_trace=None,
                            boundary_flux="balance"):
    """Conservatively capture B on a support-complete open UniformGrid.

    The grid must contain every compact kernel support ``pos +/- 2h``, which
    prevents the projection from hiding kernel clipping.  ``background=None``
    selects fixed-zero normal boundaries for a compact field;
    ``background="mean"`` preserves the raw grid volume integral; an explicit
    three-vector supplies a known uniform far-field normal trace.
    ``background="particle-mean"`` first removes the particle-volume-weighted
    mean, deposits and projects only that zero-carrier residual, then restores
    the uniform mean on every face.  The latter is the natural particle-to-grid
    decomposition for a box-threading field in a sparsely occupied open box.
    ``normal_trace`` and ``boundary_flux`` expose the general spatially varying
    fixed-normal boundary accepted by :func:`project_solenoidal_open`.
    """
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if not isinstance(grid, UniformGrid):
        raise TypeError("open magnetic capture currently requires UniformGrid")
    if any(particles.periodic):
        raise NotImplementedError("open magnetic capture requires non-periodic axes")
    if len(fields) != 3:
        raise ValueError("fields must contain the three magnetic components")
    if np.any(~np.isfinite(particles.rho)) or np.any(particles.rho <= 0.0):
        raise ValueError("magnetic B/rho carriers require finite positive rho")
    support_lo = np.min(particles.pos - 2.0 * particles.h[:, None], axis=0)
    support_hi = np.max(particles.pos + 2.0 * particles.h[:, None], axis=0)
    tol = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, grid.box)
    if np.any(grid.lo > support_lo + tol) or np.any(grid.hi < support_hi - tol):
        raise ValueError("open magnetic grid clips compact SPH kernel support; "
                         "expand it to contain pos +/- 2h")

    carrier_names = []
    source_components = []
    for axis, field in zip("xyz", fields):
        if field not in particles.values:
            raise KeyError(f"particle magnetic component {field!r} is missing")
        component = np.asarray(particles.values[field], dtype=np.float64)
        if not np.all(np.isfinite(component)):
            raise ValueError(f"particle magnetic component {field!r} is non-finite")
        key = f"__open_magnetic_B{axis}_over_rho"
        carrier_names.append(key)
        source_components.append(component)
    source_B = np.stack(source_components, axis=1)
    particle_volume = particles.mass / particles.rho
    source_integral = np.sum(particle_volume[:, None] * source_B, axis=0)
    source_volume = float(particle_volume.sum())
    particle_background = np.zeros(3, dtype=np.float64)
    deposited_B = source_B
    projection_background = background
    particle_mean_background = (
        isinstance(background, str) and background == "particle-mean")
    if particle_mean_background:
        particle_background = source_integral / source_volume
        deposited_B = source_B - particle_background[None, :]
        projection_background = None
    if normal_trace is not None and background is not None \
            and not particle_mean_background:
        raise ValueError(
            "normal_trace can only be combined with background='particle-mean'")

    projection_trace = normal_trace
    if particle_mean_background and normal_trace is not None \
            and not isinstance(normal_trace, str):
        projection_trace = {}
        for key, axis in (("Bx_lo", 0), ("Bx_hi", 0),
                          ("By_lo", 1), ("By_hi", 1),
                          ("Bz_lo", 2), ("Bz_hi", 2)):
            if key not in normal_trace:
                raise ValueError(f"normal_trace is missing {key}")
            projection_trace[key] = (
                np.asarray(normal_trace[key], dtype=np.float64)
                - particle_background[axis])

    carrier_values = {}
    for key, component in zip(carrier_names, deposited_B.T):
        carrier_values[key] = component / particles.rho
    volume_key = "__open_magnetic_volume_over_mass"
    carrier_values[volume_key] = 1.0 / particles.rho
    carrier_particles = Particles(
        particles.pos, particles.mass, particles.rho, particles.h,
        carrier_values, particles.box, particles.periodic)
    deposited = interpolate(
        carrier_particles, grid, method="petkova",
        values=carrier_names + [volume_key], conservative_fields=True)

    cell_data = {}
    contents = []
    for field, key in zip(fields, carrier_names):
        content = np.asarray(deposited.data[f"_content_{key}"], dtype=np.float64)
        contents.append(content)
        cell_data[field] = content / grid.cell_volume
    raw_residual_B = np.stack([cell_data[name] for name in fields], axis=-1)
    result = GridResult(cell_data, None, grid, "petkova-B-over-rho-open")
    projected = project_solenoidal_open(
        result, fields=fields, add_to_result=True,
        background=projection_background, normal_trace=projection_trace,
        boundary_flux=boundary_flux)
    if particle_mean_background:
        for axis, value in zip("xyz", particle_background):
            projected[f"B{axis}_face"] += value
            projected[f"B{axis}_c"] += value
    table = OpenFaceMagneticTable.from_projection(grid, projected)

    deposited_volume = float(np.asarray(
        deposited.data[f"_content_{volume_key}"], dtype=np.float64).sum())
    residual_raw_integral = np.asarray([content.sum() for content in contents])
    material_raw_integral = (
        residual_raw_integral + particle_background * deposited_volume)
    raw_B = raw_residual_B + particle_background
    geometric_raw_integral = (
        residual_raw_integral + particle_background * np.prod(grid.box))
    projected_integral = table.cell_centred().sum(axis=(0, 1, 2)) * grid.cell_volume
    correction = table.cell_centred() - raw_B
    div = table.cell_divergence()
    source_energy = float(0.5 * np.sum(
        particle_volume * np.sum(source_B ** 2, axis=1)))
    raw_energy = float(0.5 * grid.cell_volume * np.sum(raw_B ** 2))
    projected_energy = table.magnetic_energy()
    if normal_trace is not None:
        boundary_condition = (
            "fixed-spatial-normal-plus-particle-mean"
            if particle_mean_background else "fixed-spatial-normal")
    elif background is None:
        boundary_condition = "fixed-zero-normal"
    elif isinstance(background, str):
        boundary_condition = (
            "fixed-particle-mean-normal" if particle_mean_background
            else "fixed-raw-mean-normal")
    else:
        boundary_condition = "fixed-uniform-background-normal"
    boundary_background = [
        float(projected["Bx_face"][0, 0, 0]),
        float(projected["By_face"][0, 0, 0]),
        float(projected["Bz_face"][0, 0, 0]),
    ]
    audit = {
        "schema": "sph-interp-open-face-magnetic-v1",
        "boundary_condition": boundary_condition,
        "boundary_background": boundary_background,
        "boundary_projection": projected["_boundary_audit"],
        "support_lo": support_lo.tolist(),
        "support_hi": support_hi.tolist(),
        "particle_to_raw_integral_B": _component_budget(
            source_integral, material_raw_integral),
        "raw_to_projected_integral_B": _component_budget(
            geometric_raw_integral, projected_integral),
        "residual_raw_integral_B": residual_raw_integral.tolist(),
        "geometric_raw_integral_B": geometric_raw_integral.tolist(),
        "particle_volume": source_volume,
        "deposited_volume": deposited_volume,
        "volume_relative_error": ((deposited_volume - source_volume) / source_volume
                                  if source_volume != 0.0 else None),
        "hodge_correction_rms": float(np.sqrt(np.mean(correction ** 2))),
        "hodge_correction_max": float(np.max(np.abs(correction))),
        "magnetic_energy": {
            "source_particle_volume_quadrature": source_energy,
            "raw_cell_constant": raw_energy,
            "projected_rt0": projected_energy,
        },
        "face_divB_rms": float(np.sqrt(np.mean(div ** 2))),
        "face_divB_max": float(np.max(np.abs(div))),
        "boundary_flux_max": float(max(
            np.max(np.abs(table.Bx_face[[0, -1]])),
            np.max(np.abs(table.By_face[:, [0, -1]])),
            np.max(np.abs(table.Bz_face[:, :, [0, -1]])))),
    }
    return table, audit, result


def sph_divergence(positions, mass, rho, h, B, box=None, periodic=True):
    """Measure the standard difference-form SPH divergence on target particles.

    The operator is

    ``divB_i = (1/rho_i) sum_j m_j (B_j-B_i) . grad_i W(r_ij,h_i)``.

    It annihilates a constant field exactly.  It is a diagnostic of the sampled
    particle representation, not a proof of the face field's finite-volume
    constraint; a field can have zero RT0 divergence and nonzero discrete SPH
    divergence on a disordered particle set.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    n = len(pos)
    if pos.shape != (n, 3) or B.shape != (n, 3):
        raise ValueError("positions and B must both have shape (N,3)")
    if mass.shape != (n,) or rho.shape != (n,) or h.shape != (n,):
        raise ValueError("mass, rho, and h must have shape (N,)")
    if np.any(rho <= 0.0) or np.any(h <= 0.0):
        raise ValueError("rho and h must be positive")
    if not all(np.all(np.isfinite(a)) for a in (pos, mass, rho, h, B)):
        raise ValueError("SPH divergence inputs must be finite")

    if periodic:
        if box is None:
            raise ValueError("periodic SPH divergence requires box")
        period = np.asarray(box, dtype=np.float64).reshape(3)
        if np.any(period <= 0.0):
            raise ValueError("box lengths must be positive")
        origin = np.min(pos, axis=0)
        tree_pos = np.mod(pos - origin, period)
        tree = cKDTree(tree_pos, boxsize=period)
    else:
        period = None
        tree_pos = pos
        tree = cKDTree(pos)

    neighbours = tree.query_ball_point(tree_pos, 2.0 * h)
    out = np.zeros(n, dtype=np.float64)
    for i, ids in enumerate(neighbours):
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[ids != i]
        if ids.size == 0:
            continue
        delta = pos[i] - pos[ids]
        if periodic:
            delta -= period * np.rint(delta / period)
        radius = np.linalg.norm(delta, axis=1)
        active = (radius > 0.0) & (radius < 2.0 * h[i])
        if not np.any(active):
            continue
        ids = ids[active]
        delta = delta[active]
        radius = radius[active]
        q = radius / h[i]
        dwdq = np.where(q < 1.0,
                        -3.0 * q + 2.25 * q * q,
                        -0.75 * (2.0 - q) ** 2)
        dWdr = dwdq / (np.pi * h[i] ** 4)
        grad = dWdr[:, None] * delta / radius[:, None]
        out[i] = np.sum(mass[ids] * np.einsum(
            "ij,ij->i", B[ids] - B[i], grad)) / rho[i]
    return out


def sph_divergence_audit(positions, mass, rho, h, B, box=None, periodic=True):
    """Return raw and dimensionless target-particle divergence diagnostics."""
    div = sph_divergence(positions, mass, rho, h, B, box=box,
                         periodic=periodic)
    Bmag = np.linalg.norm(np.asarray(B, dtype=np.float64), axis=1)
    h = np.asarray(h, dtype=np.float64)
    normalized = np.zeros_like(div)
    nonzero = Bmag > 0.0
    normalized[nonzero] = np.abs(div[nonzero]) * h[nonzero] / Bmag[nonzero]
    return {
        "operator": "difference-SPH-cubic-spline-h_i",
        "divB": div,
        "normalized": normalized,
        "divB_rms": float(np.sqrt(np.mean(div ** 2))),
        "divB_max": float(np.max(np.abs(div))),
        "normalized_mean": float(np.mean(normalized)),
        "normalized_rms": float(np.sqrt(np.mean(normalized ** 2))),
        "normalized_max": float(np.max(normalized)),
    }


def imprint_periodic_face_field(table, particles, reconstruction="rt0",
                                limiter="mc"):
    """Evaluate a face table on target particles and audit their SPH DivB.

    This is the final periodic transfer step after glass relaxation.  Target
    density and smoothing lengths must already have been recomputed on the new
    particle set, because they define the diagnostic SPH operator.
    """
    if not isinstance(table, FaceMagneticTable):
        raise TypeError("table must be a FaceMagneticTable")
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if not all(particles.periodic):
        raise NotImplementedError("periodic magnetic imprint requires periodic particles")
    if particles.box is None or not np.allclose(particles.box, table.box,
                                                rtol=1e-12, atol=1e-14):
        raise ValueError("particle period and magnetic table box must match")
    if reconstruction == "rt0":
        evaluator = table
    elif reconstruction == "balsara2":
        evaluator = table.balsara2_reconstruction(limiter=limiter)
    else:
        raise ValueError("reconstruction must be 'rt0' or 'balsara2'")
    B = evaluator.query(particles.pos)
    audit = sph_divergence_audit(
        particles.pos, particles.mass, particles.rho, particles.h, B,
        box=particles.box, periodic=True)
    audit["sampled_B_rms"] = float(np.sqrt(np.mean(B ** 2)))
    audit["sampled_B_max"] = float(np.max(np.linalg.norm(B, axis=1)))
    audit["reconstruction"] = reconstruction
    if reconstruction == "balsara2":
        audit["limiter"] = limiter
        audit["polynomial_divB_coefficient_max"] = float(
            np.max(np.abs(evaluator.divergence_coefficients()[..., 1:])))
    return B, audit


def imprint_open_face_field(table, particles, reconstruction="rt0",
                            limiter="mc"):
    """Evaluate an open uniform face table and audit target-particle SPH DivB."""
    if not isinstance(table, OpenFaceMagneticTable):
        raise TypeError("table must be an OpenFaceMagneticTable")
    if not isinstance(particles, Particles):
        raise TypeError("particles must be a sph_interp.Particles instance")
    if any(particles.periodic):
        raise NotImplementedError("open magnetic imprint requires open particles")
    if reconstruction == "rt0":
        evaluator = table
    elif reconstruction == "balsara2":
        evaluator = table.balsara2_reconstruction(limiter=limiter)
    else:
        raise ValueError("reconstruction must be 'rt0' or 'balsara2'")
    B = evaluator.query(particles.pos)
    audit = sph_divergence_audit(
        particles.pos, particles.mass, particles.rho, particles.h, B,
        box=None, periodic=False)
    audit["sampled_B_rms"] = float(np.sqrt(np.mean(B ** 2)))
    audit["sampled_B_max"] = float(np.max(np.linalg.norm(B, axis=1)))
    audit["reconstruction"] = reconstruction
    if reconstruction == "balsara2":
        audit["limiter"] = limiter
        audit["polynomial_divB_coefficient_max"] = float(
            np.max(np.abs(evaluator.divergence_coefficients()[..., 1:])))
    return B, audit
