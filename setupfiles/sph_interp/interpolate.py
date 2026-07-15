"""Top-level dispatch: interpolate particles onto a target with a chosen method.

    result = interpolate(particles, UniformGrid.centered(box, 128), method="sph-gather")
    rho_grid = result.data["rho"]

The (method, target) pair selects the engine. All of sph-gather, sph-scatter and
petkova are implemented for UniformGrid, AMRGrid, VoronoiGrid and Projection2D
(Projection2D sph-scatter delegates its `slice` mode to gather).
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .targets import UniformGrid, Projection2D, AMRGrid, VoronoiGrid
from .sph_method import (interpolate_uniform_sph, interpolate_projection2d,
                         interpolate_amr_sph, interpolate_voronoi_sph,
                         interpolate_amr_scatter, interpolate_uniform_scatter,
                         interpolate_voronoi_scatter,
                         interpolate_projection2d_scatter)
from .petkova_method import (interpolate_uniform_petkova,
                             interpolate_projection2d_petkova,
                             interpolate_amr_petkova, interpolate_voronoi_petkova)


@dataclass
class GridResult:
    """Interpolated grids plus the geometry needed to place/plot them.

    data    : dict[name -> ndarray] interpolated field grids (target-shaped).
    norm    : summed kernel weight per cell (sph-gather), accumulated mass per
              cell (sph-scatter) or None (petkova).
    target  : the target geometry used.
    method  : "sph-gather", "sph-scatter" or "petkova".
    """
    data: Dict[str, np.ndarray]
    norm: np.ndarray
    target: object
    method: str

    def edges(self):
        return self.target.edges()

    def centres(self):
        return self.target.centres()


_METHODS = ("sph-gather", "sph-scatter", "petkova")


def interpolate(particles, target, method="sph-gather", values=None, **kwargs):
    """Interpolate `particles` onto `target` using `method`.

    method : "sph-gather" (kernel density estimate AT each cell centre; fast,
             approximate, NOT mass-conserving), "sph-scatter" (AREPO-style
             conservative scatter: each particle distributes its mass to cells
             with a per-particle normalization; mass-conserving), or "petkova"
             (exact kernel-volume integral; mass-conserving). "sph" is accepted
             as a backward-compatible alias for "sph-gather".
    values : list of field names to interpolate (default: all particles.values).
    kwargs : forwarded to the engine (e.g. hmin, normalise for sph-gather).
    """
    method = method.lower()
    if method == "sph":                      # backward-compatible alias
        method = "sph-gather"
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r} (use one of {_METHODS})")

    if isinstance(target, UniformGrid):
        if method == "sph-gather":
            data, norm = interpolate_uniform_sph(particles, target,
                                                 values=values, **kwargs)
        elif method == "sph-scatter":
            data, norm = interpolate_uniform_scatter(particles, target,
                                                     values=values, **kwargs)
        else:
            data, norm = interpolate_uniform_petkova(particles, target,
                                                     values=values, **kwargs), None
        return GridResult(data=data, norm=norm, target=target, method=method)
    if isinstance(target, Projection2D):
        if method == "sph-gather":
            data = interpolate_projection2d(particles, target, values=values,
                                            **kwargs)
        elif method == "sph-scatter":
            data = interpolate_projection2d_scatter(particles, target,
                                                    values=values, **kwargs)
        else:
            data = interpolate_projection2d_petkova(particles, target,
                                                    values=values, **kwargs)
        return GridResult(data=data, norm=None, target=target, method=method)
    if isinstance(target, AMRGrid):
        if method == "sph-gather":
            data, norm = interpolate_amr_sph(particles, target, values=values,
                                             **kwargs)
        elif method == "sph-scatter":
            data, norm = interpolate_amr_scatter(particles, target,
                                                 values=values, **kwargs)
        else:
            data, norm = interpolate_amr_petkova(particles, target,
                                                 values=values, **kwargs), None
        return GridResult(data=data, norm=norm, target=target, method=method)
    if isinstance(target, VoronoiGrid):
        if method == "sph-gather":
            data, norm = interpolate_voronoi_sph(particles, target,
                                                 values=values, **kwargs)
        elif method == "sph-scatter":
            data, norm = interpolate_voronoi_scatter(particles, target,
                                                     values=values, **kwargs)
        else:
            data, norm = interpolate_voronoi_petkova(particles, target,
                                                     values=values, **kwargs), None
        return GridResult(data=data, norm=norm, target=target, method=method)
    raise NotImplementedError(
        f"target {type(target).__name__} not supported yet "
        f"(UniformGrid, Projection2D, AMRGrid and VoronoiGrid are implemented)")
