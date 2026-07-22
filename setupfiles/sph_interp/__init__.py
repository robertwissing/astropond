"""sph_interp — code-agnostic SPH -> grid interpolation.

Quick start:

    from sph_interp import Particles, UniformGrid, interpolate, from_tipsy

    p = from_tipsy("run/snap.00100", fields=("rho", "Bmag"))
    res = interpolate(p, UniformGrid.centered(p.box, 128), method="sph")
    rho_grid = res.data["rho"]            # (128,128,128)

Two interpolation METHODS — "sph" (fast kernel deposit) and "petkova" (exact,
mass-conserving) — over pluggable TARGET geometries. Implemented for BOTH
methods: UniformGrid (3D), Projection2D (2D pixel maps), AMRGrid (octree), and
VoronoiGrid (unstructured polyhedra, the native Petkova case). AMR results
export to other mesh codes via `to_cell_table` (export.py), the output-side
analog of `from_tipsy`.
"""

from .particles import Particles, from_tipsy
from .domain import OPEN_PADDING_MODES, open_domain_bounds
from .targets import (UniformGrid, AMRGrid, VoronoiGrid, Projection2D,
                      refine_smoothing_length, refine_mass, refine_count,
                      refine_band, refine_any, grade_2to1)
from .interpolate import interpolate, GridResult
from .export import (to_cell_table, to_structured_array, to_ramses,
                     to_arepo, particles_to_arepo,
                     RAMSES_HYDRO_VARS, RAMSES_MHD_VARS)
from .solenoidal import project_solenoidal, project_solenoidal_open
from .magnetic import (FaceMagneticTable, capture_periodic_face_field,
                       OpenFaceMagneticTable, capture_open_face_field,
                       imprint_open_face_field, imprint_periodic_face_field,
                       sph_divergence,
                       sph_divergence_audit)
from .magnetic_amr import (AMRFaceTopology, AMRFaceMagneticTable,
                           AMRHDivReconstruction,
                           capture_open_amr_face_field,
                           imprint_open_amr_face_field)
from .magnetic_balsara import (Balsara2Reconstruction,
                               build_balsara2_reconstruction)
from .magnetic_amr_balsara import (AMRBalsara2Reconstruction,
                                   build_amr_balsara2_reconstruction)
from .magnetic_validation import validate_magnetic_reconstruction
from .kernels import wkernel, cnorm3D
from .visualise import (visualise, field_image, sample_at_particles,
                        draw_cells, draw_cells_3d)

__all__ = [
    "Particles", "from_tipsy", "OPEN_PADDING_MODES", "open_domain_bounds",
    "UniformGrid", "AMRGrid", "VoronoiGrid", "Projection2D",
    "refine_smoothing_length", "refine_mass", "refine_count",
    "refine_band", "refine_any", "grade_2to1",
    "interpolate", "GridResult",
    "to_cell_table", "to_structured_array", "to_ramses",
    "to_arepo", "particles_to_arepo",
    "RAMSES_HYDRO_VARS", "RAMSES_MHD_VARS", "project_solenoidal",
    "project_solenoidal_open",
    "FaceMagneticTable", "capture_periodic_face_field",
    "OpenFaceMagneticTable", "capture_open_face_field",
    "imprint_open_face_field", "imprint_periodic_face_field",
    "AMRFaceTopology", "AMRFaceMagneticTable",
    "AMRHDivReconstruction", "capture_open_amr_face_field",
    "imprint_open_amr_face_field",
    "Balsara2Reconstruction", "build_balsara2_reconstruction",
    "AMRBalsara2Reconstruction", "build_amr_balsara2_reconstruction",
    "sph_divergence", "sph_divergence_audit",
    "validate_magnetic_reconstruction",
    "wkernel", "cnorm3D",
    "visualise", "field_image", "sample_at_particles", "draw_cells",
    "draw_cells_3d",
]
