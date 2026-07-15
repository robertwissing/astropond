"""GIZMO glue for the sph_interp library.

GIZMO is a particle (meshless finite-mass / SPH / MHD) code, so the IC is a 1:1
copy of the SPH particles -- the same GADGET type-0 HDF5 that the AREPO export
writer (`sph_interp.export.particles_to_arepo`) already produces. The only
GIZMO-specific artefact is its text parameterfile, generated here from the
gasoline `.param` (see `gizmo_param.write_run_param`).
"""
