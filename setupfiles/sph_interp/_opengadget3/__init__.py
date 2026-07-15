"""OpenGadget3 glue for the sph_interp library.

OpenGadget3 (MAGNETICUM/Hydro-OpenGadget3) is a Gadget-family TreeSPH + SPMHD
particle code, so the IC is a 1:1 copy of the SPH particles -- the SAME GADGET
type-0 HDF5 (ICFormat 3) that the AREPO/GIZMO export writer
(`sph_interp.export.particles_to_arepo`) already produces (Coordinates,
Velocities, Masses, InternalEnergy, Density, SmoothingLength, MagneticField,
ParticleIDs -- all dataset names OpenGadget3 reads verbatim). The only
OpenGadget3-specific artefact is its GADGET-2-style text parameterfile, generated
here from the gasoline `.param` (see `opengadget3_param.write_run_param`).
"""
