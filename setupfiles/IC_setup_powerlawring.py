#!/usr/bin/env python3
"""Initial conditions for a disk or annulus in an external power law.

In fixed test units the circular-speed law is

    v_c = r^beta,    V0 = R0 = 1

before softening.  The IC is a glass-relaxed target in the external field,
with the same radial density taper, rotation law and magnetic arms for both
supported geometries.  The default annulus excludes the singular magnetic
axis.  The optional full disk uses a softened rotation law and a center-regular
toroidal field.  Both radial surfaces are free; the runtime force is either the
softened power law, which leaves them free, or the bounded Kepler law, whose
repulsive core and confining rim act outside everything emitted.

The vertical structure follows the selected force geometry:

* spherical (``geometry=3``) has a vertical restoring force, so the target is
  the exact hydrostatic atmosphere of a compact, lowered-isothermal barotrope.
  It is Gaussian to leading order in the resolved core but reaches an exact
  vacuum surface.  The compact surface matters for beta < 0, where a strictly
  isothermal atmosphere cannot have finite support because the spherical
  power-law potential has finite escape depth.
* cylindrical (``geometry=2``) applies no vertical acceleration, so the exact
  vertical solution of the same barotrope is a constant-density slab.  The
  target is therefore ``rho(R)`` alone, uniform in z inside the science slab.
  Periodicity closes its upper and lower surfaces.  Its radial edges are
  sharp, which is the construction the cold Keplerian-ring literature uses:
  a taper only softens a free surface whose disturbance has amplitude c_s
  whatever its shape.

Gas pressure and the toroidal Lorentz term are included in the initial
azimuthal velocity in both geometries.  There is no central sink and the
generated parameter file turns off gas self-gravity, so this setup exercises
only the external-potential path.
"""

import numpy as np

import IC_distribute as distribute
import IC_smoothlength as smth
import readtipsy as tip


R_INNER = 0.5
R_OUTER = 2.0
DOMAIN_ANNULUS = 0
DOMAIN_DISK = 1
# Which runtime force law the IC is built for.  gasoline's
# iPowerLawPotentialForm uses the same spelling, so nothing is remapped.
FORCE_POWERLAW = 0
FORCE_BOUNDED_KEPLER = 1
GEOMETRY_CYLINDRICAL = 2
GEOMETRY_SPHERICAL = 3

# Hopkins (2015) cold Keplerian ring, the test Groth et al. (2023) inherited
# as their equations (49)--(50).  The published surface density carries a 0.01
# background floor and smooth r^3 / (1+(r-2)/0.1)^-3 wings, and the force
# carries modified branches below R=0.35 and above R=2.1.  Both exist for the
# MESH arm, which cannot hold vacuum cells and therefore has material at the
# axis and far out.  The PARTICLE arm realizes the same profile by deletion --
# "the initial particles are evenly spaced such that the square domain from
# -2 < x < 2, -2 < y < 2 has 256^2 particles; then particles are removed
# inside r < 0.5 and outside r > 2" -- which leaves a flat, sharp-edged ring,
# exactly uniform_ring_factor() below.
#
# So this module needs neither the wings nor the force branches: no particle is
# ever placed where they act.  The three force radii live ONLY in pkd.c
# (pkdHopkinsKeplerPotential), where they catch whatever drifts out of the ring
# during the run, with no copy here to drift from.
#
# The one published number the IC still needs is the temperature.  P=1e-6 on a
# plateau of Sigma=1.01 fixes P/Sigma, and extruding both through the slab
# preserves it, so the ring is run at the barotrope scale height that
# reproduces the same P/rho at the ring density (see hopkins_scale_height).
HOPKINS_PRESSURE_2D = 1.0e-6
HOPKINS_SURFACE_DENSITY = 1.01

# Fixed first-slice equilibrium controls.  R=0.5..2 is the science annulus.
# The SPHERICAL atmosphere tapers to vacuum across disposable support bands,
# 0.25..0.5 inside and 2..2.5 outside; a cylindrical ring instead keeps a sharp
# surface at each science edge, 0.5 and 2.  These bounds are deliberately
# separate: a taper occupies bands OUTSIDE the science domain, while a sharp
# ring occupies the science domain exactly, so neither construction moves the
# other.  Beyond the retained tapered-target bounds, a
# half-unit disposable envelope skirt falls below any intended mass resolution.
# The relaxer's resolution-dependent rho_edge may move in that skirt without
# moving the science domain or removing its buffer.  The tiny skirt floor is
# also safe when a mapped density table clamps outside its axes.  H=0.15 remains
# resolved by several h at nx=64.  The lowered-isothermal density scale is small
# enough to leave the beta=1 core nearly Gaussian, but large enough that every
# beta > -1 has a compact atmosphere for the standard unsoftened annulus.
# Unusually large softenings in beta < 0 fields are rejected if their escape
# depth is too shallow to confine this atmosphere.
EQUILIBRIUM_SCALE_HEIGHT = 0.15
EQUILIBRIUM_CUTOFF_DENSITY = 1.0e-2
TAPER_INNER_SUPPORT_RADIUS = 0.25
EQUILIBRIUM_OUTER_BUFFER = 0.5
EQUILIBRIUM_RADIAL_FLOOR = 0.05
EQUILIBRIUM_ENVELOPE_SKIRT = 0.5
EQUILIBRIUM_ENVELOPE_FLOOR = 1.0e-8
EQUILIBRIUM_TABLE_NR = 513
EQUILIBRIUM_TABLE_NZ = 257
# The magnetization is quoted at the unit radius, the same reference the
# orbital Mach number uses (V0=R0=1).
FIELD_REFERENCE_RADIUS = 1.0

# Cylindrical geometry has no vertical force, so the barotrope alone does not
# select a thickness: with no a_z the slab is set by the periodic z box, and
# the cylindrical density target is exactly independent of the scale height
# (rho_cut*expm1(ln(1+f/rho_cut)) = f).  The z period is therefore GEOMETRY,
# not temperature, which is what lets the cylindrical arm run as cold as the
# classic Keplerian-ring test without thinning the slab.
#
# It is kept MINIMAL and tied to the resolution, the rule every other thin-box
# test in this suite uses -- orzag, gresho, currentsheet, khslab, taylorgreen,
# noh, square and mhddivtest all set their half-thickness to 2*sqrt(6)*dx.  A
# FIXED thickness spends particles on depth the test does not use: 0.3 was
# about right at nx=64 but is twice the needed depth at nx=128 and four times
# at nx=256.  Scaling it keeps a FIXED number of particles across the period at
# every resolution, so nx counts particles across the disk in every direction.
#
# The close-packed layer spacing is dx*sqrt(6)/3 and the period is 2*K*dx, so
# the slab holds exactly K*sqrt(6) = 12 close-packed layers (or ~22 cubic
# planes, which pack ~1.8x denser over the same period) at EVERY nx -- an
# integer by construction, so a lattice tiles the z period with no seam.  This
# is measured, not assumed: orzag at nx=32 gives 12 FCC layers and 22 cubic
# planes over the identical 9.798*dx period.  Doubling K to 24 FCC layers was
# considered and rejected: it costs exactly twice the particles in every
# cylindrical run and departs from the rest of the thin-box family.
#
# The same constant fixes the kernel/period contract.  A pair must not fall
# inside one kernel both directly and through the wrap, which needs the support
# diameter 2h to stay below the half period.  At the science density
# 2h = (3*nSmooth/(4*pi))^(1/3)*dx = 2.72*dx against a half period of 4.90*dx,
# so the ratio is 0.55 INDEPENDENT of resolution: the contract holds at every
# nx by construction rather than by luck.  In the disposable low-density edge
# bands h grows as rho^(-1/3) and 2h/dx = (20.05*rho_science/rho)^(1/3), which
# is slab-INDEPENDENT -- no slab thickness removes that band, so create()
# reports the emitted mass landing in it instead.
SLAB_HALF_HEIGHT_SPACINGS = 2.0 * np.sqrt(6.0)


def particle_spacing(nx):
    """Mean in-plane particle spacing: ``nx`` across the R=2 science diameter."""
    value = float(nx)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("nx must be finite and positive; got %r" % (nx,))
    return 2.0 * R_OUTER / value


def slab_half_height(nx):
    """Half the periodic z period of the cylindrical slab at resolution ``nx``."""
    return SLAB_HALF_HEIGHT_SPACINGS * particle_spacing(nx)


def _validated_slab_half(slab_half):
    """Return the run's slab half height, which cylindrical targets require."""
    if slab_half is None:
        raise ValueError("this cylindrical quantity needs the run's slab half "
                         "height; pass slab_half=slab_half_height(nx)")
    value = float(slab_half)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("slab_half must be finite and positive; got %r"
                         % (slab_half,))
    return value


def validated_scale_height(scale_height):
    """Return the barotrope scale height, the test's one temperature control.

    ``c_s(rho=1) = sqrt(gamma * H^2 * rho_cut * [1/rho_cut - ln(1+1/rho_cut)])``,
    so the orbital Mach number at ``R=1`` is ``1/c_s`` and scales as ``1/H``.
    The upper bound also rejects small integers, which would otherwise be
    read as an absurdly hot disk.
    """
    value = float(scale_height)
    if not np.isfinite(value) or value <= 0.0 or value > 0.5:
        raise ValueError("scale_height must be finite in (0, 0.5]; got %r"
                         % (scale_height,))
    return value


def validated_geometry(geometry):
    """Return the force geometry as 2 (cylindrical) or 3 (spherical)."""
    numeric = float(geometry)
    integer = int(round(numeric))
    if numeric != integer or integer not in (GEOMETRY_CYLINDRICAL,
                                             GEOMETRY_SPHERICAL):
        raise ValueError("geometry must be one of %d, %d; got %r" %
                         (GEOMETRY_CYLINDRICAL, GEOMETRY_SPHERICAL, geometry))
    return integer


def validated_force_form(force_form):
    """Return the runtime force law the IC is built for.

    0 is the softened power law, which regularizes its core with the
    softening and leaves both radial surfaces free.  1 is the bounded Kepler
    force of the Hopkins ring: exactly -1/R^2 across the disc, with a
    repulsive core and a confining outer rim outside it.
    """
    numeric = float(force_form)
    integer = int(round(numeric))
    if numeric != integer or integer not in (FORCE_POWERLAW,
                                             FORCE_BOUNDED_KEPLER):
        raise ValueError(
            "force_form must be 0 (power law) or 1 (bounded Kepler); got %r"
            % (force_form,))
    return integer


def hopkins_scale_height(rho_reference=1.0):
    """Barotrope scale height reproducing Hopkins' published sound speed.

    The paper fixes ``P/Sigma = 1e-6/1.01`` on the ring, and extruding both
    through the slab leaves that ratio unchanged, so the target is simply
    ``P(rho_ring) = (P_2D/Sigma) * rho_ring``.  The barotrope has ``P`` exactly
    proportional to ``H^2``, so inverting it is one square root and the result
    tracks EQUILIBRIUM_CUTOFF_DENSITY automatically instead of being a copied
    constant.  At the default ring density this is H = 1.0188e-3, i.e. an
    orbital Mach number of about 778 at R=1.
    """
    rho_reference = float(rho_reference)
    if not np.isfinite(rho_reference) or rho_reference <= 0.0:
        raise ValueError("the reference ring density must be finite and positive")
    target = (HOPKINS_PRESSURE_2D / HOPKINS_SURFACE_DENSITY) * rho_reference
    scaled = rho_reference / EQUILIBRIUM_CUTOFF_DENSITY
    # P = H^2 * rho_cut * [rho/rho_cut - ln(1+rho/rho_cut)]
    unit_pressure = EQUILIBRIUM_CUTOFF_DENSITY * (scaled - np.log1p(scaled))
    return float(np.sqrt(target / unit_pressure))


def circular_speed(radius, beta=-0.5, softening=0.0):
    """Circular speed implied by the softened power-law acceleration."""
    radius = np.asarray(radius, dtype=float)
    softened = np.sqrt(radius * radius + softening * softening)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return radius * softened ** (beta - 1.0)


def domain_inner_radius(domain_mode):
    """Return the fixed inner radius for an annulus or full disk."""
    if int(domain_mode) == DOMAIN_ANNULUS:
        return R_INNER
    if int(domain_mode) == DOMAIN_DISK:
        return 0.0
    raise ValueError("domain_mode must be 0 (annulus) or 1 (full disk)")


def equilibrium_buffer_bounds(domain_mode):
    """Return retained bounds for tapered equilibrium targets.

    The annulus's inner bound is the taper's support edge at 0.25, distinct
    from the common science and cylindrical free-edge radius at 0.5.  A full
    disk has no inner taper.  Both domains retain the outer taper through 2.5.
    """
    domain_inner_radius(domain_mode)
    inner = (TAPER_INNER_SUPPORT_RADIUS
             if int(domain_mode) == DOMAIN_ANNULUS else 0.0)
    return inner, R_OUTER + EQUILIBRIUM_OUTER_BUFFER


def uniform_ring_bounds(domain_mode):
    """Return the sharp cylindrical free-edge bounds.

    A sharp ring has no taper, so it has no use for the support radii a taper
    needs: both of its surfaces ARE the science edges, R=0.5..2 for the
    annulus and R=0..2 for the full disk.  The 0.25/2.5 support band is
    retained only by the tapered targets (``equilibrium_buffer_bounds``),
    which carry density to zero across it.
    """
    return domain_inner_radius(domain_mode), R_OUTER


def retained_particle_bounds(domain_mode, geometry, force_form=FORCE_POWERLAW):
    """Return the emitted IC bounds, which follow from the geometry alone.

    A cylindrical target has no vertical force, so it is a sharp-edged ring
    at R=0.5..2 -- the suite's free-edge arm and the Hopkins literature ring
    carve the identical annulus out of the identical relaxed box, and the
    force law does not change what is emitted.  Only the spherical atmosphere
    tapers, and it keeps its 0.25/2.5 support band.
    """
    validated_force_form(force_form)
    if validated_geometry(geometry) == GEOMETRY_CYLINDRICAL:
        return uniform_ring_bounds(domain_mode)
    return equilibrium_buffer_bounds(domain_mode)


def equilibrium_envelope_bounds(domain_mode):
    """Return the full target extent including the disposable faint skirt."""
    buffer_inner, buffer_outer = equilibrium_buffer_bounds(domain_mode)
    return (max(0.0, buffer_inner - EQUILIBRIUM_ENVELOPE_SKIRT),
            buffer_outer + EQUILIBRIUM_ENVELOPE_SKIRT)


def uniform_ring_factor(radius, domain_mode=DOMAIN_ANNULUS):
    """Sharp-edged uniform ring: one inside the retained bounds, zero outside.

    This is the cylindrical arm's radial profile.  A smooth taper exists only
    to soften a free surface, but the disturbance a free surface launches has
    amplitude ``c_s`` whatever its shape, so at the cold Mach numbers this arm
    runs at the taper buys nothing and costs an unphysical enthalpy gradient
    (it is what drove ``v_phi^2`` negative at beta=-0.5).  With a flat profile
    ``d ln f/dR = 0`` everywhere, so ``v_phi`` is the exact circular speed and
    the current-free ``B_phi=C/R`` needs no Lorentz correction.
    """
    radius = np.asarray(radius, dtype=float)
    inner, outer = uniform_ring_bounds(domain_mode)
    factor = np.where((radius >= inner) & (radius <= outer), 1.0, 0.0)
    return factor, np.zeros_like(factor)


def _smoothstep01(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smoothstep01_derivative(value):
    value = np.asarray(value, dtype=float)
    clipped = np.clip(value, 0.0, 1.0)
    derivative = 6.0 * clipped * (1.0 - clipped)
    return np.where((value > 0.0) & (value < 1.0), derivative, 0.0)


def radial_density_factor(radius, domain_mode=DOMAIN_ANNULUS):
    """Smooth radial density factor and its logarithmic derivative.

    The factor is one in the science domain, reaches the positive physical
    floor at the fixed buffer boundary, and then reaches the much fainter
    numerical floor across the disposable envelope skirt.  Interpolating
    ``ln(f)`` with smoothsteps makes both ``f`` and ``d f/dR`` continuous and
    gives the rotation law an analytic pressure correction everywhere that
    can survive the envelope cull.
    """
    radius = np.asarray(radius, dtype=float)
    inner = domain_inner_radius(domain_mode)
    buffer_inner, buffer_outer = equilibrium_buffer_bounds(domain_mode)
    envelope_inner, envelope_outer = equilibrium_envelope_bounds(domain_mode)
    log_buffer = np.log(EQUILIBRIUM_RADIAL_FLOOR)
    log_envelope = np.log(EQUILIBRIUM_ENVELOPE_FLOOR)
    log_factor = np.zeros_like(radius)
    dlog_dradius = np.zeros_like(radius)

    if int(domain_mode) == DOMAIN_ANNULUS:
        coordinate = ((radius - buffer_inner) /
                      (inner - buffer_inner))
        step = _smoothstep01(coordinate)
        inner_mask = radius < inner
        log_factor = np.where(inner_mask, log_buffer * (1.0 - step),
                              log_factor)
        derivative = (-log_buffer * _smoothstep01_derivative(coordinate) /
                      (inner - buffer_inner))
        dlog_dradius = np.where(inner_mask, derivative, dlog_dradius)

        coordinate = ((radius - envelope_inner) /
                      (buffer_inner - envelope_inner))
        step = _smoothstep01(coordinate)
        skirt_mask = radius < buffer_inner
        log_factor = np.where(
            skirt_mask,
            log_envelope + (log_buffer - log_envelope) * step,
            log_factor)
        derivative = ((log_buffer - log_envelope) *
                      _smoothstep01_derivative(coordinate) /
                      (buffer_inner - envelope_inner))
        dlog_dradius = np.where(skirt_mask, derivative, dlog_dradius)

    coordinate = ((radius - R_OUTER) /
                  (buffer_outer - R_OUTER))
    step = _smoothstep01(coordinate)
    outer_mask = radius > R_OUTER
    log_factor = np.where(outer_mask, log_buffer * step, log_factor)
    derivative = (log_buffer * _smoothstep01_derivative(coordinate) /
                  (buffer_outer - R_OUTER))
    dlog_dradius = np.where(outer_mask, derivative, dlog_dradius)

    coordinate = ((radius - buffer_outer) /
                  (envelope_outer - buffer_outer))
    step = _smoothstep01(coordinate)
    skirt_mask = radius > buffer_outer
    log_factor = np.where(
        skirt_mask,
        log_buffer + (log_envelope - log_buffer) * step,
        log_factor)
    derivative = ((log_envelope - log_buffer) *
                  _smoothstep01_derivative(coordinate) /
                  (envelope_outer - buffer_outer))
    dlog_dradius = np.where(skirt_mask, derivative, dlog_dradius)
    return np.exp(log_factor), dlog_dradius


def vertical_slab_factor(height, geometry=GEOMETRY_SPHERICAL,
                         slab_half=None):
    """Vertical shape factor of the cylindrical constant-density slab.

    The spherical field builds its own vertical structure from the potential,
    so its factor is one everywhere.  The cylindrical field applies no vertical
    acceleration: the exact vertical solution of the barotrope is a slab of
    constant density, which this factor truncates with a smoothstep onto an
    exact vacuum surface.  The taper is disposable envelope, not equilibrium.
    """
    height = np.asarray(height, dtype=float)
    if validated_geometry(geometry) == GEOMETRY_SPHERICAL:
        return np.ones_like(height)
    if slab_half is None:
        # No slab given, no truncation.  A caller that only wants the radial
        # equilibrium (the analyzer) reads particles that are inside the slab
        # by construction, so the cut cannot fire and it need not know the
        # run's z period.
        return np.ones_like(height)
    return np.where(np.abs(height) <= _validated_slab_half(slab_half),
                    1.0, 0.0)


def powerlaw_potential_difference(radius, height, beta=-0.5,
                                  softening=0.0,
                                  geometry=GEOMETRY_SPHERICAL):
    """Return ``Phi(R,z)-Phi(R,0)`` for the softened external field.

    The logarithmic beta=0 limit is evaluated explicitly.  At an unsoftened
    singular axis, the difference is zero at z=0 and infinite immediately off
    the plane for beta <= 0; that axis lies in the disposable skirt of an
    annulus, while a full disk requires positive softening.  A cylindrical
    field has no vertical dependence at all, so the difference vanishes.
    """
    radius, height = np.broadcast_arrays(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float))
    if validated_geometry(geometry) == GEOMETRY_CYLINDRICAL:
        return np.zeros_like(radius)
    beta = float(beta)
    softening = float(softening)
    midplane2 = radius * radius + softening * softening
    spherical2 = midplane2 + height * height
    difference = np.zeros_like(spherical2)
    regular = midplane2 > 0.0

    with np.errstate(over="raise", invalid="raise", divide="ignore"):
        if abs(beta) < 1.0e-8:
            difference[regular] = 0.5 * np.log(
                spherical2[regular] / midplane2[regular])
            difference[(~regular) & (height != 0.0)] = np.inf
        else:
            log_ratio = np.zeros_like(spherical2)
            log_ratio[regular] = np.log(
                spherical2[regular] / midplane2[regular])
            difference[regular] = (
                midplane2[regular] ** beta
                * np.expm1(beta * log_ratio[regular]) / (2.0 * beta))
            singular_off_plane = (~regular) & (height != 0.0)
            if beta > 0.0:
                difference[singular_off_plane] = (
                    spherical2[singular_off_plane] ** beta / (2.0 * beta))
            else:
                difference[singular_off_plane] = np.inf
    return np.maximum(difference, 0.0)


def _midplane_density_and_enthalpy(radius, domain_mode, rho_midplane=1.0,
                                   scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                                   geometry=GEOMETRY_SPHERICAL,
                                   force_form=FORCE_POWERLAW):
    scale_height = validated_scale_height(scale_height)
    validated_force_form(force_form)
    if validated_geometry(geometry) == GEOMETRY_CYLINDRICAL:
        factor, dlog_dradius = uniform_ring_factor(radius, domain_mode)
    else:
        factor, dlog_dradius = radial_density_factor(radius, domain_mode)
    midplane_density = float(rho_midplane) * factor
    enthalpy = (scale_height ** 2
                * np.log1p(midplane_density /
                           EQUILIBRIUM_CUTOFF_DENSITY))
    denthalpy_dradius = np.zeros_like(np.asarray(radius, dtype=float))
    np.divide(scale_height ** 2 * midplane_density * dlog_dradius,
              midplane_density + EQUILIBRIUM_CUTOFF_DENSITY,
              out=denthalpy_dradius,
              where=(midplane_density + EQUILIBRIUM_CUTOFF_DENSITY) > 0.0)
    return midplane_density, enthalpy, factor, denthalpy_dradius


def equilibrium_density(radius, height, domain_mode=DOMAIN_ANNULUS,
                        rho_midplane=1.0, beta=-0.5, softening=0.0,
                        geometry=GEOMETRY_SPHERICAL,
                        scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                        force_form=FORCE_POWERLAW, slab_half=None):
    """Compact barotropic density in exact hydrostatic balance.

    Its enthalpy is ``H^2 log(1 + rho/rho_cut)``.  Therefore

    ``h(R,z) = h(R,0) - [Phi(R,z)-Phi(R,0)]``

    gives a beta-independent equation of state, a finite vacuum surface, and
    a nearly isothermal/Gaussian core when ``rho >> rho_cut``.  A cylindrical
    field has no vertical potential difference, so this reduces exactly to the
    midplane density ``rho_m(R)``; the slab factor then truncates it.
    """
    radius, height = np.broadcast_arrays(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float))
    scale_height = validated_scale_height(scale_height)
    geometry = validated_geometry(geometry)
    force_form = validated_force_form(force_form)
    _midplane_density, midplane_enthalpy, _factor, _derivative = (
        _midplane_density_and_enthalpy(radius, domain_mode, rho_midplane,
                                       scale_height, geometry, force_form))
    remaining = (midplane_enthalpy - powerlaw_potential_difference(
        radius, height, beta, softening, geometry))
    density = np.zeros_like(remaining)
    occupied = remaining > 0.0
    density[occupied] = (EQUILIBRIUM_CUTOFF_DENSITY
                         * np.expm1(remaining[occupied] /
                                    scale_height ** 2))
    return density * vertical_slab_factor(height, geometry, slab_half)


def equilibrium_pressure_from_density(density,
                                      scale_height=EQUILIBRIUM_SCALE_HEIGHT):
    """Pressure of the lowered-isothermal barotrope at ``density``.

    The single definition of "what pressure goes with this density", for every
    radial mode.  The Hopkins ring needs no special case: its density is
    exactly flat, so the barotrope IS the literature's constant pressure -- and
    since gasoline assigns ``u`` once at t=0 and then evolves it adiabatically,
    the two are not merely close but identical.  ``hopkins_scale_height`` picks
    the ``H`` that reproduces the published value.
    """
    density = np.asarray(density, dtype=float)
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("equilibrium pressure requires finite non-negative density")
    scaled = density / EQUILIBRIUM_CUTOFF_DENSITY
    difference = scaled - np.log1p(scaled)
    small = scaled < 1.0e-4
    if np.any(small):
        value = scaled[small]
        difference = np.array(difference, copy=True)
        difference[small] = (0.5 * value ** 2 - value ** 3 / 3.0
                             + 0.25 * value ** 4)
    return (validated_scale_height(scale_height) ** 2
            * EQUILIBRIUM_CUTOFF_DENSITY * difference)


def equilibrium_pressure(radius, height, domain_mode=DOMAIN_ANNULUS,
                         rho_midplane=1.0, beta=-0.5, softening=0.0,
                         geometry=GEOMETRY_SPHERICAL,
                         scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                         force_form=FORCE_POWERLAW, slab_half=None):
    """Return the pressure paired with :func:`equilibrium_density`."""
    force_form = validated_force_form(force_form)
    density = equilibrium_density(radius, height, domain_mode, rho_midplane,
                                  beta, softening, geometry, scale_height,
                                  force_form, slab_half)
    return equilibrium_pressure_from_density(density, scale_height)


def equilibrium_surface_height(radius, domain_mode=DOMAIN_ANNULUS,
                               beta=-0.5, softening=0.0,
                               geometry=GEOMETRY_SPHERICAL,
                               scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                               force_form=FORCE_POWERLAW, slab_half=None):
    """Height of the exact vacuum surface at one or more radii."""
    radius = np.asarray(radius, dtype=float)
    if validated_geometry(geometry) == GEOMETRY_CYLINDRICAL:
        # No vertical force and a periodic z box: the slab IS the half period.
        return np.full_like(radius, _validated_slab_half(slab_half))
    _density, enthalpy, _factor, _derivative = (
        _midplane_density_and_enthalpy(radius, domain_mode, 1.0, scale_height,
                                       geometry, force_form))
    beta = float(beta)
    softening = float(softening)
    midplane2 = radius * radius + softening * softening
    if beta < 0.0:
        escape = np.full_like(radius, np.inf)
        regular = midplane2 > 0.0
        escape[regular] = (midplane2[regular] ** beta / (-2.0 * beta))
        if np.any(enthalpy >= escape * (1.0 - 64.0 * np.finfo(float).eps)):
            raise ValueError("the softened beta<0 potential is too shallow "
                             "to confine the equilibrium atmosphere")

    lower = np.zeros_like(radius)
    upper = np.full_like(radius, 6.0 * scale_height)
    for _iteration in range(80):
        unresolved = (powerlaw_potential_difference(
            radius, upper, beta, softening) < enthalpy)
        if not np.any(unresolved):
            break
        upper[unresolved] *= 2.0
    else:
        raise ValueError("could not bracket the equilibrium vacuum surface")
    for _iteration in range(64):
        middle = 0.5 * (lower + upper)
        below = (powerlaw_potential_difference(
            radius, middle, beta, softening) < enthalpy)
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return 0.5 * (lower + upper)


def equilibrium_vertical_extent(domain_mode=DOMAIN_ANNULUS, beta=-0.5,
                                softening=0.0,
                                geometry=GEOMETRY_SPHERICAL,
                                scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                                force_form=FORCE_POWERLAW, slab_half=None):
    """Maximum atmosphere/slab height over the mapped table's radial domain."""
    _envelope_inner, envelope_outer = equilibrium_envelope_bounds(domain_mode)
    radial_axis = np.linspace(0.0, np.sqrt(2.0) * envelope_outer,
                              EQUILIBRIUM_TABLE_NR)
    surface = equilibrium_surface_height(radial_axis, domain_mode, beta,
                                         softening, geometry, scale_height,
                                         force_form, slab_half)
    if validated_geometry(geometry) == GEOMETRY_CYLINDRICAL:
        # Exactly the half period: no margin, or the wrap would not close.
        return float(np.max(surface))
    return float(np.max(surface) + 0.02 * validated_scale_height(scale_height))


def equilibrium_density_table(domain_mode=DOMAIN_ANNULUS, beta=-0.5,
                              softening=0.0,
                              geometry=GEOMETRY_SPHERICAL,
                              scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                              force_form=FORCE_POWERLAW, slab_half=None):
    """Build the self-describing ``(r_cyl, |z|)`` glass target arrays."""
    _envelope_inner, envelope_outer = equilibrium_envelope_bounds(domain_mode)
    radial_axis = np.linspace(0.0, np.sqrt(2.0) * envelope_outer,
                              EQUILIBRIUM_TABLE_NR)
    height_axis = np.linspace(
        0.0, equilibrium_vertical_extent(domain_mode, beta, softening,
                                         geometry, scale_height, force_form,
                                         slab_half),
        EQUILIBRIUM_TABLE_NZ)
    values = equilibrium_density(radial_axis[:, None], height_axis[None, :],
                                 domain_mode, 1.0, beta, softening, geometry,
                                 scale_height, force_form, slab_half)
    return (radial_axis, height_axis), values


def equilibrium_target_mass(domain_mode=DOMAIN_ANNULUS, beta=-0.5,
                            softening=0.0,
                            geometry=GEOMETRY_SPHERICAL,
                            scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                            force_form=FORCE_POWERLAW, slab_half=None):
    """Integrate the table target over its rectangular periodic seed box."""
    (radial_axis, height_axis), values = equilibrium_density_table(
        domain_mode, beta, softening, geometry, scale_height, force_form,
        slab_half)
    _envelope_inner, half_width = equilibrium_envelope_bounds(domain_mode)
    angle = np.zeros_like(radial_axis)
    inside = radial_axis <= half_width
    angle[inside] = 2.0 * np.pi
    corners = ((radial_axis > half_width) &
               (radial_axis < np.sqrt(2.0) * half_width))
    angle[corners] = (2.0 * np.pi -
                      8.0 * np.arccos(half_width / radial_axis[corners]))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    vertical_integral = 2.0 * trapezoid(values, height_axis, axis=1)
    return float(trapezoid(vertical_integral * angle * radial_axis,
                           radial_axis))


def sample_equilibrium_positions(count, domain_mode=DOMAIN_ANNULUS,
                                 beta=-0.5, softening=0.0,
                                 geometry=GEOMETRY_SPHERICAL,
                                 scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                                 force_form=FORCE_POWERLAW, slab_half=None):
    """Draw an equal-mass pre-glass seed from the analytic target.

    Joint rejection sampling is used because the general-beta vertical
    profile is radius dependent.  Starting from this target leaves the glass
    relaxer to remove sampling noise instead of evacuating most of the box.
    """
    count = int(count)
    if count <= 0:
        raise ValueError("equilibrium seed count must be positive")
    _envelope_inner, extent = equilibrium_envelope_bounds(domain_mode)
    height_limit = equilibrium_vertical_extent(domain_mode, beta, softening,
                                               geometry, scale_height,
                                               force_form, slab_half)
    force_form = validated_force_form(force_form)
    # Every target this seeds is normalized to rho=1 at the science density.
    maximum_density = 1.0
    position_chunks = []
    remaining = count
    while remaining:
        trial_count = max(4096, 12 * remaining)
        trial_x = np.random.uniform(-extent, extent, trial_count)
        trial_y = np.random.uniform(-extent, extent, trial_count)
        trial_z = np.random.uniform(-height_limit, height_limit, trial_count)
        density = equilibrium_density(
            np.hypot(trial_x, trial_y), trial_z, domain_mode, 1.0, beta,
            softening, geometry, scale_height, force_form, slab_half)
        accept = np.random.random(trial_count) < density / maximum_density
        accepted = np.column_stack(
            (trial_x[accept], trial_y[accept], trial_z[accept]))
        if len(accepted):
            take = min(remaining, len(accepted))
            position_chunks.append(accepted[:take])
            remaining -= take
    return np.concatenate(position_chunks, axis=0)


def toroidal_field_strength(radius, field_amp, softening=0.0,
                            domain_mode=DOMAIN_ANNULUS,
                            radial_taper=False,
                            force_form=FORCE_POWERLAW,
                            field_mode=1):
    """Toroidal field magnitude for the selected radial domain and mode.

    Mode 1 is the current-free profile: the annulus uses ``C/R``, a full disk
    instead uses ``C R/(R^2+epsilon^2)``, which is divergence-free, vanishes
    linearly at the axis, and approaches ``C/R`` outside the softened core.
    The optional taper argument remains available for explicit experiments,
    but the powerlawring IC deliberately leaves this force-free profile
    untapered.

    Mode 4 is the CONSTANT ``B_phi = C`` on an annulus, and its center-regular
    continuation ``C R/sqrt(R^2+epsilon^2)`` on a full disk -- the same
    softening, and the same construction mode 1 already uses to reach the
    axis.  It is still divergence-free (any axisymmetric toroidal field is)
    and still exactly invariant under ideal induction in a purely azimuthal
    flow, because ``v x B = 0`` when both vectors are ``phi_hat`` -- for any
    rotation law, not only solid body.  What it is NOT is force-free:
    ``R B_phi`` is not constant, so it carries a current and an inward hoop
    stress the rotation has to balance.  That is the point of the mode: unlike
    mode 1 it exercises the momentum coupling, not just induction.

    The disk form is not a cosmetic regularization.  A strictly constant
    ``B_phi`` leaves the hoop term ``C^2/rho`` finite as ``R->0`` while gravity
    vanishes, so the equilibrium would need ``v_phi -> C`` and therefore
    ``Omega -> infinity`` on the axis: no rotation law balances it.  The
    softened form's hoop term falls as ``R^2``, so ``v_phi -> 0`` and the
    equilibrium stays regular at every plasma beta.  It costs only the core:
    ``B_phi/C`` is 0.9988 at ``R=1`` and 0.9997 at ``R=2`` for
    ``epsilon=0.05``, departing from constant only inside the softened core
    where mode 1 departs from ``C/R`` as well.
    """
    radius = np.asarray(radius, dtype=float)
    if int(field_mode) == 4:
        if int(domain_mode) == DOMAIN_DISK:
            if not np.isfinite(softening) or softening <= 0.0:
                raise ValueError("a full disk requires positive softening")
            field = (field_amp * radius /
                     np.sqrt(radius * radius + softening * softening))
        elif int(domain_mode) == DOMAIN_ANNULUS:
            field = np.full_like(radius, float(field_amp))
        else:
            raise ValueError("domain_mode must be 0 (annulus) or 1 (full disk)")
        if radial_taper:
            factor = radial_density_factor(radius, domain_mode)[0]
            field = field * np.sqrt(factor)
        return field
    if int(domain_mode) == DOMAIN_DISK:
        if not np.isfinite(softening) or softening <= 0.0:
            raise ValueError("a full disk requires positive softening")
        field = field_amp * radius / (radius * radius + softening * softening)
        if radial_taper:
            factor = radial_density_factor(radius, domain_mode)[0]
            field = field * np.sqrt(factor)
        return field
    if int(domain_mode) != DOMAIN_ANNULUS:
        raise ValueError("domain_mode must be 0 (annulus) or 1 (full disk)")
    with np.errstate(divide="raise", invalid="raise"):
        field = field_amp / radius
    if radial_taper:
        factor = radial_density_factor(radius, domain_mode)[0]
        field = field * np.sqrt(factor)
    return field


def validated_plasma_beta(plasma_beta):
    """Return the reference plasma beta, the test's one magnetization control."""
    value = float(plasma_beta)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("plasma_beta must be finite and positive; got %r"
                         % (plasma_beta,))
    return value


def field_shape_at_reference(field_mode, softening=0.0,
                             domain_mode=DOMAIN_ANNULUS):
    """Field magnitude at ``R=1`` for unit amplitude, per field mode.

    Each mode is normalized through its own profile rather than by a
    hand-copied constant, so the plasma-beta conversion cannot drift if a
    profile is changed.  ``B_R=C/R`` and ``B_z=C`` are both ``C`` here.
    """
    field_mode = int(field_mode)
    if field_mode == 0:
        return 0.0
    if field_mode in (1, 4):
        return float(toroidal_field_strength(
            np.array([FIELD_REFERENCE_RADIUS]), 1.0, softening,
            domain_mode, field_mode=field_mode)[0])
    if field_mode in (2, 3):
        # B_R = C/R and B_z = C are both C at the reference radius.
        return 1.0
    raise ValueError("field_mode must be one of 0, 1, 2, 3, 4; got %r"
                     % (field_mode,))


def field_amplitude_from_plasma_beta(plasma_beta, field_mode=0,
                                     softening=0.0,
                                     domain_mode=DOMAIN_ANNULUS,
                                     scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                                     rho_reference=1.0):
    """Magnetic constant ``C`` that puts ``plasma_beta`` at ``R=1``.

    ``beta_pl = 2 P / B^2`` at the science density, so
    ``B(R=1) = sqrt(2 P(rho=1) / beta_pl)`` and ``C`` follows from the mode's
    profile shape.  Holding the AMPLITUDE fixed is the trap this replaces:
    ``P`` scales as ``H^2``, so a temperature scan at fixed ``C`` silently
    scans the magnetization as ``H^2`` (a factor 2500 over H=0.01..0.5).
    Beta is only uniform where the profile is: modes 1 and 2 fall as ``1/R``
    so their local beta rises as ``R^2``, and the full disk's toroidal core
    dips about two decades below the reference value at ``R=epsilon``, which
    is where the kinematic assumption actually binds.  Hydro
    (``field_mode=0``) carries no field, so its amplitude is zero whatever
    beta is requested.
    """
    if int(field_mode) == 0:
        return 0.0
    plasma_beta = validated_plasma_beta(plasma_beta)
    pressure = float(equilibrium_pressure_from_density(rho_reference,
                                                      scale_height))
    reference_field = np.sqrt(2.0 * pressure / plasma_beta)
    return float(reference_field /
                 field_shape_at_reference(field_mode, softening, domain_mode))


def toroidal_lorentz_v2(radius, density, field_amp, softening=0.0,
                        domain_mode=DOMAIN_ANNULUS, field_mode=1):
    """Toroidal magnetic contribution to radial equilibrium in code units.

    Gasoline uses magnetic pressure ``B^2/2``.  This returns

    ``[B_phi^2 + (R/2) d(B_phi^2)/dR] / rho``

    for the actual untapered equilibrium field.  The annular ``C/R`` profile
    cancels exactly.  The full-disk regularization retains only its resolved
    core-current term.  The constant field (mode 4) has ``d(B_phi^2)/dR = 0``
    on an annulus and so leaves the bare hoop term ``C^2/rho``: the field line
    tension pulls inward, so the equilibrium rotation is FASTER, by
    ``2 P / plasma_beta`` at the science density -- negligible at the high beta
    the induction tests run at, and a percent-level correction only as beta
    approaches unity.  Its center-regular disk form gives
    ``C^2 R^2 (R^2 + 2 eps^2) / (R^2 + eps^2)^2``, which tends to ``C^2``
    outside the core and to ``2 C^2 R^2/eps^2`` on the axis -- vanishing there,
    which is exactly what keeps ``Omega`` finite at the centre.
    """
    radius = np.asarray(radius, dtype=float)
    density = np.asarray(density, dtype=float)
    if np.any(density <= 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("toroidal Lorentz balance requires positive density")
    if int(field_mode) == 4:
        if int(domain_mode) == DOMAIN_DISK:
            if not np.isfinite(softening) or softening <= 0.0:
                raise ValueError("a full disk requires positive softening")
            radius2 = radius * radius
            eps2 = softening * softening
            numerator = (float(field_amp) ** 2 * radius2 *
                         (radius2 + 2.0 * eps2) / (radius2 + eps2) ** 2)
        elif int(domain_mode) == DOMAIN_ANNULUS:
            numerator = np.full_like(radius, float(field_amp) ** 2)
        else:
            raise ValueError("domain_mode must be 0 (annulus) or 1 (full disk)")
        return numerator / density
    if int(domain_mode) == DOMAIN_ANNULUS:
        numerator = np.zeros_like(radius)
    elif int(domain_mode) == DOMAIN_DISK:
        if not np.isfinite(softening) or softening <= 0.0:
            raise ValueError("a full disk requires positive softening")
        radius2 = radius * radius
        denominator = radius2 + softening * softening
        numerator = (2.0 * field_amp ** 2 * radius2 * softening ** 2 /
                     denominator ** 3)
    else:
        raise ValueError("domain_mode must be 0 (annulus) or 1 (full disk)")
    return numerator / density


def equilibrium_rotation_terms(radius, height, field_mode=0,
                               field_amp=0.0, softening=0.0,
                               domain_mode=DOMAIN_ANNULUS, beta=-0.5,
                               geometry=GEOMETRY_SPHERICAL,
                               scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                               force_form=FORCE_POWERLAW, slab_half=None):
    """Gravity, pressure and magnetic terms in equilibrium ``v_phi^2``."""
    radius, height = np.broadcast_arrays(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float))
    geometry = validated_geometry(geometry)
    scale_height = validated_scale_height(scale_height)
    force_form = validated_force_form(force_form)
    if force_form == FORCE_BOUNDED_KEPLER and (int(field_mode) != 0 or
                                               float(field_amp) != 0.0):
        raise ValueError("the Hopkins ring is the hydrodynamic literature test")
    # The Hopkins ring needs no branch of its own.  Its density is flat, so the
    # pressure term vanishes identically and v_phi^2 reduces to the bare
    # gravity term below -- which for beta=-1/2 and no softening is 1/R, i.e.
    # the published rotation law, because the force's modified branches lie
    # entirely outside the emitted ring.
    density = equilibrium_density(radius, height, domain_mode, 1.0, beta,
                                  softening, geometry, scale_height,
                                  force_form, slab_half)
    _midplane, _enthalpy, _factor, denthalpy_dradius = (
        _midplane_density_and_enthalpy(radius, domain_mode, 1.0, scale_height,
                                       geometry, force_form))
    midplane2 = radius * radius + float(softening) ** 2
    # Both geometries have the same midplane radial acceleration.  In the
    # spherical field vertical hydrostatic balance cancels the height
    # dependence exactly, leaving R*d[h(R,0)]/dR just as in the cylindrical
    # slab.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        gravity = radius * radius * midplane2 ** (float(beta) - 1.0)
    pressure = radius * denthalpy_dradius
    magnetic = np.zeros_like(radius)
    if int(field_mode) in (1, 4) and float(field_amp) != 0.0:
        magnetic = toroidal_lorentz_v2(
            radius, density, field_amp, softening, domain_mode,
            field_mode=int(field_mode))
    return {
        "density": density,
        "gravity": gravity,
        "pressure": pressure,
        "magnetic": magnetic,
        "total": gravity + pressure + magnetic,
    }


def equilibrium_azimuthal_speed(radius, height, field_mode=0,
                                field_amp=0.0, softening=0.0,
                                domain_mode=DOMAIN_ANNULUS, beta=-0.5,
                                geometry=GEOMETRY_SPHERICAL,
                                scale_height=EQUILIBRIUM_SCALE_HEIGHT,
                                force_form=FORCE_POWERLAW, slab_half=None):
    """Return the complete pressure- and Lorentz-balanced azimuthal speed."""
    terms = equilibrium_rotation_terms(radius, height, field_mode, field_amp,
                                       softening, domain_mode, beta, geometry,
                                       scale_height, force_form, slab_half)
    total = terms["total"]
    if not np.all(np.isfinite(total)) or np.any(total < 0.0):
        minimum = float(np.nanmin(total))
        raise ValueError("equilibrium taper gives invalid v_phi^2 "
                         "(minimum %g)" % minimum)
    return np.sqrt(total)


def rotate_x(vectors, angle_degrees):
    """Rotate one vector or an ``(N,3)`` array about the x axis."""
    vectors = np.asarray(vectors, dtype=float)
    angle = np.deg2rad(float(angle_degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.array(vectors, dtype=float, copy=True)
    y = np.array(vectors[..., 1], copy=True)
    z = np.array(vectors[..., 2], copy=True)
    result[..., 1] = cosine * y - sine * z
    result[..., 2] = sine * y + cosine * z
    return result


class setup_powerlawring(object):
    """Cold, non-self-gravitating disk/annulus for orbit and induction tests."""

    dICdensRsmooth = 0.05
    dICdensprofile = 1
    dICdensdir = 1
    dICdensR = 1.0
    dICdensinner = 1.0
    dICdensouter = 1.0
    rhoit = 0
    ns = 64
    ndim = 3
    time = 0
    gamma = 5.0 / 3.0
    inflow = 0
    periodic = 0
    # Omega(R=1)=1 for every beta, so the unit-radius orbital period is 2*pi:
    # 40 base steps per orbit, an output every eighth of one, 20 orbits at R=1.
    # The output cadence is set by the beta=1 breathing mode, whose period is
    # pi (kappa=2 at every radius, so it never phase mixes): 8 outputs per
    # orbit is 4 samples per period, which recovers its amplitude at any
    # phase.  At 4 outputs per orbit it sat exactly on Nyquist and the
    # measured amplitude depended entirely on sampling phase.
    # Only beta=1 turns as a solid body -- at beta=-0.5 the same run is 56
    # orbits at R=0.5 and 7 at R=2, which is what the outer disk needs to be
    # sampled at all.
    # deltastep is the BASE rung, not the integration accuracy: bEpsAccStep
    # with dEta=0.175 sets the per-particle step (~0.039 here) and particles
    # simply drop rungs under a coarser base, so this controls output
    # bookkeeping and cost far more than it controls error.
    deltastep = 2.0 * np.pi / 40.0
    freqout = 5
    nsteps = 800
    dmsolunit = 1.0
    dkpcunit = 1.0
    adi = 1
    grav = 1
    cosmo = 0
    molweight = 1.0
    shape = 2
    rhozero = 1.0
    rcylmin2 = R_INNER ** 2
    rcylmax2 = R_OUTER ** 2
    rmin2 = 0.0
    rmax2 = 10.0 ** 22

    def __init__(self):
        self.rhozero = type(self).rhozero
        self.npart = self.ngas = self.ndark = self.nstar = 0
        self.mass = []
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.rho = []
        self.u = []
        self.h = []
        self.P = []
        self.Bx = []
        self.By = []
        self.Bz = []
        self.Bpsi = []
        self.soft = []
        self.tform = []
        self.metals = []
        self.dxbound = self.dybound = self.dzbound = 0.0
        self.totvol = 0.0

        # Values consumed by IC_createparamfile.py for the runtime force.
        self.powerlaw_beta = -0.5
        self.powerlaw_softening = 0.0
        self.powerlaw_geometry = 3
        self.powerlaw_field_mode = 0
        self.powerlaw_field_amp = 0.0
        self.powerlaw_plasma_beta = 1.0e6
        self.powerlaw_tilt = 0.0
        self.powerlaw_domain_mode = DOMAIN_ANNULUS
        self.powerlaw_scale_height = EQUILIBRIUM_SCALE_HEIGHT
        self.powerlaw_force_form = FORCE_POWERLAW
        # Half the periodic z period; create() sets it from nx.
        self.powerlaw_slab_half = None
        self.glass_density_table = None
        self._source_mass = None

    def getrhoi(self, _index):
        return self.rhozero

    @staticmethod
    def _validated_integer(name, value, allowed):
        numeric = float(value)
        integer = int(round(numeric))
        if numeric != integer or integer not in allowed:
            choices = ", ".join(str(item) for item in sorted(allowed))
            raise ValueError("%s must be one of %s; got %r" %
                             (name, choices, value))
        return integer

    def _place_particles(self, nx, distri, entry, half_height):
        _envelope_inner, extent = equilibrium_envelope_bounds(
            self.powerlaw_domain_mode)
        retained_inner, retained_outer = retained_particle_bounds(
            self.powerlaw_domain_mode, self.powerlaw_geometry,
            self.powerlaw_force_form)
        # Keep nx as the resolution across the original R=2 science diameter;
        # the retained equilibrium buffer adds particles rather than coarsening
        # the disk itself.
        spacing = 2.0 * R_OUTER / nx
        preglass = bool(getattr(self, "preglass", False))
        cylindrical = self.powerlaw_geometry == GEOMETRY_CYLINDRICAL
        slab = cylindrical and not preglass
        #
        # CYLINDRICAL: relax a PLAIN UNIFORM glass in a periodic box whose z
        # period is the run's z period, then carve the ring out of it.  No
        # envelope, no rho_edge, no cull, an exact mass contract, and a
        # seam-free wrap because the relaxation was periodic in z too.  With no
        # a_z the constant-density slab is the exact vertical solution, and a
        # radial taper would only soften a free surface whose disturbance has
        # amplitude c_s whatever its shape -- so the ring has sharp edges.
        # Both force laws share this glass: the relaxer sees only rho=1 in a
        # box, and neither the temperature nor the force reaches it.
        #
        # SPHERICAL: unchanged.  Its pre-glass spans the whole mapped target,
        # disposable envelope skirt included, and the rebuilt IC keeps only the
        # fixed buffer around the science domain.  The skirt exists so the
        # relaxer's resolution-dependent rho_edge contour has somewhere
        # harmless to sit; its steep log-density ramp is far below any intended
        # mass resolution and its enthalpy gradient is too strong for the orbit
        # to balance at small beta (an unphysical, negative v_phi^2 at
        # beta=-0.5), so it must not reach the emitted IC.  Cropping to the
        # buffer also makes the retained range exactly the range create()
        # validates.
        #
        if cylindrical:
            # The periodic seed box is the BOUNDING BOX of what is emitted:
            # 2.0, the ring's own outer edge, which is also the paper's box,
            # -2 < x,y < 2.  A larger box would relax particles the carve
            # throws away.  create()'s uniform seed mass reads the same bound,
            # so rho stays exactly one.
            seed_extent = retained_outer
            seed_height = self.powerlaw_slab_half
        else:
            seed_extent, seed_height = extent, half_height
        retained = seed_extent if preglass else retained_outer
        crop_height = seed_height if (preglass or slab) else half_height
        distribute.setbound(self, -retained, retained, -retained, retained,
                            -crop_height, crop_height)
        if slab:
            self.periodic = 1
            self.dxPeriod = 0.0
            self.dyPeriod = 0.0
            self.dzPeriod = 2.0 * self.powerlaw_slab_half
        if preglass:
            self.rcylmin2 = 0.0
            self.rcylmax2 = 10.0 ** 22
        else:
            self.rcylmin2 = retained_inner ** 2
            self.rcylmax2 = retained_outer ** 2 * (
                1.0 + 8.0 * np.finfo(float).eps)

        if distri == 0:
            if not (cylindrical and
                    self.powerlaw_force_form == FORCE_BOUNDED_KEPLER and not preglass):
                raise ValueError(
                    "a lattice start is implemented for the Hopkins ring "
                    "(-i 1) only; every other powerlawring target is "
                    "relaxed from a glass")
            # Hopkins starts the disc from an ORDERED grid -- "the initial
            # particles are evenly spaced such that the square domain from
            # -2 < x < 2, -2 < y < 2 has nx^2 particles; then particles are
            # removed inside r < 0.5 and outside r > 2".  That is exactly this:
            # lay the lattice over the square and let checkboundary carve the
            # ring (rcylmin2/rcylmax2 above are already 0.5 and 2).  The ring
            # is FLAT, so there is no stretch map to apply -- deleting
            # particles IS the profile.
            #
            # periodic=True keeps the z wrap seamless (setcloseddist otherwise
            # overshoots the box by half a cell); the in-plane overshoot it
            # wraps falls outside the annulus and is carved away either way.
            # The close-packed layer spacing is dx*sqrt(6)/3 and the period
            # is 2*SLAB_HALF_HEIGHT_SPACINGS*dx, so the slab holds EXACTLY 12
            # layers (22 cubic planes) at every nx -- the lattice tiles the z
            # period with no seam by construction, exactly as orzag does.
            distribute.setcloseddist(self, -retained_outer, retained_outer,
                                     -retained_outer, retained_outer,
                                     -crop_height, crop_height, spacing,
                                     clip_to_rmax=False, periodic=True)
            print("powerlawring Hopkins lattice: npart=%d carved to "
                  "R=%.2f..%.2f" %
                  (len(self.x), retained_inner, retained_outer))
        elif distri == 1:
            if (preglass and cylindrical and
                    self.powerlaw_force_form in (FORCE_POWERLAW,
                                                FORCE_BOUNDED_KEPLER)):
                # Uniform periodic seed: the relaxer's target is rho=1 over the
                # whole box (dICdensprofile=1), so no density table is needed.
                distribute.setrandomdist(self, -seed_extent, seed_extent,
                                         -seed_extent, seed_extent,
                                         -seed_height, seed_height, spacing)
                print("powerlawring uniform-ring preglass: periodic box "
                      "%.3f^2 x %.3f, npart=%d"
                      % (2.0 * seed_extent, 2.0 * seed_height, len(self.x)))
            elif preglass:
                # Preserve the meaning of nx in the SCIENCE region: the
                # in-plane spacing there is exactly `spacing`, i.e.
                # m = rhozero*spacing^3.  Counting a uniform bounding box
                # would spend most particles oversampling the Gaussian/skirt
                # vacuum and would silently change the core resolution when
                # the disposable envelope extent changes.  Every target is
                # normalized to rho=1, so the rhozero factor is the plain
                # spacing^3.
                target_mass = equilibrium_target_mass(
                    self.powerlaw_domain_mode, self.powerlaw_beta,
                    self.powerlaw_softening, self.powerlaw_geometry,
                    self.powerlaw_scale_height, self.powerlaw_force_form,
                    self.powerlaw_slab_half)
                count = max(self.ns + 1,
                            int(round(target_mass /
                                      (self.rhozero * spacing ** 3))))
                seed = sample_equilibrium_positions(
                    count, self.powerlaw_domain_mode, self.powerlaw_beta,
                    self.powerlaw_softening, self.powerlaw_geometry,
                    self.powerlaw_scale_height, self.powerlaw_force_form,
                    self.powerlaw_slab_half)
                self.x = seed[:, 0].tolist()
                self.y = seed[:, 1].tolist()
                self.z = seed[:, 2].tolist()
                self.rho = equilibrium_density(
                    np.hypot(seed[:, 0], seed[:, 1]), seed[:, 2],
                    self.powerlaw_domain_mode, 1.0, self.powerlaw_beta,
                    self.powerlaw_softening, self.powerlaw_geometry,
                    self.powerlaw_scale_height, self.powerlaw_force_form,
                    self.powerlaw_slab_half).tolist()
                print("powerlawring equilibrium target-sampled preglass: "
                      "npart=%d" % count)
        else:
            gas, _dark, _star, _header, _time = tip.readtipsy(entry)
            radius2 = gas[:, 1] ** 2 + gas[:, 2] ** 2
            keep = ((radius2 >= retained_inner ** 2) &
                    (radius2 <= retained_outer ** 2) &
                    (gas[:, 3] >= -crop_height) &
                    (gas[:, 3] < crop_height))
            selected = gas[keep, 1:4].astype(float)
            self.x = selected[:, 0].tolist()
            self.y = selected[:, 1].tolist()
            self.z = selected[:, 2].tolist()
            self._source_mass = gas[keep, 0].astype(float)
            self.rho = gas[keep, 7].astype(float).tolist()

        if not self.x:
            raise ValueError("powerlawring particle placement produced no disk")
        return spacing, preglass

    def create(self, nx, distri=0, vm=0, entry=" ", beta=-0.5,
               geometry=3, field_mode=0, plasma_beta=1.0e6, softening=0.0,
               tilt=0.0, domain_mode=DOMAIN_ANNULUS,
               scale_height=EQUILIBRIUM_SCALE_HEIGHT,
               force_form=FORCE_POWERLAW):
        """Create the annulus or center-resolving full disk.

        ``geometry`` is 2 (cylindrical: no vertical force, a constant-density
        slab) or 3 (spherical: a vertically stratified hydrostatic atmosphere,
        the only geometry in which a tilted orbital plane is meaningful).

        ``scale_height`` is the barotrope temperature and the ONLY control on
        the orbital Mach number.  For geometry 2 it is decoupled from the slab
        thickness, so the classic cold Keplerian-ring regime (Mach >~ 20) is
        reachable at any resolution; for geometry 3 it also sets the vertical
        thickness and so must stay resolved.

        ``force_form=0`` is the softened power law: its core is regularized
        by ``softening`` and nothing confines the outer edge.

        ``force_form=1`` is the Hopkins (2015) cold Keplerian ring, the test
        Groth et al. (2023) inherited.  It builds the SAME sharp-edged ring as
        force_form=0 -- the paper carves its disc the same way, by removing
        particles inside R=0.5 and outside R=2 from an evenly filled square --
        and differs only in fixing the published temperature
        (``hopkins_scale_height``) and selecting the Hopkins runtime force.
        That force equals the plain unsoftened beta=-1/2 power law everywhere
        a particle is placed; its modified branches at R<0.35 and R>2.1 exist
        for the paper's mesh arm and, here, only to catch whatever drifts out
        of the ring during the run.  It is a cylindrical, hydrodynamic,
        untilted, beta=-1/2 annulus by definition.

        ``field_mode`` is 0 (hydro), 1 (toroidal: current-free ``B_phi=C/R``
        in an annulus and center-regular in a full disk), 2 (annulus-only
        radial ``B_R=C/R`` for analytic differential winding), 3 (uniform
        field normal to the orbital plane), or 4 (CONSTANT
        ``B_phi=C``, continued to the axis on a full disk as the
        center-regular ``C R/sqrt(R^2+epsilon^2)``).  Mode 4 is exactly
        invariant under ideal induction for every rotation law, like mode 1,
        but unlike mode 1 it is not force-free: it carries a current and an
        inward hoop stress that the rotation balances, so it tests the
        momentum coupling as well.  Every configuration is a glass-relaxed,
        pressure- and Lorentz-balanced radial equilibrium.

        ``plasma_beta`` is the magnetization, quoted as ``2P/B^2`` at ``R=1``
        in the science domain, and the amplitude ``C`` is derived from it.  It
        replaces the raw amplitude because ``P`` scales as ``H^2``: at fixed
        ``C`` a temperature scan is also an unintended magnetization scan.
        The suite's induction tests are kinematic, which is exactly the
        statement ``plasma_beta >> 1``.
        """
        nx_float = float(nx)
        nx = int(round(nx_float))
        if nx_float != nx or nx < 8:
            raise ValueError("nx must be an integer >= 8")
        if float(vm) != 0.0:
            raise ValueError("powerlawring currently requires equal particle masses")

        beta = float(beta)
        plasma_beta = validated_plasma_beta(plasma_beta)
        softening = float(softening)
        tilt = float(tilt)
        geometry = validated_geometry(geometry)
        scale_height = validated_scale_height(scale_height)
        field_mode = self._validated_integer("field_mode", field_mode,
                                             {0, 1, 2, 3, 4})
        domain_mode = self._validated_integer("domain_mode", domain_mode,
                                              {DOMAIN_ANNULUS, DOMAIN_DISK})
        force_form = validated_force_form(force_form)
        finite = (beta, softening, tilt)
        if not np.all(np.isfinite(finite)):
            raise ValueError("all powerlawring parameters must be finite")
        if beta <= -1.0:
            raise ValueError("powerlawring requires beta > -1 for stable circular orbits")
        if softening < 0.0:
            raise ValueError("softening must be non-negative")
        if domain_mode == DOMAIN_DISK and softening <= 0.0:
            raise ValueError("a full disk requires positive softening")
        if domain_mode == DOMAIN_DISK and field_mode == 2:
            raise ValueError("the C/R radial winding field is annulus-only")
        if geometry == GEOMETRY_CYLINDRICAL and tilt != 0.0:
            raise ValueError("an orbital-plane tilt requires spherical "
                             "geometry 3; a cylindrical field always points "
                             "at the z axis")
        # The periodic z box is GEOMETRY and scales with the resolution
        # (2*sqrt(6) spacings per half period, the suite's thin-box rule), so
        # it has to be known before any cylindrical density normalization.
        self.powerlaw_slab_half = slab_half_height(nx)
        if force_form == FORCE_BOUNDED_KEPLER:
            if geometry != GEOMETRY_CYLINDRICAL:
                raise ValueError("the Hopkins ring requires cylindrical geometry 2")
            if domain_mode != DOMAIN_ANNULUS:
                raise ValueError("the Hopkins ring requires domain_mode=0 (annulus)")
            if beta != -0.5:
                raise ValueError("the Hopkins ring requires beta=-0.5")
            if softening != 0.0:
                raise ValueError("the Hopkins force is unsoftened; use softening=0")
            if field_mode != 0:
                raise ValueError("the Hopkins ring is hydrodynamic; use field_mode=0")
            if tilt != 0.0:
                raise ValueError("the Hopkins ring is untilted; use tilt=0")
            # The published temperature, not a free knob: P=1e-6 on Sigma=1.01.
            # rhozero stays 1 -- the ring is flat and its normalization is
            # arbitrary in a non-self-gravitating test, so only P/rho matters.
            scale_height = hopkins_scale_height(self.rhozero)
        retained_inner, retained_outer = retained_particle_bounds(
            domain_mode, geometry, force_form)
        check_radius = np.linspace(retained_inner, retained_outer, 2049)
        hydro_check = equilibrium_rotation_terms(
            check_radius, np.zeros_like(check_radius), softening=softening,
            domain_mode=domain_mode, beta=beta, geometry=geometry,
            scale_height=scale_height, force_form=force_form,
            slab_half=self.powerlaw_slab_half)["total"]
        if not np.all(np.isfinite(hydro_check)) or np.any(hydro_check < 0.0):
            raise ValueError("the requested beta/softening gives negative "
                             "v_phi^2 in the fixed radial equilibrium taper")
        # Hopkins initialises on an evenly spaced grid, so the literature
        # ring also accepts an ordered start: runtest distribution 0 (FCC) and
        # 3 (cubic) both arrive here as distri==0, the kind having been
        # selected globally by IC_createsetup.  Every other target is a
        # relaxed glass.
        if (not bool(getattr(self, "preglass", False)) and distri != 2 and
                not (distri == 0 and force_form == FORCE_BOUNDED_KEPLER)):
            raise ValueError("powerlawring requires a glass IC "
                             "(runtest distribution 2), or distribution 0/3 "
                             "with the Hopkins ring (-i 1)")

        # ONE conversion, after every input the amplitude depends on has been
        # validated. Everything downstream (profiles, the .param, the run) keeps
        # working in the amplitude C; only the user-facing knob is the beta.
        field_amp = field_amplitude_from_plasma_beta(
            plasma_beta, field_mode, softening, domain_mode, scale_height)

        self.powerlaw_beta = beta
        self.powerlaw_geometry = geometry
        self.powerlaw_field_mode = field_mode
        self.powerlaw_field_amp = field_amp
        self.powerlaw_plasma_beta = plasma_beta
        self.powerlaw_softening = softening
        self.powerlaw_tilt = tilt
        self.powerlaw_domain_mode = domain_mode
        self.powerlaw_scale_height = scale_height
        # The one number the .param needs: gasoline's iPowerLawPotentialForm
        # uses the same 0/1 spelling, so nothing has to be remapped.
        self.powerlaw_force_form = force_form

        spacing = 2.0 * R_OUTER / nx
        half_height = equilibrium_vertical_extent(domain_mode, beta, softening,
                                                  geometry, scale_height,
                                                  force_form,
                                                  self.powerlaw_slab_half)
        spacing, preglass = self._place_particles(nx, distri, entry,
                                                  half_height)

        # A cylindrical target is always a sharp ring at rho=1, so it relaxes
        # a plain uniform box and needs no table.  Only the spherical
        # atmosphere hands the relaxer its own profile.
        if preglass and geometry != GEOMETRY_CYLINDRICAL:
            axes, values = equilibrium_density_table(
                domain_mode, beta, softening, geometry, scale_height,
                force_form, self.powerlaw_slab_half)
            self.glass_density_table = (axes, values, ('r_cyl', 'z'))

        # The pre-glass is only a target-sampled relaxation seed. Its velocities and
        # fields are harmless placeholders and its .param omits the potential.
        local_positions = np.column_stack((self.x, self.y, self.z)).astype(float)
        positions = local_positions.copy()
        if not preglass:
            positions = rotate_x(positions, tilt)
        self.x, self.y, self.z = (positions[:, i].tolist() for i in range(3))

        self.npart = len(self.x)
        self.ngas = self.npart
        if (preglass and geometry == GEOMETRY_CYLINDRICAL and
                force_form in (FORCE_POWERLAW, FORCE_BOUNDED_KEPLER)):
            # A uniform seed box: its mass is simply rho * volume.  The
            # bound is _place_particles' seed_extent, not the support radius.
            _inner, outer = retained_particle_bounds(domain_mode, geometry,
                                                     force_form)
            total_mass = (self.rhozero * (2.0 * outer) ** 2
                          * 2.0 * self.powerlaw_slab_half)
            self.mass = np.full(self.npart, total_mass / self.npart)
            self.rho = np.full(self.npart, self.rhozero)
        elif preglass:
            total_mass = equilibrium_target_mass(domain_mode, beta, softening,
                                                 geometry, scale_height,
                                                 force_form,
                                                 self.powerlaw_slab_half)
            self.mass = np.full(self.npart, total_mass / self.npart)
            self.rho = np.full(self.npart, self.rhozero)
        elif distri == 0:
            # Ordered lattice, carved to the ring: equal masses by
            # construction and a flat profile, so the total is simply
            # rho * the annulus volume.  The slab factor cancels out of rho.
            _inner, _outer = retained_particle_bounds(domain_mode, geometry,
                                                      force_form)
            total_mass = (self.rhozero * np.pi
                          * (_outer ** 2 - _inner ** 2)
                          * 2.0 * self.powerlaw_slab_half)
            self.mass = np.full(self.npart, total_mass / self.npart)
            radius = np.hypot(local_positions[:, 0], local_positions[:, 1])
            self.rho = equilibrium_density(
                radius, local_positions[:, 2], domain_mode, 1.0, beta,
                softening, geometry, scale_height, force_form,
                self.powerlaw_slab_half)
        else:
            if self._source_mass is None or len(self._source_mass) != self.npart:
                raise ValueError("relaxed equilibrium glass did not supply masses")
            if not np.allclose(self._source_mass, self._source_mass[0],
                               rtol=2.0e-6, atol=0.0):
                raise ValueError("equilibrium glass must have equal particle masses")
            self.mass = self._source_mass.copy()
            radius = np.hypot(local_positions[:, 0], local_positions[:, 1])
            self.rho = equilibrium_density(
                radius, local_positions[:, 2], domain_mode, 1.0, beta,
                softening, geometry, scale_height, force_form,
                self.powerlaw_slab_half)
        self.h = smth.getsmooth(self, self.ns)
        if not preglass and geometry == GEOMETRY_CYLINDRICAL:
            # z is periodic here, so a pair must not fall inside one kernel
            # both directly and through the wrap: the support diameter 2h has
            # to stay below half the z period.  This is the HARD contract and
            # it is measured at the science density, which is what the test
            # resolves.  Since the slab is 2*sqrt(6) spacings and 2h is
            # 2.72 spacings there, the check is resolution-independent and
            # passes by construction -- it stays as a guard against a future
            # change to either constant, or an unusually large nSmooth.
            slab_half = self.powerlaw_slab_half
            support = 2.0 * float(smth.getsmooth2(
                float(np.max(self.mass)), self.rhozero, self.ns))
            if support >= slab_half:
                raise ValueError(
                    "the periodic cylindrical slab needs a kernel support "
                    "radius (%g) below half its z period (%g): raise nx or "
                    "lower nSmooth" % (support, slab_half))
            # It is kept as a guard rather than a live check: a cylindrical
            # ring is flat, so 2h is the science value everywhere and nothing
            # trips it.  A low-density band would, since h grows as rho^(-1/3),
            # which is a property of the density contrast and not of the slab;
            # the report below names the emitted mass that lands in one.
            wrapped = 2.0 * np.asarray(self.h, dtype=float) >= slab_half
            if np.any(wrapped):
                wrapped_radius = np.hypot(local_positions[wrapped, 0],
                                          local_positions[wrapped, 1])
                print("powerlawring: NOTE %.2f%% of the emitted mass has "
                      "2h >= half the z period (%g), at R=%.3f..%.3f -- its "
                      "kernel wraps more than once in z.  The science density "
                      "is clear by a factor %.2f."
                      % (100.0 * np.sum(self.mass[wrapped]) /
                         np.sum(self.mass), slab_half,
                         float(np.min(wrapped_radius)),
                         float(np.max(wrapped_radius)),
                         slab_half / support))

        if preglass:
            velocity = np.zeros((self.npart, 3), dtype=float)
            magnetic = np.zeros_like(velocity)
        else:
            local_position = rotate_x(positions, -tilt)
            xlocal, ylocal = local_position[:, 0], local_position[:, 1]
            radius = np.hypot(xlocal, ylocal)
            magnetic = np.zeros((self.npart, 3), dtype=float)
            if field_mode in (1, 4):
                bphi = toroidal_field_strength(
                    radius, field_amp, softening, domain_mode,
                    radial_taper=False, force_form=force_form,
                    field_mode=field_mode)
                inverse_radius = np.divide(
                    1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
                magnetic[:, 0] = -bphi * ylocal * inverse_radius
                magnetic[:, 1] = bphi * xlocal * inverse_radius
            elif field_mode == 2:
                magnetic[:, 0] = field_amp * xlocal / radius ** 2
                magnetic[:, 1] = field_amp * ylocal / radius ** 2
            elif field_mode == 3:
                magnetic[:, 2] = field_amp

            terms = equilibrium_rotation_terms(
                radius, local_position[:, 2], field_mode, field_amp,
                softening, domain_mode, beta, geometry, scale_height,
                force_form)
            total_v2 = terms["total"]
            if not np.all(np.isfinite(total_v2)) or np.any(total_v2 < 0.0):
                raise ValueError("equilibrium taper gives invalid v_phi^2 "
                                 "(minimum %g)" % np.nanmin(total_v2))
            vphi = np.sqrt(total_v2)
            inverse_radius = np.divide(
                1.0, radius, out=np.zeros_like(radius), where=radius > 0.0)
            velocity = np.column_stack((-vphi * ylocal * inverse_radius,
                                         vphi * xlocal * inverse_radius,
                                         np.zeros(self.npart)))
            print("powerlawring equilibrium: H=%g rho_cut=%g zmax=%g "
                  "buffer_floor=%g envelope_floor=%g force=%d "
                  "v2(gravity/pressure/magnetic) ranges "
                  "[%g,%g]/[%g,%g]/[%g,%g]" %
                  (scale_height,
                   EQUILIBRIUM_CUTOFF_DENSITY, half_height,
                   EQUILIBRIUM_RADIAL_FLOOR,
                   EQUILIBRIUM_ENVELOPE_FLOOR, force_form,
                   np.min(terms["gravity"]), np.max(terms["gravity"]),
                   np.min(terms["pressure"]), np.max(terms["pressure"]),
                   np.min(terms["magnetic"]), np.max(terms["magnetic"])))
            velocity = rotate_x(velocity, tilt)
            magnetic = rotate_x(magnetic, tilt)

        self.vx, self.vy, self.vz = (velocity[:, i].tolist() for i in range(3))
        self.Bx, self.By, self.Bz = (magnetic[:, i].tolist() for i in range(3))
        density = np.asarray(self.rho, dtype=float)
        pressure = np.asarray(
            equilibrium_pressure_from_density(density, scale_height),
            dtype=float)
        self.P = pressure.tolist()
        internal_energy = np.divide(
            pressure, (self.gamma - 1.0) * density,
            out=np.zeros_like(pressure), where=density > 0.0)
        self.u = internal_energy.tolist()
        if not preglass and field_mode:
            magnetic2 = np.sum(magnetic * magnetic, axis=1)
            particle_plasma_beta = np.divide(
                2.0 * pressure, magnetic2,
                out=np.full_like(pressure, np.inf), where=magnetic2 > 0.0)
            alfven2 = np.divide(
                magnetic2, density,
                out=np.zeros_like(pressure), where=density > 0.0)
            print("powerlawring equilibrium magnetic: min(plasma_beta)=%g "
                  "max(v_A)=%g" %
                  (np.min(particle_plasma_beta), np.sqrt(np.max(alfven2))))

        # c_s at the science density, through the same one definition.
        reference_density = 1.0
        sound_speed = float(np.sqrt(
            self.gamma * equilibrium_pressure_from_density(
                reference_density, scale_height) / reference_density))
        print("powerlawring: npart=%d beta=%g geometry=%d field_mode=%d "
              "softening=%g tilt=%g domain=%d H=%g force=%d "
              "plasma_beta=%g (C=%.4g) "
              "(c_s=%.4g, Mach=%.1f "
              "at R=1)" %
              (self.npart, beta, geometry, field_mode, softening, tilt,
               domain_mode, scale_height, force_form, plasma_beta, field_amp,
               sound_speed, 1.0 / sound_speed))
