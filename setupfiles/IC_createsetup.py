"""Create the initial-condition tipsy file (+ aux fields + .param) for one test.

CLI (the runtest.sh contract -- both its lattice/rand and glass invocations):

    python IC_createsetup.py <testsim> <nx> <vm> <entry> <output> [extra ...]

    testsim  test name (run with no arguments to list the choices)
    nx       resolution (number of elements along x)
    vm       variable SPH particle masses (0 = equal-mass particles)
    entry    "0" = close-packed (FCC) lattice, "3" = simple cubic lattice,
             "1" = random, "preglass" = the random PRE-IC the glass relaxation
             starts from (the same particles as "1", but its .param is a
             glassgen input -- periodic box + dICdens* target profile -- not a
             runnable simulation param), anything else = path to a relaxed
             glass IC file to rebuild the final IC from
    output   output name: writes datafiles/<output>.00000 (+ aux, .param)
    extra    per-test parameters, matched POSITIONALLY to the selected
             setup's create() signature after (nx, distri, vm, entry);
             omitted ones keep the create() defaults

Cosmological-unit conventions (for the comoving tests):

    Boxsize = 1 = length unit. Lengths and velocities are comoving;
    velocities are peculiar from the Hubble flow. The critical density of
    the Universe in simulation units is 1, so the total mass in the box in
    simulation units is rho/rho_crit = Omega_matter at z = 0. From the
    Friedmann equation, H_0^2 = 8 pi rho_crit / 3; hence H_0 =
    sqrt(8 pi/3) = 2.894405 in simulation units. The velocity unit is
    (Hubble velocity across the box)/2.894405; the time unit is
    2.894405/(Hubble parameter).

    Example: Omega0 = 1, box size = 100 Mpc at z=0, h=0.5 (sigma8 and P(k)
    are not important): mass unit = (rho_crit in Msun/Mpc^3) * (100 Mpc)^3
    = 6.398e16 Msun; total mass in the box = 1 code mass unit; box size =
    100 Mpc/(1+z); velocity unit = 1727.47/(1+z) = (50*100/2.894405)/(1+z).

Note: closed-packed distributions are not optimal when sides are equal, as
this generates small asymmetries due to uneven spacing.
"""

import argparse
import importlib
import inspect
import os
import sys

import numpy as np

import IC_distribute as distribute
import readtipsy as tip
from IC_createparamfile import createparamfile
from IC_writeascii import convertfromcodeunits, calculatecosmoMsol

# Every name maps to IC_setup_<name>.setup_<name>, imported lazily on use.
SETUPS = [
    'disstest',
    'turbbox',
    'orzag',
    'abcflow',
    'ponomarenko',
    'evrard',
    'polytrope',
    'planet',
    'planetcollision',
    'areablob',
    'mhdrotor',
    'mhddivtest',
    'mhdblast',
    'mhdblast_garcia',
    'mhdpinch_garcia',
    'kh_garcia',
    'sedov',
    'shocktube',
    'mhdloop',
    'mhdloopshear',
    'cube',
    'alfven',
    'cosmowave',
    'taylorgreen',
    'mhdslabcond',
    'mri',
    'mti',
    'kh',
    'square',
    'khslab',
    'rt',
    'zeldovich',
    'mhdcollapse',
    'rotatingcube',
    'gresho',
    'mhdisowave',
    'wave',
    'mhdbalsaravortex',
    'currentsheet',
    'noh',
    'isentropicvortex',
    'coldkeplerian',
    'powerlawring',
    'accdisk',
    'blob',
    'starform',
    'soliton',
    'gaussOverdens',
    '1DdensityFront',
    'fdmwave',
]


def list_setups():
    print("Available options:")
    for option in SETUPS:
        print(f"- {option}")


def load_setup(setup_case):
    module = importlib.import_module(f"IC_setup_{setup_case}")
    return getattr(module, f"setup_{setup_case}")()


# The `entry` value runtest.sh passes for the random pre-IC of a glass run. It
# builds the same particles as "1" but flags the .param as a glassgen input, so
# a plain random run stays a runnable IC in its own right.
PREGLASS = "preglass"


def resolve_distribution(entry):
    """Map the CLI `entry` to (distri, name, lattice, glasssetup).

    distri      what create() sees, unchanged: 0 lattice, 1 random, 2 rebuild
                from a relaxed glass file.
    lattice     which lattice distri==0 places (IC_distribute.set_lattice).
    glasssetup  1 marks the pre-relaxation IC, whose .param carries the
                dICdens* target profile and a forced periodic box for glassgen
                instead of the test's own runtime settings.
    """
    if entry == "0":
        return 0, "FCC lattice", 'closed', 0
    if entry == "3":
        return 0, "cubic lattice", 'cubic', 0
    if entry == "1":
        return 1, "rand", None, 0
    if entry == PREGLASS:
        return 1, "pre-glass rand", None, 1
    return 2, "glass", None, 0


def coerce_extra(value, default):
    """Coerce a CLI extra to its create() parameter type.

    String-valued parameters (planet preset names, planetcollision
    target/impactor) pass through; everything else follows the historical
    all-floats convention. A non-numeric value for a None-default parameter
    also passes through as a string.
    """
    if isinstance(default, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def write_glass_density_table(path, glass_table):
    """Write a setup-provided mapped density table for ``glassgen``.

    Setups expose ``(axes, values, mapping)``.  A one-dimensional table may
    provide its axis directly; two-dimensional targets provide a pair of axes
    and mappings such as ``(('R', '|z|'), values, ('r_cyl', 'z'))``.  The
    resulting ``.npz`` is the self-describing ``TableDensity`` layout.
    """
    axes_spec, values, mapping = glass_table
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        axes = (np.asarray(axes_spec, dtype=float),)
        mappings = (mapping,)
    elif values.ndim == 2:
        if not (isinstance(axes_spec, (tuple, list)) and
                len(axes_spec) == 2):
            raise ValueError("a 2-D glass density table needs two axes")
        if not (isinstance(mapping, (tuple, list)) and len(mapping) == 2):
            raise ValueError("a 2-D glass density table needs two mappings")
        axes = tuple(np.asarray(axis, dtype=float) for axis in axes_spec)
        mappings = tuple(mapping)
    else:
        raise ValueError("setup glass density tables must be 1-D or 2-D")
    if values.shape != tuple(len(axis) for axis in axes):
        raise ValueError("glass density table values do not match its axes")

    payload = {"values": values,
               "mapping": np.asarray(mappings)}
    for index, axis in enumerate(axes):
        payload["ax%d" % index] = axis
    np.savez(path, **payload)


def main():
    if len(sys.argv) < 2:
        list_setups()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Create the IC tipsy file (+ aux, .param) for one test case.",
        usage="%(prog)s testsim nx vm entry output [extra ...]",
        epilog=__doc__.split("Cosmological")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("testsim", help="test name (no arguments lists the choices)")
    parser.add_argument("nx", type=float, help="resolution (elements along x)")
    parser.add_argument("vm", type=float, help="variable particle masses (0 = equal mass)")
    parser.add_argument("entry", help='"0" FCC lattice, "3" cubic lattice, '
                        '"1" random, "preglass" the random pre-IC for the '
                        'glass relaxation, else a relaxed glass IC file')
    parser.add_argument("output", help="written as datafiles/<output>.00000")
    # The per-test extras are sliced off BEFORE argparse so values like
    # "-1e-4" (negative scientific notation, which argparse would reject as
    # an unknown option) and arbitrary preset strings survive untouched.
    args = parser.parse_args(sys.argv[1:6])
    extras = sys.argv[6:]

    if args.testsim not in SETUPS:
        list_setups()
        sys.exit(f"Unknown setup_case: {args.testsim}")

    sim = load_setup(args.testsim)
    distri, distribution_name, lattice, glasssetup = resolve_distribution(args.entry)
    distribute.set_lattice(lattice)
    # A standalone random IC and the random PRE-glass IC both place particles
    # with distri==1, so distri cannot tell them apart -- but the pre-glass one
    # is only a seed for the relaxation. Setups that shape the IC after placing
    # it (stretch maps, cloud replication, box doubling) or that export the
    # relaxer's density table branch on this flag, not on distri.
    sim.preglass = (glasssetup == 1)

    standard_args = {
        'nx': args.nx,
        'distri': distri,
        'vm': args.vm,
        'entry': args.entry,
    }
    sig = inspect.signature(sim.create)
    extra_args = {param.name: param.default for param in sig.parameters.values()
                  if param.name not in standard_args}
    for name, value in zip(extra_args, extras):
        extra_args[name] = coerce_extra(value, extra_args[name])
    all_args = {**standard_args, **extra_args}
    print(f"IC_createsetup[{args.testsim}]: {distribution_name} IC, parameters {all_args}")

    ## Generate gas IC
    sim.create(**all_args)

    # A setup may expose an analytic glass-relaxation target as
    # (axes, values, coordinate_mapping). Store the self-describing 1-D/2-D
    # .npz format consumed by glassgen.TableDensity. Cosmowave uses signed x;
    # powerlawring's equilibrium profile uses (r_cyl, |z|).
    glass_table = getattr(sim, 'glass_density_table', None)
    if glass_table is not None:
        table_path = os.path.join("datafiles", args.output + ".density.npz")
        if os.path.exists(table_path):
            print("density table already exists, preserving:", table_path)
        else:
            write_glass_density_table(table_path, glass_table)
            print("created density table:", table_path)
    # Likewise a force table for glassgen --force-table (relax.config GLASS_FORCEMATCH), as the
    # dict of arrays glassgen.forcematch.ForceTable reads (accdisk: (r_cyl, |z|)).
    force_table = getattr(sim, 'glass_force_table', None)
    if force_table is not None:
        force_path = os.path.join("datafiles", args.output + ".force.npz")
        if os.path.exists(force_path):
            print("force table already exists, preserving:", force_path)
        else:
            np.savez(force_path, **force_table)
            print("created force table:", force_path)

    zero = [0.0] * len(sim.x)
    # A setup OWNS its metallicity if it filled the array per-particle itself
    # (mti's buoyancy tracer, blob's cloud tag, starform's -l knob); anything
    # that left it empty gets the metal-free default. This is a property of the
    # setup, not of its name, so a new test that wants metals just fills the
    # array -- no edit here.
    if len(getattr(sim, 'metals', ())) != len(sim.x):
        sim.metals = zero
    sim.pot = zero
    sim.Bpsi = zero
    # COSMO: determine the final mass unit before converting the thermal field.
    # convertfromcodeunits() mutates sim.u, so doing this after the generic
    # conversion would convert the same internal energy twice.
    if sim.cosmo == 1:
        h0 = 1.0
        ascale = sim.a
        sim.dmsolunit = calculatecosmoMsol(h0, sim.dkpcunit)

    ## (data,kpc,msol,ifnottemp,phystocodevel,phystocodeB)
    if args.testsim == 'mhdcollapse':
        sim = convertfromcodeunits(sim, 0.001, 1000, 1, 1, 1)
    elif args.testsim in ('areablob', 'planet', 'planetcollision'):
        sim = convertfromcodeunits(sim, 1, 1, 0, 0, 0)
    else:
        sim = convertfromcodeunits(sim, sim.dkpcunit, sim.dmsolunit, 1, 0, 0)

    npart = len(sim.x)
    tgdata = np.column_stack((sim.mass, sim.x, sim.y, sim.z,
                              sim.vx, sim.vy, sim.vz,
                              sim.rho, sim.u, sim.h, sim.metals, sim.pot))
    B = np.column_stack((sim.Bx, sim.By, sim.Bz))

    # Tipsy header (none of the tests include DM or stars at this point)
    data_header = np.array([npart, 3, npart, 0, 0, 0])

    if sim.cosmo == 1:
        # The Gasoline/ChaNGa reader starts with room for 100 entries and has a
        # legacy off-by-one capacity check (it reallocates only after attempting
        # entry 101).  Exactly 100 log-spaced outputs are safe and resolve the
        # fastest cosmowave mode well; nSteps remains independent of this list.
        a2 = np.geomspace(ascale, getattr(sim, 'a_end', 1.0), num=100)
        z2 = 1 / a2 - 1
        # Keep each redshift list beside its uniquely named IC.  This avoids a
        # later setup silently overwriting another cosmological test's list.
        redfile = os.path.join("datafiles", args.output + ".red")
        if not os.path.exists(redfile):
            np.savetxt(redfile, z2, fmt='%.4f')
        time = np.array([ascale])
    else:
        time = np.array([0.0])

    tddata = []
    tsdata = []

    # A dark_only setup (the FDM tests) places and relaxes its particles as gas,
    # but the final IC is dark matter: same particles, tipsy dark columns
    # (mass, pos, vel, eps, phi), with the gas smoothing estimate as softening.
    if glasssetup == 0 and getattr(sim, 'dark_only', False):
        tddata = np.column_stack((sim.mass, sim.x, sim.y, sim.z,
                                  sim.vx, sim.vy, sim.vz, sim.h, sim.pot))
        tgdata = np.zeros((0, tgdata.shape[1]))
        B = np.zeros((0, 3))
        data_header = np.array([npart, 3, 0, npart, 0, 0])

    # Generate dark-matter / star (e.g. central sink/BH) particles if the setup
    # defines them. Each hook returns (data_array, nparticles); the per-test logic
    # lives in the IC_setup_*.py files (keeps this driver generic).
    if glasssetup == 0 and hasattr(sim, 'create_dark'):
        tddata, ndark = sim.create_dark(tgdata)
        data_header[3] = data_header[3] + ndark
        data_header[0] = data_header[0] + ndark

    if glasssetup == 0 and hasattr(sim, 'create_star'):
        tsdata, nstar = sim.create_star(tgdata)
        data_header[4] = data_header[4] + nstar
        data_header[0] = data_header[0] + nstar

    myoutput = "datafiles/" + args.output + ".00000"
    tip.writetipsy(tgdata, tddata, tsdata, myoutput, data_header, time)
    tip.writealltipsyaux(B, myoutput)

    if args.testsim == 'planetcollision':
        tip.writealltipsyauxB(sim.Xg, myoutput, sim.labels)

    if args.testsim == 'planet':
        for i in range(1, 6):
            tip.writetipsyaux(getattr(sim, f"Mat{i}"), f"Material{i}", myoutput)

    createparamfile(args.output, sim, glasssetup, args.testsim)
    print('time', time)
    print("created file: ", args.output)


if __name__ == "__main__":
    main()
