"""Export every reconstructed field of a snapshot as per-cell tipsy renders.

Driven by ``glasstable --fields``.  Hydrodynamic fields (velocity, internal
energy, metals, ...) are captured on one raw AMR grid with the same exact
Petkova machinery the density table uses; the magnetic field, which cannot go
through the generic vertex transfer, is captured separately through the
open-AMR *face* pipeline and written twice -- the RAW field the interpolation
lands on the grid (``_Bprehodge``) and the divergence-projected field after the
Hodge solve (``_Bposthodge``).  The gap between them is exactly the magnetic
energy the divergence cleaning removes.

Every output is a gas-only tipsy snapshot with ONE gas particle per leaf cell
(pos = cell centre, mass = rho*V, rho col = density, temp col = AMR level,
eps/h = cell width) so SPLASH renders them exactly like the density table.
"""

import os

import numpy as np


TINY = np.finfo(np.float64).tiny
EPS_HFAC = 1.0            # SPLASH h = EPS_HFAC * cell width (matches table_to_tipsy)


def _cell_gas(centres, sizes, levels, density, vel=None):
    """Build the shared one-particle-per-cell gas array (cf. table_to_tipsy)."""
    n = len(density)
    g = np.zeros((n, 12), dtype=np.float32)
    g[:, 0] = density * sizes.prod(1)          # mass = rho * V
    g[:, 1:4] = centres
    if vel is not None:
        g[:, 4:7] = vel                        # tipsy velocity columns
    g[:, 7] = density
    g[:, 8] = levels                           # temp column = AMR level
    g[:, 9] = EPS_HFAC * sizes.max(1)          # eps/h = cell width
    g[:, 11] = 0.5 * sizes.max(1)              # phi = half cell width
    return g


def _write_snapshot(tip, g, outsnap):
    n = len(g)
    tip.writetipsy(g, np.zeros((0, 9), np.float32),
                   np.zeros((0, 11), np.float32), outsnap,
                   np.array([n, 3, n, 0, 0, 0], dtype=np.int32))


def _read_bfield(tip, infile):
    """Return the (N,3) BField carrier if the snapshot has BField aux, else None."""
    if not os.path.exists(infile + '.BFieldx'):
        return None
    comps = []
    for axis in 'xyz':
        values, _ = tip.readtipsyaux(infile, 'BField' + axis)
        comps.append(np.asarray(values, dtype=np.float64))
    return np.column_stack(comps)


def export_fields(reconstruct, infile, out, box, gas, *, nsmooth, count,
                  capture_kwargs):
    """Capture and dump all fields of ``gas`` as per-cell tipsy renders.

    reconstruct     : the glassgen.reconstruct module (avoids a circular import)
    infile          : snapshot path (for its aux sidecars, e.g. .BFieldx)
    out             : output prefix; files land at ``${out}_fields*``.00000
    box             : (3,) box or None for an open domain
    gas             : (N,12) tipsy gas array already read by the caller
    capture_kwargs  : method/base_npx/max_level/part_per_cell/contour_* / grade
    """
    tip = reconstruct._import_readtipsy()
    import sph_interp as si

    pos = gas[:, 1:4].astype(np.float64)
    mass = gas[:, 0].astype(np.float64)
    written = []

    # ---- collect every field carried by the snapshot ----------------------
    # Hydro scalars/vectors flatten into carrier channels; magnetic (if any)
    # rides the SAME carriers so ONE grid serves density + hydro + face capture.
    carrier_values = {}
    hydro = {}                              # display name -> component keys
    vel = gas[:, 4:7].astype(np.float64)
    if np.any(vel):
        hydro['vel'] = ['vel_x', 'vel_y', 'vel_z']
        for key, col in zip(hydro['vel'], vel.T):
            carrier_values[key] = col
    u = gas[:, 8].astype(np.float64)            # tipsy gas temperature / u
    if np.any(u):
        hydro['u'] = ['u']
        carrier_values['u'] = u
    metals = gas[:, 10].astype(np.float64)
    if np.any(metals):
        hydro['metals'] = ['metals']
        carrier_values['metals'] = metals
    B = _read_bfield(tip, infile)
    if B is not None:
        for key, col in zip(('Bx', 'By', 'Bz'), B.T):
            carrier_values[key] = col

    # ---- ONE grid build; ONE carrier set ----------------------------------
    rho, h = reconstruct.particle_density(pos, mass, box=box, nsmooth=nsmooth)
    grid, dens_raw = reconstruct.capture_amr_raw(
        pos, mass, box=box, nsmooth=nsmooth, contour_n=count,
        **{k: v for k, v in capture_kwargs.items()
           if k in ('method', 'base_npx', 'max_level', 'part_per_cell',
                    'contour_band', 'contour_beta', 'grade')})
    periodic = (False, False, False) if box is None else (True, True, True)
    carriers = si.Particles(pos, mass, rho, h, carrier_values,
                            None if box is None else box, periodic)
    centres = grid.cell_lo + 0.5 * grid.cell_size
    sizes = grid.cell_size
    levels = grid.cell_level

    # ---- hydro: mass-weighted cell averages via one conservative deposit ---
    hydro_keys = [k for keys in hydro.values() for k in keys]
    cell = {}
    if hydro_keys:
        result = si.interpolate(carriers, grid, method='petkova',
                                values=hydro_keys, conservative_fields=True)
        cell_mass = np.asarray(result.data['_mass'], dtype=np.float64)
        inv_m = 1.0 / np.maximum(cell_mass, TINY)
        for key in hydro_keys:
            cell[key] = np.asarray(result.data[f'_content_{key}'],
                                   dtype=np.float64) * inv_m

    vel_cell = (np.column_stack([cell[k] for k in hydro['vel']])
                if 'vel' in hydro else None)
    g = _cell_gas(centres, sizes, levels, dens_raw, vel=vel_cell)
    outsnap = f'{out}_fields.00000'
    _write_snapshot(tip, g, outsnap)
    aux_written = []
    for name in ('u', 'metals'):
        if name in hydro:
            tip.writetipsyaux(cell[name], name, outsnap)
            aux_written.append(name)
    vel_note = ' vel(cols 4-6)' if vel_cell is not None else ''
    print(f'wrote {outsnap} (hydro: density{vel_note}'
          + (' + aux ' + ','.join(aux_written) if aux_written else '')
          + f'; {grid.ncell} cells)', flush=True)
    written.append(outsnap)

    # ---- magnetic: raw pre-Hodge vs projected post-Hodge (same grid) ------
    if B is not None:
        try:
            _table, audit, result = si.capture_open_amr_face_field(
                carriers, grid, background='particle-mean',
                normal_trace='nearest', boundary_flux='balance',
                allow_clipped_support=True)
        except NotImplementedError:
            print('magnetic export needs an open domain (--open); '
                  'skipping BField', flush=True)
            return written
        raw_B = np.column_stack([np.asarray(result.data[c], dtype=np.float64)
                                 for c in ('Bx', 'By', 'Bz')])
        post_B = np.column_stack([np.asarray(result.data[c], dtype=np.float64)
                                  for c in ('Bx_c', 'By_c', 'Bz_c')])
        for tag, field in (('Bprehodge', raw_B), ('Bposthodge', post_B)):
            gB = _cell_gas(centres, sizes, levels, dens_raw)
            snap = f'{out}_fields_{tag}.00000'
            _write_snapshot(tip, gB, snap)
            tip.writealltipsyaux(field, snap, case='mhd')
            written.append(snap)

        # divergence in both the absolute (face-flux) and the dimensionless
        # h*|divB|/|B| forms (the cleaner's `eps` / sph_divergence_audit
        # convention), with h -> the cell length V^(1/3).
        topology = _table.topology
        bg = np.asarray(audit['boundary_background'], dtype=np.float64)
        raw_face = (topology.initial_values(raw_B - bg)
                    + bg[topology.axis])
        raw_div = topology.cell_divergence(raw_face)     # absolute [B/length]
        post_div = _table.cell_divergence()              # absolute (~machine 0)
        length = np.cbrt(grid.volumes())

        def _stats(div, field):
            bmag = np.linalg.norm(field, axis=1)
            nz = bmag > 0.0
            norm = np.zeros_like(div)
            norm[nz] = np.abs(div[nz]) * length[nz] / bmag[nz]
            return dict(abs_rms=float(np.sqrt(np.mean(div ** 2))),
                        abs_max=float(np.max(np.abs(div))),
                        norm_rms=float(np.sqrt(np.mean(norm ** 2))),
                        norm_max=float(np.max(norm)))

        pre, post = _stats(raw_div, raw_B), _stats(post_div, post_B)
        e_raw = 0.5 * float(np.sum(grid.volumes() * np.sum(raw_B ** 2, axis=1)))
        e_post = 0.5 * float(np.sum(grid.volumes()
                                    * np.sum(post_B ** 2, axis=1)))
        print(f'wrote {out}_fields_Bprehodge.00000 + _Bposthodge.00000 '
              f'(magnetic; {grid.ncell} cells)', flush=True)
        print(f'  cell energy   pre={e_raw:.3e}  post={e_post:.3e} '
              f'[{e_post / max(e_raw, TINY):.2f}x]', flush=True)
        print(f'  |divB| abs    pre rms={pre["abs_rms"]:.2e} '
              f'max={pre["abs_max"]:.2e}  ->  post rms={post["abs_rms"]:.2e} '
              f'max={post["abs_max"]:.2e}', flush=True)
        print(f'  h|divB|/|B|   pre rms={pre["norm_rms"]:.2e} '
              f'max={pre["norm_max"]:.2e}  ->  post rms={post["norm_rms"]:.2e} '
              f'max={post["norm_max"]:.2e}', flush=True)
    else:
        print('no BField aux found - skipping magnetic export', flush=True)

    return written
