import numpy as np
import pytest

from glassgen import reconstruct
from glassgen.reconstruct import AMRTable, capture_density

try:
    reconstruct._import_sph_interp()
    HAVE_SPH_INTERP = True
except ImportError:
    HAVE_SPH_INTERP = False

needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason='sph_interp not found in testsuite layout')


def _lattice(n=16, jitter=0.3, seed=0):
    """Jittered lattice in the unit box: uniform density 1, total mass 1."""
    box = np.ones(3)
    g = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    pos += (rng.random(pos.shape) - 0.5) * (jitter / n)
    mass = np.full(len(pos), 1.0 / len(pos))
    return pos, mass, box


def _gradient(nside=24, seed=0):
    """Lattice warped along x by the inverse CDF of rho(x) = 1 + (x + 0.5)
    on [-0.5, 0.5] (1:2 linear ramp): glass-like local order with a known
    analytic density (a Poisson sample would bury the ramp in ~sqrt(n/64)
    kernel-estimate noise)."""
    g = (np.arange(nside) + 0.5) / nside
    ux, y, z = np.meshgrid(g, g - 0.5, g - 0.5, indexing='ij')
    # CDF(x) = ((x+.5) + (x+.5)^2/2) / 1.5  ->  x = sqrt(1+3u) - 1.5
    x = np.sqrt(1.0 + 3.0 * ux) - 1.5
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    pos += (rng.random(pos.shape) - 0.5) * (0.3 / nside)
    n = nside ** 3
    mass = np.full(n, 1.5 / n)          # integral of rho over the unit box
    rho_true = lambda p: 1.0 + (p[:, 0] + 0.5)         # noqa: E731
    return pos, mass, np.ones(3), rho_true


@needs_sph_interp
def test_amr_table_uniform_density():
    pos, mass, box = _lattice()
    table = capture_density(pos, mass, box=box, grid='amr')
    assert isinstance(table, AMRTable)
    r = table.rho0(pos)
    assert np.all(r > 0)
    assert abs(r.mean() - 1.0) < 0.05
    assert np.abs(r - 1.0).max() < 0.15
    # arbitrary points are covered too (space-filling leaf set)
    q = (np.random.default_rng(1).random((5000, 3)) - 0.5) * box
    rq = table.rho0(q)
    assert np.all(rq > 0)


@needs_sph_interp
def test_amr_table_gradient_and_refinement():
    pos, mass, box, rho_true = _gradient()
    table = capture_density(pos, mass, box=box, grid='amr', part_per_cell=8)
    # AMR actually refined somewhere (leaf count above one base level)
    counts = np.diff(table.offsets)
    assert (counts[1:] > 0).any(), 'no refinement happened'
    # interior accuracy vs the analytic ramp (PWC table + kernel smoothing:
    # allow a few % mean, ~15% pointwise)
    inner = np.abs(pos[:, 0]) < 0.4
    r = table.rho0(pos[inner])
    rel = r / rho_true(pos[inner]) - 1.0
    assert abs(rel.mean()) < 0.03
    assert np.abs(rel).max() < 0.15


@needs_sph_interp
def test_amr_matches_uniform_at_base_level():
    # no refinement (huge per-cell budget): the AMR table is its base grid,
    # so at cell centres it must agree with the uniform-grid table exactly.
    # Pin interp='flat': the centre read IS the cell's own value (the
    # layout equivalence under test); the 'vertex' default (2026-07-13)
    # reads the corner-node interpolation there instead.
    pos, mass, box = _lattice()
    npx = 8
    amr = capture_density(pos, mass, box=box, grid='amr', base_npx=npx,
                          part_per_cell=10 ** 9, interp='flat')
    uni = capture_density(pos, mass, box=box, grid='uniform', npx=npx)
    g = -0.5 + (np.arange(npx) + 0.5) / npx
    cx, cy, cz = np.meshgrid(g, g, g, indexing='ij')
    centres = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
    assert np.allclose(amr.rho0(centres),
                       uni.rho0(centres), rtol=1e-12, atol=0)


@needs_sph_interp
def test_petkova_conserves_mass():
    # the default deposit: exact kernel-volume integral -> cell-averaged
    # density whose volume-weighted sum is the total mass
    pos, mass, box, _ = _gradient(nside=16)
    si = reconstruct._import_sph_interp()
    rho, h = reconstruct.particle_density(pos, mass, box=box)
    parts = si.Particles(pos=pos, mass=mass, rho=rho, h=h,
                         values={'rho': rho}, box=box, periodic=True)
    amr = si.AMRGrid.build(parts, base_npx=8, max_level=4,
                           criterion=si.refine_count(8))
    res = si.interpolate(parts, amr, method='petkova', values=['rho'])
    vals = reconstruct._grid_rho(res, amr.volumes(), 'petkova')
    assert np.isclose((vals * amr.volumes()).sum(), mass.sum(), rtol=1e-6)


@needs_sph_interp
def test_linear_readout_is_smoother_than_flat():
    # same table values, two readouts: trilinear must remove the
    # zeroth-order steps 'flat' has at cell faces. 'linear' is RETIRED
    # from the public surface (2026-07-13, remaps to 'limited') but stays
    # the internal substrate - set the attribute directly to test it.
    pos, mass, box, _ = _gradient(nside=16)
    lin = capture_density(pos, mass, box=box, grid='amr', interp='limited')
    lin.interp = 'linear'
    flat = AMRTable(lin.lo, lin.box, lin.dims, lin.offsets, lin.keys,
                    lin.vals, lin.order, lin.periodic, interp='flat')
    x = np.linspace(-0.49, 0.49, 4001)
    line = np.stack([x, np.full_like(x, 0.123), np.full_like(x, -0.2)], 1)
    jump_lin = np.abs(np.diff(lin.rho0(line))).max()
    jump_flat = np.abs(np.diff(flat.rho0(line))).max()
    assert jump_flat > 5 * jump_lin


@needs_sph_interp
def test_scatter_conserves_mass():
    pos, mass, box, _ = _gradient(nside=16)
    si = reconstruct._import_sph_interp()
    rho, h = reconstruct.particle_density(pos, mass, box=box)
    parts = si.Particles(pos=pos, mass=mass, rho=rho, h=h,
                         values={'rho': rho}, box=box, periodic=True)
    amr = si.AMRGrid.build(parts, base_npx=8, max_level=4,
                           criterion=si.refine_count(8))
    res = si.interpolate(parts, amr, method='sph-scatter', values=['rho'])
    vals = reconstruct._grid_rho(res, amr.volumes(), 'sph-scatter')
    assert np.isclose((vals * amr.volumes()).sum(), mass.sum(), rtol=1e-10)


@needs_sph_interp
def test_empty_cells_edge_fill():
    # gas ball in a mostly-empty box: vacuum cells must come back positive
    # (relax-safe) and carry a NEARBY edge value, not a global-min tail
    pos, mass, box = _lattice(n=20, jitter=0.2)
    inside = np.linalg.norm(pos, axis=1) < 0.25
    pos, mass = pos[inside], mass[inside]
    table = capture_density(pos, mass, box=box, grid='amr')      # 'edge'
    corner = np.array([[0.49, 0.49, 0.49]])
    centre = np.array([[0.0, 0.0, 0.0]])
    r_corner = table.rho0(corner)[0]
    r_centre = table.rho0(centre)[0]
    assert r_corner > 0 and r_centre > 0
    assert r_corner <= r_centre
    # the fill value is a real occupied-cell value
    assert r_corner >= table.vals[table.vals > 0].min() - 1e-300
    # random points everywhere are positive (no relax singularities)
    q = (np.random.default_rng(3).random((20000, 3)) - 0.5) * box
    assert np.all(table.rho0(q) > 0)
    # 'zero' keeps the vacuum at 0 (and is therefore not relax-safe)
    t0 = capture_density(pos, mass, box=box, grid='amr', empty='zero',
                         interp='flat')
    assert t0.rho0(corner)[0] == 0.0


@needs_sph_interp
def test_save_load_roundtrip(tmp_path):
    pos, mass, box, _ = _gradient(nside=12)
    table = capture_density(pos, mass, box=box, grid='amr')
    f = str(tmp_path / 'rho0.npz')
    table.save(f)
    back = AMRTable.load(f)
    q = (np.random.default_rng(2).random((2000, 3)) - 0.5) * box
    assert np.array_equal(table.rho0(q), back.rho0(q))


@needs_sph_interp
def test_relax_accepts_amr_table():
    # the whole point: the table plugs into relax() as the rho0 target
    from glassgen import relax
    pos, mass, box = _lattice(n=8)
    table = capture_density(pos, mass, box=box, grid='amr')
    res = relax(pos, mass, table, box=box, nsmooth=32,
                confine=True, reanneal=False, icr0_stop=0.0, max_iter=3)
    assert res.niter == 3
    assert np.isfinite(res.pos).all()


def _ball(n=6000, R=0.15, seed=0):
    """Dense uniform ball of radius R centered at the ORIGIN (tipsy
    convention) in a unit periodic box, surrounded by vacuum. Total mass 1."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = R * rng.random(n) ** (1.0 / 3.0)
    pos = u * r[:, None]
    mass = np.full(n, 1.0 / n)
    return pos, mass, np.ones(3)


@needs_sph_interp
def test_count_overlay_matches_builder_scheme():
    # apply_count_overlay = the leaf-set view of the builder's shell fill:
    # cells >= rho_edge keep their exact values, sub-edge cells come back
    # positive (decay taper + M6 whole-frame tail, floored at rho_frame) and
    # never above rho_edge; the input is untouched
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    raw = table.vals.copy()
    filled, info = reconstruct.apply_count_overlay(table, 2 * len(pos), c=64)
    assert np.array_equal(table.vals, raw)                # input untouched
    keep = raw >= info['rho_edge']
    assert np.array_equal(filled.vals[keep], raw[keep])   # exact where resolved
    low = filled.vals[~keep]
    assert (low >= info['rho_frame'] * (1 - 1e-12)).all() # floored, positive
    assert info['rho_edge'] > info['rho_frame'] > 0.0
    # more particles -> lower edge/floor (resolution scaling)
    _, info2 = reconstruct.apply_count_overlay(table, 16 * len(pos), c=64)
    assert info2['rho_edge'] <= info['rho_edge']


def test_count_overlay_eta_taper():
    # the eta (decay_spacing) taper must ACTUALLY reshape the fill, not just
    # be accepted: it overrides the capture-cell decay_cells with a length in
    # output-spacing units, so the sub-floor cells differ from the default
    # taper (a bug once left eta/decay_local threaded only into the M6-tail
    # frame, leaving _fill_empty on the old decay_cells path - an exact no-op)
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    raw = table.vals.copy()
    # eta / decay_local reshape the EXP taper (flat_floor=True); the M18
    # default (flat_floor=False) profile is shaped by tail_reach, not eta
    base, info = reconstruct.apply_count_overlay(table, 2 * len(pos), c=64,
                                                 flat_floor=True)
    eta, _ = reconstruct.apply_count_overlay(
        table, 2 * len(pos), c=64, decay_spacing=5.0, flat_floor=True)
    assert np.array_equal(table.vals, raw)                # input untouched
    keep = raw >= info['rho_edge']
    assert np.array_equal(eta.vals[keep], raw[keep])      # exact where resolved
    assert (eta.vals[~keep] >= info['rho_frame'] * (1 - 1e-12)).all()
    # eta reshapes the sub-floor taper away from the decay_cells default
    assert not np.allclose(eta.vals[~keep], base.vals[~keep])


def test_count_overlay_decay_local_reshapes_fill():
    # decay_local (the M16 launch-cell skirt) shortens lambda where a vacuum
    # cell's nearest occupied cell is BRIGHT (rho >> rho_edge), so on a table
    # with real dynamic range it must differ from the global-eta fill - the
    # regression guard for the "reachable but inert" bug
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    glob, info = reconstruct.apply_count_overlay(
        table, 2 * len(pos), c=64, decay_spacing=5.0, decay_local=False,
        flat_floor=True)
    loc, info_l = reconstruct.apply_count_overlay(
        table, 2 * len(pos), c=64, decay_spacing=5.0, decay_local=True,
        flat_floor=True)
    keep = table.vals >= info['rho_edge']
    assert (loc.vals[~keep] >= info_l['rho_frame'] * (1 - 1e-12)).all()
    # local launches from brighter cells -> shorter reach -> different fill
    assert not np.allclose(loc.vals[~keep], glob.vals[~keep])


def test_count_overlay_decay_local_needs_spacing():
    # the M16 skirt is defined against the eta taper; without decay_spacing
    # there is no output-spacing length to launch from, so it must refuse
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    with pytest.raises(ValueError, match='decay_spacing'):
        reconstruct.apply_count_overlay(table, 2 * len(pos), c=64,
                                        decay_local=True)


def test_count_overlay_no_floor_shelf():
    # M18: no-floor-shelf (flat_floor=False, now the DEFAULT) removes the plateau
    # from the overlay's sub-edge fill - values decay toward rho_frame instead
    # of parking at rho_floor. Regression guard that the flag is threaded to
    # _fill_empty (not inert) and the steepness knob reshapes it.
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    raw = table.vals.copy()
    shelf, info = reconstruct.apply_count_overlay(table, 2 * len(pos), c=64,
                                                  flat_floor=True)   # legacy
    noflr, info_n = reconstruct.apply_count_overlay(
        table, 2 * len(pos), c=64)                                   # M18 default
    assert np.array_equal(table.vals, raw)                 # input untouched
    assert info['flat_floor'] is True and info_n['flat_floor'] is False
    keep = raw >= info['rho_edge']
    assert np.array_equal(noflr.vals[keep], raw[keep])     # exact where resolved
    sub = ~keep
    assert sub.any()
    assert (noflr.vals[sub] >= info_n['rho_frame'] * (1 - 1e-12)).all()
    # the flag actually reshapes the sub-floor fill (not a no-op)
    assert not np.allclose(noflr.vals[sub], shelf.vals[sub])
    # steepness knob reshapes it further
    steep, info_s = reconstruct.apply_count_overlay(
        table, 2 * len(pos), c=64, flat_floor=False, tail_reach=12.0)
    assert info_s['tail_reach'] == 12.0
    assert not np.allclose(steep.vals[sub], noflr.vals[sub])



@needs_sph_interp
def test_padded_table_density_fused_matches_generic():
    # the fused one-kernel padded readout (PaddedTableDensity, the relax
    # hot path) must equal the generic PaddedDensity composition of the same
    # base/occupancy/shell pieces - inside the gas, in the taper, in the
    # deep in-domain vacuum AND beyond the table domain (M6 tail region)
    from glassgen import envelope as E
    pos, mass, _ = _ball()
    amr, vals_raw = reconstruct.capture_amr_raw(pos, mass, box=None)
    table = AMRTable.from_grid(amr, vals_raw)
    env = E.build_table_envelope(table, float(mass.sum()), 4000, box=None)
    padded = env['density']
    assert isinstance(padded, E.PaddedTableDensity)
    generic = E.PaddedDensity(padded.base, padded.occupancy, padded.shell,
                              cap=padded.cap)
    rng = np.random.default_rng(5)
    q = rng.uniform(-1.0, 1.0, (30000, 3)) * 0.5 * env['frame_box']
    np.testing.assert_allclose(padded.rho0(q), generic.rho0(q),
                               rtol=1e-10, atol=0.0)
    # and the fused kernel-matched stencil (rho0_stencil, the relax pP hot
    # path) matches the generic batched stencil average
    from glassgen.density import KernelMatchedDensity
    h = rng.uniform(0.01, 0.2, len(q))
    a = KernelMatchedDensity(padded).rho0_h(q, h)
    b = KernelMatchedDensity(generic).rho0_h(q, h)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=0.0)
    # the 'vertex' interp runs fused too (corner tables threaded into the
    # padded kernels, 2026-07-13): fused == generic on a vertex-interp build
    tv = AMRTable.from_grid(amr, vals_raw, interp='vertex')
    envv = E.build_table_envelope(tv, float(mass.sum()), 4000, box=None)
    pv = envv['density']
    assert isinstance(pv, E.PaddedTableDensity)
    gv = E.PaddedDensity(pv.base, pv.occupancy, pv.shell, cap=pv.cap)
    np.testing.assert_allclose(pv.rho0(q), gv.rho0(q), rtol=1e-10, atol=0.0)
    av = KernelMatchedDensity(pv).rho0_h(q, h)
    bv = KernelMatchedDensity(gv).rho0_h(q, h)
    np.testing.assert_allclose(av, bv, rtol=1e-10, atol=0.0)


@needs_sph_interp
def test_padded_cubic_fused_matches_generic():
    # the cubic scaffold (2026-07-13 redesign) through the fused readout:
    # PaddedTableDensity with cubic_scale must equal the generic composition
    # (PaddedDensity over the _CubicTail-wrapped shell) everywhere in the
    # frame, on the linear AND vertex interps, point and stencil reads
    from glassgen import envelope as E
    pos, mass, _ = _ball()
    amr, vals_raw = reconstruct.capture_amr_raw(pos, mass, box=None)
    rng = np.random.default_rng(5)
    for interp in ('limited', 'vertex'):
        table = AMRTable.from_grid(amr, vals_raw, interp=interp)
        env = E.build_table_envelope(table, float(mass.sum()), 4000,
                                     box=None, scaffold='cubic')
        padded = env['density']
        assert isinstance(padded, E.PaddedTableDensity)
        assert padded.cap is None                    # local start, uncapped
        generic = E.PaddedDensity(padded.base, padded.occupancy,
                                  padded.shell, cap=None)
        q = rng.uniform(-1.0, 1.0, (30000, 3)) * 0.5 * env['frame_box']
        np.testing.assert_allclose(padded.rho0(q), generic.rho0(q),
                                   rtol=1e-10, atol=0.0)
        from glassgen.density import KernelMatchedDensity
        h = rng.uniform(0.01, 0.2, len(q))
        a = KernelMatchedDensity(padded).rho0_h(q, h)
        b = KernelMatchedDensity(generic).rho0_h(q, h)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=0.0)


def test_count_overlay_cubic():
    # the glasstable render of the cubic scaffold: occupied cells exact,
    # every scaffold cell >= its raw value (max(raw, taper)) and positive
    pos, mass, box = _ball()
    table = reconstruct.capture_density(pos, mass, box=box, grid='amr',
                                        empty='zero', base_npx=8, max_level=6)
    raw = table.vals.copy()
    filled, info = reconstruct.apply_count_overlay(table, 2 * len(pos), c=64,
                                                   scaffold='cubic')
    assert info['scaffold'] == 'cubic'
    assert np.array_equal(table.vals, raw)                # input untouched
    occ = raw >= info['rho_edge']
    assert np.array_equal(filled.vals[occ], raw[occ])     # exact where gas
    low = filled.vals[~occ]
    assert (low > 0.0).all()
    assert (low >= np.clip(raw[~occ], 0.0, None) - 1e-300).all()


@needs_sph_interp
def test_local_shell_start_continues_bright_surface():
    # shell_start='local' (2026-07-13): next to a surface much brighter than
    # the global rho_edge, the vacuum taper starts at the LOCAL edge value
    # (continuous support scaffold) instead of being capped at rho_edge - and
    # the fused readout still equals the generic composition. Occupancy /
    # rho_edge / floor / frame are unchanged between the two modes.
    from glassgen import envelope as E
    rng = np.random.default_rng(2)
    # a razor-thin bright slab + a faint halo: rho_edge lands in the halo,
    # decades below the slab's local surface density (the galaxy geometry)
    n, nh = 8000, 800
    pos = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                           rng.normal(0, 0.01, n)])
    pos = np.vstack([pos, rng.uniform(-3, 3, (nh, 3))])
    mass = np.full(n + nh, 1.0 / (n + nh))
    amr, vals_raw = reconstruct.capture_amr_raw(pos, mass, box=None,
                                                nsmooth=32, grade=True)
    # 'flat' reads: the local>=capped fill property is a CELL-VALUE
    # statement - minmod slopes over the two different shell value sets
    # can locally invert pointwise reads (surfaced when 'limited' replaced
    # the retired 'linear' as the from_grid default, 2026-07-13)
    table = AMRTable.from_grid(amr, vals_raw, interp='flat')
    kw = dict(box=None, c=32)
    env_e = E.build_table_envelope(table, 1.0, 20000, shell_start='edge',
                                   **kw)
    env_l = E.build_table_envelope(table, 1.0, 20000, shell_start='local',
                                   **kw)
    for key in ('rho_edge', 'rho_frame', 'm'):
        assert env_l[key] == env_e[key]
    re = env_e['rho_edge']
    # probe just outside the slab surface (vacuum side, above the midplane):
    # the local shell must read >> rho_edge, the edge shell <= rho_edge
    q = np.column_stack([np.zeros(9), np.zeros(9),
                         np.linspace(0.05, 0.45, 9)])
    occ = E._eval(env_e['occupancy'], q)
    vac = occ < 0.5
    assert vac.any()
    v_e = env_e['density'].rho0(q)[vac]
    v_l = env_l['density'].rho0(q)[vac]
    assert (v_e <= re * (1 + 1e-12)).all()          # capped arm
    assert v_l.max() > 10.0 * re                    # local support scaffold
    assert (v_l >= v_e * (1 - 1e-12)).all()         # never below the capped
    # fused == generic on the LOCAL arm too (cap=None path)
    padded = env_l['density']
    assert isinstance(padded, E.PaddedTableDensity)
    assert padded.cap is None
    generic = E.PaddedDensity(padded.base, padded.occupancy, padded.shell,
                              cap=None)
    qq = rng.uniform(-1.0, 1.0, (20000, 3)) * 0.5 * env_l['frame_box']
    np.testing.assert_allclose(padded.rho0(qq), generic.rho0(qq),
                               rtol=1e-10, atol=0.0)


@needs_sph_interp
def test_reconstruct_resample_cascade_conserves_mass_and_fields():
    # PURE reconstruct to a DIFFERENT count via the cascade + envelope
    # (reconstruct_resample re-routed through relax_progressive, plan task 7).
    pos, mass, box = _ball()
    fields = {'vel': np.zeros((len(pos), 3)), 'u': np.full(len(pos), 2.0)}
    # contour_refine=False, grade=False: this test exercises the CASCADE
    # path (mass / field conservation, cull accounting), not the capture.
    # Either sharpening (M10 contour refine OR the M11 2:1 grading, both
    # default ON) raises the tiny ball's rho_edge (the blurred pre-M10
    # value came from coarse cells mixing edge gas with vacuum) and the
    # N-aware envelope then CORRECTLY culls the kernel-scale skirt - a 3:1
    # merge at nsmooth=64 leaves ~0.67x-0.72x of the naive count
    # (permanent: max_iter 80->400 recovers nothing). The 0.8x floor below
    # encodes the blurred-edge behaviour; the sharpened captures have their
    # own tests (test_contour_refine_splits_band_straddlers,
    # test_grade_2to1_balances_thin_slab_capture). shell_start='edge' too:
    # the local scaffold on this tiny ball parks+culls ~900 particles at
    # count=2000, putting the survivor count right ON the 0.8x floor
    # (thread-order float noise flips it) - pin the whole pre-M12 envelope.
    for count in (2000, 9000):                    # both < and > the 6000 input
        posf, massf, auxf, isf, a = reconstruct.reconstruct_resample(
            pos, mass, count, fields, box=box, nsmooth=64,
            relax_kw=dict(icr0_stop=2e-2, max_iter=80), seed=1,
            contour_refine=False, grade=False, shell_start='edge',
            flat_floor=True)   # pins the pre-M12 shelf envelope (0.8x floor)
        # exact mass conservation + equal-mass glass
        assert abs(massf.sum() - mass.sum()) < 1e-9
        assert np.allclose(massf, massf[0])
        # fields gathered onto the survivors (u ~ 2 everywhere it sampled gas)
        assert all(len(v) == len(posf) for v in auxf.values())
        assert np.allclose(auxf['u'], 2.0, atol=1e-6)
        assert not isf.any()                      # survivors are gas, bg culled
        # survivors are the compact ball, count near the target
        d = np.linalg.norm(posf, axis=1)
        assert np.percentile(d, 99) < 0.30        # ball R = 0.15
        # 0.75x floor: the two-rung (floor=rho_edge) envelope culls the
        # tiny ball's skirt slightly harder than the old rho_floor one,
        # nudging this boundary count from ~0.80x to ~0.79x (2026-07-14)
        assert 0.75 * count < len(posf) < 1.3 * count
        for key in ('n_full', 'n_survivor', 'n_culled', 'n_background',
                    'mf', 'rho_edge'):
            assert key in a


@needs_sph_interp
def test_capture_cubic_cells_on_noncubic_domain():
    # M3: a scalar base_npx on a non-cubic domain must yield ~CUBIC cells
    # (per-axis base counts scaled to the aspect), not cells whose aspect
    # tracks the domain (the 8:1 galaxy capture)
    rng = np.random.default_rng(5)
    pos = rng.random((4000, 3)) * np.array([4.0, 2.0, 0.5])   # 8:1 slab
    mass = np.full(len(pos), 1.0 / len(pos))
    amr, vals = reconstruct.capture_amr_raw(pos, mass, box=None, nsmooth=32,
                                            base_npx=16)
    assert amr.base_npx[0] > amr.base_npx[1] > amr.base_npx[2] >= 1
    pw = amr.base_pw
    assert pw.max() / pw.min() < 1.6              # ~cubic base cells
    # explicit per-axis counts still honoured verbatim
    amr2, _ = reconstruct.capture_amr_raw(pos, mass, box=None, nsmooth=32,
                                          base_npx=(8, 8, 8))
    assert tuple(amr2.base_npx) == (8, 8, 8)


def _two_level_table(child_vals, n0=4, refined=(1, 1, 1)):
    # direct-constructor two-level AMR: an n0^3 base grid with ONE cell
    # refined into 8 children carrying child_vals; every other cell 0
    n0 = int(n0)
    dims = np.array([[n0] * 3, [2 * n0] * 3], dtype=np.int64)
    rx, ry, rz = refined
    keys0, vals0 = [], []
    for ix in range(n0):
        for iy in range(n0):
            for iz in range(n0):
                if (ix, iy, iz) == (rx, ry, rz):
                    continue
                keys0.append((ix * n0 + iy) * n0 + iz)
                vals0.append(0.0)
    keys1, vals1 = [], []
    n1 = 2 * n0
    for c, (dx, dy, dz) in enumerate([(a, b, cc) for a in range(2)
                                      for b in range(2) for cc in range(2)]):
        ix, iy, iz = 2 * rx + dx, 2 * ry + dy, 2 * rz + dz
        keys1.append((ix * n1 + iy) * n1 + iz)
        vals1.append(float(child_vals[c]))
    k0 = np.argsort(keys0)
    k1 = np.argsort(keys1)
    keys = np.concatenate([np.array(keys0)[k0], np.array(keys1)[k1]])
    vals = np.concatenate([np.array(vals0)[k0], np.array(vals1)[k1]])
    offsets = np.array([0, len(keys0), len(keys)], dtype=np.int64)
    order = np.array([0, 1], dtype=np.int64)
    return AMRTable(np.zeros(3), np.ones(3), dims, offsets, keys, vals,
                    order, False, interp='linear')


def test_linear_readout_level_jump_no_bleed():
    # 2026-07-11 galaxy z-sheet fix: a stencil node inside a coarse cell
    # must read the RESTRICTED average of a finer neighbour (mean of the
    # sibling group), never point-sample one hot child - otherwise a hot
    # fine leaf smears across the whole coarse cell width (readouts of
    # 1e2-4.5e3 x rho_edge were measured in TRUE-ZERO cells)
    hot = [800.0, 0, 0, 0, 0, 0, 0, 0]        # restricted average = 100
    t = _two_level_table(hot)
    t.interp = 'linear'      # the retired trilinear IS the machinery under
    # test here (limited/vertex build on its restricted nodes)
    # sample throughout the empty coarse cell face-adjacent to the refined
    # one: with level-consistent nodes every value is <= the restricted 100
    rng = np.random.default_rng(0)
    q = (np.array([2, 1, 1]) + rng.random((4000, 3))) / 4.0
    v = t.rho0(q)
    assert v.max() <= 100.0 * (1 + 1e-12)
    # and the ramp is still there (continuous, nonzero toward the face)
    assert v.max() > 1.0
    # the refined cell's interior still sees its own sharp children: a point
    # well inside the hot child reads near the leaf value, not the average
    hotc = (np.array([2 * 1 + 0, 2, 2]) + 0.5) / 8.0
    assert t.rho0(hotc[None, :])[0] > 300.0


def test_linear_readout_constant_across_levels():
    # restriction of a constant field is the same constant: the readout must
    # be exactly constant everywhere, including across the level jump
    t = _two_level_table([7.5] * 8)
    # make the whole base grid 7.5 too (rebuild with nonzero coarse vals)
    vals = np.where(t.vals == 0.0, 7.5, t.vals)
    t2 = AMRTable(t.lo, t.box, t.dims, t.offsets, t.keys, vals, t.order,
                  False, interp='limited')
    rng = np.random.default_rng(1)
    q = rng.random((5000, 3))
    assert np.allclose(t2.rho0(q), 7.5, rtol=1e-12, atol=0)


def test_occupancy_flat_ignores_level_jump_bleed():
    # 2026-07-11 companion fix: even level-consistent trilinear legitimately
    # ramps inside a truly-EMPTY coarse cell toward a dense neighbour's
    # restricted average - so OCCUPANCY (and the padded-target dispatch)
    # must use the flat cell average, or the ramp is classified as gas
    # (15% of the galaxy survivors parked in true-zero cells and passed
    # the cull as "occupied")
    from glassgen.envelope import build_table_envelope, table_mass
    # one hot child + 7 faint siblings: rho_edge lands on the faint value
    # (8), DECADES below the hot child (800) and below the neighbour-cell
    # ramp - the galaxy configuration (rho_edge 2.5e-7 vs disk ~1e-3)
    t = _two_level_table([800.0, 8, 8, 8, 8, 8, 8, 8])
    M = table_mass(t)
    # shell_start='edge': this test pins the CAPPED composition ("never
    # above rho_edge in vacuum"); the 'local' default serves the local
    # scaffold there instead (by design - still the shell, never the ramp)
    env = build_table_envelope(t, M, 8192, box=None, c=64, k=2.0,
                               shell_start='edge')
    # the (retired) trilinear base readout DOES ramp above rho_edge inside
    # the empty neighbour cell (the hazard is real in this fixture) - flip
    # the attribute ONLY for this standalone read: the envelope structures
    # hold references to t, so it must be restored before their asserts
    # (the shell/_with_values tables are 'limited' via the constructor)
    rng = np.random.default_rng(2)
    q = (np.array([2, 1, 1]) + rng.random((4000, 3))) / 4.0
    t.interp = 'linear'
    assert t.rho0(q).max() > env['rho_edge']
    t.interp = 'limited'
    # ... but the flat occupancy classifies ALL of it as vacuum,
    assert (env['occupancy'].rho0(q) < 0.5).all()
    # and the padded target there is the capped shell, never the base ramp
    assert env['density'].rho0(q).max() <= env['rho_edge'] * (1 + 1e-9)
    # 'local' (default): still the SHELL (a monotone taper from the hot
    # cell's own value), never the trilinear base ramp - the padded read
    # must equal the shell composition exactly
    env_l = build_table_envelope(t, M, 8192, box=None, c=64, k=2.0)
    from glassgen.envelope import PaddedDensity
    gen = PaddedDensity(env_l['density'].base, env_l['occupancy'],
                        env_l['density'].shell, cap=None)
    np.testing.assert_allclose(env_l['density'].rho0(q), gen.rho0(q),
                               rtol=1e-10, atol=0.0)


def test_limited_readout_no_lobes_and_linear_exact():
    # M13 cheap limiter: 'limited' (MUSCL minmod) must (a) reproduce a
    # smooth linear field exactly where 'linear' does, (b) NEVER ramp
    # toward a bright neighbour inside a faint cell (the graded z-line
    # sawtooth lobes), staying at the faint cell's own scale.
    # (a): single-level table with linear values
    n = 8
    lo, box = np.zeros(3), np.ones(3)
    dims = np.array([[n, n, n]], dtype=np.int64)
    idx = np.arange(n ** 3, dtype=np.int64)
    iz = idx % n
    iy = (idx // n) % n
    ix = idx // (n * n)
    cx = (np.stack([ix, iy, iz], 1) + 0.5) / n
    vals = 1.0 + 2.0 * cx[:, 0] + 3.0 * cx[:, 1] - cx[:, 2]
    t = AMRTable(lo, box, dims, np.array([0, n ** 3]), idx, vals,
                 np.array([0]), False, interp='limited')
    rng = np.random.default_rng(0)
    # one full cell in from the boundary: edge cells have one-sided
    # neighbours (minmod zeroes the slope / corner means extrapolate)
    q = 1.5 / n + rng.random((4000, 3)) * (1.0 - 3.0 / n)
    expect = 1.0 + 2.0 * q[:, 0] + 3.0 * q[:, 1] - q[:, 2]
    np.testing.assert_allclose(t.rho0(q), expect, rtol=1e-12)
    # (b): two-level fixture - inside a faint cell next to the hot child,
    # linear ramps decades above the faint average; limited must not
    t2 = _two_level_table([800.0, 8, 8, 8, 8, 8, 8, 8])
    q2 = (np.array([2, 1, 1]) + rng.random((4000, 3))) / 4.0
    t2.interp = 'linear'
    lin = t2.rho0(q2)
    t2.interp = 'limited'
    lim = t2.rho0(q2)
    t2.interp = 'flat'
    flat = t2.rho0(q2)
    assert lin.max() > 10.0 * flat.max()      # the lobe exists in linear
    assert lim.max() <= flat.max() * (1 + 1e-9)   # limited: no lobe
    assert (lim >= 0).all()                   # clamped to neighbour range


def test_vertex_readout_continuous_across_level_jump():
    # M13 route (a): 'vertex' (hanging-node-slaved corner trilinear) must
    # be (a) exact on a linear field in a uniform region, (b) CONTINUOUS
    # across the level jump where 'linear'/'limited' step (the basis
    # switch), (c) fall back cleanly on the two-level fixture.
    rng = np.random.default_rng(1)
    t = _two_level_table([800.0, 8, 8, 8, 8, 8, 8, 8])
    t.interp = 'vertex'
    # (b) continuity: sample pairs straddling every face of the refined
    # region with a tiny epsilon - vertex values must match across the
    # face while linear jumps
    eps = 1e-9
    faces = []
    # the refined cell spans [0.25,0.5]^3: straddle its bounding faces with
    # tangential coords INSIDE that range (coarse side <-> fine side)
    for u in np.linspace(0.26, 0.49, 12):
        for w in np.linspace(0.26, 0.49, 12):
            for f0 in (0.25, 0.5):
                faces.append(((f0 - eps, u, w), (f0 + eps, u, w)))
                faces.append(((u, f0 - eps, w), (u, f0 + eps, w)))
                faces.append(((u, w, f0 - eps), (u, w, f0 + eps)))
    a = np.array([f[0] for f in faces])
    b = np.array([f[1] for f in faces])
    va = t.rho0(a)
    vb = t.rho0(b)
    assert np.allclose(va, vb, rtol=1e-5, atol=1e-4)
    t.interp = 'linear'
    jump = np.abs(t.rho0(a) - t.rho0(b))
    assert jump.max() > 1.0                       # linear DOES step here
    # (a) linear-field exactness on a uniform single-level table
    n = 8
    dims = np.array([[n, n, n]], dtype=np.int64)
    idx = np.arange(n ** 3, dtype=np.int64)
    iz = idx % n
    iy = (idx // n) % n
    ix = idx // (n * n)
    cx = (np.stack([ix, iy, iz], 1) + 0.5) / n
    vals = 1.0 + 2.0 * cx[:, 0] + 3.0 * cx[:, 1] - cx[:, 2]
    tu = AMRTable(np.zeros(3), np.ones(3), dims, np.array([0, n ** 3]),
                  idx, vals, np.array([0]), False, interp='vertex')
    q = 1.5 / n + rng.random((4000, 3)) * (1.0 - 3.0 / n)
    expect = 1.0 + 2.0 * q[:, 0] + 3.0 * q[:, 1] - q[:, 2]
    np.testing.assert_allclose(tu.rho0(q), expect, rtol=1e-12)


@needs_sph_interp
def test_envelope_mass_contract_closure():
    # 2026-07-13 (user): the ensemble must carry the target's WHOLE mass
    # integral - gas + backed shell + the analytic outside-domain tail -
    # so nothing is borrowed from the gas during the relax:
    # M_gas + n_bg*m == M_target to within one particle mass.
    from glassgen import envelope as E
    rng = np.random.default_rng(4)
    n, nh = 6000, 600
    slab = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                            rng.normal(0, 0.02, n)])
    halo = rng.uniform(-4, 4, (nh, 3))
    pos = np.vstack([slab, halo])
    mass = np.full(len(pos), 1.0 / len(pos))
    amr, vals_raw = reconstruct.capture_amr_raw(pos, mass, box=None,
                                                nsmooth=32)
    table = AMRTable.from_grid(amr, vals_raw)
    for ss in ('local', 'edge'):
        env = E.build_table_envelope(table, 1.0, 10000, box=None, c=32,
                                     shell_start=ss)
        carried = env['gas_mass'] + env['bg_mass'].sum()
        assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9), ss
        # the local scaffold needs MORE backing than the capped one
    env_l = E.build_table_envelope(table, 1.0, 10000, box=None, c=32,
                                   shell_start='local')
    env_e = E.build_table_envelope(table, 1.0, 10000, box=None, c=32,
                                   shell_start='edge')
    assert len(env_l['bg_pos']) >= len(env_e['bg_pos'])


def test_rho_edge_particles_mass_integral():
    # the particle-field twin of envelope.rho_edge: the cut sits where the
    # cumulative faint-end mass first reaches c*m_out
    rho = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
    mass = np.full(6, 1.0)
    # c*m_out = 3 -> after sorting, cum mass 3 is reached at the 3rd particle
    assert reconstruct.rho_edge_particles(rho, mass, 1.0, c=3) == 3.0
    # tiny cut -> the faintest particle; huge cut -> the densest
    assert reconstruct.rho_edge_particles(rho, mass, 0.1, c=1) == 1.0
    assert reconstruct.rho_edge_particles(rho, mass, 100.0, c=10) == 200.0


@needs_sph_interp
def test_contour_refine_splits_band_straddlers():
    # M10: with contour_n set, occupied cells whose mean density falls in the
    # rho_edge band must be refined below the local output sampling scale
    # beta*(m_out/rho)**(1/3) - the coarse straddlers that quantize the flat
    # occupancy verdict (the rectangular cull bites) disappear.
    pos, mass, _ = _ball(n=6000)
    rng = np.random.default_rng(3)
    # faint halo around the ball: sparse enough that its cells stay coarse
    # under the pure 8-particles-per-cell criterion
    nh = 400
    u = rng.standard_normal((nh, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    hp = u * (0.2 + 0.3 * rng.random(nh) ** 2)[:, None]
    pos = np.vstack([pos, hp])
    mass = np.concatenate([mass, np.full(nh, mass[0])])
    count = 4000
    m_out = float(mass.sum()) / count
    beta = 0.5

    def band_leaf_widths(contour_n):
        amr, vals = reconstruct.capture_amr_raw(
            pos, mass, box=None, nsmooth=32, contour_n=contour_n,
            contour_beta=beta)
        re_est = reconstruct.rho_edge_particles(
            *_particle_rho(pos, mass), m_out, c=32)
        rho_cell = (amr.cell_count * mass[0]) / amr.volumes()
        band = ((amr.cell_count > 0) & (rho_cell >= re_est / 8.0)
                & (rho_cell <= re_est * 8.0))
        width = amr.cell_size.max(axis=1)
        floor = beta * np.cbrt(m_out / rho_cell)
        return len(vals), band, width, floor

    def _particle_rho(pos, mass):
        rho, _h = reconstruct.particle_density(pos, mass, box=None,
                                               nsmooth=32)
        return rho, mass

    n0, band0, w0, f0 = band_leaf_widths(None)
    n1, band1, w1, f1 = band_leaf_widths(count)
    assert n1 > n0                       # the pass added leaves
    # pre-M10: at least one coarse straddler wider than its sampling floor
    assert (band0 & (w0 > f0)).any()
    # post-M10: every occupied band leaf is at or below the floor
    assert not (band1 & (w1 > f1 * (1 + 1e-9))).any()


def _max_touch_jump(amr):
    """Worst level difference between touching leaves (face/edge/corner)
    where the coarser one is MORE than one level coarser: 0 = 2:1 balanced."""
    from sph_interp.targets import _amr_leaf_keys, _NB26
    levels, dims = _amr_leaf_keys(amr)
    worst = 0
    for L in range(len(levels) - 1, 1, -1):
        k = levels[L]
        if k.size == 0:
            continue
        nx, ny, nz = (int(dims[L, 0]), int(dims[L, 1]), int(dims[L, 2]))
        iz, iy, ix = k % nz, (k // nz) % ny, k // (ny * nz)
        nb = (np.stack([ix, iy, iz], 1)[:, None, :]
              + _NB26[None, :, :]).reshape(-1, 3)
        ok = ((nb[:, 0] >= 0) & (nb[:, 0] < nx) & (nb[:, 1] >= 0)
              & (nb[:, 1] < ny) & (nb[:, 2] >= 0) & (nb[:, 2] < nz))
        nb = nb[ok]
        rem = np.unique((nb[:, 0] * ny + nb[:, 1]) * nz + nb[:, 2])
        for M in range(L - 2, -1, -1):
            lm = levels[M]
            if lm.size == 0 or rem.size == 0:
                continue
            sh = L - M
            izr, iyr, ixr = rem % nz, (rem // nz) % ny, rem // (ny * nz)
            kM = (((ixr >> sh) * dims[M, 1] + (iyr >> sh)) * dims[M, 2]
                  + (izr >> sh))
            idx = np.searchsorted(lm, kM)
            if (((idx < lm.size)
                 & (lm[np.minimum(idx, lm.size - 1)] == kM)).any()):
                worst = max(worst, L - M)
    return worst


@needs_sph_interp
def test_grade_2to1_balances_thin_slab_capture():
    # a razor-thin dense slab in a sparse halo drives max-level cells right
    # up against near-base vacuum cells (the galaxy z-staircase geometry);
    # grade=True must leave no leaf touching a leaf >1 level coarser while
    # keeping the leaf set space-filling and the deposit mass unchanged.
    rng = np.random.default_rng(1)
    n1, n2 = 6000, 300
    slab = np.column_stack([rng.normal(0, 2.0, n1), rng.normal(0, 2.0, n1),
                            rng.normal(0, 0.04, n1)])
    halo = rng.uniform(-8, 8, (n2, 3))
    pos = np.vstack([slab, halo])
    mass = np.full(len(pos), 1.0 / len(pos))
    amr0, v0 = reconstruct.capture_amr_raw(pos, mass, box=None, nsmooth=32,
                                           grade=False)
    amr1, v1 = reconstruct.capture_amr_raw(pos, mass, box=None, nsmooth=32,
                                           grade=True)
    assert _max_touch_jump(amr0) > 1      # the defect exists ungraded
    assert _max_touch_jump(amr1) <= 1     # graded: 2:1 everywhere
    dom = np.prod(amr1.hi - amr1.lo)
    assert abs(amr1.volumes().sum() / dom - 1) < 1e-9   # space-filling
    # grading must not change the deposited mass (both arms lose the same
    # open-box kernel spill outside the domain)
    m0 = float((v0 * amr0.volumes()).sum())
    m1 = float((v1 * amr1.volumes()).sum())
    assert abs(m1 - m0) < 1e-12
