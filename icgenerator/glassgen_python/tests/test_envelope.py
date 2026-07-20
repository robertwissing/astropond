"""Unit tests for the ported free-surface envelope (glassgen/envelope.py).

Grid-agnostic: the density field / occupancy / shell targets are simple
callables here, so these tests need neither sph_interp nor a captured table.
See RECONSTRUCT_PLAN.md.
"""
import numpy as np

from glassgen import envelope as E


# --------------------------------------------------------------------------- #
#  N-aware edge density: cumulative faint-end mass first exceeding c*m
# --------------------------------------------------------------------------- #

def test_rho_edge_cumulative_tail_cut():
    # equal-volume cells; faint tail 0,0,1,2,3,4,...: cumulative mass from the
    # bottom is 1, 3, 6, ... so c*m=6 is first reached at density 3.
    vals = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7., 8.])
    vols = np.ones_like(vals)
    assert E.rho_edge(vals, vols, m=1.0, c=6.0) == 3.0


def test_rho_edge_scales_with_c_times_m():
    vals = np.array([0., 0., 1., 2., 3., 4., 5., 6., 7., 8.])
    vols = np.ones_like(vals)
    # doubling m doubles the threshold (12): 1+2+3+4+5=15 first clears it at 5
    assert E.rho_edge(vals, vols, m=2.0, c=6.0) == 5.0
    # equivalently, halving c halves the threshold
    assert E.rho_edge(vals, vols, m=2.0, c=3.0) == 3.0


def test_rho_edge_ignores_pure_vacuum_and_is_positive():
    # lots of zero (vacuum) cells must not drag rho_edge to 0
    vals = np.concatenate([np.zeros(100), [2.0, 4.0]])
    vols = np.ones_like(vals)
    assert E.rho_edge(vals, vols, m=1.0, c=1.0) == 2.0


def test_edge_floor_spacing_relations():
    vals = np.array([0., 1., 2., 3., 4.])
    vols = np.ones_like(vals)
    # cumsum 0,1,3,6,10; c*m=3 first reached at density 2
    # rho_floor retired (two-rung ladder, 2026-07-14): returns (rho_edge, delta_bg)
    re, dbg = E.edge_floor_spacing(vals, vols, m=1.0, c=3.0, k=2.0)
    assert re == 2.0
    assert np.isclose(dbg, 2.0 * (1.0 / re) ** (1.0 / 3.0))


# --------------------------------------------------------------------------- #
#  Synthetic gas ball in vacuum (radius 1) for PaddedDensity / placement / cull
# --------------------------------------------------------------------------- #

R_GAS, RHO_EDGE, K = 1.0, 5.0, 2.0
RHO_FLOOR = RHO_EDGE / K ** 3
LAM = 0.3          # decay length of the shell taper


class _Occ:
    """1 inside the gas ball, 0 outside."""
    def rho0(self, p):
        return (np.linalg.norm(p, axis=1) < R_GAS).astype(float)


class _Shell:
    """Edge density inside; decay edge->floor outside; floor in deep vacuum."""
    def rho0(self, p):
        r = np.linalg.norm(p, axis=1)
        out = np.full(len(p), RHO_EDGE)
        d = r - R_GAS
        outside = d > 0
        out[outside] = np.maximum(RHO_FLOOR,
                                  RHO_EDGE * np.exp(-d[outside] / LAM))
        return out


def test_padded_density_gas_shell_floor():
    pad = E.PaddedDensity(base=lambda p: np.full(len(p), 10.0),
                          occupancy=_Occ(), shell=_Shell())
    v = pad.rho0(np.array([[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [3., 0, 0]],
                          float))
    assert v[0] == 10.0 and v[1] == 10.0          # gas -> exact base (not binned)
    assert RHO_FLOOR < v[2] < RHO_EDGE            # shell decays
    assert np.isclose(v[3], RHO_FLOOR)            # deep vacuum -> floor


def _m_for_spacing(d_fine):
    return RHO_EDGE * d_fine ** 3                  # (m/rho_edge)^(1/3) = d_fine


def test_place_background_in_shell_only():
    m = _m_for_spacing(0.15)
    pos, mass = E.place_background(_Occ(), _Shell(), RHO_EDGE, RHO_FLOOR, m,
                                   region_lo=[-2, -2, -2], region_hi=[2, 2, 2],
                                   seed=1)
    r = np.linalg.norm(pos, axis=1)
    assert len(pos) > 0
    assert np.allclose(mass, m)                    # all mass m_bg = m
    assert (r >= R_GAS).all()                      # none inside the gas
    # none in deep vacuum: shell > floor only while d < LAM*3ln(k)
    assert r.max() < R_GAS + LAM * 3 * np.log(K) + 0.3


def test_place_background_graded_toward_gas():
    m = _m_for_spacing(0.12)
    pos, _ = E.place_background(_Occ(), _Shell(), RHO_EDGE, RHO_FLOOR, m,
                                region_lo=[-2, -2, -2], region_hi=[2, 2, 2],
                                seed=2)
    r = np.linalg.norm(pos, axis=1)
    # number density per unit volume, near the gas vs the outer taper
    near = ((r > 1.0) & (r < 1.2)).sum() / (1.2 ** 3 - 1.0 ** 3)
    far = ((r > 1.4) & (r < 1.6)).sum() / (1.6 ** 3 - 1.4 ** 3)
    assert near > far                              # denser next to the gas


def test_cull_vacuum_conserves_mass_and_removes_vacuum():
    pos = np.array([[0, 0, 0], [0.5, 0, 0], [2., 0, 0], [0, 3., 0]], float)
    mass = np.ones(4)
    aux = {'vel': np.arange(12, dtype=float).reshape(4, 3)}
    pf, mf, af, aud = E.cull_vacuum(pos, mass, aux, _Occ(), total_mass=10.0)
    assert len(pf) == 2                            # only the two inside the ball
    assert np.isclose(mf.sum(), 10.0)              # renormalised exactly
    assert af['vel'].shape == (2, 3)               # aux sliced with the cull
    assert aud['n_culled'] == 2 and np.isclose(aud['renorm'], 5.0)


def test_pad_periodic_box_encloses_with_margin():
    rng = np.random.default_rng(0)
    pos = rng.random((100, 3)) * np.array([2., 4., 6.]) + np.array([1., 1., 1.])
    box, centre = E._pad_periodic_box(pos, pad_frac=0.1)
    # centred frame contains the shifted points with a margin on every side
    shifted = pos - centre
    assert (np.abs(shifted) < 0.5 * box).all()
    assert np.allclose(box, (pos.max(0) - pos.min(0)) * 1.2)


# --------------------------------------------------------------------------- #
#  Table-native helpers (sampling + envelope for the glassgen table builder)
# --------------------------------------------------------------------------- #

def _ramp_table(n=8):
    """3D TableDensity with a 1:3 linear ramp along x, unit box."""
    from glassgen.density import TableDensity
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, _, _ = np.meshgrid(ax, ax, ax, indexing='ij')
    vals = 2.0 + 2.0 * x                    # 1 at x=-0.5, 3 at x=+0.5
    return TableDensity((ax, ax, ax), vals)


def test_table_cells_and_mass():
    t = _ramp_table()
    centres, sizes, levels, vals = E.table_cells(t)
    assert centres.shape == (8 ** 3, 3) and sizes.shape == (8 ** 3, 3)
    assert (levels == 0).all()
    # mean density 2 over the unit box -> total mass 2
    assert np.isclose(E.table_mass(t), 2.0, rtol=1e-12)


def test_with_values_replaces_only_values():
    t = _ramp_table()
    t2 = E._with_values(t, np.full(8 ** 3, 7.0))
    q = np.array([[0.1, -0.2, 0.3], [-0.4, 0.4, 0.0]])
    assert np.allclose(t2.rho0(q), 7.0)
    assert not np.allclose(t.rho0(q), 7.0)          # original untouched


def test_sample_table_tracks_density():
    t = _ramp_table()
    pos = E.sample_table(t, 40000, seed=1)
    assert (np.abs(pos) <= 0.5).all()
    # mass in x>0 vs x<0 halves of the ramp is 2.5 : 1.5
    frac = (pos[:, 0] > 0).mean()
    assert abs(frac - 2.5 / 4.0) < 0.01


def test_build_table_envelope_ball():
    # gas ball on a table: occupancy sharp at rho_edge, shell decays to the
    # floor, background only in the shell, mass m each
    from glassgen.density import TableDensity
    n = 24
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vals = np.where(r < 0.25, 10.0, 0.0)
    t = TableDensity((ax, ax, ax), vals)
    gas_mass = E.table_mass(t)
    env = E.build_table_envelope(t, gas_mass, 5000, c=64, k=2.0)
    assert env['rho_edge'] > env['rho_frame'] > 0.0
    assert np.isclose(env['m'], gas_mass / 5000)
    # occupancy: gas at the centre, vacuum at the corner
    assert env['occupancy'].rho0(np.zeros((1, 3)))[0] == 1.0
    assert env['occupancy'].rho0(np.full((1, 3), 0.45))[0] == 0.0
    # background: each of mass m, none deep inside the gas (on a top-hat ball
    # rho_edge == the interior density, so the trilinear edge band ~1-2 cells
    # wide reads below it and may carry sites - allow that band)
    bg = env['bg_pos']
    assert len(bg) > 0
    assert np.allclose(env['bg_mass'], env['m'])
    assert (np.linalg.norm(bg, axis=1) > 0.25 - 2.0 / n).all()
    # padded target: exact table in the gas, positive (floored) in vacuum
    v = env['density'].rho0(np.array([[0., 0., 0.], [0.45, 0.45, 0.45]]))
    assert np.isclose(v[0], 10.0)
    assert 0.0 < v[1] <= env['rho_edge']
    # seam contract (2026-07-10 fix): the VACUUM side of the padded target
    # never exceeds rho_edge - without the cap/clamp, trilinear bleed from
    # the 10-density gas cells makes the shell read O(10) right outside the
    # contour while the base side hands over rho_edge (an O(1) value seam)
    rng = np.random.default_rng(3)
    q = (rng.random((20000, 3)) - 0.5)
    vac = env['occupancy'].rho0(q) < 0.5
    pv = env['density'].rho0(q[vac])
    assert pv.max() <= env['rho_edge'] * (1 + 1e-9)
    # M6: the deep vacuum decays below the old rho_floor, but never below
    # the frame floor rho_frame = rho_edge/k**6
    assert pv.min() >= env['rho_frame'] * (1 - 1e-9)
    # and the shell FILL starts at min(edge cell, rho_edge): filled (formerly
    # sub-floor) cells never carry more than rho_edge (unwrap the M6
    # _TailedShell to reach the cell table)
    shell = env['density'].shell
    _, _, _, shell_vals = E.table_cells(getattr(shell, 'shell', shell))
    filled = vals.reshape(-1) < env['rho_edge']
    assert shell_vals[filled].max() <= env['rho_edge'] * (1 + 1e-9)


def test_vacuum_tail_monotone_home_pointing():
    # M6 (whole-frame vacuum decay): the deep vacuum is never flat - away
    # from the gas the padded target keeps decaying (exp taper -> power-law
    # tail in the cells, analytic _TailedShell continuation beyond the table
    # domain), reaching the frame floor rho_frame only toward the frame edge.
    # A re-anneal-kicked stray anywhere in the frame feels a home-pointing
    # gradient (the M4-rerun movie failure was the flat rho_floor).
    from glassgen.density import TableDensity
    n = 24
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vals = np.where(r < 0.25, 10.0, 0.0)
    t = TableDensity((ax, ax, ax), vals)
    env = E.build_table_envelope(t, E.table_mass(t), 5000, box=None,
                                 c=64, k=2.0)
    assert np.isclose(env['rho_frame'], env['rho_edge'] / 2.0 ** 6)
    # radial ray: taper region, deep in-domain vacuum, outside the domain.
    # Radii re-aimed 2026-07-16 (taper_len retired, CLEANUP_PLAN C1a): the
    # frame is now 1.2 (half-extent 0.6), so probes must sit INSIDE it -
    # beyond the frame the readout clamps flat at rho_frame, but positions
    # wrap there and no particle can exist (benign, culled). 0.30/0.40/0.49
    # land in distinct cells (width 1/24; finer spacing would alias two
    # probes into one cell and read diff == 0); 0.58 is in the analytic
    # _TailedShell region between the domain face (0.5) and the frame edge.
    radii = np.array([0.30, 0.40, 0.49, 0.58])
    v = env['density'].rho0(np.stack([radii, np.zeros(4), np.zeros(4)], 1))
    assert (np.diff(v) < 0).all()                  # strictly decreasing
    assert v.min() >= env['rho_frame'] * (1 - 1e-9)
    # the tail actually dips BELOW the old flat floor (where the pre-M6
    # shell was constant at rho_floor there was no gradient at all)
    assert v[-1] < env['rho_edge'] / 2.0 ** 3
    # mass-contract closure (2026-07-13): the lattice population is topped
    # up so the ensemble carries the target's WHOLE integral (gas + shell +
    # analytic outside-domain tail) - tail-mass representatives may sit
    # beyond the taper now (placement is free, the relax redistributes),
    # but the BULK of the backing still lives in the taper
    assert len(env['bg_pos']) > 0
    carried = env['gas_mass'] + env['bg_mass'].sum()
    assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9)
    r_bg = np.linalg.norm(env['bg_pos'], axis=1)
    assert np.median(r_bg) < 0.5                   # bulk in the taper


def test_fill_empty_no_floor_shelf_strictly_decreasing():
    # M18 (retire the flat rho_floor shelf): with flat_floor=False the 'decay'
    # fill is ONE strictly-decreasing profile edge*(1+d/r_scale)^-p from the
    # nearest-occupied value down to tail_floor at the frame corner - NO
    # rho_floor plateau (the parking lot). Build an explicit 1D ray from a
    # single gas cell at the origin so the distance-from-gas is exact.
    d = np.concatenate([[0.0], np.linspace(0.03, 0.7, 40)])   # gas + ray
    centres = np.zeros((len(d), 3))
    centres[:, 0] = d
    vals = np.zeros(len(d))
    vals[0] = 10.0
    common = dict(floor=1e-3, tail_scale=0.7, tail_floor=1.25e-4)
    noflr = E._fill_empty(vals.copy(), 'decay', centres, None,
                          tail_reach=6.0, **common)
    ray = noflr[1:]                                # the bad cells, ordered by d
    assert (np.diff(ray) < 0).all()               # STRICTLY decreasing, no plateau
    assert ray[0] < 10.0 and ray[0] > common['tail_floor']
    assert np.isclose(ray[-1], common['tail_floor'], rtol=1e-6)   # reaches it @corner
    assert (ray >= common['tail_floor'] * (1 - 1e-9)).all()


def test_fill_empty_no_floor_needs_tail():
    # 'decay' is meaningless without the frame corner (tail_scale) and its
    # floor (tail_floor) - the profile has nowhere to decay to. (The only
    # fill since the taper family retired, CLEANUP_PLAN C2.)
    import pytest
    centres = np.array([[0.0, 0, 0], [0.1, 0, 0]])
    vals = np.array([10.0, 0.0])
    with pytest.raises(ValueError, match='tail_scale'):
        E._fill_empty(vals.copy(), 'decay', centres, None, floor=1e-3)


def test_fill_empty_tail_reach_steepens_midfield():
    # tail_reach is the steepness knob: r_scale = tail_scale/tail_reach, so a
    # LARGER tail_reach => steeper near the gas => LOWER mid-field density.
    # Same endpoints (start below edge, reach tail_floor at the corner).
    d = np.concatenate([[0.0], np.linspace(0.03, 0.7, 40)])
    centres = np.zeros((len(d), 3))
    centres[:, 0] = d
    vals = np.zeros(len(d))
    vals[0] = 10.0
    common = dict(floor=1e-3, tail_scale=0.7, tail_floor=1.25e-4)
    shallow = E._fill_empty(vals.copy(), 'decay', centres, None,
                            tail_reach=3.0, **common)[1:]
    steep = E._fill_empty(vals.copy(), 'decay', centres, None,
                          tail_reach=12.0, **common)[1:]
    assert np.isclose(steep[-1], shallow[-1], rtol=1e-6)   # same corner value
    mid = slice(5, 30)
    assert (steep[mid] < shallow[mid]).all()               # steeper drains more



# --------------------------------------------------------------------------- #
#  M2: per-axis padding + kernel-fits-the-box rule (frame_open_box)
# --------------------------------------------------------------------------- #

def test_kernel_support_scaling():
    # (3*64/4pi)^(1/3) ~ 2.47 at the default nsmooth
    assert abs(E.kernel_support(1.0, 64) - 2.4814) < 1e-3
    assert np.isclose(E.kernel_support(3.0, 64), 3 * E.kernel_support(1.0, 64))


def test_frame_open_box_thin_axis_capped_by_kernel():
    # thin slab: 100 x 100 x 5, background spacing 4 -> H_bg ~ 9.93; the z
    # axis must be grown to fit a kernel DIAMETER (2*H_bg ~ 19.9) while the
    # long axes keep the plain pad_frac margin
    box, centre = E.frame_open_box([-50, -50, -2.5], [50, 50, 2.5],
                                   delta_bg=4.0, nsmooth=64, pad_frac=0.1,
                                   k=2.0)
    h_bg = E.kernel_support(4.0, 64)
    assert np.all(box >= 2 * h_bg - 1e-12)
    assert np.isclose(box[0], 120.0) and np.isclose(box[1], 120.0)
    assert np.isclose(box[2], 2 * h_bg)          # capped, not span-driven
    assert np.allclose(centre, 0.0)


def test_frame_open_box_margins():
    # margins are max(pad_frac*span, H_edge) -- taper_len retired 2026-07-16
    # (CLEANUP_PLAN C1a), so only these two terms compete.
    # small kernel: the pad_frac margin (1.0 each side) wins
    box, _ = E.frame_open_box([0, 0, 0], [10, 10, 10], delta_bg=0.5,
                              nsmooth=64, pad_frac=0.1, k=2.0)
    assert np.allclose(box, 12.0)
    # bigger kernel: H_edge = H_bg/k ~ 2.48 dominates the 1.0 pad margin
    box2, _ = E.frame_open_box([0, 0, 0], [10, 10, 10], delta_bg=2.0,
                               nsmooth=64, pad_frac=0.1, k=2.0)
    assert np.allclose(box2, 10 + 2 * E.kernel_support(2.0, 64) / 2.0)


def test_build_table_envelope_frame_keys_open_only():
    from glassgen.density import TableDensity
    n = 24
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vals = np.where(r < 0.25, 10.0, 0.0)
    t = TableDensity((ax, ax, ax), vals)
    gm = E.table_mass(t)
    env_open = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0)
    fb = env_open['frame_box']
    assert fb is not None and fb.shape == (3,)
    # the frame respects the kernel-fits-the-box cap and contains the domain
    assert np.all(fb >= 2 * E.kernel_support(env_open['delta_bg'], 64) - 1e-12)
    assert np.all(fb >= 1.0)                     # >= the table domain
    env_per = E.build_table_envelope(t, gm, 5000, box=np.ones(3), c=64, k=2.0)
    assert 'frame_box' not in env_per

def _ball_table(n=24):
    from glassgen.density import TableDensity
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return TableDensity((ax, ax, ax), np.where(r < 0.25, 10.0, 0.0))


def test_frame_open_box_cap_delta_grows_cap():
    # cap_delta grows the kernel-fits-the-box cap for sparser-than-delta_bg
    # kernels (kept after the bg_tail retirement 2026-07-16: the sampled
    # backing's faintest fillers can sit at spacing up to k*delta_bg - see
    # CLEANUP_PLAN RECHECK caveat 2)
    kw = dict(delta_bg=4.0, nsmooth=64, pad_frac=0.1, k=2.0)
    box0, _ = E.frame_open_box([-50, -50, -2.5], [50, 50, 2.5], **kw)
    box1, _ = E.frame_open_box([-50, -50, -2.5], [50, 50, 2.5],
                               cap_delta=8.0, **kw)
    assert np.isclose(box0[2], 2 * E.kernel_support(4.0, 64))
    assert np.isclose(box1[2], 2 * E.kernel_support(8.0, 64))  # x2 on z
    assert np.isclose(box1[0], box0[0])          # span-driven axes unchanged


def test_sampled_backing_and_retired_kwargs():
    # the post-cubic pivot: decay_spacing (lambda in output-spacing units,
    # pinned to the legacy flat_floor=True taper it shapes - inert on the
    # M18 no-floor profile) + sampled backing, the closure as the sole
    # sampler (UNCONDITIONAL since the lattice retirement 2026-07-16,
    # CLEANUP_PLAN C1). Closure contract exact; all backing at particle
    # mass m in vacuum.
    t = _ball_table()
    gm = E.table_mass(t)
    env = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0)
    assert len(env['bg_pos']) > 0
    assert np.allclose(env['bg_mass'], env['m'])
    carried = env['gas_mass'] + env['bg_mass'].sum()
    assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9)
    assert (env['occupancy'].rho0(env['bg_pos']) < 0.5).all()
    # exp-shape target: monotone outward, floored at rho_frame (radii
    # re-aimed inside the post-C1a frame, half-extent 0.6 - see
    # test_vacuum_tail_monotone_home_pointing)
    radii = np.array([0.30, 0.40, 0.49, 0.58])
    v = env['density'].rho0(np.stack([radii, np.zeros(4), np.zeros(4)], 1))
    assert (np.diff(v) <= 0).all() and v[0] > v[-1]
    assert v.min() >= env['rho_frame'] * (1 - 1e-9)
    # backing tracks the shell: near-surface annulus denser than deep vacuum
    r_bg = np.linalg.norm(env['bg_pos'], axis=1)
    n_in = ((r_bg > 0.25) & (r_bg < 0.35)).sum()
    n_out = ((r_bg > 0.45) & (r_bg < 0.55)).sum()
    v_in = 0.35 ** 3 - 0.25 ** 3
    v_out = 0.55 ** 3 - 0.45 ** 3
    assert n_in / v_in > 2.0 * n_out / v_out
    # the retired lattice/bg_tail machinery is really gone (C1)
    import pytest
    with pytest.raises(TypeError):
        E.build_table_envelope(t, gm, 5000, box=None, bg_mode='sample')
    with pytest.raises(TypeError):
        E.build_table_envelope(t, gm, 5000, box=None, bg_tail=True)
    # ... and so is the taper family (C2)
    for kw in (dict(flat_floor=True), dict(shell_start='edge'),
               dict(decay_spacing=1.0), dict(decay_cells=4.0),
               dict(decay_local=True)):
        with pytest.raises(TypeError):
            E.build_table_envelope(t, gm, 5000, box=None, **kw)


def test_bg_over_oversubscribes_closure():
    # M21 (2026-07-18): bg_over oversubscribes the mass-contract closure by
    # bg_over*n_final extra fillers (ensemble mass ABOVE M_target), aimed at
    # cancelling the systematic post-cull count undershoot (renorm f > 1).
    # Default 0 = the exact contract, byte-identical behaviour.
    t = _ball_table()
    gm = E.table_mass(t)
    env0 = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0)
    env1 = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0,
                                  bg_over=0.1)
    assert env0['bg_over'] == 0.0
    assert env1['bg_over'] == 0.1
    # exactly bg_over*N extra fillers beyond the exact contract
    assert len(env1['bg_pos']) == len(env0['bg_pos']) + round(0.1 * 5000)
    # the extras are ordinary backing: mass m, vacuum-only, same shell draw
    assert np.allclose(env1['bg_mass'], env1['m'])
    assert (env1['occupancy'].rho0(env1['bg_pos']) < 0.5).all()
    # the target field itself is untouched (bg_over is ensemble-side only)
    probe = np.stack([np.array([0.30, 0.40, 0.49, 0.58]),
                      np.zeros(4), np.zeros(4)], 1)
    assert np.allclose(env1['density'].rho0(probe),
                       env0['density'].rho0(probe))


# --------------------------------------------------------------------------- #
#  M20: the (nl0, n_frame) reparametrization (RECONSTRUCT_PLAN M20)
# --------------------------------------------------------------------------- #

def test_resolve_bg_k_mapping():
    import pytest
    # pointwise ladder: k = (nsmooth/n_frame)^(1/6); n_frame=1 <=> old k=2
    assert np.isclose(E.resolve_bg_k(n_frame=1.0, nsmooth=64), 2.0)
    assert np.isclose(E.resolve_bg_k(n_frame=16.0, nsmooth=64),
                      4.0 ** (1.0 / 6.0))
    # default n_frame = NFRAME_DEFAULT
    assert np.isclose(E.resolve_bg_k(nsmooth=64),
                      (64.0 / E.NFRAME_DEFAULT) ** (1.0 / 6.0))
    # legacy k passes straight through and excludes n_frame
    assert E.resolve_bg_k(k=1.5) == 1.5
    with pytest.raises(ValueError):
        E.resolve_bg_k(k=1.5, n_frame=16.0)
    with pytest.raises(ValueError):
        E.resolve_bg_k(n_frame=0.0)


def test_solve_tail_reach_pins_the_launch():
    # the solved TR must give launch e-fold L(0) = rs/p = nl0 * 2h_e exactly
    ts, sp, k, ns = 90.65, 4.53, 1.25, 64
    for nl0 in (0.7, 1.35, 2.0):
        tr = E.solve_tail_reach(ts, sp, nl0, k, nsmooth=ns)
        rs = ts / tr
        p = 6.0 * np.log(k) / np.log1p(tr)
        assert np.isclose(rs / p, nl0 * E.kernel_support(sp, ns), rtol=1e-6)
    # 2h_e grows (higher N) -> the solved TR SHRINKS (launch stays pinned)
    tr20 = E.solve_tail_reach(ts, 4.53, 1.35, k, nsmooth=ns)
    tr80 = E.solve_tail_reach(ts, 5.99, 1.35, k, nsmooth=ns)
    assert tr80 < tr20
    # the measured anchor cell: k1.25 nl0=1.35 at the 20k galaxy scale
    # solves to TR ~ 12 (the k1.25-tr12 grid cell)
    assert 10.0 < tr20 < 14.0


def test_scaffold_resolvability_guard():
    import warnings as _w

    import pytest
    ts, sp, ns = 90.65, 4.53, 64
    # the M19 valley cell (k1.5, tr6): LAUNCHES in the band on a thin floor
    with pytest.warns(UserWarning, match='LAUNCHES in the N_L resonance'):
        E.check_scaffold_resolvability(ts, sp, 6.0, 1.5, nsmooth=ns)
    # k2-tr3: p = 3.0 exactly - N_L flat, the whole vacuum resonant
    with pytest.warns(UserWarning, match='flat'):
        nl0, p = E.check_scaffold_resolvability(ts, sp, 3.0, 2.0, nsmooth=ns)
    assert np.isclose(p, 3.0)
    # k2-tr12 (huge-sigma grid cell): climbs across the band UNBACKED
    with pytest.warns(UserWarning, match='UNBACKED'):
        E.check_scaffold_resolvability(ts, sp, 12.0, 2.0, nsmooth=ns)
    # k1.25-tr24: band-adjacent launch (~0.81) on the FAT buffer - silent
    # (the measured 0.84+-0.07 cell; backing pins the marginal stretch)
    with _w.catch_warnings():
        _w.simplefilter('error')
        nl0_f, _ = E.check_scaffold_resolvability(ts, sp, 24.0, 1.25,
                                                  nsmooth=ns)
    assert 0.75 <= nl0_f <= 1.15
    # the M20 default anchor (k1.25-tr12 equivalent): silent
    with _w.catch_warnings():
        _w.simplefilter('error')
        nl0, p = E.check_scaffold_resolvability(ts, sp, 12.0, 1.25,
                                                nsmooth=ns)
    assert 1.2 < nl0 < 1.5 and p < 3.0


def test_build_table_envelope_m20_defaults_and_legacy():
    import warnings as _w

    import pytest
    t = _ball_table()
    gm = E.table_mass(t)

    def _build(**kw):                   # guard warnings are advisory here
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            return E.build_table_envelope(t, gm, kw.pop('n', 2000),
                                          box=None, **kw)

    # defaults: k derived from NFRAME_DEFAULT, TR solved, launch recorded
    env = _build()
    assert np.isclose(env['bg_k'], (64.0 / E.NFRAME_DEFAULT) ** (1.0 / 6.0))
    assert env['tail_reach'] > 0 and 'nl0' in env and 'profile_p' in env
    assert np.isclose(env['nl0'], E.NL0_DEFAULT, rtol=1e-5)
    # explicit nl0 moves the solved TR monotonically (steeper launch =
    # larger TR)
    env_st = _build(nl0=0.7)
    assert env_st['tail_reach'] > env['tail_reach']
    assert np.isclose(env_st['nl0'], 0.7, rtol=1e-5)
    # legacy explicit (k, tail_reach) bypasses the solve bit-exactly and
    # warns when nl0 is also passed
    with pytest.warns(UserWarning, match='overrides nl0'):
        env_l = E.build_table_envelope(t, gm, 2000, box=None, k=1.5,
                                       tail_reach=12.0, nl0=3.0)
    env_l2 = E.build_table_envelope(t, gm, 2000, box=None, k=1.5,
                                    tail_reach=12.0)
    assert env_l['tail_reach'] == 12.0 == env_l2['tail_reach']
    q = np.random.default_rng(1).uniform(-0.55, 0.55, (256, 3))
    assert np.array_equal(env_l['density'].rho0(q), env_l2['density'].rho0(q))
    # launch pinning across N: same nl0 at two counts -> same recorded
    # launch (the solver's whole point), different TR
    env_b = _build(n=8000)
    assert np.isclose(env_b['nl0'], env['nl0'], rtol=1e-6)
    assert env_b['tail_reach'] != env['tail_reach']
