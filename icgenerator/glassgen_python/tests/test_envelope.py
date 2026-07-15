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
    # radial ray: taper region, deep in-domain vacuum, outside the domain
    radii = np.array([0.35, 0.46, 0.75, 1.05])
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
    common = dict(sizes=np.full(len(d), 0.05), floor=1e-3,
                  tail_scale=0.7, tail_floor=1.25e-4)
    noflr = E._fill_empty(vals.copy(), 'decay', centres, None,
                          flat_floor=False, tail_reach=6.0, **common)
    shelf = E._fill_empty(vals.copy(), 'decay', centres, None,
                          flat_floor=True, **common)
    ray = noflr[1:]                                # the bad cells, ordered by d
    assert (np.diff(ray) < 0).all()               # STRICTLY decreasing, no plateau
    assert ray[0] < 10.0 and ray[0] > common['tail_floor']
    assert np.isclose(ray[-1], common['tail_floor'], rtol=1e-6)   # reaches it @corner
    assert (ray >= common['tail_floor'] * (1 - 1e-9)).all()
    # de-flatten is a real change: differs from the default shelf fill
    assert not np.allclose(noflr[1:], shelf[1:])


def test_fill_empty_no_floor_needs_tail():
    # flat_floor=False is meaningless without the frame corner (tail_scale) and
    # its floor (tail_floor) - the profile has nowhere to decay to.
    import pytest
    centres = np.array([[0.0, 0, 0], [0.1, 0, 0]])
    vals = np.array([10.0, 0.0])
    with pytest.raises(ValueError, match='tail_scale'):
        E._fill_empty(vals.copy(), 'decay', centres, None,
                      sizes=np.full(2, 0.05), floor=1e-3, flat_floor=False)


def test_fill_empty_tail_reach_steepens_midfield():
    # tail_reach is the steepness knob: r_scale = tail_scale/tail_reach, so a
    # LARGER tail_reach => steeper near the gas => LOWER mid-field density.
    # Same endpoints (start below edge, reach tail_floor at the corner).
    d = np.concatenate([[0.0], np.linspace(0.03, 0.7, 40)])
    centres = np.zeros((len(d), 3))
    centres[:, 0] = d
    vals = np.zeros(len(d))
    vals[0] = 10.0
    common = dict(sizes=np.full(len(d), 0.05), floor=1e-3,
                  tail_scale=0.7, tail_floor=1.25e-4)
    shallow = E._fill_empty(vals.copy(), 'decay', centres, None,
                            flat_floor=False, tail_reach=3.0, **common)[1:]
    steep = E._fill_empty(vals.copy(), 'decay', centres, None,
                          flat_floor=False, tail_reach=12.0, **common)[1:]
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
                                   taper_len=0.0, k=2.0)
    h_bg = E.kernel_support(4.0, 64)
    assert np.all(box >= 2 * h_bg - 1e-12)
    assert np.isclose(box[0], 120.0) and np.isclose(box[1], 120.0)
    assert np.isclose(box[2], 2 * h_bg)          # capped, not span-driven
    assert np.allclose(centre, 0.0)


def test_frame_open_box_margins():
    # cubic case, kernel small: margins are max(pad_frac*span, taper, H_edge)
    box, _ = E.frame_open_box([0, 0, 0], [10, 10, 10], delta_bg=0.5,
                              nsmooth=64, pad_frac=0.1, taper_len=2.0, k=2.0)
    # pad_frac margin = 1.0 < taper 2.0 -> margin 2.0 each side
    assert np.allclose(box, 14.0)
    box2, _ = E.frame_open_box([0, 0, 0], [10, 10, 10], delta_bg=2.0,
                               nsmooth=64, pad_frac=0.1, taper_len=0.0, k=2.0)
    # H_edge = H_bg/k ~ 2.48 dominates the 1.0 pad margin
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

# --------------------------------------------------------------------------- #
#  Backed tail (bg_tail, 2026-07-11): background over the whole frame
# --------------------------------------------------------------------------- #

def _ball_table(n=24):
    from glassgen.density import TableDensity
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return TableDensity((ax, ax, ax), np.where(r < 0.25, 10.0, 0.0))


def test_frame_open_box_cap_delta_grows_cap():
    # a backed tail's sparsest kernels sit at spacing k*delta_bg: the
    # kernel-fits-the-box cap must grow with cap_delta or they wrap
    kw = dict(delta_bg=4.0, nsmooth=64, pad_frac=0.1, taper_len=0.0, k=2.0)
    box0, _ = E.frame_open_box([-50, -50, -2.5], [50, 50, 2.5], **kw)
    box1, _ = E.frame_open_box([-50, -50, -2.5], [50, 50, 2.5],
                               cap_delta=8.0, **kw)
    assert np.isclose(box0[2], 2 * E.kernel_support(4.0, 64))
    assert np.isclose(box1[2], 2 * E.kernel_support(8.0, 64))  # x2 on z
    assert np.isclose(box1[0], box0[0])          # span-driven axes unchanged


def test_backed_tail_populates_whole_frame():
    t = _ball_table()
    gm = E.table_mass(t)
    env0 = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0)
    env1 = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0,
                                  bg_tail=True)
    # default arm untouched: no tail counter, background confined to the taper
    assert 'n_bg_tail' not in env0
    # tail arm: sites beyond the taper population exist and are counted
    assert env1['n_bg_tail'] > 0
    r0 = np.linalg.norm(env0['bg_pos'], axis=1)
    r1 = np.linalg.norm(env1['bg_pos'], axis=1)
    assert r1.max() > r0.max() + 0.1             # reaches the deep vacuum
    assert len(env1['bg_pos']) > len(env0['bg_pos'])
    # every tail site: vacuum, inside the frame, target above rho_frame
    assert (env1['occupancy'].rho0(env1['bg_pos']) < 0.5).all()
    half = 0.5 * env1['frame_box']
    assert (np.abs(env1['bg_pos'] - env1['frame_centre'])
            <= half + 0.05).all()                # jitter can poke slightly out
    v = env1['density'].rho0(env1['bg_pos'])
    assert (v > env1['rho_frame'] * (1 - 1e-9)).all()
    # mass-consistent: all background carries the same per-particle mass m
    assert np.allclose(env1['bg_mass'], env1['m'])
    # thinning tracks the decaying target: the taper band is denser than the
    # deep tail (counts per volume in two radial annuli)
    n_in = ((r1 > 0.28) & (r1 < 0.42)).sum()
    n_out = ((r1 > 0.55) & (r1 < 0.69)).sum()
    v_in = 0.42 ** 3 - 0.28 ** 3
    v_out = 0.69 ** 3 - 0.55 ** 3
    assert n_in / v_in > 2.0 * n_out / v_out
    # periodic boxes reject the experiment (open builds only)
    import pytest
    with pytest.raises(ValueError):
        E.build_table_envelope(t, gm, 5000, box=np.ones(3), c=64, k=2.0,
                               bg_tail=True)


# --------------------------------------------------------------------------- #
#  Cubic resolvability scaffold (2026-07-13 redesign): one marginal profile
#  rho(s) = rho0*(1 + s/(3*alpha*h0))^-3 from the occupied surface to the
#  frame corner, lattice retired for mass-weighted scaffold sampling
# --------------------------------------------------------------------------- #

def test_fill_cubic_profile_and_max_raw():
    # a line of unit cells along +x: cell 0 occupied at rho0=8, the rest
    # vacuum except cell 4 which holds real faint structure ABOVE the taper
    # (taper at s=4 is (0.5 + 4/6)^-3 ~ 0.63 < 0.8 < rho_edge). The fill
    # must follow (rho0^(-1/3) + s/(3*alpha*m^(1/3)))^-3 exactly and max()
    # must keep the real value.
    centres = np.column_stack([np.arange(5.0), np.zeros(5), np.zeros(5)])
    vals = np.array([8.0, 0.0, 0.0, 0.0, 0.8])
    m, alpha, re = 1.0, 2.0, 1.0
    out = E._fill_cubic(vals.copy(), centres, None, m, alpha, re)
    scale = 3.0 * alpha * m ** (1.0 / 3.0)          # = 6
    s = np.array([1.0, 2.0, 3.0, 4.0])
    expect = (8.0 ** (-1.0 / 3.0) + s / scale) ** -3.0
    assert out[0] == 8.0                            # occupied untouched
    np.testing.assert_allclose(out[1:4], expect[:3], rtol=1e-12)
    assert out[4] == 0.8                            # raw beats the taper
    assert expect[3] < 0.8
    # u-linearity: rho^(-1/3) grows linearly with distance, slope 1/scale
    u = out[1:4] ** (-1.0 / 3.0)
    np.testing.assert_allclose(np.diff(u), [1.0 / scale, 1.0 / scale],
                               rtol=1e-12)
    # monotone decreasing along the ray (up to the real-structure bump)
    assert out[1] > out[2] > out[3] > 0.0


def test_fill_cubic_steeper_near_bright_launch():
    # resolvability self-adjustment: from a BRIGHT launch the taper drops a
    # fixed factor over 3*alpha*h0 with h0 = (m/rho0)^(1/3) - so the descent
    # is steeper (in absolute length) than from a faint launch
    m, alpha, re = 1.0, 2.0, 1.0
    for rho0 in (8.0, 1000.0):
        centres = np.column_stack([np.arange(2.0), np.zeros(2), np.zeros(2)])
        vals = np.array([rho0, 0.0])
        out = E._fill_cubic(vals.copy(), centres, None, m, alpha, re)
        h0 = (m / rho0) ** (1.0 / 3.0)
        # value at s=1 in units of the launch: (1 + 1/(3*alpha*h0))^-3
        np.testing.assert_allclose(
            out[1] / rho0, (1.0 + 1.0 / (3.0 * alpha * h0)) ** -3.0,
            rtol=1e-12)
    # and the FAR field is universal: launch-independent (3*alpha)^3*m/s^3
    s = 1e3
    for rho0 in (8.0, 1000.0):
        v = (rho0 ** (-1.0 / 3.0) + s / (3.0 * alpha)) ** -3.0
        np.testing.assert_allclose(v, (3.0 * alpha / s) ** 3.0, rtol=2e-2)


def test_cubic_tail_u_linear_continuation():
    # outside the domain the continuation needs only the boundary value and
    # the excess distance (u-linear): continuous at the face, exact profile
    shell = lambda p: np.full(len(p), 8.0)            # noqa: E731
    scale = 6.0
    tail = E._CubicTail(shell, -0.5 * np.ones(3), 0.5 * np.ones(3), scale)
    d = np.array([0.0, 0.3, 1.0, 5.0])
    q = np.column_stack([0.5 + d, np.zeros(4), np.zeros(4)])
    v = tail.rho0(q)
    expect = (8.0 ** (-1.0 / 3.0) + d / scale) ** -3.0
    np.testing.assert_allclose(v, expect, rtol=1e-12)
    assert v[0] == 8.0                                # continuous at the face


def test_build_table_envelope_cubic_ball():
    # the cubic arm end-to-end on the gas ball: closure-exact mass-weighted
    # background (no lattice), monotone home-pointing padded target, keys
    t = _ball_table()
    gm = E.table_mass(t)
    env = E.build_table_envelope(t, gm, 5000, box=None, c=64,
                                 scaffold='cubic')
    assert env['scaffold'] == 'cubic'
    assert env['alpha'] == E.ALPHA_DEFAULT
    # occupancy unchanged by the scaffold choice
    assert env['occupancy'].rho0(np.zeros((1, 3)))[0] == 1.0
    assert env['occupancy'].rho0(np.full((1, 3), 0.45))[0] == 0.0
    # background: pure mass-weighted sampling at particle mass m, count from
    # the closure - the ensemble carries the target's whole integral
    assert len(env['bg_pos']) > 0
    assert np.allclose(env['bg_mass'], env['m'])
    carried = env['gas_mass'] + env['bg_mass'].sum()
    assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9)
    # all background sites are vacuum cells (sampled from the scaffold)
    assert (env['occupancy'].rho0(env['bg_pos']) < 0.5).all()
    # backing tracks the target: the near-surface annulus is denser than the
    # deep vacuum (the k15/ladder cusp lever, now by construction)
    r_bg = np.linalg.norm(env['bg_pos'], axis=1)
    n_in = ((r_bg > 0.25) & (r_bg < 0.35)).sum()
    n_out = ((r_bg > 0.45) & (r_bg < 0.55)).sum()
    v_in = 0.35 ** 3 - 0.25 ** 3
    v_out = 0.55 ** 3 - 0.45 ** 3
    assert n_in / v_in > 2.0 * n_out / v_out
    # padded target: exact in the gas, strictly decreasing along an outward
    # ray through the taper, in-domain vacuum and beyond the domain (the
    # _CubicTail region), positive everywhere - never flat
    assert np.isclose(env['density'].rho0(np.zeros((1, 3)))[0], 10.0)
    radii = np.array([0.35, 0.46, 0.75, 1.05])
    v = env['density'].rho0(np.stack([radii, np.zeros(4), np.zeros(4)], 1))
    assert (np.diff(v) < 0).all()
    assert v.min() > 0.0
    # rho_frame is the DERIVED frame-corner value of the profile: the far
    # target stays at or above it inside the frame
    assert v[-1] >= env['rho_frame'] * (1 - 1e-9)


def test_cubic_scaffold_validation():
    import pytest
    t = _ball_table()
    gm = E.table_mass(t)
    with pytest.raises(ValueError):                  # unknown scaffold
        E.build_table_envelope(t, gm, 5000, scaffold='quartic')
    with pytest.raises(ValueError):                  # bg_tail is exp-only
        E.build_table_envelope(t, gm, 5000, box=None, scaffold='cubic',
                               bg_tail=True)


def test_build_uniform_envelope_cubic():
    # the glassgen-native (analytic target) path: same closure contract
    from glassgen.density import CallableDensity

    def rho(p):
        r = np.linalg.norm(np.asarray(p, dtype=np.float64), axis=1)
        return np.where(r < 0.25, 10.0, 0.0)

    rng = np.random.default_rng(1)
    pos = rng.uniform(-0.25, 0.25, (3000, 3))
    pos = pos[np.linalg.norm(pos, axis=1) < 0.25]
    gm = 10.0 * 4.0 / 3.0 * np.pi * 0.25 ** 3
    env = E.build_uniform_envelope(CallableDensity(rho), pos, gm, 3000,
                                   c=64, scaffold='cubic')
    assert env['scaffold'] == 'cubic'
    assert len(env['bg_pos']) > 0
    assert np.allclose(env['bg_mass'], env['m'])
    carried = env['gas_mass'] + env['bg_mass'].sum()
    assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9)
    # padded target monotone along an outward ray, bottoming out at the
    # rho_frame floor (this toy's frame is small - the M6 guidance tail
    # reaches the floor within the probe range, like the exp arm)
    radii = np.array([0.30, 0.40, 0.60, 0.90])
    v = env['density'].rho0(np.stack([radii, np.zeros(4), np.zeros(4)], 1))
    assert (np.diff(v) <= 0).all() and v[0] > v[1]
    assert v.min() >= env['rho_frame'] * (1 - 1e-9)


def test_eta_lambda_and_sampled_backing():
    # the post-cubic pivot: decay_spacing (lambda in output-spacing units)
    # + bg_mode='sample' (closure as the sole sampler). Same exp shape;
    # closure contract exact; all backing at particle mass m in vacuum.
    t = _ball_table()
    gm = E.table_mass(t)
    env = E.build_table_envelope(t, gm, 5000, box=None, c=64, k=2.0,
                                 decay_spacing=1.0, bg_mode='sample')
    assert env['bg_mode'] == 'sample'
    assert env['decay_spacing'] == 1.0
    assert len(env['bg_pos']) > 0
    assert np.allclose(env['bg_mass'], env['m'])
    carried = env['gas_mass'] + env['bg_mass'].sum()
    assert abs(carried - env['M_target']) <= env['m'] * (1 + 1e-9)
    assert (env['occupancy'].rho0(env['bg_pos']) < 0.5).all()
    # exp-shape target: monotone outward, floored at rho_frame
    radii = np.array([0.35, 0.46, 0.75, 1.05])
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
    # guards
    import pytest
    with pytest.raises(ValueError):
        E.build_table_envelope(t, gm, 5000, box=None, bg_mode='grid')
    with pytest.raises(ValueError):
        E.build_table_envelope(t, gm, 5000, box=None, bg_mode='sample',
                               bg_tail=True)


def _two_brightness_table(n=32):
    # bright ball (r<0.12, rho=10) + faint annulus (0.3<r<0.45, rho=0.05):
    # the annulus carries the faint-end mass so rho_edge lands at 0.05,
    # ORDERS below the bright surface - the local-vs-global lambda split
    from glassgen.density import TableDensity
    ax = (np.arange(n) + 0.5) / n - 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vals = np.where(r < 0.12, 10.0, np.where((r > 0.3) & (r < 0.45),
                                             0.05, 0.0))
    return TableDensity((ax, ax, ax), vals)


def test_decay_local_shrinks_bright_skirt():
    # M16 sink fix: with lambda evaluated at the LAUNCH cell's own density
    # the skirt above the bright surface collapses (~(rho_edge/rho_local)
    # ^(1/3) shorter), while rho_edge-level launches are unchanged - the
    # global-lambda near-flat shelf above bright surfaces was the mass
    # sink the cusp drained into (table_res_probe vertical panel).
    t = _two_brightness_table()
    gm = E.table_mass(t)
    kw = dict(box=None, c=64, k=2.0, decay_spacing=1.0, bg_mode='sample')
    envg = E.build_table_envelope(t, gm, 2000, **kw)
    envl = E.build_table_envelope(t, gm, 2000, decay_local=True, **kw)
    assert envl['decay_local'] and not envg['decay_local']
    assert envl['rho_edge'] == envg['rho_edge'] < 0.06   # annulus sets it
    # probe the vacuum gap just outside the bright surface: local decays
    # much faster from the same (uncapped local-start) launch
    pts = np.array([[0.18, 0.0, 0.0], [0.22, 0.0, 0.0]])
    vg = envg['density'].rho0(pts)
    vl = envl['density'].rho0(pts)
    assert (vl < 0.2 * vg).all()
    # both stay relax-safe: monotone outward, floored at rho_frame
    radii = np.array([0.16, 0.20, 0.26])
    for env in (envg, envl):
        v = env['density'].rho0(np.stack([radii, np.zeros(3),
                                          np.zeros(3)], 1))
        assert (np.diff(v) <= 1e-12).all()
        assert v.min() >= env['rho_frame'] * (1 - 1e-9)
    # smaller demanded skirt mass -> the closure needs fewer (never more)
    # backing particles
    assert len(envl['bg_pos']) <= len(envg['bg_pos'])
    # guard: decay_local needs the eta taper
    import pytest
    with pytest.raises(ValueError):
        E.build_table_envelope(t, gm, 2000, box=None, decay_local=True)
