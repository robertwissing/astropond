"""Multi-resolution (progressive) glass relaxation.

Relax a COARSE particle set first to capture the large-scale density field
cheaply, then `split` particles up to full resolution and run a FINE, LOCAL
relaxation (move capped at one smoothing length, no cross-box jumps, no
re-anneal loop) that only fixes small-scale structure. Because the coarse glass
already encodes the large-scale field, the fine phase needs neither the big
box-crossing sweep nor the ICRLOOP0 re-anneal ladder.

Schedule is a list of per-level configs (see DEFAULT level builder); a MEDIUM
stage is added simply by inserting another level dict. The coarse stage uses a
FIXED ABSOLUTE nx_coarse: if nx_full <= nx_coarse the coarse stage is skipped
entirely (the driver reduces to a single plain relax of the input).
"""
import time

import numpy as np

from .density import DensityField
from .diagnostics import residual_force
from . import envelope as _envelope
from .envelope import _pad_periodic_box
from .relax import relax
from . import resample


def _level_diag(pos, mass, h, rho, density, box, nsmooth, flavour, rhopow,
                nsmoothmin):
    """End-of-stage error on a level's glass: denserr |1-rho/rho0| and the
    residual descent force |icvel|, each as mean/p95/max. Lets us see how well
    the coarse stage already fit the field before the fine stage refines it."""
    pos = np.asarray(pos, float)
    if box is not None:
        box = np.asarray(box, float)
        pos = pos - box * np.rint(pos / box)
    rho = np.asarray(rho, float)
    rho0 = density.rho0(pos)
    err = np.abs(1.0 - rho / rho0)
    force = residual_force(pos, mass, h, rho, rho0, box, rhopow=rhopow,
                           nsmooth=nsmooth, flavour=flavour,
                           nsmoothmin=nsmoothmin)
    return dict(denserr_mean=float(err.mean()),
                denserr_p95=float(np.percentile(err, 95)),
                denserr_max=float(err.max()),
                force_mean=float(force.mean()),
                force_p95=float(np.percentile(force, 95)),
                force_max=float(force.max()))


class _ShiftedField(DensityField):
    """Evaluate an open-coordinate density/occupancy field at centred-frame
    positions: ``rho0(p) = field(p + shift)``. Used when relax_progressive frames
    an open IC in a padded periodic box (positions shifted by ``-shift``), so the
    envelope fields - built by the caller in OPEN/world coordinates - are read at
    the right place during the (centred-frame) relax and cull. Handles both a
    DensityField and a plain callable via ``envelope._eval``."""

    def __init__(self, field, shift):
        self.field = field
        self.shift = np.asarray(shift, dtype=np.float64)

    def rho0(self, pos):
        return _envelope._eval(self.field, np.asarray(pos) + self.shift)

    def rho0_h(self, pos, h):
        # forward the kernel-matched readout (task M4) into the open frame
        if hasattr(self.field, 'rho0_h'):
            return self.field.rho0_h(np.asarray(pos) + self.shift, h)
        return self.rho0(pos)

    @property
    def smooth_readout(self):
        return bool(getattr(self.field, 'smooth_readout', False))

    __call__ = rho0


def _measure(pos, mass, density, box, nsmooth, relax_kw):
    """Measure (pos_wrapped, h, rho) on a STATIC particle set via one zero-step
    relax iteration (icr0=0 -> particles do not move), so the post-split state
    is measured with exactly the same hmode/flavour conventions the fine
    stage uses (the split itself only produces positions + mass; rho/h are not
    defined until a neighbour pass)."""
    mkw = {k: v for k, v in relax_kw.items()
           if k not in ('icr0', 'max_iter', 'icr0_stop',
                        'confine', 'reanneal', 'callback', 'callback_every')}
    r = relax(pos, mass, density, box=box, nsmooth=nsmooth, max_iter=1,
              icr0=0.0, icr0_stop=0.0, **mkw)
    return r.pos, r.h, r.rho


def _nearest_pow2(x):
    """Nearest power of two to x (>=1)."""
    x = max(1.0, float(x))
    lo = 1 << int(np.floor(np.log2(x)))
    hi = lo << 1
    return lo if (x - lo) <= (hi - x) else hi


def _floor_pow2(x):
    """Largest power of two <= x (>=1)."""
    return 1 << max(0, int(np.floor(np.log2(max(1.0, float(x))))))


def auto_coarse_count(density, box, alpha=2.0, grid=128, eps_frac=2e-4):
    """Minimum coarse particle count that still resolves the target density
    field everywhere, for an EQUAL-MASS glass:

        N_coarse_min = max_x  M * |grad rho(x)|^3 / (alpha^3 * rho(x)^4),
        M = integral rho dV.

    Derivation: equal mass m=M/N -> local spacing Delta=(m/rho)^(1/3); requiring
    Delta <= alpha * (rho/|grad rho|) (resolve the gradient length with ~1/alpha
    particles) and solving for N gives the bound above. The max picks whichever
    point is hardest to resolve (interface flank, sphere edge, ...) with NO
    geometry assumed - it is purely a function of the density table + alpha, and
    is INDEPENDENT of the input particle count (that only enters the merge
    factor F = floor_pow2(N_full / N_coarse_min)).

    Computed on a grid^3 lattice over the box; |grad rho| by central differences
    with step eps_frac*box, DECOUPLED from the grid spacing so a thin interface
    is not under-resolved (converged from grid=96 on the mhdcollapse 360:1 case).
    Uniform target (grad rho == 0) -> 0 (the caller then merges to its floor).
    alpha=2.0 is calibrated on mhdcollapse 360:1 -> N_coarse_min ~34k (the
    nx_coarse~25-32 sweet spot; nx12 < that -> F=1, no merge)."""
    rho0 = density.rho0 if isinstance(density, DensityField) else density
    box = np.asarray(box, dtype=np.float64)
    # UNIFORM physical spacing (anisotropic per-axis counts), not an isotropic
    # grid^3 - on an elongated box (e.g. shocktube x=2.0) an isotropic grid
    # over-samples the thin axes and leaves the long axis too coarse to resolve
    # a thin feature (the shock interface), badly under-estimating |grad rho|.
    # Same total budget (grid^3 points), spacing = (vol/budget)^(1/3).
    budget = grid ** 3
    spacing = (float(np.prod(box)) / budget) ** (1.0 / 3.0)
    ns = np.maximum(2, np.round(box / spacing).astype(np.int64))
    ax = [(np.arange(ns[d]) + 0.5) / ns[d] * box[d] - 0.5 * box[d]
          for d in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    rho = np.asarray(rho0(pts), dtype=np.float64)
    eps = eps_frac * box   # local-derivative step (decoupled from the spacing)
    g2 = np.zeros(len(pts))
    for d in range(3):
        o = np.zeros(3)
        o[d] = eps[d]
        gd = (np.asarray(rho0(pts + o), dtype=np.float64)
              - np.asarray(rho0(pts - o), dtype=np.float64)) / (2 * eps[d])
        g2 += gd * gd
    gmag = np.sqrt(g2)
    M = float(rho.mean()) * float(np.prod(box))
    S = float(np.max(gmag ** 3 / np.maximum(rho, 1e-30) ** 4))
    return M * S / alpha ** 3


def default_levels(nx_full, nx_coarse, coarse_kw=None, fine_kw=None,
                   momentum=0.0, icr0_stop=3e-3, polish_iter=0,
                   cap=None, start_icr0=1.0, cap_mode='h'):
    """Build the default coarse->fine schedule for a target nx_full given a
    fixed nx_coarse. Returns a list of level dicts. When nx_full <= nx_coarse
    the coarse stage is dropped and a single full-resolution level is returned.

    Each level dict: name, split_factor (applied BEFORE the level's relax),
    plus relax() kwargs (confine, reanneal, icr0, icrloop0, max_iter,
    icr0_stop, ...). coarse_kw / fine_kw override the respective defaults.

    momentum : heavy-ball coefficient applied to the POST-SPLIT fine stage only
        (confined, reanneal=False - where it is validated to accelerate the slow
        frms tail, ~2x fewer iters at beta=0.5). The coarse box-wide reanneal=True
        stage is left at beta=0 (its wide icr0 range + re-anneal ladder are an
        untested regime for momentum, and it already moves fast via box
        amplification). 0.0 = off (memoryless steepest descent everywhere).
    icr0_stop : median-smoothed icr0 stop (max step as a fraction of h), the
        finish for the coarse AND fine stages (default 3e-3; it is
        amplification-independent so it applies to the box-amplified coarse
        stage too). The short intermediate heals are limited by max_iter=
        inter_iter instead. 0.0 = off (then max_iter / polish_iter ends the run).
    polish_iter : fixed polish-iteration budget on the POST-SPLIT fine stage
        only (the full-resolution run) - stop after this many polish iters
        regardless of icr0_stop (which still cuts it short if it converges
        first). 0 = off; the coarse/intermediate stages keep relax()'s off
        default.
    cap : per-step move cap on the POST-SPLIT fine stage, in units of h
        (relax's max_step_frac). None = follow confine = clamp each move
        component to 1*h (the default; only the momentum term can then exceed
        1*h, see relax/stepper). 0.0 = no cap. The fine stage keeps confine's
        box=None controller either way, so start_icr0 stays in units of h.
    start_icr0 : starting icr0 for the POST-SPLIT fine stage in units of h
        (default 1.0). Because the fine stage is confined (box=None controller,
        no box amplification) this IS the iter-0 step as a fraction of h; the
        controller then shrinks it (/~2.2 on iter 1) and self-regulates to its
        critical-step setpoint, so this mostly sets the initial transient."""
    # icr0_stop (step < icr0_stop*h) is the finish for EVERY stage - it is
    # amplification-independent, so the box-amplified coarse stage uses it too.
    coarse_defaults = dict(confine=False, reanneal=True, icr0=1.0,
                           icrloop0=0.5, max_iter=4800, icr0_stop=icr0_stop)
    fine_defaults = dict(confine=True, reanneal=False, icr0=start_icr0,
                         icrloop0=0.5, max_iter=4800,
                         momentum=momentum, icr0_stop=icr0_stop,
                         polish_iter=polish_iter, cap_mode=cap_mode)
    if cap is not None:
        fine_defaults['max_step_frac'] = cap
    if coarse_kw:
        coarse_defaults.update(coarse_kw)
    if fine_kw:
        fine_defaults.update(fine_kw)
    if nx_full <= nx_coarse:
        # no coarse stage: one full-res level (split_factor 1), behaving like a
        # plain relax but keeping the move cap / reanneal choice configurable.
        lvl = dict(name='full', split_factor=1,
                   icr0_stop=icr0_stop, polish_iter=polish_iter)
        lvl.update(coarse_defaults)
        return [lvl]
    factor = _nearest_pow2((nx_full / nx_coarse) ** 3)
    coarse = dict(name='coarse', split_factor=1)
    coarse.update(coarse_defaults)
    fine = dict(name='fine', split_factor=factor)
    fine.update(fine_defaults)
    return [coarse, fine]


def cascade_levels(nx_full=None, nx_coarse=None, *, factor=None, inter_iter=10,
                   coarse_kw=None, fine_kw=None,
                   inter_kw=None, momentum=0.0, icr0_stop=3e-3, polish_iter=0,
                   cap=None, start_icr0=1.0, cap_mode='h'):
    """Coarse -> CASCADE -> fine schedule (a multigrid-style prolongation).

    Instead of the single big x(nx_full/nx_coarse)^3 jump that default_levels
    does, relax the coarse set, then repeatedly binary-split (x2) and run a few
    confined icgen iterations, climbing one resolution level at a time up to
    nx_full, then finish with a full confined relax. One x2 split at a time
    keeps each prolongation a SMALL local perturbation (children land ~0.4 dr
    off the parent) that the handful of intermediate iters heals, so the final
    relax starts from a far healthier glass than after one x8/x64 jump (whose
    post-split denserr is ~0.2 and residual force ~50x the converged value).

    inter_iter : icgen iterations run after each intermediate x2 split (the
        final split is followed by a full relax, not this short smooth). The
        intermediates are fixed short heals: icr0_stop is off, so max_iter=
        inter_iter limits them.

    When nx_full <= nx_coarse a single full-resolution level is returned (as in
    default_levels). coarse_kw/fine_kw/inter_kw override the respective defaults.

    momentum : heavy-ball coefficient applied to the FINAL fine relax only. The
        short intermediate heals (inter_iter iters) and the coarse box-wide
        stage stay beta=0 - momentum needs more iterations than the heals run
        to settle, and on a high-force post-split state it overshoots. 0.0=off.
    """
    # icr0_stop (step < icr0_stop*h, amplification-independent) finishes the
    # coarse and fine stages. The intermediates are fixed short heals limited by
    # max_iter=inter_iter (icr0_stop off - they won't converge in inter_iter).
    coarse_defaults = dict(confine=False, reanneal=True, icr0=1.0,
                           icrloop0=0.5, max_iter=4800, icr0_stop=icr0_stop)
    fine_defaults = dict(confine=True, reanneal=False, icr0=start_icr0,
                         icrloop0=0.5, max_iter=4800,
                         momentum=momentum, icr0_stop=icr0_stop,
                         polish_iter=polish_iter, cap_mode=cap_mode)
    if cap is not None:
        fine_defaults['max_step_frac'] = cap
    # NB momentum is NOT applied to the intermediate heals: they run only
    # inter_iter (~10) iterations from a high-force post-split state, too few
    # for the heavy-ball term to settle, so it overshoots (leaving the next
    # split a worse start). Momentum is reserved for the final fine relax,
    # which has enough iterations to benefit. (Steepest descent here.)
    inter_defaults = dict(confine=True, reanneal=False, icr0=1.0,
                          icrloop0=0.5, max_iter=inter_iter,
                          momentum=0.0, icr0_stop=0.0)
    if coarse_kw:
        coarse_defaults.update(coarse_kw)
    if fine_kw:
        fine_defaults.update(fine_kw)
    if inter_kw:
        inter_defaults.update(inter_kw)
    # total resolution gain as k binary (x2) splits - either an explicit
    # power-of-two `factor` (the merge path, count-based) or the nx ratio.
    if factor is not None:
        total = _floor_pow2(factor)
    else:
        total = _nearest_pow2((nx_full / nx_coarse) ** 3)
    if total <= 1:
        # single full-res level (factor==1, or nx_full <= nx_coarse): honour the
        # caller's icr0_stop / polish_iter (a passed icr0_stop=0 must disable the
        # finish, not inherit relax()'s 3e-3 default) - see default_levels.
        lvl = dict(name='full', split_factor=1,
                   icr0_stop=icr0_stop, polish_iter=polish_iter)
        lvl.update(coarse_defaults)
        return [lvl]
    k = int(round(np.log2(total)))
    coarse = dict(name='coarse', split_factor=1)
    coarse.update(coarse_defaults)
    levels = [coarse]
    # k-1 intermediate (x2 split + short smooth) stages ...
    for i in range(1, k):
        lv = dict(name=f'cascade{i}', split_factor=2)
        lv.update(inter_defaults)
        levels.append(lv)
    # ... then the final x2 split brings us to nx_full and gets the full relax
    fine = dict(name='fine', split_factor=2)
    fine.update(fine_defaults)
    levels.append(fine)
    return levels


def relax_progressive(pos, mass, density, box, *, nx_full=None, nx_coarse=64,
                      levels=None, nsmooth=64, coarse_seed='merge',
                      coarse_target='auto', alpha=2.0, inter_iter=10,
                      coarse_floor=13824, max_iter=4800,
                      split_margin=8, split_dist=0.4,
                      momentum=0.0, icr0_stop=3e-3, polish_iter=0,
                      cap=None, start_icr0=1.0, cap_mode='h',
                      envelope=None, open_pad=0.1, cascade=True,
                      filler_field_hook=None, verbose=False, **relax_kw):
    """Progressive coarse->split->fine glass relaxation.

    envelope : optional free-surface padding (RECONSTRUCT_PLAN.md). A dict
        with keys `density` (the PaddedDensity target, used instead of the
        `density` arg), `occupancy` (a field/callable ->~0/1 for the final
        spatial vacuum cull), `bg_pos`/`bg_mass` (background particles seeded
        into the ensemble so the mass contract holds at every cascade stage;
        they split down with the reals and are culled at the end), and
        `gas_mass` (the real gas mass the survivors are renormalised to). If
        `box is None` the whole pipeline is framed in a padded periodic box
        (`_pad_periodic_box`, margin `open_pad`) and the output is shifted back
        to the open coordinates. `None` = no padding (today's behaviour).

    pos, mass : the FULL-resolution pre-IC (N = nx_full**3). In the default
        two-level path these positions are NOT relaxed directly - the coarse
        stage starts from `coarse_seed` (or a uniform-random fallback at
        nx_coarse) and the fine stage is produced by splitting the coarse glass;
        the input is used only for the total mass and as the seed when the
        coarse stage is skipped (nx_full <= nx_coarse).
    density : DensityField or callable rho0(pos). box : (3,) periodic lengths.
    nx_full : target resolution (elements/side). nx_coarse : fixed absolute
        coarse resolution. levels : explicit schedule (overrides the default).
    coarse_seed : (M,3) array OR callable(n_target, box, total_mass) ->
        (pos, mass) supplying the coarse seed; None -> uniform random. The
        string 'merge' instead COARSENS the full-res input down by the cascade's
        total split factor (resample.coarsen) - so splitting it back restores the
        input count EXACTLY (1:1), with the coarse seed derived from the high-res
        pre-IC itself (no separate low-res file).
    relax_kw : forwarded to every relax() call (e.g. flavour, hmode,
        one_sided, diagnostics). Per-level keys (confine, reanneal, icr0,
        icrloop0, max_iter, icr0_stop, momentum) come from the level dict.
    momentum : heavy-ball coefficient injected into the POST-SPLIT levels only
        (via default_levels) when `levels` is not given explicitly; the coarse
        box-wide stage stays beta=0. When you pass `levels` yourself, bake
        momentum into the level dicts via default_levels/cascade_levels(
        momentum=...) instead - this arg is then ignored.

    Returns the final full-resolution RelaxResult."""
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    # --- envelope framing: pad open boxes, seed the background, use the
    #     PaddedDensity target (see RECONSTRUCT_PLAN.md) --------------
    env_shift = None
    env_gas_mass = None
    env_occupancy = None
    if envelope is not None:
        env_gas_mass = float(envelope.get('gas_mass', mass.sum()))
        bg_pos = envelope.get('bg_pos')
        bg_pos = None if bg_pos is None or len(bg_pos) == 0 \
            else np.asarray(bg_pos, dtype=np.float64)
        density = envelope['density']         # padded target (OPEN coords)
        env_occupancy = envelope['occupancy']
        if box is None:                       # open IC -> padded periodic frame
            if envelope.get('frame_box') is not None:
                # M2 rule: the envelope sized the frame itself (per-axis
                # margins + the kernel-fits-the-box cap from Delta_bg)
                box = np.asarray(envelope['frame_box'], dtype=np.float64)
                env_shift = np.asarray(envelope['frame_centre'],
                                       dtype=np.float64)
            else:
                allp = pos if bg_pos is None \
                    else np.concatenate([pos, bg_pos], 0)
                box, env_shift = _pad_periodic_box(allp, pad_frac=open_pad)
            pos = pos - env_shift
            if bg_pos is not None:
                bg_pos = bg_pos - env_shift
            # the envelope fields are in OPEN coords but the relax/cull run in the
            # centred frame - evaluate them shifted so they read the right place
            density = _ShiftedField(density, env_shift)
            env_occupancy = _ShiftedField(env_occupancy, env_shift)
        if bg_pos is not None:                # seed the ensemble (mass-consistent)
            pos = np.concatenate([pos, bg_pos], axis=0)
            mass = np.concatenate(
                [mass, np.asarray(envelope['bg_mass'], dtype=np.float64)], 0)
    box = np.asarray(box, dtype=np.float64)
    total_mass = float(mass.sum())
    if not isinstance(density, DensityField):
        fn = density
        density = type('_Wrap', (DensityField,),
                       {'rho0': lambda self, p: np.asarray(fn(p),
                                                           dtype=np.float64)})()
    if levels is None:
        if isinstance(coarse_seed, str) and coarse_seed == 'merge':
            # DEFAULT: pick the merge/cascade factor from the target density.
            # N_coarse_min = max_x M|grad rho|^3/(alpha^3 rho^4) (resolves the
            # steepest feature; density-table only); F = floor_pow2(N_full /
            # N_coarse_min), capped so the coarse set keeps >= coarse_floor
            # particles. F==1 => no merge => a single full-res relax (e.g. a
            # low-res run, or a near-discontinuous interface). An explicit
            # coarse_target (a particle count) overrides the 'auto' criterion.
            n_full = len(pos)
            if coarse_target == 'auto':
                # coarse count = max(density criterion, floor): the floor
                # (~24^3) is the minimum coarse resolution, so a weak/zero
                # gradient (incl. uniform) coarsens only to the floor, never
                # below; a stronger gradient bumps it up. An input already
                # below the floor (low-res) gives F=1 (no merge).
                target = max(auto_coarse_count(density, box, alpha=alpha),
                             coarse_floor)
            else:
                target = float(coarse_target)   # explicit count bypasses floor
            # cascade=False (e.g. --no-progressive): force a single full-res
            # level - relax the placed gas+background ensemble directly against
            # the fixed padded target (the envelope single-relax fallback).
            factor = 1 if not cascade \
                else max(1, _floor_pow2(n_full / max(target, 1.0)))
            levels = cascade_levels(
                factor=factor, inter_iter=inter_iter, momentum=momentum,
                icr0_stop=icr0_stop, polish_iter=polish_iter,
                cap=cap, start_icr0=start_icr0, cap_mode=cap_mode,
                coarse_kw=dict(max_iter=max_iter),
                fine_kw=dict(max_iter=max_iter))
            if verbose:
                print(f"[progressive] auto merge: coarse target={target:.0f} "
                      f"-> factor={factor} (coarse N~{n_full // factor})",
                      flush=True)
        else:
            levels = default_levels(nx_full, nx_coarse, momentum=momentum,
                                    icr0_stop=icr0_stop, polish_iter=polish_iter,
                                    cap=cap, start_icr0=start_icr0,
                                    cap_mode=cap_mode)

    skip_coarse = (len(levels) == 1 and levels[0].get('split_factor', 1) == 1)
    if skip_coarse:
        # full-res single relax of the input directly
        cur_pos, cur_mass, aux = pos.copy(), mass.copy(), {}
    elif isinstance(coarse_seed, str) and coarse_seed == 'merge':
        # coarsen the FULL-RES input down by the cascade's total split factor,
        # so splitting it back up restores the input count exactly (1:1) - no
        # separate low-res file, coarse seed derived from the high-res pre-IC.
        total_factor = 1
        for lv in levels:
            total_factor *= int(lv.get('split_factor', 1))
        cur_pos, cur_mass, aux = resample.coarsen(
            pos, mass, box=box, factor=total_factor, margin=split_margin)
        # The glass must be EQUAL-mass even for a non-uniform input mass
        # distribution (which coarsen's renormalisation would preserve): the
        # coarse positions, denser where the input was denser, already encode
        # the target density. Reset to a uniform coarse mass; the splits then
        # keep it equal down to full res.
        cur_mass = np.full(len(cur_pos), total_mass / len(cur_pos))
    else:
        n_coarse = nx_coarse ** 3
        if coarse_seed is None:
            rng = np.random.default_rng(0)
            cur_pos = (rng.random((n_coarse, 3)) - 0.5) * box
            cur_mass = np.full(n_coarse, total_mass / n_coarse)
        elif callable(coarse_seed):
            cur_pos, cur_mass = coarse_seed(n_coarse, box, total_mass)
            cur_pos = np.asarray(cur_pos, dtype=np.float64)
            cur_mass = np.asarray(cur_mass, dtype=np.float64)
        else:
            cur_pos = np.asarray(coarse_seed, dtype=np.float64)
            cur_mass = np.full(len(cur_pos), total_mass / len(cur_pos))
        aux = {}

    result = None
    level_stats = []
    diag_flavour = relax_kw.get('flavour', 'gdforce')
    diag_rhopow = relax_kw.get('rhopow', 2.0)
    diag_nsmoothmin = relax_kw.get('nsmoothmin', 32)
    for lvl in levels:
        sf = int(lvl.get('split_factor', 1))
        if sf > 1:
            cur_pos, cur_mass, aux = resample.split(
                cur_pos, cur_mass, aux, box=box, factor=sf, nsmooth=nsmooth,
                margin=split_margin, dist=split_dist)
            # post-split diagnostic: the quality the fine stage INHERITS from
            # the split (before any fine relaxation), measured with the same
            # conventions the fine stage will use. Shows how much the split
            # degrades the coarse glass and the starting point the fine stage
            # refines from.
            ps_pos, ps_h, ps_rho = _measure(cur_pos, cur_mass, density, box,
                                            nsmooth, relax_kw)
            ps = dict(name=f"{lvl.get('name')}.split", n=len(cur_pos),
                      niter=0, converged=True, wall=0.0,
                      stop_reason='post-split')
            ps.update(_level_diag(ps_pos, cur_mass, ps_h, ps_rho, density, box,
                                  nsmooth, diag_flavour, diag_rhopow,
                                  diag_nsmoothmin))
            level_stats.append(ps)
            if verbose:
                print(f"[progressive] post-split (N={len(cur_pos)}): "
                      f"denserr(mean/p95/max)={ps['denserr_mean']:.4f}/"
                      f"{ps['denserr_p95']:.4f}/{ps['denserr_max']:.4f}  "
                      f"force(mean/p95/max)={ps['force_mean']:.2e}/"
                      f"{ps['force_p95']:.2e}/{ps['force_max']:.2e}", flush=True)
        lvl_kw = {k: v for k, v in lvl.items()
                  if k not in ('name', 'split_factor')}
        kw = dict(relax_kw)
        kw.update(lvl_kw)
        if verbose:
            print(f"[progressive] level '{lvl.get('name')}' : "
                  f"N={len(cur_pos)} split_factor={sf} "
                  f"confine={kw.get('confine')} reanneal={kw.get('reanneal')}",
                  flush=True)
        t_lvl = time.perf_counter()
        result = relax(cur_pos, cur_mass, density, box=box, nsmooth=nsmooth,
                       verbose=verbose, **kw)
        wall = time.perf_counter() - t_lvl
        stats = dict(name=lvl.get('name'), n=len(cur_pos), niter=result.niter,
                     converged=result.converged, wall=wall,
                     stop_reason=result.stop_reason)
        # end-of-stage error (denserr + residual force) so we can see how well
        # the coarse stage fit the field before the fine stage refined it
        stats.update(_level_diag(result.pos, cur_mass, result.h, result.rho,
                                 density, box, nsmooth, diag_flavour,
                                 diag_rhopow, diag_nsmoothmin))
        level_stats.append(stats)
        if verbose:
            print(f"[progressive] level '{lvl.get('name')}' done: "
                  f"niter={result.niter} converged={result.converged} "
                  f"wall={wall:.1f}s  denserr(mean/p95/max)="
                  f"{stats['denserr_mean']:.4f}/{stats['denserr_p95']:.4f}/"
                  f"{stats['denserr_max']:.4f}  force(mean/p95/max)="
                  f"{stats['force_mean']:.2e}/{stats['force_p95']:.2e}/"
                  f"{stats['force_max']:.2e}", flush=True)
        cur_pos = result.pos
    result.levels = level_stats
    # mass-consistency sanity (legacy nx-based path only; the merge path is
    # exact by construction - split(coarsen(x,F),F) restores the input count).
    if not skip_coarse and nx_full is not None and not (
            isinstance(coarse_seed, str) and coarse_seed == 'merge'):
        expect = total_mass / (nx_full ** 3)
        got = total_mass / len(cur_pos)
        if not np.isclose(got, expect, rtol=0.5):
            import warnings
            warnings.warn(
                f"progressive: final per-particle mass {got:.3e} differs from "
                f"the nx_full target {expect:.3e} (final N={len(cur_pos)} vs "
                f"nx_full**3={nx_full**3}); check nx_coarse / split factor.")
    # --- envelope: spatial vacuum cull (remove background + strays), renorm,
    #     unframe. Equal-mass glass -> survivors carry env_gas_mass/n each. ----
    if envelope is not None:
        keep = _envelope._eval(env_occupancy, result.pos) >= 0.5
        n_surv = int(keep.sum())
        bgp = envelope.get('bg_pos')
        n_bg = 0 if bgp is None else len(bgp)
        # pre-cull diagnostic stash (caller's coordinates, like the output):
        # the full relaxed ensemble + the cull verdict, for with/without-cull
        # audits - it exists nowhere else once the cull runs.
        precull_pos = (result.pos + env_shift if env_shift is not None
                       else result.pos.copy())
        result.pos = np.ascontiguousarray(result.pos[keep])
        result.rho = np.ascontiguousarray(result.rho[keep])
        result.h = np.ascontiguousarray(result.h[keep])
        if env_shift is not None:                 # back to open coordinates
            result.pos = result.pos + env_shift
        mf = env_gas_mass / max(n_surv, 1)
        result.envelope_audit = dict(
            n_full=int(len(keep)), n_survivor=n_surv,
            n_culled=int((~keep).sum()), n_background=int(n_bg),
            gas_mass=env_gas_mass, mf=mf,
            precull_pos=precull_pos, precull_keep=keep)
        if envelope.get('m'):
            # renorm health gate: f = realised survivor mass / the contract
            # mass m. f-1 measures the NET mass imbalance inside the
            # occupancy contour at stop time (leaked reals vs meshed-in
            # background). On a polished run the equilibrium pins it to a
            # surface term (~one skin layer): f ~ 1.00 +- O(0.1%). A large
            # f means the run was culled in a scattered state (e.g. stopped
            # mid-re-anneal-kick) and the renorm uniformly biases rho/rho0
            # of every survivor - don't trust the output.
            f = mf / float(envelope['m'])
            result.envelope_audit['renorm'] = f
            if abs(f - 1.0) > 0.01:
                import warnings
                warnings.warn(
                    f"envelope renorm f={f:.4f} (survivor mass / contract "
                    f"mass) is >1% from 1: {int((~keep).sum())} of "
                    f"{len(keep)} particles were culled outside the "
                    "resolved gas - the run likely stopped in an unsettled "
                    "state (mid-anneal kick?); the uniform mass renorm "
                    "biases rho/rho0 of every survivor by this factor.")
        # gather per-survivor fields from the source at the FINAL positions (in
        # the caller's coordinate frame - post-unframe here) - reconstruct's
        # velocity/temperature/etc. The reconstructed glass has no history to
        # inherit; the fields are read fresh from the input at each final point.
        if filler_field_hook is not None:
            result.aux = filler_field_hook(result.pos)
        if verbose:
            a = result.envelope_audit
            print(f"[progressive] envelope cull: {a['n_full']} -> "
                  f"{a['n_survivor']} (culled {a['n_culled']}, "
                  f"bg {a['n_background']}), mf={a['mf']:.3e}", flush=True)
    return result


def build_from_table(table, count, box=None, *, gas_mass=None, nsmooth=64,
                     bg_c=None, bg_k=None, envelope='auto',
                     cascade=True, seed=0, open_pad=0.1, kernel_match=True,
                     filler_field_hook=None, alpha=None,
                     km_hmode='particle', km_hscale=1.0,
                     tail_reach=None, n_frame=None, nl0=None, bg_over=0.0,
                     verbose=False, **relax_kw):
    """THE glassgen table builder (RECONSTRUCT_PLAN.md, decided 2026-07-10).

    Given a density table (AMRTable / 3D TableDensity) and a target count,
    sample the initial ``count`` positions from the table itself
    (:func:`envelope.sample_table`), derive the N-aware envelope from
    ``m = gas_mass/count`` (:func:`envelope.build_table_envelope`), run the
    progressive relax against the padded target, and cull/renorm.
    ``./glassgen --table --count`` and ``glassresample --op reconstruct``
    both drive this one function.

    envelope : 'auto' (build it; drop it when the table places no background
        AND the box is periodic - a filled box), 'on', or 'off'. 'off' with
        ``box=None`` is an error: an open relax cannot contain a free surface.
    kernel_match : evaluate the relax target as each particle's own KERNEL
        AVERAGE of the table (:class:`density.KernelMatchedDensity`, plan
        task M4 - default ON, approved 2026-07-10): where the table is
        sharper than the N-particle kernel (thin-disk z profiles) the
        pseudo-pressure ratio -> 1 instead of blowing the feature open.
        Resolved regions are unaffected to second order. False = the point
        read (pre-M4 behaviour).
    alpha : the cascade resolvability tolerance (auto_coarse_count);
        defaults to envelope.ALPHA_DEFAULT. The vacuum fill is the M18
        no-floor profile (taper family retired C2, cubic scaffold deleted
        C3, 2026-07-17); the background is always sampled backing (the
        closure as sole sampler; lattice/bg_tail retired 2026-07-16, C1).
    gas_mass : the mass contract. Defaults to the table integral sum(rho*V);
        pass the input snapshot's exact gas mass when reconstructing.
    relax_kw : forwarded to :func:`relax_progressive` - its named cascade
        knobs (max_iter, icr0_stop, momentum, ...) bind there; the rest reach
        every relax() call (callback, flavour, hmode, ...).

    Returns the final RelaxResult: ``res.envelope_audit`` (always set; mf =
    the survivors' per-particle mass) + ``res.table_build`` (rho_edge/
    rho_floor/delta_bg/m when the envelope ran), ``res.aux`` = the
    ``filler_field_hook`` gather at the final positions (if given)."""
    from .envelope import (ALPHA_DEFAULT, build_table_envelope, sample_table,
                           table_mass)
    if envelope is True:
        envelope = 'on'
    elif envelope is False:
        envelope = 'off'
    if gas_mass is None:
        gas_mass = table_mass(table)
    if alpha is None:
        alpha = ALPHA_DEFAULT
    env = None
    if envelope in ('auto', 'on'):
        env = build_table_envelope(table, gas_mass, count, box=box,
                                   c=(bg_c or nsmooth), k=bg_k, seed=seed,
                                   nsmooth=nsmooth, pad_frac=open_pad,
                                   tail_reach=tail_reach,
                                   n_frame=n_frame, nl0=nl0,
                                   bg_over=bg_over)
        if envelope == 'auto' and len(env['bg_pos']) == 0 and box is not None:
            env = None                       # filled periodic box: no vacuum
        if env is not None and verbose:
            print(f"[progressive] envelope: bg_k {env['bg_k']:.3f} "
                  f"tail_reach {env['tail_reach']:.2f}"
                  + (f" nl0 {env['nl0']:.2f} p {env['profile_p']:.2f}"
                     if 'nl0' in env else ''), flush=True)
    if env is None and box is None:
        raise ValueError("an open-box table build needs the envelope "
                         "(envelope='off' requires a periodic box)")
    target = env['density'] if env is not None else table
    if kernel_match:
        from .density import KernelMatchedDensity
        # km_hmode='target' (2026-07-13): equilibrium-h readout - the
        # averaging scale from the target's own density instead of each
        # particle's state (breaks the drained-region h-growth feedback)
        target = KernelMatchedDensity(target, hmode=km_hmode,
                                      m=float(gas_mass) / int(count),
                                      nsmooth=nsmooth,
                                      rho_edge=(env['rho_edge']
                                                if env is not None else None),
                                      h_scale=km_hscale)
        if env is not None:
            env['density'] = target       # relax_progressive rereads the dict
    pos = sample_table(table, count, seed=seed)
    mass = np.full(int(count), float(gas_mass) / int(count))
    rkw = dict(relax_kw)
    rkw.pop('reanneal', None)                # the cascade sets these per stage
    rkw.pop('confine', None)
    # ONE resolvability constant: the same alpha drives the cubic scaffold
    # (above) and the cascade's auto_coarse_count bound (relax_progressive) -
    # callers that used to pass alpha through relax_kw keep working
    rkw.setdefault('alpha', alpha)
    res = relax_progressive(pos, mass, target,
                            box, coarse_seed='merge', nsmooth=nsmooth,
                            envelope=env, filler_field_hook=filler_field_hook,
                            cascade=cascade, open_pad=open_pad,
                            verbose=verbose, **rkw)
    if getattr(res, 'envelope_audit', None) is None:   # envelope-off build
        res.envelope_audit = dict(
            n_full=int(len(res.pos)), n_survivor=int(len(res.pos)),
            n_culled=0, n_background=0, gas_mass=float(gas_mass),
            mf=float(gas_mass) / max(len(res.pos), 1))
        if filler_field_hook is not None:
            res.aux = filler_field_hook(res.pos)
    info = dict(gas_mass=float(gas_mass), count=int(count))
    if env is not None:
        info.update(rho_edge=env['rho_edge'],
                    rho_frame=env.get('rho_frame'),
                    delta_bg=env['delta_bg'], m=env['m'],
                    tail_reach=env.get('tail_reach'),
                    nl0=env.get('nl0'))
    res.table_build = info
    return res
