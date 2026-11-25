"""Glass relaxation driver: the per-iteration loop of the C icgenerator.

Per iteration: neighbors/h -> SPH density -> target density -> pseudo
pressure pP = (rho/rho0)^rhopow -> pair forces -> displacement
normalization -> adaptive-step move x += icvel * ICR0 * h -> step control.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .ball import BallNeighborList
from .density import DensityField
from .kernels import kernel_spec
from .neighbors import NeighborList, _gather3, cell_order
from .sph_core import (density_loop, density_loop_csr, get_force_loop,
                       get_force_loop_csr)
from .stepper import (Icr0Stop, StepController,
                      apply_step, normalize_displacements, pseudo_pressure)
from .coarse import coarse_correction

_DUMMY = np.empty((1, 1))
COARSE_R = 2.0     # coarse_fft smoothing length / largest (99th percentile) h
COARSE_CAP = 0.5   # coarse_fft: largest shift per application, in units of the particle's h
COARSE_HPCT = 99.0      # coarse_fft: R = COARSE_R x this percentile of h (99 = the sparsest region)
COARSE_CAP_GLOBAL = True  # True: one factor min(1, CAP/max |u|/h) for all particles; False: per particle
# coarse_every > 0 (transport mode, see relax()): R from the median h, so a compact core is on the mesh, and the
# shift capped by its strain (max |grad u| <= COARSE_STRAIN_CAP keeps x -> x + u one-to-one), which does not
# shrink with resolution as the |u|/h cap does (soliton 128^3: x0.03 per application under the h cap)
COARSE_EVERY_HPCT = 50.0
COARSE_STRAIN_CAP = 0.5


@dataclass
class IterStats:
    it: int
    frms: float          # true per-particle RMS residual pseudo-pressure force -> 0
    dens_err: float      # <|1 - rho/rho0|> density error
    icr0: float
    icrloop0: float
    polish: bool
    pos: np.ndarray
    h: np.ndarray
    rho: np.ndarray
    rho0: np.ndarray
    q0: float = float('nan')   # mean partition of unity <sum m_j/rho_j W> -> 1 (zeroth-order)
    e0: float = float('nan')   # mean |zeroth-order force error| (diagnostics=True)


@dataclass
class RelaxResult:
    pos: np.ndarray
    h: np.ndarray
    rho: np.ndarray
    niter: int
    converged: bool
    history: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    # per-iteration error tails (when diagnostics=True): (it, denserr_mean,
    # denserr_p95, denserr_max, force_mean, force_p95, force_max, polish)
    tail_history: list = field(default_factory=list)
    # which finish criterion fired: 'icr0_abs' (the median-smoothed icr0 < the
    # icr0_stop quality threshold - the default finish), 'polish_iter' (the fixed
    # polish-iteration budget), or 'max_iter' (the hard cap)
    stop_reason: str = 'max_iter'
    # per-stage breakdown for relax_progressive: one dict per level with
    # name/niter/converged/wall/n (empty for a plain single relax)
    levels: list = field(default_factory=list)
    # (name, history) of every relax_progressive level, in order (empty for a
    # plain single relax); plot_history(levels=...) concatenates them
    level_history: list = field(default_factory=list)
    # glass state captured at polish ENTRY (the end of the sweep phase, before
    # the gentle polish ladder runs) - lets the caller measure the pre-polish
    # diagnostic and see how much the polish phase changed it. None if polish
    # was never entered. niter at entry is polish_iter0.
    pos_prepolish: np.ndarray = None
    h_prepolish: np.ndarray = None
    rho_prepolish: np.ndarray = None
    polish_iter0: int = -1
    # envelope (open-box / free-surface) build audit, set by relax_progressive
    # when envelope!=None: n_full/n_survivor/n_culled/gas_mass/mf. None otherwise.
    envelope_audit: dict = None
    # per-survivor gathered fields (reconstruct: vel/u/eps/...), set by
    # relax_progressive's filler_field_hook after the cull. None otherwise.
    aux: dict = None


def relax(pos, mass, density, box=None, nsmooth=64,
          sigma=0.0, sigma_f=1.0, gradient='onesided-isph', rhopow=2.0,
          icr0=1.0, icrloop0=0.5, icr0rate=1.5, max_iter=4800,
          margin=8, reorder=True, one_sided=False,
          hmode='knn', nsmoothmin=32, self_bias=True, kernel='wendlandc2',
          confine=False, max_step_frac=None,
          cap_mode='h', reanneal=True, momentum=0.0, icr0_stop=3e-3,
          stop_window=10, coarse_fft=0.0, coarse_serial=False, coarse_every=0,
          polish_iter=0, polish_down=0.1, polish_up=0.002,
          diagnostics=False, force_table=None,
          callback=None, callback_every=100, verbose=False):
    """Relax particles into a glass matching a target density field.

    force_table : forcematch.ForceTable or None. When set (force-matched mode,
        see forcematch.py) the relax works on the real pressure A(r) rho^Gamma
        plus the table's g instead of the pseudo-pressure; rho0 then only
        enters the diagnostics.

    pos : (N,3) positions (box centered on 0 when periodic); modified copy
        is returned. mass : (N,). density : DensityField (or callable taken
        as rho(pos) over (N,3)). box : (3,) periodic box lengths or None
        for an open domain. reorder : sort the particle
        arrays into cell order at each neighbor rebuild (cache locality;
        results are returned in the input order regardless). one_sided :
        SPH-EXA-style transpose-free force walk - faster, but pair forces
        are no longer exactly antisymmetric (negligible for relaxation;
        keep False when validating against the C generator). hmode : which
        smoothing length the force kernel (and the move step) use -
        'knn' (default, h = 1/2 dist to the nSmooth-th neighbour),
        'measured' (h = (m*hfact3/rho_sph)^1/3, the density-based h the C
        DenDVDX sets on fBall2 before the force resmooth, hfact3 =
        3*nSmooth/32pi), or 'target' (h = (m*hfact3/rho0)^1/3 from the
        analytic/table target density - noise-free, the cleanest choice for
        a glass generator). sigma : CORRPARTITION density partition exponent
        (dFacQ); only 0 = the plain SPH density estimate is implemented.
        sigma_f, gradient : the pair force, gasoline's
        CORRPARTITION family (sph_core): sigma_f the force volume exponent
        (dFacQf, 1 = GDSPH, 0 = TSPH), gradient one of sph_core.GRADIENTS;
        the default (1, 'onesided-isph') is gasoline's GDFORCE+ISPH build,
        (1, 'average-sph') the C icgenerator; each reproduces the matching
        gasoline build's force to round-off. Density and pP are always
        computed with kNN h
        (matching the C, which reuses the DenDVDX density), so only the
        force/move length changes. nsmoothmin : floor on the density-based
        h (measured/target modes only) - h is never allowed below the kNN
        radius enclosing ~nsmoothmin real neighbours, so a particle that
        sits in a sparse region while rho0 says "dense" (analytic h -> ~0)
        still gathers neighbours instead of an empty ball / zero force.
        confine : when True, cap each move component to one smoothing length
        (max_step_frac=1.0) and forbid cross-box jumps - the fine progressive
        phase, where the large-scale field is already in place and only local
        relaxation is wanted (default False = the unbounded coarse/single-relax
        move). reanneal : when False, skip the ICRLOOP0 re-anneal/halve ladder
        in the step controller and drop straight into polish once icr0 decays
        below nfac (also the fine progressive phase; default True = validated
        behaviour). icr0_stop : PRIMARY finish criterion (default 3e-3) - stop
        once the median-smoothed icr0 (the maximum step as a FRACTION of h, the
        move being icvel*icr0*h with icvel normalised so its max component <= 1)
        falls below this threshold. icr0 is resolution-, gradient- and
        particle-count invariant (h scales out), so a fixed threshold travels
        across cases - unlike the residual force frms, whose absolute scale
        depends on mass/h and the density gradient. It IS the quality dial:
        ~3e-3 stops as icr0 settles onto its critical-step setpoint in the
        confined refine stage (denserr ~+7% above asymptote, the fast/"good"
        glass); tighten to ~1e-3 for a near-asymptote glass. The trailing-median
        smooth over stop_window kills the controller's per-iteration icr0
        sawtooth so the crossing is reproducible. This is the SINGLE finish for
        every regime (confined or box-amplified single relax): icr0 is the step
        as a fraction of h, so the threshold is amplification-independent. Set
        icr0_stop=0.0 to disable (then polish_iter or max_iter ends the run).
        polish_iter :
        when > 0, a FIXED polish-iteration budget - stop after exactly this many
        iterations in the polish phase (the analog of the cascade's inter_iter
        fixed-length heal, applied to the full-resolution run: the deep polish
        past entry barely improves quality, so just do a short fixed number).
        Composes with icr0_stop - whichever fires first wins (an early icr0_stop
        convergence still cuts it short); set icr0_stop=0 to make polish_iter the
        sole finish. 0 = off,
        stop_reason='polish_iter'. momentum : heavy-ball coefficient on the move
        step (0.0 = memoryless steepest descent); > 0 blends
        the previous normalised descent direction in (see stepper.apply_step),
        accelerating the slow frms tail that pure steepest descent cannot beat
        on the stiff 360:1 gradient. Composes with confine (the cap clamps the
        final blended move) and the icr0 feedback controller (which then sees
        momentum-smoothed moves). polish_down/polish_up : polish-phase icr0
        shrink/grow coefficients (StepController; C defaults 0.1/0.002 - a
        47:1 ratchet at rate 1.5 that freezes the run ~250 iters after
        polish entry; ~0.01 keeps icr0 mobile through a long polish budget).
        Momentum is HARDWIRED to the polish phase: beta stays 0 while the
        re-anneal ladder / icr0 decay is active and turns on only once the
        controller enters polish (icfmax = -1); direction memory across anneal
        kicks costs ~30% more iterations at identical quality. The disp_prev
        buffer starts clean at polish entry because beta=0 never writes it.
        coarse_fft : > 0 (periodic box only) = the
        FFT coarse-grid correction (coarse.py) does the long-wavelength modes,
        so the re-anneal ladder starts at coarse_fft*h (no box-crossing rungs)
        and at every rung end and at polish entry the particles are shifted by
        the correction (capped at COARSE_CAP*h). 0 = off. The FFT probe (the
        correction still needed) is printed at the end of every periodic relax.
        coarse_serial : deposit the correction on one numba thread, so
        --coarse-fft runs are bitwise reproducible (A/B tests); default False =
        parallel (fastest; the run-dependent summation order makes reruns
        differ like different seeds).
        coarse_every : > 0 (with coarse_fft) = transport mode for a smooth
        profile far from the pre-IC (soliton): the correction is also applied
        every coarse_every iterations outside polish, with R = COARSE_R x the
        median h and the strain cap COARSE_STRAIN_CAP instead of the |u|/h cap.
        The rung-end-only default never fires early enough there (the first
        rung needs the residual gone; soliton 128^3 at n400: ~3000 iterations)
        and cannot see a core below 2 x h_99. Not for sharp compact objects:
        areablob's edge error runs away (blob emptied). 0 = off (default).
        Remaining parameters mirror the gdicgen ICCONFIG flags.
    """
    if sigma != 0.0:
        raise ValueError('sigma != 0 needs the CORRPARTITION partition '
                         'iteration, not implemented; use sigma=0')
    if hmode not in ('knn', 'measured', 'target'):
        raise ValueError(f"hmode must be knn/measured/target, got {hmode!r}")
    if nsmoothmin >= nsmooth:
        nsmoothmin = nsmooth - 1
    hfact3 = 3.0 * nsmooth / (32.0 * np.pi)
    # kernel (kernels.KERNELS) and its self-density bias correction: the W(0)
    # self term scaled as gasoline's smf->Wzero so a uniform glass relaxes to
    # rho -> ~1.0; self_bias=False -> bare W(0).
    kid, kp, w0self, _ = kernel_spec(kernel, nsmooth, self_bias)
    pos = np.array(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    if not isinstance(density, DensityField):
        fn = density
        density = type('_Wrap', (DensityField,), {'rho0': lambda self, p: np.asarray(fn(p), dtype=np.float64)})()
    # kernel-matched target readout: fields flagged smooth_readout
    # get rho0 evaluated as the KERNEL AVERAGE at each particle's own h
    # (rho0_h) wherever the pseudo-pressure is formed - SPH rho is a kernel
    # average at that same h, so the ratio compares like with like and a
    # target sharper than the kernel cannot manufacture fake overpressure.
    smooth = bool(getattr(density, 'smooth_readout', False))
    boxarr = None if box is None else np.asarray(box, dtype=np.float64)
    periodic = boxarr is not None
    if periodic:
        # the njit pair loops use compare-based min-image, which requires
        # wrapped coordinates
        pos -= boxarr * np.rint(pos / boxarr)

    n = len(pos)
    # current slot -> original particle id (identity unless reorder swaps)
    order = np.arange(n)

    def to_input_order(arr):
        out = np.empty_like(arr)
        out[order] = arr
        return out
    rho = np.empty(n)
    icvel = np.empty((n, 3))
    q0d = np.empty(n) if diagnostics else _DUMMY[0]
    e0 = np.empty((n, 3)) if diagnostics else _DUMMY
    tail_history = []

    # gather backend: 'ball' for the density-based h modes (no kNN - h sets a
    # mean nSmooth via the h-rho relation and the force/density are a fixed-
    # radius ball gather, as in the CORRPARTITION gasoline path), 'knn' for
    # the kNN candidate lists. 'auto' = ball iff hmode is measured/target.
    # the density-based-h modes (measured/target) use a fixed-radius ball gather
    # (CORRPARTITION gasoline path, consistent density-based h, no kNN); knn h
    # uses the kNN gather. The choice follows hmode directly.
    use_ball = hmode != 'knn'
    if use_ball and not periodic:
        raise ValueError('the measured/target h modes need a periodic box; '
                         'use hmode="knn" for an open domain')

    nblist = NeighborList(boxarr, nsmooth, margin=margin,
                          build_transpose=not one_sided)
    force_loop = get_force_loop(sigma_f, gradient, one_sided=one_sided,
                                kernel=kernel, nsmooth=nsmooth)
    ballnl = BallNeighborList(boxarr, nsmoothmin=nsmoothmin,
                              build_transpose=not one_sided) if use_ball \
        else None
    ball_force = get_force_loop_csr(sigma_f, gradient, one_sided=one_sided,
                                    kernel=kernel, nsmooth=nsmooth) \
        if use_ball else None
    # In confine mode the per-step move is capped at one h, so the controller's
    # periodic box-amplification (icr0 -> 0.99*box/h_avg, meant to cross the
    # whole box in the early sweep) is exactly what we forbid; passing box=None
    # to the controller disables both that amplification and the step_zero
    # /h_avg, so capped steps start at ~icr0*h and anneal down. The actual move
    # + wrap in apply_step still use the real periodic box.
    if coarse_fft > 0.0:
        if not periodic:
            raise ValueError('coarse_fft needs a periodic box')
        if force_table is not None:
            raise ValueError('coarse_fft assumes the pseudo-pressure (rho a = grad ln pP); '
                             'not valid with force_table')
        # ladder in units of h from coarse_fft down: box=None = no box amplification
        ctrl = StepController(coarse_fft, 1.0, icr0rate, nsmooth, box=None,
                              reanneal=reanneal, polish_down=polish_down,
                              polish_up=polish_up)
    else:
        ctrl = StepController(icr0, icrloop0, icr0rate, nsmooth,
                              box=(None if confine else boxarr), reanneal=reanneal,
                              polish_down=polish_down, polish_up=polish_up)
    fraw = np.empty((n, 3))   # raw force of the current iteration (coarse correction/probe)
    dummybox = boxarr if periodic else np.ones(3)
    # The per-step move cap (apply_step clamps each component to max_step_frac*h)
    # is normally tied to confine: confine -> cap at 1*h, else no cap. They are
    # decoupled here so an experiment can keep confine's box=None controller
    # (icr0 stays in units of h, starting at ~icr0*h with no box amplification)
    # while DISABLING the cap (max_step_frac=0.0): pure steepest descent then
    # still moves <= icr0*h <= 1*h, so only the momentum term can push a particle
    # above one smoothing length. max_step_frac=None = follow confine (default).
    if max_step_frac is None:
        max_step_frac = 1.0 if confine else 0.0
    # cap_mode chooses what the cap is a fraction OF:
    #   'h'    : cap = max_step_frac * h        (fixed length).
    #            Only binds while icr0/(1-momentum) > max_step_frac, i.e. the
    #            first few iters before icr0 anneals - inert thereafter.
    #   'icr0' : cap = max_step_frac * icr0 * h (a fraction of the CURRENT
    #            max-force step icr0*h). Since the move is d*icr0*h with the
    #            heavy-ball direction |d| <= 1/(1-momentum), this caps |d| at
    #            max_step_frac directly - an icr0-INDEPENDENT bound on how far
    #            momentum may amplify the step, consistent across the whole run.
    #            max_step_frac=1 => no particle exceeds the max-force step icr0*h
    #            (momentum can push a particle UP TO that step but not past it).
    if cap_mode not in ('h', 'icr0'):
        raise ValueError(f"cap_mode must be 'h' or 'icr0', got {cap_mode!r}")
    cap_icr0 = cap_mode == 'icr0'

    # finish criterion (all gated to the polish phase):
    #   icr0_stop (default 3e-3) : stop when the median-smoothed icr0 (the max
    #     step as a FRACTION of h - resolution/gradient/amplification invariant)
    #     falls below icr0_stop. The smooth makes the crossing reproducible. This
    #     is the single finish for every regime (confined or box-amplified).
    #   polish_iter (independent) : a fixed polish-iteration budget - stop after
    #     polish_iter iterations in polish regardless of icr0_stop; composes,
    #     first to fire wins.
    # max_iter is the hard cap (set icr0_stop=0 and polish_iter=0 to run it out).
    if icr0_stop > 0.0:
        pstop = Icr0Stop(threshold=icr0_stop, window=stop_window)
        pstop_reason = 'icr0_abs'
    else:
        pstop = None
        pstop_reason = None
    history = []
    converged = False
    stop_reason = 'max_iter'
    prepolish = None   # (pos, h, rho) snapshot at the first polish iteration
    polish_count = 0   # iterations spent in the polish phase (for polish_iter)
    it = 0
    t_start = time.perf_counter()
    if verbose:
        print(f'relax: n={n} nsmooth={nsmooth} kernel={kernel} sigma_f={sigma_f} '
              f'gradient={gradient} '
              f'rhopow={rhopow} '
              f'box={None if boxarr is None else boxarr.tolist()}', flush=True)
    pP = np.empty(n)
    pdisp = np.zeros(n)
    # heavy-ball velocity workspace (input-order, permuted alongside pos on
    # reorder so momentum stays particle-aligned; NOT reset on rebuild).
    disp_prev = np.zeros((n, 3))
    nblist.track_displacements(pdisp)
    if use_ball:
        ballnl.track_displacements(pdisp)
        # bootstrap density-based h (measured uses the previous rho; iter 1
        # seeds from the target, matching the C bStarting step)
        rho[:] = density.rho0(pos)
    # per-phase wall-time accumulators (printed in the verbose summary)
    pc = time.perf_counter
    tprof = {'nbr': 0.0, 'dens': 0.0, 'pP': 0.0, 'frc': 0.0, 'step': 0.0,
             'move': 0.0}

    def _hin(pos, mass, rho, rho0):
        return np.cbrt(mass * hfact3 / (rho0 if hmode == 'target' else rho))

    for it in range(1, max_iter + 1):
        if use_ball:
            # ball gather: h = (m*hfact3/rho)^1/3 sets a *mean* nSmooth via
            # the h-rho relation; density, pP and force all use this same h
            # (consistent, no kNN). target h is known a priori from rho0;
            # measured h from the previous iteration's SPH density. Reorder
            # into cell order at each rebuild for cache-local CSR gathers.
            t = pc()
            rho0 = density.rho0(pos) if hmode == 'target' else None
            h_in = _hin(pos, mass, rho, rho0)
            # decide staleness once, on the pre-reorder state; the reorder
            # permutes pos/h_in but not the ball's cached rcached/pdisp, so a
            # second needs_rebuild() inside update() would test misaligned
            # arrays. Pass the decision through instead.
            rebuild = ballnl.needs_rebuild(h_in)
            if reorder and rebuild:
                perm = cell_order(pos, boxarr)
                if perm is not None:
                    newpos = np.empty_like(pos)
                    _gather3(pos, perm, newpos)
                    pos = newpos
                    if momentum > 0.0:
                        newdp = np.empty_like(disp_prev)
                        _gather3(disp_prev, perm, newdp)
                        disp_prev = newdp
                    mass = np.ascontiguousarray(mass[perm])
                    rho = np.ascontiguousarray(rho[perm])
                    order = order[perm]
                    rho0 = density.rho0(pos) if hmode == 'target' else None
                    h_in = _hin(pos, mass, rho, rho0)
            indptr, indices, hforce, rev_indptr, rev_indices = \
                ballnl.update(pos, h_in, rebuild=rebuild)
            tprof['nbr'] += pc() - t
            t = pc()
            density_loop_csr(pos, mass, hforce, indptr, indices, dummybox,
                             periodic, rho, w0self, kid, kp)
            tprof['dens'] += pc() - t
            t = pc()
            if hmode != 'target':
                rho0 = density.rho0_h(pos, hforce) if smooth \
                    else density.rho0(pos)
            elif smooth:
                # kernel-matched pP target at the force h; the h_in seed above
                # keeps the cheap point read (it only sizes the ball)
                rho0 = density.rho0_h(pos, hforce)
            dens_err = pseudo_pressure(rho, rho0, rhopow, pP)
            if force_table is not None:
                force_table.pressure(pos, rho, pP)
            tprof['pP'] += pc() - t
            t = pc()
            ball_force(pos, mass, hforce, rho, pP, indptr, indices,
                       rev_indptr, rev_indices, dummybox, periodic, icvel,
                       q0d, e0, diagnostics)
            if force_table is not None:
                force_table.add_g(pos, icvel)
            tprof['frc'] += pc() - t
        else:
            t = pc()
            if reorder and periodic and nblist.needs_rebuild():
                perm = cell_order(pos, boxarr)
                if perm is not None:
                    newpos = np.empty_like(pos)
                    _gather3(pos, perm, newpos)
                    pos = newpos
                    if momentum > 0.0:
                        newdp = np.empty_like(disp_prev)
                        _gather3(disp_prev, perm, newdp)
                        disp_prev = newdp
                    mass = np.ascontiguousarray(mass[perm])
                    order = order[perm]
            idx, h, rev_indptr, rev_indices = nblist.update(pos)
            tprof['nbr'] += pc() - t
            t = pc()
            density_loop(pos, mass, h, idx, dummybox, periodic, rho, w0self,
                         kid, kp)
            tprof['dens'] += pc() - t
            t = pc()
            rho0 = density.rho0_h(pos, h) if smooth else density.rho0(pos)
            dens_err = pseudo_pressure(rho, rho0, rhopow, pP)
            if force_table is not None:
                force_table.pressure(pos, rho, pP)
            # the kNN-gather path is taken only for hmode='knn' (measured/target
            # use the ball path above), so the force kernel uses the same kNN h
            # as the density.
            hforce = h
            tprof['pP'] += pc() - t
            t = pc()
            force_loop(pos, mass, hforce, rho, pP, idx, rev_indptr,
                       rev_indices, dummybox, periodic, icvel, q0d, e0,
                       diagnostics)
            if force_table is not None:
                force_table.add_g(pos, icvel)
            tprof['frc'] += pc() - t
        if diagnostics:
            # per-particle tails (before normalize clobbers icvel): the RAW
            # residual force magnitude and the relative density error. The mean
            # is in history; the p95/max are interface-dominated, so these show
            # whether deep polish is still tightening the TAIL (the 360:1 shell)
            # after the mean has flattened. percentile cost is why it's gated.
            fmag = np.sqrt((icvel * icvel).sum(1))
            err = np.abs(1.0 - rho / rho0)
            tail_history.append((
                it, dens_err, float(np.percentile(err, 95)), float(err.max()),
                float(fmag.mean()), float(np.percentile(fmag, 95)),
                float(fmag.max()), ctrl.polish))
        # snapshot the glass the moment we ENTER polish (this iteration's
        # incoming pos/h/rho, before any polish move), so the caller can
        # measure how much the polish ladder changes the sweep-phase glass.
        if ctrl.polish and prepolish is None:
            prepolish = (to_input_order(pos).copy(),
                         to_input_order(hforce).copy(),
                         to_input_order(rho).copy(), it)
        t = pc()
        if periodic:
            np.copyto(fraw, icvel)
        frms, _ = normalize_displacements(icvel)
        ctrl.update_step(frms, float(np.mean(hforce)))
        tprof['step'] += pc() - t

        t = pc()
        # cap argument: in 'icr0' mode the cap is a fraction of the current
        # max-force step (icr0*h), so fold icr0 into the fraction passed to
        # apply_step (which still computes cap = arg*h) - no kernel change. The
        # move itself already scales by icr0, so this caps |d| at max_step_frac.
        cap_arg = max_step_frac * ctrl.icr0 if cap_icr0 else max_step_frac
        # Momentum is polish-only: beta=0 through the anneal ladder / icr0
        # decay (big coherent transient moves + kicks, where direction memory
        # costs ~30% more iterations), on in the quiescent polish descent. disp_prev is only written when the
        # effective beta > 0, so polish entry starts with a clean buffer.
        mom_now = momentum if ctrl.polish else 0.0
        max_step = apply_step(pos, icvel, hforce, ctrl.icr0, dummybox,
                              periodic, pdisp, cap_arg, disp_prev,
                              mom_now)
        (ballnl if use_ball else nblist).record_move(max_step)
        tprof['move'] += pc() - t
        if not np.isfinite(dens_err):
            raise RuntimeError(
                f"relaxation diverged at iteration {it} (density is no "
                "longer finite). With box=None there is no confinement: "
                "the pseudo-pressure is positive, so a cloud with a free "
                "surface expands without bound - use a periodic box with "
                "a floor density as the C icgenerator does.")
        rung_end = ctrl.icr0 < ctrl.nfac and not ctrl.polish
        was_polish = ctrl.polish
        ctrl.update_restart(dens_err)
        if coarse_fft > 0.0 and (rung_end or (ctrl.polish and not was_polish)
                                 or (coarse_every > 0 and not ctrl.polish and it % coarse_every == 0)):
            hpct = COARSE_EVERY_HPCT if coarse_every > 0 else COARSE_HPCT
            u, info = coarse_correction(pos, mass, fraw, boxarr, COARSE_R * np.percentile(hforce, hpct),
                                        hforce, rho, serial=coarse_serial, strain=coarse_every > 0)
            uh = np.abs(u).max(axis=1) / hforce
            if coarse_every > 0:
                s = si = min(1.0, COARSE_STRAIN_CAP / max(info['strain'], 1e-300))
            elif COARSE_CAP_GLOBAL:
                s = si = min(1.0, COARSE_CAP / max(uh.max(), 1e-300))
            else:
                si = np.minimum(1.0, COARSE_CAP / np.maximum(uh, 1e-300))
                s = float(si.min())
            pos += (si if np.ndim(si) == 0 else si[:, None]) * u
            pos -= boxarr * np.rint(pos / boxarr)
            pdisp += si * uh * hforce
            (ballnl if use_ball else nblist).record_move(float((si * uh * hforce).max()))
            if verbose:
                print(f'iter {it:5d}  coarse fft: needed max|u|/h {uh.max():.3e}, '
                      f'rms dln rho {info["dlnrho_rms"]:.3e}, applied x{s:.3f}'
                      f'{"  (polish entry)" if ctrl.polish else ""}', flush=True)

        if verbose and (it % 25 == 0 or it == 1):
            print(f"iter {it:5d}  frms {frms:.3e}  denserr {dens_err:.3e}  "
                  f"icr0 {ctrl.icr0:.3e}  icrloop0 {ctrl.icrloop0:.3e}  "
                  f"polish {ctrl.polish}  "
                  f"rebuilds {(ballnl if use_ball else nblist).nbuilds}  "
                  f"t {time.perf_counter() - t_start:.0f}s",
                  flush=True)
        history.append((it, frms, dens_err, ctrl.icr0, ctrl.icrloop0, ctrl.polish))
        if callback is not None and it % callback_every == 0:
            callback(IterStats(it, frms, dens_err, ctrl.icr0, ctrl.icrloop0,
                               ctrl.polish, to_input_order(pos),
                               to_input_order(hforce), to_input_order(rho),
                               to_input_order(rho0)))
        # icr0_stop finish in polish; pstop.update has state, feed only in polish
        if pstop is not None and ctrl.polish:
            if pstop.update(ctrl.icr0):
                converged = True
                stop_reason = pstop_reason
                break
        if polish_iter > 0 and ctrl.polish:
            # fixed polish-iteration budget: do exactly polish_iter polish iters
            # then stop (composes with icr0_stop - first to fire wins)
            polish_count += 1
            if polish_count >= polish_iter:
                converged = True
                stop_reason = 'polish_iter'
                break

    # final state at the relaxed positions, mapped back to input order
    if use_ball:
        if hmode == 'target':
            h_in = np.cbrt(mass * hfact3 / density.rho0(pos))
        else:
            h_in = np.cbrt(mass * hfact3 / rho)
        indptr, indices, hret, _, _ = ballnl.update(pos, h_in)
        density_loop_csr(pos, mass, hret, indptr, indices, dummybox,
                         periodic, rho, w0self, kid, kp)
    else:
        idx, h, rev_indptr, rev_indices = nblist.update(pos)
        density_loop(pos, mass, h, idx, dummybox, periodic, rho, w0self,
                         kid, kp)
        hret = h   # kNN-gather path is hmode='knn' only
    if verbose and periodic:
        hpct = COARSE_EVERY_HPCT if coarse_every > 0 else COARSE_HPCT
        u, info = coarse_correction(pos, mass, fraw, boxarr, COARSE_R * np.percentile(hforce, hpct),
                                    hforce, rho, serial=coarse_serial)
        print(f'coarse fft probe (correction still needed): max|u|/h '
              f'{(np.abs(u).max(axis=1) / hforce).max():.3e}, rms dln rho {info["dlnrho_rms"]:.3e}',
              flush=True)
    if verbose:
        tot = sum(tprof.values()) or 1.0
        nb = (ballnl if use_ball else nblist).nbuilds
        print('phase timings (s, %% of summed loop): ' + '  '.join(
            f'{k} {v:.1f} ({100 * v / tot:.0f}%)' for k, v in tprof.items())
            + f'  | rebuilds {nb}/{it} ({100 * nb / it:.0f}%)'
            + f'  | stop {stop_reason} @ {it}', flush=True)
    return RelaxResult(pos=to_input_order(pos), h=to_input_order(hret),
                       rho=to_input_order(rho), niter=it,
                       converged=converged, history=history,
                       tail_history=tail_history,
                       timings=tprof, stop_reason=stop_reason,
                       pos_prepolish=prepolish[0] if prepolish else None,
                       h_prepolish=prepolish[1] if prepolish else None,
                       rho_prepolish=prepolish[2] if prepolish else None,
                       polish_iter0=prepolish[3] if prepolish else -1)
