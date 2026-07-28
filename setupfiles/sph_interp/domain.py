"""Domain selection for open SPH-to-grid captures.

The capture domain is deliberately derived only from the INPUT particle set.
It therefore does not change when a later glass is requested at another
particle count.
"""

import numpy as np


OPEN_PADDING_MODES = ("full_pad", "rho_edge_pad", "constant_pad")


def kernel_support_bounds(pos, h):
    """Smallest axis-aligned box containing the union of all SPH ``2h`` supports."""
    pos = np.asarray(pos, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) != len(h):
        raise ValueError("pos must be (N,3) and h must have one value per particle")
    if len(pos) == 0:
        raise ValueError("cannot determine support bounds from no particles")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(h)):
        raise ValueError("kernel-support inputs must be finite")
    if np.any(h <= 0.0):
        raise ValueError("smoothing lengths must be positive")
    support = 2.0 * h[:, None]
    return (np.ascontiguousarray(np.min(pos - support, axis=0)),
            np.ascontiguousarray(np.max(pos + support, axis=0)))


def open_domain_bounds(pos, mass, rho, h, mode="rho_edge_pad", nsmooth=64,
                       constant_fraction=0.1):
    """Return ``(lo, hi, audit)`` for an open particle capture.

    ``full_pad``
        Pad every side by the largest compact-support radius ``max(2*h)``.
    ``rho_edge_pad``
        Find the input-particle ``rho_edge`` at a faint-end mass of
        ``nsmooth * (M/N)`` and pad every side by ``2*h_edge,in``, where
        ``h_edge,in = [3*nsmooth*(M/N)/(32*pi*rho_edge,in)]**(1/3)``.
    ``constant_pad``
        Pad every side by ``constant_fraction`` times the longest unpadded
        particle-box span (0.1 by default).

    The padding is isotropic so the meaning is stable for thin/anisotropic
    systems.  ``audit`` records the selected scale and bounds for CLI output.
    """
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    n = len(pos)
    if n == 0:
        raise ValueError("cannot determine an open domain from no particles")
    if len(mass) != n or len(rho) != n or len(h) != n:
        raise ValueError("mass, rho and h must have one value per particle")
    if mode not in OPEN_PADDING_MODES:
        raise ValueError(
            f"unknown open padding {mode!r}; choose {OPEN_PADDING_MODES}")
    if (not np.all(np.isfinite(pos)) or not np.all(np.isfinite(mass))
            or not np.all(np.isfinite(rho)) or not np.all(np.isfinite(h))):
        raise ValueError("open-domain inputs must be finite")
    if np.any(mass <= 0.0) or np.any(rho <= 0.0) or np.any(h <= 0.0):
        raise ValueError("open-domain mass, rho and h must be positive")
    if int(nsmooth) <= 0:
        raise ValueError("nsmooth must be positive")
    if not np.isfinite(constant_fraction) or constant_fraction < 0.0:
        raise ValueError("constant_fraction must be finite and non-negative")

    particle_lo = pos.min(axis=0)
    particle_hi = pos.max(axis=0)
    span = particle_hi - particle_lo
    rho_edge_in = None
    h_edge_in = None
    if mode == "full_pad":
        pad = float(np.max(2.0 * h))
    elif mode == "rho_edge_pad":
        mean_mass = float(mass.sum()) / n
        order = np.argsort(rho, kind="stable")
        cumulative_mass = np.cumsum(mass[order])
        edge_mass = int(nsmooth) * mean_mass
        idx = min(int(np.searchsorted(cumulative_mass, edge_mass,
                                      side="left")), n - 1)
        rho_edge_in = float(rho[order[idx]])
        h_edge_in = float((3.0 * int(nsmooth) * mean_mass
                           / (32.0 * np.pi * rho_edge_in)) ** (1.0 / 3.0))
        pad = 2.0 * h_edge_in
    else:
        pad = float(constant_fraction) * float(span.max())

    lo = np.ascontiguousarray(particle_lo - pad)
    hi = np.ascontiguousarray(particle_hi + pad)
    audit = dict(mode=mode, pad=float(pad), particle_lo=particle_lo.copy(),
                 particle_hi=particle_hi.copy(), lo=lo.copy(), hi=hi.copy(),
                 rho_edge_in=rho_edge_in, h_edge_in=h_edge_in,
                 nsmooth=int(nsmooth),
                 constant_fraction=float(constant_fraction))
    return lo, hi, audit


def cubic_domain_bounds(lo, hi, about="domain"):
    """Grow ``[lo, hi]`` into the smallest CUBE containing it.

    Some consumers require a cubic domain with equal cells on every axis --
    POLARIS's octree is the strict case (`GridOcTree.cpp:548-552` pins the root
    to ``[-L/2, L/2]**3``), and a single-root octree over a cubic base grid is
    the only shape that maps onto it. A thin disc captured on its bounding box
    (the galaxy's ``(16,15,2)``-style domain) is the usual offender.

    Only ever EXPANDS: the short axes grow symmetrically to the longest span, so
    no particle can fall outside a domain that already contained it, and the
    added volume is empty padding that a deposit fills with genuine zeros.

    about : ``"domain"`` keeps the existing domain centre; ``"origin"`` centres
            the cube on 0 instead, which is what POLARIS's implicit recentring
            assumes -- with it, the exported grid's `recentre_shift` is zero and
            the object sits where the detector geometry expects it. Note this
            grows the cube to cover the FURTHEST corner from the origin, so it
            can be considerably larger than the ``"domain"`` cube if the input
            domain is off-centre.

    Returns ``(lo, hi, audit)``; ``audit['side']`` is the cube edge and
    ``audit['grew']`` the per-axis expansion (0 on an already-cubic axis).
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    if np.any(hi <= lo):
        raise ValueError("cubic_domain_bounds needs hi > lo on every axis")
    if about not in ("domain", "origin"):
        raise ValueError(f"about must be 'domain' or 'origin', got {about!r}")

    span = hi - lo
    if about == "origin":
        # smallest origin-centred cube containing the domain
        half = float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
        new_lo, new_hi = np.full(3, -half), np.full(3, half)
    else:
        side = float(np.max(span))
        centre = 0.5 * (lo + hi)
        new_lo, new_hi = centre - 0.5 * side, centre + 0.5 * side
    audit = dict(mode=f"cubic_{about}", side=float(new_hi[0] - new_lo[0]),
                 span_in=span.copy(), grew=(new_hi - new_lo) - span,
                 lo=new_lo.copy(), hi=new_hi.copy())
    return (np.ascontiguousarray(new_lo), np.ascontiguousarray(new_hi), audit)
