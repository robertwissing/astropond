"""Domain selection for open SPH-to-grid captures.

The capture domain is deliberately derived only from the INPUT particle set.
It therefore does not change when a later glass is requested at another
particle count.
"""

import numpy as np


OPEN_PADDING_MODES = ("full_pad", "rho_edge_pad", "constant_pad")


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
