#!/usr/bin/env python
"""Shared helpers for the ICconvert_tipsyto*.py converters.

Kept converter-agnostic so every ICconvert_* script stays usable STANDALONE
(outside runtest.sh): the converters auto-detect the run physics from the IC
itself rather than being told by the caller.
"""
import os

__all__ = ["has_nonzero_b"]


def has_nonzero_b(tipsy_input):
    """True only if a BField aux exists AND carries a nonzero field.

    Hydro suite ICs (sedov, ...) ship all-zero BField auxes (writealltipsyaux
    writes a `.BField` aux for EVERY IC), so file presence alone is not a
    reliable MHD test -- probe the actual values. This is the single source of
    truth for "is this an MHD IC?" across the converters (and mirrors the
    RUNPHYSICS label runtest.sh declares per test for binary selection).
    """
    if not os.path.exists(tipsy_input + ".BFieldx"):
        return False
    import numpy as np
    import readtipsy as tip
    for lab in ("BFieldx", "BFieldy", "BFieldz"):
        b, _ = tip.readtipsyaux(tipsy_input, lab)
        if np.any(b != 0.0):
            return True
    return False
