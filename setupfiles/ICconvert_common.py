#!/usr/bin/env python
"""Shared helpers for the ICconvert_tipsyto*.py converters.

Kept converter-agnostic so every ICconvert_* script stays usable STANDALONE
(outside runtest.sh): the converters auto-detect the run physics from the IC
itself rather than being told by the caller.
"""
import os

__all__ = ["has_nonzero_b", "find_sibling_param", "gadget_particle_argv",
           "read_gasoline_param"]


def read_gasoline_param(path, missing="empty", duplicates="last"):
    """Parse a gasoline `.param` (`key = value`, `#` comments) into a dict of
    value STRINGS. No numeric coercion - callers convert what they need, so a
    value is never silently retyped (see `_f` helpers in the converters).

    missing : 'empty' (default) returns {} when `path` is falsy or absent -
              what the converters want, since a sibling .param is optional.
              'raise' lets open() raise, for callers that require the file.
    duplicates : 'last' (default) - a later assignment overrides an earlier
              one, which is gasoline's own semantics. 'first' keeps the FIRST
              occurrence.

    **Why `duplicates` is a flag and not a hardcoded rule.** Repeated keys do
    occur in real .param files here - `dg1_superbubble.param` repeats
    `bVDetails`, `PGAL.param` repeats `bDoDensity` - because the suite
    concatenates a few IC variants into one file.
    `sph_interp/_arepo/arepo_param.py` deliberately takes 'first' for that
    reason; the converters take 'last'.

    A malformed line with an empty key (` = orphan`) is skipped, as in
    arepo_param: no caller can look up `''`.
    """
    if duplicates not in ("last", "first"):
        raise ValueError("duplicates must be 'last' or 'first'")
    vals = {}
    if missing == "empty" and (not path or not os.path.exists(path)):
        return vals
    with open(path) as f:
        for line in f:
            line = line.split("#")[0]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if not k:
                continue
            if duplicates == "last" or k not in vals:
                vals[k] = v
    return vals


def find_sibling_param(tipsy_input):
    """The gasoline `.param` beside a tipsy snapshot `<name>.NNNNN`: try
    `<name>.param` (strip the snapshot index). Returns None if absent.
    """
    base = tipsy_input
    if "." in os.path.basename(base):
        base = base.rsplit(".", 1)[0]
    cand = base + ".param"
    return cand if os.path.exists(cand) else None


def gadget_particle_argv(args):
    """Argv for a 1:1 particle copy -> GADGET type-0 HDF5.

    Shared by the GIZMO and OpenGadget3 converters. The "arepo" output format IS the
    GADGET writer; `--voronoi-particles` selects the straight copy path. Kept
    together deliberately: both consumers must stay in lockstep on how the
    GADGET writer is invoked.

    NB the AREPO converter has its own `build_argv` - it carries the grid/AMR
    refinement modes and is genuinely different, not a third copy of this.
    """
    argv = [args.input, args.output, "--output-format", "arepo",
            "--voronoi-particles", "--fields", args.fields]
    if args.tu_fac is not None:
        argv += ["--tu-fac", str(args.tu_fac)]
    return argv


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
