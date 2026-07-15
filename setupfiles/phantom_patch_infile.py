#!/usr/bin/env python
"""Patch run-control keys into a Phantom `.in` file, preserving its structure.

Phantom rejects an incomplete `.in`: it rewrites a COMPLETE `.in` (filling in
default values for every mandatory key of *this* binary's compile options) and
exits with "try again using revised input file" without evolving. Rather than
hardcode the full, build-dependent key list in our converter, the suite lets
Phantom emit its own complete `.in` (pass 1), then runs this helper to re-assert
the handful of run-control values we actually care about (dumpfile, tmax, dtmax,
gamma, ...) before running Phantom again to evolve (pass 2).

    phantom_patch_infile.py <infile> key=value [key=value ...]
    phantom_patch_infile.py <infile> --from <minimal.in>

The `--from` form reads `key = value` lines from another `.in` (e.g. the minimal
`.in` our converter wrote with the values we care about) and applies them all,
so the suite re-asserts exactly the converter's run-control values without having
to recompute them in shell.

Each matching `   key =  value   ! comment` line has only its value replaced; the
key, alignment column and trailing `! comment` are preserved. A requested key that
is NOT already in the file is reported on stderr and skipped (never appended):
Phantom's rewritten `.in` is the authoritative list of valid keys for this build,
and appending an unknown key (e.g. `gamma`, which this build reads from the dump,
not the `.in`) only re-triggers Phantom's "unknown variable" warning on the next
run. Dropping a key Phantom doesn't accept is the correct outcome.
"""
import re
import sys

# `<indent>key =  value   ! comment`  — capture the pieces so we can swap only value.
_LINE = re.compile(r"^(?P<pre>\s*)(?P<key>\S+)(?P<mid>\s*=\s*)(?P<val>\S+)(?P<post>.*)$")


def patch(path, overrides):
    with open(path) as f:
        lines = f.readlines()

    seen = set()
    out = []
    for line in lines:
        m = _LINE.match(line)
        if m and m.group("key") in overrides:
            key = m.group("key")
            seen.add(key)
            out.append("%s%s%s%s%s\n" % (m.group("pre"), key, m.group("mid"),
                                         overrides[key], m.group("post").rstrip("\n")))
        else:
            out.append(line if line.endswith("\n") else line + "\n")

    # Warn (don't append) on any requested key the file doesn't already contain:
    # the rewritten `.in` is Phantom's authoritative key list for this build.
    for key in overrides:
        if key not in seen:
            sys.stderr.write("phantom_patch_infile: key %r not in %s; skipping "
                             "(not a valid key for this Phantom build)\n" % (key, path))

    with open(path, "w") as f:
        f.writelines(out)


def _overrides_from_infile(path):
    """Pull key->value from a `.in`-format file (ignores comments/blank lines)."""
    ov = {}
    with open(path) as f:
        for line in f:
            line = line.split("!")[0].split("#")[0]
            if "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k and v:
                    ov[k] = v
    return ov


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    path = argv[0]
    if argv[1] == "--from":
        if len(argv) != 3:
            sys.stderr.write(__doc__)
            return 2
        overrides = _overrides_from_infile(argv[2])
    elif "=" in "".join(argv[1:]):
        overrides = {}
        for tok in argv[1:]:
            k, _, v = tok.partition("=")
            overrides[k.strip()] = v.strip()
    else:
        sys.stderr.write(__doc__)
        return 2
    patch(path, overrides)
    print("Patched %s: %s" % (path, ", ".join("%s=%s" % kv for kv in overrides.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
