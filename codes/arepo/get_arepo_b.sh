#!/usr/bin/env bash
# Get the public AREPO moving-mesh code.
#
# AREPO is already laid out with this directory AS the repo root (codes/arepo/),
# so this just clones it in place if missing (e.g. a fresh checkout of the
# testsuite). Idempotent: skips the clone if the source tree is already here.
#
# Public repo: https://gitlab.mpcdf.mpg.de/vrs/arepo  (Springel et al.)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

AREPO_URL="${AREPO_URL:-https://gitlab.mpcdf.mpg.de/vrs/arepo.git}"

if [ -e "$here/Makefile" ] && [ -e "$here/src" ]; then
    echo "AREPO source already present in $here (skipping clone)"
else
    # Clone into a temp dir then move contents up, since codes/arepo/ is the repo
    # root itself (not a subdir) and may already hold these get/build scripts.
    tmp="$(mktemp -d "$here/.arepo_clone.XXXXXX")"
    git clone "$AREPO_URL" "$tmp/arepo"
    shopt -s dotglob
    mv "$tmp/arepo"/* "$here/"
    shopt -u dotglob
    rmdir "$tmp/arepo" "$tmp"
    echo "Cloned AREPO into $here"
fi
