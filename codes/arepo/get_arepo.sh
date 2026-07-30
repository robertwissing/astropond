#!/usr/bin/env bash
# Get the public AREPO moving-mesh code (code=3 in runtest.sh and the optional
# sph_interp Voronoi backend).
#
# Public repo: https://gitlab.mpcdf.mpg.de/vrs/arepo (Springel et al.).
# The upstream checkout lives in AREPO/ so this directory can retain the suite's
# tracked get/build scripts and physics configs. Idempotent: skip when present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

AREPO_URL="${AREPO_URL:-https://gitlab.mpcdf.mpg.de/vrs/arepo.git}"

if [ -e AREPO/Makefile ] && [ -e AREPO/src ]; then
    echo "AREPO source already present in $here/AREPO (skipping clone)"
else
    git clone "$AREPO_URL" AREPO
    echo "Cloned AREPO into $here/AREPO"
fi
