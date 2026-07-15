#!/usr/bin/env bash
# Get the public SPH-EXA code (code=2 in runtest.sh).
#   - sphexa : unibas-dmi-hpc/SPH-EXA (SPH-EXA mini-app)
# Idempotent: skips the clone if the source tree is already present. Clones with
# submodules (cstone / ryoanji are submodules of the CMake build).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

SPHEXA_URL="${SPHEXA_URL:-https://github.com/unibas-dmi-hpc/SPH-EXA.git}"
# NB the upstream default/active branch is `develop` -- it carries the UNIFIED
# `sphexa` mini-app (--init HDF5 / --prop std, binary at build/main/src/sphexa/)
# that runtest.sh code=2 drives. The stale `master` branch is the OLD per-test
# executable architecture (src/evrard, src/sedov; no --init/--prop) and does NOT
# work with this suite. (`magnetohydro` is the SPMHD branch for the MHD tests.)
SPHEXA_BRANCH="${SPHEXA_BRANCH:-develop}"

if [ -d sphexa ]; then
    echo "SPH-EXA source already present in $here/sphexa (skipping clone)"
else
    git clone --recurse-submodules -b "$SPHEXA_BRANCH" "$SPHEXA_URL" sphexa
    echo "Cloned SPH-EXA into $here/sphexa"
fi
