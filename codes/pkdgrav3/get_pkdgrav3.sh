#!/usr/bin/env bash
# Get the public PKDGRAV3 code (code=8 in runtest.sh).
#   - pkdgrav3 : dpotter/pkdgrav3 (Potter, Stadel & Teyssier 2017)
# Clones with submodules (mdl2 etc. are submodules of the CMake build).
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

PKDGRAV3_URL="${PKDGRAV3_URL:-https://bitbucket.org/dpotter/pkdgrav3.git}"

if [ -d pkdgrav3 ]; then
    echo "PKDGRAV3 source already present in $here/pkdgrav3 (skipping clone)"
else
    git clone --recurse-submodules "$PKDGRAV3_URL" pkdgrav3
    echo "Cloned PKDGRAV3 into $here/pkdgrav3"
fi
