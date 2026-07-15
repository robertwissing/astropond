#!/usr/bin/env bash
# Get the public Phantom SPH code (code=6 in runtest.sh).
#   - phantom : danieljprice/phantom
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

PHANTOM_URL="${PHANTOM_URL:-https://github.com/danieljprice/phantom.git}"

if [ -d phantom ]; then
    echo "Phantom source already present in $here/phantom (skipping clone)"
else
    git clone "$PHANTOM_URL" phantom
    echo "Cloned Phantom into $here/phantom"
fi
