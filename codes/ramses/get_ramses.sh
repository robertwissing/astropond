#!/usr/bin/env bash
# Get the public RAMSES AMR code (code=4 in runtest.sh).
#   - ramses : ramses-organisation/ramses (formerly bitbucket rteyssie/ramses)
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

RAMSES_URL="${RAMSES_URL:-https://github.com/ramses-organisation/ramses.git}"

if [ -d ramses ]; then
    echo "RAMSES source already present in $here/ramses (skipping clone)"
else
    git clone "$RAMSES_URL" ramses
    echo "Cloned RAMSES into $here/ramses"
fi
