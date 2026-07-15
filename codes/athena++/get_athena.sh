#!/usr/bin/env bash
# Get the public Athena++ grid MHD code (code=7 in runtest.sh).
#   - athena : PrincetonUniversity/athena
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

ATHENA_URL="${ATHENA_URL:-https://github.com/PrincetonUniversity/athena.git}"

if [ -d athena ]; then
    echo "Athena++ source already present in $here/athena (skipping clone)"
else
    git clone "$ATHENA_URL" athena
    echo "Cloned Athena++ into $here/athena"
fi
