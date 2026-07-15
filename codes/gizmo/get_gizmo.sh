#!/usr/bin/env bash
# Get the public GIZMO code (code=5 in runtest.sh).
#   - gizmo : phopkins/gizmo-public (Hopkins 2015 meshless finite-mass / SPH / MHD)
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

GIZMO_URL="${GIZMO_URL:-https://bitbucket.org/phopkins/gizmo-public.git}"

if [ -d gizmo ]; then
    echo "GIZMO source already present in $here/gizmo (skipping clone)"
else
    git clone "$GIZMO_URL" gizmo
    echo "Cloned GIZMO into $here/gizmo"
fi
