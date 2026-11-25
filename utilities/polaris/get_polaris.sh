#!/usr/bin/env bash
# Get POLARIS, the 3D dust/line/synchrotron radiative-transfer code
# (Reissl, Wolf & Brauer 2016; Reissl et al. 2019).
#   - POLARIS : polaris-MCRT/POLARIS
#
# Used by the synchrotron export pipeline: the testsuite
# writes an octree grid with setupfiles/IC_export_polaris.py and ray-traces it
# with CMD_SYNCHROTRON. POLARIS is a CONSUMER of our ICs, not a code under test,
# which is why it lives in utilities/ rather than codes/.
#
# Idempotent: skips the clone if the source tree is already present.
#
# NOTE: the clone is ~350 MB -- input/ alone is 72 MB of dust/gas reference data
#   (silicate + graphite refractive indices, line databases). Those files are
#   needed at runtime, so this is not a --depth 1 candidate for size; --depth 1
#   is still used to skip the history.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

POLARIS_URL="${POLARIS_URL:-https://github.com/polaris-MCRT/POLARIS.git}"
POLARIS_REF="${POLARIS_REF:-}"          # optional tag/branch/commit to pin

if [ -d POLARIS ]; then
    echo "POLARIS source already present in $here/POLARIS (skipping clone)"
else
    if [ -n "$POLARIS_REF" ]; then
        git clone --branch "$POLARIS_REF" "$POLARIS_URL" POLARIS
    else
        git clone --depth 1 "$POLARIS_URL" POLARIS
    fi
    echo "Cloned POLARIS into $here/POLARIS"
fi

echo "  HEAD: $(git -C POLARIS log --oneline -1)"
echo "Next: ./build_polaris.sh"
