#!/usr/bin/env bash
# Get the public Gasoline SPH code (code=0 in runtest.sh).
#   - gasoline-public : N-BodyShop/gasoline
# Idempotent: skips the clone if the source tree is already present.
#
# MHD NOTE: the suite's reference binary is `gasolinemhd` from a private
#   gasoline_claude tree (the MHD variant). The public N-BodyShop release is
#   hydro+gravity only; MHD tests need the private tree until the MHD scheme is
#   public. Override GASOLINE_URL to point at the private/fork repo if available.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

GASOLINE_URL="${GASOLINE_URL:-https://github.com/N-BodyShop/gasoline.git}"

if [ -d gasoline-public ]; then
    echo "Gasoline source already present in $here/gasoline-public (skipping clone)"
else
    git clone "$GASOLINE_URL" gasoline-public
    echo "Cloned Gasoline into $here/gasoline-public"
fi
