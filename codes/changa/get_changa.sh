#!/usr/bin/env bash
# Get the public ChaNGa code and the utilities it needs (code=1 in runtest.sh).
#   - changa-public : N-BodyShop/ChaNGa (the simulation code)
#   - utility       : N-BodyShop/utility (provides STRUCT_DIR for the build)
#   - charm         : charmplusplus/charm (Charm++ runtime, built by build_changa.sh)
# Idempotent: skips any clone whose directory already exists. To build the MHD fork
# instead, set CHANGA_URL=https://github.com/robertwissing/changa_uw and check out
# branch MHD_tp_replication in build_changa.sh (see its MHD NOTE).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

CHANGA_URL="${CHANGA_URL:-https://github.com/N-BodyShop/changa.git}"
UTIL_URL="${UTIL_URL:-https://github.com/N-BodyShop/utility.git}"
CHARM_URL="${CHARM_URL:-https://github.com/charmplusplus/charm.git}"

echo "Getting public ChaNGa + required utilities into $here"
[ -d changa-public ] || git clone "$CHANGA_URL" changa-public
[ -d utility ]       || git clone "$UTIL_URL"
[ -d charm ]         || git clone "$CHARM_URL"
