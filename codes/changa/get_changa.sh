#!/usr/bin/env bash
# Get the public ChaNGa code and the utilities it needs (code=1 in runtest.sh).
#   - changa-public : N-BodyShop/ChaNGa (the simulation code)
#   - utility       : N-BodyShop/utility (provides STRUCT_DIR for the build)
#   - charm         : charmplusplus/charm (Charm++ runtime; build one architecture
#                     before invoking the content-addressed ChaNGa resolver)
# Idempotent: skips any clone whose directory already exists. Fork capability
# mappings belong in CHANGA_BUILD_CONFIG; the resolver never changes branches.
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

cat <<'EOF'
Sources are present. If Charm++ has not been built yet, build the desired
architecture in codes/changa/charm first. Then run build_changa.sh; its default
profile uses changa-public, or set CHANGA_BUILD_CONFIG for a compatible fork.
EOF
