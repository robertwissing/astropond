#!/usr/bin/env bash
# Get MUSIC, the multi-scale cosmological initial-conditions generator
# (Hahn & Abel 2011).
#   - MUSIC : https://bitbucket.org/ohahn/music
#
# Why it lives here: MUSIC generates nested-grid "zoom" cosmological ICs and
# ships output plugins for Pkdgrav/Gasoline and TIPSY (src/plugins/output_tipsy.cc),
# i.e. it produces ICs this testsuite can read directly - a generator alongside
# glassgen_python, which is why it sits in icgenerator/ rather than codes/.
#
# NOTE this is MUSIC *1*, the original. MUSIC2/monofonIC is a separate code at
#   https://github.com/cosmo-sims/MUSIC (and .../monofonic); it does NOT have
#   the tipsy plugin, so v1 is the one that matters here. Set MUSIC_URL to
#   point elsewhere if you want a fork.
#
# Idempotent: skips the clone if the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

MUSIC_URL="${MUSIC_URL:-https://bitbucket.org/ohahn/music.git}"
MUSIC_REF="${MUSIC_REF:-}"          # optional tag/branch/commit to pin

if [ -d MUSIC ]; then
    echo "MUSIC source already present in $here/MUSIC (skipping clone)"
else
    if [ -n "$MUSIC_REF" ]; then
        git clone --branch "$MUSIC_REF" "$MUSIC_URL" MUSIC
    else
        git clone --depth 1 "$MUSIC_URL" MUSIC
    fi
    echo "Cloned MUSIC into $here/MUSIC"
fi

echo "  HEAD: $(git -C MUSIC log --oneline -1)"
echo "Next: ./build_music.sh"
