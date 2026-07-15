#!/usr/bin/env bash
# Get the public OpenGadget3 code (code=9 in runtest.sh).
#   - opengadget3 : MAGNETICUM/Hydro-OpenGadget3 (Dolag et al. TreeSPH + SPMHD;
#     induction-equation magnetic fields with Dedner divB cleaning + artificial
#     magnetic dissipation -- the Gadget-family SPH sibling of the suite's GIZMO).
# Public, anonymously cloneable from gitlab.lrz.de. Idempotent: skips the clone if
# the source tree is already present.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

OPENGADGET3_URL="${OPENGADGET3_URL:-https://gitlab.lrz.de/MAGNETICUM/Hydro-OpenGadget3.git}"

if [ -d opengadget3 ]; then
    echo "OpenGadget3 source already present in $here/opengadget3 (skipping clone)"
else
    git clone "$OPENGADGET3_URL" opengadget3
    echo "Cloned OpenGadget3 into $here/opengadget3"
fi
