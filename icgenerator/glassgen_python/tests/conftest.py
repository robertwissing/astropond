import os
import sys

# NB no `np.trapz = np.trapezoid` monkeypatch here any more (CONSOLIDATION_PLAN
# D5, 2026-07-29). Mutating the global numpy module from a conftest made every
# np.trapz caller depend on this file having been imported first - test_kernels
# and test_relax_e2e were silently relying on it. Each now carries its own local
# `_trapezoid` alias instead.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
