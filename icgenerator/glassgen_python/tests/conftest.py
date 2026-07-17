import os
import sys

import numpy as np
if not hasattr(np, "trapz"):        # numpy>=2.0 renamed trapz -> trapezoid (removed in 2.4)
    np.trapz = np.trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
