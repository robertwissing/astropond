#!/usr/bin/env python3
"""FDM static 3D Gaussian overdensity: Q and a_QP vs analytic.
The run is static; see IC_analysis_fdm.py for the measures.
Usage: python IC_analysis_gaussOverdens.py <run-dir> [<run-dir> ...] [--labels ...] [--save out]"""
from IC_analysis_fdm import static_main

if __name__ == "__main__":
    static_main("gaussOverdens", "FDM static 3D Gaussian overdensity: Q and a_QP vs analytic.")
