"""
XPCStools
=========
A Python package for X-ray Photon Correlation Spectroscopy (XPCS) data analysis.
"""

from .config import config

from .det_analysis import (
    get_It,
    bin_Itp,
)

from .matrix_comp import (

    get_G2t,
    get_g2,
    plot_G2t,
)

from .multitau import (
    get_G2tmt_4sparse,
    print_Nf_choices,
    plot_G2tmt,
    get_g2mt,
    get_g2mt_cut,
    cut_G2tmt,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "config",
    # Detector Analysis
    "get_It",
    "bin_Itp",
    # Matrix Computation & Standard Correlation
    "get_G2t",
    "get_g2",
    "plot_G2t",
    # Multi-tau Analysis
    "get_G2tmt_4sparse",
    "print_Nf_choices",
    "plot_G2tmt",
    "get_g2mt",
    "get_g2mt_cut",
    "cut_G2tmt",
]