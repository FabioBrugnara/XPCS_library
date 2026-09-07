"""
XPCStools
=========
A Python package for X-ray Photon Correlation Spectroscopy (XPCS) data analysis.
"""

from .config import config, set_config

from .det_analysis import (
    det_analysis,
    gen_mask,
    gen_Qmask,
)

from .matrix_comp import (
    get_It,
    bin_Itp,
    get_G2t,
    get_G2t_bybunch,
    get_g2,
    get_g2mt_fromling2,
    get_g2_mt,
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

from .utils import (
    E2lambda,
    lambda2E,
    theta2Q,
    Q2theta,
    decorrelation_f,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "config",
    "set_config",
    # Detector Analysis
    "det_analysis",
    "gen_mask",
    "gen_Qmask",
    "get_It",
    "bin_Itp",
    # Matrix Computation & Standard Correlation
    "get_G2t",
    "get_G2t_bybunch",
    "get_g2",
    "get_g2mt_fromling2",
    "get_g2_mt",
    "plot_G2t",
    # Multi-tau Analysis
    "get_G2tmt_4sparse",
    "print_Nf_choices",
    "plot_G2tmt",
    "get_g2mt",
    "get_g2mt_cut",
    "cut_G2tmt",
    # Utilities
    "E2lambda",
    "lambda2E",
    "theta2Q",
    "Q2theta",
    "decorrelation_f",
]