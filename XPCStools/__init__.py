"""
XPCStools
=========
A Python package for X-ray Photon Correlation Spectroscopy (XPCS) data analysis.
"""

from .config import (
    config,
    set_config
)

from .det_analysis import (
    get_Ip,
    plot_Ip,
    get_mask,
    get_Qmask,
    bin_det
)

from .matrix_comp import (
    get_It,
    get_Itp_bin,
    get_G2t,
    get_G2t_bybunch,
    get_g2,
    plot_G2t,
)

from .multitau import (
    get_G2tmt_4sparse,
    get_G2tmt_4sparse_wl,
    plot_G2tmt,
    get_g2mt,
    save_G2tmt,
    load_G2tmt,
    mt_corr
)

from .utils import (
    E2lambda,
    lambda2E,
    theta2Q,
    Q2theta,
    decorrelation_f,
)

from .Sq_utils import (
    get_Sq,
    plot_time_Sq,
    plot_temperature_Sq,
)

from .newtools import (
    get_It_fast,
    normalize_Itp,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "config",
    "set_config",
    # Detector Analysis
    "get_Ip",
    "plot_Ip",
    "get_mask",
    "get_Qmask",
    "bin_det",
    # Matrix Computation & Standard Correlation
    "get_It",
    "get_Itp_bin",
    "get_G2t",
    "get_G2t_bybunch",
    "get_g2",
    "plot_G2t",
    # Multi-tau Analysis
    "get_G2tmt_4sparse",
    "plot_G2tmt",
    "get_g2mt",
    "save_G2tmt",
    "load_G2tmt",
    "get_G2tmt_4sparse_wl",
     "mt_corr",
    # Utilities
    "E2lambda",
    "lambda2E",
    "theta2Q",
    "Q2theta",
    "decorrelation_f",
    # Sq_utils
    "get_Sq",
    "plot_Sq",
    "plot_time_Sq",
    "plot_temperature_Sq",
    # New Tools
    "get_It_fast",
    "normalize_Itp",
    
]