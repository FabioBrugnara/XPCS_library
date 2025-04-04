"""
My Library
==========

This is a sample Python library.

Author: Your Name
"""


### IMPORT SCIENTIFIC LIBRARIES ###
# standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# sparse library
from scipy import sparse

# Gaussian filter for G2t plotting
from scipy.ndimage import gaussian_filter

# fast matrix multiplication and other (mkl and numexpr)
from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl
import numexpr as ne

# C-implemented functions
from XPCScy_tools.XPCScy_tools import mean_trace_float64


### VARIABLES ###
of_value4plot = 2**32-1 # value for the overflow pixels in the plot
 
#############################################
##### SET BEAMLINE AND EXP VARIABLES #######
#############################################

def set_beamline(beamline_toset:str):
    '''
    Set the beamline parameters for the XPCS data analysis. The function load the correct varaibles (Nx, Ny, Npx, lxp, lyp) from the beamline tools.

    Parameters:
        test(float): Test value to check the function

    Returns:
        None: The function sets the global variables for the beamline parameters.
    '''

    global beamline, Nx, Ny, Npx, lxp, lyp

    if beamline_toset == 'PETRA3':
        import PETRA3_tools as PETRA
        beamline = 'PETRA3'
        Nx, Ny, Npx, lxp, lyp = PETRA.Nx, PETRA.Ny, PETRA.Npx, PETRA.lxp, PETRA.lyp
    
    elif beamline_toset == 'ID10':
        import ID10_tools as ID10
        beamline = 'ID10'
        Nx, Ny, Npx, lxp, lyp = ID10.Nx, ID10.Ny, ID10.Npx, ID10.lxp, ID10.lyp
        
    else:
        raise ValueError('Beamline not recognized!')
    