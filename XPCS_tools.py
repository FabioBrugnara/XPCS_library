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

    Parameters
    ----------
        beamline: str
            Beamline name ('PETRA3' or 'ID10')
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
    

def set_expvar(X0_toset:int, Y0_toset:int, L_toset:float):
    '''
    Set the experimental variables for the data analysis.

    Parameters
    ----------
        X0: int
            X0 position of the beam center in pixels
        Y0: int
            Y0 position of the beam center in pixels
        L: float
            Distance from the sample to the detector in meters
    '''
    global X0, Y0, L
    X0, Y0, L = X0_toset, Y0_toset, L_toset

############################################
########## GENERALFUNCTIONS ################
############################################

def E2lambda(E):
    return 12.39842/E

def lambda2E(l):
    return 12.39842/l

def theta2Q(Ei, theta):
    return 4*np.pi*np.sin(np.deg2rad(theta)/2)/E2lambda(Ei)

def Q2theta(Ei, Q):
    return 2*np.rad2deg(np.arcsin(E2lambda(Ei)*Q/4/np.pi))


decorelation_f = lambda t, tau, beta, c, y0: c*np.exp(-(t/tau)**beta) + y0

#########################################################################################################################


#################################
########### MASK PLOTS ##########
#################################

def gen_plots4mask(e4m_data, itime, Ith_high=None, Ith_low=None, Imaxth_high=None, OF=None, e4m_mask=None, Qmask=None, mask_geom=None, Nfi=None, Nff=None, max_plots=False, wide_plots = False):
    '''
    Function that generates a number of different plots to create the mask! By default it generates the average flux per pixel map and histogram.
    
    Parameters
    ----------
        e4m_data: sparse.csr_matrix
            Sparse matrix of the e4m detector data
        itime: float
            Integration time of the e4m detector
        Ith_high: float
            Threshold (above) for the mean photon flux of the pixels [ph/s/px]
        Ith_low: float
            Threshold (below) for the mean photon flux of the pixels [ph/s/px]
        Imaxth_high: float
            Maximum number of counts per pixel treshold [ph/px]
        OF: np.array
            Overflow mask of the e4m detector
        e4m_mask: np.array
            Mask of the e4m detector lines (slightly wider than the overflow lines, as pixels on the adges are not reliable)
        Qmask: np.array
            Q mask of the e4m detector
        mask_geom: list of dicts
            List of geometries to mask.
        Nfi: int
            First frame to consider
        Nff: int
            Last frame to consider
        max_plots: bool
            If True, plot the maximum counts per pixel map and histogram.
        wide_plots: bool
            If True, plot the wide histogram of the mean flux per pixel and maximum counts per pixel (if max_plots is True).
    '''

    # CHECK TYPE OF e4m_data
    if sparse.issparse(e4m_data): issparse = True
    else: issparse = False
    
    # LOAD DATA in Nfi:Nff
    e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK
    if OF is       None: OF =       np.zeros(Npx, dtype=bool)
    if e4m_mask is None: e4m_mask = np.ones(Npx, dtype=bool)
    if Qmask    is None: Qmask =    np.ones(Npx, dtype=bool)
    mask = ~OF*e4m_mask*Qmask

    # COMPUTE THE MEAN FLUX PER PX [ph/s/px]
    I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])
    I_mean[~mask] = of_value4plot # set the mask to the of_value4plot value

    # COMPUTE THE MAXIMUM COUNTS PER PX [ph/px] (only if needed)
    if (Imaxth_high is not None) or max_plots:
        I_max = e4m_data.max(axis=0).toarray()
        I_max[~mask] = of_value4plot # set the mask to the of_value4plot value

    # PRINT INFORMATIONS    
    print('################################################################################')
    if issparse: print('Maximum count in the hull run ->', e4m_data.data.max())
    else: print('Maximum value in the hull run ->', e4m_data.max())
    if Ith_high is not None: print('# of pixels above Ith_high treshold -> ', I_mean[mask][I_mean[mask]>Ith_high].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]>Ith_high].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Ith_low is not None: print('# of pixels below Ith_low treshold -> ',   I_mean[mask][I_mean[mask]<Ith_low].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]<Ith_low].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Imaxth_high is not None: print('# of pixels above Imaxth_high treshold -> ', I_max[mask][I_max[mask]>Imaxth_high].shape[0], 'pixels (of', I_max.shape[0], '=>', round(I_max[mask][I_max[mask]>Imaxth_high].shape[0]/I_max[mask].shape[0]*100, 2), '%)')
    print('################################################################################\n')


    ########## MEAN FLUX PER PX FIGURE #########################################################################
    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    ax5 = plt.subplot(413)

    # MEAN FLUX PER PX IMAGE
    im = ax4.imshow(I_mean.reshape(Nx, Ny), vmin=Ith_low, vmax=Ith_high, origin='lower')

    # add the colorbar and labels
    plt.colorbar(im, ax=ax4)
    ax4.set_title('Mean flux per px [ph/s/px]')
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')

    # plot the beam center
    ax4.plot(Y0, X0, 'ro', markersize=10)

    # plot the mask geometry (mask_geom)
    if mask_geom is not None:
        for obj in mask_geom:
            if obj['geom'] == 'Circle':
                ax4.add_artist(plt.Circle((obj['Cy'], obj['Cx']), obj['r'], color='r', fill=False))
            elif obj['geom'] == 'Rectangle':
                ax4.add_artist(plt.Rectangle((obj['y0'], obj['x0']), obj['yl'], obj['xl'], color='r', fill=False))

    # MEAN FLUX PER PX HISTOGRAM (ZOOM)
    if (Ith_high is not None) and (Ith_low is not None): ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, Ith_high*1.5),       label='(zoom)')
    elif (Ith_high is not None) and (Ith_low is None):   ax5.hist(I_mean[mask], bins=200, range=(0, Ith_high*1.5),                label='(zoom)')
    elif (Ith_high is None) and (Ith_low is not None):   ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, I_mean[mask].max()), label='(zoom)')
    else:                                                ax5.hist(I_mean[mask], bins=200,                                         label='(full range)')

    # plot the Ith_high and Ith_low limits
    if Ith_high is not None: ax5.axvline(Ith_high, color='r', label='Ith_high')
    if Ith_low is not None:  ax5.axvline(Ith_low,  color='g', label='Ith_low')

    # add the labels and legend
    ax5.set_yscale('log')
    ax5.set_xlabel('Mean flux per px [ph/s/px]')
    ax5.legend()

    # MEAN FLUX PER PX HISTOGRAM (FULL RANGE)
    if wide_plots:
        ax6 = plt.subplot(414)
        ax6.hist(I_mean[mask], bins=200, label='(full range)')
        if Ith_high is not None: ax6.axvline(Ith_high, color='r', label='Ith_high')
        if Ith_low  is not None: ax6.axvline(Ith_low,  color='g', label='Ith_low')
        ax6.set_yscale('log')
        ax6.set_xlabel('Mean flux per px[ph/s/px]')
        ax6.legend()

    plt.tight_layout()
    plt.show()

    ########## MAXIMUM COUNTS PER PX FIGURE #########################################################################
    if max_plots:
        plt.figure(figsize=(8,13))
        ax4 = plt.subplot(211)
        ax5 = plt.subplot(413)

        # MAX COUNTS PER PX IMAGE
        im = ax4.imshow(I_max.reshape(Nx, Ny), vmin=0, vmax=Imaxth_high, origin='lower')
        plt.colorbar(im, ax=ax4)
        ax4.set_title('Max counts per px [ph/px]')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # MAX COUNTS PER PX HISTOGRAM (ZOOM)
        if Imaxth_high is not None: 
            ax5.hist(I_max[mask], bins=100, label='(zoom)', range=(0, Imaxth_high*1.5))
            ax5.axvline(Imaxth_high, color='r', label='Imaxth_high')
        else:
            ax5.hist(I_max[mask], bins=100, label='(full range)')

        # add labels and legend
        ax5.set_yscale('log')
        ax5.set_xlabel('Max counts per px [ph/px]')
        ax5.legend()

        # MAX COUNTS PER PX HISTOGRAM (FULL RANGE)
        if wide_plots:
            ax6 = plt.subplot(414)
            ax6.hist(I_max[mask], bins=200, label='(full range)')
            ax6.set_yscale('log')
            ax6.set_xlabel('Max counts per px [ph/px]')
            ax6.legend()

        plt.tight_layout()
        plt.show()

    
#################################
########### MASK GEN ############
#################################

def gen_mask(e4m_data, itime, OF=None, e4m_mask=None, Qmask=None, mask_geom=None, Ith_high=None, Ith_low=None, Imaxth_high=None, Nfi=None, Nff=None, hist_plots=False):
    '''
    Generate a mask for the e4m detector from various options. The function plot the so-obtained mask, and also return some histograms to look at the results (if hist_plots is True).

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    OF: np.array
        Overflow mask of the e4m detector
    e4m_mask: np.array
        Mask of the e4m detector lines (slightly wider than the overflow lines, as pixels on the adges are not reliable)
    Qmask: np.array
        Q mask of the e4m detector
    mask_geom: list of dicts
        List of geometries to mask (in dictionary form). The supported objects are:
        - Circle: {'geom': 'Circle', 'Cx': x0, 'Cy': y0, 'r': r, 'inside': True/False}
        - Rectangle: {'geom': 'Rectangle', 'x0': x0, 'y0': y0, 'xl': xl, 'yl': yl, 'inside': True/False}

            Example:
            mask_geom = [   {'geom': 'Circle', 'Cx': 100, 'Cy': 100, 'r': 10, 'inside': True}, {'geom': 'Rectangle', 'x0': 50, 'y0': 50, 'xl': 20, 'yl': 10, 'inside': False}]
    Ith_high: float
        Threshold (above) for the mean photon flux of the pixels
    Ith_low: float
        Threshold (below) for the mean photon flux of the pixels
    Imaxth_high: float
        Maximum number of counts per pixel treshold
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider  
    hist_plots: bool
        If True, plot the histograms of the mean flux per pixel and maximum counts per pixel.

    Returns
    -------
    mask: np.array
        Mask of the e4m detector
    '''

    # CHECK TYPE OF e4m_data
    if sparse.issparse(e4m_data): issparse = True
    else: issparse = False
    
    # LOAD DATA in Nfi:Nff
    e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK OF ONES
    mask = np.ones(e4m_data.shape[1], dtype=bool)
    
    # FILTER OVERFLOWS, E4M_MASK AND QMASK
    if OF is not None:       mask = ~OF * mask
    if e4m_mask is not None: mask = mask * e4m_mask
    if Qmask is not None:    mask = mask * Qmask

    # APPLAY GEOMETRIC MASKS
    if (mask_geom is not None) and (mask_geom!=[]):
        mask = mask.reshape(Nx, Ny)
        X, Y = np.mgrid[:Nx, :Ny]
        for obj in mask_geom:
            if obj['geom']=='Circle':
                if obj['inside']:
                    mask = mask * ((Y-obj['Cy'])**2 + (X-obj['Cx'])**2 <= obj['r']**2)
                else:
                    mask = mask * ((Y-obj['Cy'])**2 + (X-obj['Cx'])**2 > obj['r']**2)
            elif obj['geom']=='Rectangle':
                if obj['inside']:
                    mask = mask * ((Y>obj['y0']) & (Y<obj['y0']+obj['yl']) & (X>obj['x0']) & (X<obj['x0']+obj['xl']))
                else:
                    mask = mask * ((Y<obj['y0']) | (Y>obj['y0']+obj['yl']) | (X<obj['x0']) | (X>obj['x0']+obj['xl']))
        mask = mask.flatten()

    # FILTER USING THRESHOLDS
    if ((Ith_high is not None) or (Ith_low is not None)):
        I_mean = e4m_data.sum(axis=0)/(itime*(Nff-Nfi))
        if Ith_high is not None: mask = mask * (I_mean<=Ith_high)
        if Ith_low is not None : mask = mask * (I_mean>=Ith_low)
    if Imaxth_high!=None:
        if issparse: I_max = e4m_data.max(axis=0).toarray()
        else       : I_max = e4m_data.max(axis=0)
        mask = mask * (I_max<Imaxth_high)

    # PRINT PERCENTAGE OF MASKED PIXELS
    print('#################################################')
    print('Masked area = ', mask.sum()/Npx*100, '%')
    print('#################################################\n')

    # PLOT THE MASK
    plt.figure(figsize=(8,8))
    plt.imshow(mask.reshape((Nx, Ny)), origin='lower')
    plt.xlabel('Y [px]')
    plt.ylabel('X [px]')
    plt.tight_layout()
    plt.show()

    # PLOT THE HISTOGRAMS (if hist_plots is True)
    if hist_plots==True:
        plt.figure(figsize=(8,6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # Masked histogram of px flux
        if ((Ith_high is None) and (Ith_low is None)): I_mean = e4m_data.sum(axis=0)/(itime*(Nff-Nfi))
        ax1.set_title('Masked histogram of px flux')
        ax1.hist(I_mean[mask], bins=100)
        ax1.set_yscale('log')
        ax1.set_xlabel('Mean flux per px')

        # Maked histogram of max counts per px
        if Imaxth_high==None:
            if issparse: I_max = e4m_data.max(axis=0).toarray()
            else       : I_max = e4m_data.max(axis=0)
        ax2.set_title('Masked histogram of max counts per px')
        ax2.hist(I_max[mask].data, bins=30, label='no zero counts')
        ax2.set_yscale('log')
        ax2.set_xlabel('Max counts per px')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return mask


#################################
########## Q MASK GEN ###########
#################################

def gen_Qmask(Ei, theta, Q, dq, Qmap_plot=False):
    '''
    Generate the Q masks for the given Q values at the working angle. The function also plot the Qmap for the given energy and angle (if Qmap_plot is True).

    Parameters
    ----------
    Ei: float
        Energy of the beam in keV
    theta: float
        Working angle in degrees
    Q: float or list of floats
        Q value(s) to mask in [1/A]
    dq: float or list of floats
        Q width(s) to mask in [1/A]
    Qmap_plot: bool
        If True, plot the Qmap for the given energy and angle

    Returns
    -------
    Qmask: np.array or dict of np.array
        Q mask(s) of the e4m detector
    '''

    # GET THE X-Y MAPS
    X, Y = np.mgrid[:Nx, :Ny]

    # COMPUTE THE THETA MAP FOR THE GIVEN DETECTOR POSITION AND ENERGY
    if beamline=='ID10':
        dY0 = L*np.tan(np.deg2rad(theta))
        dY0_map =np.sqrt(((X-X0)*lxp)**2+(dY0-(Y-Y0)*lyp)**2)
        theta_map = np.arctan(dY0_map/L)
    elif beamline=='PETRA3':
        dX0 = L*np.tan(np.deg2rad(theta))
        dX0_map =np.sqrt(((dX0-(X-X0)*lxp))**2+((Y-Y0)*lyp)**2)
        theta_map = np.arctan(dX0_map/L)

    # GET THE Q MAP
    Q_map = theta2Q(Ei, np.rad2deg(theta_map))

    # GET THE Q REGION
    # case of a single Q value
    if (type(Q) == float) or (type(Q) == int) or (type(Q) == np.float64) or (type(Q) == np.float32):
        Qmask = (np.abs(Q_map-Q)<dq).flatten()
    # case of a list of Q values
    else:
        Qmask = {}
        for i in range(len(Q)):
            if (type(dq) == float) or (type(dq) == int) or (type(dq) == np.float64) or (type(dq) == np.float32):
                Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq).flatten()
            else:
                Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq[i]).flatten()
    
    ########## QMASK IMAGE #########################################################################
    plt.figure(figsize=(8,8))

    # PLOT THE Q MASK
    # case of a single Q value
    if (type(Q) == float) or (type(Q) == int) or (type(Q) == np.float64) or (type(Q) == np.float32):    
        plt.imshow(Qmask.reshape((Nx, Ny)), cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)
        plt.scatter([],[], color=plt.cm.viridis(1.), label=str(Q)+'$\\AA^{-1}$')
    # case of a list of Q values
    else:
        Qmask2plot = 0
        s = 1/len(Q)
        for i, q in enumerate(Qmask.keys()):
            Qmask2plot += Qmask[q].reshape((Nx, Ny))*s*(i+1)
            plt.scatter([],[], color=plt.cm.viridis(s*(i+1)), label=str(Q[i])+'$\\AA^{-1}$')
        plt.imshow(Qmask2plot, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)

    # add labels and legend
    plt.xlabel('Y [px]')
    plt.ylabel('X [px]')
    plt.legend()

    plt.tight_layout()
    plt.show()


    ########## QMAP IMAGE (if Qmap_plot=True) #########################################################################
    if Qmap_plot:
        plt.figure(figsize=(8,8))
        plt.imshow(Q_map, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.xlabel('Y [px]')
        plt.ylabel('X [px]')
        plt.title('Q [$\\AA^{-1}$]')

        plt.tight_layout()
        plt.show()

    return Qmask



############################
######## GET It ############
############################

def get_It(e4m_data, itime, mask=None, Nfi=None, Nff=None, Lbin=None, Nstep=None):
    '''
    Compute the average frame intensity [ph/px/s] vector from the e4m_data, properly masked with the mask. 
    
    Parameters
    ----------
    e4m_data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    Nstep: int
        Step for the frames

    Returns
    -------
    t_Idt: np.array
        Time array for the It vector
    It: np.array
        It vector
    '''

    # CREATE EMPTY MASK IF mask=None
    if mask is None: mask = np.ones(e4m_data.shape[1], dtype=bool)

    # DEFAULT VALUES FOR Nfi, Nff, Lbin and Nstep
    if Nfi is None: Nfi = 0
    if Nff is None: Nff = e4m_data.shape[0]
    if Lbin is None: Lbin = 1
    if Nstep is None: Nstep = 1

    # GET THE CORRECT INDEXES FROM Nfi, Nff, Lbin and Nstep
    idx = [i for i in range(Nff-Nfi) if i % Nstep < Lbin][:((Nff-Nfi)//Nstep-1)*Nstep]
    idx = Nfi + np.array(idx)
    
    # COMPUTE It (masked)
    It = e4m_data[idx][:,mask].sum(axis=1)/mask.sum()

    # BIN It (if Lbin > 1)
    if Lbin != 1: 
        It = It[:(It.size//Lbin)*Lbin].reshape(-1, Lbin).sum(axis=1) / Lbin
    
    # NORMALIZE It
    It /= itime

    # GET THE TIME VECTOR    
    t_Idt = np.arange(Nfi*itime, Nff*itime, itime*Nstep)

    return t_Idt, It


#################################
######### COMUPTE G2t ###########
#################################

def get_G2t(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None, MKL_library=True, NumExpr_library=True):
    '''
    Compute the G2t matrix from the e4m, properly masked with the matrix mask.

    Parameters
    ----------
    e4m_data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    MKL_library: boolean
        If True, use the MKL library for the matrix multiplication
    NumExpr_library: boolean
        If True, use the NumExpr library for the normalization
    
    Returns
    -------
    G2t: np.array
        G2t matrix
    '''

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # BIN DATA
    if Lbin != 1:
        t0 = time.time()
        if MKL_library:
            print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
            Itp = (Itp[:Itp.shape[0]//Lbin*Lbin])
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = dot_product_mkl(BIN_matrix, Itp, dense=False, cast=True)
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    
        else:
            print('Binning frames (Lbin = '+str(Lbin)+') ...')
            Itp = Itp[:Itp.shape[0]//Lbin*Lbin]
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = BIN_matrix@Itp
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    if isinstance(Itp, sparse.sparray):
        print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
        print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3,3), 'GB')
    else:
        print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3,3), 'GB')

    #  MASK DATA
    t0 = time.time()
    if mask is not None:
        print('Masking data ...')
        Itp = Itp[:,mask]
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    if isinstance(Itp, sparse.sparray):
        print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
        print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3,3), 'GB')
    else:
        print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3,3), 'GB')

    # Compute G2t
    t0 = time.time()
    if MKL_library:
        print('Computing G2t (using MKL library)...')
        #G2t = np.zeros((Itp.shape[0], Itp.shape[0]), dtype=np.float64)
        G2t = gram_matrix_mkl(Itp, dense=True, transpose=True, cast=True)
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(G2t.shape[0])+' X '+str(G2t.shape[1])+' squared matrix')
        print('\t | memory usage (np.array @ '+str(G2t.dtype)+') =', round(G2t.nbytes/1024**3,3), 'GB')

    else:
        print('Computing G2t (and converting the sparse product to np.array)...')
        if isinstance(Itp, sparse.sparray): G2t = (Itp@Itp.T).toarray()
        else: G2t = Itp@Itp.T
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(G2t.shape[0])+' X '+str(G2t.shape[1])+' squared matrix')
        print('\t | memory usage (np.array @ '+str(G2t.dtype)+') =', round(G2t.nbytes/1024**3,3), 'GB')
        
    # Normalize G2t
    t0 = time.time()
    if NumExpr_library: print('Normalizing G2t (using NumExpr library)...')
    else: print('Normalizing G2t...')
    It = Itp.sum(axis=1, dtype=np.float64)
    np.divide(np.sqrt(Itp.shape[1]), It, where=It>0, out=It)
    if NumExpr_library:
        Itr = It[:, None] # q[:, None] -> q.reshape(N, 1)
        Itc = It[None, :] # q[None, :] -> q.reshape(1, N)
        ne.evaluate('G2t*Itr*Itc', out=G2t)
    else:
        np.multiply(G2t, It[:,None], out=G2t) 
        np.multiply(G2t, It[None,:], out=G2t) 

    # Remove diagonal and fill no counts frames
    G2t[G2t.diagonal()==0, :] = 1
    G2t[:, G2t.diagonal()==0] = 1
    np.fill_diagonal(G2t, 1)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')
    return G2t



##########################################
######### COMUPTE G2t bunnched ###########
##########################################

def get_G2t_bybunch(e4m_data, Nbunch, mask=None, Nfi=None, Nff=None, Lbin=None, MKL_library=True, NumExpr_library=True):
    '''
    Compute the G2t matrix from the e4m, bunching the frames in Nbunch bunches, thus averaging the G2t matrix over the bunches. 

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    Nbunch: int
        Number of bunches to consider
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    MKL_library: boolean
        If True, use the MKL library for the matrix multiplication
    NumExpr_library: boolean
        If True, use the NumExpr library for the normalization

    Returns
    -------
    G2t: np.array
        G2t matrix
    '''

    # DEFAULT VALUES FOR Nfi, Nff, Lbin
    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    # GET BUNCHES LENGHT [fms]
    Lbunch = (Nff-Nfi)//Nbunch

    # PREPARE THE G2t MATRIX
    G2t = np.zeros((Lbunch//Lbin, Lbunch//Lbin), dtype=np.float64)
    
    # COMPUTE G2t FOR EACH BUNCH
    for i in range(Nbunch):
        print('Computing G2t for bunch', i+1, '(Nfi =', Nfi+i*Lbunch, ', Nff =', Nfi+(i+1)*Lbunch, ') ...')
        G2t += get_G2t(e4m_data, mask, Nfi=Nfi+i*Lbunch, Nff=Nfi+(i+1)*Lbunch, Lbin=Lbin, MKL_library=MKL_library, NumExpr_library=NumExpr_library)
        print('Done!\n')

    return G2t/Nbunch



##############################
######### GET g2 #############
##############################

def get_g2(dt, G2t, cython=True):
    '''
    Compute the g2 from the G2t matrix.

    Parameters
    ----------
    dt: float
        Time step between frames
    G2t: np.array
        G2t matrix
    cython: boolean
        If True, use the cython code to compute the g2

    Returns
    -------
    t: np.array
        Time array
    g2: np.array
        g2 array
    '''
    
    t0 = time.time()
    if cython:
        print('Computing g2 (using cython code)...')
        g2 = mean_trace_float64(G2t)
    else:
        print('Computing g2...')
        g2 = np.array([G2t.diagonal(i).mean() for i in range(1,G2t.shape[0])])
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')

    return np.arange(len(g2))*dt, g2



##############################
######### GET g2 multitau ####
##############################

def get_g2_mt(dt, g2):
    '''
    Compute the multitau g2 from the g2 array.

    Parameters
    ----------
    dt: float
        Time step between frames
    g2: np.array
        g2 array    

    Returns
    -------
    t_multit: np.array
        Time array for the multitau g2
    g2_multit: np.array
        Multitau g2 array
    '''

    t = np.arange(len(g2))*dt

    g2_multit = []
    t_multit = []

    for i in range(int(np.log10(len(g2)))):
        g2_multit.append(g2[10**i:10**(i+1)].reshape(-1,10**(i)).mean(axis=1))
        t_multit.append(t[10**i:10**(i+1)].reshape(-1,10**(i)).mean(axis=1))

    i +=1
    out = g2[10**i:10**(i+1)].shape[0] % 10**i

    g2_multit.append(g2[10**i:10**(i+1)][:-out].reshape(-1,10**(i)).mean(axis=1))
    t_multit.append(t[10**i:10**(i+1)][:-out].reshape(-1,10**(i)).mean(axis=1))

    g2_multit = np.hstack(g2_multit)
    t_multit = np.hstack(t_multit)

    return t_multit, g2_multit



###########################
##### PLOT X/Y PROFILE ####
###########################

def plot_XYprofile(e4m_data, itime, ax='Y', mask=None, Nfi=None, Nff=None):
    '''
    Plot the X or Y profiles of the e4m detector.

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    ax: str
        Axis to plot ('X' or 'Y')
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    '''

    # DEFAULT VALUES FOR Nfi, Nff
    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]

    # COMPUTE It
    It = (e4m_data[Nfi:Nff].sum(axis=0)/((Nff-Nfi)*itime)).reshape(Nx, Ny)

    # APPLY MASK
    if mask is not None: It[~mask.reshape(Nx, Ny)] = 0

    # PLOT
    plt.figure(figsize=(8,5))
    if (ax=='Y') or (ax=='y'):
        plt.plot(It.sum(axis=0)/mask.reshape(Nx, Ny).sum(axis=0))
        plt.xlabel('Y [px]')
    if (ax=='X') or (ax=='x'):
        plt.plot(It.sum(axis=1)/mask.reshape(Nx, Ny).sum(axis=1))
        plt.xlabel('X [px]')
    plt.ylabel('Mean flux per px [ph/s/px]')
    
    plt.tight_layout()
    plt.show()



######################
###### PLOT G2T ######
######################

def plot_G2t(G2t, vmin, vmax, itime=None, t1=None, t2=None, x1=None, x2=None, sigma_filter=None, full=False):
    ''''
    Plot the G2t matrix.
    
    Parameters
    ----------
    G2t: np.array
        G2t matrix
    vmin: float
        Minimum value for the color scale
    vmax: float
        Maximum value for the color scale
    itime: float
        Integration time of the e4m detector
    t1: float
        First time to consider (in [s] if itime is provided, otherwise in [frames])
    t2: float
        Last time to consider (in [s] if itime is provided, otherwise in [frames])
    x1: float
        If provided, shift the x axis to the given initial value
    x2: float
        If provided, shift the x axis to the given final value
    sigma_filter: float
        Sigma for the Gaussian filter (in [frames]) 
    full: boolean
        If True, plot the full G2t matrix mirroring the lower part
    '''

    # BRHAVIORS WHEN t1, t2 ARE NONE
    if t1 is None: t1 = 0
    if (t2 is None) and (itime is None): t2 = G2t.shape[0]
    elif (t2 is None) and (itime is not None): t2 = G2t.shape[0]*itime

    # BEHAVIOURS WHEN t2 IS BIGGER THAN THE G2t MATRIX
    if (itime is None) and (t2>G2t.shape[0]): t2 = G2t.shape[0]
    elif (itime is not None) and (t2>G2t.shape[0]*itime): t2 = G2t.shape[0]*itime

    # BEHAVIOURS WHEN x1, x2 ARE NONE
    if (x1 is None) and (x2 is None): x1, x2 = t1, t2
    elif x1 is None: x1 = 0
    elif x2 is None: x2 = G2t.shape[1]

    # CUT THE G2t MATRIX
    if itime is None: G2t = G2t[t1:t2, x1:x2]
    else: G2t = G2t[int(t1//itime):int(t2//itime), int(x1//itime):int(x2//itime)]

    # APPLY GAUSSIAN FILTER (if sigma_filter is not None)
    if sigma_filter is not None:
        # default values for the filter
        truncate = 4
        radius = 2*round(truncate*sigma_filter) + 1 + truncate

        # enlarge the matrix above the diagonal
        for i in range(1, int(radius)+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = G2t.diagonal(offset=i)

        # apply the filter
        G2t = gaussian_filter(G2t, sigma=sigma_filter, mode='nearest', truncate=4)

        # remove the enlarged part
        for i in range(1, int(radius)*4+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = 0

    # ADD THE MIRRORING (if full==True)
    if full==True:
        G2t += G2t.T


    # PLOT
    plt.figure(figsize=(8,8))
    plt.imshow(G2t, vmin=vmin, vmax=vmax, origin='lower')

    # add ticks
    plt.yticks(np.round(np.linspace(0, G2t.shape[0], 6)).astype(int), np.round(np.linspace(x1, x2, 6)).astype(int))
    plt.xticks(np.round(np.linspace(0, G2t.shape[1], 6)).astype(int), np.round(np.linspace(t1, t2, 6)).astype(int))

    # add labels and colorbar
    plt.xlabel('Time [s]')
    plt.ylabel('Time [s]')
    plt.colorbar()

    plt.tight_layout()
    plt.show()