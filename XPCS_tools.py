### IMPORT SCIENTIFIC LIBRARIES ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy import sparse
from scipy.ndimage import gaussian_filter

from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl
import numexpr as ne

from XPCScy_tools.XPCScy_tools import mean_trace_float64


### VARIABLES ###
of_value4plot = 2**32-1
 
#############################################
##### SET BEAMLINE AND EXP VARIABLES #######
#############################################

def set_beamline(beamline_toset):
    '''
    Set the beamline parameters for the data analysis.

    Args:
    beamline: str
        Beamline name
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
    

def set_expvar(X0_toset, Y0_toset, L_toset):
    '''
    Set the experimental variables for the data analysis.

    Args:
    X0: float
        X coordinate of the beam center
    Y0: float
        Y coordinate of the beam center
    L: float
        Sample to detector distance
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

def gen_plots4mask(OF, e4m_data, itime, Ith_high=None, Ith_low=None, Imaxth_high=None, e4m_mask=None, Qmask=None, mask_geom=None, Nfi=None, Nff=None, max_plots=False, counts_plot=False, wide_plots = False):
    '''
    Function that generates a number of different plots to create the mask! By default it generates the average flux per pixel map and histogram.
        
    Args:
    - OF: boolean array of the overflows
    - e4m_data: sparse matrix of the data
    - itime: integration time
    - Ith: threshold for the mask
    - mask_geom: list of tuples with the mask geometry
    - wide_plots: boolean to plot the full range of the histograms
    - max_plots: boolean to plot the maximum flux per pixel map and histogram
    - counts_plot: boolean to plot the counts histogram
    - vmax_max: maximum value for the maximum flux per pixel map
    '''

    if Nfi==None: Nfi = 0
    if (Nff==None) or (Nff>e4m_data.shape[0]): Nff = e4m_data.shape[0]
    if mask_geom==None: mask_geom = []
    if e4m_mask is None: e4m_mask = np.ones_like(OF)
    if Qmask is None: Qmask = np.ones_like(OF)

    if sparse.issparse(e4m_data): issparse = True
    else: issparse = False
    
    e4m_data = e4m_data[Nfi:Nff]
    mask = ~OF*e4m_mask*Qmask

    I_mean = e4m_data.sum(axis=0)/(itime*(Nff-Nfi))
    I_mean[~mask] = of_value4plot
    if (Imaxth_high is not None) or max_plots:
        I_max = e4m_data.max(axis=0).toarray()
        I_max[~mask] = of_value4plot
    
    print('################################################################################')
    if issparse: print('Maximum count in the hull run ->', e4m_data.data.max())
    else: print('Maximum value in the hull run ->', e4m_data.max())
    if Ith_high is not None: print('# of pixels above Ith_high treshold -> ', I_mean[mask][I_mean[mask]>Ith_high].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]>Ith_high].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Ith_low is not None: print('# of pixels below Ith_low treshold -> ',   I_mean[mask][I_mean[mask]<Ith_low].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]<Ith_low].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Imaxth_high is not None: print('# of pixels above Imaxth_high treshold -> ', I_max[mask][I_max[mask]>Imaxth_high].shape[0], 'pixels (of', I_max.shape[0], '=>', round(I_max[mask][I_max[mask]>Imaxth_high].shape[0]/I_max[mask].shape[0]*100, 2), '%)')
    print('################################################################################\n')

    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    ax5 = plt.subplot(413)

    ### AVERAGE ANALYSIS ###
    if Ith_low is not None: im = ax4.imshow(I_mean.reshape(Nx, Ny), vmin=Ith_low, vmax=Ith_high, origin='lower', label='average ph flux per px')
    else:                   im = ax4.imshow(I_mean.reshape(Nx, Ny),               vmax=Ith_high, origin='lower', label='average ph flux per px')
    plt.colorbar(im, ax=ax4)
    ax4.set_title('Average photon flux per px (not masked!)')
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')

    # plot the beam center
    ax4.plot(Y0, X0, 'ro', markersize=10)

    # plot the mask geometry
    for obj in mask_geom:
        if obj['geom'] == 'Circle':
            ax4.add_artist(plt.Circle((obj['Cy'], obj['Cx']), obj['r'], color='r', fill=False))
        elif obj['geom'] == 'Rectangle':
            ax4.add_artist(plt.Rectangle((obj['y0'], obj['x0']), obj['yl'], obj['xl'], color='r', fill=False))

    # HISTOGRAM OVER THE PIXELS
    if (Ith_high is not None) and (Ith_low is not None): ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, Ith_high*1.5),       label='$\\phi$ per px (zoom)')
    elif (Ith_high is not None) and (Ith_low is None):   ax5.hist(I_mean[mask], bins=200, range=(0, Ith_high*1.5),                label='$\\phi$ per px (zoom)')
    elif (Ith_high is None) and (Ith_low is not None):   ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, I_mean[mask].max()), label='$\\phi$ per px (zoom)')
    else:                                                ax5.hist(I_mean[mask], bins=200,                                         label='$\\phi$ per px')

    if Ith_high is not None: ax5.axvline(Ith_high, color='r', label='Ith_high')
    if Ith_low is not None:  ax5.axvline(Ith_low,  color='g', label='Ith_low')
    ax5.set_yscale('log')
    ax5.set_xlabel('Mean flux per px')
    ax5.legend()

    if wide_plots:
        ax6 = plt.subplot(414)
        ax6.hist(I_mean[mask], bins=200, label='$\\phi$ per px')
        if Ith_high is not None: ax6.axvline(Ith_high, color='r', label='$I_{th}$')
        if Ith_low  is not None: ax6.axvline(Ith_low,  color='g', label='$I_{th}$')
        ax6.set_yscale('log')
        ax6.set_xlabel('Mean flux per px')
        ax6.legend()

    plt.tight_layout()
    plt.show()

    if max_plots:
        plt.figure(figsize=(8,13))
        ax4 = plt.subplot(211)
        ax5 = plt.subplot(413)

        ### MAXIMUM ANALYSIS ###
        im = ax4.imshow(I_max.reshape(Nx, Ny), vmax=Imaxth_high, origin='lower')
        plt.colorbar(im, ax=ax4)
        ax4.set_title('Max # of photons per px (in the run)')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # HISTOGRAM OVER THE PIXELS
        if Imaxth_high is not None: 
            ax5.hist(I_max[mask], bins=100, label='zoom', range=(0, Imaxth_high*1.5))
            ax5.axvline(Imaxth_high, color='r', label='Imaxth_high')
        else:
            ax5.hist(I_max[mask], bins=100)
        ax5.set_yscale('log')
        ax5.set_xlabel('max counts per px')
        ax5.legend()

        if wide_plots:
            # HISTOGRAM OVER THE PIXELS
            ax6 = plt.subplot(414)
            ax6.hist(I_max[mask], bins=200)
            ax6.set_yscale('log')
            ax6.set_xlabel('max counts per px')
            ax6.legend()

        plt.tight_layout()
        plt.show()

    
    if counts_plot:
        plt.figure(figsize=(8,6))
        ax1 = plt.subplot(211)

        ### COUNTS HISTOGRAM OF ALL DATA ##
        ax1.set_title('Ph counts for all px and frames')
        ax1.hist(e4m_data[:, mask].data, bins=100, label = 'counts without zeros (zoom)', range=(0, Imaxth_high*2))
        if Imaxth_high is not None: ax1.axvline(Imaxth_high, color='r', label='Imaxth_high')
        ax1.set_yscale('log')
        ax1.set_xlabel('counts')
        ax1.legend()

        if wide_plots:
            ax2 = plt.subplot(212)
            ax2.hist(e4m_data[:,mask].data, bins=200, label = 'counts without zeros ')
            ax2.set_yscale('log')
            ax2.set_xlabel('counts')
            ax2.legend()

        plt.tight_layout()
        plt.show()



#################################
########### MASK GEN ############
#################################

def gen_mask(OF, e4m_data, itime, Ith_high=None, Ith_low=None, Imaxth_high=None,  e4m_mask=None, Qmask=None, mask_geom=None, Nfi=None, Nff=None, hist_plots=False):
    '''
    Generate a mask for the e4m detector from various options. Also return some histograms to look at the result.

    Args:
    OF: np.array
        Overflow mask of the e4m detector
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    e4m_mask: np.array
        Mask of the e4m detector lines (slightly wider than the overflow lines, as pixels on the adges are not reliable)
    Ith: float
        Threshold (above) for the mean photon flux of the pixels
    Imax: float
        Maximum number of counts per pixel treshold
    mask_geom: list of dicts
        List of geometries to mask.
    
    Returns:
    mask: np.array
        Mask of the e4m detector
    '''

    if Nfi is None: Nfi = 0
    if Nff is None: Nff = e4m_data.shape[0]
    if sparse.issparse(e4m_data): issparse = True
    else: issparse = False
    if (Nfi is not None) or (Nff is not None): e4m_data = e4m_data[Nfi:Nff]

    mask = np.ones(e4m_data.shape[1], dtype=bool)

    # apply geometry
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

    # remove overflows
    mask[OF] = False

    # filter using thresholds
    if ((Ith_high is not None) or (Ith_low is not None)):
        I_mean = e4m_data.sum(axis=0)/(itime*(Nff-Nfi))
        if Ith_high is not None: mask = mask * (I_mean<=Ith_high)
        if Ith_low is not None : mask = mask * (I_mean>=Ith_low)
    if Imaxth_high!=None:
        if issparse: I_max = e4m_data.max(axis=0).toarray()
        else       : I_max = e4m_data.max(axis=0)
        mask = mask * (I_max<Imaxth_high)

    # filter e4m detector lines
    if e4m_mask is not None: mask = mask * e4m_mask

    # multiply by Qmask
    if Qmask is not None: mask = mask * Qmask

    print('################################################################################')
    print('Mask area -> ', mask.sum()/Npx*100, '%')
    print('################################################################################\n')

    # plot the mask
    plt.figure(figsize=(8,8))
    plt.imshow(mask.reshape((Nx, Ny)), origin='lower')
    plt.xlabel('Y [px]')
    plt.ylabel('X [px]')
    plt.tight_layout()
    plt.show()

    # PLOT THE HISTOGRAMS
    if hist_plots==True:
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        # Masked counts histogram (all frames and px)
        ax1.set_title('Masked counts histogram (all frames and px)')
        if issparse:
            ax1.hist(e4m_data[:, mask].data, bins=10, label='no zero counts')
        ax1.set_yscale('log')
        ax1.set_xlabel('# of counts')
        ax1.legend

        # Masked histogram of px flux
        if ((Ith_high is None) and (Ith_low is None)): I_mean = e4m_data.sum(axis=0)/(itime*(Nff-Nfi))
        ax2.set_title('Masked histogram of px flux')
        ax2.hist(I_mean[mask], bins=100)
        ax2.set_yscale('log')
        ax2.set_xlabel('Mean flux per px')

        # Maked histogram of max counts per px
        if Imaxth_high==None:
            if issparse: I_max = e4m_data.max(axis=0).toarray()
            else       : I_max = e4m_data.max(axis=0)
        ax3.set_title('Masked histogram of max counts per px')
        plt.hist(I_max[mask].data, bins=30, label='no zero counts')
        ax3.set_yscale('log')
        ax3.set_xlabel('Max counts per px')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    return mask



#################################
########## Q MASK GEN ###########
#################################

def gen_Qmask(Ei, theta, Q, dq, OF=None, e4m_data=None, itime=None, Ith_high=None, mask=None, Qplot=False):
    '''
    Generate the Q masks for the given Q values at the working angle.

    Args:
    - theta (float): the working angle in degrees.
    - Q (float or list of floats): the Q values to mask.
    - dq (float or list of floats): the tolerance in Q values.
    - Qplot (bool): if True, plot the Q map.

    Returns:
    - Qmask (array or list of arrays): the Q mask(s).
    '''
    if beamline=='ID10':
        dY0 = L*np.tan(np.deg2rad(theta))
    elif beamline=='PETRA3':
        dX0 = L*np.tan(np.deg2rad(theta))
    X, Y = np.mgrid[:Nx, :Ny]
    if beamline=='ID10':
        dY0_map =np.sqrt(((X-X0)*lxp)**2+(dY0-(Y-Y0)*lyp)**2)
        theta_map = np.arctan(dY0_map/L)
    elif beamline=='PETRA3':
        dX0_map =np.sqrt(((dX0-(X-X0)*lxp))**2+((Y-Y0)*lyp)**2)
        theta_map = np.arctan(dX0_map/L)

    Q_map = theta2Q(Ei, np.rad2deg(theta_map))

    if (e4m_data is not None):
        if (itime is not None) : I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])
        else: I_mean = e4m_data.sum(axis=0)/e4m_data.shape[0]

        if (Ith_high is None):
            if OF is not None: Ith = I_mean[OF].mean()
            else: Ith_high = I_mean.mean()

        if OF is not None: I_mean[OF] = of_value4plot
        if mask is not None: I_mean[~mask] = of_value4plot


    if (type(Q) == float) or (type(Q) == int) or (type(Q) == np.float64) or (type(Q) == np.float32):
        Qmask = (np.abs(Q_map-Q)<dq).flatten()
    else:
        Qmask = {}
        for i in range(len(Q)):
            if (type(dq) == float) or (type(dq) == int) or (type(dq) == np.float64) or (type(dq) == np.float32):
                Qmask[i] = (np.abs(Q_map-Q[i])<dq).flatten()
            else:
                Qmask[i] = (np.abs(Q_map-Q[i])<dq[i]).flatten()
    
    plt.figure(figsize=(8,8))
    if (e4m_data is not None):
        plt.imshow(I_mean.reshape(Nx, Ny), vmax=Ith_high, origin='lower')
        alpha = 0.5
    else: alpha = 1

    if (type(Q) == float) or (type(Q) == int) or (type(Q) == np.float64) or (type(Q) == np.float32):    
        plt.imshow(Qmask.reshape((Nx, Ny)), cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=alpha)
        plt.scatter([],[], color=plt.cm.viridis(1.), label=str(Q)+'$\\AA^{-1}$')
        plt.xlabel('Y [px]')
        plt.ylabel('X [px]')
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        Qmask2plot = 0
        s = 1/len(Q)
        for i, q in enumerate(Qmask.keys()):
            Qmask2plot += Qmask[q].reshape((Nx, Ny))*s*(i+1)
            plt.scatter([],[], color=plt.cm.viridis(s*(i+1)), label=str(Q[i])+'$\\AA^{-1}$')
        plt.imshow(Qmask2plot, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=alpha)
        plt.xlabel('Y [px]')
        plt.ylabel('X [px]')
        plt.legend()

        plt.tight_layout()
        plt.show()

    if Qplot:
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
    Compute the It (average frame time intensity) vector from the e4m, properly masked with the matrix mask.

    Args:
    e4m_data: sparse.csc_matrix
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

    Returns:
    t_Idt: np.array
        Time array for the It
    It: np.array
        It vector
    '''

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if (Lbin is not None) and (Nstep is not None): raise ValueError('Lbin and Nstep cannot be used together! (will be implemented in the future)')
    if Lbin == None: Lbin = 1
    if Nstep == None: Nstep = 1
    Nff = (Nff-Nfi)//Lbin*Lbin+Nfi

    It = e4m_data[Nfi:Nff:Nstep,mask].sum(axis=1)/mask.sum()

    if Lbin != 1: It = It[:(It.size//Lbin)*Lbin].reshape(-1, Lbin).sum(axis=1) / (itime*Lbin)
    else: It /= itime
    
    t_Idt = np.arange(Nfi*itime, Nff*itime, itime*Lbin*Nstep)

    return t_Idt, It

def get_Imax(e4m_data, itime, mask=None, Nfi=None, Nff=None, Lbin=None, Nstep=None):

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if (Lbin is not None) and (Nstep is not None): raise ValueError('Lbin and Nstep cannot be used together! (will be implemented in the future)')
    if Lbin == None: Lbin = 1
    if Nstep == None: Nstep = 1
    Nff = (Nff-Nfi)//Lbin*Lbin+Nfi

    Imax = (e4m_data[Nfi:Nff:Nstep,mask].max(axis=1)).toarray()

    if Lbin != 1: Imax = Imax[:(Imax.size//Lbin)*Lbin].reshape(-1, Lbin).max(axis=1)
    
    t_Imax = np.arange(Nfi*itime, Nff*itime*Lbin, itime*Lbin*Nstep)

    return t_Imax, Imax


#################################
######### COMUPTE G2t ###########
#################################

def get_G2t(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None, MKL_library=True, NumExpr_library=True):
    '''
    Compute the G2t matrix from the e4m, properly masked with the matrix mask.

    Args:
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
    
    Returns:
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

def get_G2t_bybunch(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None, Nbunch=2, MKL_library=True, NumExpr_library=True):
    '''
    Compute the G2t matrix from the e4m, bunching the frames in Nbunch bunches.

    Args:
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    Nbunch: int
        Number of bunches to consider
    MKL_library: boolean
        If True, use the MKL library for the matrix multiplication
    NumExpr_library: boolean
        If True, use the NumExpr library for the normalization

    Returns:
    G2t: np.array
        G2t matrix
    '''


    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    Lbunch = (Nff-Nfi)//Nbunch

    G2t = np.zeros((Lbunch//Lbin, Lbunch//Lbin), dtype=np.float64)
    
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

    Args:
    dt: float
        Time step between frames
    G2t: np.array
        G2t matrix
    cython: boolean
        If True, use the cython code to compute the g2
    parallel: boolean
        If True, use the parallel cython code to compute the g2 (still not implemented!)

    Returns:
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

    Args:
    dt: float
        Time step between frames
    g2: np.array
        g2 array    

    Returns:
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

def plot_XYprofile(e4m_data, ax='Y', mask=None, Nfi=None, Nff=None):
    '''
    Plot the X and Y profiles of the e4m detector.

    Args:
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    '''

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]

    It = (e4m_data[Nfi:Nff].sum(axis=0)/(Nff-Nfi)).reshape(Nx, Ny)
    if mask is not None: It[~mask.reshape(Nx, Ny)] = 0

    plt.figure(figsize=(8,5))
    if (ax=='Y') or (ax=='y'):
        plt.plot(It.sum(axis=0)/mask.reshape(Nx, Ny).sum(axis=0))
        plt.xlabel('Y [px]')
    if (ax=='X') or (ax=='x'):
        plt.plot(It.sum(axis=1)/mask.reshape(Nx, Ny).sum(axis=1))
        plt.xlabel('X [px]')
    
    plt.tight_layout()
    plt.show()



######################
###### PLOT G2T ######
######################

def plot_G2t(G2t, vmin, vmax, itime=None, t1=None, t2=None, x1=None, x2=None, sigma_filter=None, full=False):

    # Behaviours when t1, t2 are None
    if t1 is None: t1 = 0
    if (t2 is None) and (itime is None): t2 = G2t.shape[0]
    elif (t2 is None) and (itime is not None): t2 = G2t.shape[0]*itime

    # Behaviours when t2 is to big
    if (itime is None) and (t2>G2t.shape[0]): t2 = G2t.shape[0]
    elif (itime is not None) and (t2>G2t.shape[0]*itime): t2 = G2t.shape[0]*itime

    # Behaviours when x1, x2 are None
    if (x1 is None) and (x2 is None): x1, x2 = t1, t2
    elif x1 is None: x1 = 0
    elif x2 is None: x2 = G2t.shape[1]

    # Cut G2t
    if itime is None: G2t = G2t[t1:t2, x1:x2]
    else: G2t = G2t[int(t1//itime):int(t2//itime), int(x1//itime):int(x2//itime)]

    # Gaussian filter
    if sigma_filter is not None:
        truncate = 4
        radius = 2*round(truncate*sigma_filter) + 1 + truncate

        for i in range(1, int(radius)+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = G2t.diagonal(offset=i)

        G2t = gaussian_filter(G2t, sigma=sigma_filter, mode='nearest', truncate=4)

        for i in range(1, int(radius)*4+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = 0

    if full==True:
        G2t += G2t.T


    # plot
    plt.figure(figsize=(8,8))
    plt.imshow(G2t, vmin=vmin, vmax=vmax, origin='lower')

    plt.xlabel('Time [s]')
    plt.ylabel('Time [s]')
    plt.yticks(np.round(np.linspace(0, G2t.shape[0], 6)).astype(int), np.round(np.linspace(x1, x2, 6)).astype(int))
    plt.xticks(np.round(np.linspace(0, G2t.shape[1], 6)).astype(int), np.round(np.linspace(t1, t2, 6)).astype(int))
    plt.colorbar()
    plt.tight_layout()
    plt.show()