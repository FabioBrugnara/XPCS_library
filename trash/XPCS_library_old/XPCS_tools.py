### IMPORT SCIENTIFIC LIBRARIES ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy import sparse
from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl
import numexpr as ne

from .PETRA3_tools import Nx, Ny, Npx, of_value, lxp, lyp, X0, Y0, L

from .XPCScy_tools import mean_trace_float64, mean_trace_float64_parallel

# OVERFLOW VALUE IN PLOTS
of_value4plot = of_value

 
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


#################################
########### MASK PLOTS ##########
#################################

def gen_plots4mask(OF, e4m_data, itime, Ith, mask_geom, nozoom = False, max_plots=False, counts_plot=False, vmax_max=5):
    '''
    Function that generates a number of different plots to create the mask! By default it generates the average flux per pixel map and histogram.
        
    Args:
    - OF: boolean array of the overflows
    - e4m_data: sparse matrix of the data
    - itime: integration time
    - Ith: threshold for the mask
    - mask_geom: list of tuples with the mask geometry
    - nozoom: boolean to plot the full range of the histograms
    - max_plots: boolean to plot the maximum flux per pixel map and histogram
    - counts_plot: boolean to plot the counts histogram
    - vmax_max: maximum value for the maximum flux per pixel map
    '''
    I_max = e4m_data.max(axis=0).toarray()[0]
    I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])
    I_mean[OF] = of_value4plot

    print('################################################################################')
    print('Maximum value in the hull run ->', e4m_data.data.max())
    print('# of pixel above treshold -> ', I_mean[~OF][I_mean[~OF]>Ith].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[~OF][I_mean[~OF]>Ith].shape[0]/I_mean.shape[0]*100, 2), '%)')
    print('################################################################################\n')

    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    ax5 = plt.subplot(413)
    
    ### AVERAGE ANALYSIS ###
    im = ax4.imshow(I_mean.reshape(Nx, Ny), vmax=Ith, origin='lower', label='average ph flux per px')
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

    ax5.hist(I_mean[~OF], bins=200, range=(0, Ith*2), label='$\\phi$ per px (low counts zoom)')
    ax5.axvline(Ith, color='r', label='$I_{th}$')
    ax5.set_yscale('log')
    ax5.set_xlabel('Mean flux per px')
    ax5.legend()

    if nozoom:
        ax6 = plt.subplot(414)
        ax6.hist(I_mean[~OF], bins=200, label='$\\phi$ per px (no zoom)')
        ax6.axvline(Ith, color='r', label='$I_{th}$')
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
        im = ax4.imshow(I_max.reshape(Nx, Ny), vmax=vmax_max, origin='lower')
        cbar = plt.colorbar(im, ax=ax4)
        ax4.set_title('Max # of photons per px (in the run)')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # HISTOGRAM OVER THE PIXELS
        ax5.hist(I_max, bins=20, label='low counts zoom', range=(0, vmax_max))
        ax5.set_yscale('log')
        ax5.set_xlabel('max counts per px')
        ax5.legend()

        if nozoom:
            # HISTOGRAM OVER THE PIXELS
            ax6 = plt.subplot(414)
            ax6.hist(I_max, bins=100, label='no zoom')
            ax6.set_yscale('log')
            ax6.set_xlabel('max counts per px')
            ax6.legend()

        plt.tight_layout()
        plt.show()

    
    if counts_plot:
        fig = plt.figure(figsize=(8,6))
        ax1 = plt.subplot(211)

        ### COUNTS HISTOGRAM OF ALL DATA ##
        ax1.set_title('Ph counts for all px and frames')
        ax1.hist(e4m_data.data, bins=20, label = 'low counts zoom, no zeros', range=(0, vmax_max))
        ax1.set_yscale('log')
        ax1.set_xlabel('counts')
        ax1.legend()

        if nozoom:
            ax2 = plt.subplot(212)
            ax2.hist(e4m_data.data, bins=200, label = 'no zoom, no zeros')
            ax2.set_yscale('log')
            ax2.set_xlabel('counts')
            ax2.legend()

        plt.tight_layout()
        plt.show()


#################################
########### MASK GEN ############
#################################

def gen_mask(OF, e4m_data, itime, e4m_mask=None, Ith=None, Imax=None,  mask_geom=None):
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

    e4m_data = e4m_data.tocsc()

    I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])

    mask = np.ones_like(I_mean, dtype=bool).reshape(Nx, Ny)
    if mask_geom!=None:
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

    # filter hot pixels and eiger borders
    if Ith!=None : mask = mask * (I_mean<=Ith)
    if Imax!=None: mask = mask * (e4m_data.tocsc().max(axis=0).toarray()[0]<(Imax+1))

    # filter e4m detector lines
    if e4m_mask is not None: mask = mask * e4m_mask

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

    # Histograms
    plt.figure(figsize=(8,8))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    # Masked counts histogram (all frames and px)
    ax1.set_title('Masked counts histogram (all frames and px)')
    ax1.hist(e4m_data[:, mask].data, bins=10, label='no zero counts')
    ax1.set_yscale('log')
    ax1.set_xlabel('# of counts')
    ax1.legend

    # Masked histogram of px flux
    ax2.set_title('Masked histogram of px flux')
    ax2.hist(I_mean[mask], bins=100)
    ax2.set_yscale('log')
    ax2.set_xlabel('Mean flux per px')

    # Maked histogram of max xounts per px
    ax3.set_title('Masked histogram of max counts per px')
    plt.hist(e4m_data[:, mask].max(axis=0).data, bins=30, label='no zero counts')
    ax3.set_yscale('log')
    ax3.set_xlabel('Max counts per px')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return mask


def gen_Qmask(Ei, theta, Q, dq, OF=None, e4m_data=None, itime=None, Ith=None, mask=None, Qplot=False):
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

    dX0 = L*np.tan(np.deg2rad(theta))
    X, Y = np.mgrid[:Nx, :Ny]
    dX0_map =np.sqrt((dX0-(X-X0)*lxp)**2+((Y-Y0)*lyp)**2)
    theta_map = np.arctan(dX0_map/L)
    Q_map = theta2Q(Ei, np.rad2deg(theta_map))

    if (e4m_data is not None):

        if (itime is not None) : I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])
        else: I_mean = e4m_data.sum(axis=0)/e4m_data.shape[0]

        if (Ith is None):
            if OF is not None: Ith = I_mean[OF].mean()
            else: Ith = I_mean.mean()

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
        plt.imshow(I_mean.reshape(Nx, Ny), vmax=Ith, origin='lower')
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


#################################
######### Q MASK GEN ############
#################################

def get_G2t(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None, MKL_library=True, NumExpr_library=True):
    '''
    Compute the G2t matrix from the e4m, properly masked with the matrix mask.

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
    Itp = e4m_data.tocsr()[Nfi:Nff]
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # BIN DATA
    if Lbin != 1:
        t0 = time.time()
        if MKL_library:
            print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
            Itp = (Itp[:Itp.shape[0]//Lbin*Lbin]).astype(np.float64)
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0], dtype=np.float64), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = dot_product_mkl(BIN_matrix, Itp, dense=False)
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    
        else:
            print('Binning frames (Lbin = '+str(Lbin)+') ...')
            Itp = Itp[:Itp.shape[0]//Lbin*Lbin]
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = BIN_matrix@Itp
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
    print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**2,3), 'MB')

    #  MASK DATA
    t0 = time.time()
    if mask is not None:
        print('Masking data ...')
        Itp = Itp[:,mask]
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
    print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**2,3), 'MB')

    # Compute G2t
    t0 = time.time()
    if MKL_library:
        print('Computing G2t (using MKL library)...')
        Itp = Itp.astype(np.float64)
        G2t = np.zeros((Itp.shape[0], Itp.shape[0]), dtype=np.float64)
        gram_matrix_mkl(Itp, dense=True, transpose=True, out=G2t)
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(G2t.shape[0])+' X '+str(G2t.shape[1])+' squared matrix')
        print('\t | memory usage (np.array @ '+str(G2t.dtype)+') =', round(G2t.nbytes/1024**3,3), 'GB')

    else:
        print('Computing G2t (and converting the sparse product to np.array)...')
        G2t = (Itp@Itp.T).toarray()
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
    np.fill_diagonal(G2t, 0)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')

    return G2t

def get_G2t_bunched(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None, Nbunch=2, MKL_library=True):

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    Lbunch = (Nff-Nfi)//Nbunch

    G2t = np.zeros((Lbunch//Lbin, Lbunch//Lbin), dtype=np.float64)
    
    for i in range(Nbunch):
        print('##### Computing G2t for bunch', i+1, '(Nfi =', Nfi+i*Lbunch, ', Nff =', Nfi+(i+1)*Lbunch, ') #####\n')
        G2t += get_G2t(e4m_data, mask, Nfi=Nfi+i*Lbunch, Nff=Nfi+(i+1)*Lbunch, Lbin=Lbin, MKL_library=MKL_library)

    return G2t/Nbunch


def get_g2(dt, G2t, cython=True, parallel=False):
    '''
    Compute the g2 from the G2t matrix.

    Args:
    dt: float
        Time step between frames
    G2t: np.array
        G2t matrix

    Returns:
    t: np.array
        Time array
    g2: np.array
        g2 array
    '''
    
    t0 = time.time()
    if cython:
        if parallel:
            print('Computing g2 (using parallel cython code)...')
            'WARNING!!! WORK IN PROGRESS; NOT CORRECTLY IMPLEMENTED YET!!!'
            g2 = mean_trace_float64_parallel(G2t)
        else:
            print('Computing g2 (using cython code)...')
            g2 = mean_trace_float64(G2t)
    else:
        print('Computing g2...')
        g2 = np.array([G2t.diagonal(i).mean() for i in range(1,G2t.shape[0])])
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')

    return np.arange(len(g2))*dt, g2


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