import time
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.ndimage import gaussian_filter
import warnings

import os
os.environ["MKL_INTERFACE_LAYER"] = "ILP64"

from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl

# Cython mean trace functions import fallback
try:
    from .cython_comp import mean_trace_float32, mean_trace_float64
except ImportError:
    warnings.warn(
        "Compiled Cython extensions (cython_comp) are not available. "
        "Falling back to pure NumPy mean trace calculation, which may be significantly slower.",
        ImportWarning
    )

    # Pure Python / Numpy fallback for diagonal mean trace calculations
    def _mean_trace_fallback(G2t):
        N = G2t.shape[0]
        g2 = np.zeros(N, dtype=G2t.dtype)
        dg2 = np.zeros(N, dtype=G2t.dtype)
        for k in range(1, N):
            diag = np.diagonal(G2t, offset=k)
            if len(diag) > 0:
                g2[k - 1] = np.mean(diag)
                dg2[k - 1] = np.std(diag) / np.sqrt(len(diag))
        return g2, dg2

    mean_trace_float32 = _mean_trace_fallback
    mean_trace_float64 = _mean_trace_fallback

from .config import config




def get_It(data, itime, Lbin=None, Nstep=None):
    '''
    Compute the average frame intensity [ph/px/s] vector from the data, properly masked with the mask. 
    
    Parameters
    ----------
    data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    mask: np.array
        Mask of the e4m detector
    Lbin: int
        Binning factor for the frames
    Nstep: int
        Step for the frames

    Returns
    -------
    t_It: np.array
        Time array for the It vector
    It: np.array
        It vector
    '''
    # DEFAULT VALUES
    if Lbin is None: Lbin = 1
    if Nstep is None: Nstep = 1

    if Lbin > Nstep:
        raise ValueError("Lbin cannot be greater than Nstep")

    # COMPUTE It (masked)
    idx = np.array([i for i in range(data.shape[0]) if i % Nstep < Lbin][:((data.shape[0])//Nstep-1)*Nstep]) # GET THE CORRECT INDEXES FROM Nfi, Nff, Lbin and Nstep
    It = data[idx].sum(axis=1) / data.shape[1]
    if Lbin != 1: It = It[:(It.size//Lbin)*Lbin].reshape(-1, Lbin).sum(axis=1) / Lbin                   # BIN It (if Lbin > 1)
    It /= itime                                                                                        # NORMALIZE It
    t_It = np.linspace(itime, itime, It.shape[0])                                              # BUILD THE TIME VECTOR    

    return np.vstack(t_It, It)




def get_Itp_bin(data, Lbin, bin2dense=False):

    # LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    Itp = data
    # convert to float32
    if Itp.dtype != np.float32:
        Itp = Itp.astype(np.float32)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # BIN DATA
    t0 = time.time()
    print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
    Itp = (Itp[:Itp.shape[0]//Lbin*Lbin]) # throw the last frames 
    BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))), dtype=np.float32)
    Itp = dot_product_mkl(BIN_matrix, Itp, dense=bin2dense)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    if isinstance(Itp, (sparse.sparray, sparse.spmatrix)):
        print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
        print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3, 3), 'GB')
    else:
        print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3, 3), 'GB')

    return Itp




def get_G2t(data, Lbin=None, bin2dense=False):
    '''
    Compute the G2t matrix from the e4m, properly masked with the matrix mask.

    Parameters
    ----------
    data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    Lbin: int
        Binning factor for the frames
    bin2dense: boolean
        If True, return dense matrix
    
    Returns
    -------
    G2t: np.array
        G2t matrix
        
    '''

    Itp = data
    if Itp.dtype != np.float32: raise ValueError('Data type must be float32, but got {}'.format(Itp.dtype))
    
    # BIN DATA
    if Lbin is not None:
        print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
        Itp = (Itp[:Itp.shape[0]//Lbin*Lbin]) # throw the last frames 
        BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))), dtype=np.float32)
        Itp = dot_product_mkl(BIN_matrix, Itp, dense=bin2dense)
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
        if isinstance(Itp, (sparse.sparray, sparse.spmatrix)):
            print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
            print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3, 3), 'GB')
        else:
            print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3, 3), 'GB')
    
    # Compute G2t
    t0 = time.time()
    print('Computing G2t (using MKL library)...')
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    print('\t | '+str(G2t.shape[0])+' X '+str(G2t.shape[1])+' squared matrix')
    print('\t | memory usage (np.array @ '+str(G2t.dtype)+') =', round(G2t.nbytes/1024**3, 3), 'GB')
           
    # Normalize G2t
    t0 = time.time()
    print('Normalizing G2t (using NumExpr library)...')
    It = Itp.sum(axis=1, dtype=np.float32)
    np.divide(np.sqrt(Itp.shape[1]), It, where=It > 0, out=It, dtype=np.float32)
    Itr = It[:, None] # q[:, None] -> q.reshape(N, 1)
    Itc = It[None, :] # q[None, :] -> q.reshape(1, N)
    ne.evaluate('G2t*Itr*Itc', out=G2t)
    
    # Remove diagonal and fill no counts frames
    G2t[G2t.diagonal() == 0, :] = 1
    G2t[:, G2t.diagonal() == 0] = 1
    np.fill_diagonal(G2t, 0)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')
    return G2t


##########################################
######### COMUPTE G2t bunnched ###########
##########################################

def get_G2t_bybunch(data, Nbunch, Lbin=None, bin2dense=False):
    '''
    Compute the G2t matrix from the e4m, bunching the frames in Nbunch bunches, thus averaging the G2t matrix over the bunches. 
    '''

    # GET BUNCHES LENGHT [fms]
    Lbunch = data.shape[0]//Nbunch

    # PREPARE THE G2t MATRIX
    G2t = np.zeros((Lbunch//Lbin, Lbunch//Lbin), dtype=np.float64)
    
    # COMPUTE G2t FOR EACH BUNCH
    for i in range(Nbunch):
        print('Computing G2t for bunch', i+1, '...')
        G2t += get_G2t(data[i*Lbunch:(i+1)*Lbunch,:], Lbin=Lbin, bin2dense=bin2dense)
        print('Done!\n')

    return G2t/Nbunch




##############################
######### GET g2 #############
##############################

def get_g2(dt, G2t):
    '''
    Compute the g2 from the G2t matrix.

    Parameters
    ----------
    dt: float
        Time step between frames
    G2t: np.array
        G2t matrix

    Returns
    -------
    t: np.array
        Time array
    g2: np.array
        g2 array
    '''
    
    t0 = time.time()

    print('Computing g2 (using cython code)...')
    if G2t.dtype == np.float32: g2, dg2 = mean_trace_float32(G2t)
    elif G2t.dtype == np.float64: g2, dg2 = mean_trace_float64(G2t)
    else: raise ValueError('G2t dtype not implemented in cython code!')
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')

    g2, dg2 = g2[:-1], dg2[:-1]
    idx = np.where(dg2 == 0)[0]
    g2 = g2[:idx[0]] if len(idx) > 0 else g2
    dg2 = dg2[:idx[0]] if len(idx) > 0 else dg2

    idx = np.where(np.isnan(dg2))[0]
    g2 = g2[:idx[0]] if len(idx) > 0 else g2
    dg2 = dg2[:idx[0]] if len(idx) > 0 else dg2

    return np.arange(1, len(g2)+1)*dt, g2, dg2




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

    # BEHAVIORS WHEN t1, t2 ARE NONE
    if t1 is None: t1 = 0
    if (t2 is None) and (itime is None): t2 = G2t.shape[0]
    elif (t2 is None) and (itime is not None): t2 = G2t.shape[0]*itime

    # BEHAVIOURS WHEN t2 IS BIGGER THAN THE G2t MATRIX
    if (itime is None) and (t2 > G2t.shape[0]): t2 = G2t.shape[0]
    elif (itime is not None) and (t2 > G2t.shape[0]*itime): t2 = G2t.shape[0]*itime

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
    if full is True:
        G2t += G2t.T

    # PLOT
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(G2t, vmin=vmin, vmax=vmax, origin='lower')

    # add ticks
    ax.set_yticks(np.round(np.linspace(0, G2t.shape[0], 6)).astype(int))
    ax.set_yticklabels(np.round(np.linspace(x1, x2, 6)).astype(int))
    ax.set_xticks(np.round(np.linspace(0, G2t.shape[1], 6)).astype(int))
    ax.set_xticklabels(np.round(np.linspace(t1, t2, 6)).astype(int))

    # add labels and colorbar
    ax.set_xlabel('$t_1$ [s]')
    ax.set_ylabel('$t_2$ [s]')
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, ax