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




import numpy as np

def get_It(data, Lbin=None, Nstep=None):
    '''
    Compute the average frame intensity [ph/px/s] vector from the data. 
    '''

    itime = config["itime"]

    # DEFAULT VALUES
    if Lbin is None: Lbin = 1
    if Nstep is None: Nstep = 1

    if Lbin > Nstep:
        raise ValueError("Lbin cannot be greater than Nstep")

    max_limit = (data.shape[0] // Nstep) * Nstep
    idx = np.array([i for i in range(data.shape[0]) if i % Nstep < Lbin and i < max_limit])
    
    It = data[idx].sum(axis=1) / data.shape[1]
    if Lbin != 1: 
        It = It[:(It.size//Lbin)*Lbin].reshape(-1, Lbin).sum(axis=1) / Lbin  # BIN It (if Lbin > 1)

    It /= itime                                                             # NORMALIZE It
    t_It = np.arange(1, It.size + 1) * itime * Nstep                        # BUILD THE TIME VECTOR    

    return np.vstack((t_It, It))





def get_Itp_bin(data, Lbin, bin2dense=False, verbose=True):
    '''
    Bin ..
    '''

    Itp = data

    # BIN DATA
    if verbose:
        t0 = time.time()
        print('Binning frames (Lbin = '+str(Lbin)+') ...')

    # BIN DATA (truncate excess frames if not divisible)
    remainder = Itp.shape[0] % Lbin
    if remainder != 0:
        n_frames_to_keep = (Itp.shape[0] // Lbin) * Lbin
        if verbose:
            print(f"Warning: The number of frames is not divisible by Lbin ({Lbin}).")
            print(f"Discarding the last {remainder} frames.")
        Itp = Itp[:n_frames_to_keep]

    BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))), dtype=np.float32)
    Itp = dot_product_mkl(BIN_matrix, Itp, dense=bin2dense)

    if verbose:
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
        if isinstance(Itp, (sparse.sparray, sparse.spmatrix)):
            print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
            print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3, 3), 'GB')
        else:
            print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3, 3), 'GB')

    return Itp




def get_G2t(data, verbose=True):
    '''
    Compute the G2t matrix from the e4m, properly masked.  
    '''
    Itp = data

    # 1. Compute G2t
    t0 = time.time()
    if verbose:
        print('Computing G2t (using MKL library)...')
    
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)
    
    if verbose:
        print(f"Done! (elapsed time = {time.time() - t0:.2f} s)")
        print(f"\t | {G2t.shape[0]} X {G2t.shape[1]} squared matrix")
        print(f"\t | memory usage (np.array @ {G2t.dtype}) = {G2t.nbytes / 1024**3:.3f} GB")
            
    # 2. Normalize G2t
    t0 = time.time()
    if verbose:
        print('Normalizing G2t (using NumExpr library)...')
        
    It = Itp.sum(axis=1, dtype=np.float32)
    np.divide(np.sqrt(Itp.shape[1]), It, where=It > 0, out=It, dtype=np.float32)
    Itr = It[:, None] 
    Itc = It[None, :] 
    ne.evaluate('G2t * Itr * Itc', out=G2t)
    
    if verbose:
        print(f"Done! (elapsed time = {time.time() - t0:.2f} s)\n")
        
    return G2t




##############################
######### GET g2 #############
##############################

def get_g2(dt, G2t):
    '''
    Compute the g2 from the G2t matrix.
    '''

    t0 = time.time()
    print('Computing g2 (using cython code)...')
    
    if G2t.dtype == np.float32: 
        g2, dg2 = mean_trace_float32(G2t)
    elif G2t.dtype == np.float64: 
        g2, dg2 = mean_trace_float64(G2t)
    else: 
        raise ValueError(f"G2t dtype not implemented in cython code! Got {G2t.dtype}")
    print(f"Done! (elapsed time = {time.time() - t0:.2f} s)\n")

    return np.arange(1, len(g2)+1)*dt, g2, dg2



def plot_G2t(G2t, vmin, vmax, t1=None, t2=None, x1=None, x2=None):
    '''
    Plot the G2t matrix.
    '''
    itime = config.itime

    # BEHAVIORS WHEN t1, t2 ARE NONE
    if t1 is None: 
        t1 = 0
    if t2 is None: 
        t2 = G2t.shape[0] * itime

    # BEHAVIOURS WHEN x1, x2 ARE NONE
    if (x1 is None) and (x2 is None): 
        x1, x2 = t1, t2
    elif x1 is None: 
        x1 = 0
    elif x2 is None: 
        x2 = G2t.shape[1]

    # CUT THE G2t MATRIX
    G2t = G2t[int(t1 // itime):int(t2 // itime), int(x1 // itime):int(x2 // itime)]

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