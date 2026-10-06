"""
XPCStools.multitau
------------------
Multi-tau correlation functions, sparse multi-tau algorithms, 
and visualization utilities for XPCS data.
"""

import time
import numpy as np
import pandas as pd
import numexpr as ne
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import numba as nb

# Internal imports from matrix_comp
from .matrix_comp import gram_matrix_mkl, dot_product_mkl, get_Itp_bin

from .config import config




def _get_symG2t(Itp):
    # Compute G2t upper triangle
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)

    # Normalize G2t (directly accounting for 0-counts frames)
    
    #It = Itp.sum(axis=1, dtype=np.float32)
    #It = np.where(It > 0, It, np.sqrt(Itp.shape[1], dtype=np.float32))
    #np.divide(np.sqrt(Itp.shape[1]), It, where=It > 0, out=It, dtype=np.float32)

    sqrt_N = np.float32(np.sqrt(Itp.shape[1]))
    It_sum = Itp.sum(axis=1, dtype=np.float32)
    It = np.full_like(It_sum, np.nan)
    np.divide(sqrt_N, It_sum, out=It, where=It_sum > 0)

    Itr, Itc = It[:, None], It[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)

    # Replace exact zeros with 1
    #G2t[G2t == 0] = 1.0

    return G2t




def _get_nonsymG2t(Itp1, Itp2):
    # Compute full G2t
    G2t = dot_product_mkl(Itp1, Itp2.T, dense=True)
           
    # Normalize G2t (directly accounting for 0-counts frames)
    #It1 = Itp1.sum(axis=1, dtype=np.float32)
    #It2 = Itp2.sum(axis=1, dtype=np.float32)
    #It1 = np.where(It1 > 0, It1, np.sqrt(Itp1.shape[1], dtype=np.float32))
    #It2 = np.where(It2 > 0, It2, np.sqrt(Itp2.shape[1], dtype=np.float32))
    #np.divide(np.sqrt(Itp1.shape[1]), It1, out=It1, dtype=np.float32)
    #np.divide(np.sqrt(Itp2.shape[1]), It2, out=It2, dtype=np.float32)

    sqrt_N1 = np.float32(np.sqrt(Itp1.shape[1]))
    sqrt_N2 = np.float32(np.sqrt(Itp2.shape[1]))
    It1_sum = Itp1.sum(axis=1, dtype=np.float32)
    It2_sum = Itp2.sum(axis=1, dtype=np.float32)
    It1 = np.full_like(It1_sum, np.nan)
    It2 = np.full_like(It2_sum, np.nan)
    np.divide(sqrt_N1, It1_sum, out=It1, where=It1_sum > 0)
    np.divide(sqrt_N2, It2_sum, out=It2, where=It2_sum > 0)

    Itr, Itc = It1[:, None], It2[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)

    # Replace exact zeros with 1
    #G2t[G2t == 0] = 1.0

    return G2t




def _G2t2G2tmt(G2t, type, ch_depth):
    """
    Transform G2t matrix slices into multi-tau array format.
    """
    # check if G2t is a square matrix
    if G2t.shape[0] != G2t.shape[1]:
        raise ValueError('G2t must be a square matrix! Current shape is ' + str(G2t.shape[0]) + 'x' + str(G2t.shape[1]) + '!')

    # check if G2t shape is a power of 2
    if int(np.log2(G2t.shape[0])) != np.log2(G2t.shape[0]):
        raise ValueError('G2t must be a square matrix with size 2^n (n integer)! Current size is ' + str(G2t.shape[0]) + 'x' + str(G2t.shape[1]) + '!')
    
    # check ch_depth minimum value
    if ch_depth < 1:
        raise ValueError('ch_depth must be greater than or equal to 1! Otherwise you are working with less then 2 channels (the 0-channel compute the variance)!')

    # check if log2(G2t.shape[0]) is greater than or equal to ch_depth
    if np.log2(G2t.shape[0]) <= ch_depth:
        print(np.log2(G2t.shape[0]), ch_depth)
        raise ValueError('log2(G2t.shape[0]) must be greater than or equal to ch_depth! Current log2(G2t.shape[0]) is ' + str(int(np.log2(G2t.shape[0]))) + ' and ch_depth is ' + str(ch_depth) + '!')

    # prepare the G2tmt list & Correlators & Channels ranges
    G2tmt = [[] for _ in range(int(np.log2(G2t.shape[0])) - ch_depth + 1)]
    Correlators = range(len(G2tmt))
    Channels    = range(2**ch_depth)

    # Compute the G2tmt for each correlator and channel
    for corr in Correlators:
        for ch in Channels:
            if (ch == 0) or ((corr > 0) and (ch < 2**ch_depth // 2)):
                G2tmt[corr].append(np.array([]))
            elif type == 'sym':
                G2tmt[corr].append(G2t.diagonal(offset=ch).copy())
            elif type == 'non-sym':
                G2tmt[corr].append(G2t.diagonal(offset=-G2t.shape[0] + ch).copy())

        # Bin G2tmt by a factor of 2
        if corr != Correlators[-1]:
            BIN_matrix = sparse.csr_array((np.ones(G2t.shape[0]), (np.arange(G2t.shape[0]) // 2, np.arange(G2t.shape[0]))), dtype=np.float32)

            # build nan mask
            #nan_mask = np.isnan(G2t)

            # remove nans
            #G2t[nan_mask] = 0

            #nan_mask = nan_mask.astype(np.float32)

            # bin G2t and nan_mask
            G2t = dot_product_mkl(BIN_matrix, G2t)
            G2t = dot_product_mkl(BIN_matrix, G2t.T)
            G2t = G2t.T / 4

            #nan_mask = dot_product_mkl(BIN_matrix, nan_mask)
            #nan_mask = dot_product_mkl(BIN_matrix, nan_mask.T)

            # apply nan mask to G2t
            #G2t[nan_mask == 4] = np.nan

    return G2tmt





def _process_sparse_block(N: int, Itp, sparse_depth: int, ch_depth: int, Nfi:int, N_sparseloops: int, skip_dense: bool, skip_norm: bool = False):
    """Processes a single sparse block N independently across worker threads/processes.
       Outputs the dense frame sums and G2tmt correlations for the block.
    """
    S = 2**sparse_depth
    ch_step = 2**(sparse_depth - ch_depth)
    n_dense_rows = 2**(ch_depth - 1)

    # Extract slice for current block N
    ItpN = Itp[Nfi + N * S : Nfi + (N + 1) * S]

    # Calculate dense frame sums for block N
    if not skip_dense:
        dense_rows = np.zeros((n_dense_rows, Itp.shape[1]), dtype=np.float32)
        for ch in range(0, 2**ch_depth, 2):
            dense_rows[ch // 2] = ItpN[ch * ch_step : (ch + 2) * ch_step].sum(axis=0)
    else:
        dense_rows = None

    # Calculate symmetric G2t correlation
    if not skip_norm:
        G2t_sym = _get_symG2t(ItpN)
    else:
        G2t_sym = gram_matrix_mkl(ItpN, dense=True, transpose=True) 

    G2tmt_sym = _G2t2G2tmt(G2t_sym, type='sym', ch_depth=ch_depth)

    # Calculate non-symmetric G2t correlation with adjacent block N+1
    G2tmt_nonsym = None
    if N != N_sparseloops - 1:
        ItpN_next = Itp[(N + 1) * S : (N + 2) * S]
        if not skip_norm:
            G2t_nonsym = _get_nonsymG2t(ItpN, ItpN_next)
        else:
            G2t_nonsym = dot_product_mkl(ItpN, ItpN_next.T, dense=True)

        G2tmt_nonsym = _G2t2G2tmt(G2t_nonsym, type='non-sym', ch_depth=ch_depth)

    return N, dense_rows, G2tmt_sym, G2tmt_nonsym





def get_G2tmt_4sparse(data, sparse_depth: int, ch_depth: int = 4, Nfi: int = 0, Nff: int = -1, n_jobs: int = 1, skip_dense: bool = False, skip_norm = False, verbose: bool = True):
    """
    Compute the multitau (mt) G2t correlation from sparse data in parallel.

    Parameters
    ----------
    data : sparse.csr_matrix
        Sparse data of shape (Nf, Npx).
    sparse_depth : int
        The number of sparse multitau levels.
    ch_depth : int, optional
        Channel depth parameter. Default is 4.
    Nfi : int, optional
        Initial frame to consider (inclusive).
    Nff : int, optional
        Final frame to consider (exclusive).
    n_jobs : int, optional
        Number of parallel CPU cores (-1 uses all cores). Default is -1.

    Returns
    -------
    G2tmt : list of np.ndarray
        List containing the sparse multitau G2t correlation arrays.
    """
    # Set automatic value for Nff if not provided
    if Nff==-1: 
        Nff = (data.shape[0] - Nfi) // 2**sparse_depth * 2**sparse_depth + Nfi
        if verbose:
            print(f'Nff set to {Nff} => (Nff-Nfi) = {(data.shape[0]-Nfi) // 2**sparse_depth}*2^sparse_depth, thrown frames = {(data.shape[0]-Nfi-(Nff-Nfi))} ({round((data.shape[0]-Nfi-(Nff-Nfi))/(data.shape[0]-Nfi)*100, 2)}%)')

    Itp = data

    ### CHECK ARGUMENTS CONDITIONS
    if ch_depth < 1:
        raise ValueError('ch_depth must be greater than or equal to 1!')

    if (Nff-Nfi) / 2**sparse_depth != int((Nff-Nfi) / 2**sparse_depth):
        raise ValueError('(Nff-Nfi) must be a multiple of 2**sparse_depth!')
    
    if (Nff-Nfi) < 2**(sparse_depth - ch_depth):
        raise ValueError('(Nff-Nfi) must be greater than or equal to 2**(sparse_depth-ch_depth)!')

    ############################ PARALLEL SPARSE COMPUTATION ############################
    
    if verbose:
        t0 = time.time()
        print(f'Computing sparse multitau G2t in parallel (n_jobs={n_jobs})...')

    N_sparseloops = (Nff-Nfi) // 2**sparse_depth
    Correlators = range(sparse_depth - ch_depth + 1)
    Channels = range(2**ch_depth)

    if not skip_dense:
        n_dense_rows = 2**(ch_depth - 1)
        Itp_dense = np.zeros((N_sparseloops * n_dense_rows, Itp.shape[1]), dtype=np.float32)
    G2tmt_chunks = [[[] for _ in Channels] for _ in Correlators]

    # Parallel loop across block iterations
    jobs = (delayed(_process_sparse_block)(N, Itp, sparse_depth, ch_depth, Nfi, N_sparseloops, skip_dense, skip_norm) for N in range(N_sparseloops))

    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_gen = parallel(jobs)
        results = list(tqdm(results_gen, total=N_sparseloops, desc="Sparse blocks", disable=not verbose))

    # Re-order outputs by original iteration index N
    results.sort(key=lambda x: x[0])

    # Assemble outputs into pre-allocated memory structures
    for N, dense_rows, G2tmt_sym, G2tmt_nonsym in results:
        if not skip_dense:
            Itp_dense[N * n_dense_rows : (N + 1) * n_dense_rows] = dense_rows
        
        for corr in Correlators:
            for ch in Channels:
                G2tmt_chunks[corr][ch].append(G2tmt_sym[corr][ch])
                if G2tmt_nonsym is not None:
                    G2tmt_chunks[corr][ch].append(G2tmt_nonsym[corr][ch])

    # Single vector concatenation per correlator/channel
    G2tmt = [
        [np.concatenate(G2tmt_chunks[corr][ch]) if len(G2tmt_chunks[corr][ch]) > 0 else np.array([])
         for ch in Channels]
        for corr in Correlators
    ]

    if verbose:
        print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    if skip_dense:
        return G2tmt
    
############################ DENSE COMPUTATION ############################
    
    if verbose:
        t0 = time.time()
        print('Computing dense multitau G2t ...') 

    num_channels = 2**ch_depth
    half_channels = 2**(ch_depth - 1)
    n_pixels = Itp_dense.shape[1]

    mem_gb = round(Itp_dense.nbytes / 1024**3, 3)
    if verbose:
        print(f"\t | {Itp_dense.shape[0]} frames X {n_pixels} pixels (memory = {mem_gb} GB)")

    # Calculate exact number of dense iterations needed
    n_dense_levels = (Itp_dense.shape[0] // (num_channels + 1)).bit_length() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if skip_norm:
        Itp_dense /= 2**(sparse_depth - ch_depth+1)

    for _ in tqdm(range(n_dense_levels), desc="Dense levels", disable=not verbose):
        if not skip_norm:
            norm = np.float32(np.sqrt(n_pixels)) / Itp_dense.sum(axis=1, dtype=np.float32)

        level_g2tmt = [np.array([]) for _ in range(half_channels)]
        
        for ch in range(half_channels, num_channels):
            G2t_diag = np.einsum('ij,ij->i', Itp_dense[:-ch], Itp_dense[ch:])
            if not skip_norm:
                level_g2tmt.append(G2t_diag * norm[ch:] * norm[:-ch])
            else:
                level_g2tmt.append(G2t_diag)

        G2tmt.append(level_g2tmt)

        # Truncate to even frame count and bin by factor of 2
        n_even = (Itp_dense.shape[0] // 2) * 2
        Itp_dense = Itp_dense[:n_even].reshape(-1, 2, n_pixels).sum(axis=1)
        
        if skip_norm:
            Itp_dense /= 2

    if verbose:
        print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    return G2tmt




def plot_G2tmt(G2tmt, vmin, vmax, lower_corr=4, upper_corr=None, yscale='log', filter_layer=None, xlims=None, vlines=None):
    """
    Plot a multi-tau correlation matrix (G2tmt) using broken bar plot.
    """

    itime = config["itime"]

    #linewidth = 0.2 if borders else 0
    linewidth = 0

    if upper_corr is None: 
        N_corr = len(G2tmt)
    else:                  
        N_corr = upper_corr
    N_ch = len(G2tmt[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    T = (G2tmt[0][1].shape[0] + 1) * itime

    for corr in range(lower_corr, N_corr):
        itime_corr = itime * 2**corr
        for ch in range(N_ch):
            if (ch == 0) or ((corr > 0) and (ch < N_ch // 2)):
                pass
            else:
                x = np.arange(G2tmt[corr][ch].size) * itime_corr + (1 + ch) * itime_corr / 2
                dx = itime_corr
                y = np.ones(G2tmt[corr][ch].size) * itime_corr * ch
                dy = itime_corr

                xranges = [(x[i] - dx / 2, dx) for i in range(len(x))]
                yrange = (y[0], dy)

                if (filter_layer is None) or (corr >= filter_layer):
                    BB = ax.broken_barh(
                        xranges, yrange, array=G2tmt[corr][ch], cmap="viridis", clim=(vmin, vmax), edgecolor="black", linewidth=linewidth
                    )
                else:
                    filtered_data = gaussian_filter1d(G2tmt[corr][ch], 2 ** (filter_layer - corr), mode="nearest")
                    BB = ax.broken_barh(
                        xranges, yrange, array=filtered_data, cmap="viridis", clim=(vmin, vmax), edgecolor="black", linewidth=linewidth
                    )

    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("$t_w$ [s]")
    ax.set_ylabel("$\\tau$ [s]")

    if xlims is None:
        ax.set_xlim(0, T)
    else:
        ax.set_xlim(xlims)
    if yscale == "log":
        ax.set_yscale("log")
    fig.colorbar(BB, ax=ax)
    fig.tight_layout()
    return fig, ax




def get_g2mt(G2tmt, t1=None, t2=None):
    """
    Calculate time delays, mean, and standard error of g2 values for multi-tau XPCS.
    Optionally cuts the data within a time window [t1, t2].
    """

    itime = config["itime"]

    t1 = 0 if t1 is None else t1
    t2 = np.inf if t2 is None else t2
    use_time_cut = (t1 > 0) or (t2 < np.inf)

    N_corr, N_ch = len(G2tmt), len(G2tmt[0])

    t_g2mt, g2mt, dg2mt = [], [], []

    for corr in range(N_corr):
        itime_corr = itime * (2**corr)
        
        for ch in range(N_ch):
            # Skip multi-tau redundant channels
            if (ch == 0) or (corr > 0 and ch < N_ch // 2):
                continue

            arr = G2tmt[corr][ch]

            # Apply time window mask if requested
            if use_time_cut:
                x = np.arange(arr.size) * itime_corr + (1 + ch) * itime_corr / 2
                mask = (x >= t1) & (x <= t2) # Simplified mask (x ± dx/2 bounds simplify to x directly)
                
                if not np.any(mask):
                    continue
                arr = arr[mask]
                
            #arr = arr[~np.isnan(arr)]  # Remove NaN values
            

            # Compute statistics
            t_g2mt.append(itime_corr * ch)
            g2mt.append(np.mean(arr))
            dg2mt.append(np.std(arr) / np.sqrt(arr.size))

    return np.vstack((np.array(t_g2mt), np.array(g2mt), np.array(dg2mt)))



def save_G2tmt(filepath, G2tmt):
    """
    Save G2tmt (2D nested list of np.ndarrays) to a compressed .npz file.
    """
    if not filepath.endswith('.npz'):
        filepath += '.npz'
        
    n_corr = len(G2tmt)
    n_ch = len(G2tmt[0]) if n_corr > 0 else 0
    
    # Store matrix dimensions and flattened array dict
    data_dict = {"_shape": np.array([n_corr, n_ch])}
    
    for corr in range(n_corr):
        for ch in range(n_ch):
            data_dict[f"arr_{corr}_{ch}"] = G2tmt[corr][ch]
            
    np.savez_compressed(filepath, **data_dict)
    print(f"G2tmt successfully saved to {filepath}")



def load_G2tmt(filepath):
    """
    Load G2tmt from a .npz file and reconstruct the 2D nested list structure.
    """
    if not filepath.endswith('.npz'):
        filepath += '.npz'
        
    data = np.load(filepath)
    n_corr, n_ch = data["_shape"]
    
    # Reconstruct 2D nested list
    G2tmt = [
        [data[f"arr_{corr}_{ch}"] for ch in range(n_ch)]
        for corr in range(n_corr)
    ]
    
    print(f"G2tmt successfully loaded ({n_corr} correlators x {n_ch} channels) from {filepath}")
    return G2tmt



################################################################################################################

def _process_sparse_block_wl(N: int, load_f, sparse_depth: int, ch_depth: int, Nfi:int, N_sparseloops: int):

    S = 2**sparse_depth
    ch_step = 2**(sparse_depth - ch_depth)
    n_dense_rows = 2**(ch_depth - 1)
    # Extract slice for current block N
    if N != N_sparseloops - 1:
        Itp = load_f(Nfi + N * S, Nfi + (N + 2) * S)
    else:
        Itp = load_f(Nfi + N * S, Nfi + (N + 1) * S)
    ItpN = Itp[:S]

    # Calculate dense frame sums for block N
    dense_rows = np.zeros((n_dense_rows, Itp.shape[1]), dtype=np.float32)
    for ch in range(0, 2**ch_depth, 2):
        dense_rows[ch // 2] = ItpN[ch * ch_step : (ch + 2) * ch_step].sum(axis=0)

    # Calculate symmetric G2t correlation
    G2t_sym = _get_symG2t(ItpN)
    G2tmt_sym = _G2t2G2tmt(G2t_sym, type='sym', ch_depth=ch_depth)

    # Calculate non-symmetric G2t correlation with adjacent block N+1
    G2tmt_nonsym = None
    if N != N_sparseloops - 1:
        ItpN_next = Itp[S:]
        G2t_nonsym = _get_nonsymG2t(ItpN, ItpN_next)
        G2tmt_nonsym = _G2t2G2tmt(G2t_nonsym, type='non-sym', ch_depth=ch_depth)

    return N, dense_rows, G2tmt_sym, G2tmt_nonsym




def get_G2tmt_4sparse_wl(load_f, sparse_depth: int, ch_depth: int = 4, Nfi: int = 0, Nff:int = -1,  n_jobs: int = 1):

    Nff_new = (Nff - Nfi) // 2**sparse_depth * 2**sparse_depth + Nfi
    print(f'Nff set to {Nff_new} => (Nff-Nfi) = {((Nff_new - Nfi)-Nfi) // 2**sparse_depth}*2^sparse_depth, thrown frames = {((Nff - Nfi)-Nfi-(Nff_new-Nfi))} ({round(((Nff - Nfi)-Nfi-(Nff_new-Nfi))/((Nff - Nfi)-Nfi)*100, 2)}%)')
    Nff=Nff_new

    ### CHECK ARGUMENTS CONDITIONS
    if ch_depth < 1:
        raise ValueError('ch_depth must be greater than or equal to 1!')

    if (Nff-Nfi) / 2**sparse_depth != int((Nff-Nfi) / 2**sparse_depth):
        raise ValueError('(Nff-Nfi) must be a multiple of 2**sparse_depth!')
    
    if (Nff-Nfi) < 2**(sparse_depth - ch_depth):
        raise ValueError('(Nff-Nfi) must be greater than or equal to 2**(sparse_depth-ch_depth)!')

    Npx = load_f(0,1).shape[1]

    ############################ PARALLEL SPARSE COMPUTATION ############################
    t0 = time.time()
    print(f'Computing sparse multitau G2t in parallel (n_jobs={n_jobs})...')

    N_sparseloops = (Nff-Nfi) // 2**sparse_depth
    Correlators = range(sparse_depth - ch_depth + 1)
    Channels = range(2**ch_depth)

    n_dense_rows = 2**(ch_depth - 1)
    Itp_dense = np.zeros((N_sparseloops * n_dense_rows, Npx), dtype=np.float32)
    G2tmt_chunks = [[[] for _ in Channels] for _ in Correlators]

    # Parallel loop across block iterations
    jobs = (delayed(_process_sparse_block_wl)(N, load_f, sparse_depth, ch_depth, Nfi, N_sparseloops) for N in range(N_sparseloops))

    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_gen = parallel(jobs)
        results = list(tqdm(results_gen, total=N_sparseloops, desc="Sparse blocks"))

    # Re-order outputs by original iteration index N
    results.sort(key=lambda x: x[0])

    # Assemble outputs into pre-allocated memory structures
    for N, dense_rows, G2tmt_sym, G2tmt_nonsym in results:
        Itp_dense[N * n_dense_rows : (N + 1) * n_dense_rows] = dense_rows
        
        for corr in Correlators:
            for ch in Channels:
                G2tmt_chunks[corr][ch].append(G2tmt_sym[corr][ch])
                if G2tmt_nonsym is not None:
                    G2tmt_chunks[corr][ch].append(G2tmt_nonsym[corr][ch])

    # Single vector concatenation per correlator/channel
    G2tmt = [
        [np.concatenate(G2tmt_chunks[corr][ch]) if len(G2tmt_chunks[corr][ch]) > 0 else np.array([])
         for ch in Channels]
        for corr in Correlators
    ]

    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')


############################ DENSE COMPUTATION ############################
    t0 = time.time()
    print('Computing dense multitau G2t ...')

    num_channels = 2**ch_depth
    half_channels = 2**(ch_depth - 1)
    n_pixels = Itp_dense.shape[1]

    mem_gb = round(Itp_dense.nbytes / 1024**3, 3)
    print(f"\t | {Itp_dense.shape[0]} frames X {n_pixels} pixels (memory = {mem_gb} GB)")

    # Calculate exact number of dense iterations needed
    n_dense_levels = (Itp_dense.shape[0] // (num_channels + 1)).bit_length()

    for _ in tqdm(range(n_dense_levels), desc="Dense levels"):
        norm = np.float32(np.sqrt(n_pixels)) / Itp_dense.sum(axis=1, dtype=np.float32)

        level_g2tmt = [np.array([]) for _ in range(half_channels)]
        
        for ch in range(half_channels, num_channels):
            G2t_diag = np.einsum('ij,ij->i', Itp_dense[:-ch], Itp_dense[ch:])
            level_g2tmt.append(G2t_diag * norm[ch:] * norm[:-ch])

        G2tmt.append(level_g2tmt)

        # Truncate to even frame count and bin by factor of 2
        n_even = (Itp_dense.shape[0] // 2) * 2
        Itp_dense = Itp_dense[:n_even].reshape(-1, 2, n_pixels).sum(axis=1)

    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    return G2tmt



@nb.njit(parallel=True, fastmath=True)
def _csr_multi_channel_autocorr(data, indices, indptr, ch_arr):
    """
    Autocorrelazione sparse multi-canale parallelizzata sulle righe (tempo).
    indices must be sorted in ascending order for each row (indptr[i]:indptr[i+1]) !!!!!
    """
    n_channels = len(ch_arr)
    n_rows = len(indptr) - 1
    results = np.empty(n_channels, dtype=np.float64)

    # Loop over channels
    for c in range(n_channels):
        ch = ch_arr[c]
        rows_for_ch = n_rows - ch
        ch_sum = 0.0 # Accumulatore float64

        # Parallel loop over rows (time) for the current channel
        for i in nb.prange(rows_for_ch):
            p1, end1 = indptr[i], indptr[i + 1]
            p2, end2 = indptr[i + ch], indptr[i + ch + 1]
            
            row_sum = 0.0
            while p1 < end1 and p2 < end2:
                c1 = indices[p1]
                c2 = indices[p2]
                if c1 == c2:
                    row_sum += data[p1] * data[p2]
                    p1 += 1
                    p2 += 1
                elif c1 < c2:
                    p1 += 1
                else:
                    p2 += 1
            
            ch_sum += row_sum
            
        results[c] = ch_sum / rows_for_ch
        
    return results



@nb.njit(parallel=True, fastmath=True)
def _csc_multi_channel_autocorr(data, indices, indptr, ch_arr, n_rows):
    """
    Autocorrelazione CSC con cast sicuro a float64 per evitare overflow nei prodotti.
    """
    n_channels = len(ch_arr)
    n_cols = len(indptr) - 1
    results = np.empty(n_channels, dtype=np.float64)

    for c in range(n_channels):
        ch = ch_arr[c]
        rows_for_ch = n_rows - ch
        ch_sum = 0.0

        for j in nb.prange(n_cols):
            start = indptr[j]
            end = indptr[j + 1]
            
            p1 = start
            p2 = start
            pix_sum = 0.0

            while p1 < end and p2 < end:
                dt = indices[p2] - indices[p1]
                
                if dt == ch:
                    # Cast esplicito dei dati prima della moltiplicazione
                    pix_sum += np.float64(data[p1]) * np.float64(data[p2])
                    p1 += 1
                    p2 += 1
                elif dt < ch:
                    p2 += 1
                else:  # dt > ch
                    p1 += 1

            ch_sum += pix_sum

        results[c] = ch_sum / rows_for_ch

    return results



@nb.njit(parallel=True, fastmath=True)
def _dense_multi_channel_autocorr(Itp, ch_arr):
    """
    Autocorrelazione dense multi-canale vettorizzata.
    """
    n_channels = len(ch_arr)
    n_rows, n_cols = Itp.shape
    results = np.empty(n_channels, dtype=np.float64)
    
    for c in range(n_channels):
        ch = ch_arr[c]
        rows_for_ch = n_rows - ch
        ch_sum = 0.0
        
        for i in nb.prange(rows_for_ch):
            row_sum = 0.0  # Accumulatore float64
            # Vettorizzazione diretta sui pixel j (float32 -> accumulo float64)
            for j in range(n_cols):
                row_sum += Itp[i, j] * Itp[i + ch, j]
            ch_sum += row_sum
            
        results[c] = ch_sum / rows_for_ch
        
    return results



@nb.njit(parallel=True, fastmath=True)
def _csr_temporal_bin2x(data, indices, indptr):
    """
    Esegue il binning temporale di fattore 2 su una matrice CSR:
    combina le righe (2*i) e (2*i + 1) e le moltiplica per 0.5.
    """
    n_out_rows = (len(indptr) - 1) // 2
    row_counts = np.empty(n_out_rows, dtype=np.int64)
    
    # PASS 1: Build the new indptr. Calcolo (parallelo) degli elementi non-zero per ogni coppia di righe.
    # for cycle running on the rows (time) pairs
    for i in nb.prange(n_out_rows):
        # take two consecutive pairs indices
        p1, end1 = indptr[2 * i], indptr[2 * i + 1]
        p2, end2 = indptr[2 * i + 1], indptr[2 * i + 2]
        cnt = 0
        # cycle on the data of these pairs
        while p1 < end1 and p2 < end2:
            c1, c2 = indices[p1], indices[p2]
            if c1 == c2:
                cnt += 1
                p1 += 1
                p2 += 1
            elif c1 < c2:
                cnt += 1
                p1 += 1
            else:
                cnt += 1
                p2 += 1
        # count all remaining elements in one of the two rows when one of them is finished (i.e. p1 == end1 or p2 == end2)
        cnt += (end1 - p1) + (end2 - p2)

        row_counts[i] = cnt

    # Costruzione del nuovo indptr
    new_indptr = np.empty(n_out_rows + 1, dtype=indptr.dtype)
    new_indptr[0] = 0
    new_indptr[1:] = np.cumsum(row_counts)

    # pre-allocate new data and indices arrays (I already know the size from new_indptr)
    nnz = new_indptr[-1]
    new_data = np.empty(nnz, dtype=data.dtype)
    new_indices = np.empty(nnz, dtype=indices.dtype)

    # PASS 2: Fusione e scaling (x 0.5) dei dati (in parallelo)
    for i in nb.prange(n_out_rows):
        out_p = new_indptr[i]
        p1, end1 = indptr[2 * i], indptr[2 * i + 1]
        p2, end2 = indptr[2 * i + 1], indptr[2 * i + 2]

        # Fusione dei dati delle due righe
        while p1 < end1 and p2 < end2:
            c1, c2 = indices[p1], indices[p2]
            if c1 == c2:
                new_indices[out_p] = c1
                new_data[out_p] = (data[p1] + data[p2]) * 0.5
                out_p += 1
                p1 += 1
                p2 += 1
            elif c1 < c2:
                new_indices[out_p] = c1
                new_data[out_p] = data[p1] * 0.5
                out_p += 1
                p1 += 1
            else:
                new_indices[out_p] = c2
                new_data[out_p] = data[p2] * 0.5
                out_p += 1
                p2 += 1
        # Handle any remaining elements in either row
        while p1 < end1:
            new_indices[out_p] = indices[p1]
            new_data[out_p] = data[p1] * 0.5
            out_p += 1
            p1 += 1

        while p2 < end2:
            new_indices[out_p] = indices[p2]
            new_data[out_p] = data[p2] * 0.5
            out_p += 1
            p2 += 1

    return new_data, new_indices, new_indptr



@nb.njit(parallel=True, fastmath=True)
def _csc_temporal_bin2x(data, indices, indptr, n_rows):
    """
    Binning temporale di fattore 2 su matrice CSC.
    Converte e gestisce i dati in float64 per evitare overflow.
    """
    n_cols = len(indptr) - 1
    # Tronca l'ultimo frame se n_rows è dispari (identico a SciPy)
    n_even = (n_rows // 2) * 2
    col_counts = np.empty(n_cols, dtype=np.int64)

    # PASS 1: Conteggio elementi non-zero (solo entro n_even)
    for j in nb.prange(n_cols):
        start = indptr[j]
        end = indptr[j + 1]
        cnt = 0
        k = start
        while k < end:
            r1 = indices[k]
            if r1 >= n_even:
                break  # Ignora i frame oltre n_even
            
            if k + 1 < end and indices[k + 1] < n_even and (indices[k + 1] // 2 == r1 // 2):
                cnt += 1
                k += 2
            else:
                cnt += 1
                k += 1
        col_counts[j] = cnt

    # Costruzione nuovo indptr
    new_indptr = np.empty(n_cols + 1, dtype=indptr.dtype)
    new_indptr[0] = 0
    new_indptr[1:] = np.cumsum(col_counts)

    nnz = new_indptr[-1]
    # new_data DEVE essere float64 per preservare la precisione dopo x0.5
    new_data = np.empty(nnz, dtype=np.float64)
    new_indices = np.empty(nnz, dtype=indices.dtype)

    # PASS 2: Scrittura dati con cast sicuro a float64
    for j in nb.prange(n_cols):
        out_p = new_indptr[j]
        start = indptr[j]
        end = indptr[j + 1]
        k = start
        
        while k < end:
            r1 = indices[k]
            if r1 >= n_even:
                break
                
            binned_row = r1 // 2
            
            if k + 1 < end and indices[k + 1] < n_even and (indices[k + 1] // 2 == binned_row):
                new_indices[out_p] = binned_row
                # Cast esplicito a float64 prima dell'addizione
                new_data[out_p] = (np.float64(data[k]) + np.float64(data[k + 1])) * 0.5
                k += 2
            else:
                new_indices[out_p] = binned_row
                new_data[out_p] = np.float64(data[k]) * 0.5
                k += 1
                
            out_p += 1

    return new_data, new_indices, new_indptr



def mt_corr(Itp, ch_depth=4, sparse_depth=None, verbose=True):

    # Determine the type of the input sparse matrix (CSR or CSC)
    if (type(Itp) == sparse.csr_matrix) or (type(Itp) == sparse.csr_array):
        Itp_type = 'csr'
    elif (type(Itp) == sparse.csc_matrix) or (type(Itp) == sparse.csc_array):
        Itp_type = 'csc'
    else:
        raise TypeError("Itp must be a sparse matrix (csr or csc)")

    # Set sparse_depth automatically if not provided, and determine whether to skip dense computation
    if sparse_depth is None:
        sparse_depth = int(np.log2(Itp.shape[0])) - ch_depth + 1
        skip_dense = True
    else:
        if sparse_depth > int(np.log2(Itp.shape[0])) - ch_depth + 1:
            raise ValueError(f"sparse_depth must be less than or equal to {int(np.log2(Itp.shape[0])) - ch_depth + 1} for the given Itp shape and ch_depth.")
        skip_dense = False

    # Variables for time and number of pixels
    itime = config["itime"]
    Npx = Itp.shape[1]

    g2tmt = []
    t_g2mt = []

    # --- PARTE SPARSE ---
    for corr in tqdm(range(sparse_depth), desc="Sparse levels", disable=not verbose):
        
        ch_range = np.arange(1 if corr == 0 else 2**(ch_depth - 1), 2**ch_depth, dtype=np.int64)

        # Parallel numba autocorrelation calculation
        if Itp_type == 'csr':
            g2_res = _csr_multi_channel_autocorr(Itp.data, Itp.indices, Itp.indptr, ch_range)
        elif Itp_type == 'csc':
            g2_res = _csc_multi_channel_autocorr(Itp.data, Itp.indices, Itp.indptr, ch_range, Itp.shape[0])

        # Append results to the output lists
        g2tmt.extend(g2_res)
        itime_corr = itime * (2**corr)
        t_g2mt.extend([itime_corr * ch for ch in ch_range])

        if (corr != sparse_depth - 1) or (not skip_dense):
            if Itp_type == 'csr':
                # WITH NUMBA: Binning in-place su matrice CSR
                new_data, new_indices, new_indptr = _csr_temporal_bin2x(Itp.data, Itp.indices, Itp.indptr)
                new_shape = (len(new_indptr) - 1, Npx)
                Itp = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
            elif Itp_type == 'csc':
                # WITH NUMBA: Binning in-place su matrice CSC
                new_data, new_indices, new_indptr = _csc_temporal_bin2x(Itp.data, Itp.indices, Itp.indptr, Itp.shape[0])
                new_shape = (Itp.shape[0]//2, len(new_indptr) - 1)
                Itp = sparse.csc_matrix((new_data, new_indices, new_indptr), shape=new_shape)

            # WITHOUT NUMBA: Binning in-place su matrice CSR
            #n_even = (Itp.shape[0] // 2) * 2
            #Itp = Itp[0:n_even:2] + Itp[1:n_even:2]
            #Itp.data *= 0.5


    # --- PARTE DENSE ---
    if not skip_dense:
        Itp = Itp.toarray() 

        if verbose:
            mem_gb = round(Itp.nbytes / 1024**3, 3)
            print(f"\t | {Itp.shape[0]} frames X {Itp.shape[1]} pixels (memory = {mem_gb} GB)")

        n_dense_levels = int(np.log2(Itp.shape[0])) - ch_depth + 1

        for corr in tqdm(range(n_dense_levels), desc="Dense levels", disable=not verbose):
            itime_corr = itime * (2**(corr + sparse_depth))
            
            ch_range = range(2**(ch_depth - 1), 2**ch_depth)
            ch_arr = np.ascontiguousarray(ch_range, dtype=np.int64)

            # Calcolo parallelo SIMD vettorizzato
            g2_dense = _dense_multi_channel_autocorr(Itp, ch_arr)
            
            g2tmt.extend(g2_dense)
            t_g2mt.extend([itime_corr * ch for ch in ch_range])

            n_even = (Itp.shape[0] // 2) * 2
            # Binning in-place su array NumPy denso
            Itp = 0.5 * (Itp[0:n_even:2] + Itp[1:n_even:2])

    return np.array([t_g2mt, g2tmt])
