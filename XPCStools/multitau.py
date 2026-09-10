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
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from joblib import Parallel, delayed

# Internal imports from matrix_comp
from .matrix_comp import gram_matrix_mkl, dot_product_mkl




def _get_symG2t(Itp):
    # Compute G2t upper triangle
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)

    # Normalize G2t (directly accounting for 0-counts frames)
    It = Itp.sum(axis=1, dtype=np.float32)
    It = np.where(It > 0, It, np.sqrt(Itp.shape[1], dtype=np.float32))
    np.divide(np.sqrt(Itp.shape[1]), It, where=It > 0, out=It, dtype=np.float32)
    Itr, Itc = It[:, None], It[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)
    return G2t




def _get_nonsymG2t(Itp1, Itp2):
    # Compute full G2t
    G2t = dot_product_mkl(Itp1, Itp2.T, dense=True)
           
    # Normalize G2t (directly accounting for 0-counts frames)
    It1 = Itp1.sum(axis=1, dtype=np.float32)
    It2 = Itp2.sum(axis=1, dtype=np.float32)
    It1 = np.where(It1 > 0, It1, np.sqrt(Itp1.shape[1], dtype=np.float32))
    It2 = np.where(It2 > 0, It2, np.sqrt(Itp2.shape[1], dtype=np.float32))
    np.divide(np.sqrt(Itp1.shape[1]), It1, out=It1, dtype=np.float32)
    np.divide(np.sqrt(Itp2.shape[1]), It2, out=It2, dtype=np.float32)
    Itr, Itc = It1[:, None], It2[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)
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
            G2t = dot_product_mkl(BIN_matrix, G2t)
            G2t = dot_product_mkl(BIN_matrix, G2t.T)
            G2t = G2t.T / 4

    return G2tmt






def get_G2tmt_4sparse(data, sparse_depth: int, ch_depth: int = 4, Nfi: int = 0, Nff: int = -1, n_jobs: int = -1):
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
    mask : np.ndarray, optional
        Boolean mask to select pixels for the computation.
    keep_symmetric : bool, optional
        Whether to keep symmetric dimensions during binning. Default is False.
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
        print(f'Nff set to {Nff} => (Nff-Nfi) = {(data.shape[0]-Nfi) // 2**sparse_depth}*2^sparse_depth, thrown frames = {(data.shape[0]-Nfi-(Nff-Nfi))} ({round((data.shape[0]-Nfi-(Nff-Nfi))/(data.shape[0]-Nfi)*100, 2)}%)')

    # LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    Itp = data[Nfi:Nff]
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### CHECK ARGUMENTS CONDITIONS
    if ch_depth < 1:
        raise ValueError('ch_depth must be greater than or equal to 1!')

    if Itp.shape[0] / 2**sparse_depth != int(Itp.shape[0] / 2**sparse_depth):
        raise ValueError('Itp.shape[0] must be a multiple of 2**sparse_depth!')
    
    if Itp.shape[0] < 2**(sparse_depth - ch_depth):
        raise ValueError('Itp.shape[0] must be greater than or equal to 2**(sparse_depth-ch_depth)!')

    ############################ PARALLEL SPARSE COMPUTATION ############################
    t0 = time.time()
    print(f'Computing sparse multitau G2t in parallel (n_jobs={n_jobs})...')

    N_sparseloops = Itp.shape[0] // 2**sparse_depth
    Correlators = range(sparse_depth - ch_depth + 1)
    Channels = range(2**ch_depth)

    n_dense_rows = 2**(ch_depth - 1)
    Itp_dense = np.zeros((N_sparseloops * n_dense_rows, Itp.shape[1]), dtype=np.float32)
    G2tmt_chunks = [[[] for _ in Channels] for _ in Correlators]

    # Parallel loop across block iterations
    jobs = (delayed(_process_sparse_block)(N, Itp, sparse_depth, ch_depth, N_sparseloops) for N in range(N_sparseloops))

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



def plot_G2tmt(G2tmt, itime, vmin, vmax, lower_corr=4, upper_corr=None, yscale='log', filter_layer=None, borders=False, xlims=None, vlines=None):
    """
    Plot a multi-tau correlation matrix (G2tmt) using broken bar plot.
    """
    linewidth = 0.2 if borders else 0

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




def get_g2mt(itime, G2tmt):
    """
    Calculate time delays, mean, and standard error of g2 values for multi-tau XPCS.
    """
    N_corr, N_ch = len(G2tmt), len(G2tmt[0])

    t_g2mt, g2mt, dg2mt = [], [], []
    for corr in range(N_corr):
        for ch in range(N_ch):
            if (ch == 0) or ((corr > 0) and (ch < N_ch // 2)):
                pass
            else:
                t_g2mt.append(itime * 2**corr * ch)
                g2mt.append(np.mean(G2tmt[corr][ch]))
                dg2mt.append(np.std(G2tmt[corr][ch]) / np.sqrt(G2tmt[corr][ch].size))

    return np.array(t_g2mt), np.array(g2mt), np.array(dg2mt)





def get_g2mt_cut(itime, G2tmt, t1, t2):
    """
    Calculate time delays and mean g2 cut within a time window [t1, t2].
    """

    if t1 is None:
        t1 = 0
    if t2 is None:
        t2 = np.inf

    N_corr, N_ch = len(G2tmt), len(G2tmt[0])

    t_g2mt, g2mt_cut, dg2mt_cut = [], [], []
    for corr in range(N_corr):
        itime_corr = itime * 2**corr
        for ch in range(N_ch):
            if (ch == 0) or ((corr > 0) and (ch < N_ch // 2)):
                pass
            else:
                x = np.arange(G2tmt[corr][ch].size) * itime_corr + (1 + ch) * itime_corr / 2
                dx = itime_corr
                mask = ((x - dx / 2) >= t1) & ((x + dx / 2) <= t2)

                if mask.sum() != 0:
                    t_g2mt.append(itime * 2**corr * ch)
                    g2mt_cut.append(np.mean(G2tmt[corr][ch][mask]))
                    dg2mt_cut.append(np.std(G2tmt[corr][ch][mask]) / np.sqrt(G2tmt[corr][ch][mask].size))

    return np.array(t_g2mt), np.array(g2mt_cut), np.array(dg2mt_cut)



def _process_sparse_block(N: int, Itp, sparse_depth: int, ch_depth: int, N_sparseloops: int):
    """Processes a single sparse block N independently across worker threads/processes."""
    S = 2**sparse_depth
    ch_step = 2**(sparse_depth - ch_depth)
    n_dense_rows = 2**(ch_depth - 1)

    # Extract slice for current block N
    ItpN = Itp[N * S : (N + 1) * S]

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
        ItpN_next = Itp[(N + 1) * S : (N + 2) * S]
        G2t_nonsym = _get_nonsymG2t(ItpN, ItpN_next)
        G2tmt_nonsym = _G2t2G2tmt(G2t_nonsym, type='non-sym', ch_depth=ch_depth)

    return N, dense_rows, G2tmt_sym, G2tmt_nonsym


