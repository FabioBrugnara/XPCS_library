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




def get_G2tmt_4sparse(e4m_data, sparse_depth: int, ch_depth: int = 4, Nfi: int = None, Nff: int = None, mask=None, keep_symmetric=False):
    """
    Compute the multitau (mt) G2t correlation from sparse e4m_data.

    Parameters
    ----------
    e4m_data : sparse.csr_matrix
        Sparse e4m_data of shape (Nf, Npx).
    sparse_depth : int
        The number of sparse multitau levels.
    ch_depth : int, optional
        Channel depth parameter. Default is 4.
    Nfi : int, optional
        Initial frame to consider (inclusive).
    Nff : int, optional
        Final frame to consider (exclusive).
    mask : np.ndarray, optional
        Boolean mask to select pixels for the computation. If None, all pixels are used.
    keep_symmetric : bool, optional
        Whether to keep symmetric dimensions during binning. Default is False.

    Returns
    -------
    G2tmt : list of np.ndarray
        List containing the sparse multitau G2t correlation arrays for each level.
    """
    
    ### DEFAULT VALUES
    if Nfi is None: 
        Nfi = 0
    if Nff is None: 
        Nff = (e4m_data.shape[0] - Nfi) // 2**sparse_depth * 2**sparse_depth + Nfi
        print(f'Nff set to {Nff} => (Nff-Nfi) = {(e4m_data.shape[0]-Nfi) // 2**sparse_depth}*2^sparse_depth, thrown frames = {(e4m_data.shape[0]-Nfi-(Nff-Nfi))} ({round((e4m_data.shape[0]-Nfi-(Nff-Nfi))/(e4m_data.shape[0]-Nfi)*100, 2)}%)')

    ### LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi != 0) or (Nff != e4m_data.shape[0]): 
        Itp = e4m_data[Nfi:Nff]
    else: 
        Itp = e4m_data
    if Itp.dtype != np.float32:
        Itp = Itp.astype(np.float32)
        print('and converting to float32 ...')
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### MASK DATA
    if mask is not None:
        t0 = time.time()
        print('Masking data ...')
        Itp = Itp[:, mask]
        print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### PRINT DATA INFO
    print('\t | ' + str(Itp.shape[0]) + ' frames X ' + str(Itp.shape[1]) + ' pixels')
    print('\t | sparsity = {:.2e}'.format(Itp.data.size / (Itp.shape[0] * Itp.shape[1])))
    print('\t | memory usage (sparse.csr_array @ ' + str(Itp.dtype) + ') =', round((Itp.data.nbytes + Itp.indices.nbytes + Itp.indptr.nbytes) / 1024**3, 3), 'GB')

    ### CHECK ARGUMENTS CONDITIONS
    if ch_depth < 1:
        raise ValueError('ch_depth must be greater than or equal to 1! Otherwise you are working with less then 2 channels (the 0-channel compute the variance)!')

    if Itp.shape[0] / 2**sparse_depth != int(Itp.shape[0] / 2**sparse_depth):
        raise ValueError('Itp.shape[0] must be a multiple of 2**sparse_depth! Current Itp.shape[0] is ' + str(Itp.shape[0]) + ' and sparse_depth is ' + str(sparse_depth) + '.')
    
    if Itp.shape[0] < 2**(sparse_depth - ch_depth):
        raise ValueError('Itp.shape[0] must be greater than or equal to 2**(sparse_depth-ch_depth)! Current Itp.shape[0] is ' + str(Itp.shape[0]) + ' and sparse_depth is ' + str(sparse_depth) + ' and ch_depth is ' + str(ch_depth) + '.')
    

    ############################ SPARSE COMPUTATION ############################
    t0 = time.time()
    print('Computing sparse multitau G2t...')

    N_sparseloops = Itp.shape[0] // 2**sparse_depth
    Correlators = range(sparse_depth - ch_depth + 1)
    Channels = range(2**ch_depth)

    G2tmt = [[np.array([]) for _ in Channels] for _ in Correlators]
    Itp_dense = np.zeros((N_sparseloops * 2**(ch_depth - 1), Itp.shape[1]), dtype=np.float32)
    
    Itp1 = Itp[:2**sparse_depth]
    Itp2 = Itp[2**sparse_depth:2**(sparse_depth + 1)]
    for N in tqdm(range(N_sparseloops)):

        for ch in Channels:
            if ch % 2 == 0:
                if N % 2 == 0: 
                    Itp_dense[2**(ch_depth - 1) * N + ch // 2] = Itp1[ch * 2**(sparse_depth - ch_depth): (ch + 2) * 2**(sparse_depth - ch_depth)].sum(axis=0)
                else:        
                    Itp_dense[2**(ch_depth - 1) * N + ch // 2] = Itp2[ch * 2**(sparse_depth - ch_depth): (ch + 2) * 2**(sparse_depth - ch_depth)].sum(axis=0)

        # Compute central G2t
        if N % 2 == 0: 
            G2t = _get_symG2t(Itp1)
        else:        
            G2t = _get_symG2t(Itp2)

        G2tmt_2add = _G2t2G2tmt(G2t, type='sym', ch_depth=ch_depth)

        for corr in Correlators:
            for ch in Channels:
                G2tmt[corr][ch] = np.append(G2tmt[corr][ch], G2tmt_2add[corr][ch])
        
        # Compute shifted G2t
        if N != N_sparseloops - 1:
            if N % 2 == 0: 
                G2t = _get_nonsymG2t(Itp1, Itp2)
            else:        
                G2t = _get_nonsymG2t(Itp2, Itp1)

            G2tmt_2add = _G2t2G2tmt(G2t, type='non-sym', ch_depth=ch_depth)

            for corr in Correlators:
                for ch in Channels:
                    G2tmt[corr][ch] = np.append(G2tmt[corr][ch], G2tmt_2add[corr][ch])

            if N % 2 == 0: 
                Itp1 = Itp[(N + 2) * 2**sparse_depth:(N + 3) * 2**sparse_depth]
            else:        
                Itp2 = Itp[(N + 2) * 2**sparse_depth:(N + 3) * 2**sparse_depth]
  
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')


    ############################ DENSE COMPUTATION ############################
    t0 = time.time()
    print('Computing dense multitau G2t...') 

    while Itp_dense.shape[0] > 2**(ch_depth):
        print(f"\t-> computing channels on {Itp_dense.shape[0]} frames ...")

        G2tmt.append([])
        norm = np.divide(np.sqrt(Itp_dense.shape[1]), Itp_dense.sum(axis=1), dtype=np.float32)
        
        for ch in Channels:
            if ch < 2**ch_depth // 2:
                G2tmt[-1].append(np.array([]))
            else:
                G2t_diag = (Itp_dense[:-ch] * Itp_dense[ch:]).sum(axis=1)
                G2tmt[-1].append(np.array(G2t_diag * norm[ch:] * norm[:-ch]))  

        # Bin Itp_dense by a factor 2
        if Itp_dense.shape[0] / 2 == Itp_dense.shape[0] // 2:
            Itp_dense = np.sum(Itp_dense.reshape((Itp_dense.shape[0] // 2, 2, Itp_dense.shape[1])), axis=1)
        else:
            if not keep_symmetric:
                Itp_dense = Itp_dense[0:Itp_dense.shape[0] // 2 * 2]
                Itp_dense = np.sum(Itp_dense.reshape((Itp_dense.shape[0] // 2, 2, Itp_dense.shape[1])), axis=1)
            else:
                break
    
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    return G2tmt




def print_Nf_choices(Nf: int):
    """
    Print the possible choices for reduced Nf, sparse depth, and thrown frames.
    """
    print(f'       Nf = {Nf}    =>    log2(Nf) = {round(np.log2(Nf), 2)}')
    print('----------------------------------------------------')
    exp_max = int(np.log2(Nf))
    df = pd.DataFrame(columns=['red Nf', 'sparse depth', 'subG2t mem (GBy)', 'thrown frames %', 'thrown frames'])

    # exp_max case
    Nf_red = Nf - 2**(exp_max)
    df.loc[0] = [f'2**{exp_max}', exp_max, round(2**(2 * exp_max) * 4 / 1024**3, 1), round(Nf_red / Nf * 100, 1), Nf_red]

    # next cases
    for minus in range(1, exp_max - 11):
        n = int(Nf / (2**(exp_max - minus)))
        Nf_red = Nf - n * 2**(exp_max - minus)
        df.loc[len(df)] = [f'{n}*2**{exp_max - minus}', exp_max - minus, round(2**(2 * (exp_max - minus)) * 4 / 1024**3, 1), round(Nf_red / Nf * 100, 1), Nf_red]

    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df)
    print('----------------------------------------------------')




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




def cut_G2tmt(itime, G2tmt, tmin=None, tmax=None):
    """
    Cuts the G2tmt arrays based on specified minimum and maximum time thresholds.
    """
    G2tmt_cut = []
    for b in range(len(G2tmt)):
        if tmin is None: 
            tmin = 0
        if tmax is None: 
            tmax = (G2tmt[0].shape[0] + 1) * itime

        sel = (np.arange(2**b, G2tmt[0].shape[0] + 1, 2**b) * itime - itime * 2**b >= tmin) * (np.arange(2**b, G2tmt[0].shape[0] + 1, 2**b) * itime + itime * 2**b <= tmax)
        if sel.sum() == 0:
            return G2tmt_cut
        else:
            G2tmt_cut.append(G2tmt[b][sel])
    return G2tmt_cut




def get_g2mt_cut(itime, G2tmt, t1, t2):
    """
    Calculate time delays and mean g2 cut within a time window [t1, t2].
    """
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