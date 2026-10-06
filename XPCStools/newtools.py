import numpy as np
import numba as nb
from scipy import sparse


from .config import config





######################
##### GET IT FAST ####
######################

@nb.njit(parallel=True, fastmath=True)
def _csr_direct_binned_sums(data, indptr, Nfi, Nbin, n_out_bins):
    """
    Sfrutta la contiguità di memoria della CSR per sommare blocchi di Nbin frame
    con un unico ciclo ultra-veloce su data.data.
    """
    binned_sums = np.empty(n_out_bins, dtype=np.float64)
    for b in nb.prange(n_out_bins):
        p_start = indptr[Nfi + b * Nbin]
        p_end = indptr[Nfi + (b + 1) * Nbin]
        
        s = 0.0
        for p in range(p_start, p_end):
            s += data[p]
        binned_sums[b] = s
        
    return binned_sums


@nb.njit(parallel=True, fastmath=True)
def _csc_direct_binned_sums(data, indices, Nfi, Nff, Nbin, n_out_bins):
    """
    Versione parallela 1D a blocchi (chunking): suddivide l'array data equamente 
    tra i thread CPU in totale sicurezza (thread-safe).
    """
    n_threads = nb.get_num_threads()
    thread_sums = np.zeros((n_threads, n_out_bins), dtype=np.float64)
    
    n_elements = len(data)
    chunk_size = (n_elements + n_threads - 1) // n_threads

    for t in nb.prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, n_elements)
        
        for k in range(start, end):
            f = indices[k]
            if Nfi <= f < Nff:
                b = (f - Nfi) // Nbin
                thread_sums[t, b] += data[k]

    # Riduzione dei parziali compatibile con Numba
    out = np.zeros(n_out_bins, dtype=np.float64)
    for b in range(n_out_bins):
        s = 0.0
        for t in range(n_threads):
            s += thread_sums[t, b]
        out[b] = s

    return out



def get_It_fast(data, Nbin, Nfi=None, Nff=None):
    """
    Calcola l'intensità media nel tempo binnata su blocchi di Nbin frame.
    Versione ultra-veloce a zero allocazioni intermedie.
    """
    itime = config['itime']

    if not isinstance(data, (sparse.csr_matrix, sparse.csr_array, sparse.csc_matrix, sparse.csc_array)):
        raise TypeError("Input 'data' deve essere una matrice sparse CSR o CSC di SciPy.")

    is_csr = isinstance(data, (sparse.csr_matrix, sparse.csr_array))
    n_frames, n_pixels = data.shape

    Nfi = 0 if Nfi is None else int(Nfi)
    Nff = n_frames if Nff is None else int(Nff)
    out_frames = Nff - Nfi

    if out_frames % Nbin != 0:
        raise ValueError(f'Numero di frame ({out_frames}) non divisibile per Nbin ({Nbin})')

    n_out_bins = out_frames // Nbin

    # 1. SOMME BINNATE DIRETTE SUI CORE CPU
    if is_csr:
        binned_sums = _csr_direct_binned_sums(data.data, data.indptr, Nfi, Nbin, n_out_bins)
    else:
        # CORRETTO: Passati esattamente i 6 argomenti richiesti
        binned_sums = _csc_direct_binned_sums(
            data.data, data.indices, Nfi, Nff, Nbin, n_out_bins
        )

    # 2. SCALATURA IN UN UNICO PASSAGGIO SCALARE
    scale = 1.0 / (n_pixels * Nbin * itime)
    It = binned_sums * scale

    # 3. VETTORE DEI TEMPI
    t = np.arange(n_out_bins, dtype=np.float64) * (Nbin * itime)

    return np.vstack((t, It))







########################
##### NORMALIZE ITP ####
########################

# =============================================================================
# KERNEL NUMBA (CSR)
# =============================================================================

@nb.njit(parallel=True, fastmath=True, cache=True)
def _csr_row_sums(data, indptr, Nfi, Nff):
    """Calcola la somma per riga nell'intervallo [Nfi, Nff) per CSR."""
    out_frames = Nff - Nfi
    sums = np.empty(out_frames, dtype=np.float64)
    for i in nb.prange(out_frames):
        r = Nfi + i
        s = 0.0
        for p in range(indptr[r], indptr[r + 1]):
            s += data[p]
        sums[i] = s
    return sums


@nb.njit(parallel=True, fastmath=True)
def _csr_apply_norm(data, indptr, norm):
    """Moltiplica ogni riga della matrice CSR per il rispettivo fattore di scala."""
    n_rows = len(indptr) - 1
    for i in nb.prange(n_rows):
        s = norm[i]
        for p in range(indptr[i], indptr[i + 1]):
            data[p] *= s


# =============================================================================
# KERNEL NUMBA (CSC)
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def _csc_row_sums(data, indices, out_frames, Nfi, Nff):
    """Calcola la somma per riga tramite scansione 1D sequenziale velocissima."""
    sums = np.zeros(out_frames, dtype=np.float64)
    for k in range(len(indices)):
        f = indices[k]
        if Nfi <= f < Nff:
            sums[f - Nfi] += data[k]
    return sums


@nb.njit(parallel=True, fastmath=True)
def _csc_apply_norm_full(data, indices, norm):
    """Normalizza CSC in-place quando non c'è slicing sui frame."""
    for k in nb.prange(len(data)):
        data[k] *= norm[indices[k]]


@nb.njit(parallel=True, fastmath=True)
def _csc_count_nnz(indices, indptr, Nfi, Nff, n_pixels):
    """Conta gli elementi validi per colonna per ricreare indptr."""
    counts = np.zeros(n_pixels, dtype=np.int32)
    for p in nb.prange(n_pixels):
        for k in range(indptr[p], indptr[p + 1]):
            if Nfi <= indices[k] < Nff:
                counts[p] += 1
    return counts


@nb.njit(parallel=True, fastmath=True)
def _csc_fill_norm(data, indices, indptr, new_data, new_indices, new_indptr, norm, Nfi, Nff, n_pixels):
    """Filtra, ri-indicizza e normalizza la matrice CSC in un unico passaggio."""
    for p in nb.prange(n_pixels):
        write_pos = new_indptr[p]
        for k in range(indptr[p], indptr[p + 1]):
            f = indices[k]
            if Nfi <= f < Nff:
                new_indices[write_pos] = f - Nfi
                new_data[write_pos] = data[k] * norm[f - Nfi]
                write_pos += 1


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def normalize_Itp(data, t_bin, t_norm=1e-3, Nfi=None, Nff=None):
    """
    Normalizza i dati XPCS/TEMPUS supportando sia CSR che CSC.
    Ottimizzata per velocità e basso consumo di memoria RAM.
    """
    # 1. CONTROLLI E PARAMETRI BASE
    if not isinstance(data, (sparse.csr_matrix, sparse.csr_array, sparse.csc_matrix, sparse.csc_array)):
        raise TypeError("Input 'data' deve essere una matrice sparse CSR o CSC di SciPy.")

    is_csr = isinstance(data, (sparse.csr_matrix, sparse.csr_array))
    n_frames, n_pixels = data.shape

    Nfi = 0 if Nfi is None else int(Nfi)
    Nff = n_frames if Nff is None else int(Nff)
    out_frames = Nff - Nfi

    N_norm = round(t_norm / t_bin)
    if out_frames % N_norm != 0:
        raise ValueError(f'Numero di frame ({out_frames}) non divisibile per N_norm ({N_norm})')

    # 2. CALCOLO DELLA SOMMA PER RIGA
    if is_csr:
        row_sums = _csr_row_sums(data.data, data.indptr, Nfi, Nff)
    else:
        row_sums = _csc_row_sums(data.data, data.indices, out_frames, Nfi, Nff)

    # 3. VETTORE DI NORMALIZZAZIONE
    row_means = row_sums / n_pixels
    M_binned = row_means.reshape(-1, N_norm).mean(axis=1)

    if M_binned.min() <= 0:
        raise ValueError('Vettore di normalizzazione contiene valori <= 0, impossibile normalizzare.')

    scale_factor = 1.0 / (M_binned * np.sqrt(n_pixels))
    norm = np.repeat(scale_factor, N_norm).astype(data.dtype)

    # 4. APPLICAZIONE E COSTRUZIONE OUTPUT
    if is_csr:
        # Slicing istantaneo su indptr per CSR
        r_start, r_end = data.indptr[Nfi], data.indptr[Nff]
        new_indptr = data.indptr[Nfi : Nff + 1] - r_start
        new_indices = data.indices[r_start : r_end].copy()
        new_data = data.data[r_start : r_end].copy()

        _csr_apply_norm(new_data, new_indptr, norm)
        return type(data)((new_data, new_indices, new_indptr), shape=(out_frames, n_pixels))

    else: # CSC
        if Nfi == 0 and Nff == n_frames:
            # CSC senza slicing temporale
            new_data = data.data.copy()
            _csc_apply_norm_full(new_data, data.indices, norm)
            return type(data)((new_data, data.indices.copy(), data.indptr.copy()), shape=(out_frames, n_pixels))
        else:
            # CSC con slicing temporale
            counts = _csc_count_nnz(data.indices, data.indptr, Nfi, Nff, n_pixels)
            new_indptr = np.zeros(n_pixels + 1, dtype=np.int32)
            np.cumsum(counts, out=new_indptr[1:])

            total_nnz = new_indptr[-1]
            new_data = np.empty(total_nnz, dtype=data.dtype)
            new_indices = np.empty(total_nnz, dtype=np.int32)

            _csc_fill_norm(data.data, data.indices, data.indptr, new_data, new_indices, new_indptr, norm, Nfi, Nff, n_pixels)
            return type(data)((new_data, new_indices, new_indptr), shape=(out_frames, n_pixels))





