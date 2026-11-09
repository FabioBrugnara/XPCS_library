"""
ID10_tools
==========

A python library for loading and processing data from the ID10 beamline at the European Synchrotron Radiation Facility (ESRF). The library provides functions for loading and processing data from the Eiger 4M detector, including converting dense data to sparse format, loading scan information, and handling overflow values.

Author: Fabio Brugnara
"""

### IMPORT LIBRARIES ###
import os, time, gc
import numpy as np
from scipy import sparse                # for sparse array in scipy
import h5py, hdf5plugin                 # for hdf5 file reading
from joblib import Parallel, delayed    # for parallel processing
from tqdm import tqdm


Nfmax_dense_file_e4m   = None     # this is the default value
Nfmax_sparse_file_e4m  = 10000     # this is the default value
Nfmax_sparse_file_pitec = 5000    # this is the default value
len_dataset_string = 4
len_scan_string    = 4
len_fileidx_string = 5
dense_eiger4m_filename   = 'eiger4m_v2_frame_0_'
sparse_eiger4m_filename  = 'eiger4m_v2_sparse_frame_0_'
sparse_pitec_filename    = 'pitec_sparse_frame_'
Npx_e4m = 2162*2068
Npx_pitec = 512*512*3



####################################
####### LOAD SCAN INFO #############
###################################b#

def load_scan(raw_folder, sample_name, Ndataset, Nscan):
    '''  
    Parameters
    ----------
    raw_folder: string
        path to raw data folder
    sample_name: string
        name of the sample
    Ndataset: int
        number of the dataset
    Nscan: int
        number of the scan

    Returns
    -------
    scan: dict
        dictionary with scan parameters
    '''

    # LOAD H5 FILE
    h5file = h5py.File(f"{raw_folder}/{sample_name}/{sample_name}_{Ndataset:0{len_dataset_string}d}/{sample_name}_{Ndataset:0{len_dataset_string}d}.h5", 'r')
    
    # LOAD SCAN PARAMETERS
    scan = {}
    # general
    scan['command']       = h5file[f"{Nscan}.1"]['title'][()].decode("utf-8")
    scan['start_time']    = h5file[f"{Nscan}.1"]['start_time'][()].decode("utf-8")
    try: scan['end_time'] = h5file[f"{Nscan}.1"]['end_time'][()].decode("utf-8")
    except: pass

    # triggers
    try: scan['fast_timer_trig']   = h5file[f"{Nscan}.1"]['measurement']['fast_timer_trig'][:]
    except: pass
    try: scan['fast_timer_period'] = h5file[f"{Nscan}.1"]['measurement']['fast_timer_period'][:]
    except: pass
    try: scan['fast_epoch_trig'] = h5file[f"{Nscan}.1"]['measurement']['fast_epoch_trig'][:]
    except: pass
    try: scan['slow_timer_trig']   = h5file[f"{Nscan}.1"]['measurement']['slow_timer_trig'][:]
    except: pass
    try: scan['slow_timer_period'] = h5file[f"{Nscan}.1"]['measurement']['slow_timer_period'][:]
    except: pass
    try: scan['slow_epoch_trig'] = h5file[f"{Nscan}.1"]['measurement']['slow_epoch_trig'][:]
    except: pass

    # energy
    try: scan['monoe'] = h5file[f"{Nscan}.1"]['instrument']['positioners']['monoe'][()]
    except: pass

    # diodes
    try: scan['eh2diode'] = h5file[f"{Nscan}.1"]['measurement']['eh2diode'][:]
    except: pass
    try: scan['ch2_saxs'] = h5file[f"{Nscan}.1"]['measurement']['ch2_saxs'][:]
    except: pass
    try: scan['tetramm_diode'] = h5file[f"{Nscan}.1"]['measurement']['tetramm_diode'][:]
    except: pass
    try: scan['ch1_sample'] = h5file[f"{Nscan}.1"]['measurement']['ch1_sample'][:]
    except: pass


    # motor positions
    try:    scan['delcoup'] = h5file[f"{Nscan}.1"]['instrument']['positioners']['delcoup'][:]
    except: scan['delcoup'] = h5file[f"{Nscan}.1"]['instrument']['positioners']['delcoup'][()]
    try:    scan['ys']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['ys'][:]
    except: scan['ys']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['ys'][()]
    try:    scan['zs']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['zs'][:]
    except: scan['zs']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['zs'][()]
    try:    scan['th']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['th'][:]
    except: scan['th']      = h5file[f"{Nscan}.1"]['instrument']['positioners']['th'][()]

    # PID temperatures
    try:    scan['omega_sample'] = h5file[f"{Nscan}.2"]['measurement']['omega_sample'][:]
    except: 
        try: scan['omega_sample'] = h5file[f"{Nscan}.1"]['measurement']['omega_sample'][:]
        except: pass
    try:    scan['omega_body']   = h5file[f"{Nscan}.2"]['measurement']['omega_body'][:]
    except: 
        try:    scan['omega_body']   = h5file[f"{Nscan}.1"]['measurement']['omega_body'][:]
        except: pass
    try:    scan['epoch']        = h5file[f"{Nscan}.2"]['measurement']['epoch'][:]
    except:
        try:    scan['epoch']        = h5file[f"{Nscan}.1"]['measurement']['epoch'][:]
        except: pass

    # stufs
    try: scan['elapsed_time'] = h5file[f"{Nscan}.2"]['measurement']['elapsed_time'][:]
    except: pass
    try: scan['current']      = h5file[f"{Nscan}.2"]['measurement']['current'][:]
    except: pass
    try: scan['elapsed_time_1']      = h5file[f"{Nscan}.1"]['measurement']['elapsed_time'][:]
    except: pass


    return scan




#######################################
######### LOAD E4M DATA ###############
#######################################

def load_dense_e4m(raw_folder, sample_name, Ndataset, Nscan, Nfi=None, Nff=None, n_jobs=-1, load_mask=None, tosparse=False):

    t0 = time.time()

    ### E4M DATA FOLDER
    h5_folder = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + 'scan'+str(Nscan).zfill(len_dataset_string) + '/'
    print(f"Loading HDF5 files from {h5_folder}{dense_eiger4m_filename}{'X'*len_fileidx_string}.h5 ...")

    #### GET THE INDEXES OF THE FILES THAT SHOULD BE LOADED
    if Nfi is None: file_i = 0 # if Nfi is None, start from the file 0
    else: file_i = Nfi//Nfmax_dense_file_e4m
    if Nff is None: file_f = len([i for i in os.listdir(h5_folder) if i.startswith(dense_eiger4m_filename)]) # if Nff is None, finish at the last file
    else : file_f = Nff//Nfmax_dense_file_e4m+1

    total_files = file_f - file_i

    ### LOAD FRAMES BY FILE FUNCTION (file i, with Nfi, Nff refered to the full run)
    def _load_framesbyfile(i):
        import hdf5plugin # import hdf5plugin (needed for the parallel loop)

        # LOAD H5 FILE (setting the correct file name for the version)
        h5file = h5py.File(h5_folder + dense_eiger4m_filename + str(i).zfill(len_fileidx_string) + '.h5', 'r')

        # LOAD DATA
        if (file_i==i)  and (Nfi is not None):   a = Nfi%Nfmax_dense_file_e4m                     # get initial frame for the file
        else: a = 0
        if (file_f==i+1) and (Nff is not None):  b = Nff%Nfmax_dense_file_e4m                     # get final frame for the file
        else: b = Nfmax_dense_file_e4m
        npA = h5file['entry_0000']['measurement']['data'][a:b]                                # load data
        npA = npA.reshape((npA.shape[0], Npx_e4m))                                            # reshape data from (Nx, Ny) to (Npx_e4m)                                     # remove overflow values for version v1 (if OF_mask_v1 is not None)
        if load_mask is not None: npA = npA[:,load_mask]                                      # apply load_mask (if load_mask is not None)
        if tosparse: return sparse.csr_array(npA, dtype=np.float32)                           # convert to sparse array and return (if tosparse is True)              
        else:        return npA.astype(np.float32)                                            # return dense array (if tosparse is False)

    # 1. Define distinct job generator
    jobs = (delayed(_load_framesbyfile)(i) for i in range(file_i, file_f))

    # 2. Parallel loop with generator and tqdm progress bar
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_gen = parallel(jobs)
        A = list(tqdm(results_gen, total=total_files, desc="Files loaded"))

    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    ### CONCATENATE THE RESULT
    t0 = time.time()
    print('Concatenating vectors ...')
    if not tosparse: A = np.vstack(A)
    else: A = sparse.vstack(A)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    ### PRINT FEW INFO
    print('--------------------------------------------------------')
    if tosparse:       print('Sparsity:    ', '{:.1e}'.format(A.size/(A.shape[0]*A.shape[1])))
    if tosparse:       print('Memory usage (scipy.csr_array):', '{:.3f}'.format((A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)/1024**3), 'GB', '(np.array usage:', '{:.3f}'.format(A.shape[0]*A.shape[1]*4/1024**3), 'GB)')
    elif not tosparse: print('Memory usage (numpy.array):', '{:.3f}'.format(A.nbytes/1024**3), 'GB')
    print('--------------------------------------------------------')

    return A


#######################################
######## LOAD E4M SPARSE ARRAY ########
#######################################

def Nf_sparse_e4m(raw_folder, sample_name, Ndataset, Nscan):

    #### E4M DATA FOLDER
    h5_folder = f"{raw_folder}/{sample_name}/{sample_name}_{str(Ndataset).zfill(len_dataset_string)}/scan{str(Nscan).zfill(len_scan_string)}/"

    #### GET THE INDEX (number) OF THE FILES THAT SHOULD BE LOADED
    file_f = len([i for i in os.listdir(h5_folder) if i.startswith(sparse_eiger4m_filename)]) 

    import hdf5plugin  # needed for parallel loop context

    # LOAD H5 FILE
    h5file = h5py.File(f"{h5_folder}{sparse_eiger4m_filename}{str(file_f-1).zfill(len_fileidx_string)}.h5", 'r')

    # LOAD DATA
    frame_ptr = h5file['entry_0000']['measurement']['data']['frame_ptr']

    return (file_f-1)*Nfmax_sparse_file_e4m + frame_ptr.shape[0] - 1



def load_sparse_e4m(raw_folder, sample_name, Ndataset, Nscan, Nfi=None, Nff=None, load_mask=None, n_jobs=-1, count_th=None, verbose=True):

    t0 = time.time()

    # Detect if load_mask is a list/tuple/iterable of masks or a single mask
    is_mask_list = isinstance(load_mask, (list, tuple))
    
    if load_mask is None:
        mask_list = [None]
    elif is_mask_list:
        mask_list = load_mask
    else:
        mask_list = [load_mask]

    #### E4M DATA FOLDER
    h5_folder = f"{raw_folder}/{sample_name}/{sample_name}_{str(Ndataset).zfill(len_dataset_string)}/scan{str(Nscan).zfill(len_scan_string)}/"
    if verbose: print(f"Loading HDF5 files from {h5_folder}{sparse_eiger4m_filename}{'X'*len_fileidx_string}.h5 ...")  

    #### GET THE INDEX (number) OF THE FILES THAT SHOULD BE LOADED
    if Nfi is None: 
        file_i = 0 
    else: 
        file_i = Nfi // Nfmax_sparse_file_e4m
        
    if Nff is None: 
        file_f = len([i for i in os.listdir(h5_folder) if i.startswith(sparse_eiger4m_filename)]) 
    else: 
        file_f = Nff // Nfmax_sparse_file_e4m + 1

    total_files = file_f - file_i

    ### LOAD FRAMES BY FILE FUNCTION
    def _load_framesbyfile(i):
            import hdf5plugin  # needed for parallel loop context
            
            # LOAD H5 FILE
            h5file = h5py.File(f"{h5_folder}{sparse_eiger4m_filename}{str(i).zfill(len_fileidx_string)}.h5", 'r')

            # LOAD DATA
            a, b = None, None
            if (file_i == i) and (Nfi is not None): 
                a = Nfi % Nfmax_sparse_file_e4m
            if (file_f == i + 1) and (Nff is not None): 
                b = Nff % Nfmax_sparse_file_e4m + 1
            
            frame_ptr = h5file['entry_0000']['measurement']['data']['frame_ptr'][a:b]
            index =     h5file['entry_0000']['measurement']['data']['index'][frame_ptr[0]:frame_ptr[-1]]
            intensity = h5file['entry_0000']['measurement']['data']['intensity'][frame_ptr[0]:frame_ptr[-1]]
            
            if (file_i == i) and (Nfi is not None): 
                frame_ptr = frame_ptr - frame_ptr[0]
                
            out = sparse.csr_array((intensity, index, frame_ptr), shape=(frame_ptr.shape[0] - 1, Npx_e4m), dtype=np.float32)

            
            # Apply each mask in the mask list
            out_masked = [out[:, m] if m is not None else out for m in mask_list]

            # eliminates counts above count_th if count_th is not None
            if count_th is not None:
                for idx, m in enumerate(mask_list):
                    if m is not None:
                        out_masked[idx] = out_masked[idx] - out_masked[idx]*(out_masked[idx] > count_th)
                        out_masked[idx].eliminate_zeros()

            return out_masked

    # 1. Define distinct job generator
    jobs = (delayed(_load_framesbyfile)(i) for i in range(file_i, file_f))

    # 2. Parallel loop with generator and tqdm progress bar
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_gen = parallel(jobs)
        if verbose: 
            file_results = list(tqdm(results_gen, total=total_files, desc="Files loaded"))
        else:
            file_results = list(results_gen)
    
    if verbose: print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### CONCATENATE THE RESULT
    t0 = time.time()
    if verbose: print('Concatenating vectors ...')
    
    # Reorganize file_results from [file_idx][mask_idx] -> [mask_idx][file_idx]
    sA_list = []
    for m_idx in range(len(mask_list)):
        mask_files = [res[m_idx] for res in file_results]
        sA_list.append(sparse.vstack(mask_files))
        # Release references to raw files for this mask immediately
        for res in file_results:
            res[m_idx] = None
    del file_results
    gc.collect()
        
    if verbose: print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### PRINT FEW INFO
    if verbose: 
        print('\t | Sparse array loaded from', h5_folder)
        for idx, sA in enumerate(sA_list):
            m_str = f" [Mask #{idx+1}]" if is_mask_list else ""
            print(f'\t | Shape{m_str}:      ', sA.shape)
            if sA.shape[0] * sA.shape[1] > 0:
                print(f'\t | Sparsity{m_str}:   ', '{:.1e}'.format(sA.size / (sA.shape[0] * sA.shape[1])))
                print(f'\t | Memory usage{m_str}:', '{:.3f}'.format((sA.data.nbytes + sA.indices.nbytes + sA.indptr.nbytes) / 1024**3), 'GB')

    # Return list of sparse arrays if load_mask was a list, otherwise return single sparse array
    return sA_list if is_mask_list else sA_list[0]



#########################################
######## LOAD pitec SPARSE ARRAY ########
#########################################

def load_sparse_pitec(raw_folder, sample_name, Nscan, Nfi=None, Nff=None, load_mask=None, n_jobs=-1):

    t0 = time.time()

    # Detect if load_mask is a list/tuple/iterable of masks or a single mask
    is_mask_list = isinstance(load_mask, (list, tuple))
    
    if load_mask is None:
        mask_list = [None]
    elif is_mask_list:
        mask_list = load_mask
    else:
        mask_list = [load_mask]

    #### PITEC DATA FOLDER
    h5_folder = f"{raw_folder}/{sample_name}/run_{str(Nscan).zfill(4)}/"
    print(f"Loading HDF5 files from {h5_folder}{sparse_pitec_filename}{'X'*len_fileidx_string}.h5 ...")  

    #### GET THE INDEX (number) OF THE FILES THAT SHOULD BE LOADED
    if Nfi is None: 
        file_i = 0 
    else: 
        file_i = Nfi // Nfmax_sparse_file_pitec
        
    if Nff is None: 
        file_f = len([i for i in os.listdir(h5_folder) if i.startswith(sparse_pitec_filename)]) 
    else: 
        file_f = Nff // Nfmax_sparse_file_pitec + 1

    total_files = file_f - file_i

    ### LOAD FRAMES BY FILE FUNCTION
    def _load_framesbyfile(i):
            import hdf5plugin # needed for parallel loop context

            # LOAD H5 FILE
            if (i==file_f) and Nff is not None:
                try:
                    h5file = h5py.File(f"{h5_folder}{sparse_pitec_filename}{str(i).zfill(len_fileidx_string)}.h5", 'r')
                except:
                    return 999
            else:
                h5file = h5py.File(f"{h5_folder}{sparse_pitec_filename}{str(i).zfill(len_fileidx_string)}.h5", 'r')

            # LOAD DATA
            a, b = None, None
            if (file_i == i) and (Nfi is not None): 
                a = Nfi % Nfmax_sparse_file_pitec
            if (file_f == i + 1) and (Nff is not None): 
                b = Nff % Nfmax_sparse_file_pitec + 1
            
            frame_ptr = h5file['entry_0000']['measurement']['data']['frame_ptr'][a:b]
            index =     h5file['entry_0000']['measurement']['data']['index'][frame_ptr[0]:frame_ptr[-1]]
            intensity = h5file['entry_0000']['measurement']['data']['intensity'][frame_ptr[0]:frame_ptr[-1]]
            
            if (file_i == i) and (Nfi is not None): 
                frame_ptr = frame_ptr - frame_ptr[0]
                
            out = sparse.csr_array((intensity, index, frame_ptr), shape=(frame_ptr.shape[0] - 1, Npx_pitec), dtype=np.float32)
            
            # Apply each mask in the mask list
            out_masked = [out[:, m] if m is not None else out for m in mask_list]
            return out_masked

    # 1. Define distinct job generator
    jobs = (delayed(_load_framesbyfile)(i) for i in range(file_i, file_f))

    # 2. Parallel loop with generator and tqdm progress bar
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_gen = parallel(jobs)
        file_results = list(tqdm(results_gen, total=total_files, desc="Files loaded"))

    if file_results[-1] == 999:
        file_results = file_results[:-1]
        print('Last file skipped !!!')
    
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### CONCATENATE THE RESULT
    t0 = time.time()
    print('Concatenating vectors ...')

    # Reorganize file_results from [file_idx][mask_idx] -> [mask_idx][file_idx]
    sA_list = []
    for m_idx in range(len(mask_list)):
        mask_files = [res[m_idx] for res in file_results]
        sA_list.append(sparse.vstack(mask_files))
        
    print('Done! (elapsed time =', round(time.time() - t0, 2), 's)')

    ### PRINT FEW INFO
    print('\t | Sparse array loaded from', h5_folder)
    for idx, sA in enumerate(sA_list):
        m_str = f" [Mask #{idx+1}]" if is_mask_list else ""
        print(f'\t | Shape{m_str}:      ', sA.shape)
        print(f'\t | Sparsity{m_str}:   ', '{:.1e}'.format(sA.size / (sA.shape[0] * sA.shape[1])))
        print(f'\t | Memory usage{m_str}:', '{:.3f}'.format((sA.data.nbytes + sA.indices.nbytes + sA.indptr.nbytes) / 1024**3), 'GB')

    # Return list of sparse arrays if load_mask was a list, otherwise return single sparse array
    return sA_list if is_mask_list else sA_list[0]


#############################
###### LOAD PILATUS #########
#############################

def load_pilatus(raw_folder, sample_name, Ndataset, Nscan, Nfi=None, Nff=None, Nstep=None):
    '''
    Load pilatus images from h5 file.
    
    Work in progress
    1) maybe, better to do not relay on the master file, but directly look at the scan folder? But pilatus files are few, so maybe is fine to load them from the master file.

    Parameters
    ----------
    raw_folder: string
        path to raw data folder
    sample_name: string
        name of the sample
    Ndataset: int
        number of the dataset
    Nscan: int
        number of the scan
    Nfi: int
        number of the first image
    Nff: int
        number of the last image
    Lbin: int
        binning factor (default=None)

    Returns
    -------
    pilatus: dict
        dictionary with pilatus images
    '''
    # Default values
    if Nstep==None: Nstep = 1

    # LOAD H5 FILE
    h5file = h5py.File(raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'.h5', 'r')[str(Nscan)+'.1']

    # LOAD PILATUS IMAGES
    pilatus_data = h5file['measurement']['pilatus300k'][Nfi:Nff:Nstep]

    return pilatus_data