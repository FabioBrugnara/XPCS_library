import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pyFAI
import time
from tqdm import tqdm




############################
######## COMPUTE SQ ########
############################

def get_Sq(pilatus_data, ponifile, mask, npt=1024, print_ponifile=False):
    """
    Perform azimuthal integration on a stack of 2D detector images to obtain the 1D scattering profile S(q).
    
    Parameters
    ----------
    pilatus_data : np.ndarray
        3D array of detector images with shape (n_frames, height, width).
    ponifile : str
        Path to the pyFAI calibration (.poni) file containing detector geometry.
    mask : np.ndarray
        2D boolean array with the same shape as a single detector image, where True values are masked (ignored).
    npt : int, optional
        Number of points in the resulting 1D q profile (default is 1024).
    print_ponifile : bool, optional
        If True, prints the calibration parameters loaded from the poni file (default is False).
    
    Returns
    -------
    Q : np.ndarray
        1D array of q values (momentum transfer) in Å⁻¹.
    azav : np.ndarray
        2D array of azimuthally averaged intensities with shape (n_frames, npt).
    dazaf : np.ndarray
        2D array of estimated errors for the azimuthally averaged intensities with shape (n_frames, npt).
    
    Notes
    -----
    This function uses pyFAI for azimuthal integration and assumes Poisson statistics for error estimation.
    """

    ai = pyFAI.load(ponifile)   # Load the calibration file

    if print_ponifile:          # Print the calibration parameters
        print(f'Calibration parameters (from \'{ponifile}\'):')
        print('-----------------------------------------------------------------------------------------------')
        print(ai)
        print('-----------------------------------------------------------------------------------------------\n')
    
    t0 = time.time()
    print('Computing azimuthal integration...')
    if len(pilatus_data.shape) == 3:  # Check if the input is a stack of images
        azav, dazaf  = np.zeros((pilatus_data.shape[0], npt)), np.zeros((pilatus_data.shape[0], npt))  # Initialize arrays for azimuthal integration and errors
        for f in tqdm(range(pilatus_data.shape[0])):
            Q, azav[f], dazaf[f] = ai.integrate1d(data=pilatus_data[f], npt=npt, mask=mask, polarization_factor=-1, unit="q_A^-1", error_model="poisson") # Perform azimuthal integration
    else:
        Q, azav, dazaf  = ai.integrate1d(data=pilatus_data, npt=npt, mask=mask, polarization_factor=-1, unit="q_A^-1", error_model="poisson") # Perform azimuthal integration for single frame
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    return Q, azav, dazaf


#########################
######## PLOT SQ ########
#########################
def plot_time_Sq(q, Sq, dSq=None, itime=None, cmap=cm.copper,lw=2, alpha=0.7, xlims=None, ylims=None):
    """
    Plot the static structure factor S(Q) as a function of Q for multiple datasets.
    
    Parameters
    ----------
    q : array-like
        1D array of Q values (momentum transfer) in inverse angstroms [$\\AA^{-1}$].
    Sq : array-like
        2D array of S(Q) values with shape (n_curves, n_q), where each row corresponds to a dataset to plot.
    dSq : array-like, optional
        2D array of uncertainties for S(Q), same shape as Sq. If provided, error bars are shown.
    itime : array-like or None, optional
        Array of time values corresponding to each dataset, used for colorbar labeling. If None, colorbar is labeled as 'frame'.
    cmap : matplotlib colormap, optional
        Colormap to use for distinguishing datasets. Default is `cm.copper`.
    lw : float, optional
        Line width for the plots. Default is 2.
    alpha : float, optional
        Transparency for the plot lines. Default is 0.7.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cmap(np.linspace(0, 1, Sq.shape[0]))
    for i in range(len(Sq)):
        if dSq is None: ax.plot(q, Sq[i], color=colors[i], alpha=alpha, lw=lw)
        else: ax.errorbar(q, Sq[i], yerr=dSq[i], color=colors[i], alpha=alpha, lw=lw)
    ax.set_xlabel("Q [$\\AA^{-1}$]"); ax.set_ylabel("S(Q) [a.u.]")

    if itime is None: itime_4cbar=1
    else:             itime_4cbar = itime
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Sq.shape[0]*itime_4cbar)), ax=ax, pad=0.01)
    if itime is None: cbar.set_label  - h5py('frame')
    else:             cbar.set_label('t [s]')
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.tight_layout(); plt.show()

def plot_temperature_Sq(q, Sq, dSq=None, T_step=None, T_0 = None, cmap=cm.jet, cmap_label='T [°C]',lw=2, alpha=0.7, xlims=None, ylims=None):
    """
    Plot the static structure factor S(Q) as a function of Q for multiple datasets.
    
    Parameters
    ----------
    q : array-like
        1D array of Q values (momentum transfer) in inverse angstroms [$\\AA^{-1}$].
    Sq : array-like
        2D array of S(Q) values with shape (n_curves, n_q), where each row corresponds to a dataset to plot.
    dSq : array-like, optional
        2D array of uncertainties for S(Q), same shape as Sq. If provided, error bars are shown.
    itime : array-like or None, optional
        Array of time values corresponding to each dataset, used for colorbar labeling. If None, colorbar is labeled as 'frame'.
    cmap : matplotlib colormap, optional
        Colormap to use for distinguishing datasets. Default is `cm.copper`.
    lw : float, optional
        Line width for the plots. Default is 2.
    alpha : float, optional
        Transparency for the plot lines. Default is 0.7.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cmap(np.linspace(0, 1, Sq.shape[0]))
    for i in range(len(Sq)):
        if dSq is None: ax.plot(q, Sq[i], color=colors[i], alpha=alpha, lw=lw)
        else: ax.errorbar(q, Sq[i], yerr=dSq[i], color=colors[i], alpha=alpha, lw=lw)
    ax.set_xlabel("Q [$\\AA^{-1}$]"); ax.set_ylabel("S(Q) [a.u.]")

    if T_step is None: T_step_4cbar=1; cmap_label = 'frame'
    else:             T_step_4cbar = T_step
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T_0, vmax= T_0 + Sq.shape[0]*T_step_4cbar)), ax=ax, pad=0.01)
    if T_step is None: cbar.set_label  - h5py('frame')
    else:             cbar.set_label(cmap_label)
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.tight_layout(); plt.show()