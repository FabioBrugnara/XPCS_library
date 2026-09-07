# XPCStools/masking.py

import matplotlib.pyplot as plt
import numpy as np

# Import configuration dictionary and helpers
from .config import config
from .utils import theta2Q



def det_analysis(e4m_data, itime, Ith_high=None, Ith_low=None, Imaxth_high=None, mask=None, load_mask=None, mask_geom=None, Nfi=None, Nff=None, max_plots=False, wide_plots=False, plot_center=True):
    '''
    Function that generates a number of different plots to create the mask! By default it generates the average flux per pixel map and histogram.
    
    Parameters
    ----------
        e4m_data: sparse.csr_matrix
            Sparse matrix of the e4m detector data
        itime: float
            Integration time of the e4m detector
        Ith_high: float
            Threshold (above) for the mean photon flux of the pixels [ph/s/px]
        Ith_low: float
            Threshold (below) for the mean photon flux of the pixels [ph/s/px]
        Imaxth_high: float
            Maximum number of counts per pixel treshold [ph/px]
        mask: np.array
            If e4m_data.shape[1]==config["Npx"], is the mask to apply to the data for plotting and histogram generation.\n
            If e4m_data.shape[1]!=config["Npx"], mask is assumed is the same used to load the data, thus mask.sum()==e4m_data.shape[1]. In this case mask is used to plot the correct X-Y profile.
        mask_geom: list of dicts
            List of geometries to mask. The function just plot the geometries on top of the XY profile.
        Nfi: int
            First frame to consider
        Nff: int
            Last frame to consider
        max_plots: bool
            If True, plot the maximum counts per pixel map and histogram.
        wide_plots: bool
            If True, plot the wide histogram of the mean flux per pixel and maximum counts per pixel (if max_plots is True).
    '''
    # Fetch parameters from config
    Nx, Ny = config["Nx"], config["Ny"]
    X0, Y0 = config["X0"], config["Y0"]
    of_value4plot = config["of_value4plot"]
    total_px = config["Npx"]

    # CHECK e4m_data AND load_mask DIMENSION
    if (e4m_data.shape[1] != total_px) and (load_mask is None):     raise ValueError('Data are masked at loading! Please provide the load_mask!')
    if (load_mask is not None) and (mask is not None):         raise ValueError('Cannot apply mask to masked loaded data!')
    if (e4m_data.shape[1] == total_px) and (load_mask is not None): raise ValueError('Cannot use load_mask with already masked data!')
    if load_mask is not None:
        if (e4m_data.shape[1] != load_mask.sum()):             raise ValueError('The load_mask does not match the e4m_data.shape[1]! Please check the mask dimensions!')
    
    # LOAD DATA in Nfi:Nff
    e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK
    if mask is None: mask = np.ones(total_px, dtype=bool)

    # COMPUTE THE MEAN FLUX PER PX [ph/s/px]
    I_mean = np.ones(total_px) * of_value4plot
    if load_mask is None:
        I_mean[mask]      = e4m_data[:,mask].sum(axis=0)/(itime*e4m_data.shape[0])
    else:
        I_mean[load_mask] = e4m_data.sum(axis=0)        /(itime*e4m_data.shape[0])
        mask = load_mask

    # COMPUTE THE MAXIMUM COUNTS PER PX [ph/px] (only if needed)
    if (Imaxth_high is not None) or max_plots:
        I_max = np.ones(total_px) * of_value4plot
        if e4m_data.shape[1] == total_px: I_max[mask] = (e4m_data[:,mask].max(axis=0)).toarray()
        else:                        I_max[mask] = (e4m_data.max(axis=0)).toarray()

    # PRINT INFORMATIONS    
    print('################################################################################')
    print('Maximum count in the whole run ->', e4m_data.max())
    if Ith_high is not None: print('# of pixels above Ith_high treshold -> ', I_mean[mask][I_mean[mask]>Ith_high].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]>Ith_high].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Ith_low is not None: print('# of pixels below Ith_low treshold -> ',   I_mean[mask][I_mean[mask]<Ith_low].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]<Ith_low].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Imaxth_high is not None: print('# of pixels above Imaxth_high treshold -> ', I_max[mask][I_max[mask]>Imaxth_high].shape[0], 'pixels (of', I_max.shape[0], '=>', round(I_max[mask][I_max[mask]>Imaxth_high].shape[0]/I_max.shape[0]*100, 2), '%)')
    print('################################################################################\n')

    # MEAN FLUX PER PX FIGURE
    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    if Ith_high is None: vmax = I_mean[mask].max()
    else:                vmax = Ith_high
    if Ith_low is None:  vmin = I_mean[mask].min()
    else:                vmin = Ith_low
    im = ax4.imshow(I_mean.reshape(Nx, Ny), vmin=vmin, vmax=vmax, origin='lower')                                                                    # plot the mean flux per px 
    plt.colorbar(im, ax=ax4, label='Mean flux per pixel [ph/s/px]')                                                                                            
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')
    ax4.set_xlim(0, Ny)
    ax4.set_ylim(0, Nx)  
    if plot_center:
        ax4.plot(Y0, X0, 'ro', markersize=10)                                                                                                                   # plot the beam center
    if mask_geom is not None:                                                                                                                               # plot the mask geometry (mask_geom)
        for obj in mask_geom:                                                                                                                               # loop over the objects ...  
            if obj['geom'] == 'Circle':
                ax4.add_artist(plt.Circle((obj['Cy'], obj['Cx']), obj['r'], color='r', fill=False))
            elif obj['geom'] == 'Rectangle':
                ax4.add_artist(plt.Rectangle((obj['y0'], obj['x0']), obj['yl'], obj['xl'], color='r', fill=False))
            elif obj['geom'] == 'Line':
                xx = np.array((obj['y0'],obj['y1'],1))
                mm = (obj['x1']-obj['x0'])/(obj['y1']-obj['y0'])
                qq = obj['x0'] - obj['y0']*mm
                ax4.add_artist(plt.plot(xx, mm*xx+qq+int(obj['linewidth']/2), color='r')[0])
                ax4.add_artist(plt.plot(xx, mm*xx+qq-int(obj['linewidth']/2), color='r')[0])

    # MEAN FLUX PER PX HISTOGRAM (ZOOM)
    ax5 = plt.subplot(413)                                                                                                                                  # create the subplot
    if (Ith_high is not None) and (Ith_low is not None): ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, Ith_high*1.5),       label='pixels')           # plot the histogram
    elif (Ith_high is not None) and (Ith_low is None):   ax5.hist(I_mean[mask], bins=200, range=(0, Ith_high*1.5),                label='pixels')           # ..
    elif (Ith_high is None) and (Ith_low is not None):   ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, I_mean[mask].max()), label='pixels')           # ..
    else:                                                ax5.hist(I_mean[mask], bins=200,                                         label='pixels')     # ..
    if Ith_high is not None: ax5.axvline(Ith_high, color='r', label='Higher treshold')                                                                             # plot the Ith_high limit
    if Ith_low is not None:  ax5.axvline(Ith_low,  color='g', label='Lower treshold')                                                                              # plot the Ith_low limit
    ax5.set_yscale('log')                                                                                                                                   # add the labels and legend
    ax5.set_xlabel('Mean flux per pixel [ph/s/px]')                                                                                                          # ..   
    ax5.legend()                                                                                                                                            # ..        

    # MEAN FLUX PER PX HISTOGRAM (FULL RANGE)
    if wide_plots:
        ax6 = plt.subplot(414)                                                                                                                              # create the subplot
        ax6.hist(I_mean[mask], bins=200, label='(full range)')                                                                                              # plot the histogram    
        if Ith_high is not None: ax6.axvline(Ith_high, color='r', label='Higher treshold')
        if Ith_low  is not None: ax6.axvline(Ith_low,  color='g', label='Lower treshold')
        ax6.set_yscale('log')
        ax6.set_xlabel('Mean flux per pixel [ph/s/px]')
        ax6.legend()

    plt.tight_layout(); plt.show()

    # MAXIMUM COUNTS PER PX FIGURE
    if max_plots:
        plt.figure(figsize=(8,13))
        ax4 = plt.subplot(211)

        # MAX COUNTS PER PX IMAGE
        im = ax4.imshow(I_max.reshape(Nx, Ny), vmin=0, vmax=Imaxth_high, origin='lower')
        plt.colorbar(im, ax=ax4, label='Max counts per pixel [ph/px]')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # MAX COUNTS PER PX HISTOGRAM (ZOOM)
        ax5 = plt.subplot(413)
        if Imaxth_high is not None: 
            ax5.hist(I_max[mask], bins=100, label='pixels', range=(0, Imaxth_high*1.5))
            ax5.axvline(Imaxth_high, color='r', label='Higher max treshold')
        else:
            ax5.hist(I_max[mask], bins=100, label='pixels')

        # add labels and legend
        ax5.set_yscale('log')
        ax5.set_xlabel('Max counts per pixel [ph/px]')
        ax5.legend()

        # MAX COUNTS PER PX HISTOGRAM (FULL RANGE)
        if wide_plots:
            ax6 = plt.subplot(414)
            ax6.hist(I_max[mask], bins=200, label='pixels')
            ax6.set_yscale('log')
            ax6.set_xlabel('Max counts per pixel [ph/px]')
            ax6.legend()

        plt.tight_layout()
        plt.show()




def gen_mask(e4m_data=None, itime=None, mask=None, mask_geom=None, Ith_high=None, Ith_low=None, Imaxth_high=None, Nfi=None, Nff=None, hist_plots=False):
    '''
    Generate a mask for the e4m detector from various options. The function plot the so-obtained mask, and also return some histograms to look at the results (if hist_plots is True).

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    mask: np.array
        Mask of the e4m detector lines (slightly wider than the overflow lines, as pixels on the adges are not reliable)
    mask_geom: list of dicts
        List of geometries to mask (in dictionary form). The supported objects are:\n
        - Circle: {'geom': 'Circle', 'Cx': x0, 'Cy': y0, 'r': r, 'inside': True/False}\n
        - Rectangle: {'geom': 'Rectangle', 'x0': x0, 'y0': y0, 'xl': xl, 'yl': yl, 'inside': True/False}\n
        Example:\n
        mask_geom = [   {'geom': 'Circle', 'Cx': 100, 'Cy': 100, 'r': 10, 'inside': True}, {'geom': 'Rectangle', 'x0': 50, 'y0': 50, 'xl': 20, 'yl': 10, 'inside': False}]
    Ith_high: float
        Threshold (above) for the mean photon flux of the pixels
    Ith_low: float
        Threshold (below) for the mean photon flux of the pixels
    Imaxth_high: float
        Maximum number of counts per pixel treshold
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider  
    hist_plots: bool
        If True, plot the histograms of the mean flux per pixel and maximum counts per pixel.

    Returns
    -------
    np.array
        Mask of the e4m detector
    '''
    Nx, Ny = config["Nx"], config["Ny"]
    total_px = config["Npx"]

    # CHECK e4m_data DIMENSION & LOAD DATA in Nfi:Nff
    if e4m_data is not None: 
        if e4m_data.shape[1] != total_px: raise ValueError('Cannot generate a mask from already masked data!')
        e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK OF ONES (if mask is None)
    if mask is None: mask = np.ones(total_px, dtype=bool)

    # APPLAY GEOMETRIC MASKS (if mask_geom is not None)
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
            elif obj['geom']=='Line':
                    mm = (obj['x1']-obj['x0'])/(obj['y1']-obj['y0'])
                    qq = obj['x0'] - obj['y0']*mm
                    mask = mask * ~(((X>mm*Y+qq-int(obj['linewidth']/2))) & (X<mm*Y+qq+int(obj['linewidth']/2)))
        mask = mask.flatten()

    # FILTER USING THRESHOLDS (Ith_high, Ith_low, Imaxth_high) & AND COMPUTING I_mean, I_max (if needed)
    if (Ith_high is not None) or (Ith_low is not None) or (hist_plots==True):
        I_mean = np.array(e4m_data.sum(axis=0)/(itime*e4m_data.shape[0]))
        if Ith_high is not None: mask = mask * (I_mean<=Ith_high)
        if Ith_low is not None : mask = mask * (I_mean>=Ith_low)
    if (Imaxth_high!=None) or (hist_plots==True):
        I_max = np.array(e4m_data.max(axis=0))
        if Imaxth_high is not None : mask = mask * (I_max<=Imaxth_high)

    # PRINT PERCENTAGE OF MASKED PIXELS
    print('#################################################')
    print('Masked area = ', mask.sum()/total_px*100, '%')
    print('#################################################\n')

    # PLOT THE MASK
    plt.figure(figsize=(8,8))
    plt.imshow(mask.reshape((Nx, Ny)), origin='lower')
    plt.xlabel('Y [px]')
    plt.ylabel('X [px]')
    plt.tight_layout()
    plt.show()

    # PLOT THE HISTOGRAMS (if hist_plots is True)
    if hist_plots==True:
        plt.figure(figsize=(8,6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # Masked histogram of px flux
        ax1.set_title('Masked histogram of px flux')
        ax1.hist(I_mean[mask], bins=100)
        ax1.set_yscale('log')
        ax1.set_xlabel('Mean flux per px')

        # Maked histogram of max counts per px
        ax2.set_title('Masked histogram of max counts per px')
        ax2.hist(I_max[mask].data, bins=30, label='no zero counts')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.set_xlabel('Max counts per px')
        plt.tight_layout()
        plt.show()

    return mask




def gen_Qmask(theta, Q, dq, Qmap_plot=False):
    '''
    Generate the Q masks for the given Q values at the working angle. The function also plot the Qmap for the given energy and angle (if Qmap_plot is True).

    Parameters
    ----------
    theta: float
        Working angle in degrees
    Q: float or list of floats
        Q value(s) to mask in [1/A]
    dq: float or list of floats
        Q width(s) to mask in [1/A]
    Qmap_plot: bool
        If True, plot the Qmap for the given energy and angle

    Returns
    -------
    np.array or dict of np.array
        Q mask(s) of the e4m detector
    '''
    Nx, Ny = config["Nx"], config["Ny"]
    X0, Y0 = config["X0"], config["Y0"]
    L = config["L"]
    lxp, lyp = config["lxp"], config["lyp"]
    movement_axis = config["movement_axis"]

    # GET THE X-Y MAPS
    X, Y = np.mgrid[:Nx, :Ny]

    # COMPUTE THE Q MAP FOR THE GIVEN DETECTOR DISTANCE, DETECTOR POSITION, AND X-RAY ENERGY
    if movement_axis == 'Y':
        dY0 = L * np.tan(np.deg2rad(theta))
        dY0_map = np.sqrt(((X - X0) * lxp)**2 + (dY0 - (Y - Y0) * lyp)**2)
        theta_map = np.arctan(dY0_map / L)
    elif movement_axis == 'X':
        dX0 = L * np.tan(np.deg2rad(theta))
        dX0_map = np.sqrt(((dX0 - (X - X0) * lxp))**2 + ((Y - Y0) * lyp)**2)
        theta_map = np.arctan(dX0_map / L)
    Q_map = theta2Q(config["Ei"], np.rad2deg(theta_map))

    # GET THE Q REGION
    if (type(Q) == float) or (type(Q) == int):           Qmask       = (np.abs(Q_map-Q)<dq      ).flatten() # case of a single Q value
    else:
        Qmask = {}
        for i in range(len(Q)):
            if (type(dq) == float) or (type(dq) == int): Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq   ).flatten() # case of a list of Q values, single dq value
            else:                                        Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq[i]).flatten() # case of a list of Q values, list of dq values
    
    # QMASK PLOT
    plt.figure(figsize=(8,8))                                          
    if (type(Q) == float) or (type(Q) == int):                                                              # case of a single Q value
        plt.imshow(Qmask.reshape((Nx, Ny)), cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)
        plt.scatter([],[], color=plt.cm.viridis(1.), label=str(Q)+'$\\AA^{-1}$')
    else:                                                                                                   # case of a list of Q values
        Qmask2plot = 0
        s = 1/len(Q)
        for i, q in enumerate(Qmask.keys()):
            Qmask2plot += Qmask[q].reshape((Nx, Ny))*s*(i+1)
            plt.scatter([],[], color=plt.cm.viridis(s*(i+1)), label=str(Q[i])+'$\\AA^{-1}$')
        plt.imshow(Qmask2plot, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)

    plt.xlabel('Y [px]'); plt.ylabel('X [px]'); plt.legend()
    plt.tight_layout(); plt.show()

    # QMAP PLOT (if Qmap_plot=True)
    if Qmap_plot:
        plt.figure(figsize=(8,8))
        plt.imshow(Q_map, cmap='viridis', origin='lower')
        plt.colorbar(); plt.xlabel('Y [px]'); plt.ylabel('X [px]'); plt.title('Q [$\\AA^{-1}$]')
        plt.tight_layout(); plt.show()

    return Qmask