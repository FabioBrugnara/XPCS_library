# XPCStools/masking.py

import matplotlib.pyplot as plt
import numpy as np

# Import configuration dictionary and helpers
from .config import config
from .utils import theta2Q


def get_Ip(data, load_mask=None, max_comp=False):

    itime = config["itime"]

    # Fetch parameters from config
    of_value4plot = config["of_value4plot"]
    total_px = config["Npx"]

    # CHECK data AND load_mask DIMENSION
    if (data.shape[1] != total_px) and (load_mask is None):            raise ValueError('Data are masked at loading! Please provide the load_mask!')
    if (load_mask is not None) and (data.shape[1] != load_mask.sum()): raise ValueError('The load_mask does not match the data.shape[1]! Please check the mask dimensions!')

    # COMPUTE THE MEAN FLUX PER PX [ph/s/px]
    Ip = np.ones(total_px) * of_value4plot
    if load_mask is None: Ip            = data.sum(axis=0)/(itime*data.shape[0])
    else:                 Ip[load_mask] = data.sum(axis=0)/(itime*data.shape[0])

    # COMPUTE THE MAXIMUM COUNTS PER PX [ph/px] (only if needed)
    if max_comp:
        Ip_max = np.ones(total_px) * of_value4plot
        if load_mask is None: Ip_max            = (data.max(axis=0)).toarray()
        else:                 Ip_max[load_mask] = (data.max(axis=0)).toarray()

    if not max_comp: return Ip
    else:            return Ip, Ip_max




def plot_Ip(Ip, Ip_max=None, mask=None, I1=None, I2=None, Imax1=None, Imax2=None, mask_geom=None):
    '''
    Function that generates different plots to create the mask! By default it generates the average flux per pixel map and histogram.
    '''
    # Fetch parameters from config
    Nx, Ny = config["Nx"], config["Ny"]
    of_value4plot = config["of_value4plot"]
    total_px = config["Npx"]

    # CHECK mask DIMENSION
    if mask is not None:
        if mask.size != total_px:
            raise ValueError("mask must have config['Npx'] elements!")

    if mask is None: mask = Ip != of_value4plot
    else:            mask = (Ip != of_value4plot) & mask

    # PRINT INFORMATIONS    
    print('################################################################################')
    if I2 is not None: print('# of pixels above I2 treshold -> ', Ip[mask][Ip[mask]>I2].shape[0], 'pixels (of', Ip.shape[0], '=>', round(Ip[mask][Ip[mask]>I2].shape[0]/Ip[mask].shape[0]*100, 2), '%)')
    if I1 is not None: print('# of pixels below I1 treshold -> ',   Ip[mask][Ip[mask]<I1].shape[0], 'pixels (of', Ip.shape[0], '=>', round(Ip[mask][Ip[mask]<I1].shape[0]/Ip[mask].shape[0]*100, 2), '%)')
    if (Ip_max is not None): print('Maximum count in the whole run ->', Ip_max.max())
    if (Ip_max is not None) and (Imax1 is not None): print('# of pixels with max below Imax1 treshold -> ', Ip_max[mask][Ip_max[mask]<Imax1].shape[0], 'pixels (of', Ip_max.shape[0], '=>', round(Ip_max[mask][Ip_max[mask]<Imax1].shape[0]/Ip_max.shape[0]*100, 2), '%)')
    if (Ip_max is not None) and (Imax2 is not None): print('# of pixels with max above Imax2 treshold -> ', Ip_max[mask][Ip_max[mask]>Imax2].shape[0], 'pixels (of', Ip_max.shape[0], '=>', round(Ip_max[mask][Ip_max[mask]>Imax2].shape[0]/Ip_max.shape[0]*100, 2), '%)')
    print('################################################################################\n')

    # MEAN FLUX PER PX FIGURE
    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    if I2 is None: vmax = Ip[mask].max()
    else:                vmax = I2
    if I1 is None:  vmin = Ip[mask].min()
    else:                vmin = I1
    im = ax4.imshow(Ip.reshape(Nx, Ny), vmin=vmin, vmax=vmax, origin='lower')  # plot the mean flux per px 
    plt.colorbar(im, ax=ax4, label='Mean flux per pixel [ph/s/px]')                                                                                            
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')
    ax4.set_xlim(0, Ny)
    ax4.set_ylim(0, Nx)  
    if (config["X0"] is not None) and (config["Y0"] is not None):
        ax4.plot(config["Y0"], config["X0"], 'ro', markersize=10) # plot the beam center

    if mask_geom is not None: # plot the mask geometry (mask_geom)
        for obj in mask_geom: # loop over the objects ...  
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

    # MEAN FLUX PER PX HISTOGRAM

    # Compute fallback limits only if needed
    d_min = Ip[mask].min() if (I1 is None and I2 is not None) else None
    d_max = Ip[mask].max() if (I2 is None and I1 is not None) else None

    r_min = I1 * 0.5 if I1 is not None else d_min
    r_max = I2 * 1.5 if I2 is not None else d_max

    # Pass tuple only if at least one bound was defined
    hist_range = (r_min, r_max) if (r_min is not None and r_max is not None) else None

    ax5 = plt.subplot(413)
    ax5.hist(Ip[mask], bins=200, range=hist_range, label='pixels')

    if I2 is not None: ax5.axvline(I2, color='r', label='I2') # plot the I2 limit
    if I1 is not None: ax5.axvline(I1,  color='g', label='I1') # plot the I1 limit
    ax5.set_yscale('log')                                                                                                                                  
    ax5.set_xlabel('Mean flux per pixel [ph/s/px]')                                                                                                      
    ax5.legend()                                                                                                                                           


    plt.tight_layout(); plt.show()

    # MAXIMUM COUNTS PER PX FIGURE
    if Ip_max is not None:

        if Imax2 is None: Imax2 = Ip_max[mask].max()

        plt.figure(figsize=(8,13))
        ax4 = plt.subplot(211)
        
        # MAX COUNTS PER PX IMAGE
        im = ax4.imshow(Ip_max.reshape(Nx, Ny), vmin=Imax1, vmax=Imax2, origin='lower')
        plt.colorbar(im, ax=ax4, label='Max counts per pixel [ph/px]')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # MAX COUNTS PER PX HISTOGRAM
        d_max = Ip_max[mask].max() if (Imax2 is None and Imax1 is not None) else None

        r_min = 0
        r_max = Imax2 * 1.5 if Imax2 is not None else d_max

        hist_range = (r_min, r_max) if (r_min is not None and r_max is not None) else None

        ax5 = plt.subplot(413)
        ax5.hist(Ip_max[mask], bins=200, range=hist_range, label='pixels')
        if Imax2 is not None: ax5.axvline(Imax2, color='r', label='Imax2') # plot the Imax2 limit
        if Imax1 is not None: ax5.axvline(Imax1, color='g', label='Imax1') # plot the Imax1 limit

        ax5.set_yscale('log')
        ax5.set_xlabel('Max counts per pixel [ph/px]')
        ax5.legend()


        plt.tight_layout()
        plt.show()




def get_mask(Ip=None, Ip_max=None, I2=None, I1=None, Imax1=None, Imax2=None, mask=None, mask_geom=None):
    '''
    Generate a mask for the e4m detector from various options. The function plot the so-obtained mask, and also return some histograms to look at the results (if hist_plots is True).
    '''
    Nx, Ny = config["Nx"], config["Ny"]
    total_px = config["Npx"]


    # GENERATE MASK OF ONES
    if mask is None:
        mask = np.ones(total_px, dtype=bool)
    else:
        if mask.size != total_px:
            raise ValueError("mask must have config['Npx'] elements!")
        mask = np.asarray(mask, dtype=bool).ravel()


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

    # FILTER USING THRESHOLDS
    if Ip is not None:
        if I2 is not None:
            mask &= Ip <= I2
        if I1 is not None:
            mask &= Ip >= I1

    if Ip_max is not None:
        if Imax2 is not None:
            mask &= Ip_max <= Imax2
        if Imax1 is not None:
            mask &= Ip_max >= Imax1

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

    return mask




def get_Qmask(theta, Q, dq, Qmap_plot=False):
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