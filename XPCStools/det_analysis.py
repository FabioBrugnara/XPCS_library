# XPCStools/masking.py

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from sparse_dot_mkl import dot_product_mkl

# Import configuration dictionary and helpers
from .config import config
from .utils import theta2Q


def get_Ip(data, load_mask=None, max_comp=False):
    total_px = config.Nx * config.Ny

    if load_mask is None:
        if data.shape[1] != total_px:
            raise ValueError(
                'Data are masked at loading! (Or you set wrong config["Nx"] and config["Ny"] values). '
                'Please provide a load_mask to compute the mean flux per pixel!'
            )
        load_mask = np.ones(total_px, dtype=bool)
    else:
        load_mask = load_mask.flatten()

    if data.shape[1] != load_mask.sum():
        raise ValueError('The load_mask does not match the data.shape[1]! Please check the mask dimensions!')

    # Calcolo della media Ip
    Ip = np.full(total_px, config.ofvalue, dtype=float)
    Ip[load_mask] = data.sum(axis=0) / (config.itime * data.shape[0])

    if not max_comp:
        return Ip.reshape(config.Nx, config.Ny)

    # Calcolo del massimo Ip_max (se richiesto)
    Ip_max = np.full(total_px, config.ofvalue, dtype=float)
    max_vals = data.max(axis=0)
    if hasattr(max_vals, "toarray"):
        max_vals = max_vals.toarray().ravel()
        
    Ip_max[load_mask] = max_vals

    return Ip.reshape(config.Nx, config.Ny), Ip_max.reshape(config.Nx, config.Ny)




def plot_Ip(Ip, I1=None, I2=None, mask_geom=None):
    '''
    Function that generates the flux map and its histogram. 
    Pixels equal to config.ofvalue are automatically treated as masked.
    '''
    Nx, Ny = config.Nx, config.Ny

    # Define valid pixels (those that are not ofvalue)
    valid_pixels = (Ip != config.ofvalue)

    # Print Information
    num_valid = valid_pixels.sum()
    
    if I2 is not None:
        count = (Ip[valid_pixels] > I2).sum()
        pct = (count / num_valid * 100) if num_valid > 0 else 0
        print(f"# pixels above upper threshold ({I2}) -> {count} pixels (of {num_valid} => {pct:.2f}%)")
        
    if I1 is not None:
        count = (Ip[valid_pixels] < I1).sum()
        pct = (count / num_valid * 100) if num_valid > 0 else 0
        print(f"# pixels below lower threshold ({I1}) -> {count} pixels (of {num_valid} => {pct:.2f}%)")
        
    print(f"# Absolute max value -> {Ip[valid_pixels].max()}\n")

    # --- FIGURE: IMAGE MAP & HISTOGRAM ---
    fig = plt.figure(figsize=(8, 13))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    # 1. Image Map (in alto)
    ax4 = fig.add_subplot(gs[0])
    vmin = I1 if I1 is not None else Ip[valid_pixels].min()
    vmax = I2 if I2 is not None else Ip[valid_pixels].max()
    
    Ip_grid = Ip.reshape(Nx, Ny).copy()
    Ip_grid[Ip_grid == config.ofvalue] = np.nan
    
    cmap = plt.colormaps.get_cmap('viridis').copy()
    cmap.set_bad(color='black')
    
    im = ax4.imshow(Ip_grid, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax4, label="Mean flux per pixel [ph/s/px]")
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')
    ax4.set_xlim(0, Ny)
    ax4.set_ylim(0, Nx)

    if config.X0 is not None and config.Y0 is not None:
        ax4.plot(config.Y0, config.X0, 'ro', markersize=10)

    if mask_geom is not None:
        for obj in mask_geom:
            geom_type = obj.get('geom')
            if geom_type == 'Circle':
                ax4.add_artist(plt.Circle((obj['Cy'], obj['Cx']), obj['r'], color='r', fill=False))
            elif geom_type == 'Rectangle':
                ax4.add_artist(plt.Rectangle((obj['y0'], obj['x0']), obj['yl'], obj['xl'], color='r', fill=False))
            elif geom_type == 'Line':
                y0, y1, x0, x1 = obj['y0'], obj['y1'], obj['x0'], obj['x1']
                xx = np.array((y0, y1, 1))
                mm = (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else 0
                qq = x0 - y0 * mm
                lw_offset = int(obj.get('linewidth', 1) / 2)
                ax4.plot(xx, mm * xx + qq + lw_offset, color='r')
                ax4.plot(xx, mm * xx + qq - lw_offset, color='r')

    # 2. Histogram (in basso)
    r_min = max(0, I1 - I1 * 0.5) if I1 is not None else Ip[valid_pixels].min()
    r_max = I2 + I2 * 0.5 if I2 is not None else Ip[valid_pixels].max()
    hist_range = (r_min, r_max)

    ax5 = fig.add_subplot(gs[1])
    ax5.hist(Ip[valid_pixels], bins=123, range=hist_range, label='pixels')
    if I2 is not None: ax5.axvline(I2, color='r', label='Threshold 2 (I2)')
    if I1 is not None: ax5.axvline(I1, color='g', label='Threshold 1 (I1)')
    ax5.set_yscale('log')
    ax5.set_xlabel("Mean flux per pixel [ph/s/px]")
    ax5.legend()

    plt.tight_layout()

    return fig, ax4, ax5




def get_mask(Ip=None, I2=None, I1=None, mask=None, mask_geom=None):
    '''
    Generate a boolean mask for the detector combining geometry and thresholds.
    Returns the boolean mask (True for valid pixels, False for masked).
    '''
    Nx, Ny = config.Nx, config.Ny
    total_px = Nx * Ny

    # 1. Generate or validate base mask
    if mask is None:
        mask = np.ones((Nx, Ny), dtype=bool)
    elif mask.ndim == 2:
        if mask.shape != (Nx, Ny):
            raise ValueError(f"Provided mask shape {mask.shape} does not match expected shape ({Nx}, {Ny}).")
        else:
            mask = mask.astype(bool)
    else:
        if mask.ndim ==1:
            if mask.size != total_px:
                raise ValueError(f"Provided mask size {mask.size} does not match expected size {total_px}.")
            else:
                mask = mask.astype(bool).reshape(Nx, Ny)

    # 2. Apply geometric masks if provided
    if mask_geom:
        X, Y = np.mgrid[:Nx, :Ny]
        for obj in mask_geom:
            geom_type = obj.get('geom')
            if geom_type == 'Circle':
                cond = (Y - obj['Cy'])**2 + (X - obj['Cx'])**2 <= obj['r']**2
                mask &= cond if obj['inside'] else ~cond
            elif geom_type == 'Rectangle':
                cond = (Y > obj['y0']) & (Y < obj['y0'] + obj['yl']) & (X > obj['x0']) & (X < obj['x0'] + obj['xl'])
                mask &= cond if obj['inside'] else ~cond
            elif geom_type == 'Line':
                mm = (obj['x1'] - obj['x0']) / (obj['y1'] - obj['y0']) if (obj['y1'] - obj['y0']) != 0 else 0
                qq = obj['x0'] - obj['y0'] * mm
                lw_half = int(obj.get('linewidth', 1) / 2)
                line_cond = (X > mm * Y + qq - lw_half) & (X < mm * Y + qq + lw_half)
                mask &= ~line_cond

    # 3. Filter using thresholds on Ip
    if Ip is not None:
        if Ip.ndim == 2:
            if Ip.shape != (Nx, Ny):
                raise ValueError(f"Provided Ip shape {Ip.shape} does not match expected shape ({Nx}, {Ny}).")
        elif Ip.ndim == 1:
            if Ip.size != total_px:
                raise ValueError(f"Provided Ip size {Ip.size} does not match expected size {total_px}.")
            else:
                Ip = Ip.reshape(Nx, Ny)

        if I2 is not None:
            mask &= (Ip <= I2)
        if I1 is not None:
            mask &= (Ip >= I1)

        # Handle ofvalue automatically if present in Ip
        mask &= (Ip != config.ofvalue)

    # 4. Print masked percentage
    masked_pct = (1.0 - mask.sum() / total_px) * 100
    print(f"# Masked area: {masked_pct:.2f}% ({mask.sum()} valid pixels of {total_px})\n")

    return mask



def get_Qmask(theta, Q, dq, mask_plot=True, Qmap_plot=False, legend=True):
    '''
    Generate the Q masks (2D) for the given Q values at the working angle. 
    Optionally plots the Q mask and the Q map.
    '''
    Nx, Ny = config.Nx, config.Ny
    X0, Y0 = config.X0, config.Y0
    L = config.L
    lxp, lyp = config.lxp, config.lyp
    movement_axis = config.movement_axis

    # 1. GET THE X-Y MAPS
    X, Y = np.mgrid[:Nx, :Ny]

    # 2. COMPUTE THE Q MAP FOR THE GIVEN GEOMETRY AND ENERGY
    if movement_axis == 'Y':
        dY0 = L * np.tan(np.deg2rad(theta))
        dY0_map = np.sqrt(((X - X0) * lxp)**2 + (dY0 - (Y - Y0) * lyp)**2)
        theta_map = np.arctan(dY0_map / L)
    elif movement_axis == 'X':
        dX0 = L * np.tan(np.deg2rad(theta))
        dX0_map = np.sqrt(((dX0 - (X - X0) * lxp))**2 + ((Y - Y0) * lyp)**2)
        theta_map = np.arctan(dX0_map / L)
    elif movement_axis == '-Y':
        dY0 = L * np.tan(np.deg2rad(theta))
        dY0_map = np.sqrt(((X - X0) * lxp)**2 + ((-dY0 - (Y - Y0) * lyp))**2)
        theta_map = np.arctan(dY0_map / L)
    elif movement_axis == '-X':
        dX0 = L * np.tan(np.deg2rad(theta))
        dX0_map = np.sqrt(((-dX0 - (X - X0) * lxp))**2 + ((Y - Y0) * lyp)**2)
        theta_map = np.arctan(dX0_map / L)
        
    Q_map = theta2Q(config.Ei, np.rad2deg(theta_map))

    # 3. GET THE Q REGION IN 2D
    is_scalar_Q = isinstance(Q, (float, int, np.floating, np.integer))
    is_scalar_dq = isinstance(dq, (float, int, np.floating, np.integer))

    if is_scalar_Q:
        Qmask = (np.abs(Q_map - Q) < dq)
    else:
        Qmask = {}
        for i, q_val in enumerate(Q):
            if is_scalar_dq:
                Qmask[q_val] = (Q_map - q_val >= -dq) & (Q_map - q_val < dq)
            else:
                dq_val = dq[i]
                Qmask[q_val] = (q_val - dq_val <= Q_map) & (Q_map < q_val + dq_val)

    # 4. QMASK PLOT (if mask_plot=True)
    if mask_plot:
        fig_mask, ax_mask = plt.subplots(figsize=(8, 8))
        
        if is_scalar_Q:
            ax_mask.imshow(Qmask, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)
            ax_mask.scatter([], [], color=plt.cm.viridis(1.), label=f"{Q} $\\AA^{{-1}}$")
        else:
            Qmask2plot = np.zeros((Nx, Ny))
            s = 1.0 / len(Q)
            for i, q_val in enumerate(Qmask.keys()):
                Qmask2plot += Qmask[q_val] * s * (i + 1)
                ax_mask.scatter([], [], color=plt.cm.viridis(s * (i + 1)), label=f"{q_val} $\\AA^{{-1}}$")
            ax_mask.imshow(Qmask2plot, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)

        ax_mask.set_xlabel('Y [px]')
        ax_mask.set_ylabel('X [px]')
        if legend:
            ax_mask.legend()
        fig_mask.tight_layout()
        plt.show()

    # 5. QMAP PLOT (if Qmap_plot=True)
    if Qmap_plot:
        fig_qmap, ax_qmap = plt.subplots(figsize=(8, 8))
        im = ax_qmap.imshow(Q_map, cmap='viridis', origin='lower')
        fig_qmap.colorbar(im, ax=ax_qmap, label='Q [$\\AA^{-1}$]')
        ax_qmap.set_xlabel('Y [px]')
        ax_qmap.set_ylabel('X [px]')
        ax_qmap.set_title('Q Map')
        fig_qmap.tight_layout()
        plt.show()

    return Qmask


def bin_det(data, Npx_bin):
    
    Nx = config["Nx"]
    Ny = config["Ny"]

    BIN_Y = sparse.csr_array((np.ones(data.shape[1]), (np.arange(data.shape[1]), np.arange(data.shape[1])//Npx_bin)), dtype=np.float32)

    r1 = np.hstack([np.arange(Ny//Npx_bin)+i*Npx_bin*Ny//Npx_bin for i in range(0,Nx//Npx_bin)])
    R = np.hstack([r1+i*Ny//Npx_bin for i in range(Npx_bin)])

    C = np.hstack([np.arange(len(r1))]*Npx_bin)

    BIN_X = sparse.csr_array((np.ones(len(R)), (R, C)), dtype=np.float32)

    return dot_product_mkl(dot_product_mkl(data, BIN_Y), BIN_X)