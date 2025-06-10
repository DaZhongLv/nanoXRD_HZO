import h5py
import hdf5plugin
#
from scipy import ndimage
from scipy.optimize import curve_fit
#
import numpy as np
#
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
#
import os, sys
#
sys.path.append('./tools/')
import Toolbox as tl
import math_nem as ne
#

###


def RSM_Q(scan_path, plot = None):
    with h5py.File(scan_path, 'r') as wp:
        gamma = np.deg2rad((wp['entry/snapshots/pre_scan/gamma'][0])) # to return only scalars. [:] retuns an array
        delta = np.deg2rad((wp['entry/snapshots/pre_scan/delta'][0])) # gamma clockwise (positive) 
        gontheta = np.deg2rad((wp['entry/snapshots/pre_scan/gontheta'][0]))
        gonphi = np.deg2rad((wp['entry/snapshots/pre_scan/gontheta'][0]))
        #the beam has to be projected in the negative x-axis
        energy = wp['entry/snapshots/pre_scan/energy'][0] # in eV
        radius = (wp['/entry/snapshots/post_scan/radius'][0])*1e-3 #from mm to m
        #
        npixels = np.shape(wp['entry/measurement/eiger500k/frames'])[1:]
        pixel_size = 75*1e-6 #[m]
        xdi = np.arange(-npixels[1]/2, npixels[1]/2)*pixel_size
        ydi = np.arange(npixels[0]/2, -npixels[0]/2, -1)*pixel_size
        #
        YD, XD = np.meshgrid(ydi, xdi,indexing='ij')
        #
        distance = np.ones_like(XD)*radius        
        D = np.sqrt(XD**2 + YD**2 + distance**2)
        coord_det = np.array([XD / D, YD / D, distance/D]) # this is for angle set to zero. it needs to be rotated further
        #
        h, speed, ev_to_j  = 6.62e-34, 2.99e8, 1.60e-19  #j*s, m/s
        wl = h*speed/(ev_to_j*energy) # in meters
        #
        k = 2*np.pi/wl # in meters
        ki = np.array([0, 0, 1])
        ## detector ans sample rotation are independent
        Rot_ylab = np.array([
            [np.cos(gamma), 0, np.sin(gamma)],  # Rotation around the beam axis (cosd(0) = 1, sind(0) = 0)
            [0, 1, 0],
            [-np.sin(gamma), 0, np.cos(gamma)]
        ])
        #when you rotate the y-lab axis, is gamma that moves
        Rot_xlab = np.array([
            [1, 0 ,0],
            [0, np.cos(delta), -np.sin(delta)],
            [0, np.sin(delta), np.cos(delta)]
        ])
        ## sample_rotation
        Rot_sample_gontheta = np.array([
            [1, 0 ,0],
            [0, np.cos(gontheta), -np.sin(gontheta)],
            [0, np.sin(gontheta), np.cos(gontheta)]
        ])
        ###
        Rot_sample_gonphi = np.array([
            [np.cos(gonphi), 0, np.sin(gonphi)],  # Rotation around the beam axis (cosd(0) = 1, sind(0) = 0)
            [0, 1, 0],
            [-np.sin(gonphi), 0, np.cos(gonphi)]
        ])
        ###
        #when you rotate the x-lab axis, is delta that moves
        # Combine the rotations into a single matrix
        Rot_matrix = Rot_sample_gontheta.T @ Rot_sample_gonphi.T @ Rot_ylab.T @ Rot_xlab.T 
        #the transpose alters only where the negative values will be. 
        # I defined the system as left-handed
        ## now the calculation of a point in the detector considering the center point as (0,0). 
        ## this point will be rotated using the roration matrix defined before
        kf_rot = Rot_matrix @ coord_det.reshape(3, -1)
        # ki_rot = Rot_matrix @ ki
        # print(np.shape(ki), 'ki_rot')
        Q_lab = k*(kf_rot.reshape(3, *YD.shape) - ki[:, np.newaxis, np.newaxis])*1e-10
        intensity = wp['entry/measurement/eiger500k/frames'][:]
        #bolean masking
        intensity[(intensity > 1e7) | (intensity < 0)] = 0
        #
        intensity[np.isinf(intensity)] = 0
        intensity[np.isnan(intensity)] = 0
        #
        if plot == 'off':
            data = np.zeros_like(intensity[0])
            return Q_det, data  # Exit early if plotting is off
        
        if plot == 'max':
            max_frames = np.nanmax(intensity, axis=(1, 2)) #vectorized
            max_frame = intensity[np.argmax(max_frames)]
            data = max_frame#
        
        if plot == 'sum':
            sumQ = np.sum(intensity, axis = 0)
            data = sumQ
        
        fig, ax = plt.subplots(3, 1, figsize = (5, 5), layout = 'constrained')
        #
        print(np.shape(data), 'data')
        QyQx = ax[0].pcolormesh(Q_lab[0], Q_lab[1], data, cmap='terrain', norm = 'log',  shading='nearest') ##Qy vs Qx
        fig.colorbar(QyQx, ax = ax [0], label = "Intensity (a. u.)")
        ax[0].set_xlabel(r"$Q_{x}\; (Å^{-1})$") #\; is just for spacing
        ax[0].set_ylabel(r"$Q_{y}\; (Å^{-1})$")
        #
        QyQz = ax[1].pcolormesh(Q_lab[2], Q_lab[1], data, cmap='terrain', norm = 'log') ##Qy vs Qx
        fig.colorbar(QyQz, ax = ax [1], label = "Intensity (a. u.)")
        #fig.colorbar(QxQz, ax = ax[1], label = "Intensity (a. u.)")
        ax[1].set_ylabel(r"$Q_{y}\; (Å^{-1})$") #\; is just for spacing
        ax[1].set_xlabel(r"$Q_{z}\; (Å^{-1})$")
        #
        QzQx = ax[2].pcolormesh(Q_lab[0], Q_lab[2], data, cmap='terrain', norm = 'log') ##Qy vs Qx
        fig.colorbar(QzQx, ax = ax [2], label = "Intensity (a. u.)")
        #fig.colorbar(QxQz, ax = ax[1], label = "Intensity (a. u.)")
        ax[2].set_ylabel(r"$Q_{z}\; (Å^{-1})$") #\; is just for spacing
        ax[2].set_xlabel(r"$Q_{x}\; (Å^{-1})$")
        #
        short_path = "/".join(scan_path.split('/')[-2:])
        plt.suptitle(short_path, fontweight = 'bold')
        plt.show()
    return Q_lab,data

def ROI(yc, xc, h,w):
    '''ROI in reciprocal (detector) space. this functions is not meant for realspace images'''
    ROI_center = [yc, xc] # y and x 
    ROI_shape = [h, w]
    ROI = (slice(ROI_center[0]-ROI_shape[0]//2, ROI_center[0]+ROI_shape[0]//2), slice(ROI_center[1]-ROI_shape[1]//2, ROI_center[1]+ROI_shape[1]//2))
    return {"slice": ROI, "center": ROI_center, "shape": ROI_shape}

def preprocess(scan_path, lim = None, ROI = None):
    'processs all the data (frames, pixels_x, pixels_y) without reshaping'
    with h5py.File(scan_path, 'r') as fp:
        data = np.array(fp['entry/measurement/eiger500k/frames'][:, ROI['slice'][0], ROI['slice'][1]])
        #size0 = data.nbytes
        data[data > lim] = 0
        #data = np.log10(data.astype('float16'))
        data = np.log10(data)
        data[np.isinf(data)| np.isnan(data)] = 0
        #print(f'the data in {scan_path} is now at log10 scale')
        #size1 = data.nbytes
        #delta = 100 * size0/size1
        #print(f'original data reduced to{delta}%')
    return data


def scan_steps(scan_path):
    command = tl.get_command(scan_path)
    if command.split()[1] == "sy":
        flip_sx_sy = 1
        nvertical, step_vertical, nhorizontal, step_horizontal, scan_orientation = tl.get_scan_parameter(command.split())
    else:
        flip_sx_sy = 0
        nhorizontal, step_horizontal, nvertical, step_vertical, scan_orientation = tl.get_scan_parameter(command.split())
    return nhorizontal, step_horizontal, nvertical, step_vertical, scan_orientation


def plotscan(scan_path, ROI = None, norm = None, percentile = None, scale = None):
    with h5py.File(scan_path, 'r') as fp:
        data = fp['entry/measurement/eiger500k/frames'][:] #faster than np.array
        #norm_factor = np.mean(fp['entry/measurement/alba2/1'][:])
        data[data > 1e7] = 0 #masking faster than np.where
        frame_sum = np.sum(data, axis = 0)
    ####
    command = tl.get_command(scan_path)
    if command.split()[1] == "sy":
        flip_sx_sy = 1
        nvertical, step_vertical, nhorizontal, step_horizontal, scan_orientation = tl.get_scan_parameter(command.split())
    else:
        flip_sx_sy = 0
        nhorizontal, step_horizontal, nvertical, step_vertical, scan_orientation = tl.get_scan_parameter(command.split())
    ####
    fig, ax = plt.subplots(2, 1, layout = 'constrained')
    ax[0].imshow(np.log10(frame_sum), vmax = np.percentile(np.log10(frame_sum), percentile), cmap = 'terrain') # vmax = np.percentile(masked_data, 99.999))    
    #
    left, bottom, width, height = (ROI['center'][1] - ROI['shape'][1]/2, ROI['center'][0] - ROI['shape'][0]/2, ROI['shape'][1], ROI['shape'][0])
    rect = patches.Rectangle((left, bottom), width, height, linewidth=0.5, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].scatter(ROI['center'][1], ROI['center'][0], s = 10, marker = "x", color = "r")
    #
    data_roi = frame_sum[ROI['slice']]
    croped_data = data[:, ROI['slice'][0], ROI['slice'][1]]
    ####
    XRD_4dmaps = croped_data.reshape(nvertical, nhorizontal, np.shape(croped_data)[1], np.shape(croped_data)[-1])
    integrated_data_xrd =(np.sum(XRD_4dmaps, axis=(2, 3)))
    #fig, ax = plt.subplots(dpi = 200)
    ext = [0, nhorizontal*step_horizontal, 0, nvertical*step_vertical] #(left, right, bottom, top) 
    
    if scale == 'normal':
        integrated_data_xrd = integrated_data_xrd
    elif scale == 'log10':
        integrated_data_xrd = np.log10(integrated_data_xrd)

    plot = ax[1].imshow(integrated_data_xrd, cmap = 'magma', 
                         vmin = np.percentile(integrated_data_xrd, 1), vmax = np.percentile(integrated_data_xrd, 95), 
                           extent = ext)
    ax[1].set_title("ROI integration")
    ax[1].set_ylabel(r"y ($\mu$m)")
    ax[1].set_xlabel(r"x ($\mu$m)")
    
    cax = fig.add_axes([0.8, 0.12, 0.02, 0.29])
    
    fig.colorbar(plot, cax = cax, orientation = "vertical", shrink = 1/2)
    ###
    short_path = "/".join(scan_path.split('/')[-2:])
    plt.suptitle(short_path, fontweight = "bold")
    plt.show()
    return XRD_4dmaps, ext, short_path

def RSM_to_map(scan_path, ROI = None):
    with h5py.File(scan_path, 'r') as fp:
        croped_data = fp['entry/measurement/eiger500k/frames'][:, ROI['slice'][0], ROI['slice'][1]]
        croped_data[croped_data > 1e8] = 0
    command = tl.get_command(scan_path)
    if command.split()[1] == "sy":
        flip_sx_sy = 1
        nvertical, step_vertical, nhorizontal, step_horizontal, scan_orientation = tl.get_scan_parameter(command.split())
    else:
        flip_sx_sy = 0
        nhorizontal, step_horizontal, nvertical, step_vertical, scan_orientation = tl.get_scan_parameter(command.split())
    XRD_4dmaps = croped_data.reshape(nvertical, nhorizontal, np.shape(croped_data)[1], np.shape(croped_data)[-1])
    ext = [0, nhorizontal*step_horizontal, 0, nvertical*step_vertical]
    return XRD_4dmaps, ext

def clean(data, reduce = None, lim = None):
    copy = np.copy(data)
    copy[copy > lim] = 0
    if reduce == 'y':
        copy = copy.astype('float16')
        print('data reduced to float16')
    copy[np.isinf(copy)| np.isnan(copy)] = 0
    return copy


def filter(array, kernel = None):
    kernel_size = kernel if kernel else 3
    if array.ndim == 4:
        return ndimage.median_filter(array, size=(1, 1, kernel_size, kernel_size))
    if array.ndim == 2:
        return ndimage.median_filter(array, size = kernel_size)

def RSM3D(Q_det, data):
    Qx = Q_det[0]
    Qy = Q_det[1]
    Qz = Q_det[2]
    #intensity = max_frame_clean  # Use your intensity or normalized version
    # Set up figure with subplots
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(221, projection='3d')  # 3D plot
    # 3D scatter plot
    scatter = ax1.scatter(Qx.flatten(), Qy.flatten(), Qz.flatten(),
                          c=data.flatten(), cmap='terrain', norm=LogNorm(), s=1)
    fig.colorbar(scatter, ax=ax1, label="Intensity (log scale)")
    ax1.set_xlabel(r"$Q_x\; (Å^{-1})$")
    ax1.set_ylabel(r"$Q_y\; (Å^{-1})$")
    ax1.set_zlabel(r"$Q_z\; (Å^{-1})$")
    ax1.set_title("3D Reciprocal Space")
    plt.tight_layout()
    plt.show()
    return

def get_COM(XRD_4dmaps, Q_det_ROI):
    dir1, dir2, _ , _ = np.shape(XRD_4dmaps)
    integrated_data_xrd = np.sum(XRD_4dmaps, axis = (0,1))
    #
    COM_x = np.zeros((dir1, dir2))
    COM_y = np.zeros((dir1, dir2))
    #COM_z = np.zeros((dir1, dir2))
    for i in np.arange(dir1):
        for j in np.arange(dir2):
            total_int = np.sum(XRD_4dmaps[i,j,:,:])
            COM_x[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_det_ROI[0])/total_int
            COM_y[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_det_ROI[1])/total_int
            COM_z[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_det_ROI[2])/total_int
    return COM_x, COM_y, COM_z

def RSMpixel(XRD_4dmaps, pixel = (None,None), lim = None, title = None):
    fig, ax = plt.subplots(1,2, figsize = (12,4))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    ax[0].imshow(np.sum(XRD_4dmaps, axis =(2,3)),aspect = 'auto')
    ax[0].scatter(pixel[1], pixel[0], marker = 'x', color = 'red')
    chosen = np.copy(XRD_4dmaps[pixel[0],pixel[1],:,:])
    chosen[chosen > lim] = 0
    chosen[chosen < 0] = 0
    chosen[np.isinf(chosen)| np.isnan(chosen)] = 0
    #
    RSM = ax[1].imshow((chosen), cmap = 'terrain')
    fig.colorbar(RSM, ax = ax[1])
    plt.suptitle(title, fontweight = 'bold')
    plt.show()
    return chosen

def id_RSM(XRD_4dmaps):
    summed = np.sum(XRD_4dmaps, axis = (2,3))
    id_max = np.unravel_index(np.argmax(summed, axis=None), summed.shape)
    id_min = np.unravel_index(np.argmin(summed, axis=None), summed.shape)
    print(id_max, 'max', id_min, 'min')
    return id_max, id_min

def gaussian_2d(x, y, amplitude, xo, yo, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - xo)**2 / (2 * sigma_x**2) + (y - yo)**2 / (2 * sigma_y**2)))

def multi_gaussian_2d(coords, *params):
    """Sum of multiple 2D Gaussians."""
    x, y = coords  # Unpack the coordinates
    Z = np.zeros_like(x, dtype=np.float32)  # Ensure Z is float32. The precision affects the fit
    n_gaussians = len(params) // 5  # 5 parameters per Gaussian (amplitude, xo, yo, sigma_x, sigma_y)
    for i in range(n_gaussians):
        amplitude = params[5 * i]
        xo = params[5 * i + 1]
        yo = params[5 * i + 2]
        sigma_x = params[5 * i + 3]
        sigma_y = params[5 * i + 4]
        Z += gaussian_2d(x, y, amplitude, xo, yo, sigma_x, sigma_y)
    return Z
    
def fit_2d_gaussians(data_frame, coords, guess, lower_bounds, upper_bounds):
    try:
        params_opt, _ = curve_fit(
            lambda coords, *params: multi_gaussian_2d(coords, *params),
            coords,
            data_frame.flat,
            p0 = guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev = 20000, method = 'trf',
        )

    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        params_opt = np.full(len(guess), np.nan)  # Return NaNs for failed fit
        r_squared = np.nan
        return params_opt, r_squared
    
    # residuals and R-squared
    residual = data_frame - multi_gaussian_2d(coords, *params_opt).reshape(data_frame.shape)
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((data_frame - np.mean(data_frame))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return params_opt, r_squared

def get_q_2D(scan):
    ''' this is the simplied cooridinate system, but for horizontal scaterring
        the berengue article is on vertical geometry
        this code is for 4D-SDXM. here we dont map q along rockig direction ''' 
    with h5py.File(scan, 'r') as h5:
        energy = h5['entry/snapshots/pre_scan/energy'][:] # in eV
        theta = h5['entry/snapshots/pre_scan/gamma'][:]/2
        radius = h5['entry/snapshots/pre_scan/radius'][:]* 1e-3 # from mm to m
        npixels = np.shape(h5['entry/measurement/eiger500k/frames'])[1:]
    pixel_size = 75*1e-6 #[m]
    #
    h, speed, ev_to_j  = 6.62e-34, 2.99e8, 1.60e-19  #j*s, m/s
    wavelength = h*speed/(ev_to_j*energy) # in meters
    #
    sintheta = np.sin(np.deg2rad(theta))
    costheta = np.cos(np.deg2rad(theta))
    k = 2 * np.pi /wavelength 
    G = 2 * k * sintheta
    #
    dq12 = k * pixel_size  / radius
    q1 = np.linspace(dq12 * npixels[0]/2, -dq12 * npixels[0]/2, npixels[0], -1)
    q2 = np.linspace(-dq12 * npixels[1]/2 + G/costheta, dq12 * npixels[1]/2 + G/costheta, npixels[1]) 
    #
    Q1, Q2 = np.meshgrid(q1,q2, indexing='ij')
    #
    Qv = Q1 
    Qh = Q2*costheta
    #
    q_values = [Qv, Qh]
    q_values = np.array(q_values)
    return q_values

def get_moments(XRD_4dmaps, Q_ROI = None):
    integrated_data_xrd = np.sum(XRD_4dmaps, axis = (0,1))
    ##
    dir_y, dir_x, _, _ = np.shape(XRD_4dmaps)
    ##
    COM_x = np.zeros((dir_y, dir_x))
    COM_y = np.zeros((dir_y, dir_x))
    #COM_z = np.zeros((dir_y, dir_x))
    ##
    var_x = np.zeros((dir_y, dir_x))
    var_y = np.zeros((dir_y, dir_x))
    #var_z = np.zeros((dir_y, dir_x))
    ##
    for i in np.arange(dir_y):
        for j in np.arange(dir_x):
            total_int = np.sum(XRD_4dmaps[i,j,:,:])
            COM_x[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_ROI[1])/total_int
            var_x[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*(Q_ROI[1] - COM_x[i,j])**2)/ total_int
            #
            COM_y[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_ROI[0])/total_int
            var_y[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*(Q_ROI[0] - COM_y[i,j])**2)/ total_int
            #
            #COM_z[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*Q_ROI[0])/total_int
            #var_z[i,j] = np.sum(XRD_4dmaps[i,j,:,:]*(Q_ROI[0] - COM_z[i,j])**2)/ total_int
            
    FWHM_x = 2.35482*np.sqrt(var_x)
    FWHM_y = 2.35482*np.sqrt(var_y)
  
    return COM_x, COM_y, FWHM_x, FWHM_y