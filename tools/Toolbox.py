"""
Toolbox for the data analysis from the beamline Nanomax, Maxiv

@author: Dima
"""
import math
import os, sys
import hdf5plugin
import h5py
import numpy as np
import scipy.ndimage.measurements
import skimage.transform
from skimage.registration import phase_cross_correlation
import scipy.signal
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

from scipy.interpolate import NearestNDInterpolator

sys.path.append('./tool/')
import math_nem as nem

import scipy.io as scio
from tifffile import imsave

def cosd(input):
    """
    Calculate cosine for degrees value
    """
    return np.cos(np.radians(input))

def sind(input):
    """
    Calculate sine for degrees value
    """
    return np.sin(np.radians(input))

def tand(input):
    """
    Calculate tangent for degrees value
    """
    return np.tan(np.radians(input))


# alignment methods

## The fisrt two is just use for alignment of detector or the axis = -1, which is horizontal axis
def roll(im, pixels, roll_center=None):
    """
    roll the image to align the dataset
    """
    if roll_center is None:
        rolled_im = np.roll(im, pixels, axis=-1)
    else:
        # the masked pixels has a value of -1
        im = im.astype(float)
        im[np.where(im < 0)] = np.nan

        # approximate number of pixel per degree
        dist = np.sqrt(np.sum((np.array(roll_center) - np.array(im.shape)/2)**2))
        angle = pixels / dist / np.pi * 180
        rolled_im = skimage.transform.rotate(im, angle=-angle, roll_center=roll_center[::-1], mode='reflect')

        rolled_im[np.where(np.isnan(rolled_im))] = -1
    return rolled_im


def align_detector_rolls(data, roll_center):
    """
    To mitigate the influence of detector roll angle omega, which mainly happens in horizontal direction.
    move it to the center

    :param data: diffraction pattern
    :param roll_center: defalt as None
    :return rolled im and shift data.
    """
    shifts = np.zeros(len(data), dtype=np.int)
    ii, jj = np.indices(data[0].shape)
    mask = (data[0] > 0)

    for k in range(len(data)):
        com = np.sum(jj * data[k]* mask) / np.sum(data[k] * mask)
        try:
            shift = int(np.round(data.shape[-1]//2-com))
        except ValueError:
            print("not pre-aligned frame %d"%k)
            shift = 0
        data[k] = roll(data[k], shift, roll_center)
        shifts[k] = shift
    return data, shifts

## These two works for 2D frames, a reference is needed here

def calculate_shift(dataset, method, reference_position=None):
    """
    align the whole dataset with the first image of the data.

    :param dataset: whole dataset for the scan
    :param method: choose from 'pc' for phase_cross_correlation, 'com' for COM
    :param reference_position: defalt as the begin of the dataset, otherwise 'middle'
    :return: shift
    """
    shifts = []
    if method == 'pc':
        for k in range(dataset.shape[0]):
            # Shifts are (Z, Y, X): vertical,horizontal
            if reference_position is None:
                current_shift, error, diffphase = phase_cross_correlation(dataset[0, :, :], dataset[k, :, :])
                position_ = 0

            elif reference_position == 'highest':
                sum_data = np.sum(dataset, axis=(1, 2))
                peak_position = np.argmax(sum_data)
                position_ = peak_position

                current_shift, error, diffphase = phase_cross_correlation(dataset[peak_position, :, :], dataset[k, :, :])
            else:
                print("please select a reference position")
                raise ValueError

            shifts.append([int(current_shift[0]), int(current_shift[1])])
    elif method == 'com':
        coms = []
        for k in range(dataset.shape[0]):
            com_ = scipy.ndimage.measurements.center_of_mass(dataset[k])
            coms.append(com_)

        coms = np.array(coms)
        if reference_position is None:
            shifts = [coms[0, 0]-coms[:, 0], coms[0, 1]-coms[:, 1]]
            position_ = 0

        elif reference_position == 'highest':
            sum_data = np.sum(dataset, axis=(1,2))
            peak_position = np.argmax(sum_data)
            position_ = peak_position

            shifts = [coms[peak_position, 0]-coms[:, 0], coms[peak_position, 1]-coms[:, 1]]
        else:
            print("please select a reference position")

        shifts = np.transpose(shifts)
    else:
        print("invalid alignment method")
        raise ValueError

    shifts = np.round(shifts)
    return shifts, position_

def get_new_coordinate(nhorizontal, nvertical, shifts, ref_position=0):
    """
    This function calculates linear coordinates of the scan-points in the area of overlap between angular positions after
    calculation of the shift that has to be read from h5 dataset

    alarm: there is also a small possibility the common scan range is not continuous, then it might be aligned to a large area

    :param: ref_position, the frame num of reference point, defalt as the begining
    :param:input: size of the array, and shift array with shape (n_angles,2)
    :return: size of a new map (vertical, horizontal), list of scan point coordinates to be read from h5 dataset for each shift value
    """
    # Shifts are (Z, Y, X): vertical,horizontal
    shifts = np.array(shifts)
    nangles = len(shifts)

    hh = np.arange(0, nhorizontal)
    vv = np.arange(0, nvertical)
    h, v = np.meshgrid(hh, vv)

    # Add shift to every coordinate of the 2D scan
    shifted_coordinates = []
    for k in range(len(shifts)):
        tmp_ = []
        for j in range(nvertical):
            for i in range(nhorizontal):
                tmp_.append(str(v[j, i] + shifts[k, 0])+ ',' + str(h[j, i] + shifts[k, 1]))
        shifted_coordinates.append(tmp_)

    # Reference 2D map coordinates
    ref_position = int(ref_position)
    reference_coordinates = shifted_coordinates[ref_position].copy()

    # Find intersection, continuously searching with the whole scan
    for k in range(len(shifted_coordinates)):
        tmp_set = frozenset(shifted_coordinates[k])
        tmp_coordinates = [x.split(',') for x in reference_coordinates if x in tmp_set]
        reference_coordinates = []
        for i in range(len(tmp_coordinates)):
            reference_coordinates.append(tmp_coordinates[i][0]+','+ tmp_coordinates[i][1])

    common_coordinates = reference_coordinates

    if common_coordinates:
        # Find the position of common_coordinate in each frame
        data_coordinates = []
        for k in range(nangles):
            tmp_coordinates = []
            for coordinate in common_coordinates:
                tmp_coordinates.append(shifted_coordinates[k].index(coordinate))
            data_coordinates.append(tmp_coordinates)

        coordinate_list = []
        for k in range(len(common_coordinates)):
            tmp_coordinates = common_coordinates[k].split(',')
            coordinate_list.append([int(float(tmp_coordinates[0])), int(float(tmp_coordinates[1]))])
        coordinate_list = np.array(coordinate_list)

        # Size of the common scan range
        nvertical_new = np.max(coordinate_list[:,0]) - np.min(coordinate_list[:,0]) + 1
        nhorizontal_new = np.max(coordinate_list[:,1]) - np.min(coordinate_list[:,1]) + 1

    else:
        print('there is no overlap region among all frames, stop here')
        raise ValueError

    return data_coordinates, nvertical_new, nhorizontal_new

def roll_2D(im, shift):
    vertical_shift = shift[0]
    horizontal_shift = shift[1]

    rolled_im = im.copy()
    rolled_im = np.roll(rolled_im, -vertical_shift, axis=0)
    rolled_im = np.roll(rolled_im, -horizontal_shift, axis=1)
    return rolled_im


# Processing on the image

## bining methods
def binPixels_1D(arr, n=2):
    size = np.array(len(arr))
    size = size // n
    new = np.zeros(size, dtype=arr.dtype)

    for i in range(size):
        tmp = np.mean(arr[i * n: (i + 1) * n])
        if issubclass(arr.dtype.type, np.integer):
            new[i] = np.round(tmp)
        else:
            new[i] = tmp

    return new

def binPixels(im, n=2):
    """
    downsampling 
    :param im: image
    :param n: integer
    :return: downsampled_im
    """
    size = np.array(im.shape)
    size[0] = size[0] // n             #vertical 
    size[1] = size[1] // n             #hortizontal
    new = np.zeros(size, dtype=im.dtype)
    
    for k in range(size[0]):
        for i in range(size[1]):
            tmp = np.mean(im[k * n: (k + 1)* n], im[i * n : (i + 1) * n])
            if issubclass(im.dtype.type, np.integer):
                new[k, i] = np.round(tmp)
            else:
                new[k, i] = tmp
    return new

def fastBinPixels(im, n=2):
    """ Downsamples an image by stride-tricks downsampling. """
    strided = as_strided(im,
                shape=(im.shape[0]//n, im.shape[1]//n, n, n),
                strides=((im.strides[0]*n, im.strides[1]*n)+im.strides))
    return strided.sum(axis=-1).sum(axis=-1)

def gaussian2D(n, sigma):
    """
    Returns an n-by-n matrix containing a circular 2d gaussian with variance sigma**2 in pixels.
    :param n:
    :param sigma:
    :return:
    """
    gau = np.zeros(n, n)
    mu = (n - 1) / 2.0
    twosigma = float(2 * sigma**2)
    factor = 1 / (twosigma * np.pi)

    for i in range(n):
        for j in range(n):
            gau[i, j] = factor * np.exp(-((i - mu)**2 + (j - mu)**2) / twosigma)
            # the convolution darkens the image, not sure why, but this happens both with the scipy methods and
            # with a slow manual doulbe loop convolution... this number was obtained from a numerical test.
            gau[i, j] *= 100.0/77.9484
    return gau

def smoothImage(im, sigma):
    """
    Returns a smoothened copy of the input image, which is convolved by a gaussian of standard deviation sigma
    :param im: input image
    :param sigma: standard derivation of Gaussian distribuion
    :return: smoothed image
    """
    size = max(3, 3 * int(round(sigma)))
    gaussian = gaussian2D(size, sigma)
    smoothed_im = scipy.signal.fftconvolve(im, gaussian, mode='same')
    return smoothed_im

def poisson(mean, k):
    """
     Returns the normalized Poisson probability for observing k counts in a distribution described by the mean.
    :param mean:
    :param k:
    :return:
    """

    if type(k) in [list, tuple, np.ndarray]:
        result = []
        for k_ in k:
            result.append(mean**k_ * np.exp(-mean) / math.factorial(k_))
        return np.array(result)
    else:
        return mean**k * np.exp(-mean) / math.factorial(k)


def noisyImage(im, photonsPerPixel=None, photonsAtMax=None, photonsTotal=None, dtype=None):
    """
    Returns a noisy copy of the input image, with simulated
    photon-counting noise (Poisson noise) corresponding to:
    - an overall average number of photons per pixel, or
    - a certain number of photons expected in the maximum pixel, or
    - a certain number of total expected photons.

    You can specify the dtype of the output, by default it is the same
    as the input image.

    normally: photonsTotal=frame.sum() * photons_per_intensity
    photons_per_intensity = photons_in_central_frame / frames_ref[central].sum()
    for given photons_in_central_frame
    """

    if not dtype:
        dtype = im.dtype

    if photonsPerPixel and not photonsAtMax and not photonsTotal:
        photonsTotal = np.prod(im.shape) * photonsPerPixel
    elif photonsAtMax and not photonsPerPixel and not photonsTotal:
        photonsTotal = np.sum(im) / np.max(im) * photonsAtMax
    elif photonsTotal and not photonsAtMax and not photonsPerPixel:
        pass
    else:
        raise ValueError('Confusing input to noisyImage')

    result = np.zeros(im.shape, dtype=dtype)
    totalSum = im.sum()
    if len(im.shape) == 2:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                expected = im[i, j] / totalSum * photonsTotal
                result[i, j] = np.random.poisson(expected)
    if len(im.shape) == 3:
        for k in range(result.shape[0]):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    expected = im[k, i, j] / totalSum * photonsTotal
                    result[k, i, j] = np.random.poisson(expected)
    return result

def biggestBlob_detect(im):
    """ Takes an image and returns a version with only the biggest continuous blob of non-zero elements left."""
    label_im, N = scipy.ndimage.label(im)
    areas = []
    for i in range(1, N + 1): #N doesn't include the background
        areas.append(sum(sum(label_im == i)))
    biggest = np.where(areas == max(areas))[0] + 1
    return (label_im == biggest)


def blobs_detect(im):
    """ Takes an image and returns a version with only all the continuous blobs of non-zero elements left."""
    label_im, N = scipy.ndimage.label(im)
    areas = []
    for i in range(1, N + 1): #N doesn't include the background
        areas.append(sum(sum(label_im == i)))
    blobs = np.array(areas)
    return blobs

## create some pattern

def circle(n, radius, dtype = 'float'):
    """return an n by n array of zeros with a filled circle of ones in its center, defalt radius as n/2."""

    result = np.zeros((n, n),dtype=dtype)
    if not radius:
        radius = n / 2.0
    Y, X = result.shape
    for i in range(Y):
        for j in range(X):
            result[i,j] = int((i - (Y-1)/2.0 **2) + (j - (X-1))/2.0 **2 < radius **2)
    return result

def pseudoCircle(n, radius=None, exponent=1.5, dtype='float'):
    """ Returns an n-by-n array of zeros with a filled psuedo-circle of ones in its center,
    with default radius n/2. For exponent=1 this is a rhomb, for exponent=2 a circle, for high exponents a rounded-corner square."""

    result = np.zeros((n,n), dtype=dtype)
    if not radius:
        radius = n / 2.0
    Y, X = result.shape
    for i in range(Y):
        for j in range(X):
            result[i, j] = int(  np.abs(i-(Y-1)/2.0)**exponent + np.abs((j-(X-1)/2.0))**exponent < radius**exponent )
    return result
    
# load the data

def get_meta_data(path, **kwargs):
    """ load the data save in h5 file, return the meta data of the scan number"""
    h5file = h5py.File(path,'r')
    try:
        motor_positions = {
            # Detector positions
            "delta": h5file['entry']['snapshots']['post_scan']['delta'][()],
            "gamma": h5file['entry']['snapshots']['post_scan']['gamma'][()],
            "gonphi": h5file['entry']['snapshots']['post_scan']['gonphi'][()],
            "gontheta": h5file['entry']['snapshots']['post_scan']['gontheta'][()],
            "radius": h5file['entry']['snapshots']['post_scan']['radius'][()],
            "energy": h5file['entry']['snapshots']['post_scan']['energy'][()]
        }
    except:
        print('Some motor positions are not recorded')
    
    try:
        scan_position_x = h5file['entry']['measurement']['pseudo']['x'][()]
    except:
        print("-- Current dataset has no x lateral scanning, continue with single position")
        scan_position_x = []
    
    try:
        scan_position_y = h5file['entry']['measurement']['pseudo']['y'][()]
    except:
        print("-- Current dataset has no y lateral scanning, continue with single position")
        scan_position_y = []
    
    try:
        scan_position_z = h5file['entry']['measurement']['pseudo']['z'][()]
    except:
        print("-- Current dataset has no z lateral scanning, continue with single position")
        scan_position_z = []
    
    try:
        rocking_motor = "gonphi"
        rocking_angles = h5file['entry']['measurement'][rocking_motor][()]
        print("-- Rocking motor is %s --" % rocking_motor)
    except:
        try:
            rocking_motor = "gontheta"
            rocking_angles = h5file['entry']['measurement'][rocking_motor][()]
            print("-- Rocking motor is %s" % rocking_motor)
        except:
            print("-- No rocking motor positions, pass or specify it separately!")
            rocking_angles = []
            rocking_motor = []
            pass
    
    h5file.close()
    return motor_positions, rocking_motor, rocking_angles, scan_position_x, scan_position_y, scan_position_z

def get_braggmax_map(data, rc_ang):
    """
    Input the data after applying a mask and maybe cropped to roi, the shape should be ( pointlist , detector y, detector x), and return map of maximum intensity value
    :param data: the input diffraction data, it should be a map
    :return: Maximum Intensity Angles Map
    """
    
    int_vals =  data[0, :, :]
    bragg_map = np.zeros([data.shape[1],data.shape[2]])
    
    for i in range(data.shape[0]):        
        if i>0:
            im1 = data[i, :, :]
            im2 = int_vals
            bragg_map[im1>im2] = rc_ang[i]
            int_vals[im1>im2] = im1[im1>im2] 
        else:
            bragg_map[:,:] = rc_ang[0] 
    return bragg_map


def get_command(path, **kwargs):
    """ return the command for the specific scan """
    h5file = h5py.File(path, 'r')

    try:
        command = str(h5file['entry']['description'][()])[3:-2]
    except:
        raise NameError('Command is not found, please check the data with PyMca')

    return command

def get_scan_parameter(command):
    """ Parse the command, and return the number of points in both direction of the scan and also its range"""

    coordinate_to_direction = {
        'sx': 'h',
        'sy': 'v',
        'sz': 'b'}

    if command[0] == 'npointflyscan':
        fast_axis = command[1]
        fast_axis_range = [float(command[2]), float(command[3])]
        fast_axis_points = int(command[4]) + 1

        slow_axis = command[5]
        slow_axis_range = [float(command[6]), float(command[7])]
        slow_axis_points = int(command[8]) + 1

    else:
        print('it is not a map, please check the data with PyMca!')

    scan_orientation = coordinate_to_direction[fast_axis] + coordinate_to_direction[slow_axis]

    if scan_orientation == 'hv':
        nhorizontal = fast_axis_points
        nvertical = slow_axis_points
        step_horizontal = (fast_axis_range[-1] - fast_axis_range[0]) / (fast_axis_points - 1)
        step_vertical = (slow_axis_range[-1] - slow_axis_range[0]) / (slow_axis_points - 1)
    elif scan_orientation == 'vh':
        nhorizontal = slow_axis_points
        nvertical = fast_axis_points
        step_horizontal = (slow_axis_range[-1] - slow_axis_range[0]) / (slow_axis_points - 1)
        step_vertical = (fast_axis_range[-1] - fast_axis_range[0]) / (fast_axis_points - 1)

    return nhorizontal, step_horizontal, nvertical, step_vertical, scan_orientation

def load_data_xrf(path, roi=None, module = 3):
    """ roi is applied to the sepctrum, module 3 is as defalt """

    h5file = h5py.File(path, 'r')
    if roi:
        data = h5file['entry']['measurement']['xspress3']['data'][:,module,roi[0]:roi[1]]
        #data = np.sum(data, 1)  #sum up all the selected sepectrum
    else:
        data = h5file['entry']['measurement']['xspress3']['data'][:,module,:]
        print ('now this is the whole spectrum, please check the roi')
    h5file.close()

    return data

def get_incoming_intensity(path):
    """ get the incoming intensity for normalization"""

    h5file = h5py.File(path, 'r')
    try:
        incoming_intensity = h5file['entry']['measurement']['alba2']['1'][()]
    except:
        print("-- Normalization data is not found, consider manual normalization, continue without it")
        incoming_intensity = []

    h5file.close()

    return incoming_intensity

def load_data_xrd_eiger500k(path, roi=None, point_list=[]):
    """
    load the data from eiger500k detector
    :param path: the data path
    :param roi: roi in detector, should contain the interested diffraction pattern
    :param point_list: since the position information of map is listed as 1D data, this means the region in real map position
    :return: xrd map for selected peak
    """

    h5file = h5py.File(path, 'r')

    if roi != []:
        if point_list != []:
            data = h5file['entry']['measurement']['eiger500k']['frames'][point_list, roi[0]:roi[1], roi[2]:roi[3]]
        else:
            data = h5file['entry']['measurement']['eiger500k']['frames'][:, roi[0]:roi[1], roi[2]:roi[3]]
    else:
        data = h5file['entry']['measurement']['eiger500k']['frames'][()]
    ang = h5file['entry']['snapshots']['post_scan']['gonphi'][()]
    h5file.close()
    return data, ang

def load_mask(path, roi):
    """ the whole mask data could be found in the data file in maxiv server, the roi should be the same with xrd"""
    data = np.load(path)
    mask = data['mask'][roi[0]: roi[1], roi[2]:roi[3]]
    return mask


def get_COM_map(data):
    """
    Input the data after applying a mask and maybe cropped to roi, the shape should be ( pointlist , detector y, detector x), and return the COM map, which can somehow reveal the strain distribution
    :param data: the input diffraction data, it should be a map
    #:param direction: "vertical", "horizontal", "magnitude"
    :return: COM map
    """

    com = []
    for i in range(data.shape[0]):
        im = data[i, :, :]
        com_ = scipy.ndimage.measurements.center_of_mass(im)
        if np.any(np.isnan(com_)):
            com_ = (0, 0)
        com.append(com_)

    com_vert = com[:, 0] - np.mean(com[:, 0])
    com_hor = com[:, 1] - np.mean(com[:, 1])
    com_mag = np.sum((com - np.mean(com, axis=0))**2, axis = 1)

    return com_vert, com_hor, com_mag


def save_data(save_path, save_type, data):
    if save_type == 'mat':
        scio.savemat(save_path, {'scan': data}, do_compression=True, oned_as='column')
    elif save_type == 'npz':
        np.savez_compressed(save_path, data= data)
    elif save_type == 'tif':
        imsave(save_path, data)
    else:
        print('Unrecognizd save type!')


class IndexTracker:
    def __init__(self, ax, data, axis):
        self.ax = ax
        self.scroll_axis = axis
        ax.set_title('use scroll wheel to navigate images')
        self.data = data

        # Start slice to show
        self.ind = 0

        if self.scroll_axis == 0:
            self.slices, rows, cols = data.shape
            self.im = ax.imshow(self.data[self.ind, :, :])
        elif self.scroll_axis == 1:
            rows, self.slices, cols = data.shape
            self.im = ax.imshow(self.data[:, self.ind, :])
        elif self.scroll_axis == 2:
            rows, cols, self.slices = data.shape
            self.im = ax.imshow(self.data[:, :, self.ind])
        self.im.set_cmap('turbo')
        # plt.colorbar(self.im, self.ax)
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.key == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update(self):
        if self.scroll_axis == 0:
            self.im.set_data(self.data[self.ind, :, :])
        elif self.scroll_axis == 1:
            self.im.set_data(self.data[:, self.ind, :])
        elif self.scroll_axis == 2:
            self.im.set_data(self.data[:, :, self.ind])

        self.ax.set_title('Slice %d along axis %d ' % (self.ind, self.scroll_axis))
        self.im.set_cmap('turbo')
        self.im.axes.figure.canvas.draw()


def scroll_data(data, axis=None, colormap=None):
    fig, ax = plt.subplots(1, 1)

    if axis:
        tracker = IndexTracker(ax, data, axis)
    else:
        tracker = IndexTracker(ax, data, 0)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    if colormap:
        plt.set_cmap(colormap)
    else:
        plt.set_cmap('turbo')

    plt.show()

    return tracker

# use to analysis the strain map

def get_q_coordinates(setup, number_of_scans, roi_xrd=None, gonphi_correction=0, chi_correction=0):
    """
    Calculate q-space coordinates for each pixel in the rocking curve dataset

    """
    # Constants. They are needed for correct labeling of axes
    H = 4.1357e-15  # Plank's constant
    C = 2.99792458e8  # Speed of light in vacuum

    number_of_pixels_horizontal = setup.detector.nb_pixel_x
    number_of_pixels_vertical = setup.detector.nb_pixel_y
    # 3rd dimension: number_of_scans

    # Correction angles
    detector_delta_correction = 0
    detector_gamma_correction = 0

    # NanoMax convention:
    # gamma - horizontal detector
    # delta - vertical detector
    # gonphi - rotation about vertical axis
    # gontheta - rotation about horizontal axis
    radius = setup.distance
    photon_energy = setup.energy

    gonphi_increment = setup.tilt_angle  # [deg] - can be a range of angles
    gontheta = setup.sample_outofplane
    gonphi = setup.sample_inplane

    delta = setup.outofplane_angle + detector_delta_correction  # [deg] these angles are corrected with the sign respecting the rotation rules
    gamma = setup.inplane_angle + detector_gamma_correction  # [deg]

    if setup.detector.pixelsize_x != setup.detector.pixelsize_y:
        print('Pixels of the detector are not square! Not implemented, stop here.')
        raise ValueError
    else:
        detector_pitch = setup.detector.pixelsize_x  # [m]

    direct_beam = np.round([251.7859, 250.4288])  #merlin?
    direct_beam = np.round([512/2, 1028/2])  #eiger500k

    wavelength = H * C / photon_energy

    k = 2 * np.pi / wavelength  # wave vector

    # According to the gratting equation. or you can say it comes from |q| = 2|k|sintheta
    dq = k * 2 * np.arctan(detector_pitch / (2 * radius))  # q-space pitch at the detector plane

    hd, vd = np.meshgrid(np.arange(-number_of_pixels_horizontal / 2, number_of_pixels_horizontal / 2),
                         np.arange(number_of_pixels_vertical / 2, -number_of_pixels_vertical / 2, -1))

    # move the center to the beam position, now it is in real space, the detector plane
    hd = (hd + number_of_pixels_horizontal / 2 - direct_beam[1]) * detector_pitch;
    vd = (vd + number_of_pixels_vertical / 2 - direct_beam[0]) * detector_pitch;
    bd = np.ones(vd.shape) * radius  # bd for beam direction

    # Add functionality to image display
    # gu.imagesc(hd[0,:],vd[:,0],(np.sum(data,0)));

    # Data reduction
    # data = data[roi_xrd[0]:roi_xrd[1],roi_xrd[2]:roi_xrd[3],:]
    hd = hd[roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]
    vd = vd[roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]
    bd = bd[roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]

    number_of_pixels_horizontal = roi_xrd[3] - roi_xrd[2]
    number_of_pixels_vertical = roi_xrd[1] - roi_xrd[0]

    d = np.array([hd.flatten(), vd.flatten(), bd.flatten()])

    # the biggest range of the
    r = np.sqrt(np.sum(d ** 2, 0))

    # consider about the Edwald's sphere
    hq = k * (d[0, :] / r)
    vq = k * (d[1, :] / r)
    bq = k * (1 - d[2, :] / r)

    q = [hq, vq, bq]

    # Sample orientation matrix. Bounds the sample crystal with the laboratory frame
    # Angles alpha beta gamma were manually adjusted so that known peaks
    # are exactly in their places

    # X is horizontal, perp to the beam, Y is vertical

    Rh = np.array([[1, 0, 0],  # detector rotation around horizontal axis
                   [0, cosd(delta), -sind(delta)],
                   [0, sind(delta), cosd(delta)]])

    Rv = np.array([[cosd(gamma), 0, sind(gamma)],  # detector rotation around vertical axis
                   [0, 1, 0],
                   [-sind(gamma), 0, cosd(gamma)]])

    Rb = np.array([[cosd(0), -sind(0), 0],  # detector rotation around beam axis
                   [sind(0), cosd(0), 0],
                   [0, 0, 1]])


    U = Rh @ Rv @ Rb
    qR = (U @ q)  # correct so far in real space

    # Initial coordinate of ki
    ki = np.array([0, 0, k])
    kf = U @ ki
    Q = kf - ki

    # Lab coordinate system: accosiated with the ki
    QLab = [qR[0, :] + Q[0], qR[1, :] + Q[1], qR[2, :] + Q[2]]

    # Small corrections to misalignment of the sample
    # Here the rocking curve should be introduced
    # alpha
    # beta

    # Gonphi correction
    sample_alpha_correction = 0  # Qx+Qz

    # if gonphi_correction != None:
    sample_beta_correction = -gonphi_increment * number_of_scans / 2 + gonphi_correction  # Qz
    # else:
    #     sample_beta_correction = -gonphi_increment*number_of_scans/2

    sample_gamma_correction = chi_correction  # Qz

    dphi = 0.1

    q_values = []

    for ii in range(number_of_scans):
        # Rotations to bring the q vector into sample coordinate system
        Rsh = np.array([[1, 0, 0],  # detector rotation around horizintal axis
                        [0, cosd(gontheta + sample_alpha_correction),
                         -sind(gontheta + sample_alpha_correction)],
                        [0, sind(gontheta + sample_alpha_correction),
                         cosd(gontheta + sample_alpha_correction)]])

        Rsv = np.array([[cosd(-gamma / 2 + gonphi_increment * (ii - 1) + sample_beta_correction), 0,
                         sind(-gamma / 2 + gonphi_increment * (ii - 1) + sample_beta_correction)],
                        # detector rotation around vertical axis
                        [0, 1, 0],
                        [-sind(-gamma / 2 + gonphi_increment * (ii - 1) + sample_beta_correction), 0,
                         cosd(-gamma / 2 + gonphi_increment * (ii - 1) + sample_beta_correction)]])

        Rsb = np.array([[cosd(sample_gamma_correction), -sind(sample_gamma_correction), 0],
                        [sind(sample_gamma_correction), cosd(sample_gamma_correction), 0],
                        [0, 0, 1]])

        Rs = Rsh @ Rsv @ Rsb

        # Sample coordinate system: accosiated with the ki
        q_values.append(Rs @ QLab)

    q_values = np.array(q_values)

    return q_values

# need to be fixed, associated with the setup and also consider about the roi
def get_q_values_simple(setup, scan_number, roi_xrd):

    """
    from sanna, the reference paper is Berenguer's classic paper.
    :param psize: (gonphi_increment, psize_vertical, psize_horizontal)
    :param shape: (gonphi step, nvertical, nhorizontal )
    :param energy: incoming energy
    :param distance: distance from sample to detector
    :param theta_bragg: bragg angle
    :return: q_values in cartesian coordinate
    """
    
    #shape = [scan_number, 514, 514]
    shape = [scan_number, roi_xrd[1]-roi_xrd[0], roi_xrd[3]-roi_xrd[2]]
    theta_bragg = setup.inplane_angle/2.0
    distance = setup.distance
    energy = setup.energy /1000
    
    if setup.detector.pixelsize_x != setup.detector.pixelsize_y:
        print('Pixels of the detector are not square! Not implemented, stop here.')
        raise ValueError
    else:
        psizex = setup.detector.pixelsize_x # [m]
        psizey = setup.detector.pixelsize_y
    
    gonphi_increment = setup.tilt_angle
        
    wavelength = 1.23984E-9 / energy # energy in units of keV
    #wavelength = setup.wavelength

    sintheta = np.sin(np.deg2rad(theta_bragg))
    costheta = np.cos(np.deg2rad(theta_bragg))
    tantheta = np.tan(np.deg2rad(theta_bragg))


    dq1 = psizey * 2 * np.pi / distance / wavelength
    dq2 = psizex * 2 * np.pi / distance / wavelength
    # from d = |Q|= 2k * sintheta
    dq3 = np.deg2rad(gonphi_increment) * 4 * np.pi / wavelength * sintheta

    # from d = |Q|= 2k * sintheta
    Q_abs = 4 * np.pi / wavelength * sintheta

    # reference: Berenguer's paper. q1 vertical, q2 - horizonatal, q3 - beam axis
    q1 = np.linspace(-dq1 * shape[1] / 2. + Q_abs / costheta, dq1 * shape[1] / 2. + Q_abs / costheta,
                     shape[1])
    q3 = np.linspace(-dq3 * shape[0] / 2. + sintheta * q1.min(),
                     dq3 * shape[0] / 2. + sintheta * q1.max(), shape[0])
    q2 = np.linspace(-dq2 * shape[2] / 2., dq2 * shape[2] / 2., shape[2])

    # make a meshgrid of q3, q1, q2 and transform it to qx, qz, qy

    Q3, Q2, Q1 = np.meshgrid(q3, q2, q1, indexing='ij')

    # COM analysis is more exact if the measured data is transformed to a orthogonal grid
    # transform the Q-space grid to from experimental grid to orthoganal
    # go from natura to cartesian coordinate system
    Qv = costheta * Q1
    Qh = -Q2
    Qb = Q3 - sintheta * Q1
    
    #Qv = Qv[:, roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]
    #Qh = Qh[:, roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]
    #Qb = Qb[:, roi_xrd[0]:roi_xrd[1], roi_xrd[2]:roi_xrd[3]]
    
    q_values = [Qh, Qv, Qb]
    q_values = np.array(q_values)

    # q_values = [qh, qv, qb]

    return q_values

def intepolate_q_values(setup, data, q_values, scale_coefficient=1):
    
    if setup.detector.pixelsize_x == setup.detector.pixelsize_y:
        dqh = scale_coefficient * (2 * np.pi * setup.detector.pixelsize_x / (setup.distance * setup.wavelength))
        dqv = scale_coefficient * (2 * np.pi * setup.detector.pixelsize_x / (setup.distance * setup.wavelength))
        dqb = scale_coefficient * (2 * np.pi * setup.detector.pixelsize_x / (setup.distance * setup.wavelength))
    else:
        raise ValueError("the pixel size didn't match, please check!")
    
    shape_q = len(q_values.shape)
    print(shape_q)
    
    if shape_q == 3:

        qh = q_values[:, 0, :]
        qv = q_values[:, 1, :]
        qb = q_values[:, 2, :]
    else:
        qh = q_values[0]
        qv = q_values[1]
        qb = q_values[2]

    qb = qb.flatten()
    qv = qv.flatten()
    qh = qh.flatten()

    qrb, qrv, qrh = np.meshgrid(np.arange(min(qb), max(qb), dqb), \
                                np.arange(min(qv), max(qv), dqv), \
                                np.arange(min(qh), max(qh), dqh))
    # interpolate Q
    a = list(zip(qb, qv, qh))
    # a = np.transpose(a,(0,2,1))
    print(np.shape(a))
    interp = NearestNDInterpolator(a, data.flatten())
    data_interpolated = interp(qrb, qrv, qrh)
    # data_interpolated[math.isnan(data_interpolated)] = 0

    return qrh, qrv, qrb, data_interpolated




# get the strain, represented as COM of q values in each direction

def COM_voxels_reciprocal(data, q_values):

    #print("the shape of q should be (nhorizontal, nvertical, steps")
    
    shape_q = len(q_values.shape)
   # print(shape_q)
    
    if shape_q == 3:

        qh = q_values[:, 0, :]
        qv = q_values[:, 1, :]
        qb = q_values[:, 2, :]
    else:
        qh = q_values[0]
        qv = q_values[1]
        qb = q_values[2]


    COM_qh = np.sum(data.flatten() * qh.flatten())/ np.sum(data.flatten())
    COM_qv = np.sum(data.flatten() * qv.flatten()) / np.sum(data.flatten())
    COM_qb = np.sum(data.flatten() * qb.flatten()) / np.sum(data.flatten())

    return COM_qh, COM_qv, COM_qb


def imagesc_central_slices(*args, cmap='turbo', xlabel=None, ylabel=None, title=None, levels=300):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    image = args[0]
    ax1.imshow(image[int(image.shape[0] / 2), :, :])
    ax2.imshow(image[:, int(image.shape[1] / 2), :])
    ax3.imshow(image[:, :, int(image.shape[2] / 2)])


def imagesc_ortho_projections(*args, cmap=None, xlabel=None, ylabel=None, title=None, levels=300):
    def set_cmap_ortho(ax1, ax2, ax3, d1, d2, d3, cmap):
        if cmap != None:
            d1 = ax1.set_cmap(cmap)
            d2 = ax2.set_cmap(cmap)
            d3 = ax3.set_cmap(cmap)
        else:
            d1.set_cmap('turbo')
            d2.set_cmap('turbo')
            d3.set_cmap('turbo')

    if len(args) == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        image = args[0]
        d1 = ax1.imshow(np.sum(image, 0))
        d2 = ax2.imshow(np.sum(image, 1))
        d3 = ax3.imshow(np.sum(image, 2))
        set_cmap_ortho(ax1, ax2, ax3, d1, d2, d3, cmap)
        if title != None:
            ax.title(title)

    elif len(args) == 4:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        # fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        bv = args[0] * 1e-10
        vv = args[1] * 1e-10
        hv = args[2] * 1e-10

        d1 = ax1.imshow(np.sum(args[3], 1), extent=(np.min(hv), np.max(hv), np.min(bv), np.max(bv)))
        d2 = ax2.imshow(np.sum(args[3], 0), extent=(np.min(hv), np.max(hv), np.min(vv), np.max(vv)))
        d3 = ax3.imshow(np.sum(args[3], 2), extent=(np.min(vv), np.max(vv), np.min(bv), np.max(bv)))

    #         if xlabel!=None:
    #             ax.set_xlabel(xlabel)

    #         if ylabel!=None:
    #             ax.set_ylabel(ylabel)
    set_cmap_ortho(ax1, ax2, ax3, d1, d2, d3, cmap)
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    ax3.set_aspect('auto')
    plt.colorbar(d1, ax=ax1)
    plt.colorbar(d2, ax=ax2)
    plt.colorbar(d3, ax=ax3)
    plt.tight_layout()
    plt.show()


def imagesc(*args, cmap='turbo', xlabel=None, ylabel=None, title=None, levels=300):
    if len(args) == 1:
        plt.figure()
        plt.imshow(args[0], cmap=cmap)
        plt.colorbar()
        plt.show()

        if title != None:
            plt.title(title)

        if xlabel != None:
            plt.xlabel(xlabel)

        if ylabel != None:
            plt.ylabel(ylabel)

    elif len(args) == 3:
        hv = args[0]
        vv = args[1]
        fig, ax = plt.subplots(ncols=1)  # figsize=(10,3)
        display = ax.imshow(args[2],
                            interpolation='none',
                            extent=[hv[0], hv[-1], vv[0], vv[-1]])

        # fig, ax = plt.subplots(nrows=1, ncols=1)

        # display = ax.imshow(args[2],extent=(np.min(hv), np.max(hv), np.min(vv), np.max(vv)))
        tick_interval = 5

        plt.yticks(range(0, len(vv), tick_interval), np.round(vv[::tick_interval], 2))
        plt.xticks(range(0, len(hv), tick_interval), np.round(hv[::tick_interval], 2))

        if cmap != None:
            display.set_cmap(cmap)
        else:
            display.set_cmap('turbo')

        if xlabel != None:
            ax.set_xlabel(xlabel)

        if ylabel != None:
            ax.set_ylabel(ylabel)

        cbar = fig.colorbar(display)

        if title != None:
            ax.title(title)

        ax.set_aspect("equal")
        
        fig.tight_layout()

