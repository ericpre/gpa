# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np

from hyperspy.signal import BaseSignal
from hyperspy.roi import CircleROI


def get_array_module(array):
    module = np
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            module = cp
    except ImportError:
        pass

    return module


def relative2value(axis, relative):
    """
    Return the value corresponding to the relative coordinate on a
    :py:class:`hyperspy.axes.DataAxis`

    Parameters
    ----------
    axis : :py:class:`hyperspy.axes.DataAxis`
        The DataAxis from which the value is calculated.
    relative : float
        Float value between 0 and 1. Relative coordinate on the DataAxis

    Returns
    -------
    float
        The value of the axis corresponding to the relative coordinate.

    """
    return (axis.axis[-1] - axis.axis[0]) * relative + axis.axis[0]


def vector_from_roi(roi):

    if isinstance(roi, CircleROI):
        vector = np.array([roi.cx, roi.cy])
    else:
        raise ValueError('Only "CircleROI" are supported.')

    return vector


def get_mask_from_roi(signal, roi, axes=None, gaussian=True):
    if axes is None and signal in roi.signal_map:
        axes = roi.signal_map[signal][1]
    else:
        axes = roi._parse_axes(axes, signal.axes_manager)

    radius = roi.r

    if hasattr(roi, 'cx'):
        # CircleROI
        cx = roi.cx + 0.5001 * axes[0].scale
        cy = roi.cy + 0.5001 * axes[1].scale
        ranges = [[cx - radius, cx + radius],
                  [cy - radius, cy + radius]]
    else:
        ranges = roi._get_ranges()

    if hasattr(roi, 'cx'):
        slices = roi._make_slices(axes, axes, ranges=ranges)

        if gaussian:
            mask = BaseSignal(np.zeros(signal.data.shape))
            mask.axes_manager.set_signal_dimension(
                signal.axes_manager.signal_dimension)
            x = np.arange(slices[0].stop - slices[0].start)
            y = np.arange(slices[1].stop - slices[1].start)
            xx, yy = np.meshgrid(x, y)
            sigma = len(x) / 2

            import hyperspy.api as hs
            gaussian2d = hs.model.components2D.Gaussian2D(
                sigma_x=sigma,
                sigma_y=sigma,
                centre_x=len(x)/2,
                centre_y=len(y)/2,
                A=2*np.pi*sigma**2)
            mask_circle = gaussian2d.function(xx, yy)

        else:
            mask = BaseSignal(np.full(signal.data.shape, True, dtype=bool))
            mask.axes_manager.set_signal_dimension(
                signal.axes_manager.signal_dimension)

            natax = signal.axes_manager._get_axes_in_natural_order()
            ir = [slices[natax.index(axes[0])],
                  slices[natax.index(axes[1])]]
            vx = axes[0].axis[ir[0]] - cx
            vy = axes[1].axis[ir[1]] - cy
            gx, gy = np.meshgrid(vx, vy)
            gr = gx**2 + gy**2
            mask_circle = gr > roi.r**2

        mask.isig[slices] = mask_circle

        # If signal.data is cupy array, transfer the array to the GPU
        xp = get_array_module(signal.data)
        mask.data = xp.asarray(mask.data)

    return mask


def normalise_to_range(data, vmin, vmax):
    """
    Normalise the data to the speficied range [vmin, vmax].

    Parameters
    ----------
    data : numpy.ndarray
        Data to normalise.
    vmin : float
        Minimum value after normalisation.
    vmax : float
        Maximum value after normalisation.

    Returns
    -------
    numpy.ndarray
        Normalised data.

    """
    dmin = data.min()
    dmax = data.max()
    return (vmax - vmin) * (data - dmin) / (dmax - dmin) + vmin


def rotation_matrix(angle):
    theta = np.radians(angle)

    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


def rotate_strain_tensor(angle, exx, eyy, eyx, exy):
    st = np.sin(angle/360*np.pi*2)
    ct = np.cos(angle/360*np.pi*2)

    nexx = exx*ct**2 + eyy*st*ct + eyx*ct*st + exy*st**2
    nexy = -exx*ct*st + eyy*ct**2 - eyx*st**2 + exy*ct*st
    neyx = -exx*ct*st - eyy*st**2 + eyx*ct**2 + exy*st*ct
    neyy = exx*st**2 - eyy*st*ct - eyx*ct*st + exy*ct**2

    return np.array([[nexx, nexy], [neyx, neyy]])


def gradient_phase(phase, flatten=False):
    """ Calculate the gradient of the phase

    Parameters
    ----------
    phase : numpy.ndarray
        Phase image
    flatten : float, default is False
        If True, returns flattened array.

    Notes
    -----
    Appendix D in Hytch et al. Ultramicroscopy 1998
    """

    phase = 1j * phase
    x, y = np.imag(np.exp(-phase) * np.gradient(np.exp(phase), axis=[1, 0]))

    if flatten:
        return np.array([x.flatten(), y.flatten()])
    else:
        return np.array([x, y])
