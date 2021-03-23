# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import matplotlib.animation as animation
import numpy as np

from hyperspy.roi import CircleROI
from hyperspy.signal import BaseSignal


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

    # Needs to add support for other type of ROI
    if hasattr(roi, 'cx'):
        # CircleROI
        radius = roi.r
        cx = roi.cx
        cy = roi.cy
        r = np.linalg.norm([cx, cy]) * 0.8
        # The factor of 3 come from an estimate of how far the tail of the
        # Gaussian goes; to avoid getting the zero-frequency component in
        # the mask, we clip its radius_slice value
        radius_slice = np.clip(radius * 3, a_min=radius, a_max=r)
        ranges = [[cx - radius_slice, cx + radius_slice],
                  [cy - radius_slice, cy + radius_slice]]
    else:
        ranges = roi._get_ranges()

    if hasattr(roi, 'cx'):
        # The 'else' part is missing
        slices = roi._make_slices(axes, axes, ranges=ranges)

        if not gaussian:
            # in case of Bragg Filtering
            radius_slice = radius

        # Calculate a disk mask
        sig_axes = signal.axes_manager.signal_axes
        ir = [slices[sig_axes.index(axes[0])],
              slices[sig_axes.index(axes[1])]]
        vx = axes[0].axis[ir[0]] - cx
        vy = axes[1].axis[ir[1]] - cy
        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        disk_mask = gr > radius_slice**2

        if gaussian:
            import hyperspy.api as hs
            mask = hs.signals.Signal2D(np.zeros(signal.data.shape))
            x = np.linspace(ranges[0][0], ranges[0][1], disk_mask.shape[1])
            y = np.linspace(ranges[1][0], ranges[1][1], disk_mask.shape[0])
            xx, yy = np.meshgrid(x, y)

            gaussian2d = hs.model.components2D.Gaussian2D(
                sigma_x=radius,
                sigma_y=radius,
                centre_x=cx,
                centre_y=cy,
                A=2*np.pi*radius**2)
            mask_circle = gaussian2d.function(xx, yy) * ~disk_mask
        else:
            mask = BaseSignal(np.full(signal.data.shape, True, dtype=bool))
            mask.axes_manager.set_signal_dimension(
                signal.axes_manager.signal_dimension)

            mask_circle = disk_mask

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


def rotation_matrix(angle, like):
    theta = np.radians(angle)

    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]], like=like)


def rotate_strain_tensor(angle, exx, eyy, eyx, exy, like):
    st = np.sin(angle/360*np.pi*2)
    ct = np.cos(angle/360*np.pi*2)

    nexx = exx*ct**2 + eyy*st*ct + eyx*ct*st + exy*st**2
    nexy = -exx*ct*st + eyy*ct**2 - eyx*st**2 + exy*ct*st
    neyx = -exx*ct*st - eyy*st**2 + eyx*ct**2 + exy*st*ct
    neyy = exx*st**2 - eyy*st*ct - eyx*ct*st + exy*ct**2

    return np.array([[nexx, nexy], [neyx, neyy]], like=like)


def gradient_phase(phase, axis, flatten=False):
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
    x, y = np.imag(np.exp(-phase) * np.array(np.gradient(np.exp(phase), axis=axis), like=phase))

    if flatten:
        return np.array([x.flatten(), y.flatten()], like=phase)
    else:
        return np.array([x, y], like=phase)


def is_cupy_array(array):
    """
    Convenience function to determine if an array is a cupy array.

    Parameters
    ----------
    array : array
        The array to determine whether it is a cupy array or not.

    Returns
    -------
    bool
        True if it is cupy array, False otherwise.
    """
    try:
        import cupy as cp
        return isinstance(array, cp.ndarray)
    except ImportError:
        return False


def to_numpy(array):
    """
    Returns the array as an numpy array

    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used.

    Returns
    -------
    array : numpy.ndarray
    """
    if is_cupy_array(array):
        import cupy as cp
        array = cp.asnumpy(array)

    return array


def get_array_module(array):
    """
    Returns the array module for the given array.

    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used.

    Returns
    -------
    module : module
    """
    module = np
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            module = cp
    except ImportError:
        pass

    return module


def get_ndi_module(array):
    """
    Returns the array module for the given array.

    Parameters
    ----------
    array : numpy or cupy array
        Array to determine whether numpy or cupy should be used.

    Returns
    -------
    module : module
    """

    if is_cupy_array(array):
        from cupyx.scipy import ndimage
        return ndimage
    else:
        from scipy import ndimage
        return ndimage


def export_signal_as_animation(signal, filename, **kwargs):
    """
    Generate a matplotlib animation of a plotted signal and save it as a file.
    Only the signal figure is saved and the signal will iterate over the
    navigation indices.

    Parameters
    ----------
    signal : BaseSignal instance
        The signal to save as an animation.
    filename : str
        Name of the file.
    **kwargs : dict
        The keyword argument are passed to
        `matplotlib.animation.Animation.save`

    Returns
    -------
    matplotlib.animation.Animation
        The matplotlib animation of the signal.

    """

    if signal._plot is None or not signal._plot.is_active:
        raise RuntimeError("The signal must be plotted.")

    _plot = signal._plot.signal_plot
    signal.axes_manager.indices = (0, )
    fig = _plot.ax.figure

    frames = signal.axes_manager.navigation_axes[0].size

    def update(i):
        signal.axes_manager.indices = (i, )
        return _plot.ax.images

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  blit=_plot.figure.canvas.supports_blit,
                                  repeat=False)

    ani.save(filename, **kwargs)
    return ani
