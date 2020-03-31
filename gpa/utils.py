import numpy as np

from hyperspy.signal import BaseSignal
from hyperspy.roi import CircleROI


def get_atomic_resolution_tem_signal2d(size_x=200, size_y=200,
                                       spacing_x=10, spacing_y=10,
                                       gaussian_width_x=2, gaussian_width_y=2,
                                       rotation_angle=0):
    """Get an artificial atomic resolution TEM Signal2D.

    Returns
    -------
    artificial_tem_image : HyperSpy Signal2D

    Example
    -------
    >>> s = hs.datasets.artificial_data.get_atomic_resolution_tem_signal2d()
    >>> s.plot()

    """
    from hyperspy.signals import Signal2D
    from hyperspy import components2d
    
    x_array, y_array = np.mgrid[0:size_x, 0:size_y]
    image = np.zeros_like(x_array, dtype=np.float32)
    gaussian2d = components2d.Gaussian2D(sigma_x=gaussian_width_x,
                                         sigma_y=gaussian_width_y,
                                         centre_x=spacing_x/2,
                                         centre_y=spacing_y/2,
                                         )
    
    gaussian_peak = gaussian2d.function(*np.mgrid[0:spacing_x, 0:spacing_y])

    for i, x in enumerate(range(int(spacing_x/2), int(size_x-spacing_x/2), spacing_x)):
        for j, y in enumerate(range(int(spacing_y/2), int(size_y-spacing_y/2), spacing_y)):
            image[i*spacing_x:(i+1)*spacing_x, j*spacing_x:(j+1)*spacing_x] += gaussian_peak
    
    s = Signal2D(image)
    
    if rotation_angle != 0:
        from scipy.ndimage import rotate
        s.map(rotate, angle=rotation_angle, reshape=False)
        
        w, h = s.axes_manager.signal_axes[0].size,s.axes_manager.signal_axes[1].size
        wr, hr = get_largest_rectangle_from_rotation(w, h, rotation_angle)
        w_remove, h_remove = (w - wr), (h - hr)
        s.crop_image(int(w_remove/2), int(w-w_remove/2),
                     int(h_remove/2), int(h-h_remove/2))

    for axis in s.axes_manager.signal_axes:
        axis.scale = 0.015
        axis.units = 'nm'

    return s


def get_largest_rectangle_from_rotation(width, height, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    from: http://stackoverflow.com/a/16778797/1018861
    In hyperspy, it is centered around centre coordinate of the signal.
    """
    import math
    angle = math.radians(angle)
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

    return wr, hr


def get_mask_from_roi(signal, roi, axes=None):
    if axes is None and signal in roi.signal_map:
        axes = roi.signal_map[signal][1]
    else:
        axes = roi._parse_axes(axes, signal.axes_manager)

    if hasattr(roi, 'cx'):
        # CircleROI
        cx = roi.cx + 0.5001 * axes[0].scale
        cy = roi.cy + 0.5001 * axes[1].scale
        ranges = [[cx - roi.r, cx + roi.r],
                  [cy - roi.r, cy + roi.r]]
    else:
        ranges = roi._get_ranges()

    mask = BaseSignal(np.full(signal.data.shape, True, dtype=bool))
    mask.axes_manager.set_signal_dimension(
        signal.axes_manager.signal_dimension)

    if hasattr(roi, 'cx'):
        slices = roi._make_slices(axes, axes, ranges=ranges)
        natax = signal.axes_manager._get_axes_in_natural_order()
        ir = [slices[natax.index(axes[0])],
              slices[natax.index(axes[1])]]
        vx = axes[0].axis[ir[0]] - cx
        vy = axes[1].axis[ir[1]] - cy
        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        mask_circle = gr > roi.r**2
    
        mask.isig[slices] = mask_circle

    return mask
