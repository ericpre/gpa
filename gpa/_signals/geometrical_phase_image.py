import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase
from hyperspy._signals.signal2d import Signal2D
from hyperspy.roi import BaseROI, RectangularROI

from gpa.utils import normalise_to_range


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'
    _signal_dimension = 2

    def __init__(self, vector=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_refinement_roi(self, roi=None):
        """
        Add a roi to the figure to define the refinement area.

        Parameters
        ----------
        roi : ROI, optional
            ROI defining the refinement area. If None, a RectangularROI is
            added. The default is None.

        """
        if roi is not None:
            if not isinstance(roi, BaseROI):
                raise ValueError("A valid hyperspy ROI must be provided. "
                                 f"Provided ROI: {roi}")
        elif roi is None:
            max_value = []
            for axis in self.axes_manager.signal_axes:
                max_value.append(axis.index2value(int(axis.size/2)))

            start = [(axis.axis[-1] - axis.axis[0]) / 4 + axis.offset
                     for axis in self.axes_manager.signal_axes]
            end = [3 * (axis.axis[-1] - axis.axis[0]) / 4 + axis.offset
                   for axis in self.axes_manager.signal_axes]
            roi = RectangularROI(*start, *end)
        if self._plot is None or not self._plot.is_active:
            self.plot()
        roi.interactive(self)

        return roi

    def refine_phase(self, fft, roi, refinement_roi, normalise=True,
                     inplace=True, unwrap=True):
        """
        Refine the geometrical phase by calculing the gradient of the phase in
        the area defined by the roi and substracting the median of the gradient
        to the phase.

        Parameters
        ----------
        inplace : bool, optional
            DESCRIPTION. The default is True.

        """

        # Take the gradient of the area defined by the ROI
        data = refinement_roi(self).data
        # if unwrap:
        #     data = self._unwrap(data, normalise=True)
        grad_x, grad_y = np.gradient(data)
        # Take the median of the gradient field to correct the phase
        correction_x, correction_y = np.median(grad_x), np.median(grad_y)

        print("correction", correction_x, correction_y)

        roi.cx = roi.cx + correction_x
        roi.cy = roi.cy + correction_y

        self.metadata.set_item('GPA.phase_from_roi', str(roi))

        data = np.angle(fft._bragg_filtering(roi, return_real=False, centre=True).data)
        if unwrap:
            data = self._unwrap(data, normalise=True)

        if normalise:
            data = normalise_to_range(data, -np.pi, np.pi)

        if inplace:
            self.data = data
            self.events.data_changed.trigger(self)
        else:
            return self._deepcopy_with_new_data(data)

    def gradient(self, flatten=False, median_filter_size=5):
        """ Calculate the gradient of the phase

        Parameters
        ----------
        median_filter_size : float, default is 5
            Size of the median filter applied to the gradient of the phase map.

        Notes
        -----
        Appendix D in Hytch et al. Ultramicroscopy 1998
        """
        x, y = np.imag(np.exp(-1j*self.data) * np.gradient(np.exp(1j*self.data)))

        if median_filter_size >0:
            x = ndimage.median_filter(x, median_filter_size)
            y = ndimage.median_filter(y, median_filter_size)
        if flatten:
            return np.array([x.flatten(), y.flatten()])
        else:
            return np.array([x, y])

    def _calc_derivative(self, axis):
        """
        Calculate the derivative of a phase image, see appendix D.
        """
        s1 = np.exp(-1j * self.data)
        s2 = np.exp(1j * self.data)
        d1 = np.diff(s2, axis=axis)  # will have 1 axis reduced by 1 pix
        nd = np.min(d1.shape)
        dP1 = s1[:nd, :nd] * d1[:nd, :nd]
        return dP1.imag

    def _unwrap(self, data, normalise=True):
        data = unwrap_phase(data)

        if normalise:
            data = normalise_to_range(data, -np.pi, np.pi)

        return data

    def unwrap(self, inplace=True, normalise=True):
        data = self._unwrap(self.data, normalise=normalise)

        if inplace:
            self.data = data
            self.events.data_changed.trigger(self)
        else:
            return self._deepcopy_with_new_data(data)
