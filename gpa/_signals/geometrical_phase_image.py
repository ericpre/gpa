# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
from hyperspy._signals.signal2d import Signal2D
from hyperspy.roi import BaseROI, RectangularROI

from gpa.utils import normalise_to_range


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
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

    def gradient(self, flatten=False):
        """ Calculate the gradient of the phase

        Parameters
        ----------
        flatten : float, default is False
            If True, returns flattened array.

        Notes
        -----
        Appendix D in Hytch et al. Ultramicroscopy 1998
        """

        phase = 1j * self.data
        x, y = np.imag(np.exp(-phase) * np.gradient(np.exp(phase), axis=[1, 0]))

        if flatten:
            return np.array([x.flatten(), y.flatten()])
        else:
            return np.array([x, y])

    def _get_phase_ramp(self):
        """
        Get the phase ramp corresponding to a g vector.

        Parameters
        ----------
        g : numpy.ndarray
            g vectors in calibrated units.

        Returns
        -------
        numpy.ndarray

        """
        # Ramp over 2pi
        shape = self.data.shape
        x = np.arange(shape[0]) / shape[0] * 2 * np.pi
        y = np.arange(shape[1]) / shape[1] * 2 * np.pi
        xx, yy = np.meshgrid(x, y)

        # convert to pixel unit
        g_px = 2 * self.g_vector / self.axes_manager.signal_axes[0].scale / shape[0]
        print("g_px", g_px)

        # phase ramp corresponding to g vector
        return g_px[0] * xx + g_px[1] * yy