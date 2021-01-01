# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
from hyperspy._signals.signal2d import Signal2D
from hyperspy.roi import BaseROI

from gpa.utils import gradient_phase


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_refinement_roi(self, roi=None):
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

        if self._plot is not None or not self._plot.is_active:
            print('here')
            roi.add_widget(self, self.axes_manager.signal_axes)

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
        # Take the average of the gradient phase
        grad_phase = gradient_phase(data)
        g1_r = np.sum(grad_phase, axis=0) / (2 * np.pi)

        correction_x = np.average(grad_phase[0])
        correction_y = np.average(grad_phase[1])

        print("correction", correction_x, correction_y)

        roi.cx = roi.cx + correction_x
        roi.cy = roi.cy + correction_y

        self.metadata.set_item('GPA.phase_from_roi', str(roi))

        data = np.angle(fft._bragg_filtering(roi, return_real=False, centre=True).data)

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

        return gradient_phase(self.data, flatten=flatten)

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