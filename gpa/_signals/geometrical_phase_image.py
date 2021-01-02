# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

from hyperspy._signals.signal2d import Signal2D
from hyperspy.roi import BaseROI

from gpa.utils import gradient_phase


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gradient = None

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
            roi.add_widget(self, self.axes_manager.signal_axes)

    def refine_phase(self, refinement_roi):
        """
        Refine the geometrical phase by calculing the gradient of the phase in
        the area defined by the roi and substracting the average of the gradient
        to the phase.

        """
        if self._gradient is None:
            raise RuntimeError("Gradient needs to be calculated first.")

        correction = refinement_roi(self._gradient).mean(axis=[-2, -1])
        self._gradient -= correction

    def gradient(self):
        """ Calculate the gradient of the phase.

        Returns
        -------
        gradient : Signal2D

        Notes
        -----
        Appendix D in Hytch et al. Ultramicroscopy 1998
        """
        # Unfortunatelly, BaseSignal.map doesn't work in this case, axes_manager
        # set wrongly, we need to workaround it
        self._gradient = Signal2D(gradient_phase(self.data), flatten=False)
        for ax1, ax2 in zip(self._gradient.axes_manager.signal_axes,
                            self.axes_manager.signal_axes):
            ax1.scale = ax2.scale
            ax1.offset = ax2.offset
            ax1.units = ax1.units
        return self._gradient
