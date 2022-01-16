# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
from hyperspy._signals.signal2d import Signal2D

from gpa.utils import gradient_phase


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gradient = None

    def refine_phase(self, refinement_roi):
        """
        Refine the geometrical phase by calculating the geometrical mean of the
        gradient of the phase in the area defined by the refinement roi and
        substracting it to the gradient of the phase.

        Parameters
        ----------
        refinement_roi : hyperspy ROI
            The area used to refine to calculate the geometrical mean of the
            gradient of the phase.

        Returns
        -------
        g_refinement : array
            The shift in g corresponding to the change in phase reference.

        """
        if self._gradient is None:
            raise RuntimeError("Gradient needs to be calculated first.")

        if self.axes_manager.navigation_size == 0:
            reference = self._gradient
        else:
            # Average over the navigation dimensions, the last navigation
            # axis of the gradient are the two x, y compoments of the gradient
            nav_axis = self._gradient.axes_manager.navigation_axes[:-1]
            reference = self._gradient.mean(axis=nav_axis)

        axis = reference.axes_manager.signal_indices_in_array
        # Refine the gradient of the phase
        grad_refinement = reference.isig[refinement_roi].data.mean(axis=axis)
        # cupy broadcasting is not yet supported in hyperspy
        for i in range(2):
            self._gradient.data[i] -= grad_refinement[i]

        return grad_refinement / (-2*np.pi)

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
        # set wrongly, we need to workaround
        axis = self.axes_manager.signal_indices_in_array
        self._gradient = Signal2D(gradient_phase(self.data, axis=axis))
        for ax1, ax2 in zip(self._gradient.axes_manager.signal_axes,
                            self.axes_manager.signal_axes):
            ax1.scale = ax2.scale
            ax1.offset = ax2.offset
            ax1.units = ax2.units
        return self._gradient
