# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np

from hyperspy.signals import Signal2D, ComplexSignal2D

from gpa.utils import get_mask_from_roi, vector_from_roi, get_ndi_module
from gpa.tools import GeometricalPhaseAnalysisTool



class AtomicResolution(Signal2D):

    signal_type = 'atomic_resolution'
    _signal_dimension = 2


    def create_gpa_tool(self, synchronise_roi_radius=True):
        """
        Create the GPA tool for the current signal.

        Parameters
        ----------
        synchronise_roi_radius : bool, optional
            If True, the radius of the roi used to select the g vectors will
            be synchronised. The default is True.

        Returns
        -------
        GeometricalPhaseAnalysisTool
            The GPA tool.

        """
        return GeometricalPhaseAnalysisTool(self, synchronise_roi_radius)

    def fft(self, *args, **kwargs):
        fft = super().fft(*args, **kwargs)
        fft.set_signal_type('spectral_domain')

        return fft

    def rescale(self, scale, order=1, inplace=True, **kwargs):
        """
        Performs interpolation to up-scale the signal space.

        Parameters
        ----------
        scale : {float, tuple of floats}
            Scale factors for the signal dimension
        order : int, optional
            The order of the spline interpolation, default is 1.
        inplace : bool, optional
            Determine if the operation is performed in place. Otherwise, the
            output is returned. The default is True.
        **kwargs : dict
            Keyword argument passed to ``map_coordinates``.

        Returns
        -------
        data : array or None
            Rescaled version of the input if inplace=False, otherwise None.

        Note
        ----
        Convenience method adapted from scikit-image rescale to support numpy
        and cupy array.

        """
        scale = tuple(np.atleast_1d(scale))
        # Normalise scale argument to match signal shape
        if len(scale) == 1:
            scale = scale * self.axes_manager.signal_dimension
        elif len(scale) > self.axes_manager.signal_dimension:
            raise ValueError("The length of scale must be the same as the"
                             "signal dimension.")

        # Add scale of 1 for all navigation dimensions
        scale = tuple([1.0]*self.axes_manager.navigation_dimension) + scale

        orig_shape = np.asarray(self.data.shape)
        output_shape = np.maximum(np.round(scale * orig_shape), 1)

        input_shape = self.data.shape
        factors = (np.asarray(input_shape, dtype=float) /
                   np.asarray(output_shape, dtype=float))

        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        grid = np.meshgrid(*coord_arrays, sparse=False, indexing='ij')
        coord_map = np.array(grid, like=self.data)

        ndi = get_ndi_module(self.data)

        s = self if inplace else self._deepcopy_with_new_data(
            None, copy_variance=True)
        s.data = ndi.map_coordinates(self.data, coord_map, order=order)

        s.get_dimensions_from_data()
        for axis, factor in zip(s.axes_manager.signal_axes, factors):
            axis.scale /= factor

        if inplace:
            s.events.data_changed.trigger(obj=s)
        else:
            return s


class ComplexAtomicResolution(ComplexSignal2D, AtomicResolution):

    pass


class AtomicResolutionFFT(ComplexSignal2D):
    """
    Attributes
    ----------

    rois : dict
        Dictionary containing HyperSpy ROI for each g.
    """


    signal_type = 'spectral_domain'
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rois = {}

    def _bragg_filtering(self, roi, return_real, centre=False, gaussian=True):
        """
        Perform Bragg filtering from a circle ROI.

        Parameters
        ----------
        roi : CircleROI
            ROI used to define the Bragg filter in the spectral domain..
        return_real : bool
            If True, return real value, otherwise complex value.
        centre : bool, optional
            Centre the ROI in the spectral domain to subtract the central
            frequency component of the ROI.
        gaussian : bool, optional
            Apply Gaussian smoothing to the edge of the ROI.

        Returns
        -------
        Signal2D or ComplexSignal2D
            Bragg filtered image of the spectra component defined by the ROI.

        """
        # Add gaussian mask
        mask = get_mask_from_roi(self, roi, gaussian)
        # check inplace, out, multi-dimensional. etc.
        signal = self * mask

        shifted = self.metadata.Signal.FFT.shifted
        if centre:
            values = (roi.cx, roi.cy)
            for axis, value in zip(signal.axes_manager.signal_axes, values):
                zero_frequency_index = axis.size / 2 if shifted else 0
                shift = int(zero_frequency_index - axis.value2index(value))
                signal.data = np.roll(signal.data, shift, axis.index_in_array)

        signal = signal.ifft(shift=shifted, return_real=return_real)
        signal.set_signal_type('atomic_resolution')

        return signal

    def bragg_filtering(self, roi):
        """
        Perform Bragg filtering from a circle ROI.

        Parameters
        ----------
        roi : CircleROI
            ROI used to define the Bragg filter in the spectral domain.

        Returns
        -------
        Signal2D or ComplexSignal2D
            Bragg filtered image of the spectra component defined by the ROI.

        """
        signal = self._bragg_filtering(roi, return_real=True)
        signal.metadata.General.title = 'Bragg filtered image'

        return signal

    def get_phase_from_roi(self, roi, reduced=False, name='g', unwrap=True):
        """
        Get the geometrical phase of the area defined by the provided ROI.

        Parameters
        ----------
        roi : CircleROI
            ROI used to define the Bragg filter in the spectral domain.
        reduced : bool
            If True, the phase is centered around the zero component frequency
            (substract the average phase from the phase).
            The default is False.

        Returns
        -------
        phase : PhaseImage
            Geometrical phase as defined by the ROI.

        """
        phase = self._bragg_filtering(roi, return_real=False)

        if unwrap:
            phase = phase.unwrapped_phase(show_progressbar=False)
        else:
            phase.data = np.angle(phase.data)

        if reduced:
            g_vector_px = np.array(roi[:2]) / self._get_g_convertion_factor(like=np.ones(1))
            phase.data -= self._calculate_phase_from_g(g_vector_px)

        phase.set_signal_type('geometrical_phase')
        phase.g_vector = vector_from_roi(roi)
        title = 'Reduced phase image' if reduced else 'Phase image'
        phase.metadata.General.title = title
        phase.metadata.set_item('GPA.phase_from_roi', f'{roi}')

        return phase

    def _calculate_phase_from_g(self, g):
        shape = self.axes_manager.signal_shape
        R_x, R_y = np.meshgrid(np.arange(0, shape[0], like=self.data),
                               np.arange(0, shape[1], like=self.data))

        return 2 * np.pi * ((R_x * g[0]) + (R_y * g[1]))

    def _get_g_convertion_factor(self, like=None):
        if like is None:
            like = self.data
        return np.array([axis.scale * axis.size for axis in
                         self.axes_manager.signal_axes], like=like)
