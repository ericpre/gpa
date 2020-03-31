import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase

import hyperspy.api as hs
from hyperspy.signals import Signal2D, ComplexSignal2D

from gpa.utils import get_mask_from_roi


class AtomicResolution(Signal2D):

    signal_type = 'atomic_resolution'
    _signal_dimension = 2

    def adjust_g_position(self, fft, **kwargs):
        if fft._plot is None or not fft._plot.is_active:
            fft.plot()

        self.roi_g1 = hs.roi.CircleROI(6.5, -1.5)

    def fft(self, *args, **kwargs):
        fft = super().fft(*args, **kwargs)
        fft.set_signal_type('spectral_domain')

        return fft


class AtomicResolutionFFT(ComplexSignal2D):

    signal_type = 'spectral_domain'
    _signal_dimension = 2

    # def __init__(self, *args, **kwargs):
    #     self.roi_g1 = None
    #     self.roi_g2 = None

    def _bragg_filtering(self, roi, real, centre=False):
        mask = get_mask_from_roi(self, roi)
        # check inplace, out, multi-dimensional. etc.
        signal = (~mask.data) * self.deepcopy()
        shifted = self.metadata.Signal.FFT.shifted
        if centre:
            values = (roi.cx, roi.cy)
            for axis, value in zip(signal.axes_manager.signal_axes, values):
                zero_frequency_index = axis.size / 2 if shifted else 0
                shift = int(zero_frequency_index - axis.value2index(value))
                signal.data = np.roll(signal.data, shift, axis.index_in_array)

        return signal.ifft(shift=shifted, real=real)

    def bragg_filtering(self, roi):
        """
        Perform Bragg filtering from a circle ROI.

        Parameters
        ----------
        roi : CircleROI
            ROI used to define the Bragg filter in the spectral domain.

        Returns
        -------
        signal : Signal2D
            Bragg filtered image.

        """
        signal = self._bragg_filtering(roi, real=True)
        signal.set_signal_type(self.signal_type)
        signal.metadata.General.title = 'Bragg filtered image'

        return signal

    def get_phase_from_roi(self, roi, centre=False):
        """
        Get the geometrical phase of the area defined by the provided ROI.

        Parameters
        ----------
        roi : CircleROI
            ROI used to define the Bragg filter in the spectral domain.
        centre : bool
            If True, the phase is centered around the zero component frequency
            (substract the average phase from the phase).
            The default is False.

        Returns
        -------
        phase : PhaseImage
            Geometrical phase as defined by the ROI.

        """

        phase = self._bragg_filtering(roi, real=False, centre=centre)
        phase.data = np.angle(phase.data)
        phase._dtype = 'real'
        phase.set_signal_type('geometrical_phase')
        phase._assign_subclass()
        phase.metadata.General.title = 'Phase image'
        phase.metadata.set_item('GPA.phase_from_roi', str(roi))

        return phase


class GeometricalPhaseImage(Signal2D):

    signal_type = 'geometrical_phase'
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refinement_roi = None

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
            if isinstance(roi, hs.roi.BaseROI):
                raise ValueError(f"A valid hyperspy ROI must be provided.")
            self.refinement_roi = roi
        elif self.refinement_roi is None:
            max_value = []
            for axis in self.axes_manager.signal_axes:
                max_value.append(axis.index2value(int(axis.size/2)))
            self.refinement_roi = hs.roi.RectangularROI(
                0, 0, max_value[0], max_value[1])
            if self._plot is None or not self._plot.is_active:
                self.plot()
            self.refinement_roi.interactive(self)        

    def refine_phase(self, fft, roi, inplace=True):
        """
        Refine the geometrical phase by calculing the gradient of the phase in
        the area defined by the roi and substracting the median of the gradient
        to the phase.

        Parameters
        ----------
        inplace : bool, optional
            DESCRIPTION. The default is True.

        """
        if self.refinement_roi is None:
            self.add_refinement_roi()

        # Take the gradient of the area defined by the ROI
        grad_x, grad_y = np.gradient(self.refinement_roi(self).data)
        # Take the median of the gradient field to correct the phase
        correction_x, correction_y = np.median(grad_x), np.median(grad_y)

        roi.cx = roi.cx + correction_x
        roi.cy = roi.cy + correction_y

        self.data = np.angle(fft._bragg_filtering(roi, real=False, centre=True).data)
        self.events.data_changed.trigger(self)

    def unwrap(self, inplace=True):
        out = unwrap_phase(self.data) 
        if inplace:
            self.data = out
            self.events.data_changed.trigger(self)
        else:
            return self._deepcopy_with_new_data(out)
