import numpy as np

import hyperspy.api as hs
from hyperspy.signals import Signal2D, ComplexSignal2D

from gpa.utils import get_mask_from_roi, vector_from_roi



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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi = {}

    def _bragg_filtering(self, roi, return_real, centre=False, gaussian=True):
        # Add gaussian mask
        mask = get_mask_from_roi(self, roi, gaussian)
        # check inplace, out, multi-dimensional. etc.
        signal = self * mask
        shifted = self.metadata.Signal.FFT.shifted
        if centre:
            # alternative would be to substract a ramp of a corresponding g
            # followed by normalisation to -pi, pi
            values = (roi.cx, roi.cy)
            for axis, value in zip(signal.axes_manager.signal_axes, values):
                zero_frequency_index = axis.size / 2 if shifted else 0
                shift = int(zero_frequency_index - axis.value2index(value))
                signal.data = np.roll(signal.data, shift, axis.index_in_array)

        return signal.ifft(shift=shifted, return_real=return_real)

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
        signal = self._bragg_filtering(roi, return_real=True)
        signal.set_signal_type(self.signal_type)
        signal.metadata.General.title = 'Bragg filtered image'

        return signal

    def get_phase_from_roi(self, roi, reduced=False):
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

        phase = self._bragg_filtering(roi, return_real=False, centre=reduced)
        phase.data = np.angle(phase.data)
        phase._dtype = 'real'
        phase.set_signal_type('geometrical_phase')
        phase._assign_subclass()
        phase.vector = vector_from_roi(roi)
        phase.metadata.General.title = 'Phase image'
        phase.metadata.set_item('GPA.phase_from_roi', str(roi))

        if self.roi.get('g1') is None:
            self.roi['g1'] = roi
        elif self.roi.get('g2') is None:
            self.roi['g2'] = roi

        return phase

    # def find_peaks(self):
    #     if self.metadata.Signal.FFT.shifted:
    #         offset = [int(v/2) for v in data.shape]
    #     else:
    #         offset = [0] * len(data.shape)
    #     data = self.data[offset[0]:, offset[1]:]
