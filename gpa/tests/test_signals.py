# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
import hyperspy.api as hs

import gpa.api as gpa


def test_atomic_resolution_signal():
    s = gpa.signals.AtomicResolution(np.arange(100).reshape(10, 10))
    assert s.signal_type == 'atomic_resolution'
    assert isinstance(s, hs.signals.Signal2D)

    assert isinstance(s.create_gpa_tool(), gpa.GeometricalPhaseAnalysisTool)

    for axis in s.axes_manager.signal_axes:
        axis.units = 'nm'
    assert isinstance(s.fft(), gpa.signals.AtomicResolutionFFT)


def test_atomic_resolution_fft_signal():
    s = gpa.signals.AtomicResolutionFFT(np.arange(100).reshape(10, 10))
    assert s.signal_type == 'spectral_domain'
    assert isinstance(s, hs.signals.ComplexSignal2D)
