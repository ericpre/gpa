# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import hyperspy.api as hs
import numpy as np
import pytest

import gpa


def test_atomic_resolution_signal():
    s = gpa.signals.AtomicResolution(np.arange(100).reshape(10, 10))
    assert s.signal_type == 'atomic_resolution'
    assert isinstance(s, hs.signals.Signal2D)

    assert isinstance(s.create_gpa_tool(), gpa.GeometricalPhaseAnalysisTool)

    for axis in s.axes_manager.signal_axes:
        axis.units = 'nm'
    assert isinstance(s.fft(), gpa.signals.AtomicResolutionFFT)


def test_refine_phase(gpa_tool, rois, refinement_roi):
    gpa_tool.add_rois(rois[:1])
    gpa_tool.calculate_phase()
    phase = gpa_tool.phases['g1']

    with pytest.raises(RuntimeError):
        # Gradient needs to be calculate first
        phase.refine_phase(refinement_roi)

    grad = phase.gradient()
    assert isinstance(grad, hs.signals.Signal2D)
    assert grad.axes_manager.navigation_size == 2
    for ax1, ax2 in zip(phase.axes_manager.signal_axes,
                        grad.axes_manager.signal_axes):
        assert ax1.scale == ax2.scale
        assert ax1.offset == ax2.offset
        assert ax1.units == ax2.units

    g_refinement = phase.refine_phase(refinement_roi)
    assert isinstance(g_refinement, np.ndarray)
    np.testing.assert_allclose(g_refinement.data, np.array([-6.038e-03, 0.0]), atol=5e-5)


def test_atomic_resolution_fft_signal():
    s = gpa.signals.AtomicResolutionFFT(np.arange(100).reshape(10, 10))
    assert s.signal_type == 'spectral_domain'
    assert isinstance(s, hs.signals.ComplexSignal2D)


def test_rescale():
    s = gpa.datasets.get_atomic_resolution_interface(
        size=256, spacing=14, strain=-0.1)
    s.add_gaussian_noise(100)
    s.set_signal_type('atomic_resolution')
    s.rescale(1.25)
    assert s.data.shape == (320, 320)

    s.rescale((1.25, 1.0))
    assert s.data.shape == (400, 320)

    np.testing.assert_allclose([ax.scale for ax in s.axes_manager.signal_axes],
                               [0.0234375, 0.01875])


def test_rescale_navigation():
    s = hs.signals.Signal2D(np.arange(1E3).reshape([10]*3))
    s.set_signal_type('atomic_resolution')
    with pytest.raises(ValueError):
        s.rescale((1.25, 1.25, 1.0))

    s2 = s.rescale((1.25, 1.25), inplace=False)
    s3 = s.rescale((1.25, ), inplace=False)
    s.rescale(1.25)
    assert s == s2
    assert s == s3
