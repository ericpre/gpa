# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.
import matplotlib
matplotlib.use('agg')

import pytest

import gpa.api as gpa

@pytest.fixture
def rois():
    rois = [[4.35, 0.0, 1.5], [0.0, -4.7, 1.5]]
    return rois


@pytest.fixture
def gpa_tool():
    s = gpa.datasets.get_atomic_resolution_interface(size=512, spacing=14, strain=-0.1)
    s.add_gaussian_noise(100)
    s.set_signal_type('atomic_resolution')
    gpa_tool = gpa.GeometricalPhaseAnalysisTool(s)
    gpa_tool.set_fft(True)

    return gpa_tool