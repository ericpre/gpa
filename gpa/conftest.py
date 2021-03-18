# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.
import matplotlib
matplotlib.use('agg')

import hyperspy.api as hs
import pytest

import gpa


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
    gpa_tool.set_fft()

    return gpa_tool


@pytest.fixture
def refinement_roi_args():
    refinement_roi_args = [0.1, 0.1, 3.4, 7.]

    return refinement_roi_args


@pytest.fixture
def refinement_roi(refinement_roi_args):
    refinement_roi = hs.roi.RectangularROI(*refinement_roi_args)

    return refinement_roi
