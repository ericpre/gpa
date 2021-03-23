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
    rois = [[4.35, 0.0, 1.0], [0.0, -4.7, 1.0]]
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


@pytest.fixture
def strain_values():
    strain_values = [0.05, 0.075, 0.1]

    return strain_values


@pytest.fixture
def gpa_tool_stack(strain_values):
    def get_interface_image(strain=0.1):
        _s = gpa.datasets.get_atomic_resolution_interface(
            size=512, spacing=14, strain=-strain)
        _s.add_gaussian_noise(100)
        return _s

    strain_values = [0.05, 0.075, 0.1]

    s = hs.stack([get_interface_image(strain) for strain in strain_values],
                 show_progressbar=False)
    s.set_signal_type('atomic_resolution')
    gpa_tool = gpa.GeometricalPhaseAnalysisTool(s)
    gpa_tool.set_fft()

    return gpa_tool
