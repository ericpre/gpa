# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import matplotlib.pyplot as plt
import pytest

from gpa.drawing import add_roi_to_signal_plot


@pytest.mark.parametrize('angle', [None, 30])
def test_plot_gpa2D(gpa_tool, rois, angle):
    gpa_tool.add_rois(rois)

    gpa_tool.calculate_phase()
    gpa_tool.plot_phase()
    gpa_tool.plot_amplitude()
    for phase in gpa_tool.phases.values():
        assert phase._plot is not None
        phase._plot.close()
    for amplitude in gpa_tool.amplitudes.values():
        assert amplitude._plot is not None
        amplitude._plot.close()

    gpa_tool.calculate_strain()
    gpa_tool.plot_strain()
    plt.close('all')

    gpa_tool.plot_strain(same_figure=True)
    # components is None, set the following default
    default_components = ['e_xx', 'e_yy', 'omega']
    for component in default_components:
        _ = getattr(gpa_tool, component)
    plt.close('all')

    gpa_tool.plot_strain(same_figure=False)
    plt.close('all')


def test_plot_phase_amplitude_refinement_roi(gpa_tool, rois, refinement_roi):
    gpa_tool.add_rois(rois[:1])
    gpa_tool.calculate_phase()
    phase = gpa_tool.phases['g1']
    with pytest.raises(ValueError):
        add_roi_to_signal_plot(phase, rois[0])

    phase.plot()
    assert phase._plot is not None
    assert phase._plot.is_active
    add_roi_to_signal_plot(phase, refinement_roi)

    amplitude = gpa_tool.amplitudes['g1']
    with pytest.raises(ValueError):
        add_roi_to_signal_plot(amplitude, rois[0])

    amplitude.plot()
    assert amplitude._plot is not None
    assert amplitude._plot.is_active
    add_roi_to_signal_plot(amplitude, refinement_roi)


def test_plot_refinement_roi(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()

    refinement_roi = [1., 5., 12., 24.]
    gpa_tool.set_refinement_roi(refinement_roi)
    roi = gpa_tool.refinement_roi
    assert [roi.left, roi.top, roi.right, roi.bottom] == refinement_roi

    gpa_tool.plot_phase()


def test_plot_error_strain_calculation_missing(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()
    with pytest.raises(ValueError):
        gpa_tool.plot_strain()
