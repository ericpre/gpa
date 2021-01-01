# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize('angle', [None, 30])
def test_plot_gpa2D(gpa_tool, rois, angle):
    gpa_tool.add_rois(rois)

    gpa_tool.calculate_phase()
    gpa_tool.plot_phase()
    for phase in gpa_tool.phases.values():
        assert phase._plot is not None
        phase._plot.close()

    gpa_tool.calculate_strain()
    gpa_tool.plot_strain()
    plt.close('all')

    gpa_tool.plot_strain(same_figure=True)
    # components is None, set the following default
    default_components = ['e_xx', 'e_yy', 'omega']
    for component in default_components:
        c = getattr(gpa_tool, component)
        assert c._plot is None
    plt.close('all')

    gpa_tool.plot_strain(same_figure=False)
    for component in default_components:
        c = getattr(gpa_tool, component)
        assert c._plot is not None
        c._plot.close()
