# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import gpa

s = gpa.datasets.get_atomic_resolution_interface(size=2048, spacing=14, strain=-0.1)
s.add_gaussian_noise(100)
s.set_signal_type('atomic_resolution')

s.plot()

gpa_tool = s.create_gpa_tool()
gpa_tool.set_fft(True)

gpa_tool.plot_power_spectrum()

# Add ROIs for the two g_vectors
g_rois = [[4.7, 0.0, 1.5], [0.0, -4.7, 1.5]]
gpa_tool.add_rois(g_rois)
gpa_tool.calculate_phase()

# Add refinement ROI and refine phase
refinement_roi = [1., 5., 12., 29.]
gpa_tool.set_refinement_roi(refinement_roi)
gpa_tool.refine_phase()

# Calculate and plot strain
gpa_tool.calculate_strain()
gpa_tool.plot_strain()


def assert_strain_values(gpa_tool):
    # check that measured strain is correct
    import hyperspy.api as hs
    import numpy as np

    # In reference area, the strain is 0
    refinement_roi = [1., 5., 12., 29.]
    reference_area_roi = hs.roi.RectangularROI(*refinement_roi)
    strain_area = reference_area_roi(gpa_tool.e_xx).data.mean()
    np.testing.assert_almost_equal(strain_area, 1E-10)

    # In reference area, the strain is 0.01
    strained_area_roi_args = [17.0, 0.5, 29., 29.]
    strained_area_roi = hs.roi.RectangularROI(*strained_area_roi_args)
    strain_area = strained_area_roi(gpa_tool.e_xx).data.mean()
    np.testing.assert_almost_equal(strain_area, 0.0997, decimal=3)

    print('All tests passed!')
