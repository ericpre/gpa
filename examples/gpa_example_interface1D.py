# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import gpa.api as gpa

s = gpa.datasets.get_atomic_resolution_interface(size=2048, spacing=14, strain=-0.1)
s.add_gaussian_noise(100)
s.set_signal_type('atomic_resolution')

s.plot()

gpa_tool = s.create_gpa_tool()
gpa_tool.set_fft(True)

gpa_tool.plot_fft(True)

# Add ROIs for the two g_vectors
g_rois = [[4.7, 0.0, 1.5], [0.0, -4.7, 1.5]]
gpa_tool.add_rois(g_rois)
gpa_tool.calculate_phase()

# Add refinement ROI and refine phase
refinement_roi = [1., 5., 12., 24.]
gpa_tool.set_refinement_roi(refinement_roi)
gpa_tool.refine_phase()

# Calculate and plot strain
gpa_tool.calculate_strain()
gpa_tool.plot_strain()