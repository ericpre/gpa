# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import gpa
import hyperspy.api as hs

size = 1024

def get_interface_image(strain=0.1):
    _s = gpa.datasets.get_atomic_resolution_interface(
        size=size, spacing=14, strain=-strain)
    _s.add_gaussian_noise(100)
    return _s

s = hs.stack([get_interface_image(strain) for strain in [0, 0.025, 0.05, 0.075, 0.1]],
              show_progressbar=False)
s.set_signal_type('atomic_resolution')
s.plot()

gpa_tool = s.create_gpa_tool()
gpa_tool.set_fft()
gpa_tool.plot_power_spectrum()

# Add ROIs for the two g_vectors
g_rois = [[4.7, 0.0, 1.5], [0.0, -4.7, 1.5]]
gpa_tool.add_rois(g_rois)
gpa_tool.spatial_resolution = 0.6

gpa_tool.calculate_phase()

# # Add refinement ROI and refine phase
scale = 0.015
height = width = size * scale
refinement_roi = [0.05 * height, 0.05 * width, 0.45 * width, 0.95 * height]
gpa_tool.set_refinement_roi(refinement_roi)
gpa_tool.refine_phase()

# # Calculate and plot strain
gpa_tool.calculate_strain()
gpa_tool.plot_strain(vmin=-0.1, vmax=0.1, same_figure=False, components='e_xx')


# To export a multi-dimensional strain component as a gif animation, the
# 'imagemagick' will generate smaller file than the default 'pillow' writter
# of matplotlib, however, it is an optional dependency of matplotlib and
# may not be already installed.

gpa_tool.plot_strain(vmin=-0.1, vmax=0.1, same_figure=False, components='e_xx',
                     save_figure=True, filename='strain-e_xx.gif',
                     save_kwds={'writer':'imagemagick'}, display_figure=False)
