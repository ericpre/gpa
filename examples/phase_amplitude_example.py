# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import gpa

s = gpa.datasets.get_atomic_resolution_interface(size=256, spacing=14, strain=-0.1)
s.add_gaussian_noise(100)
s.set_signal_type('atomic_resolution')

s.plot()

gpa_tool = s.create_gpa_tool()
gpa_tool.set_fft()

gpa_tool.plot_power_spectrum()

g_rois = [[4.7, 0.0, 1.5], [0.0, -4.7, 1.5]]
gpa_tool.add_rois(g_rois)
gpa_tool.spatial_resolution = 0.6

roi = gpa_tool.rois['g1']

# return a complex signal from which we can extract the phase and the amplitude
out = gpa_tool.fft_signal._bragg_filtering(roi, return_real=False)

gpa_tool.calculate_phase()
gpa_tool.plot_amplitude()
