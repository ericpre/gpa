# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import gpa
from gpa.utils import get_mask_from_roi

"""
Generate synthetic dataset
"""
s = gpa.datasets.get_atomic_resolution_interface(size=1024, spacing=14, strain=-0.1)
s.add_gaussian_noise(100)
s.set_signal_type('atomic_resolution')
s.plot()

"""
Create GPA tool and display power spectrum
"""
gpa_tool = s.create_gpa_tool()
gpa_tool.set_fft()
gpa_tool.plot_power_spectrum()

"""
Add a ROI to power spectrum
"""
roi_args = [4.7, 0.0, 1.5]
gpa_tool.add_rois([roi_args])

"""
Display the Gaussian mask used for the GPA calculation and overlay the ROI
"""
fft = gpa_tool.fft_signal
roi = gpa_tool.rois['g1']
g_mask = get_mask_from_roi(fft, roi, gaussian=True)
g_mask = fft.real._deepcopy_with_new_data(g_mask.data) # need to have the signal calibration
g_mask.plot(cmap='viridis')
roi.interactive(g_mask, snap=False)
