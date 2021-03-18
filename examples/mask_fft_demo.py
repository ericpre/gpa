# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import matplotlib.pyplot as plt
import numpy as np

import hyperspy.api as hs

import gpa
from gpa.utils import get_mask_from_roi

"""
Generate synthetic dataset
"""
s = gpa.datasets.get_atomic_resolution_interface(size=2048, spacing=14, strain=-0.1)
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
roi = hs.roi.CircleROI(*roi_args)
gpa_tool.add_rois([tuple(roi)])

"""
Display the Gaussina mask used for the GPA calculation and overlay the ROI
"""
fft = gpa_tool.fft_signal
g_mask = get_mask_from_roi(fft, roi, gaussian=True)
g_mask = fft.real._deepcopy_with_new_data(g_mask.data) # need to have the signal calibration
g_mask.plot(cmap='viridis')
roi.interactive(g_mask)
