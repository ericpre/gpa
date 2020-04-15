import os
import numpy as np
import hyperspy.api as hs

import gpa.api as gpa

s = gpa.datasets.get_atomic_resolution_tem_signal2d(
    1024, 1024, rotation_angle=15) * 1E4
s.add_gaussian_noise(40)

s.set_signal_type('atomic_resolution')

gp_analysis = gpa.GeometricalPhaseAnalysis(s)
gp_analysis.set_fft(True)
gp_analysis.plot_fft(True)


gp_analysis.add_rois([[4.3010, -1.1947, 1.5], [1.19474, 4.3010, 1.5]])

gp_analysis.calculate_phase()

gp_analysis.plot_phase()

gp_analysis.refine_phase()

gp_analysis.calculate_displacement()
gp_analysis.calculate_strain()

gp_analysis.plot_strain(components='all', same_figure=True,
                        vmin=-0.01, vmax=0.01)
