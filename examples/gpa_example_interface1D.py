import gpa.api as gpa

s = gpa.datasets.utils.get_atomic_resolution_interface(size=2048, spacing=14, strain=-0.1)
s.add_gaussian_noise(100)
s.plot()

s.set_signal_type('atomic_resolution')

gp_analysis = gpa.GeometricalPhaseAnalysis(s)
gp_analysis.set_fft(True)
gp_analysis.plot_fft(True)

gp_analysis.add_rois([[4.3, 0.0, 1.5]])

gp_analysis.calculate_phase()

gp_analysis.plot_phase()
