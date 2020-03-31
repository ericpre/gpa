import hyperspy.api as hs

from gpa.utils import get_atomic_resolution_tem_signal2d

s = get_atomic_resolution_tem_signal2d(1024, 1024, rotation_angle=15) * 1E4
s.add_gaussian_noise(40)

s.set_signal_type('atomic_resolution')


s.plot()
fft = s.fft(True, apodization=True)

fft.plot(True)

# Add the rois
roi_g1 = hs.roi.CircleROI(6.45161, -1.75229, 2.0)
roi_g1.interactive(fft, color='C0')

roi_g2 = hs.roi.CircleROI(-1.67264, -6.45161, 2.0)
roi_g2.interactive(fft, color='C1')

# Get phase image
phase_g1 = fft.get_phase_from_roi(roi_g1, centre=True)
phase_g2 = fft.get_phase_from_roi(roi_g2, centre=True)

# Refine phase image g1
phase_g1.plot()
phase_g1.add_refinement_roi()

phase_g1.refine_phase(fft, roi_g1)
phase_g1.unwrap()

phase_g2.plot()
phase_g2.add_refinement_roi()
phase_g2.refine_phase(fft, roi_g2)
phase_g2.unwrap()
