# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
import pytest
import hyperspy.api as hs

import gpa


def assert_g_a_matrices_product(g_matrix, a_matrix):
    np.testing.assert_allclose(g_matrix @ a_matrix, np.eye(2))


def assert_strain_components(gpa_tool):
    for component in ['e_xx', 'e_yy', 'omega', 'theta']:
        assert isinstance(getattr(gpa_tool, component), hs.signals.Signal2D)


def test_gpa_tool_error(gpa_tool, rois):
    gpa_tool.fft_signal = None
    with pytest.raises(RuntimeError):
        gpa_tool.add_rois(rois[:1])


def test_single_g_vectors(gpa_tool, rois):
    gpa_tool.add_rois(rois[:1])
    g_vectors = gpa_tool.g_vectors()
    assert len(g_vectors) == 1
    np.testing.assert_allclose(g_vectors['g1'], rois[0][:2])

    g_matrix = gpa_tool._g_matrix(calibrated=True)
    np.testing.assert_allclose(g_matrix, np.array([[4.35, 0],
                                                   [0, -4.35]]))

    a_matrix = gpa_tool._a_matrix(calibrated=True)
    np.testing.assert_allclose(a_matrix, np.array([[0.22988506, 0],
                                                   [0, -0.22988506]]))

    assert_g_a_matrices_product(g_matrix, a_matrix)

    g_matrix = gpa_tool._g_matrix(calibrated=False)
    np.testing.assert_allclose(g_matrix, np.array([[0.06525, 0],
                                                   [0, -0.06525]]), rtol=1E-6)

    a_matrix = gpa_tool._a_matrix(calibrated=False)
    np.testing.assert_allclose(a_matrix, np.array([[15.32567, 0],
                                                   [0, -15.32567]]))

    assert_g_a_matrices_product(g_matrix, a_matrix)


def test_g_vectors(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    g_vectors = gpa_tool.g_vectors()
    assert len(g_vectors) == 2
    for g_vector, roi in zip(g_vectors.values(), rois):
        np.testing.assert_allclose(g_vector, roi[:2])

    g_matrix = gpa_tool._g_matrix(calibrated=True)
    np.testing.assert_allclose(g_matrix, np.array([[4.35, 0],
                                                   [0, -4.7]]))

    a_matrix = gpa_tool._a_matrix(calibrated=True)
    np.testing.assert_allclose(a_matrix, np.array([[0.22988506, 0],
                                                   [0, -0.21276596]]))

    assert_g_a_matrices_product(g_matrix, a_matrix)

    g_matrix = gpa_tool._g_matrix(calibrated=False)
    np.testing.assert_allclose(g_matrix, np.array([[0.06525, 0],
                                                   [0, -0.0705]]), rtol=1E-6)

    a_matrix = gpa_tool._a_matrix(calibrated=False)
    np.testing.assert_allclose(a_matrix, np.array([[15.32567, 0],
                                                   [0, -14.184397]]))

    assert_g_a_matrices_product(g_matrix, a_matrix)


def test_gpa1D(gpa_tool, rois):
    gpa_tool.add_rois(rois[:1])
    gpa_tool.calculate_phase()
    assert len(gpa_tool.phases) == 1
    phase = gpa_tool.phases['g1']
    roi = rois[0]
    assert isinstance(phase, gpa.signals.GeometricalPhaseImage)
    np.testing.assert_allclose(phase.g_vector, roi[:2])

    gpa_tool.calculate_displacement()
    gpa_tool.calculate_strain()
    assert_strain_components(gpa_tool)


def test_gpa2D(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()
    assert len(gpa_tool.phases) == 2
    for g, roi in zip(['g1', 'g2'], rois):
        phase = gpa_tool.phases[g]
        assert isinstance(gpa_tool.phases['g1'], gpa.signals.GeometricalPhaseImage)
        np.testing.assert_allclose(phase.g_vector, roi[:2])


    gpa_tool.calculate_displacement()
    gpa_tool.calculate_strain()
    assert_strain_components(gpa_tool)


def test_gpa2D_angle(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()
    assert gpa_tool.angle is None
    gpa_tool.calculate_strain(angle=30.0)
    assert gpa_tool.angle == 30.0
    np.testing.assert_allclose(gpa_tool._g_matrix(),
                               np.array([[ 0.05650816, -0.032625],
                                         [-0.03525, -0.06105479]]))


def test_refine_phase(gpa_tool, rois, refinement_roi, refinement_roi_args):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()

    gpa_tool.set_refinement_roi(refinement_roi_args)
    gpa_tool.refine_phase()
    np.testing.assert_allclose(gpa_tool.g_vectors()['g1'],
                               np.array([4.7512, 0.0]), atol=5E-3)
    np.testing.assert_allclose(gpa_tool.g_vectors()['g2'],
                               np.array([-1.02912e-04, -4.7519]), atol=5E-3)

    gpa_tool.calculate_strain()

    ref_area_strain = refinement_roi(gpa_tool.e_xx).data.mean()
    np.testing.assert_almost_equal(ref_area_strain, 1E-8)

    strained_area_roi_args = [4.5, 0.2, 7.2, 7.2]
    strained_area_roi = hs.roi.RectangularROI(*strained_area_roi_args)
    strain_area = strained_area_roi(gpa_tool.e_xx).data.mean()
    # strain error due to sampling, larger number of pixels would improve
    np.testing.assert_almost_equal(strain_area, 0.0974, decimal=3)


def test_refine_phase_default(gpa_tool, rois):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()

    with pytest.raises(RuntimeError):
        gpa_tool.refine_phase()

    gpa_tool.set_refinement_roi()
    assert isinstance(gpa_tool.refinement_roi, hs.roi.RectangularROI)
    assert gpa_tool.refinement_roi.left == 1.91625
    assert gpa_tool.refinement_roi.top == 1.91625
    assert gpa_tool.refinement_roi.right == 5.74875
    assert gpa_tool.refinement_roi.bottom == 5.74875


def test_refine_phase_strain_values(gpa_tool, rois, refinement_roi_args):
    gpa_tool.add_rois(rois)
    gpa_tool.calculate_phase()
    gpa_tool.plot_phase()

    gpa_tool.set_refinement_roi(refinement_roi_args)

    assert isinstance(gpa_tool.refinement_roi, hs.roi.RectangularROI)
    assert gpa_tool.refinement_roi.left == refinement_roi_args[0]
    assert gpa_tool.refinement_roi.top == refinement_roi_args[1]
    assert gpa_tool.refinement_roi.right == refinement_roi_args[2]
    assert gpa_tool.refinement_roi.bottom == refinement_roi_args[3]
    gpa_tool.refine_phase()

    gpa_tool.calculate_strain()

    # Check strain in reference area
    reference_area_roi = hs.roi.RectangularROI(*refinement_roi_args)
    strain_area = reference_area_roi(gpa_tool.e_xx).data.mean()
    np.testing.assert_almost_equal(strain_area, 1E-10)

    # Check strain in strained area
    strained_area_roi_args = [4.2, 0.1, 7.4, 7.4]
    strained_area_roi = hs.roi.RectangularROI(*strained_area_roi_args)
    strain_area = strained_area_roi(gpa_tool.e_xx).data.mean()
    np.testing.assert_almost_equal(strain_area, 0.0970, decimal=3)
