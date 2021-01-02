# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
import pytest
import hyperspy.api as hs

import gpa.api as gpa


def assert_g_a_matrices_product(g_matrix, a_matrix):
    np.testing.assert_allclose(g_matrix @ a_matrix, np.eye(2))


def assert_strain_components(gpa_tool):
    for component in ['e_xx', 'e_yy', 'omega', 'theta']:
        assert isinstance(getattr(gpa_tool, component), hs.signals.Signal2D)


def test_gpa_tool_error(gpa_tool, rois):
    gpa_tool.fft = None
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
