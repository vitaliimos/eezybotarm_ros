"""Unit tests for WorkspaceCalibrationNode."""

from unittest.mock import patch

from eezybotarm_vision.calibration_node import WorkspaceCalibrationNode
from eezybotarm_vision.zone_detector import Zone, ZoneDetectionResult
import numpy as np
import pytest
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image


def _make_test_node(**kwargs) -> WorkspaceCalibrationNode:
    params = [
        Parameter('recalibration_interval_sec', Parameter.Type.DOUBLE, 3.0),
        Parameter('calibration_smoothing_alpha', Parameter.Type.DOUBLE, 0.25),
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('sticker_width_m', Parameter.Type.DOUBLE, 0.030),
        Parameter('sticker_height_m', Parameter.Type.DOUBLE, 0.015),
    ]
    return WorkspaceCalibrationNode(parameter_overrides=params, **kwargs)


def _make_dummy_calibration(corners: np.ndarray | None = None) -> ZoneDetectionResult:
    if corners is None:
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dummy_corners = np.array([[0, 0], [30, 0], [30, 100], [0, 100]], dtype=np.float32)
    return ZoneDetectionResult(
        zones={
            'left': Zone('left', (15, 50), dummy_corners),
            'middle': Zone('middle', (45, 50), dummy_corners),
            'right': Zone('right', (75, 50), dummy_corners),
        },
        field_corners_px=corners.astype(np.float32),
        homography_matrix=np.eye(3, dtype=np.float32),
        field_width_mm=300.0,
        field_height_mm=200.0,
        marker_inset_x_mm=15.0,
        marker_inset_y_mm=7.5,
    )


def _create_mock_image_msg() -> Image:
    msg = Image()
    msg.height = 100
    msg.width = 100
    msg.encoding = 'bgr8'
    msg.step = 300
    msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    return msg


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_missing_required_parameters_raises():
    with pytest.raises(ParameterUninitializedException):
        WorkspaceCalibrationNode()


@patch('eezybotarm_vision.calibration_node.detect_zones')
def test_initial_calibration_lock(mock_detect_zones):
    mock_calib = _make_dummy_calibration()
    mock_detect_zones.return_value = mock_calib

    node = _make_test_node()
    try:
        assert node.calibration is None
        with patch.object(node.calibration_pub, 'publish') as mock_pub:
            node._handle_image(_create_mock_image_msg())
            assert node.calibration is mock_calib
            mock_pub.assert_called_once()
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.calibration_node.detect_zones')
def test_throttling_skips_cv_bridge_before_interval(mock_detect_zones):
    node = _make_test_node()
    try:
        node.calibration = _make_dummy_calibration()
        node.recalibration_interval_sec = 3.0
        node.last_recalibration_time = 100.0

        with (
            patch('time.monotonic', return_value=101.5),
            patch.object(node, '_convert_image') as mock_convert,
        ):
            node._handle_image(_create_mock_image_msg())
            mock_convert.assert_not_called()
            mock_detect_zones.assert_not_called()
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.calibration_node.detect_zones')
def test_periodic_recalibration_smoothing_and_occlusion(mock_detect_zones):
    calib1 = _make_dummy_calibration(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    )
    calib2 = _make_dummy_calibration(
        np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    )

    node = _make_test_node()
    try:
        node.calibration = calib1
        node.recalibration_interval_sec = 1.0
        node.last_recalibration_time = 0.0

        mock_detect_zones.return_value = calib2
        node._handle_image(_create_mock_image_msg())

        mock_detect_zones.assert_called_once()
        assert node.calibration is not None
        assert np.allclose(node.calibration.field_corners_px[0], [2.5, 2.5], atol=1e-1)

        # Next interval: occlusion (detect_zones returns None) preserves previous calibration
        prev_calib = node.calibration
        node.last_recalibration_time = 0.0
        mock_detect_zones.return_value = None
        node._handle_image(_create_mock_image_msg())
        assert node.calibration is prev_calib
    finally:
        node.destroy_node()
