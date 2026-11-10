"""Unit tests for CubeDetectorNode."""

from unittest.mock import patch

from eezybotarm_vision.cube_detector_node import CubeDetectorNode
from eezybotarm_vision.yolo_detector import CubeDetection
from eezybotarm_vision.zone_detector import Zone, ZoneDetectionResult
import numpy as np
import pytest
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image


def _make_test_node(**kwargs) -> CubeDetectorNode:
    params = [
        Parameter('weights_path', Parameter.Type.STRING, '/usr/share/models/model.engine'),
        Parameter('conf_threshold', Parameter.Type.DOUBLE, 0.5),
    ]
    return CubeDetectorNode(parameter_overrides=params, **kwargs)


def _create_mock_image_msg() -> Image:
    msg = Image()
    msg.height = 100
    msg.width = 100
    msg.encoding = 'bgr8'
    msg.step = 300
    msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    return msg


def _make_cube(
    class_id: int = 0,
    class_name: str = 'green_cube',
    confidence: float = 0.9,
    center: tuple[float, float] = (150.0, 100.0),
) -> CubeDetection:
    cx, cy = center
    bbox = (cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0)
    corners = ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]))
    return CubeDetection(class_id, class_name, confidence, bbox, center, 0.0, corners)


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    rclpy.init()
    yield
    rclpy.shutdown()


@patch('eezybotarm_vision.cube_detector_node.YoloCubeDetector')
def test_missing_required_parameters_raises(mock_detector):
    with pytest.raises(ParameterUninitializedException):
        CubeDetectorNode()


@patch('eezybotarm_vision.cube_detector_node.YoloCubeDetector')
def test_uncalibrated_detection_publishes_uncalibrated_zone(mock_detector):
    mock_detector.return_value.predict.return_value = [_make_cube()]
    node = _make_test_node()
    try:
        with patch.object(node.detections_pub, 'publish') as mock_pub:
            node._handle_image(_create_mock_image_msg())
            mock_pub.assert_called_once()
            published_msg = mock_pub.call_args[0][0]
            assert len(published_msg.cubes) == 1
            assert published_msg.cubes[0].zone == 'uncalibrated'
            assert published_msg.cubes[0].x_mm == 0.0
            assert published_msg.cubes[0].y_mm == 0.0
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.cube_detector_node.YoloCubeDetector')
def test_calibrated_detection_maps_coordinates_and_zone(mock_detector):
    mock_detector.return_value.predict.return_value = [
        _make_cube(center=(150.0, 100.0))
    ]
    node = _make_test_node()
    try:
        dummy_corners = np.array([[0, 0], [30, 0], [30, 100], [0, 100]], dtype=np.float32)
        calib = ZoneDetectionResult(
            zones={
                'left': Zone('left', (15, 50), dummy_corners),
                'middle': Zone('middle', (45, 50), dummy_corners),
                'right': Zone('right', (75, 50), dummy_corners),
            },
            field_corners_px=np.array(
                [[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32
            ),
            homography_matrix=np.eye(3, dtype=np.float64),
            field_width_mm=300.0,
            field_height_mm=200.0,
            marker_inset_x_mm=15.0,
            marker_inset_y_mm=7.5,
        )
        node._calibration_callback(calib.to_msg())

        with patch.object(node.detections_pub, 'publish') as mock_pub:
            node._handle_image(_create_mock_image_msg())
            mock_pub.assert_called_once()
            published_msg = mock_pub.call_args[0][0]
            assert len(published_msg.cubes) == 1
            assert published_msg.cubes[0].zone == 'middle'
            assert np.isclose(published_msg.cubes[0].x_mm, 150.0)
            assert np.isclose(published_msg.cubes[0].y_mm, 100.0)
    finally:
        node.destroy_node()
