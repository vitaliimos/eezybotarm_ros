"""Unit tests for SortingCoordinatorNode state machine and transitions."""

from unittest.mock import MagicMock, patch

from eezybotarm_vision.debug_visualizer import LocalizedCube
from eezybotarm_vision.sorting_node import SortingCoordinatorNode, SortingState
from eezybotarm_vision.yolo_detector import CubeDetection
from eezybotarm_vision.zone_detector import Zone, ZoneDetectionResult
import numpy as np
import pytest
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image


def _make_test_node(**kwargs) -> SortingCoordinatorNode:
    params = [
        Parameter('weights_path', Parameter.Type.STRING, '/usr/share/models/model.engine'),
        Parameter('conf_threshold', Parameter.Type.DOUBLE, 0.5),
        Parameter('iou_threshold', Parameter.Type.DOUBLE, 0.45),
        Parameter('stable_frames_required', Parameter.Type.INTEGER, 8),
        Parameter('cooldown_duration_sec', Parameter.Type.DOUBLE, 2.5),
        Parameter('debug_image_jpeg_quality', Parameter.Type.INTEGER, 80),
        Parameter('action_name', Parameter.Type.STRING, 'pick_and_place'),
        Parameter('recalibration_interval_sec', Parameter.Type.DOUBLE, 30.0),
        Parameter('calibration_smoothing_alpha', Parameter.Type.DOUBLE, 0.25),
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('robot_x_offset_m', Parameter.Type.DOUBLE, 0.102),
        Parameter('robot_y_offset_m', Parameter.Type.DOUBLE, 0.0),
        Parameter('table_z_m', Parameter.Type.DOUBLE, 0.095),
        Parameter('cube_height_m', Parameter.Type.DOUBLE, 0.025),
        Parameter('sticker_width_m', Parameter.Type.DOUBLE, 0.030),
        Parameter('sticker_height_m', Parameter.Type.DOUBLE, 0.015),
    ]
    return SortingCoordinatorNode(parameter_overrides=params, **kwargs)


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


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    rclpy.init()
    yield
    rclpy.shutdown()


def _create_mock_image_msg() -> Image:
    msg = Image()
    msg.height = 100
    msg.width = 100
    msg.encoding = 'bgr8'
    msg.step = 300
    msg.data = np.zeros((100, 100, 3), dtype=np.uint8).tobytes()
    return msg


def _make_cube(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    center: tuple[float, float],
    angle: float = 0.0,
) -> CubeDetection:
    x1, y1, x2, y2 = bbox
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return CubeDetection(class_id, class_name, confidence, bbox, center, angle, corners)


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_missing_required_parameters_raises(mock_detector):
    with pytest.raises(ParameterUninitializedException):
        SortingCoordinatorNode()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_initial_state_and_publishers(mock_detector):
    node = _make_test_node()
    try:
        assert node.state == SortingState.CALIBRATING
        assert node.calibration is None
        assert node.visualizer.debug_image_pub.qos_profile.depth == 1
        assert node.visualizer.debug_compressed_pub.qos_profile.depth == 1
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_state_change_logging(mock_detector):
    node = _make_test_node()
    try:
        with patch.object(node.get_logger(), 'info') as mock_log:
            node.state = SortingState.CALIBRATING
            mock_log.assert_not_called()

            node.state = SortingState.SCANNING
            mock_log.assert_called_once_with('State changed: CALIBRATING -> SCANNING')

            mock_log.reset_mock()
            node.state = SortingState.SCANNING
            mock_log.assert_not_called()
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.detect_zones')
@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_handle_image_calibrating_success(mock_detector, mock_detect_zones):
    mock_calib = _make_dummy_calibration()
    mock_detect_zones.return_value = mock_calib

    node = _make_test_node()
    try:
        assert node.state == SortingState.CALIBRATING
        node._handle_image(_create_mock_image_msg())

        assert node.calibration is mock_calib
        assert node.state == SortingState.SCANNING
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.detect_zones')
@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_handle_image_calibrating_failure_bypasses_yolo(mock_detector, mock_detect_zones):
    mock_detect_zones.return_value = None

    node = _make_test_node()
    try:
        assert node.state == SortingState.CALIBRATING
        node._handle_image(_create_mock_image_msg())

        assert node.calibration is None
        assert node.state == SortingState.CALIBRATING
        node.detector.predict.assert_not_called()
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_convert_image(mock_detector):
    node = _make_test_node()
    try:
        valid_msg = _create_mock_image_msg()
        frame = node._convert_image(valid_msg)
        assert frame is not None
        assert frame.shape == (100, 100, 3)

        invalid_msg = Image()
        invalid_msg.encoding = 'invalid_encoding'
        invalid_msg.data = b'invalid'
        assert node._convert_image(invalid_msg) is None
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_handle_cooldown(mock_detector):
    node = _make_test_node()
    try:
        node.state = SortingState.COOLDOWN
        node.cooldown_start_time = 100.0
        node.cooldown_duration_sec = 2.5
        node.candidate_cube = _make_cube(0, 'green_cube', 0.9, (0, 0, 10, 10), (5.0, 5.0))
        node.candidate_frames_count = 5

        with patch('time.monotonic', return_value=101.0):
            node._handle_cooldown()
            assert node.state == SortingState.COOLDOWN
            assert node.candidate_cube is not None

        with patch('time.monotonic', return_value=103.0):
            node._handle_cooldown()
            assert node.state == SortingState.SCANNING
            assert node.candidate_cube is None
            assert node.candidate_frames_count == 0
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_is_cube_match(mock_detector):
    node = _make_test_node()
    try:
        base_cube = _make_cube(0, 'green_cube', 0.9, (10, 10, 30, 30), (20.0, 20.0))
        same_cube = _make_cube(0, 'green_cube', 0.85, (12, 12, 32, 32), (22.0, 22.0))
        diff_class = _make_cube(1, 'white_cube', 0.85, (12, 12, 32, 32), (22.0, 22.0))
        far_cube = _make_cube(0, 'green_cube', 0.85, (80, 80, 100, 100), (90.0, 90.0))

        assert node._is_cube_match(base_cube, same_cube) is True
        assert node._is_cube_match(base_cube, diff_class) is False
        assert node._is_cube_match(base_cube, far_cube) is False
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_update_candidate_tracking(mock_detector):
    node = _make_test_node()
    try:
        node.stable_frames_required = 2
        det = _make_cube(0, 'green_cube', 0.9, (10, 10, 30, 30), (20.0, 20.0))
        item = LocalizedCube(detection=det, zone='middle', x_mm=150.0, y_mm=100.0)

        node._update_candidate_tracking([item])
        assert node.state == SortingState.CONFIRMING
        assert node.candidate_cube == det
        assert node.candidate_frames_count == 1

        with patch.object(node, '_dispatch_pick_and_place') as mock_dispatch:
            node._update_candidate_tracking([item])
            assert node.candidate_frames_count == 2
            mock_dispatch.assert_called_once_with(det, 150.0, 100.0)

        node._update_candidate_tracking([])
        assert node.state == SortingState.SCANNING
        assert node.candidate_cube is None
        assert node.candidate_frames_count == 0
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_publish_debug_image(mock_detector):
    node = _make_test_node()
    try:
        dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        header = MagicMock()

        with patch.object(node.visualizer, 'publish') as mock_viz_publish:
            node._publish_debug_image(dummy_frame, [], header)
            mock_viz_publish.assert_called_once_with(
                frame=dummy_frame,
                header=header,
                state_text=node.state.value,
                calibration=node.calibration,
                detections=[],
                status_extra='',
            )
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.detect_zones')
@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_periodic_recalibration_in_scanning_success(mock_detector, mock_detect_zones):
    mock_calib1 = _make_dummy_calibration(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    )
    mock_calib2 = _make_dummy_calibration(
        np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    )
    mock_detector.return_value.predict.return_value = []

    node = _make_test_node()
    try:
        node.calibration = mock_calib1
        node.state = SortingState.SCANNING
        node.recalibration_interval_sec = 1.0
        node.last_recalibration_time = 0.0

        mock_detect_zones.return_value = mock_calib2
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        node._handle_scanning(dummy_frame)

        mock_detect_zones.assert_called_once()
        assert node.calibration is not None
        assert np.allclose(node.calibration.field_corners_px[0], [2.5, 2.5], atol=1e-1)
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.detect_zones')
@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_periodic_recalibration_occlusion_preserves_calibration(mock_detector, mock_detect_zones):
    mock_calib1 = _make_dummy_calibration()
    mock_detector.return_value.predict.return_value = []

    node = _make_test_node()
    try:
        node.calibration = mock_calib1
        node.state = SortingState.SCANNING
        node.recalibration_interval_sec = 1.0
        node.last_recalibration_time = 0.0

        mock_detect_zones.return_value = None
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        node._handle_scanning(dummy_frame)

        mock_detect_zones.assert_called_once()
        assert node.calibration is mock_calib1
        assert node.state == SortingState.SCANNING
    finally:
        node.destroy_node()
