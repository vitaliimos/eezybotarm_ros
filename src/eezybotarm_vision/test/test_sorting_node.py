"""Unit tests for SortingCoordinatorNode state machine and transitions."""

from unittest.mock import MagicMock, patch

from eezybotarm_vision.sorting_node import SortingCoordinatorNode, SortingState
from eezybotarm_vision.yolo_detector import CubeDetection
from eezybotarm_vision.zone_detector import ZoneDetectionResult
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterUninitializedException


def _make_test_node(**kwargs) -> SortingCoordinatorNode:
    params = [
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('robot_x_offset_m', Parameter.Type.DOUBLE, 0.127),
        Parameter('robot_y_offset_m', Parameter.Type.DOUBLE, 0.0),
        Parameter('table_z_m', Parameter.Type.DOUBLE, 0.075),
        Parameter('cube_height_m', Parameter.Type.DOUBLE, 0.025),
    ]
    return SortingCoordinatorNode(parameter_overrides=params, **kwargs)


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    """Manage rclpy init and shutdown for each test."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _create_mock_image_msg() -> Image:
    """Create a dummy Image message for testing."""
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
    """Verify that omitting required workspace parameters raises ParameterUninitializedException."""
    with pytest.raises(ParameterUninitializedException):
        SortingCoordinatorNode()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_initial_state_and_properties(mock_detector):
    """Verify initial state is CALIBRATING and properties work as expected."""
    node = _make_test_node()
    try:
        assert node.state == SortingState.CALIBRATING
        assert node.calibration is None
        assert node.active_calibration is None
        assert node.debug_image_pub.qos_profile.depth == 1
        assert node.debug_compressed_pub.qos_profile.depth == 1
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_state_change_logging(mock_detector):
    """Verify state setter logs changes only when the state actually changes."""
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
    """Verify transition from CALIBRATING to SCANNING on zone detection."""
    mock_calib = ZoneDetectionResult(
        zones={'left': MagicMock(), 'middle': MagicMock(), 'right': MagicMock()},
        field_corners_px=np.zeros((4, 2), dtype=np.float32),
        homography_matrix=np.eye(3, dtype=np.float32),
        field_width_mm=300.0,
        field_height_mm=200.0,
    )
    mock_detect_zones.return_value = mock_calib

    node = _make_test_node()
    try:
        assert node.state == SortingState.CALIBRATING
        node._handle_image(_create_mock_image_msg())

        assert node.calibration is mock_calib
        assert node.active_calibration is mock_calib
        assert node.state == SortingState.SCANNING
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.detect_zones')
@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_handle_image_calibrating_failure_bypasses_yolo(mock_detector, mock_detect_zones):
    """Verify node stays in CALIBRATING without calling YOLO if zones not detected."""
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
    """Verify _convert_image properly handles valid and invalid images."""
    node = _make_test_node()
    try:
        valid_msg = _create_mock_image_msg()
        frame = node._convert_image(valid_msg)
        assert frame is not None
        assert frame.shape == (100, 100, 3)

        # Invalid message encoding to trigger conversion exception
        invalid_msg = Image()
        invalid_msg.encoding = 'invalid_encoding'
        invalid_msg.data = b'invalid'
        assert node._convert_image(invalid_msg) is None
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_handle_cooldown(mock_detector):
    """Verify cooldown countdown and transition to SCANNING."""
    node = _make_test_node()
    try:
        node.state = SortingState.COOLDOWN
        node.cooldown_start_time = 100.0
        node.cooldown_duration_sec = 2.5
        node.candidate_cube = _make_cube(0, 'green_cube', 0.9, (0, 0, 10, 10), (5.0, 5.0))
        node.candidate_frames_count = 5

        # Timer not reached yet
        with patch('time.monotonic', return_value=101.0):
            node._handle_cooldown()
            assert node.state == SortingState.COOLDOWN
            assert node.candidate_cube is not None

        # Timer expired
        with patch('time.monotonic', return_value=103.0):
            node._handle_cooldown()
            assert node.state == SortingState.SCANNING
            assert node.candidate_cube is None
            assert node.candidate_frames_count == 0
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_is_cube_match(mock_detector):
    """Verify cube matching based on class ID and pixel distance."""
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
    """Verify candidate stability counting, dispatching, and resetting."""
    node = _make_test_node()
    try:
        node.stable_frames_required = 2
        det = _make_cube(0, 'green_cube', 0.9, (10, 10, 30, 30), (20.0, 20.0))
        item = (det, 'middle', 150.0, 100.0)

        # 1. First detection sets candidate and count to 1
        node._update_candidate_tracking([item])
        assert node.state == SortingState.CONFIRMING
        assert node.candidate_cube == det
        assert node.candidate_frames_count == 1

        # 2. Second detection reaches stability threshold and dispatches pick-and-place
        with patch.object(node, '_dispatch_pick_and_place') as mock_dispatch:
            node._update_candidate_tracking([item])
            assert node.candidate_frames_count == 2
            mock_dispatch.assert_called_once_with(det, 150.0, 100.0)

        # 3. No detection in middle resets candidate
        node._update_candidate_tracking([])
        assert node.state == SortingState.SCANNING
        assert node.candidate_cube is None
        assert node.candidate_frames_count == 0
    finally:
        node.destroy_node()


@patch('eezybotarm_vision.sorting_node.YoloCubeDetector')
def test_publish_debug_image(mock_detector):
    """Verify debug image is published via visualizer."""
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
