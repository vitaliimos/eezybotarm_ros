"""Unit tests for SortingCoordinatorNode state machine and transitions."""

from unittest.mock import patch

from eezybotarm_interfaces.msg import LocalizedCubeArray
from eezybotarm_vision.localized_cube import LocalizedCube
from eezybotarm_vision.sorting_node import SortingCoordinatorNode, SortingState
from eezybotarm_vision.yolo_detector import CubeDetection
import pytest
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter


def _make_test_node(**kwargs) -> SortingCoordinatorNode:
    params = [
        Parameter('stable_frames_required', Parameter.Type.INTEGER, 8),
        Parameter('cooldown_duration_sec', Parameter.Type.DOUBLE, 2.5),
        Parameter('action_name', Parameter.Type.STRING, 'pick_and_place'),
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


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    rclpy.init()
    yield
    rclpy.shutdown()


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


def test_missing_required_parameters_raises():
    with pytest.raises(ParameterUninitializedException):
        SortingCoordinatorNode()


def test_initial_state_and_publishers():
    node = _make_test_node()
    try:
        assert node.state == SortingState.SCANNING
        assert node.subscription.qos_profile.depth == 1
        assert node.status_pub.qos_profile.depth == 1
    finally:
        node.destroy_node()


def test_state_change_logging():
    node = _make_test_node()
    try:
        with patch.object(node.get_logger(), 'info') as mock_log:
            node.state = SortingState.SCANNING
            mock_log.assert_not_called()

            node.state = SortingState.CONFIRMING
            mock_log.assert_called_once_with('State changed: SCANNING -> CONFIRMING')

            mock_log.reset_mock()
            node.state = SortingState.CONFIRMING
            mock_log.assert_not_called()
    finally:
        node.destroy_node()


def test_handle_cooldown():
    node = _make_test_node()
    try:
        node.state = SortingState.COOLDOWN
        node.cooldown_start_time = 100.0
        node.cooldown_duration_sec = 2.5
        node.tracker.candidate = LocalizedCube(
            detection=_make_cube(0, 'green_cube', 0.9, (0, 0, 10, 10), (5.0, 5.0)),
            zone='middle', x_mm=0.0, y_mm=0.0,
        )
        node.tracker.frame_count = 5

        with patch('time.monotonic', return_value=101.0):
            node._handle_cooldown()
            assert node.state == SortingState.COOLDOWN
            assert node.tracker.candidate is not None

        with patch('time.monotonic', return_value=103.0):
            node._handle_cooldown()
            assert node.state == SortingState.SCANNING
            assert node.tracker.candidate is None
            assert node.tracker.frame_count == 0
    finally:
        node.destroy_node()


def test_update_candidate_tracking():
    node = _make_test_node()
    try:
        node.tracker.stable_frames_required = 2
        det = _make_cube(0, 'green_cube', 0.9, (10, 10, 30, 30), (20.0, 20.0))
        item = LocalizedCube(detection=det, zone='middle', x_mm=150.0, y_mm=100.0)

        node._update_candidate_tracking([item])
        assert node.state == SortingState.CONFIRMING
        assert node.tracker.candidate.detection == det
        assert node.tracker.frame_count == 1
        assert node._format_status() == 'CONFIRMING (1/2)'

        with patch.object(node, '_dispatch_pick_and_place') as mock_dispatch:
            node._update_candidate_tracking([item])
            assert node.tracker.frame_count == 2
            mock_dispatch.assert_called_once_with(det, 150.0, 100.0)

        node._update_candidate_tracking([])
        assert node.state == SortingState.SCANNING
        assert node.tracker.candidate is None
        assert node.tracker.frame_count == 0
    finally:
        node.destroy_node()


def test_detections_callback_filters_middle_zone():
    node = _make_test_node()
    try:
        det_left = _make_cube(0, 'green_cube', 0.9, (10, 10, 30, 30), (20.0, 20.0))
        det_mid = _make_cube(1, 'white_cube', 0.95, (50, 50, 70, 70), (60.0, 60.0))
        cube_left = LocalizedCube(detection=det_left, zone='left', x_mm=40.0, y_mm=100.0)
        cube_mid = LocalizedCube(detection=det_mid, zone='middle', x_mm=150.0, y_mm=100.0)

        msg = LocalizedCubeArray()
        msg.cubes = [cube_left.to_msg(), cube_mid.to_msg()]

        node.detections_callback(msg)
        assert node.state == SortingState.CONFIRMING
        assert node.tracker.candidate is not None
        assert node.tracker.candidate.detection.class_name == 'white_cube'
        assert node.tracker.frame_count == 1
    finally:
        node.destroy_node()
