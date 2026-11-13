"""Unit tests for SortingCoordinatorNode state machine and transitions."""

from unittest.mock import MagicMock, patch

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
        Parameter('tracking_tolerance_mm', Parameter.Type.DOUBLE, 10.0),
        Parameter('cooldown_duration_sec', Parameter.Type.DOUBLE, 2.5),
        Parameter('action_name', Parameter.Type.STRING, 'pick_and_place'),
        Parameter('table_width_m', Parameter.Type.DOUBLE, 0.300),
        Parameter('table_depth_m', Parameter.Type.DOUBLE, 0.200),
        Parameter('robot_x_offset_m', Parameter.Type.DOUBLE, 0.102),
        Parameter('robot_y_offset_m', Parameter.Type.DOUBLE, 0.0),
        Parameter('table_z_m', Parameter.Type.DOUBLE, 0.095),
        Parameter('cube_height_m', Parameter.Type.DOUBLE, 0.025),
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


def test_reset_to_scanning():
    node = _make_test_node()
    try:
        node.state = SortingState.EXECUTING
        node.current_action_phase = 'SENDING_GOAL'
        node.tracker.candidate = LocalizedCube(
            detection=_make_cube(0, 'green_cube', 0.9, (0, 0, 10, 10), (5.0, 5.0)),
            zone='middle',
            x_mm=0.0,
            y_mm=0.0,
        )
        node.tracker.frame_count = 3

        node._reset_to_scanning()
        assert node.state == SortingState.SCANNING
        assert node.current_action_phase == ''
        assert node.tracker.candidate is None
        assert node.tracker.frame_count == 0
        assert node._last_published_status == 'SCANNING'
    finally:
        node.destroy_node()


def test_cooldown_one_shot_timer():
    node = _make_test_node()
    try:
        assert node._cooldown_timer is None
        node.state = SortingState.COOLDOWN
        node.cooldown_duration_sec = 2.5
        node.tracker.candidate = LocalizedCube(
            detection=_make_cube(0, 'green_cube', 0.9, (0, 0, 10, 10), (5.0, 5.0)),
            zone='middle',
            x_mm=0.0,
            y_mm=0.0,
        )
        node.tracker.frame_count = 5

        node._start_cooldown_timer()
        assert node._cooldown_timer is not None
        assert node._cooldown_timer.timer_period_ns == 2_500_000_000

        with patch.object(
            node, '_reset_to_scanning', wraps=node._reset_to_scanning
        ) as mock_reset:
            node._on_cooldown_expired()
            mock_reset.assert_called_once()
            assert node._cooldown_timer is None
            assert node.state == SortingState.SCANNING
            assert node.tracker.candidate is None
            assert node.tracker.frame_count == 0
    finally:
        node.destroy_node()


def test_on_cooldown_expired_ignores_non_cooldown_state():
    node = _make_test_node()
    try:
        node.state = SortingState.SCANNING
        with patch.object(node, '_reset_to_scanning') as mock_reset:
            node._on_cooldown_expired()
            mock_reset.assert_not_called()
    finally:
        node.destroy_node()


def test_action_result_callback_starts_cooldown():
    node = _make_test_node()
    try:
        node.state = SortingState.EXECUTING
        node.current_action_phase = 'MOVING'

        future = MagicMock()
        future.result.return_value.result.success = True
        future.result.return_value.result.execution_time = 1.2
        future.result.return_value.result.error_code = ''

        with patch.object(node, '_start_cooldown_timer') as mock_start:
            node._action_result_callback(future)
            mock_start.assert_called_once()
            assert node.state == SortingState.COOLDOWN
            assert node.current_action_phase == ''
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

        with patch.object(
            node, '_reset_to_scanning', wraps=node._reset_to_scanning
        ) as mock_reset:
            node._update_candidate_tracking([])
            mock_reset.assert_called_once()
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
        cube_left = LocalizedCube(
            detection=det_left, zone='left', x_mm=40.0, y_mm=100.0
        )
        cube_mid = LocalizedCube(
            detection=det_mid, zone='middle', x_mm=150.0, y_mm=100.0
        )

        msg = LocalizedCubeArray()
        msg.cubes = [cube_left.to_msg(), cube_mid.to_msg()]

        node.detections_callback(msg)
        assert node.state == SortingState.CONFIRMING
        assert node.tracker.candidate is not None
        assert node.tracker.candidate.detection.class_name == 'white_cube'
        assert node.tracker.frame_count == 1
    finally:
        node.destroy_node()
