"""Unit tests for sequential workflow runner."""

import threading
from unittest.mock import MagicMock

from eezybotarm_control.pick_and_place_server import _offset_z, PickAndPlaceActionServer
from eezybotarm_control.workflow import run_workflow
from geometry_msgs.msg import PoseStamped
import pytest
from rclpy.action import CancelResponse, GoalResponse


class FakeAction:
    class Feedback:
        def __init__(self):
            self.current_phase = ''
            self.progress_percentage = 0.0

    class Result:
        def __init__(self):
            self.success = False
            self.execution_time = 0.0
            self.error_code = ''


def make_goal_handle():
    goal_handle = MagicMock()
    goal_handle.is_cancel_requested = False
    return goal_handle


def run(steps, goal_handle):
    return run_workflow(steps, goal_handle, FakeAction, MagicMock())


def test_successful_execution():
    """All steps run in order with correct feedback and result."""
    actions = [MagicMock(return_value=True) for _ in range(3)]
    steps = [(f'STEP_{i + 1}', action) for i, action in enumerate(actions)]
    goal_handle = make_goal_handle()

    published = []
    goal_handle.publish_feedback.side_effect = lambda fb: published.append(
        (fb.current_phase, fb.progress_percentage))

    result = run(steps, goal_handle)

    for action in actions:
        action.assert_called_once()
    goal_handle.succeed.assert_called_once()
    assert result.success is True
    assert result.error_code == ''
    assert result.execution_time >= 0.0

    # 3 steps + 1 completion feedback
    assert published == [
        ('STEP_1', pytest.approx(0.0)),
        ('STEP_2', pytest.approx(100.0 / 3)),
        ('STEP_3', pytest.approx(200.0 / 3)),
        ('COMPLETE', 100.0),
    ]


def test_step_failure_aborts():
    """A failing step aborts the workflow and reports the error."""
    action_1 = MagicMock(return_value=True)
    action_2 = MagicMock(return_value=False)
    action_3 = MagicMock(return_value=True)
    steps = [('STEP_1', action_1), ('STEP_2', action_2), ('STEP_3', action_3)]
    goal_handle = make_goal_handle()

    result = run(steps, goal_handle)

    action_1.assert_called_once()
    action_2.assert_called_once()
    action_3.assert_not_called()
    goal_handle.abort.assert_called_once()
    goal_handle.succeed.assert_not_called()
    assert result.success is False
    assert result.error_code == 'STEP_2 failed'


def test_step_exception_aborts():
    """An unexpected exception in a step aborts gracefully."""
    steps = [('STEP_1', MagicMock(side_effect=RuntimeError('Planning scene crash')))]
    goal_handle = make_goal_handle()
    logger = MagicMock()

    result = run_workflow(steps, goal_handle, FakeAction, logger)

    goal_handle.abort.assert_called_once()
    assert result.success is False
    assert result.error_code == 'STEP_1 failed'
    logger.error.assert_called()


def test_cancel_requested():
    """Cancellation stops execution and marks the goal as canceled."""
    goal_handle = make_goal_handle()

    def cancel_during_step_1():
        goal_handle.is_cancel_requested = True
        return True

    action_1 = MagicMock(side_effect=cancel_during_step_1)
    action_2 = MagicMock(return_value=True)

    result = run([('STEP_1', action_1), ('STEP_2', action_2)], goal_handle)

    action_1.assert_called_once()
    action_2.assert_not_called()
    goal_handle.canceled.assert_called_once()
    goal_handle.abort.assert_not_called()
    goal_handle.succeed.assert_not_called()
    assert result.success is False
    assert result.error_code == 'Goal canceled by client'


def test_cancel_during_step_with_failure():
    """If a step fails because cancel was requested, mark canceled not aborted."""
    goal_handle = make_goal_handle()

    def cancel_and_fail_during_step():
        goal_handle.is_cancel_requested = True
        return False

    action = MagicMock(side_effect=cancel_and_fail_during_step)
    result = run([('STEP_1', action)], goal_handle)

    action.assert_called_once()
    goal_handle.canceled.assert_called_once()
    goal_handle.abort.assert_not_called()
    assert result.success is False
    assert result.error_code == 'Goal canceled by client'


def test_goal_callback_accepts_when_idle_and_rejects_when_pending():
    """goal_callback accepts first goal, sets pending flag, and rejects when pending."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    server._current_goal = None
    server._is_goal_pending = False
    server.get_logger = MagicMock()

    request = MagicMock()
    request.object_class = 'cube'
    request.target_pose.pose.position.x = 0.1
    request.target_pose.pose.position.y = 0.2

    resp1 = server.goal_callback(request)
    assert resp1 == GoalResponse.ACCEPT
    assert server._is_goal_pending is True

    resp2 = server.goal_callback(request)
    assert resp2 == GoalResponse.REJECT


def test_goal_callback_rejects_when_active_goal_exists():
    """goal_callback rejects if a goal is already executing/active."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    server._current_goal = MagicMock(is_active=True)
    server._is_goal_pending = False
    server.get_logger = MagicMock()

    request = MagicMock()
    request.object_class = 'cube'
    request.target_pose.pose.position.x = 0.1
    request.target_pose.pose.position.y = 0.2

    resp = server.goal_callback(request)
    assert resp == GoalResponse.REJECT


def test_handle_accepted_callback_sets_active_goal():
    """handle_accepted_callback stores current goal and clears pending flag."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    server._current_goal = None
    server._is_goal_pending = True

    goal_handle = MagicMock()
    server.handle_accepted_callback(goal_handle)

    assert server._current_goal == goal_handle
    assert server._is_goal_pending is False
    goal_handle.execute.assert_called_once()


def test_cancel_callback_stops_trajectory_for_active_goal():
    """cancel_callback invokes stop_execution on TrajectoryExecutionManager for active goal."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    goal_handle = MagicMock(is_active=True)
    server._current_goal = goal_handle
    server.get_logger = MagicMock()
    tem = MagicMock()
    server.moveit = MagicMock()
    server.moveit.get_trajectory_execution_manager.return_value = tem

    resp = server.cancel_callback(goal_handle)
    assert resp == CancelResponse.ACCEPT
    tem.stop_execution.assert_called_once()


def test_cancel_callback_rejects_for_unknown_or_inactive_goal():
    """cancel_callback rejects cancellation when handle does not match current active goal."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    active_handle = MagicMock(is_active=True)
    server._current_goal = active_handle
    server.get_logger = MagicMock()
    tem = MagicMock()
    server.moveit = MagicMock()
    server.moveit.get_trajectory_execution_manager.return_value = tem

    unknown_handle = MagicMock(is_active=True)
    resp = server.cancel_callback(unknown_handle)
    assert resp == CancelResponse.REJECT
    tem.stop_execution.assert_not_called()

    # Inactive current handle
    active_handle.is_active = False
    resp2 = server.cancel_callback(active_handle)
    assert resp2 == CancelResponse.REJECT
    tem.stop_execution.assert_not_called()


def test_execute_callback_resets_goal_state():
    """execute_callback always resets _current_goal and _is_goal_pending in finally block."""
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server._goal_lock = threading.Lock()
    goal_handle = make_goal_handle()
    server._current_goal = goal_handle
    server._is_goal_pending = True
    server.approach_distance = 0.03
    server.get_logger = MagicMock()
    server._build_steps = MagicMock(side_effect=RuntimeError('step build failed'))

    with pytest.raises(RuntimeError):
        server.execute_callback(goal_handle)

    assert server._current_goal is None
    assert server._is_goal_pending is False


def _make_pose(x: float, y: float, z: float, frame_id: str = 'base_link') -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.w = 1.0
    return pose


def _make_server_stub():
    server = PickAndPlaceActionServer.__new__(PickAndPlaceActionServer)
    server.arm = 'arm_component'
    server.gripper = 'gripper_component'
    calls = []
    server._move_to_named = MagicMock(
        side_effect=lambda comp, name: (calls.append(('named', comp, name)) or True)
    )
    server._move_to_pose = MagicMock(
        side_effect=lambda pose: (
            calls.append((
                'pose',
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )) or True
        )
    )
    return server, calls


def test_offset_z_returns_shifted_copy():
    """_offset_z shifts along the approach vector without mutating the original PoseStamped."""
    original = _make_pose(0.15, -0.05, 0.02, frame_id='base_link')
    shifted = _offset_z(original, 0.04)

    assert shifted is not original
    assert original.pose.position.x == pytest.approx(0.15)
    assert original.pose.position.z == pytest.approx(0.02)
    assert shifted.header.frame_id == 'base_link'
    assert shifted.pose.position.x == pytest.approx(0.19)
    assert shifted.pose.position.y == pytest.approx(-0.05)
    assert shifted.pose.position.z == pytest.approx(0.06)


def test_build_steps():
    """_build_steps always builds the full 10-phase sequence."""
    server, calls = _make_server_stub()
    pick_pose = _make_pose(0.15, 0.0, 0.02)
    place_pose = _make_pose(0.15, -0.10, 0.02)

    steps = server._build_steps(pick_pose, place_pose, approach_distance=0.04)
    phases = [phase for phase, _ in steps]

    assert phases == [
        'OPENING_GRIPPER',
        'MOVING_TO_PRE_GRASP',
        'APPROACHING_OBJECT',
        'CLOSING_GRIPPER',
        'LIFTING_OBJECT',
        'MOVING_TO_PRE_PLACE',
        'PLACING_OBJECT',
        'OPENING_GRIPPER',
        'RETREATING_FROM_PLACE',
        'MOVING_TO_HOME',
    ]

    for _, action in steps:
        assert action() is True

    assert calls == [
        ('named', 'gripper_component', 'open'),
        ('pose', pytest.approx(0.19), pytest.approx(0.0), pytest.approx(0.06)),
        ('pose', pytest.approx(0.15), pytest.approx(0.0), pytest.approx(0.02)),
        ('named', 'gripper_component', 'close'),
        ('pose', pytest.approx(0.19), pytest.approx(0.0), pytest.approx(0.06)),
        ('pose', pytest.approx(0.19), pytest.approx(-0.10), pytest.approx(0.06)),
        ('pose', pytest.approx(0.15), pytest.approx(-0.10), pytest.approx(0.02)),
        ('named', 'gripper_component', 'open'),
        ('pose', pytest.approx(0.19), pytest.approx(-0.10), pytest.approx(0.06)),
        ('named', 'arm_component', 'home'),
    ]
