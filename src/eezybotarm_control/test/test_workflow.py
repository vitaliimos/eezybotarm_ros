"""Unit tests for sequential workflow runner."""

from unittest.mock import MagicMock

from eezybotarm_control.pick_and_place_server import _offset_z, PickAndPlaceActionServer
from eezybotarm_control.workflow import run_workflow
from geometry_msgs.msg import PoseStamped
import pytest


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
