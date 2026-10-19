"""Unit tests for sequential workflow runner."""

from unittest.mock import MagicMock

from eezybotarm_control.workflow import run_workflow
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
