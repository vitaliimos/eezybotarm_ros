"""Sequential workflow runner for action servers."""

import time
from typing import Callable

# A step is (phase name, function returning True on success).
Step = tuple[str, Callable[[], bool]]


def run_workflow(steps: list[Step], goal_handle, action_type, logger):
    """
    Run steps one by one, publishing feedback before each of them.

    Stops at the first step that returns False or raises, or when the client
    cancels. `action_type` is the action class (provides `Feedback`/`Result`).
    """
    started_at = time.monotonic()
    feedback = action_type.Feedback()
    result = action_type.Result()

    def finish(success: bool, error_code: str = ''):
        result.success = success
        result.error_code = error_code
        result.execution_time = time.monotonic() - started_at
        return result

    for index, (phase, action) in enumerate(steps):
        if goal_handle.is_cancel_requested:
            logger.warn(f'Canceled before step {phase}')
            goal_handle.canceled()
            return finish(False, 'Goal canceled by client')

        feedback.current_phase = phase
        feedback.progress_percentage = 100.0 * index / len(steps)
        goal_handle.publish_feedback(feedback)

        try:
            ok = action()
        except Exception as exc:
            logger.error(f'Step {phase} raised: {exc}')
            ok = False

        if goal_handle.is_cancel_requested:
            logger.warn(f'Canceled during step {phase}')
            goal_handle.canceled()
            return finish(False, 'Goal canceled by client')

        if not ok:
            logger.error(f'Step {phase} failed')
            goal_handle.abort()
            return finish(False, f'{phase} failed')

    feedback.current_phase = 'COMPLETE'
    feedback.progress_percentage = 100.0
    goal_handle.publish_feedback(feedback)

    goal_handle.succeed()
    return finish(True)
