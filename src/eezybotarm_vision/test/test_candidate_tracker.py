"""Unit tests for CandidateTracker without rclpy dependency."""

from eezybotarm_vision.candidate_tracker import CandidateTracker
from eezybotarm_vision.localized_cube import LocalizedCube
from eezybotarm_vision.yolo_detector import CubeDetection


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


def _make_localized_cube(
    class_id: int,
    class_name: str,
    center: tuple[float, float],
    x_mm: float = 150.0,
    y_mm: float = 100.0,
    zone: str = 'middle',
) -> LocalizedCube:
    cx, cy = center
    det = _make_cube(
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0),
        center=center,
    )
    return LocalizedCube(detection=det, zone=zone, x_mm=x_mm, y_mm=y_mm)


def test_is_match():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_px=25.0)
    base = _make_localized_cube(0, 'green_cube', (20.0, 20.0))
    same = _make_localized_cube(0, 'green_cube', (22.0, 22.0))
    diff_class = _make_localized_cube(1, 'white_cube', (22.0, 22.0))
    far_x = _make_localized_cube(0, 'green_cube', (60.0, 20.0))
    far_y = _make_localized_cube(0, 'green_cube', (20.0, 60.0))

    assert tracker.is_match(base, same)
    assert not tracker.is_match(base, diff_class)
    assert not tracker.is_match(base, far_x)
    assert not tracker.is_match(base, far_y)


def test_update_empty_resets():
    tracker = CandidateTracker(stable_frames_required=3)
    assert tracker.update([]) is None
    assert tracker.candidate is None

    assert tracker.update([_make_localized_cube(0, 'green_cube', (20.0, 20.0))]) is None
    assert tracker.frame_count == 1

    assert tracker.update([]) is None
    assert tracker.candidate is None
    assert tracker.frame_count == 0


def test_update_confirms_after_required_frames():
    tracker = CandidateTracker(stable_frames_required=3)
    frames = [
        _make_localized_cube(0, 'green_cube', (20.0, 20.0)),
        _make_localized_cube(0, 'green_cube', (22.0, 21.0)),
        _make_localized_cube(0, 'green_cube', (21.0, 23.0)),
    ]

    assert tracker.update([frames[0]]) is None
    assert tracker.update([frames[1]]) is None
    assert tracker.frame_count == 2
    assert tracker.update([frames[2]]) == frames[2]
    assert tracker.frame_count == 3


def test_update_replaces_unconfirmed_candidate():
    tracker = CandidateTracker(stable_frames_required=3)
    cube_a = _make_localized_cube(0, 'green_cube', (20.0, 20.0))
    cube_b = _make_localized_cube(1, 'white_cube', (80.0, 80.0))

    tracker.update([cube_a])
    assert tracker.update([cube_b]) is None
    assert tracker.candidate == cube_b
    assert tracker.frame_count == 1


def test_update_prefers_match_over_list_order():
    tracker = CandidateTracker(stable_frames_required=2)
    cube_a1 = _make_localized_cube(0, 'green_cube', (20.0, 20.0))
    cube_a2 = _make_localized_cube(0, 'green_cube', (22.0, 21.0))
    cube_b = _make_localized_cube(1, 'white_cube', (80.0, 80.0))

    tracker.update([cube_a1])
    assert tracker.update([cube_b, cube_a2]) == cube_a2
    assert tracker.frame_count == 2
