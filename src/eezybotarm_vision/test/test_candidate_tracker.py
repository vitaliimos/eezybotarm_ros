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
    x_mm: float = 150.0,
    y_mm: float = 100.0,
    confidence: float = 0.9,
    center: tuple[float, float] = (20.0, 20.0),
    zone: str = 'middle',
) -> LocalizedCube:
    cx, cy = center
    det = _make_cube(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0),
        center=center,
    )
    return LocalizedCube(detection=det, zone=zone, x_mm=x_mm, y_mm=y_mm)


def test_is_match_euclidean_mm():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_mm=10.0)
    base = _make_localized_cube(0, 'green_cube', x_mm=150.0, y_mm=100.0)
    # Distance = sqrt(6^2 + 7^2) = ~9.22 < 10.0
    same = _make_localized_cube(0, 'green_cube', x_mm=156.0, y_mm=107.0)
    diff_class = _make_localized_cube(1, 'white_cube', x_mm=151.0, y_mm=101.0)
    # Per-axis < 10.0, but Euclidean = sqrt(8^2 + 8^2) = ~11.31 >= 10.0
    diagonal_far = _make_localized_cube(0, 'green_cube', x_mm=158.0, y_mm=108.0)

    assert tracker.is_match(base, same)
    assert not tracker.is_match(base, diff_class)
    assert not tracker.is_match(base, diagonal_far)


def test_update_empty_resets():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_mm=10.0)
    assert tracker.update([]) is None
    assert tracker.candidate is None

    assert tracker.update([_make_localized_cube(0, 'green_cube')]) is None
    assert tracker.frame_count == 1

    assert tracker.update([]) is None
    assert tracker.candidate is None
    assert tracker.frame_count == 0


def test_update_confirms_after_required_frames():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_mm=10.0)
    frames = [
        _make_localized_cube(0, 'green_cube', x_mm=150.0, y_mm=100.0),
        _make_localized_cube(0, 'green_cube', x_mm=152.0, y_mm=101.0),
        _make_localized_cube(0, 'green_cube', x_mm=151.0, y_mm=103.0),
    ]

    assert tracker.update([frames[0]]) is None
    assert tracker.update([frames[1]]) is None
    assert tracker.frame_count == 2
    assert tracker.update([frames[2]]) == frames[2]
    assert tracker.frame_count == 3


def test_initial_acquisition_picks_most_confident():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_mm=10.0)
    low_conf = _make_localized_cube(
        0, 'green_cube', x_mm=100.0, y_mm=100.0, confidence=0.6
    )
    high_conf = _make_localized_cube(
        1, 'white_cube', x_mm=200.0, y_mm=100.0, confidence=0.95
    )

    assert tracker.update([low_conf, high_conf]) is None
    assert tracker.candidate == high_conf
    assert tracker.frame_count == 1


def test_lost_candidate_adopts_nearest_cube():
    tracker = CandidateTracker(stable_frames_required=3, tolerance_mm=10.0)
    cube_a = _make_localized_cube(0, 'green_cube', x_mm=150.0, y_mm=100.0)
    far_cube = _make_localized_cube(
        1, 'white_cube', x_mm=220.0, y_mm=100.0, confidence=0.99
    )
    near_cube = _make_localized_cube(
        1, 'white_cube', x_mm=165.0, y_mm=100.0, confidence=0.7
    )

    tracker.update([cube_a])
    assert tracker.update([far_cube, near_cube]) is None
    assert tracker.candidate == near_cube
    assert tracker.frame_count == 1


def test_update_prefers_nearest_match_over_list_order():
    tracker = CandidateTracker(stable_frames_required=2, tolerance_mm=10.0)
    cube_a1 = _make_localized_cube(0, 'green_cube', x_mm=150.0, y_mm=100.0)
    cube_a_further = _make_localized_cube(0, 'green_cube', x_mm=158.0, y_mm=100.0)
    cube_a_closer = _make_localized_cube(0, 'green_cube', x_mm=151.0, y_mm=100.0)
    cube_b = _make_localized_cube(1, 'white_cube', x_mm=220.0, y_mm=100.0)

    tracker.update([cube_a1])
    assert tracker.update([cube_b, cube_a_further, cube_a_closer]) == cube_a_closer
    assert tracker.frame_count == 2
