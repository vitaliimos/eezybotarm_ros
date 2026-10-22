import math

from eezybotarm_vision.yolo_detector import YoloCubeDetector
import numpy as np


def test_letterbox_preserves_aspect_ratio_and_pads():
    """Verify letterbox resizes to exact target shape and computes proper scale."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    padded, scale, left, top = YoloCubeDetector._letterbox(img, new_shape=(640, 640))

    assert padded.shape == (640, 640, 3)
    assert math.isclose(scale, 640 / 400, rel_tol=1e-5)
    assert left == 0.0
    expected_top = (640 - round(300 * (640 / 400))) / 2.0
    assert abs(top - expected_top) <= 1.0


def test_postprocess_yolo_obb_format():
    """Verify postprocessing decodes raw YOLO-OBB output with correct NMS and coordinates."""
    detector = YoloCubeDetector.__new__(YoloCubeDetector)
    detector.conf_threshold = 0.5
    detector.iou_threshold = 0.45

    # Create dummy output tensor: (1, 7, 8400)
    # Channels: [cx, cy, w, h, score_green, score_white, angle]
    out = np.zeros((1, 7, 8400), dtype=np.float32)

    # Box 1: White cube at letterbox center (320, 320), size (50, 50), conf 0.85
    out[0, 0, 100] = 320.0
    out[0, 1, 100] = 320.0
    out[0, 2, 100] = 50.0
    out[0, 3, 100] = 50.0
    out[0, 4, 100] = 0.05  # green
    out[0, 5, 100] = 0.85  # white
    out[0, 6, 100] = 0.0   # angle

    # Box 2: Duplicate of Box 1 with lower confidence (should be suppressed by NMS)
    out[0, 0, 101] = 322.0
    out[0, 1, 101] = 321.0
    out[0, 2, 101] = 50.0
    out[0, 3, 101] = 50.0
    out[0, 4, 101] = 0.02
    out[0, 5, 101] = 0.65
    out[0, 6, 101] = 0.0

    # Box 3: Below confidence threshold (should be filtered out)
    out[0, 0, 200] = 100.0
    out[0, 1, 200] = 100.0
    out[0, 2, 200] = 40.0
    out[0, 3, 200] = 40.0
    out[0, 4, 200] = 0.30  # < 0.5
    out[0, 5, 200] = 0.10
    out[0, 6, 200] = 0.0

    # Box 4: Green cube elsewhere, conf 0.90
    out[0, 0, 300] = 200.0
    out[0, 1, 300] = 150.0
    out[0, 2, 300] = 60.0
    out[0, 3, 300] = 60.0
    out[0, 4, 300] = 0.90  # green
    out[0, 5, 300] = 0.05
    out[0, 6, 300] = 0.25  # angle rad

    scale = 1.0
    pad_x = 0.0
    pad_y = 0.0
    orig_w = 640
    orig_h = 640

    detections = detector._postprocess(out, scale, pad_x, pad_y, orig_w, orig_h)

    assert len(detections) == 2, f'Expected 2 detections after NMS, got {len(detections)}'

    # Check top detection is green cube
    d0 = detections[0]
    assert d0.class_id == 0
    assert d0.class_name == 'green_cube'
    assert math.isclose(d0.confidence, 0.90, rel_tol=1e-5)
    assert math.isclose(d0.center_px[0], 200.0, rel_tol=1e-5)
    assert math.isclose(d0.center_px[1], 150.0, rel_tol=1e-5)
    assert math.isclose(d0.angle, 0.25, rel_tol=1e-5)
    assert len(d0.corners) == 4

    # Check second detection is white cube (box 2 suppressed by NMS)
    d1 = detections[1]
    assert d1.class_id == 1
    assert d1.class_name == 'white_cube'
    assert math.isclose(d1.confidence, 0.85, rel_tol=1e-5)
    assert math.isclose(d1.center_px[0], 320.0, rel_tol=1e-5)
    assert math.isclose(d1.center_px[1], 320.0, rel_tol=1e-5)


def test_postprocess_empty_and_none():
    """Verify postprocessing handles empty, None, and all-zero outputs gracefully."""
    detector = YoloCubeDetector.__new__(YoloCubeDetector)
    detector.conf_threshold = 0.5
    detector.iou_threshold = 0.45

    assert detector._postprocess(None, 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess([], 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess(np.zeros((1, 7, 8400)), 1.0, 0, 0, 640, 640) == []
