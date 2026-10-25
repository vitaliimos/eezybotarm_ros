"""Unit tests for YOLO cube detector and inference runners."""

import math

from eezybotarm_vision.yolo_detector import YoloCubeDetector
import numpy as np
import pytest


def test_letterbox_preserves_aspect_ratio_and_pads():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    padded, scale, left, top = YoloCubeDetector._letterbox(img, new_shape=(640, 640))

    assert padded.shape == (640, 640, 3)
    assert math.isclose(scale, 640 / 400, rel_tol=1e-5)
    assert left == 0.0
    expected_top = (640 - round(300 * (640 / 400))) / 2.0
    assert abs(top - expected_top) <= 1.0


def test_letterbox_16_9_inverse_mapping():
    orig_h, orig_w = 576, 1024
    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = YoloCubeDetector._letterbox(img, (640, 640))

    assert padded.shape == (640, 640, 3)
    assert abs(scale - (640.0 / 1024.0)) < 1e-6
    assert abs(pad_x - 0.0) < 1e-6
    assert abs(pad_y - 140.0) < 1e-6

    det = object.__new__(YoloCubeDetector)
    det.conf_threshold = 0.5
    fake_out = np.array([[[320.0, 320.0, 64.0, 64.0, 0.98, 0.0, 0.1]]], dtype=np.float32)
    results = det._postprocess(fake_out, scale, pad_x, pad_y, orig_w, orig_h)

    assert len(results) == 1
    assert abs(results[0].center_px[0] - 512.0) < 1e-3
    assert abs(results[0].center_px[1] - 288.0) < 1e-3


def test_detector_fail_fast_on_missing_model():
    with pytest.raises(RuntimeError):
        YoloCubeDetector(weights_path='/non/existent/model.engine')


def test_postprocess_yolo_obb_format():
    detector = YoloCubeDetector.__new__(YoloCubeDetector)
    detector.conf_threshold = 0.5

    out = np.zeros((1, 300, 7), dtype=np.float32)

    # Box 1: White cube at (320, 320), size (50, 50), conf 0.85, class 1
    out[0, 100] = [320.0, 320.0, 50.0, 50.0, 0.85, 1.0, 0.0]

    # Box 3: Below confidence threshold (filtered out)
    out[0, 200] = [100.0, 100.0, 40.0, 40.0, 0.30, 0.0, 0.0]

    # Box 4: Green cube elsewhere, conf 0.90, class 0
    out[0, 250] = [200.0, 150.0, 60.0, 60.0, 0.90, 0.0, 0.25]

    detections = detector._postprocess(out, 1.0, 0.0, 0.0, 640, 640)
    assert len(detections) == 2

    d0 = detections[0]
    assert d0.class_id == 0
    assert d0.class_name == 'green_cube'
    assert math.isclose(d0.confidence, 0.90, rel_tol=1e-5)
    assert math.isclose(d0.center_px[0], 200.0, rel_tol=1e-5)
    assert math.isclose(d0.center_px[1], 150.0, rel_tol=1e-5)
    assert math.isclose(d0.angle, 0.25, rel_tol=1e-5)
    assert len(d0.corners) == 4

    d1 = detections[1]
    assert d1.class_id == 1
    assert d1.class_name == 'white_cube'
    assert math.isclose(d1.confidence, 0.85, rel_tol=1e-5)
    assert math.isclose(d1.center_px[0], 320.0, rel_tol=1e-5)
    assert math.isclose(d1.center_px[1], 320.0, rel_tol=1e-5)


def test_postprocess_empty_and_none():
    detector = YoloCubeDetector.__new__(YoloCubeDetector)
    detector.conf_threshold = 0.5

    assert detector._postprocess(None, 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess([], 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess(np.zeros((1, 300, 7)), 1.0, 0, 0, 640, 640) == []


def test_normalize_input_blob_properties():
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    blob, scale, pad_x, pad_y = YoloCubeDetector._normalize_input(img)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() <= blob.max() <= 1.0
    assert blob.flags['C_CONTIGUOUS']
