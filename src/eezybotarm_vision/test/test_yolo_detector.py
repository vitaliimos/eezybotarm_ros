"""Unit tests for YOLO cube detector and inference runners."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from eezybotarm_vision.inference_runners import OnnxRunner
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
    det.iou_threshold = 0.45
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
    detector.iou_threshold = 0.45

    out = np.zeros((1, 300, 7), dtype=np.float32)

    # Box 1: White cube at (320, 320), size (50, 50), conf 0.85, class 1
    out[0, 100] = [320.0, 320.0, 50.0, 50.0, 0.85, 1.0, 0.0]

    # Box 2: Duplicate of Box 1 with lower confidence (suppressed by NMS)
    out[0, 101] = [322.0, 321.0, 50.0, 50.0, 0.65, 1.0, 0.0]

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
    detector.iou_threshold = 0.45

    assert detector._postprocess(None, 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess([], 1.0, 0, 0, 640, 640) == []
    assert detector._postprocess(np.zeros((1, 300, 7)), 1.0, 0, 0, 640, 640) == []


def test_onnx_runner_basic_opt_and_cpu_fallback(monkeypatch):
    expected_out = np.ones((1, 7, 8400), dtype=np.float32)

    cuda_session = MagicMock()
    cuda_session.get_providers.return_value = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
    cuda_session.get_inputs.return_value = [SimpleNamespace(name='images')]
    cuda_session.run.side_effect = RuntimeError('cudaErrorNoKernelImageForDevice')

    cpu_session = MagicMock()
    cpu_session.get_providers.return_value = ['CPUExecutionProvider']
    cpu_session.get_inputs.return_value = [SimpleNamespace(name='images')]
    cpu_session.run.return_value = [expected_out]

    sessions = [cuda_session, cpu_session]
    created_calls = []

    def fake_inference_session(path, sess_options=None, providers=None):
        created_calls.append((path, sess_options, providers))
        return sessions.pop(0)

    fake_ort = SimpleNamespace(
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC='ORT_ENABLE_BASIC'),
        SessionOptions=lambda: SimpleNamespace(graph_optimization_level=None),
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        InferenceSession=fake_inference_session,
    )
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake_ort)

    runner = OnnxRunner('dummy.onnx')
    assert len(created_calls) == 1
    assert created_calls[0][1].graph_optimization_level == 'ORT_ENABLE_BASIC'
    assert created_calls[0][2] == ['CUDAExecutionProvider', 'CPUExecutionProvider']

    blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
    result = runner.infer(blob)

    assert np.array_equal(result, expected_out)
    assert len(created_calls) == 2
    assert created_calls[1][2] == ['CPUExecutionProvider']
