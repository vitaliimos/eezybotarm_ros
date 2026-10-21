"""Object detector for white and green cubes using TensorRT or ONNX."""

import ctypes
from dataclasses import dataclass
import os

import cv2
import numpy as np


@dataclass
class CubeDetection:
    """Represents a detected cube."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    angle: float
    corners: tuple[tuple[float, float], ...]


class TrtRunner:
    """TensorRT inference runner for NVIDIA Jetson."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine from {engine_path}')

        self.context = self.engine.create_execution_context()
        self.cudart = self._load_cudart()

        self.input_name = 'images'
        self.output_name = 'output0'
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        in_bytes = int(np.prod(self.input_shape) * 4)
        out_bytes = int(np.prod(self.output_shape) * 4)

        self.d_input = self._cuda_malloc(in_bytes)
        self.d_output = self._cuda_malloc(out_bytes)
        self.stream = self._cuda_stream_create()

    def _load_cudart(self):
        import glob

        for lib in ['libcudart.so', 'libcudart.so.13', 'libcudart.so.12']:
            try:
                return ctypes.CDLL(lib)
            except OSError:
                pass

        for path in glob.glob(
            '/usr/local/lib/python3*/dist-packages/nvidia/*/lib/libcudart.so*'
        ):
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass

        raise RuntimeError('libcudart.so not found on system.')

    def _cuda_malloc(self, size_bytes: int):
        ptr = ctypes.c_void_p()
        ret = self.cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
        if ret != 0:
            raise RuntimeError(f'cudaMalloc failed with code {ret}')
        return ptr

    def _cuda_stream_create(self):
        stream = ctypes.c_void_p()
        ret = self.cudart.cudaStreamCreate(ctypes.byref(stream))
        if ret != 0:
            raise RuntimeError(f'cudaStreamCreate failed with code {ret}')
        return stream

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        self.cudart.cudaMemcpyAsync(
            self.d_input,
            input_blob.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(input_blob.nbytes),
            ctypes.c_int(1),
            self.stream,
        )

        self.context.set_input_shape(self.input_name, input_blob.shape)
        self.context.set_tensor_address(self.input_name, self.d_input.value)
        self.context.set_tensor_address(self.output_name, self.d_output.value)
        self.context.execute_async_v3(self.stream.value)

        out_buf = np.empty(self.output_shape, dtype=np.float32)
        self.cudart.cudaMemcpyAsync(
            out_buf.ctypes.data_as(ctypes.c_void_p),
            self.d_output,
            ctypes.c_size_t(out_buf.nbytes),
            ctypes.c_int(2),
            self.stream,
        )
        self.cudart.cudaStreamSynchronize(self.stream)
        return out_buf

    def __del__(self):
        if hasattr(self, 'cudart'):
            if getattr(self, 'd_input', None):
                self.cudart.cudaFree(self.d_input)
            if getattr(self, 'd_output', None):
                self.cudart.cudaFree(self.d_output)
            if getattr(self, 'stream', None):
                self.cudart.cudaStreamDestroy(self.stream)


class OnnxRunner:
    """ONNX Runtime inference runner for workstation development and testing."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: input_blob})[0]


class YoloCubeDetector:
    """Detector for green_cube and white_cube using TensorRT or ONNX."""

    CLASS_NAMES = {0: 'green_cube', 1: 'white_cube'}

    def __init__(
        self,
        weights_path: str = '',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._runner = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.weights_path:
            raise RuntimeError('YOLO weights_path parameter is empty.')

        path = self.weights_path
        # Allow automatic fallback to counterpart .onnx if .engine does not exist in same location
        if not os.path.exists(path) and path.endswith('.engine'):
            onnx_path = path[:-7] + '.onnx'
            if os.path.exists(onnx_path):
                path = onnx_path

        if not os.path.exists(path):
            raise RuntimeError(f"YOLO model file not found: '{self.weights_path}'")

        if path.endswith('.engine'):
            try:
                self._runner = TrtRunner(path)
                return
            except Exception:
                onnx_path = path[:-7] + '.onnx'
                if os.path.exists(onnx_path):
                    self._runner = OnnxRunner(onnx_path)
                    return
                raise

        self._runner = OnnxRunner(path)

    def is_loaded(self) -> bool:
        return self._runner is not None

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, float, float]:
        """Resize and pad image while preserving aspect ratio (Ultralytics LetterBox)."""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = new_shape
        scale = min(target_h / orig_h, target_w / orig_w)

        new_unpad_w = round(orig_w * scale)
        new_unpad_h = round(orig_h * scale)

        dw = (target_w - new_unpad_w) / 2.0
        dh = (target_h - new_unpad_h) / 2.0

        top = round(dh - 0.1)
        bottom = int(target_h - new_unpad_h - top)
        left = round(dw - 0.1)
        right = int(target_w - new_unpad_w - left)

        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(
                image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized = image

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, scale, float(left), float(top)

    @classmethod
    def _normalize_input(cls, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        padded, scale, pad_x, pad_y = cls._letterbox(image, (640, 640))
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))
        return blob, scale, pad_x, pad_y

    def predict(self, image: np.ndarray) -> list[CubeDetection]:
        """Detect cubes in image and return list of CubeDetection."""
        if not self.is_loaded():
            raise RuntimeError('Cannot predict: model is not loaded.')

        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        blob, scale, pad_x, pad_y = self._normalize_input(image)

        out = self._runner.infer(blob)
        return self._postprocess(out, scale, pad_x, pad_y, orig_w, orig_h)

    def _postprocess(
        self,
        out: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        orig_w: int,
        orig_h: int,
    ) -> list[CubeDetection]:
        if out is None or len(out) == 0:
            return []

        # yolo26-obb output format: [1, 7, 8400] -> transpose to [8400, 7]
        preds = out[0].T

        # Columns: cx, cy, w, h, score_green, score_white, angle
        boxes = preds[:, :4]
        cls_scores = preds[:, 4:6]
        angles = preds[:, 6]

        confidences = np.max(cls_scores, axis=1)
        cls_ids = np.argmax(cls_scores, axis=1)

        # Confidence threshold filtering
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        cls_ids = cls_ids[mask]
        angles = angles[mask]

        # Rescale coordinates to original image space
        cx = (boxes[:, 0] - pad_x) / scale
        cy = (boxes[:, 1] - pad_y) / scale
        w = boxes[:, 2] / scale
        h = boxes[:, 3] / scale

        # Calculate oriented box corners
        cos_val = np.cos(angles)
        sin_val = np.sin(angles)
        vec1_x, vec1_y = (w / 2.0) * cos_val, (w / 2.0) * sin_val
        vec2_x, vec2_y = -(h / 2.0) * sin_val, (h / 2.0) * cos_val

        p1_x, p1_y = cx + vec1_x + vec2_x, cy + vec1_y + vec2_y
        p2_x, p2_y = cx + vec1_x - vec2_x, cy + vec1_y - vec2_y
        p3_x, p3_y = cx - vec1_x - vec2_x, cy - vec1_y - vec2_y
        p4_x, p4_y = cx - vec1_x + vec2_x, cy - vec1_y + vec2_y

        corners_x = np.stack([p1_x, p2_x, p3_x, p4_x], axis=0)
        corners_y = np.stack([p1_y, p2_y, p3_y, p4_y], axis=0)

        min_x = np.clip(corners_x.min(axis=0), 0.0, float(orig_w))
        min_y = np.clip(corners_y.min(axis=0), 0.0, float(orig_h))
        max_x = np.clip(corners_x.max(axis=0), 0.0, float(orig_w))
        max_y = np.clip(corners_y.max(axis=0), 0.0, float(orig_h))

        # Class-wise Non-Maximum Suppression (NMS)
        nms_boxes = np.stack([min_x, min_y, max_x - min_x, max_y - min_y], axis=1)

        keep_indices = []
        for c in np.unique(cls_ids):
            c_mask = np.where(cls_ids == c)[0]
            c_keep = cv2.dnn.NMSBoxes(
                nms_boxes[c_mask],
                confidences[c_mask],
                score_threshold=float(self.conf_threshold),
                nms_threshold=float(self.iou_threshold),
            )
            if len(c_keep) > 0:
                keep_indices.extend(c_mask[np.array(c_keep).reshape(-1)])

        detections: list[CubeDetection] = []
        for idx in keep_indices:
            cls_id = int(cls_ids[idx])
            detections.append(
                CubeDetection(
                    class_id=cls_id,
                    class_name=self.CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                    confidence=float(confidences[idx]),
                    bbox_xyxy=(
                        float(min_x[idx]),
                        float(min_y[idx]),
                        float(max_x[idx]),
                        float(max_y[idx]),
                    ),
                    center_px=(float(cx[idx]), float(cy[idx])),
                    angle=float(angles[idx]),
                    corners=(
                        (float(p1_x[idx]), float(p1_y[idx])),
                        (float(p2_x[idx]), float(p2_y[idx])),
                        (float(p3_x[idx]), float(p3_y[idx])),
                        (float(p4_x[idx]), float(p4_y[idx])),
                    ),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
