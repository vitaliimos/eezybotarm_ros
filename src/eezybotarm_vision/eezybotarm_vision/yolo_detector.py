"""YOLO-OBB cube detector for white and green cubes."""

from dataclasses import dataclass

import cv2
import numpy as np

from .inference_runners import create_inference_runner


@dataclass
class CubeDetection:
    """Detected cube with oriented bounding box."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    angle: float
    corners: tuple[tuple[float, float], ...]


class YoloCubeDetector:
    """Detector for green_cube and white_cube using TensorRT or ONNX."""

    CLASS_NAMES = {0: 'green_cube', 1: 'white_cube'}
    INPUT_SHAPE = (640, 640)

    def __init__(
        self,
        weights_path: str = '',
        conf_threshold: float = 0.5,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self._runner = create_inference_runner(weights_path)

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: tuple[int, int] = INPUT_SHAPE,
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
        padded, scale, pad_x, pad_y = cls._letterbox(image, cls.INPUT_SHAPE)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=cls.INPUT_SHAPE,
            swapRB=True,
            crop=False,
        )
        return blob, scale, pad_x, pad_y

    def predict(self, image: np.ndarray) -> list[CubeDetection]:
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

        # YOLO26-OBB (ONNX / TensorRT) format: [1, N, 7]
        # Columns: cx, cy, w, h, score, class_id, angle
        preds = out[0]
        boxes = preds[:, :4]
        confidences = preds[:, 4]
        cls_ids = preds[:, 5].astype(int)
        angles = preds[:, 6]

        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        cls_ids = cls_ids[mask]
        angles = angles[mask]

        cx = (boxes[:, 0] - pad_x) / scale
        cy = (boxes[:, 1] - pad_y) / scale
        w = boxes[:, 2] / scale
        h = boxes[:, 3] / scale

        cos_val = np.cos(angles)
        sin_val = np.sin(angles)
        v1 = np.stack([(w / 2.0) * cos_val, (w / 2.0) * sin_val], axis=1)
        v2 = np.stack([-(h / 2.0) * sin_val, (h / 2.0) * cos_val], axis=1)
        center = np.stack([cx, cy], axis=1)
        corners = np.stack(
            [center + v1 + v2, center + v1 - v2, center - v1 - v2, center - v1 + v2],
            axis=1,
        )  # (N, 4, 2)

        mins = np.clip(corners.min(axis=1), 0.0, [orig_w, orig_h])
        maxs = np.clip(corners.max(axis=1), 0.0, [orig_w, orig_h])

        detections = [
            CubeDetection(
                class_id=int(cls_ids[i]),
                class_name=self.CLASS_NAMES.get(int(cls_ids[i]), f'class_{int(cls_ids[i])}'),
                confidence=float(confidences[i]),
                bbox_xyxy=(
                    float(mins[i, 0]), float(mins[i, 1]), float(maxs[i, 0]), float(maxs[i, 1])
                ),
                center_px=(float(cx[i]), float(cy[i])),
                angle=float(angles[i]),
                corners=tuple((float(x), float(y)) for x, y in corners[i]),
            )
            for i in range(len(confidences))
        ]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
