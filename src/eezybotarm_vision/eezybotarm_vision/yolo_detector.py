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
        iou_threshold: float = 0.45,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))
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

        # yolo26-obb output format: [1, 7, 8400] -> transpose to [8400, 7]
        # Columns: cx, cy, w, h, score_green, score_white, angle
        preds = out[0].T
        boxes = preds[:, :4]
        cls_scores = preds[:, 4:6]
        angles = preds[:, 6]

        confidences = np.max(cls_scores, axis=1)
        cls_ids = np.argmax(cls_scores, axis=1)

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
