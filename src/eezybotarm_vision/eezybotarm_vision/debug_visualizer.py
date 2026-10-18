"""Debug visualizer and publisher for EezyBotArm computer vision."""

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from .yolo_detector import CubeDetection
from .zone_detector import draw_zones, ZoneDetectionResult


class DebugVisualizer:
    """Renders debug overlays and publishes raw and compressed ROS 2 image streams."""

    def __init__(
        self,
        node: Node,
        raw_topic: str = '/eezybotarm_vision/debug_image',
        jpeg_quality: int = 80,
    ) -> None:
        """Initialize publishers and CV bridge."""
        self.node = node
        self.jpeg_quality = int(jpeg_quality)
        self.bridge = CvBridge()

        self.debug_image_pub = node.create_publisher(Image, raw_topic, 1)
        self.debug_compressed_pub = node.create_publisher(
            CompressedImage, f'{raw_topic}/compressed', 1
        )

    @property
    def has_subscribers(self) -> bool:
        """Check whether any client is subscribed to raw or compressed topics."""
        return (
            self.debug_image_pub.get_subscription_count() > 0
            or self.debug_compressed_pub.get_subscription_count() > 0
        )

    def publish(
        self,
        frame: np.ndarray,
        header: Header,
        state_text: str,
        calibration: ZoneDetectionResult | None = None,
        detections: list[tuple[CubeDetection, str, float, float]] | None = None,
        status_extra: str = '',
    ) -> None:
        """Render annotations and publish to active debug topics."""
        has_raw = self.debug_image_pub.get_subscription_count() > 0
        has_comp = self.debug_compressed_pub.get_subscription_count() > 0

        if not (has_raw or has_comp):
            return

        annotated = self._render_debug_overlay(
            frame,
            state_text=state_text,
            calibration=calibration,
            detections=detections or [],
            status_extra=status_extra,
        )

        if has_raw:
            debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_msg.header = header
            self.debug_image_pub.publish(debug_msg)

        if has_comp:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encoded = cv2.imencode('.jpg', annotated, encode_param)
            if success:
                comp_msg = CompressedImage()
                comp_msg.header = header
                comp_msg.format = 'jpeg'
                comp_msg.data = encoded.tobytes()
                self.debug_compressed_pub.publish(comp_msg)

    def _render_debug_overlay(
        self,
        frame: np.ndarray,
        state_text: str,
        calibration: ZoneDetectionResult | None,
        detections: list[tuple[CubeDetection, str, float, float]],
        status_extra: str,
    ) -> np.ndarray:
        """Draw calibration zones, bounding boxes, and status text."""
        annotated = frame.copy()

        if state_text == 'CALIBRATING':
            cv2.putText(
                annotated,
                'Calibrating: Looking for 3 Zones...',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 0, 255),
                thickness=2,
            )
        elif calibration is not None:
            annotated = draw_zones(annotated, calibration)

        for det, zone, x_mm, y_mm in detections:
            self._draw_cube(annotated, det, zone, x_mm, y_mm)

        label = f'State: {state_text}{status_extra}'
        cv2.putText(
            annotated,
            label,
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(0, 255, 255),
            thickness=2,
        )
        return annotated

    def _draw_cube(
        self,
        img: np.ndarray,
        det: CubeDetection,
        zone: str,
        x_mm: float,
        y_mm: float,
    ) -> None:
        """Draw bounding box and center marker for a detected cube."""
        color = (0, 255, 0) if det.class_id == 0 else (255, 255, 255)
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, y2 = int(det.bbox_xyxy[0]), int(det.bbox_xyxy[1]), int(det.bbox_xyxy[3])
        cv2.drawMarker(
            img,
            (int(det.center_px[0]), int(det.center_px[1])),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )
        label = f'{det.class_name} ({det.confidence:.2f}) [{zone}]'
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
        )
        if zone != 'unknown':
            coords_lbl = f'({x_mm:.0f}, {y_mm:.0f})mm'
            cv2.putText(
                img,
                coords_lbl,
                (x1, min(img.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45,
                color=(0, 255, 255),
                thickness=1,
            )
