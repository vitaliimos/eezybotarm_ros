"""Localized cube data type shared across vision nodes."""

from dataclasses import dataclass

from .yolo_detector import CubeDetection


@dataclass
class LocalizedCube:
    """Detected cube mapped to workspace coordinates and zone."""

    detection: CubeDetection
    zone: str
    x_mm: float
    y_mm: float

    def to_msg(self):
        """Serialize localized cube into LocalizedCube ROS 2 message."""
        from eezybotarm_interfaces.msg import LocalizedCube as LocalizedCubeMsg

        msg = LocalizedCubeMsg()
        det = self.detection
        msg.class_id = int(det.class_id)
        msg.class_name = str(det.class_name)
        msg.confidence = float(det.confidence)
        msg.bbox_xyxy = [float(v) for v in det.bbox_xyxy]
        msg.center_px = [float(det.center_px[0]), float(det.center_px[1])]
        msg.angle = float(det.angle)
        msg.corners_px = [float(coord) for pt in det.corners for coord in pt]
        msg.zone = str(self.zone)
        msg.x_mm = float(self.x_mm)
        msg.y_mm = float(self.y_mm)
        return msg

    @classmethod
    def from_msg(cls, msg) -> 'LocalizedCube':
        """Deserialize LocalizedCube ROS 2 message into LocalizedCube dataclass."""
        corners = tuple(
            (float(msg.corners_px[i]), float(msg.corners_px[i + 1]))
            for i in range(0, 8, 2)
        )
        det = CubeDetection(
            class_id=int(msg.class_id),
            class_name=str(msg.class_name),
            confidence=float(msg.confidence),
            bbox_xyxy=(
                float(msg.bbox_xyxy[0]),
                float(msg.bbox_xyxy[1]),
                float(msg.bbox_xyxy[2]),
                float(msg.bbox_xyxy[3]),
            ),
            center_px=(float(msg.center_px[0]), float(msg.center_px[1])),
            angle=float(msg.angle),
            corners=corners,
        )
        return cls(
            detection=det,
            zone=str(msg.zone),
            x_mm=float(msg.x_mm),
            y_mm=float(msg.y_mm),
        )
