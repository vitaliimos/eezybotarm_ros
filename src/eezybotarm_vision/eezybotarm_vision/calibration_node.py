"""Workspace Calibration Node for EezyBotArm fiducial zone detection and homography."""

import time

from cv_bridge import CvBridge
from eezybotarm_interfaces.msg import WorkspaceCalibration
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

from .zone_detector import detect_zones, smooth_calibration, ZoneDetectionResult


class WorkspaceCalibrationNode(Node):
    """Detect corner ring markers and publish smoothed workspace homography."""

    def __init__(self, **kwargs):
        super().__init__('calibration_node', **kwargs)

        self.bridge = CvBridge()

        self.declare_parameter('recalibration_interval_sec', Parameter.Type.DOUBLE)
        self.declare_parameter('calibration_smoothing_alpha', Parameter.Type.DOUBLE)
        self.declare_parameter('table_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_depth_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_height_m', Parameter.Type.DOUBLE)

        self.recalibration_interval_sec = self.get_parameter('recalibration_interval_sec').value
        self.calibration_smoothing_alpha = self.get_parameter('calibration_smoothing_alpha').value
        table_width_m = self.get_parameter('table_width_m').value
        table_depth_m = self.get_parameter('table_depth_m').value
        sticker_width_m = self.get_parameter('sticker_width_m').value
        sticker_height_m = self.get_parameter('sticker_height_m').value

        self.table_width_mm = table_width_m * 1000.0
        self.table_depth_mm = table_depth_m * 1000.0
        self.marker_inset_x_mm = (sticker_width_m * 1000.0) / 2.0
        self.marker_inset_y_mm = (sticker_height_m * 1000.0) / 2.0

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.calibration_pub = self.create_publisher(
            WorkspaceCalibration,
            '/eezybotarm_vision/calibration',
            latched_qos,
        )
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1,
        )

        self.calibration: ZoneDetectionResult | None = None
        self.last_recalibration_time = 0.0
        self._is_processing = False

        self.get_logger().info('Workspace Calibration Node ready and monitoring /camera/image_raw')

    def image_callback(self, msg: Image) -> None:
        if self._is_processing:
            return
        self._is_processing = True
        try:
            self._handle_image(msg)
        finally:
            self._is_processing = False

    def _convert_image(self, msg: Image) -> np.ndarray | None:
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert image frame: {exc}', throttle_duration_sec=2.0
            )
            return None

    def _detect_workspace_zones(self, frame: np.ndarray) -> ZoneDetectionResult | None:
        return detect_zones(
            frame,
            field_width_mm=self.table_width_mm,
            field_height_mm=self.table_depth_mm,
            marker_inset_x_mm=self.marker_inset_x_mm,
            marker_inset_y_mm=self.marker_inset_y_mm,
        )

    def _handle_image(self, msg: Image) -> None:
        now = time.monotonic()
        if (
            self.calibration is not None
            and now - self.last_recalibration_time < self.recalibration_interval_sec
        ):
            return

        frame = self._convert_image(msg)
        if frame is None:
            return

        if self.calibration is None:
            detected_zones = self._detect_workspace_zones(frame)
            if detected_zones is not None:
                self.calibration = detected_zones
                self.last_recalibration_time = now
                self.calibration_pub.publish(self.calibration.to_msg(msg.header))
                self.get_logger().info('Workspace zones detected and locked.')
            return

        self.last_recalibration_time = now
        detected_zones = self._detect_workspace_zones(frame)
        if detected_zones is not None:
            self.calibration = smooth_calibration(
                self.calibration,
                detected_zones,
                alpha=self.calibration_smoothing_alpha,
            )
            self.calibration_pub.publish(self.calibration.to_msg(msg.header))
            self.get_logger().debug('Workspace calibration smoothed and updated.')


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
