"""Debug Visualizer Node for EezyBotArm annotated camera stream."""

from cv_bridge import CvBridge
from eezybotarm_interfaces.msg import LocalizedCubeArray, WorkspaceCalibration
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .debug_visualizer import DebugVisualizer, LocalizedCube
from .zone_detector import ZoneDetectionResult


class DebugVisualizerNode(Node):
    """Render workspace zones, cube detections, and sorting status onto debug image topics."""

    def __init__(self, **kwargs):
        super().__init__('debug_visualizer_node', **kwargs)

        self.bridge = CvBridge()

        self.declare_parameter('debug_image_jpeg_quality', Parameter.Type.INTEGER)
        jpeg_quality = self.get_parameter('debug_image_jpeg_quality').value

        self.visualizer = DebugVisualizer(node=self, jpeg_quality=jpeg_quality)

        self.calibration: ZoneDetectionResult | None = None
        self.detections: list[LocalizedCube] = []
        self.sorting_status: str = 'SCANNING'
        self._is_processing = False

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.calibration_sub = self.create_subscription(
            WorkspaceCalibration,
            '/eezybotarm_vision/calibration',
            self._calibration_callback,
            latched_qos,
        )
        self.detections_sub = self.create_subscription(
            LocalizedCubeArray,
            '/eezybotarm_vision/detections',
            self._detections_callback,
            1,
        )
        self.status_sub = self.create_subscription(
            String,
            '/eezybotarm_vision/sorting_status',
            self._status_callback,
            latched_qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1,
        )

        self.get_logger().info('Debug Visualizer Node ready and monitoring /camera/image_raw')

    def _calibration_callback(self, msg: WorkspaceCalibration) -> None:
        self.calibration = ZoneDetectionResult.from_msg(msg)

    def _detections_callback(self, msg: LocalizedCubeArray) -> None:
        self.detections = [LocalizedCube.from_msg(c) for c in msg.cubes]

    def _status_callback(self, msg: String) -> None:
        self.sorting_status = msg.data

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

    def _handle_image(self, msg: Image) -> None:
        if not self.visualizer.has_subscribers():
            return

        frame = self._convert_image(msg)
        if frame is None:
            return

        state_text = 'CALIBRATING' if self.calibration is None else self.sorting_status
        self.visualizer.publish(
            frame=frame,
            header=msg.header,
            state_text=state_text,
            calibration=self.calibration,
            detections=self.detections,
        )


def main(args=None):
    rclpy.init(args=args)
    node = DebugVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
