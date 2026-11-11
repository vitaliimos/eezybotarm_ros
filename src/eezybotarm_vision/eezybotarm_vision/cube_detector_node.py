"""Cube Detector Node for EezyBotArm YOLO-OBB inference and workspace projection."""

from cv_bridge import CvBridge
from eezybotarm_interfaces.msg import LocalizedCubeArray, WorkspaceCalibration
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

from .localized_cube import LocalizedCube
from .yolo_detector import YoloCubeDetector
from .zone_detector import get_real_coordinates, identify_zone


class CubeDetectorNode(Node):
    """Run YOLO-OBB cube detection and map detections to table coordinates."""

    def __init__(self, **kwargs):
        super().__init__('cube_detector_node', **kwargs)

        self.bridge = CvBridge()

        self.declare_parameter('weights_path', Parameter.Type.STRING)
        self.declare_parameter('conf_threshold', Parameter.Type.DOUBLE)

        weights_path = self.get_parameter('weights_path').value
        conf_thresh = self.get_parameter('conf_threshold').value

        self.get_logger().info(f'Loading Cube Detector with: "{weights_path}"...')
        try:
            self.detector = YoloCubeDetector(
                weights_path=weights_path,
                conf_threshold=conf_thresh,
            )
            self.get_logger().info('Cube detector initialized successfully.')
        except Exception as exc:
            self.get_logger().fatal(
                f'Failed to initialize Cube detector from "{weights_path}": {exc}. Stopping node.'
            )
            raise

        self.calibration_msg: WorkspaceCalibration | None = None
        self.homography_matrix: np.ndarray | None = None
        self._is_processing = False

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.calibration_sub = self.create_subscription(
            WorkspaceCalibration,
            '/eezybotarm_vision/calibration',
            self._calibration_callback,
            latched_qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1,
        )
        self.detections_pub = self.create_publisher(
            LocalizedCubeArray,
            '/eezybotarm_vision/detections',
            1,
        )

        self.get_logger().info('Cube Detector Node ready and monitoring /camera/image_raw')

    def _calibration_callback(self, msg: WorkspaceCalibration) -> None:
        self.calibration_msg = msg
        self.homography_matrix = np.array(msg.homography_matrix, dtype=np.float64).reshape(3, 3)

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

    def _detect_and_map_cubes(self, frame: np.ndarray) -> list[LocalizedCube]:
        raw_detections = self.detector.predict(frame)
        cubes: list[LocalizedCube] = []

        for det in raw_detections:
            if self.calibration_msg is not None and self.homography_matrix is not None:
                x_mm, y_mm = get_real_coordinates(
                    det.center_px[0],
                    det.center_px[1],
                    self.homography_matrix,
                )
                zone = identify_zone(
                    x_mm,
                    y_mm,
                    float(self.calibration_msg.field_width_mm),
                    float(self.calibration_msg.field_height_mm),
                )
            else:
                x_mm, y_mm = 0.0, 0.0
                zone = 'uncalibrated'

            cubes.append(LocalizedCube(detection=det, zone=zone, x_mm=x_mm, y_mm=y_mm))

        return cubes

    def _handle_image(self, msg: Image) -> None:
        frame = self._convert_image(msg)
        if frame is None:
            return

        cubes = self._detect_and_map_cubes(frame)
        out_msg = LocalizedCubeArray()
        out_msg.header = msg.header
        out_msg.cubes = [cube.to_msg() for cube in cubes]
        self.detections_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CubeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
