"""Sorting Coordinator Node for EezyBotArm computer vision and pick-and-place."""

from enum import Enum
import os
import time

from cv_bridge import CvBridge
from eezybotarm_interfaces.action import PickObject
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image

from .coordinate_mapper import CoordinateMapper
from .debug_visualizer import DebugVisualizer
from .yolo_detector import CubeDetection, YoloCubeDetector
from .zone_detector import (
    detect_zones,
    get_real_coordinates,
    identify_zone,
    ZoneDetectionResult,
)


class SortingState(Enum):
    CALIBRATING = 'CALIBRATING'
    SCANNING = 'SCANNING'
    CONFIRMING = 'CONFIRMING'
    EXECUTING = 'EXECUTING'
    COOLDOWN = 'COOLDOWN'


class SortingCoordinatorNode(Node):
    """Orchestrate workspace calibration, cube detection, and pick-and-place actions."""

    def __init__(self):
        super().__init__('sorting_coordinator_node')

        self.bridge = CvBridge()

        default_model_path = '/usr/share/models/model.engine'

        self.declare_parameter('weights_path', default_model_path)
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('stable_frames_required', 8)
        self.declare_parameter('cooldown_duration_sec', 2.5)
        self.declare_parameter('paper_width_mm', 300.0)
        self.declare_parameter('paper_height_mm', 200.0)
        self.declare_parameter('robot_x_offset_m', 0.150)
        self.declare_parameter('robot_y_offset_m', 0.0)
        self.declare_parameter('table_z_m', 0.095)
        self.declare_parameter('cube_height_m', 0.025)
        self.declare_parameter('debug_image_jpeg_quality', 80)
        self.declare_parameter('action_name', 'pick_and_place')

        weights_path = self.get_parameter('weights_path').value or default_model_path
        if not os.path.exists(weights_path) and os.path.exists(default_model_path):
            weights_path = default_model_path
        conf_thresh = self.get_parameter('conf_threshold').value
        self.stable_frames_required = self.get_parameter('stable_frames_required').value
        self.cooldown_duration_sec = self.get_parameter('cooldown_duration_sec').value
        jpeg_quality = self.get_parameter('debug_image_jpeg_quality').value

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

        self.coord_mapper = CoordinateMapper(
            paper_width_mm=self.get_parameter('paper_width_mm').value,
            paper_height_mm=self.get_parameter('paper_height_mm').value,
            robot_x_offset_m=self.get_parameter('robot_x_offset_m').value,
            robot_y_offset_m=self.get_parameter('robot_y_offset_m').value,
            table_z_m=self.get_parameter('table_z_m').value,
            cube_height_m=self.get_parameter('cube_height_m').value,
        )

        action_name = self.get_parameter('action_name').value
        self.action_client = ActionClient(self, PickObject, action_name)

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
        )
        self.visualizer = DebugVisualizer(node=self, jpeg_quality=jpeg_quality)

        # State management
        self._state = SortingState.CALIBRATING
        self.calibration: ZoneDetectionResult | None = None
        self.candidate_cube: CubeDetection | None = None
        self.candidate_frames_count = 0
        self.cooldown_start_time = 0.0
        self.current_action_phase = ''
        self._is_processing = False

        self.get_logger().info(f'Initial state: {self._state.value}')
        self.get_logger().info('Sorting Coordinator Node ready and monitoring /camera/image_raw')

    @property
    def state(self) -> SortingState:
        """Get the current sorting state."""
        return self._state

    @state.setter
    def state(self, new_state: SortingState) -> None:
        """Set a new sorting state and log on transition."""
        if self._state != new_state:
            self.get_logger().info(f'State changed: {self._state.value} -> {new_state.value}')
            self._state = new_state

    @property
    def active_calibration(self) -> ZoneDetectionResult | None:
        """Get current calibration for backward compatibility."""
        return self.calibration

    @property
    def debug_image_pub(self):
        """Get the debug image publisher."""
        return self.visualizer.debug_image_pub

    @property
    def debug_compressed_pub(self):
        """Get the debug compressed image publisher."""
        return self.visualizer.debug_compressed_pub

    def image_callback(self, msg: Image) -> None:
        if self._is_processing:
            return
        self._is_processing = True
        try:
            self._handle_image(msg)
        finally:
            self._is_processing = False

    def _convert_image(self, msg: Image) -> np.ndarray | None:
        """Convert ROS Image message to OpenCV BGR image."""
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert image frame: {exc}', throttle_duration_sec=2.0
            )
            return None

    def _handle_calibrating(self, frame: np.ndarray) -> None:
        """Handle workspace calibration by attempting zone detection."""
        detected_zones = detect_zones(
            frame,
            field_width_mm=self.coord_mapper.paper_width_mm,
            field_height_mm=self.coord_mapper.paper_height_mm,
        )
        if detected_zones is not None:
            self.calibration = detected_zones
            self.get_logger().info('Workspace zones detected and locked.')
            self.state = SortingState.SCANNING

    def _handle_cooldown(self) -> None:
        """Handle cooldown timer after completing a pick-and-place action."""
        now = time.monotonic()
        if now - self.cooldown_start_time >= self.cooldown_duration_sec:
            self.state = SortingState.SCANNING
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.get_logger().info('Cooldown finished, resumed scanning.')

    def _handle_executing(self) -> None:
        """Handle frames during active action execution (no tracking)."""
        pass

    def _detect_and_map_cubes(
        self, frame: np.ndarray
    ) -> tuple[
        list[tuple[CubeDetection, str, float, float]],
        list[tuple[CubeDetection, str, float, float]],
    ]:
        """Detect cubes using YOLO and compute real workspace coordinates and zones."""
        assert self.calibration is not None
        raw_detections = self.detector.predict(frame)

        detections_with_zones: list[tuple[CubeDetection, str, float, float]] = []
        middle_detections: list[tuple[CubeDetection, str, float, float]] = []

        for det in raw_detections:
            x_mm, y_mm = get_real_coordinates(
                det.center_px[0],
                det.center_px[1],
                self.calibration.homography_matrix,
            )
            zone = identify_zone(
                x_mm,
                y_mm,
                self.coord_mapper.paper_width_mm,
                self.coord_mapper.paper_height_mm,
            )

            item = (det, zone, x_mm, y_mm)
            detections_with_zones.append(item)

            if zone == 'middle':
                middle_detections.append(item)

        return detections_with_zones, middle_detections

    def _is_cube_match(
        self, a: CubeDetection, b: CubeDetection, max_dist_px: float = 25.0
    ) -> bool:
        """Check if two cube detections represent the same object."""
        return (
            a.class_id == b.class_id
            and abs(a.center_px[0] - b.center_px[0]) < max_dist_px
            and abs(a.center_px[1] - b.center_px[1]) < max_dist_px
        )

    def _update_candidate_tracking(
        self, middle_detections: list[tuple[CubeDetection, str, float, float]]
    ) -> None:
        """Update candidate stability tracking and trigger action when stable."""
        middle_candidate = None
        if self.candidate_cube is not None:
            for item in middle_detections:
                if self._is_cube_match(self.candidate_cube, item[0]):
                    middle_candidate = item
                    break

        if middle_candidate is None and middle_detections:
            middle_candidate = middle_detections[0]

        if middle_candidate is not None:
            det, _, x_mm, y_mm = middle_candidate
            if self.candidate_cube is not None and self._is_cube_match(self.candidate_cube, det):
                self.candidate_cube = det
                self.candidate_frames_count += 1
            else:
                self.candidate_cube = det
                self.candidate_frames_count = 1

            self.state = SortingState.CONFIRMING
            if self.candidate_frames_count >= self.stable_frames_required:
                self.get_logger().info(
                    f'Cube confirmed: {det.class_name} at ({x_mm:.1f}, {y_mm:.1f}) mm. '
                    'Dispatching pick and place goal!'
                )
                self._dispatch_pick_and_place(det, x_mm, y_mm)
        else:
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.state = SortingState.SCANNING

    def _handle_scanning(
        self, frame: np.ndarray
    ) -> list[tuple[CubeDetection, str, float, float]]:
        """Detect cubes and track middle-zone candidate."""
        all_detections, middle_detections = self._detect_and_map_cubes(frame)
        self._update_candidate_tracking(middle_detections)
        return all_detections

    def _publish_debug_image(
        self,
        frame: np.ndarray,
        detections: list[tuple[CubeDetection, str, float, float]],
        header,
    ) -> None:
        """Publish annotated debug overlay if subscribers exist."""
        status_extra = ''
        if self.state == SortingState.CONFIRMING:
            status_extra = f' ({self.candidate_frames_count}/{self.stable_frames_required})'
        elif self.state == SortingState.EXECUTING:
            status_extra = f' [{self.current_action_phase}]'

        self.visualizer.publish(
            frame=frame,
            header=header,
            state_text=self.state.value,
            calibration=self.calibration,
            detections=detections,
            status_extra=status_extra,
        )

    def _handle_image(self, msg: Image) -> None:
        frame = self._convert_image(msg)
        if frame is None:
            return

        detections_with_zones: list[tuple[CubeDetection, str, float, float]] = []

        if self.state == SortingState.CALIBRATING:
            self._handle_calibrating(frame)
        elif self.state == SortingState.COOLDOWN:
            self._handle_cooldown()
        elif self.state in (SortingState.SCANNING, SortingState.CONFIRMING):
            detections_with_zones = self._handle_scanning(frame)
        elif self.state == SortingState.EXECUTING:
            self._handle_executing()

        self._publish_debug_image(frame, detections_with_zones, msg.header)

    def _dispatch_pick_and_place(
        self, det: CubeDetection, pick_x_mm: float, pick_y_mm: float
    ) -> None:
        if not self.action_client.server_is_ready():
            self.get_logger().warn('Pick-and-place action server not ready!')
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.state = SortingState.SCANNING
            return

        self.state = SortingState.EXECUTING
        self.current_action_phase = 'SENDING_GOAL'

        # Calculate pick pose
        x_m, y_m, z_m = self.coord_mapper.paper_to_robot_xyz(pick_x_mm, pick_y_mm)
        pick_pose = self.coord_mapper.create_pose_stamped(x_m, y_m, z_m)

        # White cube -> left zone, green cube -> right zone
        target_zone = 'left' if det.class_name == 'white_cube' else 'right'
        px_m, py_m, pz_m = self.coord_mapper.get_zone_center_robot_xyz(target_zone)
        place_pose = self.coord_mapper.create_pose_stamped(px_m, py_m, pz_m)

        goal = PickObject.Goal()
        goal.target_pose = pick_pose
        goal.place_pose = place_pose
        goal.object_class = det.class_name
        goal.approach_distance = 0.05

        self.get_logger().info(
            f'Sending Goal: {det.class_name} | Pick: ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) -> '
            f'Place ({target_zone}): ({px_m:.2f}, {py_m:.2f}, {pz_m:.2f})'
        )

        send_goal_future = self.action_client.send_goal_async(
            goal, feedback_callback=self._action_feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by pick-and-place server!')
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.state = SortingState.SCANNING
            return

        self.get_logger().info('Goal accepted, executing...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._action_result_callback)

    def _action_feedback_callback(self, feedback_msg):
        self.current_action_phase = feedback_msg.feedback.current_phase
        self.get_logger().debug(
            f'Action phase: {self.current_action_phase} '
            f'({feedback_msg.feedback.progress_percentage:.0f}%)'
        )

    def _action_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Pick-and-place completed. Success={result.success}, '
            f'Time={result.execution_time:.1f}s, Error="{result.error_code}"'
        )
        self.state = SortingState.COOLDOWN
        self.cooldown_start_time = time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = SortingCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
