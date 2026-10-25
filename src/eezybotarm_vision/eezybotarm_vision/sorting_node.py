"""Sorting Coordinator Node for EezyBotArm computer vision and pick-and-place."""

from enum import Enum
import time

from cv_bridge import CvBridge
from eezybotarm_interfaces.action import PickObject
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from .coordinate_mapper import CoordinateMapper
from .debug_visualizer import DebugVisualizer, LocalizedCube
from .yolo_detector import CubeDetection, YoloCubeDetector
from .zone_detector import (
    detect_zones,
    get_real_coordinates,
    identify_zone,
    smooth_calibration,
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

    TARGET_ZONE_BY_CLASS = {'white_cube': 'left', 'green_cube': 'right'}
    APPROACH_DISTANCE_M = 0.05

    def __init__(self, **kwargs):
        super().__init__('sorting_coordinator_node', **kwargs)

        self.bridge = CvBridge()

        # Vision and coordinator parameters
        self.declare_parameter('weights_path', Parameter.Type.STRING)
        self.declare_parameter('conf_threshold', Parameter.Type.DOUBLE)
        self.declare_parameter('stable_frames_required', Parameter.Type.INTEGER)
        self.declare_parameter('cooldown_duration_sec', Parameter.Type.DOUBLE)
        self.declare_parameter('debug_image_jpeg_quality', Parameter.Type.INTEGER)
        self.declare_parameter('action_name', Parameter.Type.STRING)
        self.declare_parameter('recalibration_interval_sec', Parameter.Type.DOUBLE)
        self.declare_parameter('calibration_smoothing_alpha', Parameter.Type.DOUBLE)

        # Physical workspace parameters (supplied via environment.yaml)
        self.declare_parameter('table_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_depth_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_x_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_y_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_z_m', Parameter.Type.DOUBLE)
        self.declare_parameter('cube_height_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_height_m', Parameter.Type.DOUBLE)

        weights_path = self.get_parameter('weights_path').value
        conf_thresh = self.get_parameter('conf_threshold').value
        self.stable_frames_required = self.get_parameter('stable_frames_required').value
        self.cooldown_duration_sec = self.get_parameter('cooldown_duration_sec').value
        jpeg_quality = self.get_parameter('debug_image_jpeg_quality').value
        action_name = self.get_parameter('action_name').value
        self.recalibration_interval_sec = self.get_parameter('recalibration_interval_sec').value
        self.calibration_smoothing_alpha = self.get_parameter('calibration_smoothing_alpha').value

        self.coord_mapper = CoordinateMapper(
            table_width_m=self.get_parameter('table_width_m').value,
            table_depth_m=self.get_parameter('table_depth_m').value,
            robot_x_offset_m=self.get_parameter('robot_x_offset_m').value,
            robot_y_offset_m=self.get_parameter('robot_y_offset_m').value,
            table_z_m=self.get_parameter('table_z_m').value,
            cube_height_m=self.get_parameter('cube_height_m').value,
            sticker_width_m=self.get_parameter('sticker_width_m').value,
            sticker_height_m=self.get_parameter('sticker_height_m').value,
        )

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

        self.action_client = ActionClient(self, PickObject, action_name)

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
        )
        self.visualizer = DebugVisualizer(node=self, jpeg_quality=jpeg_quality)

        self._state = SortingState.CALIBRATING
        self.calibration: ZoneDetectionResult | None = None
        self.candidate_cube: CubeDetection | None = None
        self.candidate_frames_count = 0
        self.cooldown_start_time = 0.0
        self.last_recalibration_time = 0.0
        self.current_action_phase = ''
        self._is_processing = False

        self.get_logger().info(f'Initial state: {self._state.value}')
        self.get_logger().info('Sorting Coordinator Node ready and monitoring /camera/image_raw')

    @property
    def state(self) -> SortingState:
        return self._state

    @state.setter
    def state(self, new_state: SortingState) -> None:
        if self._state != new_state:
            self.get_logger().info(f'State changed: {self._state.value} -> {new_state.value}')
            self._state = new_state

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
            field_width_mm=self.coord_mapper.table_width_mm,
            field_height_mm=self.coord_mapper.table_depth_mm,
            marker_inset_x_mm=self.coord_mapper.marker_inset_x_mm,
            marker_inset_y_mm=self.coord_mapper.marker_inset_y_mm,
        )

    def _handle_calibrating(self, frame: np.ndarray) -> None:
        detected_zones = self._detect_workspace_zones(frame)
        if detected_zones is not None:
            self.calibration = detected_zones
            self.last_recalibration_time = time.monotonic()
            self.get_logger().info('Workspace zones detected and locked.')
            self.state = SortingState.SCANNING

    def _handle_cooldown(self) -> None:
        if time.monotonic() - self.cooldown_start_time >= self.cooldown_duration_sec:
            self.state = SortingState.SCANNING
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.get_logger().info('Cooldown finished, resumed scanning.')

    def _detect_and_map_cubes(self, frame: np.ndarray) -> list[LocalizedCube]:
        assert self.calibration is not None
        raw_detections = self.detector.predict(frame)
        cubes: list[LocalizedCube] = []

        for det in raw_detections:
            x_mm, y_mm = get_real_coordinates(
                det.center_px[0],
                det.center_px[1],
                self.calibration.homography_matrix,
            )
            zone = identify_zone(
                x_mm,
                y_mm,
                self.coord_mapper.table_width_mm,
                self.coord_mapper.table_depth_mm,
            )
            cubes.append(LocalizedCube(detection=det, zone=zone, x_mm=x_mm, y_mm=y_mm))

        return cubes

    @staticmethod
    def _is_cube_match(
        a: CubeDetection, b: CubeDetection, max_dist_px: float = 25.0
    ) -> bool:
        return (
            a.class_id == b.class_id
            and abs(a.center_px[0] - b.center_px[0]) < max_dist_px
            and abs(a.center_px[1] - b.center_px[1]) < max_dist_px
        )

    def _update_candidate_tracking(self, middle_cubes: list[LocalizedCube]) -> None:
        matched_cube: LocalizedCube | None = None
        if self.candidate_cube is not None:
            for cube in middle_cubes:
                if self._is_cube_match(self.candidate_cube, cube.detection):
                    matched_cube = cube
                    break

        if matched_cube is None and middle_cubes:
            matched_cube = middle_cubes[0]

        if matched_cube is not None:
            det = matched_cube.detection
            if self.candidate_cube is not None and self._is_cube_match(self.candidate_cube, det):
                self.candidate_cube = det
                self.candidate_frames_count += 1
            else:
                self.candidate_cube = det
                self.candidate_frames_count = 1

            self.state = SortingState.CONFIRMING
            if self.candidate_frames_count >= self.stable_frames_required:
                self.get_logger().info(
                    f'Cube confirmed: {det.class_name} at '
                    f'({matched_cube.x_mm:.1f}, {matched_cube.y_mm:.1f}) mm. '
                    'Dispatching pick and place goal!'
                )
                self._dispatch_pick_and_place(det, matched_cube.x_mm, matched_cube.y_mm)
        else:
            self.candidate_cube = None
            self.candidate_frames_count = 0
            self.state = SortingState.SCANNING

    def _recalibrate_if_needed(self, frame: np.ndarray) -> None:
        if self.state != SortingState.SCANNING or self.calibration is None:
            return
        now = time.monotonic()
        if now - self.last_recalibration_time < self.recalibration_interval_sec:
            return

        self.last_recalibration_time = now
        detected_zones = self._detect_workspace_zones(frame)
        if detected_zones is not None:
            self.calibration = smooth_calibration(
                self.calibration,
                detected_zones,
                alpha=self.calibration_smoothing_alpha,
            )
            self.get_logger().debug('Workspace calibration smoothed and updated.')

    def _handle_scanning(self, frame: np.ndarray) -> list[LocalizedCube]:
        self._recalibrate_if_needed(frame)
        all_cubes = self._detect_and_map_cubes(frame)
        middle_cubes = [c for c in all_cubes if c.zone == 'middle']
        self._update_candidate_tracking(middle_cubes)
        return all_cubes

    def _publish_debug_image(
        self,
        frame: np.ndarray,
        detections: list[LocalizedCube],
        header,
    ) -> None:
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

        detections: list[LocalizedCube] = []

        if self.state == SortingState.CALIBRATING:
            self._handle_calibrating(frame)
        elif self.state == SortingState.COOLDOWN:
            self._handle_cooldown()
        elif self.state in (SortingState.SCANNING, SortingState.CONFIRMING):
            detections = self._handle_scanning(frame)

        self._publish_debug_image(frame, detections, msg.header)

    def _reset_to_scanning(self) -> None:
        self.candidate_cube = None
        self.candidate_frames_count = 0
        self.state = SortingState.SCANNING

    def _dispatch_pick_and_place(
        self, det: CubeDetection, pick_x_mm: float, pick_y_mm: float
    ) -> None:
        if not self.action_client.server_is_ready():
            self.get_logger().warn('Pick-and-place action server not ready!')
            self._reset_to_scanning()
            return

        target_zone = self.TARGET_ZONE_BY_CLASS.get(det.class_name)
        if target_zone is None:
            self.get_logger().error(f'No target zone configured for class "{det.class_name}"')
            self._reset_to_scanning()
            return

        self.state = SortingState.EXECUTING
        self.current_action_phase = 'SENDING_GOAL'

        x_m, y_m, z_m = self.coord_mapper.table_to_robot_xyz(pick_x_mm, pick_y_mm)
        pick_pose = self.coord_mapper.create_pose_stamped(x_m, y_m, z_m)

        px_m, py_m, pz_m = self.coord_mapper.get_zone_center_robot_xyz(target_zone)
        place_pose = self.coord_mapper.create_pose_stamped(px_m, py_m, pz_m)

        goal = PickObject.Goal()
        goal.target_pose = pick_pose
        goal.place_pose = place_pose
        goal.object_class = det.class_name
        goal.approach_distance = self.APPROACH_DISTANCE_M

        self.get_logger().info(
            f'Sending Goal: {det.class_name} | Pick: ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) -> '
            f'Place ({target_zone}): ({px_m:.2f}, {py_m:.2f}, {pz_m:.2f})'
        )

        send_goal_future = self.action_client.send_goal_async(
            goal, feedback_callback=self._action_feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send pick-and-place goal: {exc}')
            self._reset_to_scanning()
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by pick-and-place server!')
            self._reset_to_scanning()
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
        try:
            result = future.result().result
            self.get_logger().info(
                f'Pick-and-place completed. Success={result.success}, '
                f'Time={result.execution_time:.1f}s, Error="{result.error_code}"'
            )
        except Exception as exc:
            self.get_logger().error(f'Pick-and-place action failed: {exc}')

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
