"""Sorting Coordinator Node for EezyBotArm computer vision and pick-and-place."""

from enum import Enum
import os
import time

import cv2
from cv_bridge import CvBridge
from eezybotarm_interfaces.action import PickObject
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image

from .coordinate_mapper import CoordinateMapper
from .yolo_detector import CubeDetection, YoloCubeDetector
from .zone_detector import (
    detect_zones,
    draw_zones,
    get_real_coordinates,
    identify_zone,
    ZoneDetectionResult,
)


class SortingState(Enum):
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
        self.declare_parameter('table_z_m', 0.075)
        self.declare_parameter('cube_height_m', 0.025)
        self.declare_parameter('action_name', 'pick_and_place')

        weights_path = self.get_parameter('weights_path').value or default_model_path
        if not os.path.exists(weights_path) and os.path.exists(default_model_path):
            weights_path = default_model_path
        conf_thresh = self.get_parameter('conf_threshold').value
        self.stable_frames_required = self.get_parameter('stable_frames_required').value
        self.cooldown_duration_sec = self.get_parameter('cooldown_duration_sec').value

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
        self.debug_image_pub = self.create_publisher(Image, '/eezybotarm_vision/debug_image', 10)

        # State management
        self.state = SortingState.SCANNING
        self.active_calibration: ZoneDetectionResult | None = None
        self.candidate_cube: CubeDetection | None = None
        self.candidate_frames_count = 0
        self.cooldown_start_time = 0.0
        self.current_action_phase = ''
        self._is_processing = False

        self.get_logger().info('Sorting Coordinator Node ready and monitoring /camera/image_raw')

    def image_callback(self, msg: Image) -> None:
        if self._is_processing:
            return
        self._is_processing = True
        try:
            self._handle_image(msg)
        finally:
            self._is_processing = False

    def _handle_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert image frame: {exc}', throttle_duration_sec=2.0
            )
            return

        now = time.monotonic()

        # 1. Calibrate workspace zones once, then lock
        if self.active_calibration is None:
            detected_zones = detect_zones(
                frame,
                field_width_mm=self.coord_mapper.paper_width_mm,
                field_height_mm=self.coord_mapper.paper_height_mm,
            )
            if detected_zones is not None:
                self.active_calibration = detected_zones
                self.get_logger().info('Workspace zones detected and locked.')

        # 2. State Machine Execution
        if self.state == SortingState.COOLDOWN:
            if now - self.cooldown_start_time >= self.cooldown_duration_sec:
                self.state = SortingState.SCANNING
                self.candidate_cube = None
                self.candidate_frames_count = 0
                self.get_logger().info('Cooldown finished, resumed scanning.')

        detections_with_zones: list[tuple[CubeDetection, str, float, float]] = []
        middle_detections: list[tuple[CubeDetection, str, float, float]] = []

        if self.state in (SortingState.SCANNING, SortingState.CONFIRMING):
            raw_detections = self.detector.predict(frame)

            for det in raw_detections:
                if self.active_calibration is not None:
                    x_mm, y_mm = get_real_coordinates(
                        det.center_px[0],
                        det.center_px[1],
                        self.active_calibration.homography_matrix,
                    )
                    zone = identify_zone(
                        x_mm,
                        y_mm,
                        self.coord_mapper.paper_width_mm,
                        self.coord_mapper.paper_height_mm,
                    )
                else:
                    zone = 'unknown'
                    x_mm, y_mm = 0.0, 0.0

                item = (det, zone, x_mm, y_mm)
                detections_with_zones.append(item)

                if zone == 'middle':
                    middle_detections.append(item)

            middle_candidate = None
            if self.candidate_cube is not None:
                for item in middle_detections:
                    det = item[0]
                    if (
                        det.class_id == self.candidate_cube.class_id
                        and abs(self.candidate_cube.center_px[0] - det.center_px[0]) < 25.0
                        and abs(self.candidate_cube.center_px[1] - det.center_px[1]) < 25.0
                    ):
                        middle_candidate = item
                        break
            if middle_candidate is None and middle_detections:
                middle_candidate = middle_detections[0]

            if middle_candidate is not None and self.active_calibration is not None:
                det, _, x_mm, y_mm = middle_candidate
                if (
                    self.candidate_cube is not None
                    and self.candidate_cube.class_id == det.class_id
                    and abs(self.candidate_cube.center_px[0] - det.center_px[0]) < 25.0
                    and abs(self.candidate_cube.center_px[1] - det.center_px[1]) < 25.0
                ):
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

        # 3. Publish debug overlay if subscribers exist
        if self.debug_image_pub.get_subscription_count() > 0:
            annotated = self._render_debug_overlay(frame, detections_with_zones)
            debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

    def _render_debug_overlay(
        self,
        frame: np.ndarray,
        detections: list[tuple[CubeDetection, str, float, float]],
    ) -> np.ndarray:
        annotated = frame.copy()
        if self.active_calibration is not None:
            annotated = draw_zones(annotated, self.active_calibration)
        else:
            cv2.putText(
                annotated,
                'Calibrating: Looking for 3 Zones...',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        for det, zone, x_mm, y_mm in detections:
            self._draw_cube(annotated, det, zone, x_mm, y_mm)

        status_text = f'State: {self.state.value}'
        if self.state == SortingState.CONFIRMING:
            status_text += f' ({self.candidate_frames_count}/{self.stable_frames_required})'
        elif self.state == SortingState.EXECUTING:
            status_text += f' [{self.current_action_phase}]'

        cv2.putText(
            annotated,
            status_text,
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
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
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = (0, 255, 0) if det.class_id == 0 else (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
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
            0.5,
            color,
            2,
        )
        if zone != 'unknown':
            coords_lbl = f'({x_mm:.0f}, {y_mm:.0f})mm'
            cv2.putText(
                img,
                coords_lbl,
                (x1, min(img.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

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
