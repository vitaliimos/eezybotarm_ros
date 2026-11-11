"""Sorting Coordinator Node for EezyBotArm pick-and-place state machine."""

from enum import Enum
import time
from typing import ClassVar

from eezybotarm_interfaces.action import PickObject
from eezybotarm_interfaces.msg import LocalizedCubeArray
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from .candidate_tracker import CandidateTracker
from .coordinate_mapper import CoordinateMapper
from .localized_cube import LocalizedCube
from .yolo_detector import CubeDetection


class SortingState(Enum):
    SCANNING = 'SCANNING'
    CONFIRMING = 'CONFIRMING'
    EXECUTING = 'EXECUTING'
    COOLDOWN = 'COOLDOWN'


class SortingCoordinatorNode(Node):
    """Track candidate cube stability in the middle zone and dispatch pick-and-place goals."""

    TARGET_ZONE_BY_CLASS: ClassVar[dict[str, str]] = {
        'white_cube': 'left',
        'green_cube': 'right',
    }

    def __init__(self, **kwargs):
        super().__init__('sorting_coordinator_node', **kwargs)

        # Coordinator parameters
        self.declare_parameter('stable_frames_required', Parameter.Type.INTEGER)
        self.declare_parameter('cooldown_duration_sec', Parameter.Type.DOUBLE)
        self.declare_parameter('action_name', Parameter.Type.STRING)

        # Physical workspace parameters (supplied via environment.yaml)
        self.declare_parameter('table_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_depth_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_x_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_y_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_z_m', Parameter.Type.DOUBLE)
        self.declare_parameter('cube_height_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('sticker_height_m', Parameter.Type.DOUBLE)

        stable_frames_required = self.get_parameter('stable_frames_required').value
        self.cooldown_duration_sec = self.get_parameter('cooldown_duration_sec').value
        action_name = self.get_parameter('action_name').value

        self.tracker = CandidateTracker(stable_frames_required=stable_frames_required)

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

        self.action_client = ActionClient(self, PickObject, action_name)

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            String,
            '/eezybotarm_vision/sorting_status',
            latched_qos,
        )
        self.subscription = self.create_subscription(
            LocalizedCubeArray,
            '/eezybotarm_vision/detections',
            self.detections_callback,
            1,
        )

        self._state = SortingState.SCANNING
        self.cooldown_start_time = 0.0
        self.current_action_phase = ''
        self._last_published_status = ''

        self.cooldown_timer = self.create_timer(0.1, self._handle_cooldown)
        self._publish_status()

        self.get_logger().info(f'Initial state: {self._state.value}')
        self.get_logger().info(
            'Sorting Coordinator Node ready and monitoring /eezybotarm_vision/detections'
        )

    @property
    def state(self) -> SortingState:
        return self._state

    @state.setter
    def state(self, new_state: SortingState) -> None:
        if self._state != new_state:
            self.get_logger().info(
                f'State changed: {self._state.value} -> {new_state.value}'
            )
            self._state = new_state

    def _format_status(self) -> str:
        if self.state == SortingState.CONFIRMING:
            return f'CONFIRMING ({self.tracker.frame_count}/{self.tracker.stable_frames_required})'
        if self.state == SortingState.EXECUTING and self.current_action_phase:
            return f'EXECUTING [{self.current_action_phase}]'
        return self.state.value

    def _publish_status(self) -> None:
        status_str = self._format_status()
        if status_str != self._last_published_status:
            self._last_published_status = status_str
            msg = String()
            msg.data = status_str
            self.status_pub.publish(msg)

    def _handle_cooldown(self) -> None:
        if self.state != SortingState.COOLDOWN:
            return
        if time.monotonic() - self.cooldown_start_time >= self.cooldown_duration_sec:
            self.state = SortingState.SCANNING
            self.tracker.reset()
            self.get_logger().info('Cooldown finished, resumed scanning.')
            self._publish_status()

    def _update_candidate_tracking(self, middle_cubes: list[LocalizedCube]) -> None:
        confirmed_cube = self.tracker.update(middle_cubes)
        if confirmed_cube is not None:
            det = confirmed_cube.detection
            self.state = SortingState.CONFIRMING
            self._publish_status()
            self.get_logger().info(
                f'Cube confirmed: {det.class_name} at '
                f'({confirmed_cube.x_mm:.1f}, {confirmed_cube.y_mm:.1f}) mm. '
                'Dispatching pick and place goal!'
            )
            self._dispatch_pick_and_place(det, confirmed_cube.x_mm, confirmed_cube.y_mm)
        elif self.tracker.candidate is not None:
            self.state = SortingState.CONFIRMING
            self._publish_status()
        else:
            self.state = SortingState.SCANNING
            self._publish_status()

    def detections_callback(self, msg: LocalizedCubeArray) -> None:
        if self.state == SortingState.COOLDOWN:
            self._handle_cooldown()
            return
        if self.state not in (SortingState.SCANNING, SortingState.CONFIRMING):
            return

        all_cubes = [LocalizedCube.from_msg(c) for c in msg.cubes]
        middle_cubes = [c for c in all_cubes if c.zone == 'middle']
        self._update_candidate_tracking(middle_cubes)

    def _reset_to_scanning(self) -> None:
        self.tracker.reset()
        self.current_action_phase = ''
        self.state = SortingState.SCANNING
        self._publish_status()

    def _dispatch_pick_and_place(
        self, det: CubeDetection, pick_x_mm: float, pick_y_mm: float
    ) -> None:
        if not self.action_client.server_is_ready():
            self.get_logger().warn('Pick-and-place action server not ready!')
            self._reset_to_scanning()
            return

        target_zone = self.TARGET_ZONE_BY_CLASS.get(det.class_name)
        if target_zone is None:
            self.get_logger().error(
                f'No target zone configured for class "{det.class_name}"'
            )
            self._reset_to_scanning()
            return

        self.state = SortingState.EXECUTING
        self.current_action_phase = 'SENDING_GOAL'
        self._publish_status()

        x_m, y_m, z_m = self.coord_mapper.table_to_robot_xyz(pick_x_mm, pick_y_mm)
        pick_pose = self.coord_mapper.create_pose_stamped(x_m, y_m, z_m)

        px_m, py_m, pz_m = self.coord_mapper.get_zone_center_robot_xyz(target_zone)
        place_pose = self.coord_mapper.create_pose_stamped(px_m, py_m, pz_m)

        goal = PickObject.Goal()
        goal.target_pose = pick_pose
        goal.place_pose = place_pose
        goal.object_class = det.class_name

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
        self._publish_status()
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

        self.current_action_phase = ''
        self.state = SortingState.COOLDOWN
        self.cooldown_start_time = time.monotonic()
        self._publish_status()


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
