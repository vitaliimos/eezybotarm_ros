"""NiceGUI Web Dashboard ROS 2 Node for EezyBotArm."""

import asyncio
import math
import threading
import time
from typing import Optional

from builtin_interfaces.msg import Duration
import cv2
from cv_bridge import CvBridge
from eezybotarm_interfaces.action import PickObject
from fastapi.responses import Response, StreamingResponse
from geometry_msgs.msg import PoseStamped
from nicegui import app, ui
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, Image, JointState
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class EezybotDashboardNode(Node):
    """ROS 2 node that powers the NiceGUI dashboard backend."""

    def __init__(self, **kwargs):
        super().__init__('eezybotarm_dashboard', **kwargs)

        # Configuration parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('camera_raw_topic', '/camera/image_raw/compressed')
        self.declare_parameter('camera_debug_topic', '/eezybotarm_vision/debug_image/compressed')

        # Physical workspace parameters (must be supplied via environment.yaml / launch)
        self.declare_parameter('table_z_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_depth_m', Parameter.Type.DOUBLE)
        self.declare_parameter('table_width_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_x_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('robot_y_offset_m', Parameter.Type.DOUBLE)
        self.declare_parameter('cube_height_m', Parameter.Type.DOUBLE)

        self.cb_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()

        # Telemetry storage
        self.joint_positions: dict[str, float] = {
            'waist_joint': 0.0,
            'shoulder_joint': 0.0,
            'elbow_joint': 0.0,
            'finger_left_joint': 0.0,
        }
        self.ee_pose: dict[str, float] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.last_joint_update = 0.0

        # Video frames storage (JPEG bytes)
        self._raw_frame: Optional[bytes] = None
        self._debug_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        # Action client state
        self.action_status = 'IDLE'
        self.action_phase = 'IDLE'
        self.action_progress = 0.0
        self._active_goal_handle = None

        # Setup TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Setup publishers for controllers
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10, callback_group=self.cb_group
        )
        self.hand_pub = self.create_publisher(
            JointTrajectory, '/hand_controller/joint_trajectory', 10, callback_group=self.cb_group
        )

        # Setup subscriptions
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.cb_group,
        )

        raw_topic = self.get_parameter('camera_raw_topic').value
        debug_topic = self.get_parameter('camera_debug_topic').value

        self._setup_image_subscriber(raw_topic, is_debug=False)
        self._setup_image_subscriber(debug_topic, is_debug=True)

        # Action Client
        self.action_client = ActionClient(
            self, PickObject, 'pick_and_place', callback_group=self.cb_group
        )

        # Periodic timer for TF lookup (10 Hz)
        self.create_timer(0.1, self._update_ee_pose, callback_group=self.cb_group)

        self.get_logger().info('EezyBot Dashboard ROS Node initialized successfully')

    def _setup_image_subscriber(self, topic: str, is_debug: bool):
        if 'compressed' in topic:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg: self._compressed_image_callback(msg, is_debug),
                1,
                callback_group=self.cb_group,
            )
        else:
            self.create_subscription(
                Image,
                topic,
                lambda msg: self._raw_image_callback(msg, is_debug),
                1,
                callback_group=self.cb_group,
            )

    def _joint_state_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_positions:
                self.joint_positions[name] = pos
        self.last_joint_update = time.time()

    def _compressed_image_callback(self, msg: CompressedImage, is_debug: bool):
        with self._frame_lock:
            if is_debug:
                self._debug_frame = bytes(msg.data)
            else:
                self._raw_frame = bytes(msg.data)

    def _raw_image_callback(self, msg: Image, is_debug: bool):
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, buffer = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buffer.tobytes()
            with self._frame_lock:
                if is_debug:
                    self._debug_frame = jpeg_bytes
                else:
                    self._raw_frame = jpeg_bytes
        except Exception as exc:
            self.get_logger().error(f'Error converting raw image: {exc}', throttle_duration_sec=5.0)

    def _update_ee_pose(self):
        try:
            # Look up end effector position relative to base_link
            t = self.tf_buffer.lookup_transform(
                'base_link', 'ee_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)
            )
            self.ee_pose['x'] = t.transform.translation.x * 1000.0  # in mm
            self.ee_pose['y'] = t.transform.translation.y * 1000.0
            self.ee_pose['z'] = t.transform.translation.z * 1000.0
        except Exception:
            # TF might not be ready yet during boot/startup
            pass

    def get_latest_frame(self, stream_type: str = 'debug') -> bytes:
        with self._frame_lock:
            frame = self._debug_frame if stream_type == 'debug' else self._raw_frame
            if frame is None and stream_type == 'debug':
                frame = self._raw_frame  # fallback to raw if debug not publishing

        if frame is not None:
            return frame

        # Placeholder image when camera is offline
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, 'No Camera Signal', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        _, buf = cv2.imencode('.jpg', img)
        return buf.tobytes()

    def send_gripper_command(self, position: float, duration_sec: float = 0.5):
        traj = JointTrajectory()
        traj.joint_names = ['finger_left_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        secs = int(duration_sec)
        nsecs = int((duration_sec - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)
        traj.points = [point]
        self.hand_pub.publish(traj)
        self.get_logger().info(f'Sent gripper command: {position:.2f} rad')

    def send_arm_joints(self, waist: float, shoulder: float, elbow: float, duration_sec: float = 1.5):
        traj = JointTrajectory()
        traj.joint_names = ['waist_joint', 'shoulder_joint', 'elbow_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(waist), float(shoulder), float(elbow)]
        secs = int(duration_sec)
        nsecs = int((duration_sec - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)
        traj.points = [point]
        self.arm_pub.publish(traj)
        self.get_logger().info(
            f'Sent arm joints command: waist={waist:.2f}, shoulder={shoulder:.2f}, elbow={elbow:.2f}'
        )

    def calculate_zone_xyz(self, zone_name: str) -> tuple[float, float, float]:
        """Calculates metric coordinates for zone center (in base_link frame)."""
        table_width_m = self.get_parameter('table_width_m').value
        table_depth_m = self.get_parameter('table_depth_m').value
        robot_x_offset_m = self.get_parameter('robot_x_offset_m').value
        robot_y_offset_m = self.get_parameter('robot_y_offset_m').value
        table_z_m = self.get_parameter('table_z_m').value
        cube_height_m = self.get_parameter('cube_height_m').value

        zone_multipliers = {'left': 0.5, 'middle': 1.5, 'right': 2.5}
        multiplier = zone_multipliers.get(zone_name, 1.5)

        x_paper_m = (table_width_m / 3.0) * multiplier
        y_paper_m = table_depth_m / 2.0

        x_robot = robot_x_offset_m + y_paper_m
        y_robot = robot_y_offset_m + (x_paper_m - (table_width_m / 2.0))
        z_robot = table_z_m + (cube_height_m / 2.0)
        return x_robot, y_robot, z_robot

    def trigger_pick_and_place(self, pick_zone: str = 'middle', place_zone: str = 'right'):
        if not self.action_client.wait_for_server(timeout_sec=1.0):
            self.action_status = 'SERVER NOT AVAILABLE'
            return False

        px, py, pz = self.calculate_zone_xyz(pick_zone)
        rx, ry, rz = self.calculate_zone_xyz(place_zone)

        goal_msg = PickObject.Goal()
        goal_msg.object_class = 'cube'
        goal_msg.approach_distance = 0.04

        # Pick pose
        goal_msg.target_pose.header.frame_id = 'base_link'
        goal_msg.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.target_pose.pose.position.x = px
        goal_msg.target_pose.pose.position.y = py
        goal_msg.target_pose.pose.position.z = pz
        goal_msg.target_pose.pose.orientation.w = 1.0

        # Place pose
        goal_msg.place_pose.header.frame_id = 'base_link'
        goal_msg.place_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.place_pose.pose.position.x = rx
        goal_msg.place_pose.pose.position.y = ry
        goal_msg.place_pose.pose.position.z = rz
        goal_msg.place_pose.pose.orientation.w = 1.0

        self.action_status = 'SENDING GOAL...'
        self.action_progress = 0.0

        send_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self._action_feedback_callback
        )
        send_future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.action_status = 'GOAL REJECTED'
            self._active_goal_handle = None
            return

        self._active_goal_handle = goal_handle
        self.action_status = 'EXECUTING'
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._action_result_callback)

    def _action_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.action_phase = feedback.current_phase
        self.action_progress = feedback.progress_percentage

    def _action_result_callback(self, future):
        result = future.result().result
        self._active_goal_handle = None
        if result.success:
            self.action_status = f'SUCCESS ({result.execution_time:.1f}s)'
            self.action_progress = 100.0
        else:
            self.action_status = f'FAILED: {result.error_code}'

    def cancel_active_goal(self):
        if self._active_goal_handle is not None:
            self.action_status = 'CANCELING...'
            self._active_goal_handle.cancel_goal_async()


def build_dashboard_ui(node: EezybotDashboardNode):
    """Assembles the NiceGUI layout and visual components."""
    ui.colors(primary='#1976D2', secondary='#26A69A', accent='#9C27B0', dark='#1D1D1D')

    # Header
    with ui.header().classes('items-center justify-between bg-primary text-white p-3 shadow-md'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('smart_toy', size='md')
            ui.label('EezyBot Dashboard').classes('text-xl font-bold tracking-wide')
        with ui.row().classes('items-center gap-4'):
            heartbeat_badge = ui.badge('DDS: OK', color='green').classes('text-sm px-2 py-1')

    # Main layout
    with ui.row().classes('w-full max-w-7xl mx-auto p-4 gap-4 items-start'):
        # LEFT COLUMN: Camera Feed and Telemetry
        with ui.column().classes('flex-1 gap-4'):
            # Camera Card
            with ui.card().classes('w-full shadow-lg rounded-xl p-4'):
                with ui.row().classes('justify-between items-center w-full mb-2'):
                    ui.label('Live Camera Stream').classes('text-lg font-semibold')
                    stream_select = ui.select(
                        options={'debug': 'Annotated (YOLO & Zones)', 'raw': 'Raw Camera'},
                        value='debug',
                    ).classes('w-60')

                # Video display via standard MJPEG endpoint
                video_url = f'/stream/video?type={stream_select.value}'
                video_img = (
                    ui.image(video_url)
                    .props('fit=contain :ratio="16/9"')
                    .classes('w-full aspect-[16/9] rounded-lg border border-gray-300 shadow-inner bg-black')
                )
                stream_select.on_value_change(
                    lambda e: video_img.set_source(f'/stream/video?type={e.value}&t={time.time()}')
                )

            # Robot Telemetry Card
            with ui.card().classes('w-full shadow-lg rounded-xl p-4'):
                ui.label('Robot Telemetry').classes('text-lg font-semibold mb-3')
                with ui.grid(columns=2).classes('w-full gap-4'):
                    with ui.card().classes('bg-blue-50 p-3 rounded-lg'):
                        ui.label('End-Effector Position (mm)').classes('text-sm text-gray-500 font-bold')
                        ee_label = ui.label('X: 0.0 | Y: 0.0 | Z: 0.0').classes('text-base font-mono font-bold mt-1')

                    with ui.card().classes('bg-green-50 p-3 rounded-lg'):
                        ui.label('Controllers Status').classes('text-sm text-gray-500 font-bold')
                        controller_status = ui.label('Active / Broadcasting').classes(
                            'text-base font-semibold text-green-700 mt-1'
                        )

                with ui.grid(columns=4).classes('w-full gap-2 mt-3'):
                    waist_val = ui.label('Waist: 0.0°').classes('text-sm font-mono')
                    shoulder_val = ui.label('Shoulder: 0.0°').classes('text-sm font-mono')
                    elbow_val = ui.label('Elbow: 0.0°').classes('text-sm font-mono')
                    finger_val = ui.label('Gripper: 0.0°').classes('text-sm font-mono')

        # RIGHT COLUMN: Motion Controls & Pick-and-Place
        with ui.column().classes('w-96 gap-4'):
            # Gripper Control Card
            with ui.card().classes('w-full shadow-lg rounded-xl p-4'):
                ui.label('Gripper Control').classes('text-lg font-semibold mb-2')
                with ui.row().classes('w-full gap-2 justify-center mb-3'):
                    ui.button(
                        'Open Gripper',
                        on_click=lambda: (node.send_gripper_command(0.65), grip_slider.set_value(0.65)),
                    ).props('color=positive icon=pan_tool')
                    ui.button(
                        'Close Gripper',
                        on_click=lambda: (node.send_gripper_command(0.0), grip_slider.set_value(0.0)),
                    ).props('color=negative icon=back_hand')

                ui.label('Manual Position (rad):').classes('text-xs text-gray-500')
                grip_slider = ui.slider(
                    min=0.0,
                    max=0.65,
                    step=0.05,
                    value=0.0,
                    on_change=lambda e: node.send_gripper_command(e.value),
                ).props('label-always')

            # Manual Arm Joints Card
            with ui.card().classes('w-full shadow-lg rounded-xl p-4'):
                ui.label('Manual Joint Control').classes('text-lg font-semibold mb-2')

                ui.label('Waist Joint:').classes('text-xs text-gray-500')
                waist_slider = ui.slider(min=-1.2, max=1.2, step=0.05, value=0.0).props('label-always')

                ui.label('Shoulder Joint:').classes('text-xs text-gray-500 mt-2')
                shoulder_slider = ui.slider(min=-0.45, max=0.80, step=0.05, value=0.0).props('label-always')

                ui.label('Elbow Joint:').classes('text-xs text-gray-500 mt-2')
                elbow_slider = ui.slider(min=-0.25, max=0.85, step=0.05, value=0.0).props('label-always')

                with ui.row().classes('w-full justify-between mt-3'):
                    def apply_joints():
                        node.send_arm_joints(waist_slider.value, shoulder_slider.value, elbow_slider.value)

                    ui.button('Send Target', on_click=apply_joints).props('color=primary icon=send')

                    def set_pose(w, s, e):
                        waist_slider.set_value(w)
                        shoulder_slider.set_value(s)
                        elbow_slider.set_value(e)
                        node.send_arm_joints(w, s, e)

                    ui.button('Home', on_click=lambda: set_pose(0.0, 0.0, 0.0)).props('outline icon=home')
                    ui.button('Standby', on_click=lambda: set_pose(-1.2, -0.45, -0.25)).props(
                        'outline icon=pause'
                    )

            # Pick and Place Action Card
            with ui.card().classes('w-full shadow-lg rounded-xl p-4 bg-gray-50'):
                ui.label('Pick & Place Workflow').classes('text-lg font-semibold mb-2')
                with ui.row().classes('w-full gap-2'):
                    pick_sel = ui.select(['left', 'middle', 'right'], value='middle', label='Pick Zone').classes(
                        'flex-1'
                    )
                    place_sel = ui.select(['left', 'middle', 'right'], value='right', label='Place Zone').classes(
                        'flex-1'
                    )

                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button(
                        'Run Pick & Place',
                        on_click=lambda: node.trigger_pick_and_place(pick_sel.value, place_sel.value),
                    ).props('color=secondary icon=play_arrow').classes('flex-1')
                    ui.button('Cancel', on_click=node.cancel_active_goal).props('color=negative icon=stop')

                ui.label('Action Status:').classes('text-xs text-gray-500 mt-3')
                action_status_badge = ui.label(node.action_status).classes('font-bold text-sm text-primary')

                ui.label('Current Phase:').classes('text-xs text-gray-500 mt-1')
                action_phase_label = ui.label('None').classes('font-mono text-xs')

                progress_bar = ui.linear_progress(value=0.0, show_value=False).props('instant-feedback').classes('mt-2')

    # UI Refresh Timer (10 Hz)
    def update_ui_telemetry():
        ee = node.ee_pose
        ee_label.set_text(f"X: {ee['x']:.1f} | Y: {ee['y']:.1f} | Z: {ee['z']:.1f}")

        jp = node.joint_positions
        waist_val.set_text(f"Waist: {math.degrees(jp['waist_joint']):.1f}°")
        shoulder_val.set_text(f"Shoulder: {math.degrees(jp['shoulder_joint']):.1f}°")
        elbow_val.set_text(f"Elbow: {math.degrees(jp['elbow_joint']):.1f}°")
        finger_val.set_text(f"Grip: {jp['finger_left_joint']:.2f} rad")

        if time.time() - node.last_joint_update > 2.0:
            heartbeat_badge.set_text('DDS: NO JOINTS').props('color=red')
        else:
            heartbeat_badge.set_text('DDS: CONNECTED').props('color=green')

        action_status_badge.set_text(f'Status: {node.action_status}')
        action_phase_label.set_text(f'Phase: {node.action_phase}')
        progress_bar.set_value(node.action_progress / 100.0)

    ui.timer(0.1, update_ui_telemetry)


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    node = EezybotDashboardNode()

    # Spin ROS 2 in background thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # FastAPI MJPEG Streaming Endpoint
    @app.get('/stream/video')
    async def video_stream(type: str = 'debug'):
        async def frame_gen():
            while True:
                frame = node.get_latest_frame(type)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                await asyncio.sleep(0.04)

        return StreamingResponse(frame_gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    # Setup NiceGUI pages
    @ui.page('/')
    def index():
        build_dashboard_ui(node)

    host = node.get_parameter('host').value
    port = node.get_parameter('port').value

    app.on_shutdown(lambda: (node.destroy_node(), rclpy.shutdown()))

    ui.run(host=host, port=port, title='EezyBotArm Dashboard', reload=False)


if __name__ == '__main__':
    main()
