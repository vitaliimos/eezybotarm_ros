"""Reachable workspace visualization node for EezyBotArm gripper."""

from dataclasses import dataclass
import math
from typing import Generator

from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Header
import urdf_parser_py.urdf as urdf
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class EezybotKinematics:
    """Kinematic parameters and joint limits of EezyBotArm extracted from URDF."""

    l1_arm_m: float
    l2_forearm_m: float
    l3_claw_x_m: float
    z_shoulder_offset_m: float
    z_claw_offset_m: float
    turret_y_offset_m: float
    waist_min: float
    waist_max: float
    shoulder_min: float
    shoulder_max: float
    elbow_min: float
    elbow_max: float

    @property
    def z_combined_offset_m(self) -> float:
        """Return combined vertical offset from base_link to claw frame."""
        return self.z_shoulder_offset_m + self.z_claw_offset_m

    def compute_ee_fk(self, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
        """
        Compute forward kinematics for end-effector frame (ee_link) in base_link.

        :param q1: waist_joint angle in radians (rotation about Z)
        :param q2: shoulder_joint angle in radians (rotation about Y)
        :param q3: elbow_joint angle in radians (rotation about Y)
        :return: (x, y, z) position in metres relative to base_link
        """
        r = self.l1_arm_m * math.sin(q2) + self.l2_forearm_m * math.cos(q3) + self.l3_claw_x_m
        z = (
            self.z_combined_offset_m
            + self.l1_arm_m * math.cos(q2)
            - self.l2_forearm_m * math.sin(q3)
        )
        x = r * math.cos(q1) - self.turret_y_offset_m * math.sin(q1)
        y = r * math.sin(q1) + self.turret_y_offset_m * math.cos(q1)
        return x, y, z

    @classmethod
    def from_urdf_string(cls, urdf_xml: str) -> 'EezybotKinematics':
        """
        Parse EezyBotArm URDF XML string and extract kinematic offsets and joint limits.

        :param urdf_xml: Complete URDF XML string of the robot
        :return: Configured EezybotKinematics instance
        """
        robot = urdf.Robot.from_xml_string(urdf_xml)
        joints = {j.name: j for j in robot.joints}

        waist = joints['waist_joint']
        shoulder = joints['shoulder_joint']
        shoulder_mimic = joints['shoulder_mimic_joint']
        elbow = joints['elbow_joint']
        wrist = joints['wrist_joint']
        claw_base = joints['claw_base_joint']
        ee = joints['ee_joint']

        l1_arm_m = float(shoulder_mimic.origin.xyz[2])
        l2_forearm_m = float(wrist.origin.xyz[0])
        l3_claw_x_m = float(claw_base.origin.xyz[0]) + float(ee.origin.xyz[0])
        z_shoulder_offset_m = float(waist.origin.xyz[2]) + float(shoulder.origin.xyz[2])
        z_claw_offset_m = float(claw_base.origin.xyz[2])

        turret_y_offset_m = (
            float(shoulder.origin.xyz[1])
            + float(shoulder_mimic.origin.xyz[1])
            + float(elbow.origin.xyz[1])
            + float(wrist.origin.xyz[1])
            + float(claw_base.origin.xyz[1])
            + float(ee.origin.xyz[1])
        )

        return cls(
            l1_arm_m=l1_arm_m,
            l2_forearm_m=l2_forearm_m,
            l3_claw_x_m=l3_claw_x_m,
            z_shoulder_offset_m=z_shoulder_offset_m,
            z_claw_offset_m=z_claw_offset_m,
            turret_y_offset_m=turret_y_offset_m,
            waist_min=float(waist.limit.lower),
            waist_max=float(waist.limit.upper),
            shoulder_min=float(shoulder.limit.lower),
            shoulder_max=float(shoulder.limit.upper),
            elbow_min=float(elbow.limit.lower),
            elbow_max=float(elbow.limit.upper),
        )


def height_to_color(normalized_h: float, alpha: float = 0.7) -> ColorRGBA:
    """
    Map normalized height [0.0, 1.0] to a color gradient (blue -> cyan -> green -> red).

    :param normalized_h: value in [0, 1]
    :param alpha: opacity of the point
    :return: ColorRGBA message
    """
    clamped = max(0.0, min(1.0, normalized_h))
    if clamped < 0.25:
        ratio = clamped / 0.25
        r, g, b = 0.0, ratio, 1.0
    elif clamped < 0.5:
        ratio = (clamped - 0.25) / 0.25
        r, g, b = 0.0, 1.0, 1.0 - ratio
    elif clamped < 0.75:
        ratio = (clamped - 0.5) / 0.25
        r, g, b = ratio, 1.0, 0.0
    else:
        ratio = (clamped - 0.75) / 0.25
        r, g, b = 1.0, 1.0 - ratio, 0.0

    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(alpha))


def frange(start: float, stop: float, step: float) -> Generator[float, None, None]:
    """Generate floating-point sequence inclusive of end within precision bounds."""
    current = start
    while current <= stop + 1e-9:
        yield current
        current += step


class WorkspaceVisualizer(Node):
    """Publishes RViz visualization markers representing the gripper's reachable workspace."""

    def __init__(self, **kwargs):
        """Initialize the workspace visualizer node and declare parameters."""
        super().__init__('workspace_visualizer', **kwargs)

        # Strictly required robot_description parameter (SSOT URDF)
        self.declare_parameter(
            'robot_description',
            Parameter.Type.STRING,
            ParameterDescriptor(description='URDF robot description XML string'),
        )
        robot_description = self.get_parameter('robot_description').value
        self.kinematics = EezybotKinematics.from_urdf_string(robot_description)

        # SSOT workspace geometry parameters from environment.yaml
        self.declare_parameter(
            'table_z_m',
            Parameter.Type.DOUBLE,
            ParameterDescriptor(description='Table surface height in metres'),
        )
        self.table_z_m = self.get_parameter('table_z_m').value

        self.declare_parameter(
            'cube_height_m',
            Parameter.Type.DOUBLE,
            ParameterDescriptor(description='Target cube height in metres'),
        )
        self.cube_height_m = self.get_parameter('cube_height_m').value

        # Optional grid resolution
        self.declare_parameter('step_deg', 4.0)
        self.step_deg = self.get_parameter('step_deg').value

        # QoS profile matching moveit.rviz WorkspaceMarkers display (transient local)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._marker_pub = self.create_publisher(MarkerArray, '/workspace_markers', qos_profile)

        # Precompute workspace markers
        self.markers = self.build_workspace_markers()
        self._marker_pub.publish(self.markers)
        self.get_logger().info(
            f'Published workspace markers to /workspace_markers '
            f'({len(self.markers.markers[0].points)} volume points sampled at {self.step_deg}°)'
        )

        # Periodically republish (1 Hz) to support newly connected RViz clients
        self._timer = self.create_timer(1.0, self._publish_timer_callback)

    def _publish_timer_callback(self):
        """Timer callback to republish markers."""
        stamp = self.get_clock().now().to_msg()
        for marker in self.markers.markers:
            marker.header.stamp = stamp
        self._marker_pub.publish(self.markers)

    def build_workspace_markers(self) -> MarkerArray:
        """Calculate reachable positions and construct MarkerArray for RViz."""
        step_rad = math.radians(self.step_deg)
        q1_vals = list(frange(self.kinematics.waist_min, self.kinematics.waist_max, step_rad))
        q2_vals = list(
            frange(self.kinematics.shoulder_min, self.kinematics.shoulder_max, step_rad)
        )
        q3_vals = list(frange(self.kinematics.elbow_min, self.kinematics.elbow_max, step_rad))

        volume_points: list[Point] = []
        table_points: list[Point] = []
        table_colors: list[ColorRGBA] = []

        z_min = float('inf')
        z_max = float('-inf')
        r_min = float('inf')
        r_max = float('-inf')

        # Grasping layer: cubes resting on table surface up to cube height + clearance
        grasp_z_min = self.table_z_m - 0.005
        grasp_z_max = self.table_z_m + self.cube_height_m + 0.010

        for q2 in q2_vals:
            for q3 in q3_vals:
                r = (
                    self.kinematics.l1_arm_m * math.sin(q2)
                    + self.kinematics.l2_forearm_m * math.cos(q3)
                    + self.kinematics.l3_claw_x_m
                )
                r_min = min(r_min, r)
                r_max = max(r_max, r)

                for q1 in q1_vals:
                    x, y, z = self.kinematics.compute_ee_fk(q1, q2, q3)
                    pt = Point(x=x, y=y, z=z)
                    volume_points.append(pt)

                    z_min = min(z_min, z)
                    z_max = max(z_max, z)

                    if grasp_z_min <= z <= grasp_z_max:
                        table_points.append(pt)
                        table_colors.append(ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.9))

        # Assign colors to volume points based on height
        z_span = max(1e-5, z_max - z_min)
        volume_colors = [
            height_to_color((pt.z - z_min) / z_span, alpha=0.6)
            for pt in volume_points
        ]

        header = Header()
        header.frame_id = 'base_link'
        header.stamp = self.get_clock().now().to_msg()

        # 1. 3D Volume Point Cloud Marker
        volume_marker = Marker()
        volume_marker.header = header
        volume_marker.ns = 'reachable_volume'
        volume_marker.id = 0
        volume_marker.type = Marker.POINTS
        volume_marker.action = Marker.ADD
        volume_marker.scale.x = 0.005
        volume_marker.scale.y = 0.005
        volume_marker.points = volume_points
        volume_marker.colors = volume_colors

        # 2. Table Grasping Footprint Marker
        table_marker = Marker()
        table_marker.header = header
        table_marker.ns = 'table_grasp_footprint'
        table_marker.id = 1
        table_marker.type = Marker.POINTS
        table_marker.action = Marker.ADD
        table_marker.scale.x = 0.007
        table_marker.scale.y = 0.007
        table_marker.points = table_points
        table_marker.colors = table_colors

        # 3. Outer boundary arc lines (min and max reach profile)
        boundary_marker = Marker()
        boundary_marker.header = header
        boundary_marker.ns = 'outer_boundaries'
        boundary_marker.id = 2
        boundary_marker.type = Marker.LINE_STRIP
        boundary_marker.action = Marker.ADD
        boundary_marker.scale.x = 0.003
        boundary_marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9)

        # Outer arc at table height
        for q1 in q1_vals:
            x = r_max * math.cos(q1)
            y = r_max * math.sin(q1)
            boundary_marker.points.append(Point(x=x, y=y, z=self.table_z_m))

        # 4. Text Marker with reachable space dimensions
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = 'workspace_stats'
        text_marker.id = 3
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = 0.15
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = z_max + 0.06
        text_marker.scale.z = 0.015
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        text_marker.text = (
            f'Gripper Workspace (ee_link)\n'
            f'Reach: {r_min * 100:.1f} .. {r_max * 100:.1f} cm\n'
            f'Height: {z_min * 100:.1f} .. {z_max * 100:.1f} cm\n'
            f'Table Z: {self.table_z_m * 100:.1f} cm'
        )

        marker_array = MarkerArray()
        marker_array.markers = [volume_marker, table_marker, boundary_marker, text_marker]
        return marker_array


def main(args=None):
    """Initialize and run the workspace visualizer node."""
    rclpy.init(args=args)
    node = WorkspaceVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
