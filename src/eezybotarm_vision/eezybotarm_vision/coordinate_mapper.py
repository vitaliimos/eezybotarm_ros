"""Coordinate mapping from paper workspace coordinates to robot base_link frame."""

from geometry_msgs.msg import PoseStamped


class CoordinateMapper:
    """Maps paper workspace (x_mm, y_mm) coordinates to robot base_link poses."""

    def __init__(
        self,
        paper_width_mm: float = 300.0,
        paper_height_mm: float = 200.0,
        robot_x_offset_m: float = 0.150,
        robot_y_offset_m: float = 0.0,
        table_z_m: float = 0.095,
        cube_height_m: float = 0.025,
    ):
        self.paper_width_mm = paper_width_mm
        self.paper_height_mm = paper_height_mm
        self.robot_x_offset_m = robot_x_offset_m
        self.robot_y_offset_m = robot_y_offset_m
        self.table_z_m = table_z_m
        self.cube_height_m = cube_height_m

    def paper_to_robot_xyz(self, x_mm: float, y_mm: float) -> tuple[float, float, float]:
        """Convert paper coordinates (mm) to robot base_link frame (meters)."""
        x_robot = self.robot_x_offset_m + (y_mm / 1000.0)
        center_x_mm = self.paper_width_mm / 2.0
        y_robot = self.robot_y_offset_m + ((x_mm - center_x_mm) / 1000.0)
        z_robot = self.table_z_m + (self.cube_height_m / 2.0)
        return x_robot, y_robot, z_robot

    def get_zone_center_robot_xyz(self, zone_name: str) -> tuple[float, float, float]:
        """Get pre-calculated center of specified zone in robot base_link coordinates."""
        zone_width = self.paper_width_mm / 3.0
        zone_multipliers = {'left': 0.5, 'middle': 1.5, 'right': 2.5}
        multiplier = zone_multipliers.get(zone_name, 1.5)
        return self.paper_to_robot_xyz(zone_width * multiplier, self.paper_height_mm / 2.0)

    def create_pose_stamped(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        frame_id: str = 'base_link',
    ) -> PoseStamped:
        """Create a PoseStamped for MoveIt 2 Cartesian planning."""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x_m
        pose.pose.position.y = y_m
        pose.pose.position.z = z_m
        pose.pose.orientation.w = 1.0
        return pose
