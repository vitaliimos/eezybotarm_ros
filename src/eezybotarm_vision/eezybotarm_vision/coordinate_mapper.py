"""Coordinate mapping from table workspace coordinates to robot base_link frame."""

from geometry_msgs.msg import PoseStamped


class CoordinateMapper:
    """Maps table workspace (x_mm, y_mm) coordinates to robot base_link poses."""

    ZONE_MULTIPLIERS = {'left': 0.5, 'middle': 1.5, 'right': 2.5}

    def __init__(
        self,
        table_width_m: float,
        table_depth_m: float,
        robot_x_offset_m: float,
        robot_y_offset_m: float,
        table_z_m: float,
        cube_height_m: float,
        sticker_width_m: float,
        sticker_height_m: float,
    ):
        self.table_width_m = table_width_m
        self.table_depth_m = table_depth_m
        self.table_width_mm = table_width_m * 1000.0
        self.table_depth_mm = table_depth_m * 1000.0
        self.robot_x_offset_m = robot_x_offset_m
        self.robot_y_offset_m = robot_y_offset_m
        self.table_z_m = table_z_m
        self.cube_height_m = cube_height_m
        self.sticker_width_m = sticker_width_m
        self.sticker_height_m = sticker_height_m
        self.marker_inset_x_mm = (sticker_width_m * 1000.0) / 2.0
        self.marker_inset_y_mm = (sticker_height_m * 1000.0) / 2.0

    def table_to_robot_xyz(self, x_mm: float, y_mm: float) -> tuple[float, float, float]:
        """Convert table coordinates (mm) to robot base_link frame (meters)."""
        x_robot = self.robot_x_offset_m + (y_mm / 1000.0)
        center_x_mm = self.table_width_mm / 2.0
        y_robot = self.robot_y_offset_m + ((x_mm - center_x_mm) / 1000.0)
        z_robot = self.table_z_m + (self.cube_height_m / 2.0)
        return x_robot, y_robot, z_robot

    def get_zone_center_robot_xyz(self, zone_name: str) -> tuple[float, float, float]:
        """Get center of the specified zone in robot base_link coordinates."""
        if zone_name not in self.ZONE_MULTIPLIERS:
            raise ValueError(f"Unknown zone: '{zone_name}'")
        zone_width_mm = self.table_width_mm / 3.0
        x_mm = zone_width_mm * self.ZONE_MULTIPLIERS[zone_name]
        return self.table_to_robot_xyz(x_mm, self.table_depth_mm / 2.0)

    def create_pose_stamped(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        frame_id: str = 'base_link',
    ) -> PoseStamped:
        """Create a PoseStamped for MoveIt 2 planning."""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x_m
        pose.pose.position.y = y_m
        pose.pose.position.z = z_m
        pose.pose.orientation.w = 1.0
        return pose
