"""Launch configuration to visualize EezyBotArm gripper reachable workspace in RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build launch description for reachable workspace visualization in RViz."""
    pkg_debug = FindPackageShare('eezybotarm_debug')
    pkg_description = FindPackageShare('eezybotarm_description')
    pkg_moveit_config = FindPackageShare('eezybotarm_moveit_config')

    default_env_config = PathJoinSubstitution([pkg_description, 'config', 'environment.yaml'])
    default_rviz_config = PathJoinSubstitution([pkg_debug, 'config', 'workspace.rviz'])
    urdf_xacro_path = PathJoinSubstitution(
        [pkg_moveit_config, 'config', 'eezybotarm.urdf.xacro']
    )

    environment_config = LaunchConfiguration('environment_config')
    rviz_config = LaunchConfiguration('rviz_config')
    step_deg = LaunchConfiguration('step_deg')

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        urdf_xacro_path,
        ' include_environment:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
    )

    static_world_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        output='screen',
    )

    workspace_visualizer = Node(
        package='eezybotarm_debug',
        executable='workspace_visualizer',
        parameters=[
            environment_config,
            robot_description,
            {'step_deg': step_deg},
        ],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'environment_config',
            default_value=default_env_config,
            description='Path to workspace geometry parameters YAML file (SSOT)',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz_config,
            description='Path to RViz configuration file',
        ),
        DeclareLaunchArgument(
            'step_deg',
            default_value='4.0',
            description='Angular sampling step in degrees for reachability grid',
        ),
        robot_state_publisher,
        joint_state_publisher,
        static_world_tf,
        workspace_visualizer,
        rviz_node,
    ])
