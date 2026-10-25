from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rviz_config_arg = LaunchConfiguration('rviz_config')
    include_environment = LaunchConfiguration('include_environment')

    default_rviz_config = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'config', 'moveit.rviz']
    )

    moveit_config = (
        MoveItConfigsBuilder('eezybotarm', package_name='eezybotarm_moveit_config')
        .robot_description(
            file_path='config/eezybotarm.urdf.xacro',
            mappings={'include_environment': include_environment},
        )
        .robot_description_semantic(file_path='config/eezybotarm.srdf')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        parameters=[moveit_config.to_dict()],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_arg],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz_config,
            description='Path to RViz configuration file',
        ),
        DeclareLaunchArgument(
            'include_environment',
            default_value='true',
            description='Whether to include the physical workspace environment in robot_description',
        ),
        move_group,
        rviz,
    ])
