from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_weights_path = '/usr/share/models/model.engine'

    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    use_mock_camera = LaunchConfiguration('use_mock_camera')
    use_rviz = LaunchConfiguration('use_rviz')
    start_camera = LaunchConfiguration('start_camera')
    weights_path = LaunchConfiguration('weights_path')

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('eezybotarm_moveit_config'),
                'launch',
                'bringup.launch.py',
            ])
        ),
        launch_arguments={
            'use_mock_hardware': use_mock_hardware,
            'use_mock_camera': use_mock_camera,
            'start_camera': start_camera,
            'weights_path': weights_path,
            'use_vision': 'true',
            'use_rviz': use_rviz,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_hardware',
            default_value='true',
            description='Use mock hardware interfaces for testing without physical arm',
        ),
        DeclareLaunchArgument(
            'use_mock_camera',
            default_value='true',
            description='Use image_publisher mock camera instead of physical USB camera',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz for visualization and debugging',
        ),
        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Launch camera streamer node',
        ),
        DeclareLaunchArgument(
            'weights_path',
            default_value=default_weights_path,
            description='Path to YOLO weights file',
        ),
        bringup_launch,
    ])
