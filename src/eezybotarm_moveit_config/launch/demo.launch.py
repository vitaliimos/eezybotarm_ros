from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'launch', 'bringup.launch.py']
    )
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'use_mock_hardware': 'true',
                'use_rviz': 'true',
            }.items(),
        )
    ])
