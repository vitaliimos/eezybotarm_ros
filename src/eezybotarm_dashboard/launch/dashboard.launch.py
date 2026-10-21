import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')
    camera_raw_topic = LaunchConfiguration('camera_raw_topic')
    camera_debug_topic = LaunchConfiguration('camera_debug_topic')

    # Find environment.yaml in description package if available
    try:
        desc_share = get_package_share_directory('eezybotarm_description')
        env_yaml = os.path.join(desc_share, 'config', 'environment.yaml')
        parameters = [env_yaml]
    except Exception:
        parameters = []

    parameters.append({
        'host': host,
        'port': port,
        'camera_raw_topic': camera_raw_topic,
        'camera_debug_topic': camera_debug_topic,
    })

    dashboard_node = Node(
        package='eezybotarm_dashboard',
        executable='dashboard_node',
        name='eezybotarm_dashboard',
        output='screen',
        parameters=parameters,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'host',
            default_value='0.0.0.0',
            description='Host address to bind the web server',
        ),
        DeclareLaunchArgument(
            'port',
            default_value='8080',
            description='Port for the web server',
        ),
        DeclareLaunchArgument(
            'camera_raw_topic',
            default_value='/camera/image_raw/compressed',
            description='Topic name for raw camera stream',
        ),
        DeclareLaunchArgument(
            'camera_debug_topic',
            default_value='/eezybotarm_vision/debug_image/compressed',
            description='Topic name for annotated debug image stream',
        ),
        dashboard_node,
    ])
