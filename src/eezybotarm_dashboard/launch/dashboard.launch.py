from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')
    camera_raw_topic = LaunchConfiguration('camera_raw_topic')
    camera_debug_topic = LaunchConfiguration('camera_debug_topic')
    dataset_dir = LaunchConfiguration('dataset_dir')
    environment_config = LaunchConfiguration('environment_config')

    default_env_config = PathJoinSubstitution([
        FindPackageShare('eezybotarm_description'), 'config', 'environment.yaml'
    ])

    dashboard_node = Node(
        package='eezybotarm_dashboard',
        executable='dashboard_node',
        name='eezybotarm_dashboard',
        output='screen',
        parameters=[
            environment_config,
            {
                'host': host,
                'port': port,
                'camera_raw_topic': camera_raw_topic,
                'camera_debug_topic': camera_debug_topic,
                'dataset_dir': dataset_dir,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'environment_config',
            default_value=default_env_config,
            description='Path to workspace environment parameters YAML file',
        ),
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
        DeclareLaunchArgument(
            'dataset_dir',
            default_value='dataset',
            description='Directory where captured dataset images are stored',
        ),
        dashboard_node,
    ])
