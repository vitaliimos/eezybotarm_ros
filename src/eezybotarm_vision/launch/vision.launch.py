from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('eezybotarm_vision')
    default_config_path = PathJoinSubstitution([pkg_share, 'config', 'vision_params.yaml'])
    default_env_config = PathJoinSubstitution([
        FindPackageShare('eezybotarm_description'), 'config', 'environment.yaml'
    ])
    default_mock_image = PathJoinSubstitution([pkg_share, 'resource', 'sample_workspace.jpg'])
    default_weights_path = '/usr/share/models/model.engine'

    config_file = LaunchConfiguration('config_file')
    environment_config = LaunchConfiguration('environment_config')
    start_camera = LaunchConfiguration('start_camera')
    use_mock_camera = LaunchConfiguration('use_mock_camera')
    mock_image_path = LaunchConfiguration('mock_image_path')
    weights_path = LaunchConfiguration('weights_path')
    gui = LaunchConfiguration('gui')
    use_rviz = LaunchConfiguration('use_rviz')

    # Condition: start_camera is true AND use_mock_camera is false
    run_real_camera = IfCondition(
        PythonExpression([
            "'", start_camera, "' == 'true' and '", use_mock_camera, "' == 'false'"
        ])
    )

    # Condition: start_camera is true AND use_mock_camera is true
    run_mock_camera = IfCondition(
        PythonExpression([
            "'", start_camera, "' == 'true' and '", use_mock_camera, "' == 'true'"
        ])
    )

    # 1. Real USB Camera node (v4l2_camera)
    real_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='camera',
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('image_raw/compressed', '/camera/image_raw/compressed'),
            ('image_raw/compressedDepth', '/camera/image_raw/compressedDepth'),
            ('image_raw/theora', '/camera/image_raw/theora'),
            ('image_raw/zstd', '/camera/image_raw/zstd'),
        ],
        condition=run_real_camera,
        output='screen',
    )

    # 2. Mock camera node (image_publisher)
    mock_camera_node = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='mock_camera',
        arguments=[mock_image_path],
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('image_raw/compressed', '/camera/image_raw/compressed'),
            ('image_raw/compressedDepth', '/camera/image_raw/compressedDepth'),
            ('image_raw/theora', '/camera/image_raw/theora'),
            ('image_raw/zstd', '/camera/image_raw/zstd'),
        ],
        condition=run_mock_camera,
        output='screen',
    )

    # 3. Sorting Coordinator node
    sorting_node = Node(
        package='eezybotarm_vision',
        executable='sorting_node',
        name='sorting_coordinator_node',
        parameters=[
            config_file,
            environment_config,
            {'weights_path': weights_path},
        ],
        output='screen',
    )

    # 4. GUI image viewer for annotated stream (rqt_image_view)
    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/eezybotarm_vision/debug_image'],
        condition=IfCondition(gui),
        output='screen',
    )

    # 5. Optional RViz2 for 3D marker visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_path,
            description='Path to vision parameters YAML file',
        ),
        DeclareLaunchArgument(
            'environment_config',
            default_value=default_env_config,
            description='Path to workspace environment parameters YAML file',
        ),
        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Whether to start any camera streaming node',
        ),
        DeclareLaunchArgument(
            'use_mock_camera',
            default_value='false',
            description='Whether to use image_publisher mock instead of physical camera',
        ),
        DeclareLaunchArgument(
            'mock_image_path',
            default_value=default_mock_image,
            description='Path to mock test image for image_publisher',
        ),
        DeclareLaunchArgument(
            'weights_path',
            default_value=default_weights_path,
            description='Path to TensorRT .engine model',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Launch rqt_image_view to preview the annotated debug image',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz for 3D marker visualization',
        ),
        real_camera_node,
        mock_camera_node,
        sorting_node,
        rqt_image_view_node,
        rviz_node,
    ])
