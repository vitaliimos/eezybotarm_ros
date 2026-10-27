from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    i2c_bus = LaunchConfiguration('i2c_bus')
    i2c_address = LaunchConfiguration('i2c_address')
    pwm_frequency = LaunchConfiguration('pwm_frequency')
    use_vision = LaunchConfiguration('use_vision')
    camera = LaunchConfiguration('camera')
    video_device = LaunchConfiguration('video_device')
    weights_path = LaunchConfiguration('weights_path')
    include_environment = LaunchConfiguration('include_environment')

    moveit_config = (
        MoveItConfigsBuilder('eezybotarm', package_name='eezybotarm_moveit_config')
        .robot_description(
            file_path='config/eezybotarm.urdf.xacro',
            mappings={
                'use_mock_hardware': use_mock_hardware,
                'i2c_bus': i2c_bus,
                'i2c_address': i2c_address,
                'pwm_frequency': pwm_frequency,
                'include_environment': include_environment,
            },
        )
        .robot_description_semantic(file_path='config/eezybotarm.srdf')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .moveit_cpp(file_path='config/moveit_cpp.yaml')
        .to_moveit_configs()
    )

    controller_config = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'config', 'ros2_controllers.yaml']
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[moveit_config.robot_description],
        output='screen',
    )
    static_world_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        output='screen',
    )
    ros2_control = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[moveit_config.robot_description, controller_config],
        output='screen',
    )
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )
    arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )
    hand_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['hand_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )
    # MoveItCpp (used by moveit_py) reads `planning_pipelines.pipeline_names`, while the
    # builder emits `planning_pipelines` as a plain list (the format move_group expects).
    moveit_py_params = dict(moveit_config.to_dict())
    moveit_py_params['planning_pipelines'] = {'pipeline_names': ['ompl']}
    pick_and_place_server = Node(
        package='eezybotarm_control',
        executable='pick_and_place_server',
        parameters=[moveit_py_params],
        output='screen',
    )

    start_arm_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[arm_controller],
        )
    )
    start_hand_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_controller,
            on_exit=[hand_controller],
        )
    )

    default_weights_path = '/usr/share/models/model.engine'
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('eezybotarm_vision'),
                'launch',
                'vision.launch.py',
            ])
        ),
        launch_arguments={
            'camera': camera,
            'video_device': video_device,
            'weights_path': weights_path,
        }.items(),
        condition=IfCondition(use_vision),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_hardware',
            default_value='false',
            description='Use mock hardware interfaces instead of real hardware',
        ),
        DeclareLaunchArgument(
            'i2c_bus',
            default_value='/dev/i2c-7',
            description='Path to the I2C bus device connected to PCA9685',
        ),
        DeclareLaunchArgument(
            'i2c_address',
            default_value='0x60',
            description='I2C slave address of the PCA9685 servo driver',
        ),
        DeclareLaunchArgument(
            'pwm_frequency',
            default_value='50',
            description='PWM frequency for servos in Hz',
        ),
        DeclareLaunchArgument(
            'use_vision',
            default_value='true',
            description='Launch vision and sorting coordinator nodes',
        ),
        DeclareLaunchArgument(
            'camera',
            default_value='real',
            choices=['real', 'mock', 'none'],
            description='Camera streaming mode: real (v4l2_camera), mock (image_publisher), none',
        ),
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='Path to the V4L2 capture device for the physical camera',
        ),
        DeclareLaunchArgument(
            'weights_path',
            default_value=default_weights_path,
            description='Path to YOLO weights file',
        ),
        DeclareLaunchArgument(
            'include_environment',
            default_value='true',
            description='Whether to include the physical workspace environment in robot_description',
        ),
        static_world_tf,
        robot_state_publisher,
        ros2_control,
        joint_state_broadcaster,
        start_arm_controller,
        start_hand_controller,
        pick_and_place_server,
        vision_launch,
    ])
