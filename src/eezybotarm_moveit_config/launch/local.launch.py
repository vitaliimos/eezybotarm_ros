from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'launch', 'bringup.launch.py']
    )
    dashboard_launch_file = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_dashboard'), 'launch', 'dashboard.launch.py']
    )
    default_weights_path = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_vision'), 'models', 'model.onnx']
    )

    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    i2c_bus = LaunchConfiguration('i2c_bus')
    i2c_address = LaunchConfiguration('i2c_address')
    pwm_frequency = LaunchConfiguration('pwm_frequency')
    use_rviz = LaunchConfiguration('use_rviz')
    use_vision = LaunchConfiguration('use_vision')
    use_mock_camera = LaunchConfiguration('use_mock_camera')
    start_camera = LaunchConfiguration('start_camera')
    weights_path = LaunchConfiguration('weights_path')
    start_dashboard = LaunchConfiguration('start_dashboard')

    dashboard_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(dashboard_launch_file),
        condition=IfCondition(start_dashboard),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_hardware',
            default_value='true',
            description='Use mock hardware interfaces for local development',
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
            'use_rviz',
            default_value='true',
            description='Launch RViz for visualization and planning',
        ),
        DeclareLaunchArgument(
            'use_vision',
            default_value='true',
            description='Launch vision and sorting coordinator nodes',
        ),
        DeclareLaunchArgument(
            'use_mock_camera',
            default_value='true',
            description='Use image_publisher mock camera instead of physical USB camera',
        ),
        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Whether to start camera streaming node in vision',
        ),
        DeclareLaunchArgument(
            'weights_path',
            default_value=default_weights_path,
            description='Path to YOLO ONNX weights file in eezybotarm_vision',
        ),
        DeclareLaunchArgument(
            'start_dashboard',
            default_value='true',
            description='Whether to start the NiceGUI web dashboard alongside local stack',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'use_mock_hardware': use_mock_hardware,
                'i2c_bus': i2c_bus,
                'i2c_address': i2c_address,
                'pwm_frequency': pwm_frequency,
                'use_rviz': use_rviz,
                'use_vision': use_vision,
                'use_mock_camera': use_mock_camera,
                'start_camera': start_camera,
                'weights_path': weights_path,
            }.items(),
        ),
        dashboard_launch,
    ])
