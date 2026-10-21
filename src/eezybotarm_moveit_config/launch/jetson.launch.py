from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'launch', 'bringup.launch.py']
    )

    profile = LaunchConfiguration('profile')
    i2c_bus = LaunchConfiguration('i2c_bus')
    i2c_address = LaunchConfiguration('i2c_address')
    pwm_frequency = LaunchConfiguration('pwm_frequency')
    use_vision = LaunchConfiguration('use_vision')
    start_camera = LaunchConfiguration('start_camera')
    weights_path = LaunchConfiguration('weights_path')

    default_weights_path = PythonExpression([
        "'/usr/share/models/model.onnx' if '",
        profile,
        "' == 'dev' else '/usr/share/models/model.engine'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'profile',
            default_value='prod',
            choices=['prod', 'dev'],
            description='Target profile on Jetson (prod uses TensorRT .engine, dev uses .onnx)',
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
            'start_camera',
            default_value='true',
            description='Whether to start physical USB camera streaming node',
        ),
        DeclareLaunchArgument(
            'weights_path',
            default_value=default_weights_path,
            description='Path to YOLO weights file (.engine for prod, .onnx for dev)',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'use_mock_hardware': 'false',
                'use_mock_camera': 'false',
                'use_rviz': 'false',
                'i2c_bus': i2c_bus,
                'i2c_address': i2c_address,
                'pwm_frequency': pwm_frequency,
                'use_vision': use_vision,
                'start_camera': start_camera,
                'weights_path': weights_path,
            }.items(),
        ),
    ])
