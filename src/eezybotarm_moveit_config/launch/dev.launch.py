from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare('eezybotarm_moveit_config'), 'launch', 'bringup.launch.py']
    )
    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    i2c_bus = LaunchConfiguration('i2c_bus')
    i2c_address = LaunchConfiguration('i2c_address')
    pwm_frequency = LaunchConfiguration('pwm_frequency')
    use_rviz = LaunchConfiguration('use_rviz')

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
            'use_rviz',
            default_value='true',
            description='Launch RViz for visualization and debugging/planning',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'use_mock_hardware': use_mock_hardware,
                'i2c_bus': i2c_bus,
                'i2c_address': i2c_address,
                'pwm_frequency': pwm_frequency,
                'use_rviz': use_rviz,
            }.items(),
        ),
    ])
