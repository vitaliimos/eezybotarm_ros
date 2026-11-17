from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare("eezybotarm_moveit_config"), "launch", "bringup.launch.py"]
    )

    pwm_chip = LaunchConfiguration("pwm_chip")
    pwm_frequency = LaunchConfiguration("pwm_frequency")
    use_vision = LaunchConfiguration("use_vision")
    camera = LaunchConfiguration("camera")
    weights_path = LaunchConfiguration("weights_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pwm_chip",
                default_value="/sys/bus/i2c/devices/7-0060/pwm",
                description="Path to PWM chip sysfs directory or parent pwm/ directory",
            ),
            DeclareLaunchArgument(
                "pwm_frequency",
                default_value="50",
                description="PWM frequency for servos in Hz",
            ),
            DeclareLaunchArgument(
                "use_vision",
                default_value="true",
                description="Launch vision and sorting coordinator nodes",
            ),
            DeclareLaunchArgument(
                "camera",
                default_value="real",
                choices=["real", "mock", "none"],
                description="Camera streaming mode: real (v4l2_camera), mock (image_publisher), none",
            ),
            DeclareLaunchArgument(
                "weights_path",
                default_value="/usr/share/models/model.engine",
                description="Path to YOLO TensorRT engine file",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bringup),
                launch_arguments={
                    "use_mock_hardware": "false",
                    "camera": camera,
                    "pwm_chip": pwm_chip,
                    "pwm_frequency": pwm_frequency,
                    "use_vision": use_vision,
                    "weights_path": weights_path,
                }.items(),
            ),
        ]
    )
