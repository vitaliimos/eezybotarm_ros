FROM ros:jazzy-ros-base-noble

ARG USERNAME=ubuntu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        git \
        i2c-tools \
        libi2c-dev \
        nano \
        python3-colcon-common-extensions \
        python3-pip \
        python3-rosdep \
        python3-vcstool \
        python3-venv \
        vim \
        wget \
        ros-${ROS_DISTRO}-controller-manager \
        ros-${ROS_DISTRO}-geometric-shapes \
        ros-${ROS_DISTRO}-hardware-interface \
        ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-joint-state-publisher-gui \
        ros-${ROS_DISTRO}-launch-param-builder \
        ros-${ROS_DISTRO}-moveit \
        ros-${ROS_DISTRO}-moveit-configs-utils \
        ros-${ROS_DISTRO}-moveit-planners-ompl \
        ros-${ROS_DISTRO}-moveit-py \
        ros-${ROS_DISTRO}-moveit-ros-move-group \
        ros-${ROS_DISTRO}-moveit-ros-visualization \
        ros-${ROS_DISTRO}-moveit-setup-assistant \
        ros-${ROS_DISTRO}-object-recognition-msgs \
        ros-${ROS_DISTRO}-octomap-msgs \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-ros2-control \
        ros-${ROS_DISTRO}-ros2-controllers \
        ros-${ROS_DISTRO}-rqt \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-srdfdom \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-warehouse-ros \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-v4l2-camera \
        ros-${ROS_DISTRO}-image-transport-plugins \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed "setuptools<70" "numpy<2" onnxruntime tensorrt nvidia-cuda-runtime-cu12

RUN mkdir -p /eezybot_ros \
    && chown -R ${USERNAME}:${USERNAME} /eezybot_ros

USER ${USERNAME}
WORKDIR /eezybot_ros

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/${USERNAME}/.bashrc

RUN colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
    && colcon mixin update default || true
