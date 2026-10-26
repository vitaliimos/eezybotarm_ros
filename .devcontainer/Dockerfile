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
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/pip \
    && printf "[global]\nbreak-system-packages = true\n\n[install]\nignore-installed = true\n" > /etc/pip.conf \
    && printf "numpy<2\nsetuptools<70\n" > /etc/pip/constraints.txt
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_IGNORE_INSTALLED=1 \
    PIP_CONSTRAINT=/etc/pip/constraints.txt

# Configure custom rosdep rules and install dependencies for workspace packages
COPY rosdep.yaml /etc/ros/rosdep/custom_rules/eeazybot.yaml
RUN echo "yaml file:///etc/ros/rosdep/custom_rules/eeazybot.yaml" > /etc/ros/rosdep/sources.list.d/50-eeazybot.list

COPY src/eezybotarm_control/package.xml /tmp/manifests/eezybotarm_control/package.xml
COPY src/eezybotarm_dashboard/package.xml /tmp/manifests/eezybotarm_dashboard/package.xml
COPY src/eezybotarm_description/package.xml /tmp/manifests/eezybotarm_description/package.xml
COPY src/eezybotarm_hardware/package.xml /tmp/manifests/eezybotarm_hardware/package.xml
COPY src/eezybotarm_interfaces/package.xml /tmp/manifests/eezybotarm_interfaces/package.xml
COPY src/eezybotarm_moveit_config/package.xml /tmp/manifests/eezybotarm_moveit_config/package.xml
COPY src/eezybotarm_vision/package.xml /tmp/manifests/eezybotarm_vision/package.xml
RUN rosdep init || true \
    && rosdep update \
    && apt-get update \
    && rosdep install --from-paths /tmp/manifests --ignore-src -y \
    && rm -rf /tmp/manifests \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed "setuptools<70" "numpy<2"

RUN mkdir -p /eezybot_ros \
    && chown -R ${USERNAME}:${USERNAME} /eezybot_ros

USER ${USERNAME}
WORKDIR /eezybot_ros

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/${USERNAME}/.bashrc

RUN colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
    && colcon mixin update default || true
