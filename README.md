# EezyBotArm ROS 2 & MoveIt 2 Control Stack

This repository contains the complete ROS 2 Jazzy and MoveIt 2 software stack for the **EezyBotArm** (Mk2 3-DOF robotic arm + 1-DOF parallel gripper). It features an integrated `ros2_control` hardware interface for driving servos via a PCA9685 I2C PWM driver, MoveIt 2 trajectory planning, and a MoveItPy-powered Pick-and-Place action server.

---

## Repository Structure

```text
├── .devcontainer/              # VS Code Docker development container configuration
├── move.sh                     # Sample test script to send PickObject action goal
├── src/
│   ├── eezybotarm_description/ # URDF, Xacro kinematics, and STL visual/collision meshes
│   ├── eezybotarm_moveit_config/# MoveIt 2 configuration (SRDF, kinematics, controllers, launch)
│   ├── eezybotarm_control/     # High-level MoveItPy Python Pick-and-Place action server
│   ├── eezybotarm_hardware/    # C++ ros2_control hardware interface for PCA9685 over I2C
│   └── eezybotarm_interfaces/  # Custom action definitions (PickObject.action)
└── README.md
```

---

## Packages Overview

* **`eezybotarm_description`**: Contains the URDF and 3D visual/collision meshes.
* **`eezybotarm_hardware`**: Implements a `hardware_interface::SystemInterface` plugin for Linux I2C (`/dev/i2c-*`), communicating directly with a PCA9685 PWM servo driver without external middleware.
* **`eezybotarm_moveit_config`**: Contains SRDF semantic descriptions, OMPL planner configurations, joint limits, trajectory controllers, and launch orchestrations.
* **`eezybotarm_control`**: Runs the `/pick_and_place` action server implementing Cartesian planning and gripper sequencing using `moveit_py`.
* **`eezybotarm_interfaces`**: Action and message definitions (`PickObject.action`).

---

## Getting Started

### Option 1: Using VS Code Dev Container (Recommended)

The repository provides a complete, pre-configured development environment using Docker and Dev Containers:

1. Install [Docker](https://docs.docker.com/get-docker/) and [Visual Studio Code](https://code.visualstudio.com/).
2. Install the **Dev Containers** extension in VS Code.
3. Open this folder in VS Code.
4. When prompted, click **"Reopen in Container"** (or press `Ctrl+Shift+P` / `Cmd+Shift+P` and choose `Dev Containers: Reopen in Container`).
5. All system libraries, ROS 2 Jazzy packages, MoveIt 2, and compilers will be installed automatically.

### Option 2: Native Local Setup (Ubuntu 24.04 + ROS 2 Jazzy)

If developing natively on Ubuntu 24.04 with ROS 2 Jazzy:

1. **Install ROS 2 Jazzy Base & Development Tools**:
   ```bash
   sudo apt update
   sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
   ```

2. **Initialize rosdep and register local project rules** (if not already done):
   ```bash
   test -f /etc/ros/rosdep/sources.list.d/20-default.list || sudo rosdep init
   echo "yaml file://$(pwd)/rosdep.yaml" | sudo tee /etc/ros/rosdep/sources.list.d/50-eeazybot.list
   rosdep update
   ```

3. **Install Package Dependencies**:
   From the repository root:
   ```bash
   PIP_BREAK_SYSTEM_PACKAGES=1 rosdep install --from-paths src --ignore-src -r -y
   ```
   *(Note: `PIP_BREAK_SYSTEM_PACKAGES=1` is required on Ubuntu 24.04 / Python 3.12+ because rosdep installs pip-based dependencies globally).*

4. **Hardware Access Setup (for physical I2C)**:
   Add your user to the `i2c` group to access the hardware interface without root privileges:
   ```bash
   sudo apt install -y i2c-tools libi2c-dev
   sudo usermod -aG i2c $USER
   ```
   *(Log out and back in for group changes to take effect).*

---

## Building the Workspace

Build all packages using `colcon`:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

After the build completes, source the workspace overlay:

```bash
source install/setup.bash
```

> **Tip:** Add this line to your `~/.bashrc` to source it automatically in every new terminal:
> ```bash
> echo "source $(pwd)/install/setup.bash" >> ~/.bashrc
> ```

---

## Running the Robot

### 1. Local Development & Simulation (`local.launch.py`)

Use this mode for local development and testing without physical hardware connected. By default, it launches mock joint hardware, the mock camera (`image_publisher`), MoveIt 2 planning pipelines, the pick-and-place action server, the vision sorting node (using the workspace's `model.onnx`), RViz GUI, and the NiceGUI web dashboard (port `8080`):

```bash
./docker-run.sh ros2 launch eezybotarm_moveit_config local.launch.py
```

Available launch arguments:
* `use_mock_hardware` (default: `true`): Use mock hardware interfaces.
* `camera` (default: `mock`): Camera streaming mode (`mock`, `real`, `none`).
* `use_rviz` (default: `true`): Launch RViz for visualization and planning.
* `use_dashboard` (default: `true`): Launch the NiceGUI web dashboard on port `8080`.
* `weights_path` (default: `<eezybotarm_vision>/models/model.onnx`): Path to YOLO weights file.

### 2. Target Hardware on NVIDIA Jetson (`jetson.launch.py`)

Use this mode on the actual robot hardware (auto-started at boot via `eezybotarm.service` on Yocto images). It runs headless with the real `ros2_control` PCA9685 I2C driver and physical USB camera:

```bash
ros2 launch eezybotarm_moveit_config jetson.launch.py profile:=prod
```

Available launch arguments:
* `profile` (default: `prod`, choices: `prod`, `dev`): Selects `/usr/share/models/model.engine` (TensorRT) for `prod` or `/usr/share/models/model.onnx` (ONNX Runtime) for `dev`.
* `i2c_bus` (default: `/dev/i2c-7`): Path to the Linux I2C device.
* `i2c_address` (default: `0x60`): I2C hex address of the PCA9685 driver board.
* `pwm_frequency` (default: `50`): Servo PWM frequency in Hz.
* `weights_path` (default: determined by `profile`): Explicit override for YOLO model path.

---

## Testing Motion with the Action Client

Once any of the launch files above are running, open a new terminal (with `install/setup.bash` sourced) and trigger a pick-and-place sequence using the helper script:

```bash
./move.sh
```

Or invoke the action directly via the ROS 2 CLI:

```bash
ros2 action send_goal /pick_and_place eezybotarm_interfaces/action/PickObject \
  "{object_class: 'cup', target_pose: {header: {frame_id: 'base_link'}, pose: {position: {x: 0.33, y: 0.0, z: 0.100}, orientation: {w: 1.0}}}}" \
  --feedback
```

---

## MoveIt Setup Assistant

To reconfigure joint limits, collision matrices, or planning groups:

```bash
ros2 launch eezybotarm_moveit_config setup_assistant.launch.py
```
Select **"Edit Existing MoveIt Configuration Package"** and point to `src/eezybotarm_moveit_config`.

---

## Computer Vision & TensorRT Acceleration

The vision package `eezybotarm_vision` detects white and green cubes and estimates their 2D orientation angles using YOLO accelerated with **NVIDIA TensorRT**.

### 1. Generating the TensorRT Engine on PC

Because TensorRT `.engine` plans are architecture-specific, the engine must be compiled on the machine with the target GPU from the shared ONNX file (`src/eezybotarm_vision/models/model.onnx`).

#### Method A: Using the Python Helper Script
```bash
python3 scripts/build_trt_engine.py \
    --onnx src/eezybotarm_vision/models/model.onnx \
    --output src/eezybotarm_vision/models/model.engine \
    --fp16
```

#### Method B: Using `trtexec` CLI
```bash
trtexec --onnx=src/eezybotarm_vision/models/model.onnx \
        --saveEngine=src/eezybotarm_vision/models/model.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640 \
        --fp16
```

### 2. Testing Inference on PC

Verify model detection, bounding boxes, and orientation angle on sample images:

```bash
# Test inference and save annotated image to /tmp/yolo_test_result.jpg
./docker-run.sh python3 scripts/test_yolo_inference.py --weights src/eezybotarm_vision/models/model.engine
```

Run the complete ROS 2 vision launch pipeline with mock camera:
```bash
./docker-run.sh ros2 launch eezybotarm_vision vision.launch.py \
    weights_path:=/workspaces/src/eezybotarm_vision/models/model.engine \
    camera:=mock gui:=false
```

### 3. Deployment on Target Hardware (NVIDIA Jetson)

When deployed via Yocto (`eeazybot_yocto`), the `cube-detection-model` recipe installs the pre-compiled Jetson TensorRT engine (`model.engine`, tracked via Git LFS in `eeazybot_yocto`) directly into `/usr/share/models/model.engine`. To compile or update `model.engine` from a trained `.pt` file using a Jetson running `eezybot-image-dev` over SSH, run `scripts/prepare_yolo_model.py` in `eeazybot_yocto`.

---

## License & Attribution

- **Source Code**: Licensed under the [MIT License](LICENSE).
- **3D Model Meshes**: The STL mesh models in `src/eezybotarm_description/meshes/` are from the **EEZYbotARM Mk2** open-source project by Carlo Franciscone (daGHIZmo) and are licensed under [Creative Commons Attribution - Non-Commercial 3.0 Unported (CC BY-NC 3.0)](https://creativecommons.org/licenses/by-nc/3.0/).
  - [Instructables Project](https://www.instructables.com/EEZYbotARM-Mk2-3D-Printed-Robot/)
  - [Thingiverse (thing:1454048)](https://www.thingiverse.com/thing:1454048)
  - [Onshape CAD Model](https://cad.onshape.com/documents/eb6eac15e3f68718c52d3cac/w/2b8e59d1698d09b06440ab07/e/c11caf264884cee4298c28f7)

