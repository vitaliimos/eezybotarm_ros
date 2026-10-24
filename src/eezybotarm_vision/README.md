# eezybotarm_vision

Computer vision, zone calibration, and sorting coordination package for the EezyBotArm robotic manipulator.

## Overview
* **Object Detection**: Detects `green_cube` and `white_cube` objects with orientation angles using **NVIDIA TensorRT** (`.engine`) on Jetson or **ONNX Runtime** (`.onnx`) on workstations.
* **Zone Calibration**: Detects the 4 corner ring fiducial stickers on the table surface and computes a homography transformation to project 3 virtual zones (`left`, `middle`, `right`) in metric coordinates.
* **Sorting Coordinator**: Tracks cube stability in the middle zone and dispatches MoveIt pick-and-place action requests.

---

## TensorRT Model Preparation

TensorRT engines are specific to the exact GPU architecture and runtime version.
Use the exported ONNX model (`models/model.onnx`) to compile an `.engine` plan for your GPU.

### On PC (Linux Workstation / Dev Container)
Compile with the Python helper script:
```bash
python3 scripts/build_trt_engine.py \
    --onnx src/eezybotarm_vision/models/model.onnx \
    --output src/eezybotarm_vision/models/model.engine
```

Or using `trtexec`:
```bash
trtexec --onnx=src/eezybotarm_vision/models/model.onnx \
        --saveEngine=src/eezybotarm_vision/models/model.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640 \
        --fp16
```

### On Jetson (Target Embedded Device)
The Yocto image automatically handles this on first boot:
* `recipes-model/cube-detection-model` places `model.engine` into `/usr/share/models/model.engine`.
* `sorting_node` loads `/usr/share/models/model.engine` on startup.

---

## Running and Testing

### Benchmark & Inference Test
```bash
./docker-run.sh python3 scripts/test_yolo_inference.py
```

### Launch Vision Node (Mock Camera)
```bash
./docker-run.sh ros2 launch eezybotarm_vision vision.launch.py \
    weights_path:=/workspaces/src/eezybotarm_vision/models/model.onnx \
    use_mock_camera:=true gui:=true
```

### Launch Complete Local Stack (Mock Hardware + Mock Camera + RViz + Dashboard)
```bash
./docker-run.sh ros2 launch eezybotarm_moveit_config local.launch.py
```
