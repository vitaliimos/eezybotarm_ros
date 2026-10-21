# eezybotarm_vision

Computer vision, zone detection, and sorting coordination package for the EezyBotArm robotic manipulator.

## Overview
* **Object Detection**: Detects `green_cube` and `white_cube` objects with orientation angles using **NVIDIA TensorRT** (`.engine`).
* **Zone Calibration**: Uses homography transformation to map image pixel space to metric coordinates on the workspace paper sheet (`left`, `middle`, `right` zones).
* **Sorting Coordinator**: Coordinates vision detection and triggers MoveIt pick-and-place action requests.

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
* `sorting_node` auto-discovers `/usr/share/models/model.engine` on startup.

---

## Running and Testing

### Benchmark & Inference Test
```bash
./docker-run.sh python3 scripts/test_yolo_inference.py --weights src/eezybotarm_vision/models/model.engine
```

### Launch Vision Node (Mock Camera)
```bash
./docker-run.sh ros2 launch eezybotarm_vision vision.launch.py \
    weights_path:=/workspaces/src/eezybotarm_vision/models/model.engine \
    use_mock_camera:=true gui:=true
```

### Launch Complete Local Stack (Mock Hardware + Mock Camera + RViz + Dashboard)
```bash
./docker-run.sh ros2 launch eezybotarm_moveit_config local.launch.py
```
