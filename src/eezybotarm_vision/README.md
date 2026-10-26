# eezybotarm_vision

Computer vision, zone calibration, and sorting coordination package for the EezyBotArm robotic manipulator.

## Overview
* **Object Detection**: Detects `green_cube` and `white_cube` objects with orientation angles using **NVIDIA TensorRT** (`.engine`) on Jetson or **ONNX Runtime** (`.onnx`) on workstations.
* **Zone Calibration**: Detects the 4 corner ring fiducial stickers on the table surface and computes a homography transformation to project 3 virtual zones (`left`, `middle`, `right`) in metric coordinates.
* **Sorting Coordinator**: Tracks cube stability in the middle zone and dispatches MoveIt pick-and-place action requests.

---

## Model Requirements & Export

The detector (`YoloCubeDetector`) expects an **NMS-free (End-to-End)** YOLO-OBB model (such as YOLO26m-obb):
* **Input Tensor**: `images` with shape `[1, 3, 640, 640]` (RGB, FP32 normalized to `[0, 1]`).
* **Output Tensor**: `output0` with shape `[1, N, 7]` (e.g. `[1, 300, 7]`) containing decoded detections:
  `[cx, cy, w, h, score, class_id, angle_rad]`.

> **Important**: When exporting the model from Ultralytics to ONNX, **`nms=False` must be explicitly set**:
> ```python
> model.export(
>     format="onnx",
>     imgsz=640,
>     dynamic=False,
>     simplify=True,
>     opset=17,
>     batch=1,
>     nms=False,  # Required: enables the NMS-free / End-to-End head [1, 300, 7]
> )
> ```
> If `nms=False` is omitted, Ultralytics exports the raw anchor grid `[1, 7, 8400]`, which results in invalid detection dimensions and scrambled class IDs.

---

## TensorRT Model Preparation

TensorRT engines are specific to the exact GPU architecture and runtime version.
Use the exported ONNX model (`models/model.onnx`) to compile an `.engine` plan for your GPU.

### On PC (Linux Workstation / Dev Container)
Compile with the Python helper script (uses the container's `tensorrt` library with FP16 enabled):
```bash
./docker-run.sh python3 scripts/build_trt_engine.py
```

### On Jetson (Target Embedded Device)
The Yocto image automatically handles this:
* `recipes-model/cube-detection-model` installs the pre-compiled `model.engine` if present locally in `files/model.engine`, or falls back to installing `model.onnx` into `/usr/share/models/`.
* On first boot, `eezybotarm-ensure-engine.sh` compiles `/usr/share/models/model.onnx` into `/usr/share/models/model.engine` if `model.engine` is not already present.

---

## Running and Testing

### Benchmark & Inference Test
```bash
# Test ONNX Runtime inference
./docker-run.sh python3 scripts/test_yolo_inference.py

# Test TensorRT (.engine) inference
./docker-run.sh python3 scripts/test_yolo_inference.py --trt
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
