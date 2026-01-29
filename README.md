# MuJoCo Sneaker Vision

Render sneaker models in MuJoCo, run YOLO detection, and estimate 6DoF pose with FoundationPose for robotic grasping.

## Features

- MuJoCo physics simulation with sneaker models
- YOLOv8 object detection (fine-tuned on footwear)
- **6DoF Pose Estimation** with NVIDIA FoundationPose
- Direction vector visualization for robot gripper orientation

## Folder structure

```
.
├── Sneakers/                 # 3D assets (OBJ + MTL + textures)
├── FoundationPose/           # 6DoF pose estimation
│   ├── run_sneaker.py        # Pose estimation script
│   ├── weights/              # Model weights (download separately)
│   └── demo_data/            # RGB/Depth/Mask data
├── outputs/                  # Generated images
│   ├── frames/               # Raw MuJoCo renders
│   ├── yolo/                 # YOLO detection overlays
│   └── pose_with_arrow/      # Pose visualization with direction arrows
├── run_viewer.py             # MuJoCo GUI viewer
├── run_yolo.py               # Render + YOLO inference
├── generate_pose_data.py     # Generate RGB+Depth+Mask for FoundationPose
├── render_with_arrow.py      # Visualize direction vectors in MuJoCo
├── world.xml                 # Main scene
├── world_pose_test.xml       # Pose estimation test scene
└── requirements.txt
```

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### 1. Interactive Viewer

```powershell
python run_viewer.py
```

### 2. YOLO Detection

```powershell
python run_yolo.py
```

### 3. 6DoF Pose Estimation

**Prerequisites:** Docker Desktop with GPU support

```powershell
# Download FoundationPose weights (~260MB)
cd FoundationPose
gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O weights/

# Pull Docker image (~12GB)
docker pull shingarey/foundationpose_custom_cuda121:latest
```

**Run pose estimation:**

```powershell
# Generate RGB + Depth + Mask from MuJoCo
python generate_pose_data.py world_pose_test.xml

# Run FoundationPose in Docker
docker run --rm --gpus all --env NVIDIA_DISABLE_REQUIRE=1 ^
  -v "%CD%\FoundationPose:/workspace/FoundationPose" ^
  --ipc=host shingarey/foundationpose_custom_cuda121:latest ^
  bash -c "cd /workspace/FoundationPose && python run_sneaker.py --debug 0"
```

**Visualize results:**

```powershell
python render_with_arrow.py
```

## Output Format

Pose estimation outputs `FoundationPose/sneaker_poses.json`:

```json
{
  "position": [x, y, z],
  "direction_vector": [dx, dy, dz],
  "rotation_matrix": [[...], ...],
  "pose_4x4": [[...], ...]
}
```

- **position** - 3D coordinates for robot approach
- **direction_vector** - Unit vector showing sneaker orientation (for gripper alignment)
- **rotation_matrix** - Full 3x3 rotation
- **pose_4x4** - SE(3) transformation matrix

## Tech Stack

- **Physics:** MuJoCo
- **Detection:** YOLOv8 (ultralytics)
- **Pose Estimation:** NVIDIA FoundationPose
- **GPU:** NVIDIA RTX (tested on 4070 SUPER)
