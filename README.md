# MuJoCo Sneaker Vision

Render sneaker models in MuJoCo and run YOLO detection on rendered frames.

## What's inside

- MuJoCo world with sneaker meshes (`world.xml`)
- Viewer script for interactive inspection (`run_viewer.py`)
- Renderer + YOLO inference (`run_yolo.py`)
- Sneaker assets in `Sneakers/`
- YOLO training notes in `YOLO.md`

## Folder structure

```
.
+-- Sneakers/                 # 3D assets (OBJ + MTL + textures)
+-- outputs/                  # Generated images (frames + detections)
+-- previews/                 # Preview renders of assets
+-- run_viewer.py             # MuJoCo GUI viewer
+-- run_yolo.py               # Render + YOLO inference
+-- world.xml                 # Scene definition
+-- requirements.txt
+-- YOLO.md                   # Training, inference, and improvements
```

## Setup (Windows)

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

### Viewer (interactive)

```powershell
python run_viewer.py
```

### Render + YOLO

```powershell
python run_yolo.py
```

Outputs:
- `outputs/frames/` - raw renders
- `outputs/yolo/` - YOLO overlays

## Changing sneaker models

Edit the mesh paths in `world.xml`, for example:

```xml
<mesh name="sneaker_mesh" file="Sneakers/sneaker/sneaker.obj" scale="1 1 1"/>
```

## Notes

- Default YOLO models are not trained for sneakers, so detections may be weak.
- Fine-tune YOLO on footwear or render synthetic data for best results.
