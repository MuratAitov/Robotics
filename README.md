# MuJoCo Sneaker Vision

Render sneaker models in MuJoCo and run YOLO detection on the rendered frames.

## What’s inside

- MuJoCo world with a sneaker mesh (`world.xml`)
- Viewer script for interactive inspection (`run_viewer.py`)
- Renderer + YOLO inference (`run_yolo.py`)
- Multiple sneaker assets in `Sneakers/`

## Folder structure

```
.
├── Sneakers/                 # 3D assets (OBJ + MTL + textures)
├── previews/                 # Preview renders of assets
├── run_viewer.py             # MuJoCo GUI viewer
├── run_yolo.py               # Render + YOLO inference
├── world.xml                 # Scene definition
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### macOS (Apple Silicon)

MuJoCo requires **arm64** Python on Apple Silicon.
If you see an error about x86_64/Rosetta, install arm64 Python and recreate the venv.

## Run

### Viewer (interactive)

- **macOS:**

```bash
mjpython run_viewer.py
```

- **Linux/Windows:**

```bash
python run_viewer.py
```

### Render + YOLO

```bash
python run_yolo.py
```

Outputs:
- `frame.png` – raw render
- `yolo_out.png` – YOLO overlay

## Changing the sneaker model

Edit the mesh path in `world.xml`, for example:

```xml
<mesh name="sneaker_mesh" file="Sneakers/sneaker/sneaker.obj" scale="1 1 1"/>
```

## Notes

- Default YOLO models are **not trained on sneakers**, so detections may be weak.
- For good results, fine-tune YOLO or generate a synthetic dataset in MuJoCo.
