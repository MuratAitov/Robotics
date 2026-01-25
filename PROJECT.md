MuJoCo Vision Project — Step-by-Step Guide

This guide walks you through setting up a MuJoCo-based computer vision project where you:
	•	create a simple simulated world,
	•	place a sneaker (or any object) in the scene,
	•	render images from a camera,
	•	run YOLO object detection on the rendered images.

No robot models are used at this stage — the focus is purely on vision.

⸻

0. Project Structure

You can name the project folder however you like. Inside it, create a folder called mujoco_vision.

Recommended structure:

my_project/
├── mujoco_vision/
│   ├── world.xml
│   ├── sneaker.obj
│   ├── run_viewer.py
│   └── run_yolo.py
├── .venv/
└── README.md (optional)

Important rules:
	•	Do NOT use spaces in folder names
	•	Use only English letters (no Cyrillic)

⸻

1. Environment Setup

1.1 Create a virtual environment

From the project root:

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

1.2 Install dependencies

pip install mujoco ultralytics opencv-python


⸻

2. Create the MuJoCo World (world.xml)

Place this file inside mujoco_vision/.

<mujoco model="vision_world">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002"/>

  <asset>
    <mesh name="sneaker_mesh" file="sneaker.obj" scale="1 1 1"/>
  </asset>

  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>

    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>

    <body name="sneaker" pos="0.3 0 0.05">
      <freejoint/>
      <geom type="mesh" mesh="sneaker_mesh" rgba="1 1 1 1"/>
    </body>

    <camera name="cam0" pos="1.2 0 0.7"
            xyaxes="0 1 0 -0.5 0 0.866" fovy="55"/>
  </worldbody>
</mujoco>

Put your sneaker.obj file in the same folder.

⸻

3. Open the Scene in the MuJoCo GUI

Create run_viewer.py:

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

Run:

python run_viewer.py

You should see:
	•	a floor plane
	•	a sneaker object
	•	an interactive camera view

If the sneaker is not visible:
	•	adjust scale in <mesh>
	•	adjust pos of the sneaker body

⸻

4. Render a Camera Image

Create run_yolo.py (initial version):

import mujoco
import cv2

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

for _ in range(10):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="cam0")
rgb = renderer.render()

bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("frame.png", bgr)

print("Saved frame.png")

Run:

python run_yolo.py

Check that frame.png contains the sneaker.

⸻

5. Run YOLO Object Detection

Replace the contents of run_yolo.py with:

import mujoco
import cv2
from ultralytics import YOLO

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

for _ in range(10):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="cam0")
rgb = renderer.render()

# Load YOLO model
yolo = YOLO("yolov8n.pt")
results = yolo.predict(source=rgb, verbose=False)

annotated = results[0].plot()
bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
cv2.imwrite("yolo_out.png", bgr)

print("Saved yolo_out.png")

Run again:

python run_yolo.py

Open yolo_out.png to see detections.

⸻

6. Important Notes About YOLO
	•	Default YOLO models may NOT recognize sneakers
	•	This is normal
	•	For reliable detection you should:
	•	fine-tune YOLO on sneaker images, or
	•	generate a synthetic dataset using MuJoCo

MuJoCo is ideal for synthetic data because you can:
	•	randomize lighting
	•	randomize camera pose
	•	randomize object pose
	•	get perfect ground-truth labels

⸻

7. What Comes Next (Optional)

Possible next steps:
	•	Automatic dataset generation (YOLO format)
	•	Instance segmentation
	•	6D pose estimation of the sneaker
	•	Domain randomization for sim-to-real transfer

⸻

Summary

At this point you have:
	•	MuJoCo installed with GUI
	•	A custom world with an object
	•	Camera-based rendering
	•	YOLO inference on simulated images

This is a solid foundation for any vision-based robotics or CV pipeline.