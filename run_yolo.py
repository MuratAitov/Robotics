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

# Save raw render for debugging
frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("frame.png", frame_bgr)

# Load YOLO model
yolo = YOLO("yolov8n.pt")
results = yolo.predict(source=rgb, verbose=False)

annotated = results[0].plot()
bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
cv2.imwrite("yolo_out.png", bgr)

print("Saved yolo_out.png")
