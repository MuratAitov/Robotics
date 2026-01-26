import os
import mujoco
import cv2
from ultralytics import YOLO

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

# Simulate longer so objects settle completely
for _ in range(2000):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=720, width=1280)

# Output folders for generated images
output_dir = "outputs"
frames_dir = os.path.join(output_dir, "frames")
yolo_dir = os.path.join(output_dir, "yolo")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(yolo_dir, exist_ok=True)

# Load fine-tuned model if available, otherwise fall back to base model
best_path = os.path.join("runs", "detect", "train2", "weights", "best.pt")
model_path = best_path if os.path.isfile(best_path) else "yolov8n.pt"
yolo = YOLO(model_path)

cameras = ["pedestals_cam", "scattered_cam", "top_cam", "overview_cam"]

for cam_name in cameras:
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render()

    # Save raw frame
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(frames_dir, f"frame_{cam_name}.png"), frame_bgr)

    # YOLO detection
    results = yolo.predict(source=rgb, verbose=False, conf=0.25)

    annotated = results[0].plot()
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(yolo_dir, f"yolo_{cam_name}.png"), bgr)

    # Print detections
    print(f"\n=== {cam_name} ===")
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo.names[cls_id]
            conf = float(box.conf[0])
            print(f"  {cls_name}: {conf:.1%}")
    else:
        print("  No detections")

print(f"\nModel: {model_path}")
print(f"Saved frames for all {len(cameras)} cameras to {output_dir}/")
