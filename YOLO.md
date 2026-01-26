# YOLO

This project fine-tunes YOLOv8 for a single class: **Footwear**.

## Current model

- Fine-tuned weights: `runs/detect/train2/weights/best.pt`
- Inference script: `run_yolo.py` (auto-picks best.pt if present)

## Inference

```powershell
yolo detect predict model=runs/detect/train2/weights/best.pt source=outputs/frames/frame_pedestals_cam.png conf=0.25
```

## Training (YOLOv8s)

```powershell
yolo detect train data=data/open_images_footwear/dataset.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=16 device=0
```

## Switching to YOLOv8m (more accurate, slower)

```powershell
yolo detect train data=data/open_images_footwear/dataset.yaml model=yolov8m.pt epochs=50 imgsz=640 batch=16 device=0
```

## Notes

- The Open Images class name is **Footwear** (not Shoe).
- `data/` and `outputs/` are ignored by git.
- `.pt` weights are ignored by git; upload `best.pt` to cloud to use on another machine.

## Improving results

1) Increase dataset size (more train/val samples).
2) Increase image size: `imgsz=960` or `imgsz=1024`.
3) Use a larger model: `yolov8m` or `yolov8l`.
4) Reduce false positives: raise `conf` to 0.4-0.5.
5) Reduce domain gap: render more MuJoCo images and fine-tune on them.
