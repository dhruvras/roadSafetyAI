import time

import cv2
import torch
from ultralytics import YOLO

VIDEO_IN = "data/clips/practice.mp4"
VIDEO_OUT = "outputs/step2.mp4"
PERSON, MOTORCYCLE = 0, 3   # class IDs in the COCO dataset the model was trained on

# Use the fastest hardware available: Apple GPU (mps) > NVIDIA GPU (cuda) > CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("running on:", device)

# 1. Load the pretrained model (the .pt file = the model's learned weights)
model = YOLO("models/yolo26n.pt")
print(f"model knows {len(model.names)} classes, e.g. {[model.names[i] for i in (0, 1, 2, 3, 5, 7)]}")

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_no = 0
start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    # 2. Ask the model what is in this frame
    result = model(frame, conf=0.35, device=device, verbose=False)[0]

    # 3. Read its answer: one box, class and confidence per detected object
    boxes = result.boxes.xyxy.tolist()       # [[x1, y1, x2, y2], ...] in pixels
    classes = result.boxes.cls.tolist()      # [3.0, 0.0, ...] class IDs
    confs = result.boxes.conf.tolist()       # [0.91, 0.47, ...] how sure it is (0-1)

    if frame_no == 1:
        print("\nframe 1 detections:")
        for box, cls, conf in zip(boxes, classes, confs):
            print(f"  {model.names[int(cls)]:<12} conf={conf:.2f}  box={[int(v) for v in box]}")

    # 4. Draw: motorcycles green, people orange, everything else grey
    bikes = 0
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        if cls == MOTORCYCLE:
            color = (0, 200, 0)
            bikes += 1
        elif cls == PERSON:
            color = (0, 140, 255)
        else:
            color = (170, 170, 170)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"frame {frame_no}/{total}   motorcycles: {bikes}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    writer.write(frame)
    preview = cv2.resize(frame, (960, int(960 * height / width)))
    cv2.imshow("step 2 - press q to quit", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

elapsed = time.time() - start
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"\nprocessed {frame_no} frames in {elapsed:.1f} s = {frame_no / elapsed:.1f} frames per second")
print(f"saved {VIDEO_OUT}")
