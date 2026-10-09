from collections import defaultdict

import cv2
import torch
from ultralytics import YOLO

VIDEO_IN = "data/clips/practice.mp4"
VIDEO_OUT = "outputs/step3.mp4"
MOTORCYCLE = 3
CONF = 0.35
# 10 bright BGR colours; each track ID always gets the same one
COLORS = [(0, 200, 0), (255, 120, 0), (0, 140, 255), (255, 0, 200), (0, 230, 230),
          (200, 0, 255), (255, 255, 0), (0, 80, 255), (120, 255, 120), (255, 160, 160)]

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("models/yolo26n.pt")

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

tracks = {}                  # track ID -> {"first": frame no, "last": frame no, "frames": how many frames seen}
trails = defaultdict(list)   # track ID -> list of (x, y) points, to draw the path each bike travelled

frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    # 1. Detect + track in one call. persist=True keeps the tracker's memory between calls,
    #    so a bike seen in the previous frame keeps its ID in this one.
    result = model.track(frame, persist=True, classes=[MOTORCYCLE], conf=CONF, device=device,
                         tracker="bytetrack.yaml", verbose=False)[0]

    # 2. result.boxes.id holds the track IDs. It is None when nothing is being tracked yet.
    if result.boxes.id is not None:
        ids = result.boxes.id.int().tolist()
        boxes = result.boxes.xyxy.tolist()

        for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
            # 3. Remember this bike: when we first/last saw it and for how many frames
            if track_id not in tracks:
                tracks[track_id] = {"first": frame_no, "last": frame_no, "frames": 0}
            tracks[track_id]["last"] = frame_no
            tracks[track_id]["frames"] += 1
            trails[track_id].append((int((x1 + x2) / 2), int(y2)))   # bottom-centre = where it touches the road

            # 4. Draw its box, its ID, and the path it travelled in the last 2 seconds
            color = COLORS[track_id % len(COLORS)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, f"#{track_id}", (int(x1) + 8, int(y1) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            path = trails[track_id][-60:]
            for a, b in zip(path, path[1:]):
                cv2.line(frame, a, b, color, 4)

    cv2.putText(frame, f"frame {frame_no}/{total}   bikes seen so far: {len(tracks)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    writer.write(frame)
    preview = cv2.resize(frame, (960, int(960 * height / width)))
    cv2.imshow("step 3 - press q to quit", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# 5. Summary: one line per bike
print(f"{len(tracks)} different bikes (track IDs) in {frame_no} frames\n")
print(f"{'ID':>4}  {'first seen':>10}  {'last seen':>9}  {'frames':>6}")
for track_id, t in sorted(tracks.items()):
    print(f"{track_id:>4}  {t['first'] / fps:>9.1f}s  {t['last'] / fps:>8.1f}s  {t['frames']:>6}")
short = [tid for tid, t in tracks.items() if t["frames"] < 15]
print(f"\n{len(short)} of these IDs lasted under half a second (15 frames): flicker or ID switches")
print(f"saved {VIDEO_OUT}")
