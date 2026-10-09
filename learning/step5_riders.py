import csv
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VIDEO_IN = "data/clips/practice.mp4"
VIDEO_OUT = "outputs/step5.mp4"
CSV_OUT = "outputs/step5_riders.csv"
CROPS_DIR = Path("outputs/step5_crops")

PERSON, MOTORCYCLE = 0, 3
BIKE_CONF, PERSON_CONF = 0.35, 0.35
ALPHA_W, ALPHA_H = 0.05, 1.0   # rider region = bike box, 5% wider on each side and extended up by 100% of its height
N_MIN = 8                      # only judge bikes we have seen in at least 8 frames
K_TR = 3                       # triple riding = 3+ riders seen in at least 3 frames
GREEN, RED, ORANGE, GREY = (0, 200, 0), (0, 0, 255), (0, 140, 255), (180, 180, 180)


def rider_region(bike, W, H):
    """Grow the bike box sideways and upwards so it covers the riders sitting on it (clipped to the frame)."""
    x1, y1, x2, y2 = bike
    w, h = x2 - x1, y2 - y1
    return max(0, int(x1 - ALPHA_W * w)), max(0, int(y1 - ALPHA_H * h)), min(W, int(x2 + ALPHA_W * w)), min(H, int(y2))


def on_bike(person, bike):
    """Is this person riding this bike, or a pedestrian / a rider on the next bike?"""
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike
    bike_h = by2 - by1
    overlap = max(0, min(px2, bx2) - max(px1, bx1))              # horizontal overlap with the bike, in pixels
    over_bike = overlap >= 0.6 * (px2 - px1)                     # most of the person is above the bike, not beside it
    feet_on_bike = by1 + 0.3 * bike_h <= py2 <= by2 + 0.1 * bike_h   # their lowest point is on the bike's lower half
    same_scale = (py2 - py1) >= 0.7 * bike_h                     # people further back look smaller than the bike's riders
    return over_bike and feet_on_bike and same_scale


def iou(a, b):
    """Intersection over Union: how much two boxes overlap (0 = not at all, 1 = identical)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def remove_duplicates(people):
    """The detector sometimes puts two boxes on one person: keep only one of any pair overlapping > 50%."""
    kept = []
    for p in people:
        if all(iou(p, k) <= 0.5 for k in kept):
            kept.append(p)
    return kept


def decide(stats):
    """One verdict per bike from all its frames: (typical rider count, is it triple riding?)."""
    if stats["frames"] < N_MIN:
        return None, False                                       # seen too briefly to judge
    with_riders = {n: c for n, c in stats["counts"].items() if n > 0}
    typical = max(with_riders, key=with_riders.get) if with_riders else 0
    frames_3plus = sum(c for n, c in stats["counts"].items() if n >= 3)
    return typical, frames_3plus >= K_TR


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
bike_model = YOLO("models/yolo26n.pt")    # finds + tracks motorcycles in the full frame
rider_model = YOLO("models/yolo26n.pt")   # finds people inside each bike's crop (later: your helmet model)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}")
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
CROPS_DIR.mkdir(parents=True, exist_ok=True)

bikes = {}   # track ID -> {"frames": n, "counts": Counter({riders: frames}), "best": most riders seen so far}

frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    # 1. Find and track the motorcycles (exactly Step 3)
    result = bike_model.track(frame, persist=True, classes=[MOTORCYCLE], conf=BIKE_CONF, device=device,
                              tracker="bytetrack.yaml", verbose=False)[0]
    ids = result.boxes.id.int().tolist() if result.boxes.id is not None else []
    bike_boxes = result.boxes.xyxy.tolist()[:len(ids)]

    # 2. Cut out the rider region above each bike (.copy() so later drawing doesn't change the crops)
    regions = [rider_region(b, W, H) for b in bike_boxes]
    crops = [frame[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in regions]

    # 3. Look for people inside all the crops in one batch
    person_results = rider_model.predict(crops, classes=[PERSON], conf=PERSON_CONF, device=device,
                                         verbose=False) if crops else []

    for track_id, bike, (rx1, ry1, rx2, ry2), crop, pr in zip(ids, bike_boxes, regions, crops, person_results):
        # 4. Convert each person from crop coordinates back to frame coordinates, keep only the riders
        riders = []
        for a, b, c, d in pr.boxes.xyxy.tolist():
            person = (a + rx1, b + ry1, c + rx1, d + ry1)
            if on_bike(person, bike):
                riders.append(person)
        riders = remove_duplicates(riders)
        n = len(riders)

        # 5. Remember how many riders this bike had in this frame
        stats = bikes.setdefault(track_id, {"frames": 0, "counts": Counter(), "best": -1})
        stats["frames"] += 1
        stats["counts"][n] += 1
        if n > stats["best"]:                                     # keep a picture of its fullest moment
            stats["best"] = n
            for x1, y1, x2, y2 in riders:
                cv2.rectangle(crop, (int(x1 - rx1), int(y1 - ry1)), (int(x2 - rx1), int(y2 - ry1)), ORANGE, 3)
            cv2.imwrite(str(CROPS_DIR / f"bike_{track_id}.jpg"), crop)

        # 6. Draw: rider region (grey), riders (orange), bike (green, or red once it counts as triple riding)
        _, triple = decide(stats)
        color = RED if triple else GREEN
        x1, y1, x2, y2 = map(int, bike)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), GREY, 1)
        for px1, py1, px2, py2 in riders:
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), ORANGE, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        label = f"#{track_id} riders:{n}" + ("  TRIPLE" if triple else "")
        cv2.putText(frame, label, (x1 + 8, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"frame {frame_no}/{total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    writer.write(frame)
    preview = cv2.resize(frame, (960, int(960 * H / W)))
    cv2.imshow("step 5 - press q to quit", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# 7. One verdict per bike, printed and saved as CSV
rows = []
for track_id, stats in sorted(bikes.items()):
    typical, triple = decide(stats)
    if typical is None:
        continue
    rows.append({"track_id": track_id, "frames_seen": stats["frames"], "typical_riders": typical,
                 "max_riders": stats["best"], "frames_with_3plus": sum(c for n, c in stats["counts"].items() if n >= 3),
                 "triple_riding": "YES" if triple else "no", "rider_counts": dict(sorted(stats["counts"].items()))})

print(f"{len(bikes)} track IDs, {len(rows)} seen long enough to judge (>= {N_MIN} frames)\n")
print(f"{'bike':>5} {'frames':>6} {'typical':>7} {'max':>4} {'3+ frames':>9}  triple   riders per frame -> number of frames")
for r in rows:
    print(f"{'#' + str(r['track_id']):>5} {r['frames_seen']:>6} {r['typical_riders']:>7} {r['max_riders']:>4} "
          f"{r['frames_with_3plus']:>9}  {r['triple_riding']:<6}   {r['rider_counts']}")

with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["track_id"])
    w.writeheader()
    w.writerows(rows)
print(f"\nsaved {VIDEO_OUT}, {CSV_OUT} and one crop per bike in {CROPS_DIR}/")
