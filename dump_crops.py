"""Cut rider crops out of videos (the same rider regions the pipeline looks at), pre-labelled by the current model.

    python dump_crops.py data/clips/*.mp4

Rebuilds data/v2_crops/ from scratch: images/*.jpg + labels/*.txt (YOLO format). Pass ALL clips at once.
Next: python auto_label.py   (or label by hand: python label_tool.py)
"""
import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import BIKE_MODEL, HEAD_MODEL, MOTORCYCLE, heads_from, pick_device, rider_region

OUT = Path("data/v2_crops")
BIKE_CONF = 0.35
PRELABEL_CONF = 0.30   # low on purpose: deleting a wrong box is quicker than drawing a missing one
STRIDE = 2             # look at every 2nd frame
EVERY = 8              # crop the same bike at most once every 8 analysed frames (about half a second)
MAX_PER_BIKE = 4       # ...and at most 4 times, so one slow bike can't fill the dataset with near-copies
MIN_BIKE_FRAC = 0.14   # skip bikes shorter than 14% of the frame height: their riders' heads are too small to label

if __name__ == "__main__":
    videos = [Path(p) for p in sys.argv[1:]]
    if not videos:
        sys.exit(__doc__)
    device = pick_device()
    bike_model, head_model = YOLO(BIKE_MODEL), YOLO(HEAD_MODEL)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bike_model.predictor = None                  # fresh tracker per video
        last_saved, n_saved = {}, {}                 # bike ID -> analysed-frame number of its last crop / crops so far
        frame_no = analysed = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if (frame_no - 1) % STRIDE:
                continue
            analysed += 1
            r = bike_model.track(frame, persist=True, classes=[MOTORCYCLE], conf=BIKE_CONF, device=device,
                                 tracker="bytetrack.yaml", verbose=False)[0]
            if r.boxes.id is None:
                continue
            for track_id, bike in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                too_small = bike[3] - bike[1] < MIN_BIKE_FRAC * H
                too_soon = analysed - last_saved.get(track_id, -EVERY) < EVERY
                if too_small or too_soon or n_saved.get(track_id, 0) >= MAX_PER_BIKE:
                    continue
                x1, y1, x2, y2 = rider_region(bike, W, H)
                crop = frame[y1:y2, x1:x2]
                ch, cw = crop.shape[:2]
                name = f"{video.stem}_bike{track_id}_f{frame_no}"
                cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), crop)
                heads = heads_from(head_model.predict(crop, conf=PRELABEL_CONF, device=device, verbose=False)[0])
                lines = [f"{c} {(a + c2) / 2 / cw:.6f} {(b + d) / 2 / ch:.6f} {(c2 - a) / cw:.6f} {(d - b) / ch:.6f}"
                         for a, b, c2, d, c, _ in heads]
                (OUT / "labels" / f"{name}.txt").write_text("".join(line + "\n" for line in lines))
                last_saved[track_id] = analysed
                n_saved[track_id] = n_saved.get(track_id, 0) + 1
                saved += 1
        cap.release()
        print(f"{video.name:<32} {saved:>4} crops from {len(n_saved)} bikes")

    (OUT / "classes.txt").write_text("helmet\nno_helmet\n")
    total = len(list((OUT / "images").glob("*.jpg")))
    boxes = sum(len(p.read_text().splitlines()) for p in (OUT / "labels").glob("*.txt"))
    print(f"\n{total} crops with {boxes} pre-labelled heads in {OUT}/")
