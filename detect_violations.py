"""The full system: for every motorcycle, count heads, check helmets, vote over frames, save evidence.

    python detect_violations.py data/clips/*.mp4
    python detect_violations.py data/clips/practice.mp4 --model models/helmet_v2.pt --no-show

Results for each video go to outputs/<video name>/: annotated.mp4, violations.csv, evidence/*.jpg
"""
import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

from common import BIKE_MODEL, HEAD_MODEL, MOTORCYCLE, VEHICLES, heads_from, pick_device, rider_region

BIKE_CONF, HEAD_CONF = 0.35, 0.40
NH_CONF = 0.50             # a bare head only counts towards "no helmet" when the model is at least this sure
STRIDE = 2                 # analyse every 2nd frame: twice as fast, and 12-15 looks per second is plenty
BETA = 0.10                # a head's centre may stick out up to 10% beyond the bike's sides
N_MIN = 8                  # only judge a bike after seeing it in 8 analysed frames
K_TR = 3                   # triple riding: 3+ heads in at least 3 frames
RHO, K_NH = 0.4, 3         # no helmet: in >= 40% of the frames where its heads were visible, and in >= 3 frames
MIN_MOVE = 0.5             # a bike that never moves half its own width is treated as parked and not judged
BIKE_SHARE = 0.6           # a track must be called "motorcycle" in >= 60% of its frames (not bicycle/car/truck)
LOST_AFTER = 30            # analysed frames without seeing a track before we close it

TRIPLE, NO_HELMET = "triple_riding", "no_helmet"
LABELS = {TRIPLE: "TRIPLE RIDING", NO_HELMET: "NO HELMET"}
GREEN, RED, WHITE, GREY = (0, 200, 0), (0, 0, 255), (255, 255, 255), (150, 150, 150)


# ---------------------------------------------------------------- per-track memory and the voting rules
def new_track(track_id, pos, t):
    return {"id": track_id, "classes": Counter(), "seen": 0, "frames": 0, "head_frames": 0, "triple_frames": 0,
            "nohelmet_frames": 0, "counts": Counter(), "width_sum": 0.0, "max_move": 0.0, "best": {},
            "first_pos": pos, "last_seen": 0}


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def is_motorcycle(s):
    """Fix 1: bicycles and e-rickshaws get called 'motorcycle' sometimes, but not most of the time."""
    return s["classes"][MOTORCYCLE] >= BIKE_SHARE * s["seen"]


def moved(s):
    """Parked bikes don't travel at least half their own width."""
    return s["max_move"] >= MIN_MOVE * (s["width_sum"] / s["seen"])


def decide(s):
    """The verdict for one bike from all its frames: a set containing TRIPLE and/or NO_HELMET, or empty."""
    if s["frames"] < N_MIN or not is_motorcycle(s) or not moved(s):
        return set()
    verdict = set()
    if s["triple_frames"] >= K_TR:
        verdict.add(TRIPLE)
    if s["head_frames"] and s["nohelmet_frames"] >= K_NH and s["nohelmet_frames"] / s["head_frames"] >= RHO:
        verdict.add(NO_HELMET)
    return verdict


def typical_riders(s):
    with_heads = {n: c for n, c in s["counts"].items() if n > 0}
    return max(with_heads, key=with_heads.get) if with_heads else 0


# ---------------------------------------------------------------- one video
def analyse(video, bike_model, head_model, device, show=True):
    out_dir = Path("outputs") / video.stem
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "evidence").glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open {video}, skipping")
        return []
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw = out_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / STRIDE, (W, H))
    bike_model.predictor = None               # fresh tracker for every video, so IDs start again at 1

    live, finished = {}, []
    frame_no = analysed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if (frame_no - 1) % STRIDE:
            continue
        analysed += 1
        t = (frame_no - 1) / fps

        # 1. track every vehicle type, so we can tell when a "motorcycle" is really a bicycle or an e-rickshaw
        r = bike_model.track(frame, persist=True, classes=VEHICLES, conf=BIKE_CONF, device=device,
                             tracker="bytetrack.yaml", verbose=False)[0]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
        boxes, classes = r.boxes.xyxy.tolist()[:len(ids)], r.boxes.cls.int().tolist()[:len(ids)]
        for track_id, box, cls in zip(ids, boxes, classes):
            pos = ((box[0] + box[2]) / 2, box[3])                  # bottom-centre: where it touches the road
            s = live.setdefault(track_id, new_track(track_id, pos, t))
            s["classes"][cls] += 1
            s["seen"] += 1
            s["width_sum"] += box[2] - box[0]
            s["max_move"] = max(s["max_move"], dist(s["first_pos"], pos))
            s["last_seen"] = analysed

        # 2. cut out the rider region of each motorcycle and find heads in all crops at once
        bikes = [(tid, box) for tid, box, cls in zip(ids, boxes, classes) if cls == MOTORCYCLE]
        regions = [rider_region(box, W, H) for _, box in bikes]
        crops = [frame[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in regions]
        head_results = head_model.predict(crops, conf=HEAD_CONF, device=device, verbose=False) if crops else []

        for (track_id, bike), (rx, ry, _, _), crop, hr in zip(bikes, regions, crops, head_results):
            # 3. heads in frame coordinates, only those above this bike
            margin = BETA * (bike[2] - bike[0])
            heads = [h for h in heads_from(hr, rx, ry) if bike[0] - margin <= (h[0] + h[2]) / 2 <= bike[2] + margin]
            n = len(heads)
            bare = [h for h in heads if h[4] == 1 and h[5] >= NH_CONF]       # fix 3: only confident bare heads

            # 4. update this bike's memory
            s = live[track_id]
            s["frames"] += 1
            s["counts"][n] += 1
            s["head_frames"] += n >= 1
            s["triple_frames"] += n >= 3
            s["nohelmet_frames"] += len(bare) >= 1

            # 5. keep the most convincing frame per violation as evidence (crop with its heads drawn on)
            candidates = [(TRIPLE, sum(h[5] for h in heads))] if n >= 3 else []
            if bare:
                candidates.append((NO_HELMET, max(h[5] for h in bare)))
            for v, score in candidates:
                if score > s["best"].get(v, (-1,))[0]:
                    for hx1, hy1, hx2, hy2, cls, _ in heads:
                        cv2.rectangle(crop, (int(hx1 - rx), int(hy1 - ry)), (int(hx2 - rx), int(hy2 - ry)),
                                      GREEN if cls == 0 else RED, 3)
                    s["best"][v] = (score, t, crop)

            # 6. draw: grey = parked or not really a motorcycle (not judged), red = violation so far, white = OK so far
            verdict = decide(s)
            judged = s["frames"] < N_MIN or (is_motorcycle(s) and moved(s))
            color = GREY if not judged else RED if verdict else WHITE
            note = "" if judged else " not judged"
            x1, y1, x2, y2 = map(int, bike)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"#{track_id} riders:{n}{note}" + "".join(f" {LABELS[v]}" for v in sorted(verdict)),
                        (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for hx1, hy1, hx2, hy2, cls, _ in heads:
                cv2.rectangle(frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), GREEN if cls == 0 else RED, 3)

        writer.write(frame)
        if show:
            cv2.imshow("violations - press q to skip this video", cv2.resize(frame, (960, int(960 * H / W))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # 7. tracks not seen for a while have left the scene
        for track_id in [k for k, s in live.items() if analysed - s["last_seen"] > LOST_AFTER]:
            finished.append(live.pop(track_id))

    finished += live.values()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 8. browsers can't play OpenCV's mp4v, so re-encode to H.264 (the dashboard needs this)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / "annotated.mp4")], check=True)
    raw.unlink()

    # 9. one verdict per bike
    bikes_final = [s for s in finished if s["frames"] >= N_MIN]
    records = []
    for s in bikes_final:
        for v in sorted(decide(s)):
            score, t, crop = s["best"][v]
            path = out_dir / "evidence" / f"bike{s['id']}_{v}.jpg"
            cv2.imwrite(str(path), crop)
            records.append({"video": video.name, "bike": s["id"], "violation": v,
                            "time": f"{int(t // 60):02d}:{t % 60:04.1f}", "time_s": round(t, 1),
                            "riders": typical_riders(s), "max_heads": max(s["counts"]), "confidence": round(score, 2),
                            "frames_seen": s["frames"], "evidence": str(path)})
    records.sort(key=lambda r: r["time_s"])
    with open(out_dir / "violations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "bike", "violation", "time", "time_s", "riders", "max_heads",
                                          "confidence", "frames_seen", "evidence"])
        w.writeheader()
        w.writerows(records)

    not_bikes = sum(not is_motorcycle(s) for s in bikes_final)
    parked = sum(is_motorcycle(s) and not moved(s) for s in bikes_final)
    print(f"\n{video.name}: {len(bikes_final)} vehicles seen long enough | ignored: {not_bikes} not motorcycles, "
          f"{parked} parked | {len(records)} violations")
    for r in records:
        print(f"   {r['time']}  bike #{r['bike']:<4} {LABELS[r['violation']]:<14} riders: {r['riders']}  conf: {r['confidence']}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--model", default=HEAD_MODEL, help="head model to use (default: %(default)s)")
    ap.add_argument("--no-show", action="store_true", help="don't open a preview window")
    args = ap.parse_args()
    device = pick_device()
    bike_model, head_model = YOLO(BIKE_MODEL), YOLO(args.model)
    everything = []
    for video in args.videos:
        everything += analyse(video, bike_model, head_model, device, show=not args.no_show)
    print(f"\nTOTAL: {len(everything)} violations in {len(args.videos)} video(s)")
