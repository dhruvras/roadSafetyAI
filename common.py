"""Settings and helpers shared by the pipeline scripts."""
import torch

BIKE_MODEL = "models/yolo26n.pt"      # COCO-pretrained: finds + tracks vehicles (downloads itself if missing)
HEAD_MODEL = "models/helmet_v1.pt"    # ours: finds heads, class 0 = helmet, 1 = no_helmet
BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 1, 2, 3, 5, 7   # COCO class IDs
VEHICLES = [BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK]
ALPHA_W, ALPHA_H = 0.05, 1.0          # rider region = bike box, 5% wider each side, extended up by its own height


def pick_device():
    """Fastest hardware available: Apple GPU (mps) > NVIDIA GPU (cuda) > CPU."""
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def rider_region(bike, W, H):
    """Grow a bike box sideways and upwards so it covers the riders' heads (clipped to the frame)."""
    x1, y1, x2, y2 = bike
    w, h = x2 - x1, y2 - y1
    return max(0, int(x1 - ALPHA_W * w)), max(0, int(y1 - ALPHA_H * h)), min(W, int(x2 + ALPHA_W * w)), min(H, int(y2))


def iou(a, b):
    """Intersection over Union of two (x1, y1, x2, y2, ...) boxes: 0 = no overlap, 1 = identical."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def heads_from(result, dx=0, dy=0):
    """Head-model result -> [(x1, y1, x2, y2, cls, conf)], shifted by (dx, dy), with double boxes merged:
    the model sometimes puts a helmet AND a no-helmet box on one head, so keep the more confident one."""
    heads = sorted(((a + dx, b + dy, c + dx, d + dy, int(cls), conf) for (a, b, c, d), cls, conf
                    in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())),
                   key=lambda h: -h[5])
    kept = []
    for h in heads:
        if all(iou(h, k) <= 0.5 for k in kept):
            kept.append(h)
    return kept
