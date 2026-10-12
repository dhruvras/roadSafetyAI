"""Settings and helpers shared by the pipeline scripts."""
import torch

BIKE_MODEL = "models/yolo26n.pt"      # COCO-pretrained: finds + tracks vehicles (downloads itself if missing)
HEAD_MODEL = "models/helmet_v1.pt"    # ours: finds heads, class 0 = helmet, 1 = no_helmet
BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 1, 2, 3, 5, 7   # COCO class IDs
VEHICLES = [BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK]
YOLOE_MODEL = "models/yoloe-26s-seg.pt"  # zero-shot "helmet" finder: a second opinion on every head (see add_yoloe_helmets)
HELMET_CONF = 0.30                    # YOLOE helmet boxes we trust
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


def overlaps(a, b):
    """Same head? Either box's centre lies inside the other, or they overlap > 30%."""
    def centre_in(p, q):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        return q[0] <= cx <= q[2] and q[1] <= cy <= q[3]
    return centre_in(a, b) or centre_in(b, a) or iou(a, b) > 0.3


def load_yoloe():
    """YOLOE set up to look for 'helmet' only (the text is turned into numbers once, so no text model runs later)."""
    from ultralytics import YOLOE
    model = YOLOE(YOLOE_MODEL)
    model.set_classes(["helmet"], model.get_text_pe(["helmet"]))
    return model


def add_yoloe_helmets(heads, result, dx=0, dy=0):
    """Second opinion from YOLOE. A head our model calls 'no_helmet' becomes 'helmet' when YOLOE sees a helmet on it
    (our model's weak spot: dark helmets from behind), and helmets our model missed completely are added as new heads."""
    helmets = [(a + dx, b + dy, c + dx, d + dy) for (a, b, c, d), conf
               in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()) if conf >= HELMET_CONF]
    out = [(*h[:4], 0, max(h[5], HELMET_CONF) if h[4] == 0 else HELMET_CONF) if any(overlaps(h, b) for b in helmets) else h
           for h in heads]
    for b in helmets:                    # a helmet that sits on no known head = a rider our model missed
        if not any(overlaps(b, h) for h in out):
            out.append((*b, 0, HELMET_CONF))
    return out


def claim_heads(bikes, candidates, max_off=0.4, min_size=0.12):
    """Give every head to ONE bike. Crops of neighbouring bikes overlap, so the same head shows up in several of them
    and would inflate rider counts. A head goes to the bike whose centre it is closest to (sideways, in bike widths)
    and is dropped if it is further than `max_off` bike widths away or smaller than `min_size` of the bike's width.
    bikes: list of (x1, y1, x2, y2); candidates: heads found in each bike's crop -> heads kept for each bike."""
    mine = [[] for _ in bikes]
    seen = []                                                     # heads already handed out, to skip duplicates
    pool = sorted(((h, i) for i, hs in enumerate(candidates) for h in hs), key=lambda p: -p[0][5])
    for h, _ in pool:
        if any(overlaps(h, k) for k in seen):
            continue
        seen.append(h)
        cx = (h[0] + h[2]) / 2
        offs = [abs(cx - (b[0] + b[2]) / 2) / (b[2] - b[0]) for b in bikes]
        i = min(range(len(bikes)), key=offs.__getitem__)
        if offs[i] <= max_off and (h[2] - h[0]) >= min_size * (bikes[i][2] - bikes[i][0]):
            mine[i].append(h)
    return mine
