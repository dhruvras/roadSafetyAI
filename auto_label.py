"""Auto-label the rider crops in data/v2_crops/ without drawing boxes by hand (knowledge distillation).

    python auto_label.py

Three models vote on every crop:
  YOLOE (big zero-shot model, text prompt "helmet")  finds helmets from any angle, including the back
  helmet_v1 (our model)                              finds heads, and is good at bare heads seen from the front
  yolo26n (COCO "person")                            checks that every visible rider got a head label
Crops where they agree get labels. Crops with any doubt go to skip.txt (fix them later with: python label_tool.py --review)
"""
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE

from common import BIKE_MODEL, HEAD_MODEL, heads_from, iou, pick_device

D = Path("data/v2_crops")
YOLOE_MODEL = "models/yoloe-26s-seg.pt"
HELMET_CONF = 0.30         # YOLOE helmet boxes we trust
WEAK_HELMET = 0.15         # YOLOE helmet boxes below HELMET_CONF but above this = doubt
V1_CONF = 0.30             # v1 head boxes we consider at all
BARE_CONF = 0.50           # v1 "no_helmet" must be at least this sure (and not contradicted by YOLOE)
PERSON_CONF = 0.50
HELMET, NO_HELMET = 0, 1


def overlaps(a, b):
    """Same head? Either box's centre lies inside the other, or they overlap > 30%."""
    def centre_in(p, q):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        return q[0] <= cx <= q[2] and q[1] <= cy <= q[3]
    return centre_in(a, b) or centre_in(b, a) or iou(a, b) > 0.3


def label_crop(img, yoloe, v1, persons_model, device):
    """-> (list of (cls, x1, y1, x2, y2), reason it's doubtful or None)"""
    H, W = img.shape[:2]
    ye = yoloe.predict(img, conf=WEAK_HELMET, device=device, verbose=False)[0]
    helmets = [(b, c) for b, c in zip(ye.boxes.xyxy.tolist(), ye.boxes.conf.tolist())]
    strong = [b for b, c in helmets if c >= HELMET_CONF]
    weak = [b for b, c in helmets if c < HELMET_CONF]
    heads = heads_from(v1.predict(img, conf=V1_CONF, device=device, verbose=False)[0])

    labels, used = [], set()
    for x1, y1, x2, y2, cls, conf in heads:
        head = (x1, y1, x2, y2)
        match = next((i for i, b in enumerate(strong) if overlaps(head, b)), None)
        if match is not None:                              # YOLOE sees a helmet here: it's a helmet
            labels.append((HELMET, *strong[match]))
            used.add(match)
        elif any(overlaps(head, b) for b in weak):
            return [], "weak helmet"                       # maybe a helmet, maybe not
        elif cls == NO_HELMET and conf >= BARE_CONF:       # v1 is sure it's bare and YOLOE sees no helmet
            labels.append((NO_HELMET, *head))
        else:
            return [], "v1 unsure"                         # v1 says helmet but YOLOE can't find one, or v1 weak
    for i, b in enumerate(strong):                         # helmets v1 missed (mostly rear views): v2's main lesson
        if i not in used:
            labels.append((HELMET, *b))

    # every clearly visible person in the crop must have a head label, or the crop is incomplete
    pr = persons_model.predict(img, classes=[0], conf=PERSON_CONF, device=device, verbose=False)[0]
    for px1, py1, px2, py2 in pr.boxes.xyxy.tolist():
        ph = py2 - py1
        if ph < 0.25 * H or py1 < 0.02 * H:               # small, or head cut off by the crop edge
            continue
        top = (px1, py1 - 0.1 * ph, px2, py1 + 0.35 * ph)
        if not any(top[0] <= (l[1] + l[3]) / 2 <= top[2] and top[1] <= (l[2] + l[4]) / 2 <= top[3] for l in labels):
            return [], "person without head label"
    return labels, None


def main():
    device = pick_device()
    yoloe = YOLOE(YOLOE_MODEL)
    yoloe.set_classes(["helmet"], yoloe.get_text_pe(["helmet"]))
    v1, persons_model = YOLO(HEAD_MODEL), YOLO(BIKE_MODEL)

    skipped, counts = [], {"helmet": 0, "no_helmet": 0, "empty": 0}
    for p in sorted((D / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        labels, doubt = label_crop(img, yoloe, v1, persons_model, device)
        if doubt:
            skipped.append(f"{p.stem}\t{doubt}")
            continue
        lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                 for c, x1, y1, x2, y2 in labels]
        (D / "labels" / f"{p.stem}.txt").write_text("".join(line + "\n" for line in lines))
        counts["empty"] += not labels
        for c, *_ in labels:
            counts["helmet" if c == HELMET else "no_helmet"] += 1

    (D / "skip.txt").write_text("".join(s + "\n" for s in skipped))
    total = len(list((D / "images").glob("*.jpg")))
    print(f"auto-labelled {total - len(skipped)} of {total} crops: {counts['helmet']} helmet heads, "
          f"{counts['no_helmet']} bare heads, {counts['empty']} crops with no heads")
    print(f"{len(skipped)} doubtful crops listed in {D}/skip.txt (not used for training unless you fix them)")


if __name__ == "__main__":
    main()
