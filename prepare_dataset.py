"""Convert the Kaggle helmet dataset (Pascal VOC XML) into YOLO format and split it into train / val / test."""
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

SRC_IMAGES = Path("data/kaggle_helmet/images")
SRC_XML = Path("data/kaggle_helmet/annotations")
OUT = Path("data/helmet_v1")
CLASSES = {"With Helmet": 0, "Without Helmet": 1}   # name in the XML -> YOLO class ID
NAMES = ["helmet", "no_helmet"]                     # what we call class 0 and 1 from now on
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42                                           # fixed seed = the same "random" split every run


def voc_to_yolo(xml_path):
    """Read one XML file -> (image file name, list of YOLO label lines 'class cx cy w h'), or None lines if corrupted."""
    root = ET.parse(xml_path).getroot()
    W = float(root.find("size/width").text)
    H = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        cls = CLASSES[obj.find("name").text]
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 > 1.05 * W or y2 > 1.05 * H or x2 <= x1 or y2 <= y1:
            return root.find("filename").text, None                        # corrupted box: skip the whole image
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)   # trim boxes that stick out slightly
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H                      # centre, as a fraction of the image size
        w, h = (x2 - x1) / W, (y2 - y1) / H                                # size, as a fraction of the image size
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return root.find("filename").text, lines


# 1. Convert every annotation
items = [voc_to_yolo(x) for x in sorted(SRC_XML.glob("*.xml"))]
corrupt = [name for name, lines in items if lines is None]
items = [(name, lines) for name, lines in items if lines is not None]
print(f"converted {len(items)} annotation files, skipped {len(corrupt)} with corrupted boxes: {corrupt}")

# 2. Shuffle once (with a fixed seed) and cut into train / val / test
random.Random(SEED).shuffle(items)
n_train = int(len(items) * SPLIT["train"])
n_val = int(len(items) * SPLIT["val"])
splits = {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}

# 3. Write the YOLO folder layout: images/<split>/x.png next to labels/<split>/x.txt
if OUT.exists():
    shutil.rmtree(OUT)                       # start clean every run
print(f"\n{'split':<6} {'images':>6} {'helmet':>7} {'no_helmet':>10}")
for split, split_items in splits.items():
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)
    counts = Counter()
    for filename, lines in split_items:
        src = SRC_IMAGES / filename
        dst = OUT / "images" / split / filename
        try:
            os.link(src, dst)                # a hard link: same file on disk, no extra space used
        except OSError:
            shutil.copy2(src, dst)           # fallback, e.g. on a different drive
        (OUT / "labels" / split / f"{Path(filename).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts.update(int(line.split()[0]) for line in lines)
    print(f"{split:<6} {len(split_items):>6} {counts[0]:>7} {counts[1]:>10}")

# 4. data.yaml tells the trainer where the images are and what the classes are called
(OUT / "data.yaml").write_text(
    "# helmet / no_helmet heads, converted from the Kaggle 'Helmet Detection' dataset\n"
    "train: images/train\nval: images/val\ntest: images/test\n"
    "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
)
print(f"\nwrote {OUT}/data.yaml")

# 5. Sanity check: draw the converted labels back onto a few training images
colors = [(0, 200, 0), (0, 0, 255)]          # helmet green, no_helmet red
tiles = []
for filename, _ in random.Random(1).sample(splits["train"], 8):
    img = cv2.imread(str(OUT / "images" / "train" / filename))
    H, W = img.shape[:2]
    for line in (OUT / "labels" / "train" / f"{Path(filename).stem}.txt").read_text().split("\n"):
        if not line:
            continue
        cls, cx, cy, w, h = line.split()
        cls, cx, cy, w, h = int(cls), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H   # fractions -> pixels
        cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), colors[cls], 2)
        cv2.putText(img, NAMES[cls], (int(cx - w / 2), int(cy - h / 2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)
    tiles.append(cv2.resize(img, (400, 300)))
grid = cv2.vconcat([cv2.hconcat(tiles[:4]), cv2.hconcat(tiles[4:])])
Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/labels_check.jpg", grid)
print("saved outputs/labels_check.jpg (open it: boxes should sit exactly on the heads)")
