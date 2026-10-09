"""A tiny labelling tool for the rider crops in data/v2_crops/.

    python label_tool.py            # go through every crop
    python label_tool.py --review   # only the crops auto_label.py was unsure about (skip.txt)

  drag on empty space    draw a new head box (in the current class)
  click on a box         switch it: helmet <-> no_helmet
  right-click on a box   delete it (or hover over it and press e)
  1 / 2                  current class for new boxes: helmet / no_helmet
  d or space             save + next image          a    save + previous image
  x                      delete every box in this image (no usable heads)
  q                      save + quit (you continue where you left off next time)
"""
import sys
from pathlib import Path

import cv2

D = Path("data/v2_crops")
NAMES = ["helmet", "no_helmet"]
COLORS = [(0, 200, 0), (0, 0, 255)]          # BGR: helmet green, no_helmet red
VIEW_H, MAX_W = 820, 1400                    # crops are small: show them enlarged
WINDOW = "v2 labeller"

state = {"boxes": [], "cls": 1, "drag": None, "mouse": (0, 0), "scale": 1.0, "W": 1, "H": 1}


def load(img_path):
    """Read the YOLO label file into a list of [cls, x1, y1, x2, y2] in crop pixels."""
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    boxes = []
    label = D / "labels" / f"{img_path.stem}.txt"
    if label.exists():
        for line in label.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            boxes.append([int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    state.update(boxes=boxes, W=W, H=H, scale=min(VIEW_H / H, MAX_W / W), drag=None)
    return img


def save(img_path):
    """Write the boxes back as YOLO lines: class centre_x centre_y width height (fractions of the crop)."""
    W, H = state["W"], state["H"]
    lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
             for c, x1, y1, x2, y2 in state["boxes"]]
    (D / "labels" / f"{img_path.stem}.txt").write_text("".join(line + "\n" for line in lines))


def box_at(x, y):
    """Index of the smallest box under the point (crop pixels), or None."""
    hits = [(abs(b[3] - b[1]) * abs(b[4] - b[2]), i) for i, b in enumerate(state["boxes"])
            if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
    return min(hits)[1] if hits else None


def on_mouse(event, x, y, flags, param):
    s = state["scale"]
    x, y = x / s, y / s                      # window pixels -> crop pixels
    state["mouse"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drag"] = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
        state["drag"] = (*state["drag"][:2], x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
        x0, y0, _, _ = state["drag"]
        state["drag"] = None
        if abs(x - x0) * s < 6 and abs(y - y0) * s < 6:          # barely moved: a click, not a drag
            i = box_at(x, y)
            if i is not None:
                state["boxes"][i][0] = 1 - state["boxes"][i][0]
        else:
            x1, x2 = sorted((max(0, x0), min(state["W"], x)))
            y1, y2 = sorted((max(0, y0), min(state["H"], y)))
            state["boxes"].append([state["cls"], x1, y1, x2, y2])
    elif event == cv2.EVENT_RBUTTONDOWN:
        delete_at(x, y)


def delete_at(x, y):
    i = box_at(x, y)
    if i is not None:
        state["boxes"].pop(i)


def render(img, idx, total, name):
    s = state["scale"]
    view = cv2.resize(img, (int(state["W"] * s), int(state["H"] * s)), interpolation=cv2.INTER_CUBIC)
    for c, x1, y1, x2, y2 in state["boxes"]:
        cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), COLORS[c], 2)
        cv2.putText(view, NAMES[c], (int(x1 * s), max(14, int(y1 * s) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1)
    if state["drag"]:
        x0, y0, x1, y1 = state["drag"]
        cv2.rectangle(view, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), COLORS[state["cls"]], 1)
    bar = cv2.copyMakeBorder(view, 0, 48, 0, max(0, 560 - view.shape[1]), cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(bar, f"{idx + 1}/{total}  {name[:40]}", (8, bar.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(bar, f"new boxes: {NAMES[state['cls']]}   [1/2 class, d next, a back, x clear, q quit]",
                (8, bar.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[state["cls"]], 1)
    return bar


def main():
    images = sorted((D / "images").glob("*.jpg"))
    done_file = D / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    if "--review" in sys.argv:
        skip = {line.split("\t")[0] for line in (D / "skip.txt").read_text().splitlines()}
        images = [p for p in images if p.stem in skip]
    idx = next((i for i, p in enumerate(images) if p.stem not in done), len(images) - 1)
    print(f"{len(done)} of {len(images)} already done, starting at {idx + 1}")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    img = load(images[idx])
    while True:
        cv2.imshow(WINDOW, render(img, idx, len(images), images[idx].stem))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("d"), ord(" "), ord("a"), ord("q")):
            save(images[idx])
            done.add(images[idx].stem)
            done_file.write_text("\n".join(sorted(done)) + "\n")
            if key == ord("q"):
                break
            if key == ord("a"):
                idx = max(0, idx - 1)
            elif idx == len(images) - 1:
                print("that was the last image - all done!")
                break
            else:
                idx += 1
            img = load(images[idx])
        elif key == ord("1"):
            state["cls"] = 0
        elif key == ord("2"):
            state["cls"] = 1
        elif key == ord("x"):
            state["boxes"] = []
        elif key == ord("e"):
            delete_at(*state["mouse"])
    cv2.destroyAllWindows()
    print(f"{len(done)} of {len(images)} images labelled")


if __name__ == "__main__":
    main()
