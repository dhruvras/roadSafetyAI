"""Fine-tune YOLO26n to find heads and tell helmet from no_helmet.

    python train.py data/helmet_v1/data.yaml helmet_v1          # full run (up to 100 epochs)
    python train.py data/helmet_v1/data.yaml helmet_v1 3        # quick test: 3 epochs

Results: runs/detect/<name>/ (charts) and models/<name>.pt (the trained model)
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from common import BIKE_MODEL, pick_device

# Windows (Muskan's laptop) starts helper processes that re-run this file; the guard stops them re-starting training.
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    data, name = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    device = pick_device()
    if device == "cuda":
        device = 0

    # 1. Start from the COCO-trained weights: the model already knows edges, shapes, faces and people,
    #    so it only has to learn "helmet vs no helmet" (= transfer learning / fine-tuning).
    model = YOLO(BIKE_MODEL)

    # 2. Train. Ultralytics handles the training loop, augmentation, validation after every epoch and saving.
    model.train(
        data=data,
        epochs=epochs,       # max number of full passes over the training images
        patience=20,         # stop early if val accuracy hasn't improved for 20 epochs
        imgsz=640,           # every image is resized to 640 px, the same size we use when detecting
        batch=16,            # images per step; use 8 on a 4 GB GPU
        device=device,
        name=name,           # results go to runs/detect/<name>/
        exist_ok=True,       # re-running overwrites that folder instead of making <name>2, <name>3...
        seed=0,
    )

    # 3. Final exam: score the best checkpoint on the test images it has never seen
    best = model.trainer.best
    YOLO(best).val(data=data, split="test", device=device, name=f"{name}_test", exist_ok=True)

    # 4. Keep a copy where the rest of the project looks for models
    shutil.copy(best, Path("models") / f"{name}.pt")
    print(f"\nsaved models/{name}.pt")
