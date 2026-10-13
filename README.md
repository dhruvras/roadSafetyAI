# Two-Wheeler Violation Detection

Detects **riders without helmets** and **triple riding** in traffic video, using YOLO object detection,
multi-object tracking and per-vehicle temporal voting.

B.Tech CSE minor project, THDC Institute of Hydropower Engineering and Technology, Tehri.
Team: Dhruv Rastogi, Muskan.

## How it works

```
video ─► YOLO26n (COCO) + ByteTrack ─► every vehicle gets an ID; keep the ones that are mostly "motorcycle"
           │
           ▼ for each motorcycle: cut out the region above it (its riders)
       head model (ours, fine-tuned YOLO26n) ─► heads, each "helmet" or "no_helmet"
       YOLOE ("helmet" prompt, zero-shot)     ─► second opinion: dark rear-view helmets our model misses
       person detector (COCO)                 ─► a rider with no helmet/head box on top = bare head seen from behind
           │
           ▼ every head is given to ONE bike (neighbouring crops overlap), then remember per bike, frame after frame: how many heads, any confident bare head?
       vote once per bike (ignore bikes seen too briefly or parked)
           ├─ 3+ heads in ≥ 3 frames and ≥ 30% of frames     ─► TRIPLE RIDING
           └─ bare head in ≥ 40% of frames (and ≥ 3 frames)  ─► NO HELMET
           │
           ▼
       outputs/<video>/  annotated.mp4 · violations.csv · evidence/*.jpg
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU; the scripts pick automatically.
`models/yolo26n.pt` and `models/yoloe-26s-seg.pt` (the helmet second opinion) download themselves on first run. The trained head model
`models/helmet_v1.pt` (5 MB) is included in the repo. Datasets, videos, other weights and results are
not in git; you can retrain the model yourself (below).

## Usage

| Step | Command |
|---|---|
| Website: API (terminal 1, from the project root) | `uvicorn backend.main:app --port 8000` |
| Website: frontend (terminal 2) | `cd frontend && npm install && npm run dev`, then open http://localhost:5173 |
| Detect violations in videos | `python detect_violations.py data/clips/*.mp4` (add `--model models/helmet_v2.pt`, `--no-show`, `--no-yoloe`) |
| Build the training set from the Kaggle dataset | `python prepare_dataset.py` (expects `data/kaggle_helmet/{images,annotations}`) |
| Train a head model | `python train.py data/helmet_v1/data.yaml helmet_v1` |
| Cut rider crops from your own videos | `python dump_crops.py data/clips/*.mp4` (rebuilds `data/v2_crops/` from scratch) |
| Auto-label the crops (YOLOE + v1 + person check) | `python auto_label.py` |
| Fix crops by hand | `python label_tool.py` or `python label_tool.py --review` |

## Website

```
browser (React) ──upload──► FastAPI ──queue──► one worker thread runs detect_violations.analyse()
       ▲                       │                          │ writes outputs/<job id>/ (video, csv, evidence)
       └──── polls progress ◄──┘◄─────────────────────────┘
```

`POST /api/jobs` stores the video and returns a job id; the page polls `GET /api/jobs/<id>` every second for
progress and, when done, the violations with links to the annotated video and evidence photos (`/files/...`).

## Folder layout

```
common.py            shared settings + helpers (device, rider region, IoU, head parsing, YOLOE check)
detect_violations.py the full pipeline
backend/main.py      FastAPI: upload a video, analyse it in the background, serve results
frontend/            React (Vite) website: upload, progress, annotated video, violations, evidence
prepare_dataset.py   Kaggle Pascal-VOC XML -> YOLO labels, train/val/test split
train.py             fine-tune YOLO26n as the head model
dump_crops.py        rider crops from videos, pre-labelled
auto_label.py        labels crops with a big zero-shot model (YOLOE), keeps only confident ones
label_tool.py        small OpenCV tool to correct labels
learning/            step-by-step scripts from building the project (video I/O, detection, tracking, rider counting)
docs/synopsis/       project synopsis
data/ models/ runs/ outputs/   local only (git-ignored): videos, datasets, weights, results
```

## Results so far

| | |
|---|---|
| Head model v1 (Kaggle data only), held-out test images | mAP50 0.89 (helmet 0.91, no_helmet 0.87) |
| Pipeline with head model only, 7 stock traffic clips, checked by eye | 17 reported, 8 correct |
| Pipeline + YOLOE second opinion + one-bike-per-head, same clips | 5 reported, 4 correct (no triple riding in these clips; a few real cases are now missed; tuned on these same clips, so optimistic) |
| + bare riders seen from behind (person with no helmet found by YOLOE), same clips | 9 reported, about 7 look correct by eye |

Main failure: v1 has never seen helmets **from behind**, so dark rear-view helmets get called "no helmet".
v2 adds 107 auto-labelled rider crops from Indian traffic (mostly rear views).

## Data

- Head model v1: [Helmet Detection dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/helmet-detection), 764 images, 15 dropped for corrupted boxes.
- Test clips: free stock footage (Pexels / Pixabay).
