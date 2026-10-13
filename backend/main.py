"""Web API around the detector: upload a video, it is analysed in the background, poll for the result.

    uvicorn backend.main:app --port 8000        (run from the project root)

POST   /api/jobs          upload a video (form fields: file, use_yoloe)  -> job
GET    /api/jobs          all jobs, newest first
GET    /api/jobs/{id}     one job: status, progress (0-1) and, when done, the result
DELETE /api/jobs/{id}     remove a job and its files
/files/<id>/...           annotated video + evidence images (static)
"""
import json
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from common import BIKE_MODEL, HEAD_MODEL, load_yoloe, pick_device
from detect_violations import analyse

UPLOADS, OUTPUTS = Path("data/uploads"), Path("outputs")
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

jobs = {}                     # id -> job dict; also saved as outputs/<id>/job.json so it survives a restart
todo = queue.Queue()          # job ids waiting for the model (it can only analyse one video at a time)


def save(job):
    (OUTPUTS / job["id"]).mkdir(parents=True, exist_ok=True)
    (OUTPUTS / job["id"] / "job.json").write_text(json.dumps(job))


def build_result(job_id, records, bikes_checked):
    """What the frontend needs: counts, the violations with image URLs, and the video URL."""
    return {
        "bikes_checked": bikes_checked,
        "counts": {"no_helmet": sum(r["violation"] == "no_helmet" for r in records),
                   "triple_riding": sum(r["violation"] == "triple_riding" for r in records)},
        "video_url": f"/files/{job_id}/annotated.mp4",
        "violations": [{"bike": r["bike"], "type": r["violation"], "time": r["time"], "time_s": r["time_s"],
                        "riders": r["riders"], "confidence": r["confidence"],
                        "image_url": f"/files/{job_id}/evidence/{Path(r['evidence']).name}"} for r in records],
    }


def worker(models):
    """One background thread: takes jobs off the queue and runs the detector on them."""
    bike_model, head_model, yoloe = models
    while True:
        job = jobs[todo.get()]
        job["status"] = "processing"
        save(job)

        def progress(f):
            job["progress"] = round(min(f, 1.0), 3)

        try:
            video = UPLOADS / f"{job['id']}{job['ext']}"
            records, n_bikes = analyse(video, bike_model, head_model, yoloe if job["use_yoloe"] else None,
                                       pick_device(), show=False, progress=progress)
            if not (OUTPUTS / job["id"] / "annotated.mp4").exists():
                raise RuntimeError("could not read this video")
            job["result"] = build_result(job["id"], records, n_bikes)
            job["status"], job["progress"] = "done", 1.0
        except Exception as e:                      # show the reason in the website instead of killing the worker
            job["status"], job["error"] = "error", str(e)
        job["finished_at"] = time.time()
        save(job)


@asynccontextmanager
async def lifespan(app):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    for f in OUTPUTS.glob("*/job.json"):           # jobs from earlier runs
        job = json.loads(f.read_text())
        if job["status"] in ("queued", "processing"):
            job["status"], job["error"] = "error", "server restarted while analysing"
        jobs[job["id"]] = job
    models = (YOLO(BIKE_MODEL), YOLO(HEAD_MODEL), load_yoloe())     # loaded once, not per request
    threading.Thread(target=worker, args=(models,), daemon=True).start()
    yield


app = FastAPI(title="Two-Wheeler Violation Detection", lifespan=lifespan)


@app.post("/api/jobs", status_code=201)
async def create_job(file: UploadFile = File(...), use_yoloe: bool = Form(True)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type '{ext}', use one of {sorted(ALLOWED)}")
    job_id = uuid.uuid4().hex[:8]
    with open(UPLOADS / f"{job_id}{ext}", "wb") as out:
        shutil.copyfileobj(file.file, out)         # streamed to disk, not held in memory
    job = {"id": job_id, "filename": file.filename, "ext": ext, "use_yoloe": use_yoloe, "status": "queued",
           "progress": 0.0, "created_at": time.time(), "result": None, "error": None}
    jobs[job_id] = job
    save(job)
    todo.put(job_id)
    return job


@app.get("/api/jobs")
def list_jobs():
    return sorted(jobs.values(), key=lambda j: -j["created_at"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return jobs[job_id]


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] in ("queued", "processing"):
        raise HTTPException(409, "still analysing")
    jobs.pop(job_id)
    shutil.rmtree(OUTPUTS / job_id, ignore_errors=True)
    (UPLOADS / f"{job_id}{job['ext']}").unlink(missing_ok=True)


OUTPUTS.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUTS), name="files")
