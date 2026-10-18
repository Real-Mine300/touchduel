import json
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
JOBS = WEB / "jobs"
SCRIPT = ROOT / "src" / "web_overlay.py"

jobs = {}


def ffmpeg_exe():
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_job(job_id, video_path):
    job = jobs[job_id]
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), str(video_path)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS "):
            _, done, total = line.split()
            done = int(done)
            total = int(total)
            job["frames_done"] = done
            job["frames_total"] = total
            job["percent"] = round(100 * done / total, 1) if total else 0
        elif line.startswith("JUGGLES "):
            job["juggles"] = json.loads(line[len("JUGGLES "):])
        elif line.startswith("VIDEO "):
            job["video"] = line[len("VIDEO "):].strip()
        else:
            job["log"] = (job["log"] + "\n" + line)[-4000:]

    code = proc.wait()
    src = job.get("video")
    dst = JOBS / job_id / "overlay.mp4"
    ff = ffmpeg_exe()
    if code == 0 and src and Path(src).is_file() and ff:
        converted = subprocess.run(
            [
                ff, "-y", "-i", src, "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(dst),
            ],
            capture_output=True,
            text=True,
        )
        if converted.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            job["video"] = str(dst)
    if code != 0:
        job["status"] = "error"
        job["error"] = "The overlay script stopped early."
    elif not job.get("video") or not Path(job["video"]).is_file():
        job["status"] = "error"
        job["error"] = "Scoring finished but the overlay video was not written."
    else:
        job["status"] = "done"
        job["percent"] = 100


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    folder = JOBS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"
    dest = folder / f"input{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    jobs[job_id] = {
        "status": "running",
        "percent": 0,
        "frames_done": 0,
        "frames_total": 0,
        "juggles": [],
        "video": None,
        "error": None,
        "log": "",
    }
    threading.Thread(target=run_job, args=(job_id, dest), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No job with that id.")
    return {
        "status": job["status"],
        "percent": job["percent"],
        "frames_done": job["frames_done"],
        "frames_total": job["frames_total"],
        "juggles": job["juggles"],
        "has_video": bool(job.get("video") and Path(job["video"]).is_file()),
        "error": job["error"],
        "log": job["log"],
    }


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or not job.get("video"):
        raise HTTPException(404, "Video is not ready.")
    return FileResponse(job["video"], media_type="video/mp4")


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="site")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)