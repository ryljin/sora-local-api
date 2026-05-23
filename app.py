import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from sora_client import (
    OUTPUT_DIR,
    PINS_FILE,
    batch_progress_percent,
    client,
    create_openai_video_batch,
    extract_video_id_from_batch_row,
    get_batch_counts,
    parse_batch_jsonl_text,
    read_file_content_text,
    retrieve_openai_batch,
)


app = Flask(__name__)

jobs = {}
jobs_lock = threading.Lock()
pins_lock = threading.Lock()
views_lock = threading.Lock()
VIEWS_FILE = OUTPUT_DIR / "views.json"


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def extract_video_id(filename: str):
    match = re.search(r"(video_[^.]+)", filename)
    return match.group(1) if match else None


def safe_output_path(filename: str) -> Path:
    path = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()

    if output_root not in path.parents and path != output_root:
        raise ValueError("Invalid output path.")

    return path


def load_pins():
    with pins_lock:
        if not PINS_FILE.exists():
            return set()

        try:
            data = json.loads(PINS_FILE.read_text(encoding="utf-8"))
            pins = data.get("pins", [])
            if isinstance(pins, list):
                return set(str(item) for item in pins)
        except Exception:
            return set()

        return set()


def save_pins(pins):
    with pins_lock:
        PINS_FILE.write_text(
            json.dumps({"pins": sorted(pins)}, indent=2),
            encoding="utf-8",
        )


def load_views():
    with views_lock:
        if not VIEWS_FILE.exists():
            return set()

        try:
            data = json.loads(VIEWS_FILE.read_text(encoding="utf-8"))
            viewed = data.get("viewed", [])
            if isinstance(viewed, list):
                return set(str(item) for item in viewed)
        except Exception:
            return set()

        return set()


def save_views(viewed):
    with views_lock:
        VIEWS_FILE.write_text(
            json.dumps({"viewed": sorted(viewed)}, indent=2),
            encoding="utf-8",
        )


def is_viewed(filename: str) -> bool:
    return filename in load_views()


def mark_viewed(filename: str):
    path = safe_output_path(filename)
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise FileNotFoundError("Video not found.")

    viewed = load_views()
    viewed.add(filename)
    save_views(viewed)
    return True


def is_pinned(filename: str) -> bool:
    return filename in load_pins()


def set_pin(filename: str, pinned: bool):
    path = safe_output_path(filename)
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise FileNotFoundError("Video not found.")

    pins = load_pins()

    if pinned:
        pins.add(filename)
    else:
        pins.discard(filename)

    save_pins(pins)
    return filename in pins


def find_filename_by_video_id(video_id: str):
    if not video_id:
        return None

    for path in OUTPUT_DIR.glob("*.mp4"):
        if video_id in path.name:
            return path.name

    for path in OUTPUT_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("video_id:") and line.split(":", 1)[1].strip() == video_id:
                mp4_path = path.with_suffix(".mp4")
                if mp4_path.exists():
                    return mp4_path.name

    return None


def read_metadata(video_filename: str):
    video_path = safe_output_path(video_filename)
    metadata_path = video_path.with_suffix(".txt")

    data = {
        "filename": video_filename,
        "video_id": extract_video_id(video_filename),
        "model": "sora-2",
        "seconds": "4",
        "size": "720x1280",
        "prompt": "",
        "source_video_id": "",
        "source_filename": "",
        "source_mode": "",
        "batch_id": "",
        "batch_custom_id": "",
        "pinned": is_pinned(video_filename),
        "viewed": is_viewed(video_filename),
        "is_new": not is_viewed(video_filename),
    }

    if not metadata_path.exists():
        return data

    text = metadata_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    in_prompt = False
    prompt_lines = []

    for line in lines:
        if line.strip() == "prompt:":
            in_prompt = True
            continue

        if in_prompt:
            prompt_lines.append(line)
            continue

        if line.startswith("video_id:"):
            data["video_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("model:"):
            data["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("seconds:"):
            data["seconds"] = line.split(":", 1)[1].strip()
        elif line.startswith("size:"):
            data["size"] = line.split(":", 1)[1].strip()
        elif line.startswith("source_video_id:"):
            data["source_video_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("source_filename:"):
            data["source_filename"] = line.split(":", 1)[1].strip()
        elif line.startswith("source_mode:"):
            data["source_mode"] = line.split(":", 1)[1].strip()
        elif line.startswith("batch_id:"):
            data["batch_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("batch_custom_id:"):
            data["batch_custom_id"] = line.split(":", 1)[1].strip()

    data["prompt"] = "\n".join(prompt_lines).strip()

    if data["source_video_id"]:
        source_path = OUTPUT_DIR / data["source_filename"] if data["source_filename"] else None

        if not data["source_filename"] or not source_path.exists():
            found = find_filename_by_video_id(data["source_video_id"])
            data["source_filename"] = found or data["source_filename"] or ""

    return data


def save_rich_metadata(
    video_id: str,
    output_path: Path,
    prompt: str,
    model: str,
    seconds: str,
    size: str,
    source_mode: str = "",
    source_video_id: str = "",
    source_filename: str = "",
    batch_id: str = "",
    batch_custom_id: str = "",
):
    metadata_path = output_path.with_suffix(".txt")

    metadata_path.write_text(
        f"video_id: {video_id}\n"
        f"created_at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"video_file: {output_path.name}\n"
        f"model: {model}\n"
        f"seconds: {seconds}\n"
        f"size: {size}\n"
        f"source_mode: {source_mode}\n"
        f"source_video_id: {source_video_id}\n"
        f"source_filename: {source_filename}\n"
        f"batch_id: {batch_id}\n"
        f"batch_custom_id: {batch_custom_id}\n\n"
        f"prompt:\n{prompt}\n",
        encoding="utf-8",
    )


def get_local_videos():
    videos = []

    for path in OUTPUT_DIR.glob("*.mp4"):
        videos.append(read_metadata(path.name))

    videos.sort(
        key=lambda video: (
            1 if video.get("pinned") else 0,
            safe_output_path(video["filename"]).stat().st_mtime,
        ),
        reverse=True,
    )

    return videos


def get_recent_jobs():
    with jobs_lock:
        copied_jobs = list(jobs.values())

    copied_jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    return copied_jobs


def create_or_continue_video(item):
    mode = item.get("mode", "generate")
    prompt = item["prompt"].strip()

    if mode == "remix":
        source_video_id = item.get("source_video_id", "").strip()
        if not source_video_id:
            raise ValueError("Missing source video ID for remix.")

        return client.videos.remix(
            source_video_id,
            prompt=prompt,
        )

    if mode == "extend":
        source_video_id = item.get("source_video_id", "").strip()
        if not source_video_id:
            raise ValueError("Missing source video ID for extend.")

        return client.videos.extensions.create(
            video=source_video_id,
            prompt=prompt,
        )

    return client.videos.create(
        model=item["model"],
        prompt=prompt,
        seconds=str(item["seconds"]),
        size=item["size"],
    )


def run_generation_job(job_id, items):
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")

    for index, item in enumerate(items):
        prompt = item["prompt"].strip()
        model = item.get("model", "sora-2")
        seconds = str(item.get("seconds", "4"))
        size = item.get("size", "720x1280")
        mode = item.get("mode", "generate")
        source_video_id = item.get("source_video_id") or ""
        source_filename = item.get("source_filename") or ""

        if source_video_id and not source_filename:
            source_filename = find_filename_by_video_id(source_video_id) or ""

        try:
            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "creating",
                    "progress": 0,
                    "source_video_id": source_video_id,
                    "source_filename": source_filename,
                })

            video = create_or_continue_video(item)

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "queued",
                    "video_id": video.id,
                    "progress": 0,
                })

            while True:
                video = client.videos.retrieve(video.id)
                progress = getattr(video, "progress", None)
                progress = progress if progress is not None else 0

                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": video.status,
                        "progress": progress,
                    })

                if video.status == "completed":
                    break

                if video.status == "failed":
                    error = getattr(video, "error", None)
                    message = getattr(error, "message", "Unknown video generation error")
                    raise RuntimeError(message)

                time.sleep(5)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = mode if mode in ("remix", "extend") else "sora"
            output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{video.id}.mp4"

            content = client.videos.download_content(video.id, variant="video")
            content.write_to_file(str(output_path))

            save_rich_metadata(
                video_id=video.id,
                output_path=output_path,
                prompt=prompt,
                model=model,
                seconds=seconds,
                size=size,
                source_mode=mode if mode in ("remix", "extend") else "",
                source_video_id=source_video_id,
                source_filename=source_filename,
            )

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "completed",
                    "progress": 100,
                    "filename": output_path.name,
                    "video_id": video.id,
                    "source_video_id": source_video_id,
                    "source_filename": source_filename,
                })

        except Exception as e:
            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "failed",
                    "error": str(e),
                })

    with jobs_lock:
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")


def run_openai_batch_job(job_id, items):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "uploading_batch"
            jobs[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")
            for item in jobs[job_id]["items"]:
                item["status"] = "waiting_for_batch"
                item["progress"] = 0

        batch, batch_input_file, jsonl_path = create_openai_video_batch(items, job_id)

        with jobs_lock:
            jobs[job_id].update({
                "status": batch.status,
                "batch_id": batch.id,
                "batch_input_file_id": batch_input_file.id,
                "batch_jsonl": str(jsonl_path),
                "progress": batch_progress_percent(batch),
                "request_counts": get_batch_counts(batch),
            })

        while True:
            batch = retrieve_openai_batch(batch.id)
            progress = batch_progress_percent(batch)
            counts = get_batch_counts(batch)

            with jobs_lock:
                jobs[job_id].update({
                    "status": batch.status,
                    "progress": progress,
                    "request_counts": counts,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                })

                for index, item in enumerate(jobs[job_id]["items"]):
                    if item.get("status") not in ("completed", "failed"):
                        item["status"] = batch.status
                        item["progress"] = progress

            if batch.status in TERMINAL_BATCH_STATUSES:
                break

            time.sleep(20)

        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch ended with status: {batch.status}")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError("OpenAI batch completed without an output file ID.")

        output_text = read_file_content_text(output_file_id)
        rows = parse_batch_jsonl_text(output_text)

        for row in rows:
            custom_id = row.get("custom_id", "")
            match = re.match(r"item-(\d+)$", custom_id)
            if not match:
                continue

            index = int(match.group(1))
            if index < 0 or index >= len(items):
                continue

            item = items[index]
            response = row.get("response") or {}
            error = row.get("error")
            status_code = response.get("status_code")

            if error or not status_code or int(status_code) >= 400:
                message = json.dumps(error or response, ensure_ascii=False)
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "failed",
                        "error": message,
                        "progress": 100,
                    })
                continue

            video_id = extract_video_id_from_batch_row(row)
            if not video_id:
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "failed",
                        "error": "Batch output row did not include a video ID.",
                        "progress": 100,
                    })
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"batch_{timestamp}_{video_id}.mp4"

            content = client.videos.download_content(video_id, variant="video")
            content.write_to_file(str(output_path))

            save_rich_metadata(
                video_id=video_id,
                output_path=output_path,
                prompt=item["prompt"].strip(),
                model=item.get("model", "sora-2"),
                seconds=str(item.get("seconds", "4")),
                size=item.get("size", "720x1280"),
                batch_id=batch.id,
                batch_custom_id=custom_id,
            )

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "completed",
                    "progress": 100,
                    "filename": output_path.name,
                    "video_id": video_id,
                    "batch_custom_id": custom_id,
                })

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
            for item in jobs[job_id]["items"]:
                if item.get("status") not in ("completed", "failed"):
                    item["status"] = "failed"
                    item["error"] = str(e)


def clean_item_from_request(item):
    prompt = item.get("prompt", "").strip()
    if not prompt:
        return None

    mode = item.get("mode", "generate")
    source_video_id = item.get("source_video_id") or ""
    source_filename = item.get("source_filename") or ""

    if source_video_id and not source_filename:
        source_filename = find_filename_by_video_id(source_video_id) or ""

    return {
        "prompt": prompt,
        "model": item.get("model", "sora-2"),
        "seconds": str(item.get("seconds", "4")).strip(),
        "size": item.get("size", "720x1280"),
        "mode": mode,
        "source_video_id": source_video_id,
        "source_filename": source_filename,
        "status": "waiting",
        "progress": 0,
        "filename": None,
        "video_id": None,
        "error": None,
    }


def create_local_job(items, job_type="standard"):
    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": None,
            "completed_at": None,
            "progress": 0,
            "batch_id": None,
            "items": items,
        }

    return job_id


@app.route("/")
def index():
    return render_template("index.html", videos=get_local_videos())


@app.route("/video/<path:filename>")
def video_detail(filename):
    try:
        video_path = safe_output_path(filename)
    except ValueError:
        return "Invalid path", 400

    if not video_path.exists():
        return "Video not found", 404

    try:
        mark_viewed(filename)
    except Exception:
        pass

    video = read_metadata(filename)
    return render_template("detail.html", video=video)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    items = data.get("items", [])

    clean_items = []

    for item in items:
        clean_item = clean_item_from_request(item)
        if clean_item:
            clean_items.append(clean_item)

    if not clean_items:
        return jsonify({"error": "No prompts provided."}), 400

    job_id = create_local_job(clean_items, job_type="standard")

    thread = threading.Thread(target=run_generation_job, args=(job_id, clean_items), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/start-batch", methods=["POST"])
def api_start_batch():
    data = request.get_json(force=True)
    items = data.get("items", [])

    clean_items = []

    for item in items:
        clean_item = clean_item_from_request(item)
        if clean_item:
            clean_items.append(clean_item)

    if not clean_items:
        return jsonify({"error": "No prompts provided."}), 400

    unsupported = [item for item in clean_items if item.get("mode") != "generate"]
    if unsupported:
        return jsonify({
            "error": "Real OpenAI Batch mode currently supports normal generate items only in this UI. Send remix/extend as standard jobs."
        }), 400

    models = set(item.get("model", "sora-2") for item in clean_items)
    if len(models) > 1:
        return jsonify({
            "error": "OpenAI Batch input files for this UI are restricted to one model at a time. Use one model per batch."
        }), 400

    job_id = create_local_job(clean_items, job_type="openai_batch")

    thread = threading.Thread(target=run_openai_batch_job, args=(job_id, clean_items), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify(job)


@app.route("/api/jobs")
def api_jobs():
    return jsonify({"jobs": get_recent_jobs()})


@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": get_local_videos()})


@app.route("/api/pins")
def api_pins():
    return jsonify({"pins": sorted(load_pins())})


@app.route("/api/pin", methods=["POST"])
def api_pin():
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    pinned = bool(data.get("pinned", True))

    if not filename:
        return jsonify({"error": "Missing filename."}), 400

    try:
        final_state = set_pin(filename, pinned)
    except FileNotFoundError:
        return jsonify({"error": "Video not found."}), 404
    except ValueError:
        return jsonify({"error": "Invalid filename."}), 400

    return jsonify({"filename": filename, "pinned": final_state})


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
