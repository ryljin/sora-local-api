import base64
import json
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
BATCH_DIR = OUTPUT_DIR / "batches"
PINS_FILE = OUTPUT_DIR / "pins.json"

load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def create_video(prompt: str, model: str, seconds: str, size: str):
    print("Creating video job...")

    video = client.videos.create(
        model=model,
        prompt=prompt,
        seconds=str(seconds),
        size=size,
    )

    print(f"Created video job: {video.id}")
    print(f"Initial status: {video.status}")

    return video


def wait_for_video(video_id: str):
    print("Polling video status...")

    while True:
        video = client.videos.retrieve(video_id)

        progress = getattr(video, "progress", None)
        if progress is not None:
            print(f"Status: {video.status} | Progress: {progress}%")
        else:
            print(f"Status: {video.status}")

        if video.status == "completed":
            return video

        if video.status == "failed":
            error = getattr(video, "error", None)
            message = getattr(error, "message", "Unknown video generation error")
            raise RuntimeError(f"Video generation failed: {message}")

        time.sleep(5)


def save_metadata(video_id: str, output_path: Path, prompt: str, model: str, seconds: str, size: str):
    metadata_path = output_path.with_suffix(".txt")

    metadata_path.write_text(
        f"video_id: {video_id}\n"
        f"created_at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"video_file: {output_path.name}\n"
        f"model: {model}\n"
        f"seconds: {seconds}\n"
        f"size: {size}\n\n"
        f"prompt:\n{prompt}\n",
        encoding="utf-8",
    )

    print(f"Saved metadata to: {metadata_path}")


def download_video(video_id: str, prompt: str, model: str, seconds: str, size: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"sora_{timestamp}_{video_id}.mp4"

    print("Downloading video...")

    content = client.videos.download_content(
        video_id,
        variant="video",
    )

    content.write_to_file(str(output_path))

    print(f"Saved video to: {output_path}")

    save_metadata(video_id, output_path, prompt, model, seconds, size)

    return output_path


def generate_video(prompt: str, model: str = "sora-2", seconds: str = "4", size: str = "720x1280"):
    prompt = prompt.strip()
    seconds = str(seconds).strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if not seconds:
        raise ValueError("Duration cannot be empty.")

    if not seconds.isdigit():
        raise ValueError("Duration must be a whole number of seconds, such as 4, 8, 12, 16, or 20.")

    video = create_video(prompt, model, seconds, size)
    completed_video = wait_for_video(video.id)
    output_path = download_video(completed_video.id, prompt, model, seconds, size)

    return output_path


def split_batch_prompts(raw_text: str):
    parts = raw_text.split("---")
    prompts = []

    for part in parts:
        cleaned = part.strip()
        if cleaned:
            prompts.append(cleaned)

    return prompts


def generate_batch(raw_text: str, model: str = "sora-2", seconds: str = "4", size: str = "720x1280"):
    prompts = split_batch_prompts(raw_text)

    if not prompts:
        raise ValueError("No prompts found. Add one prompt, or separate multiple prompts with ---.")

    results = []

    for index, prompt in enumerate(prompts, start=1):
        print(f"\nStarting batch item {index} of {len(prompts)}")
        output_path = generate_video(prompt, model, seconds, size)

        results.append(
            {
                "index": index,
                "prompt": prompt,
                "filename": output_path.name,
                "path": str(output_path),
            }
        )

    return results


def local_image_to_data_url(local_filename: str, mime_type: str = "") -> str:
    """Return a base64 data URL for an image cached under outputs/uploads/."""
    uploads_root = (OUTPUT_DIR / "uploads").resolve()
    local_path = (uploads_root / local_filename).resolve()

    if uploads_root not in local_path.parents and local_path != uploads_root:
        raise ValueError("Invalid image reference path.")

    if not local_path.exists():
        raise FileNotFoundError(f"Image reference file not found: {local_filename}")

    resolved_mime = (mime_type or "").strip()
    if not resolved_mime:
        resolved_mime = mimetypes.guess_type(str(local_path))[0] or "image/png"

    encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
    return f"data:{resolved_mime};base64,{encoded}"


def make_video_batch_jsonl(items: List[Dict[str, Any]], jsonl_path: Path, image_reference_mode: str = "file_id"):
    """
    Creates a Batch API input file for POST /v1/videos.
    Every line must target the same endpoint. Batch video requests must be JSON bodies.

    image_reference_mode:
      - file_id: upload local image references to OpenAI Files and use input_reference.file_id
      - base64: embed cached local image references as input_reference.image_url data URLs
    """
    image_reference_mode = "base64" if image_reference_mode == "base64" else "file_id"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(items):
            body = {
                "model": item.get("model", "sora-2"),
                "prompt": item["prompt"].strip(),
                "seconds": str(item.get("seconds", "4")).strip(),
                "size": item.get("size", "720x1280"),
            }

            input_reference_file_id = (item.get("input_reference_file_id") or "").strip()
            input_reference_image_url = (item.get("input_reference_image_url") or "").strip()
            input_reference_local_filename = (item.get("input_reference_local_filename") or "").strip()
            input_reference_mime_type = (item.get("input_reference_mime_type") or "image/png").strip() or "image/png"

            if image_reference_mode == "base64" and input_reference_local_filename:
                body["input_reference"] = {
                    "image_url": local_image_to_data_url(input_reference_local_filename, input_reference_mime_type),
                }
            elif input_reference_file_id:
                body["input_reference"] = {
                    "file_id": input_reference_file_id,
                }
            elif input_reference_image_url:
                body["input_reference"] = {
                    "image_url": input_reference_image_url,
                }

            record = {
                "custom_id": f"item-{index}",
                "method": "POST",
                "url": "/v1/videos",
                "body": body,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonl_path


def create_openai_video_batch(items: List[Dict[str, Any]], local_job_id: str, image_reference_mode: str = "file_id"):
    jsonl_path = BATCH_DIR / f"video_batch_{local_job_id}.jsonl"
    make_video_batch_jsonl(items, jsonl_path, image_reference_mode=image_reference_mode)

    with jsonl_path.open("rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch",
        )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/videos",
        completion_window="24h",
        metadata={
            "local_job_id": local_job_id,
            "description": "Sora Local UI video batch",
        },
    )

    return batch, batch_input_file, jsonl_path


def retrieve_openai_batch(batch_id: str):
    return client.batches.retrieve(batch_id)


def get_batch_counts(batch) -> Dict[str, int]:
    counts = getattr(batch, "request_counts", None)
    if not counts:
        return {"total": 0, "completed": 0, "failed": 0}

    return {
        "total": int(getattr(counts, "total", 0) or 0),
        "completed": int(getattr(counts, "completed", 0) or 0),
        "failed": int(getattr(counts, "failed", 0) or 0),
    }


def batch_progress_percent(batch) -> int:
    counts = get_batch_counts(batch)
    total = counts["total"]
    if total <= 0:
        return 0

    done = counts["completed"] + counts["failed"]
    return max(0, min(100, int((done / total) * 100)))


def read_file_content_text(file_id: str) -> str:
    response = client.files.content(file_id)

    if hasattr(response, "text"):
        return response.text

    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)

    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")

    return str(response)


def parse_batch_jsonl_text(text: str) -> List[Dict[str, Any]]:
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    return rows


def extract_video_id_from_batch_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Extract the video id from a Batch API output row.

    The video endpoint has returned slightly different JSON shapes across SDK/API
    versions, and partial-failure batches make it especially important that a
    successful row is not missed just because the id is nested differently.
    This accepts the common explicit shapes first, then falls back to a small
    recursive search for any string that looks like a Sora video id.
    """
    response = row.get("response") or {}
    body = response.get("body") or {}

    if isinstance(body, dict):
        if isinstance(body.get("id"), str) and body["id"].startswith("video_"):
            return body["id"]

        video = body.get("video")
        if isinstance(video, dict) and isinstance(video.get("id"), str) and video["id"].startswith("video_"):
            return video["id"]
        if isinstance(video, str) and video.startswith("video_"):
            return video

        data = body.get("data")
        if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].startswith("video_"):
            return data["id"]

    def walk(value: Any) -> Optional[str]:
        if isinstance(value, str):
            if value.startswith("video_"):
                return value
            return None

        if isinstance(value, dict):
            # Prefer id-like keys first.
            for key in ("id", "video_id", "video"):
                found = walk(value.get(key))
                if found:
                    return found
            for nested in value.values():
                found = walk(nested)
                if found:
                    return found

        if isinstance(value, list):
            for nested in value:
                found = walk(nested)
                if found:
                    return found

        return None

    return walk(row)
