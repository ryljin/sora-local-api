import base64
import copy
import json
import tempfile
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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

video_index_lock = threading.Lock()
video_index_cache = {
    "signature": None,
    "videos": [],
    "built_at": None,
    "dirty": True,
    "last_disk_check": 0.0,
    "loaded_from_disk": False,
    "refresh_running": False,
    "last_error": "",
}

pins_lock = threading.Lock()
views_lock = threading.Lock()
favorites_lock = threading.Lock()
archive_lock = threading.Lock()
budget_lock = threading.Lock()
presets_lock = threading.Lock()
tags_lock = threading.Lock()
VIEWS_FILE = OUTPUT_DIR / "views.json"
FAVORITES_FILE = OUTPUT_DIR / "favorites.json"
ARCHIVE_FILE = OUTPUT_DIR / "archive.json"
BUDGET_FILE = OUTPUT_DIR / "budget.json"
JOBS_FILE = OUTPUT_DIR / "jobs.json"
PRESETS_FILE = OUTPUT_DIR / "presets.json"
TAGS_FILE = OUTPUT_DIR / "tags.json"
THUMBNAIL_DIR = OUTPUT_DIR / "thumbnails"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
THUMBNAIL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

thumbnail_lock = threading.Lock()
thumbnail_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "generated": 0,
    "failed": 0,
    "last_error": "",
    "ffmpeg_path": "",
    "started_at": "",
    "completed_at": "",
}


TERMINAL_JOB_STATUSES = {"completed", "completed_with_errors", "failed", "expired", "cancelled", "interrupted"}
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
NON_CANCELLABLE_LOCAL_BATCH_STATUSES = TERMINAL_JOB_STATUSES | {"processing_batch_output"}
GALLERY_DISK_RESCAN_SECONDS = 300
GALLERY_INDEX_FILE = OUTPUT_DIR / "gallery_index.json"



def load_persisted_jobs():
    if not JOBS_FILE.exists():
        return {}

    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        raw_jobs = data.get("jobs", [])
        if not isinstance(raw_jobs, list):
            return {}

        loaded = {}
        for job in raw_jobs:
            if not isinstance(job, dict):
                continue

            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                continue

            # A browser page reload should not lose jobs. If the Flask dev server itself
            # restarted while a background thread was running, keep the row visible instead
            # of pretending it never existed.
            if job.get("status") not in TERMINAL_JOB_STATUSES:
                job["status"] = "interrupted"
                job["error"] = "Local server restarted before this job finished. The job record was preserved, but this local worker is no longer polling it."
                job["completed_at"] = job.get("completed_at") or datetime.now().isoformat(timespec="seconds")
                for item in job.get("items", []):
                    if isinstance(item, dict) and item.get("status") not in ("completed", "failed"):
                        item["status"] = "interrupted"
                        item["error"] = job["error"]

            loaded[job_id] = job

        return loaded
    except Exception:
        return {}


def save_jobs_snapshot_unlocked():
    try:
        job_values = list(jobs.values())
        job_values.sort(key=lambda job: job.get("created_at") or "", reverse=True)
        JOBS_FILE.write_text(
            json.dumps({"jobs": job_values[:100]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def extract_video_id(filename: str):
    match = re.search(r"(video_[^.]+)", filename)
    return match.group(1) if match else None


def safe_output_path(filename: str) -> Path:
    path = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()

    if output_root not in path.parents and path != output_root:
        raise ValueError("Invalid output path.")

    return path


def safe_thumbnail_path(filename: str) -> Path:
    path = (THUMBNAIL_DIR / filename).resolve()
    thumbnail_root = THUMBNAIL_DIR.resolve()

    if thumbnail_root not in path.parents and path != thumbnail_root:
        raise ValueError("Invalid thumbnail path.")

    return path


def thumbnail_filename_for_video(video_filename: str) -> str:
    return f"{Path(video_filename).stem}.jpg"


def thumbnail_path_for_video(video_filename: str) -> Path:
    return THUMBNAIL_DIR / thumbnail_filename_for_video(video_filename)


def cached_thumbnail(video_filename: str) -> str:
    """Return the cached thumbnail filename only. Never runs ffmpeg."""
    thumb_path = thumbnail_path_for_video(video_filename)
    return thumb_path.name if thumb_path.exists() else ""


def missing_thumbnail_filenames(limit: int = 200):
    missing = []

    for path in sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not cached_thumbnail(path.name):
            missing.append(path.name)

        if len(missing) >= limit:
            break

    return missing


def save_browser_thumbnail(video_filename: str, image_data: str) -> str:
    # Browser fallback: the page loads one video at a time, captures a canvas frame,
    # and POSTs it here. This avoids loading every video in the gallery and avoids
    # depending solely on ffmpeg on Windows machines.
    video_path = safe_output_path(video_filename)
    if not video_path.exists():
        raise FileNotFoundError("Video file not found.")

    if not image_data.startswith("data:image/jpeg;base64,"):
        raise ValueError("Thumbnail must be a JPEG data URL.")

    encoded = image_data.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)

    if not raw:
        raise ValueError("Empty thumbnail image.")

    if len(raw) > 2_500_000:
        raise ValueError("Thumbnail image is too large.")

    thumb_path = thumbnail_path_for_video(video_filename)
    temp_path = thumb_path.with_suffix(".browser.tmp.jpg")
    temp_path.write_bytes(raw)
    temp_path.replace(thumb_path)

    with thumbnail_lock:
        thumbnail_state["generated"] = thumbnail_state.get("generated", 0) + 1
        thumbnail_state["last_error"] = ""

    return thumb_path.name


def find_ffmpeg_executable() -> str:
    env_path = (os.environ.get("FFMPEG_BINARY") or "").strip().strip('"').strip("'")

    if env_path:
        candidate = Path(env_path)

        # Allow either FFMPEG_BINARY=C:\\ffmpeg\\bin\\ffmpeg.exe
        # or FFMPEG_BINARY=C:\\ffmpeg\\bin.
        if candidate.is_dir():
            exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            candidate = candidate / exe_name

        if candidate.exists():
            return str(candidate)

        # If the user gave just a command name, let PATH resolve it.
        resolved = shutil.which(env_path)
        if resolved:
            return resolved

        # Return the raw value so the visible thumbnail error shows exactly what failed.
        return env_path

    path = shutil.which("ffmpeg")
    if path:
        return path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def set_thumbnail_error(message: str):
    with thumbnail_lock:
        thumbnail_state["last_error"] = message


def generate_thumbnail_once(video_filename: str) -> str:
    """Generate one thumbnail only if it is missing, then reuse it forever."""
    video_path = safe_output_path(video_filename)
    thumb_path = thumbnail_path_for_video(video_filename)

    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return thumb_path.name

    if not video_path.exists():
        set_thumbnail_error(f"Video file does not exist: {video_filename}")
        return ""

    ffmpeg_path = find_ffmpeg_executable()
    with thumbnail_lock:
        thumbnail_state["ffmpeg_path"] = ffmpeg_path

    if not ffmpeg_path:
        set_thumbnail_error("ffmpeg was not found. Install ffmpeg or set FFMPEG_BINARY in .env.")
        return ""

    temp_path = thumb_path.with_suffix(".tmp.jpg")

    commands = [
        [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "00:00:00.5",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "4",
            str(temp_path),
        ],
        [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "4",
            str(temp_path),
        ],
    ]

    last_error = ""

    for command in commands:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30,
            )

            if result.returncode != 0:
                last_error = (result.stderr or result.stdout or f"ffmpeg exited with code {result.returncode}").strip()
                continue

            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.replace(thumb_path)
                return thumb_path.name

            last_error = "ffmpeg ran but did not create a thumbnail file."
        except Exception as exc:
            last_error = str(exc)
        finally:
            try:
                if temp_path.exists() and not thumb_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    set_thumbnail_error(f"Thumbnail failed for {video_filename}: {last_error}")
    print(f"Thumbnail failed for {video_filename}: {last_error}")
    return ""


def generate_missing_thumbnails():
    """One-time cache warmer for all existing local outputs."""
    paths = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)

    with thumbnail_lock:
        thumbnail_state.update({
            "running": True,
            "total": len(paths),
            "done": 0,
            "generated": 0,
            "failed": 0,
            "last_error": "",
            "ffmpeg_path": find_ffmpeg_executable(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "completed_at": "",
        })

    try:
        for path in paths:
            before = cached_thumbnail(path.name)
            after = generate_thumbnail_once(path.name)

            with thumbnail_lock:
                thumbnail_state["done"] += 1
                if after:
                    if not before:
                        thumbnail_state["generated"] += 1
                else:
                    thumbnail_state["failed"] += 1
    finally:
        with thumbnail_lock:
            thumbnail_state["running"] = False
            thumbnail_state["completed_at"] = datetime.now().isoformat(timespec="seconds")


def start_thumbnail_cache_warmer():
    with thumbnail_lock:
        if thumbnail_state.get("running"):
            return False
        thumbnail_state["running"] = True

    thread = threading.Thread(target=generate_missing_thumbnails, daemon=True)
    thread.start()
    return True


def get_thumbnail_status():
    with thumbnail_lock:
        return dict(thumbnail_state)

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

def load_named_set(path: Path, key: str, lock: threading.Lock):
    with lock:
        if not path.exists():
            return set()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            values = data.get(key, [])
            if isinstance(values, list):
                return set(str(item) for item in values)
        except Exception:
            return set()

        return set()


def save_named_set(path: Path, key: str, values, lock: threading.Lock):
    with lock:
        path.write_text(
            json.dumps({key: sorted(values)}, indent=2),
            encoding="utf-8",
        )


def load_favorites():
    return load_named_set(FAVORITES_FILE, "favorites", favorites_lock)


def save_favorites(favorites):
    save_named_set(FAVORITES_FILE, "favorites", favorites, favorites_lock)


def load_archive():
    return load_named_set(ARCHIVE_FILE, "archived", archive_lock)


def save_archive(archived):
    save_named_set(ARCHIVE_FILE, "archived", archived, archive_lock)


def load_budget():
    with budget_lock:
        default_data = {
            "remaining": "",
            "updated_at": None,
        }

        if not BUDGET_FILE.exists():
            return default_data

        try:
            data = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
            return {
                "remaining": str(data.get("remaining", "")),
                "updated_at": data.get("updated_at"),
            }
        except Exception:
            return default_data


def save_budget(remaining: str):
    cleaned = str(remaining or "").strip()
    with budget_lock:
        data = {
            "remaining": cleaned,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        BUDGET_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data


def normalize_preset_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip()).strip("_").lower()
    return cleaned[:80] or f"preset_{uuid.uuid4().hex[:10]}"


def load_presets():
    with presets_lock:
        if not PRESETS_FILE.exists():
            return []

        try:
            data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            presets = data.get("presets", [])
            if not isinstance(presets, list):
                return []

            cleaned = []
            seen = set()
            for preset in presets:
                if not isinstance(preset, dict):
                    continue
                name = str(preset.get("name") or "").strip()
                template = str(preset.get("template") or "").strip()
                if not name or not template:
                    continue
                preset_id = normalize_preset_id(preset.get("id") or name)
                if preset_id in seen:
                    continue
                seen.add(preset_id)
                cleaned.append({
                    "id": preset_id,
                    "name": name[:120],
                    "template": template,
                    "updated_at": preset.get("updated_at") or "",
                })
            return cleaned
        except Exception:
            return []


def save_presets(presets):
    with presets_lock:
        PRESETS_FILE.write_text(
            json.dumps({"presets": presets}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def normalize_tag_name(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip())
    return value[:60]


def load_custom_tags():
    if not TAGS_FILE.exists():
        return {}

    try:
        data = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
        raw_tags = data.get("tags", {})
        if not isinstance(raw_tags, dict):
            return {}

        tags = {}
        for name, filenames in raw_tags.items():
            tag_name = normalize_tag_name(name)
            if not tag_name:
                continue
            if isinstance(filenames, list):
                tags[tag_name] = {str(filename) for filename in filenames if filename}
            else:
                tags[tag_name] = set()
        return tags
    except Exception:
        return {}


def save_custom_tags(tags):
    clean = {}
    for name, filenames in (tags or {}).items():
        tag_name = normalize_tag_name(name)
        if not tag_name:
            continue
        clean[tag_name] = sorted({str(filename) for filename in filenames if filename})

    with tags_lock:
        TAGS_FILE.write_text(
            json.dumps({"tags": clean}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def all_custom_tag_names():
    return sorted(load_custom_tags().keys(), key=lambda value: value.lower())


def tags_for_filename(filename: str, tags=None):
    tags = tags if tags is not None else load_custom_tags()
    return sorted([name for name, filenames in tags.items() if filename in filenames], key=lambda value: value.lower())


def ensure_custom_tag(tag_name: str):
    tag_name = normalize_tag_name(tag_name)
    if not tag_name:
        raise ValueError("Tag name cannot be empty.")
    tags = load_custom_tags()
    tags.setdefault(tag_name, set())
    save_custom_tags(tags)
    return tag_name




def rename_custom_tag(old_name: str, new_name: str):
    old_name = normalize_tag_name(old_name)
    new_name = normalize_tag_name(new_name)

    if not old_name:
        raise ValueError("Select a tag to rename.")
    if not new_name:
        raise ValueError("New tag name cannot be empty.")

    tags = load_custom_tags()
    if old_name not in tags:
        raise KeyError(old_name)

    if old_name == new_name:
        return new_name

    old_files = set(tags.get(old_name, set()))
    new_files = set(tags.get(new_name, set()))
    tags[new_name] = old_files | new_files
    tags.pop(old_name, None)

    save_custom_tags(tags)
    invalidate_video_index(files_changed=False)
    return new_name


def delete_custom_tag(tag_name: str):
    tag_name = normalize_tag_name(tag_name)
    if not tag_name:
        raise ValueError("Select a tag to remove.")

    tags = load_custom_tags()
    if tag_name not in tags:
        raise KeyError(tag_name)

    tags.pop(tag_name, None)
    save_custom_tags(tags)
    invalidate_video_index(files_changed=False)
    return tag_name

def set_video_custom_tags(filename: str, tag_names):
    path = safe_output_path(filename)
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise FileNotFoundError(filename)

    selected = {normalize_tag_name(name) for name in (tag_names or [])}
    selected = {name for name in selected if name}

    tags = load_custom_tags()
    for name in selected:
        tags.setdefault(name, set())

    for name in list(tags.keys()):
        filenames = set(tags.get(name, set()))
        if name in selected:
            filenames.add(filename)
        else:
            filenames.discard(filename)
        tags[name] = filenames

    save_custom_tags(tags)
    invalidate_video_index(files_changed=False)
    return tags_for_filename(filename, tags)


def make_unique_preset_id(base_id: str, presets):
    base_id = normalize_preset_id(base_id)
    existing_ids = {preset.get("id") for preset in presets}

    if base_id not in existing_ids:
        return base_id

    for number in range(2, 10000):
        candidate = f"{base_id}_{number}"
        if candidate not in existing_ids:
            return candidate

    return f"{base_id}_{uuid.uuid4().hex[:8]}"


def upsert_preset(preset_id: str, name: str, template: str, force_new: bool = False):
    name = str(name or "").strip()
    template = str(template or "").strip()
    if not name:
        raise ValueError("Preset name is required.")
    if not template:
        raise ValueError("Preset template is required.")

    presets = load_presets()

    if force_new:
        preset_id = make_unique_preset_id(preset_id or name, presets)
    else:
        preset_id = normalize_preset_id(preset_id or name)

    now = datetime.now().isoformat(timespec="seconds")
    new_preset = {
        "id": preset_id,
        "name": name[:120],
        "template": template,
        "updated_at": now,
    }

    replaced = False
    if not force_new:
        for index, existing in enumerate(presets):
            if existing.get("id") == preset_id:
                presets[index] = new_preset
                replaced = True
                break

    if not replaced:
        presets.append(new_preset)

    presets.sort(key=lambda item: item.get("name", "").lower())
    save_presets(presets)
    return new_preset


def delete_preset(preset_id: str):
    preset_id = normalize_preset_id(preset_id)
    presets = load_presets()
    remaining = [preset for preset in presets if preset.get("id") != preset_id]
    if len(remaining) == len(presets):
        return False
    save_presets(remaining)
    return True


def apply_prompt_preset(raw_prompt: str, preset_template: str) -> str:
    raw_prompt = str(raw_prompt or "").strip()
    preset_template = str(preset_template or "").strip()

    if not preset_template:
        return raw_prompt

    if "[prompt]" in preset_template:
        return preset_template.replace("[prompt]", raw_prompt)

    return f"{preset_template}\n\n{raw_prompt}".strip()


def parse_requested_video_size(size: str):
    """Return (width, height) from a Sora size string like 720x1280."""
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", str(size or ""))
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def center_crop_resize_image_bytes(raw: bytes, target_size: str):
    """
    Sora image references must match the requested video width and height.
    This converts any uploaded reference image into a center-cropped PNG with
    exactly the same dimensions as the selected video size.
    """
    parsed_size = parse_requested_video_size(target_size)
    if not parsed_size:
        return raw, None, None

    target_width, target_height = parsed_size

    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError(
            "Image auto-crop requires Pillow. Install it with: pip install pillow. "
            f"Original import error: {exc}"
        )

    try:
        import io
        source = Image.open(io.BytesIO(raw))
        source = ImageOps.exif_transpose(source)

        if source.mode not in ("RGB", "RGBA"):
            source = source.convert("RGB")

        source_width, source_height = source.size
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height

        if source_ratio > target_ratio:
            # Source is too wide. Crop left/right.
            crop_height = source_height
            crop_width = int(round(crop_height * target_ratio))
            left = max((source_width - crop_width) // 2, 0)
            top = 0
        else:
            # Source is too tall. Crop top/bottom.
            crop_width = source_width
            crop_height = int(round(crop_width / target_ratio))
            left = 0
            top = max((source_height - crop_height) // 2, 0)

        right = min(left + crop_width, source_width)
        bottom = min(top + crop_height, source_height)
        cropped = source.crop((left, top, right, bottom))
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

        if resized.mode == "RGBA":
            # Flatten transparency to black so the final video reference has a stable RGB canvas.
            background = Image.new("RGB", resized.size, (0, 0, 0))
            background.paste(resized, mask=resized.split()[-1])
            resized = background
        else:
            resized = resized.convert("RGB")

        output = io.BytesIO()
        resized.save(output, format="PNG", optimize=True)
        return output.getvalue(), "image/png", f"{target_width}x{target_height}"
    except Exception as exc:
        raise RuntimeError(f"Could not auto-crop image reference to {target_size}: {exc}")

def sanitize_reference_filename(filename: str, mime_type: str = "image/png") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(filename or "image_reference")).strip("._")

    if not cleaned:
        cleaned = "image_reference"

    suffix = Path(cleaned).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        return cleaned

    mime_suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    return cleaned + mime_suffixes.get(mime_type, ".png")


def parse_image_data_url(data_url: str):
    match = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$", str(data_url or ""), re.DOTALL)
    if not match:
        raise ValueError("Image reference must be a base64 image data URL.")

    mime_type = match.group(1).lower()
    encoded = match.group(2)

    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image reference type: {mime_type}")

    raw = base64.b64decode(encoded, validate=True)
    if not raw:
        raise ValueError("Image reference file is empty.")

    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Image reference must be under 20 MB.")

    return mime_type, raw


def safe_upload_path(filename: str) -> Path:
    path = (UPLOAD_DIR / filename).resolve()
    root = UPLOAD_DIR.resolve()

    if root not in path.parents and path != root:
        raise ValueError("Invalid upload path.")

    return path


def save_image_reference_upload(data_url: str, filename: str = "image_reference", target_size: str = "") -> dict:
    """
    Save a browser data URL once as a local file.

    Standard Sora video creation through openai-python is multipart-based in many
    SDK versions, so passing {file_id: ...} can fail with "expected bytes/path".
    Keeping a local file lets standard jobs pass PathLike directly, while batch jobs
    can still upload the same local file to Files and reference the resulting file_id.
    """
    original_mime_type, raw = parse_image_data_url(data_url)
    formatted_size = None

    if target_size:
        raw, converted_mime_type, formatted_size = center_crop_resize_image_bytes(raw, target_size)
        mime_type = converted_mime_type or original_mime_type
    else:
        mime_type = original_mime_type

    safe_name = sanitize_reference_filename(filename, mime_type)
    suffix = Path(safe_name).suffix.lower() or ".png"

    if mime_type == "image/png" and suffix != ".png":
        suffix = ".png"

    stored_name = f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}{suffix}"
    stored_path = safe_upload_path(stored_name)
    stored_path.write_bytes(raw)

    return {
        "input_reference_local_filename": stored_name,
        "input_reference_name": safe_name,
        "input_reference_mime_type": mime_type,
        "input_reference_size": len(raw),
        "input_reference_original_mime_type": original_mime_type,
        "input_reference_formatted_size": formatted_size or "",
        "input_reference_auto_cropped": bool(formatted_size),
    }


def upload_local_image_reference_to_openai_file(local_filename: str, filename: str = "image_reference", mime_type: str = "image/png") -> str:
    """Upload a cached local image to OpenAI Files for JSONL Batch requests."""
    local_path = safe_upload_path(local_filename)
    if not local_path.exists():
        raise FileNotFoundError(f"Image reference file not found: {local_filename}")

    safe_name = sanitize_reference_filename(filename, mime_type)
    env_purpose = (os.environ.get("OPENAI_IMAGE_FILE_PURPOSE") or "").strip()
    candidate_purposes = []

    if env_purpose:
        candidate_purposes.append(env_purpose)

    for purpose in ("user_data", "vision", "assistants"):
        if purpose not in candidate_purposes:
            candidate_purposes.append(purpose)

    errors = []

    for purpose in candidate_purposes:
        try:
            with local_path.open("rb") as file_handle:
                uploaded = client.files.create(
                    file=(safe_name, file_handle, mime_type),
                    purpose=purpose,
                )
            return uploaded.id
        except TypeError:
            try:
                with local_path.open("rb") as file_handle:
                    uploaded = client.files.create(
                        file=file_handle,
                        purpose=purpose,
                    )
                return uploaded.id
            except Exception as exc:
                errors.append(f"{purpose}: {exc}")
        except Exception as exc:
            errors.append(f"{purpose}: {exc}")

    raise RuntimeError("Could not upload image reference to OpenAI Files. " + " | ".join(errors))


def upload_image_reference_to_openai_file(data_url: str, filename: str = "image_reference", target_size: str = "") -> str:
    saved = save_image_reference_upload(data_url, filename, target_size)
    return upload_local_image_reference_to_openai_file(
        saved["input_reference_local_filename"],
        saved.get("input_reference_name") or filename,
        saved.get("input_reference_mime_type") or "image/png",
    )


def prepare_input_reference_for_api(item: dict) -> dict:
    """
    Normalize image references without losing the local file needed by standard jobs.

    Browser uploads arrive as base64 data URLs. They are saved once under
    outputs/uploads/. Standard jobs pass that local PathLike directly to
    videos.create. Batch jobs also upload the same local file to OpenAI Files and
    use input_reference.file_id in JSONL.
    """
    image_url = (item.get("input_reference_image_url") or "").strip()
    file_id = (item.get("input_reference_file_id") or "").strip()
    local_filename = (item.get("input_reference_local_filename") or "").strip()
    mime_type = (item.get("input_reference_mime_type") or "image/png").strip() or "image/png"

    if local_filename:
        item["input_reference_local_filename"] = local_filename
        item["input_reference_mime_type"] = mime_type
        item["input_reference_image_url"] = ""
        item["has_input_reference"] = True

        if not file_id:
            try:
                item["input_reference_file_id"] = upload_local_image_reference_to_openai_file(
                    local_filename,
                    item.get("input_reference_name") or "image_reference",
                    mime_type,
                )
                item["input_reference_upload_error"] = ""
            except Exception as exc:
                item["input_reference_file_id"] = ""
                item["input_reference_upload_error"] = str(exc)
        return item

    if image_url.startswith("data:image/"):
        saved = save_image_reference_upload(
            image_url,
            item.get("input_reference_name") or "image_reference",
            item.get("size") or "",
        )
        item.update(saved)
        item["input_reference_image_url"] = ""
        item["has_input_reference"] = True

        try:
            item["input_reference_file_id"] = upload_local_image_reference_to_openai_file(
                item["input_reference_local_filename"],
                item.get("input_reference_name") or saved.get("input_reference_name") or "image_reference",
                item.get("input_reference_mime_type") or "image/png",
            )
            item["input_reference_upload_error"] = ""
        except Exception as exc:
            # Standard generation can still work from the local file. Batch creation
            # will surface this error if it needs file_id and upload is unavailable.
            item["input_reference_file_id"] = ""
            item["input_reference_upload_error"] = str(exc)
        return item

    if file_id:
        item["input_reference_file_id"] = file_id
        item["input_reference_image_url"] = ""
        item["has_input_reference"] = True
        return item

    if image_url:
        # Remote image URL. This remains JSON-only and may work for Batch; standard
        # SDKs that require multipart may reject it, so the resulting error is shown.
        item["input_reference_file_id"] = ""
        item["has_input_reference"] = True
        return item

    item["input_reference_file_id"] = ""
    item["input_reference_local_filename"] = ""
    item["has_input_reference"] = bool(item.get("has_input_reference"))
    return item


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


def is_favorite(filename: str) -> bool:
    return filename in load_favorites()


def is_archived(filename: str) -> bool:
    return filename in load_archive()


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


def set_favorite(filename: str, favorite: bool):
    path = safe_output_path(filename)
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise FileNotFoundError("Video not found.")

    favorites = load_favorites()

    if favorite:
        favorites.add(filename)
    else:
        favorites.discard(filename)

    save_favorites(favorites)
    return filename in favorites


def set_archived(filename: str, archived: bool):
    path = safe_output_path(filename)
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise FileNotFoundError("Video not found.")

    archive = load_archive()

    if archived:
        archive.add(filename)
    else:
        archive.discard(filename)

    save_archive(archive)
    return filename in archive


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


def read_metadata(video_filename: str, state_sets=None, resolve_source: bool = True):
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
        "input_reference_name": "",
        "input_reference_local_filename": "",
        "has_input_reference": False,
        "pinned": False,
        "favorite": False,
        "archived": False,
        "viewed": False,
        "is_new": True,
        "thumbnail": "",
        "prompt_preview": "",
        "custom_tags": [],
    }

    if state_sets is None:
        state_sets = {
            "pins": load_pins(),
            "favorites": load_favorites(),
            "archive": load_archive(),
            "views": load_views(),
            "custom_tags": load_custom_tags(),
        }

    data["pinned"] = video_filename in state_sets.get("pins", set())
    data["favorite"] = video_filename in state_sets.get("favorites", set())
    data["archived"] = video_filename in state_sets.get("archive", set())
    data["viewed"] = video_filename in state_sets.get("views", set())
    data["is_new"] = not data["viewed"]
    data["custom_tags"] = tags_for_filename(video_filename, state_sets.get("custom_tags", {}))

    if not metadata_path.exists():
        data["thumbnail"] = cached_thumbnail(video_filename)
        data["prompt_preview"] = ""
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
        elif line.startswith("input_reference_name:"):
            data["input_reference_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("input_reference_local_filename:"):
            data["input_reference_local_filename"] = line.split(":", 1)[1].strip()
        elif line.startswith("has_input_reference:"):
            data["has_input_reference"] = line.split(":", 1)[1].strip().lower() == "true"

    data["prompt"] = "\n".join(prompt_lines).strip()

    if resolve_source and data["source_video_id"]:
        source_path = OUTPUT_DIR / data["source_filename"] if data["source_filename"] else None

        if not data["source_filename"] or not source_path.exists():
            found = find_filename_by_video_id(data["source_video_id"])
            data["source_filename"] = found or data["source_filename"] or ""

    data["thumbnail"] = cached_thumbnail(video_filename)
    data["prompt_preview"] = data.get("prompt", "")[:350]

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
    input_reference_name: str = "",
    input_reference_local_filename: str = "",
    has_input_reference: bool = False,
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
        f"batch_custom_id: {batch_custom_id}\n"
        f"input_reference_name: {input_reference_name}\n"
        f"input_reference_local_filename: {input_reference_local_filename}\n"
        f"has_input_reference: {str(bool(has_input_reference)).lower()}\n\n"
        f"prompt:\n{prompt}\n",
        encoding="utf-8",
    )

    # Keep the persistent gallery index current immediately for newly downloaded
    # standard, remix, extend, and Batch outputs. Without this, the large-library
    # cache can make new videos invisible until the next full background rescan.
    try:
        upsert_video_in_index(output_path.name)
    except Exception:
        pass


def get_outputs_signature():
    """Identify file/metadata changes without touching thumbnails or UI state files.

    This is deliberately limited to .mp4 files and their .txt sidecars. Pins,
    favorites, archive, viewed status, and thumbnail availability are applied at
    request time, so they should not force a full metadata re-index.
    """
    parts = []

    for pattern in ("*.mp4", "*.txt"):
        for path in OUTPUT_DIR.glob(pattern):
            try:
                stat = path.stat()
                parts.append((path.name, stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                continue

    return tuple(sorted(parts))


def runtime_state_sets():
    return {
        "pins": load_pins(),
        "favorites": load_favorites(),
        "archive": load_archive(),
        "views": load_views(),
        "custom_tags": load_custom_tags(),
    }


def apply_runtime_video_state(video: dict, state_sets=None):
    state_sets = state_sets or runtime_state_sets()
    filename = video.get("filename") or ""
    enriched = dict(video)
    enriched["pinned"] = filename in state_sets.get("pins", set())
    enriched["favorite"] = filename in state_sets.get("favorites", set())
    enriched["archived"] = filename in state_sets.get("archive", set())
    enriched["viewed"] = filename in state_sets.get("views", set())
    enriched["is_new"] = not enriched["viewed"]
    enriched["thumbnail"] = cached_thumbnail(filename)
    enriched["custom_tags"] = tags_for_filename(filename, state_sets.get("custom_tags", {}))
    return enriched


def load_persisted_video_index():
    if not GALLERY_INDEX_FILE.exists():
        return None

    try:
        data = json.loads(GALLERY_INDEX_FILE.read_text(encoding="utf-8"))
        videos = data.get("videos", [])
        if not isinstance(videos, list):
            return None
        return {
            "signature": data.get("signature"),
            "videos": [video for video in videos if isinstance(video, dict)],
            "built_at": data.get("built_at"),
        }
    except Exception:
        return None


def save_persisted_video_index(signature, videos, built_at):
    tmp_path = GALLERY_INDEX_FILE.with_suffix(".tmp")
    payload = {
        "version": 2,
        "signature": signature,
        "built_at": built_at,
        "videos": videos,
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(GALLERY_INDEX_FILE)


def load_video_index_from_disk_once():
    with video_index_lock:
        if video_index_cache.get("loaded_from_disk"):
            return
        video_index_cache["loaded_from_disk"] = True

    persisted = load_persisted_video_index()
    if not persisted:
        return

    with video_index_lock:
        if video_index_cache.get("videos"):
            return
        video_index_cache["signature"] = persisted.get("signature")
        video_index_cache["videos"] = persisted.get("videos") or []
        video_index_cache["built_at"] = persisted.get("built_at")
        video_index_cache["dirty"] = False
        video_index_cache["last_error"] = ""


def start_gallery_index_refresh(force: bool = False):
    with video_index_lock:
        if video_index_cache.get("refresh_running"):
            return False
        video_index_cache["refresh_running"] = True
        video_index_cache["last_error"] = ""
        old_signature = video_index_cache.get("signature")
        is_dirty = bool(video_index_cache.get("dirty", True))

    def worker():
        try:
            signature = get_outputs_signature()
            should_rebuild = force or is_dirty or old_signature != signature
            if should_rebuild:
                videos = build_video_index()
                built_at = datetime.now().isoformat(timespec="seconds")
                save_persisted_video_index(signature, videos, built_at)
                with video_index_lock:
                    video_index_cache["signature"] = signature
                    video_index_cache["videos"] = videos
                    video_index_cache["built_at"] = built_at
                    video_index_cache["dirty"] = False
                    video_index_cache["last_disk_check"] = time.monotonic()
                    video_index_cache["last_error"] = ""
            else:
                with video_index_lock:
                    video_index_cache["dirty"] = False
                    video_index_cache["last_disk_check"] = time.monotonic()
                    video_index_cache["last_error"] = ""
        except Exception as exc:
            with video_index_lock:
                video_index_cache["last_error"] = str(exc)
        finally:
            with video_index_lock:
                video_index_cache["refresh_running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return True

def build_video_index():
    empty_state_sets = {"pins": set(), "favorites": set(), "archive": set(), "views": set(), "custom_tags": {}}
    values = []

    for path in OUTPUT_DIR.glob("*.mp4"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        try:
            video = read_metadata(path.name, state_sets=empty_state_sets, resolve_source=False)
        except Exception:
            continue

        prompt = video.get("prompt", "")
        video["prompt_preview"] = prompt[:350]
        video["mtime"] = stat.st_mtime
        video["_search_text"] = " ".join([
            video.get("filename") or "",
            prompt,
            video.get("prompt_preview") or "",
            video.get("source_filename") or "",
            video.get("preset_name") or "",
            video.get("input_reference_name") or "",
            video.get("input_reference_local_filename") or "",
        ]).lower()
        video.pop("prompt", None)
        values.append(video)

    id_to_filename = {
        video.get("video_id"): video.get("filename")
        for video in values
        if video.get("video_id") and video.get("filename")
    }
    for video in values:
        if video.get("source_video_id") and not video.get("source_filename"):
            video["source_filename"] = id_to_filename.get(video.get("source_video_id"), "")
            if video.get("source_filename"):
                video["_search_text"] = f'{video.get("_search_text", "")} {video["source_filename"]}'.lower()

    values.sort(key=lambda video: video.get("mtime", 0), reverse=True)

    return values


def make_video_index_record(video_filename: str):
    """Create one gallery-index row for a newly saved output without scanning the whole library."""
    try:
        path = safe_output_path(video_filename)
        stat = path.stat()
    except Exception:
        return None

    empty_state_sets = {"pins": set(), "favorites": set(), "archive": set(), "views": set(), "custom_tags": {}}

    try:
        video = read_metadata(video_filename, state_sets=empty_state_sets, resolve_source=True)
    except Exception:
        return None

    prompt = video.get("prompt", "")
    video["prompt_preview"] = prompt[:350]
    video["mtime"] = stat.st_mtime
    video["_search_text"] = " ".join([
        video.get("filename") or "",
        prompt,
        video.get("prompt_preview") or "",
        video.get("source_filename") or "",
        video.get("preset_name") or "",
        video.get("input_reference_name") or "",
        video.get("input_reference_local_filename") or "",
    ]).lower()
    video.pop("prompt", None)
    return video


def upsert_video_in_index(video_filename: str):
    """Make a newly created/downloaded video visible immediately.

    The large-library index normally refreshes in the background so search/page
    switches stay fast with thousands of videos. New Sora outputs should not
    wait for the next full disk rescan, so this surgically inserts or replaces
    one row in the in-memory and persisted gallery index.
    """
    record = make_video_index_record(video_filename)
    if not record:
        return False

    load_video_index_from_disk_once()

    with video_index_lock:
        videos = list(video_index_cache.get("videos") or [])
        videos = [video for video in videos if video.get("filename") != video_filename]
        videos.append(record)
        videos.sort(key=lambda video: video.get("mtime", 0), reverse=True)

        built_at = datetime.now().isoformat(timespec="seconds")
        video_index_cache["videos"] = videos
        video_index_cache["built_at"] = built_at
        video_index_cache["dirty"] = False
        video_index_cache["last_disk_check"] = time.monotonic()
        video_index_cache["last_error"] = ""
        signature = video_index_cache.get("signature")

    try:
        save_persisted_video_index(signature, videos, built_at)
    except Exception as exc:
        with video_index_lock:
            video_index_cache["last_error"] = str(exc)

    return True


def invalidate_video_index(files_changed: bool = True):
    # UI state changes are applied dynamically. Real file/metadata changes mark
    # the persistent gallery index stale and refresh it in the background rather
    # than blocking page loads/search/filter switches.
    with video_index_lock:
        if files_changed:
            video_index_cache["dirty"] = True
            video_index_cache["signature"] = None
    if files_changed:
        start_gallery_index_refresh(force=True)


def get_video_index(force_disk_check: bool = False):
    load_video_index_from_disk_once()
    now = time.monotonic()

    with video_index_lock:
        has_cache = bool(video_index_cache.get("videos"))
        should_refresh = (
            force_disk_check
            or video_index_cache.get("dirty", True)
            or video_index_cache.get("signature") is None
            or (now - float(video_index_cache.get("last_disk_check") or 0)) >= GALLERY_DISK_RESCAN_SECONDS
        )
        if should_refresh and not video_index_cache.get("refresh_running"):
            # Do not scan 5,000+ metadata files inside the request path. Return
            # the current persisted/in-memory index immediately and refresh in
            # the background.
            start_needed = True
        else:
            start_needed = False
        base_videos = list(video_index_cache.get("videos", []))

    if start_needed:
        start_gallery_index_refresh(force=force_disk_check or not has_cache)

    state_sets = runtime_state_sets()
    return [apply_runtime_video_state(video, state_sets) for video in base_videos]


def gallery_index_status():
    load_video_index_from_disk_once()
    with video_index_lock:
        return {
            "ready": bool(video_index_cache.get("videos")),
            "indexing": bool(video_index_cache.get("refresh_running")),
            "count": len(video_index_cache.get("videos") or []),
            "built_at": video_index_cache.get("built_at"),
            "error": video_index_cache.get("last_error") or "",
        }


def get_local_videos(include_full_prompt: bool = False):
    if not include_full_prompt:
        return get_video_index()

    videos = []
    state_sets = {
        "pins": load_pins(),
        "favorites": load_favorites(),
        "archive": load_archive(),
        "views": load_views(),
    }

    for path in OUTPUT_DIR.glob("*.mp4"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        video = read_metadata(path.name, state_sets=state_sets)
        video["prompt_preview"] = video.get("prompt", "")[:350]
        video["mtime"] = stat.st_mtime
        videos.append(video)

    videos.sort(
        key=lambda video: (
            1 if video.get("pinned") else 0,
            video.get("mtime", 0),
        ),
        reverse=True,
    )

    return videos


def get_recent_jobs():
    with jobs_lock:
        copied_jobs = list(jobs.values())

    copied_jobs.sort(key=lambda job: job.get("created_at") or "", reverse=True)
    return copied_jobs




def object_from_json(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: object_from_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return [object_from_json(item) for item in value]
    return value


def normalize_extension_seconds(seconds) -> str:
    """Validate the number of seconds requested for an extension segment."""
    value = str(seconds or "4").strip() or "4"
    allowed = {"4", "8", "12", "16", "20"}
    if value not in allowed:
        raise ValueError("Extension seconds must be one of: 4, 8, 12, 16, or 20.")
    return value


def create_video_extension(source_video_id: str, prompt: str, seconds: str):
    """Create a Sora video extension using the stable REST endpoint.

    The supported OpenAI endpoint is POST /v1/videos/extensions with a JSON
    body shaped as {"prompt": ..., "seconds": ..., "video": {"id": ...}}.
    Calling the REST endpoint directly avoids SDK-version drift such as
    openai-python builds that do not expose client.videos.extend(...) yet, or
    older local builds where client.videos.extensions is missing.
    """
    source_video_id = (source_video_id or "").strip()
    prompt = (prompt or "").strip()
    seconds = normalize_extension_seconds(seconds)

    if not source_video_id:
        raise ValueError("Missing source video ID for extend.")
    if not source_video_id.startswith("video_"):
        raise ValueError(f"Invalid source video ID for extend: {source_video_id}")
    if not prompt:
        raise ValueError("Missing prompt for extend.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = str(getattr(client, "base_url", "https://api.openai.com/v1")).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    url = f"{base_url}/videos/extensions"
    payload = {
        "prompt": prompt,
        "seconds": seconds,
        "video": {
            "id": source_video_id,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    organization = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
    project = os.environ.get("OPENAI_PROJECT_ID") or os.environ.get("OPENAI_PROJECT")
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project

    request_obj = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            response_text = response.read().decode("utf-8")
            response_payload = json.loads(response_text)
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Extend request failed. The app sent POST /v1/videos/extensions with "
            f"source video {source_video_id} and {seconds}s. OpenAI returned HTTP {exc.code}: {error_text}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Extend request failed before OpenAI returned a video job: {exc}"
        ) from exc

    video = object_from_json(response_payload)
    if not getattr(video, "id", None):
        raise RuntimeError(
            "Extend request returned without a video id. Raw response: "
            + json.dumps(response_payload, ensure_ascii=False)
        )

    return video

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

        return create_video_extension(
            source_video_id=source_video_id,
            prompt=prompt,
            seconds=str(item.get("seconds", "4")),
        )

    create_kwargs = {
        "model": item["model"],
        "prompt": prompt,
        "seconds": str(item["seconds"]),
        "size": item["size"],
    }

    input_reference_local_filename = (item.get("input_reference_local_filename") or "").strip()
    input_reference_file_id = (item.get("input_reference_file_id") or "").strip()
    input_reference_image_url = (item.get("input_reference_image_url") or "").strip()

    if input_reference_local_filename:
        local_path = safe_upload_path(input_reference_local_filename)
        if not local_path.exists():
            raise FileNotFoundError(f"Image reference file not found: {input_reference_local_filename}")
        # openai-python video create accepts multipart FileTypes here. PathLike is
        # the most compatible form for standard image-guided generation.
        create_kwargs["input_reference"] = local_path
    elif input_reference_image_url:
        create_kwargs["input_reference"] = {
            "image_url": input_reference_image_url,
        }
    elif input_reference_file_id:
        # Some SDK versions accept ImageInputReferenceParam here, while others only
        # accept multipart FileTypes. Keep this as a last resort for older persisted
        # jobs that have only file_id and no local upload cache.
        create_kwargs["input_reference"] = {
            "file_id": input_reference_file_id,
        }

    try:
        return client.videos.create(**create_kwargs)
    except TypeError as exc:
        if "input_reference" in create_kwargs and input_reference_file_id and not input_reference_local_filename:
            raise TypeError(
                "This openai-python version requires the original local image file for standard image-guided generation. "
                "Create a new job with the image attached, or send it through real Batch so the saved file_id can be used in JSONL. "
                f"Original error: {exc}"
            )
        raise


def run_generation_job(job_id, items):
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")
        save_jobs_snapshot_unlocked()

    for index, item in enumerate(items):
        prompt = item["prompt"].strip()
        model = item.get("model", "sora-2")
        seconds = str(item.get("seconds", "4"))
        size = item.get("size", "720x1280")
        mode = item.get("mode", "generate")
        source_video_id = item.get("source_video_id") or ""
        source_filename = item.get("source_filename") or ""
        input_reference_name = item.get("input_reference_name") or ""
        has_input_reference = bool(item.get("input_reference_file_id") or item.get("input_reference_image_url") or item.get("input_reference_local_filename") or item.get("has_input_reference"))

        if source_video_id and not source_filename:
            source_filename = find_filename_by_video_id(source_video_id) or ""

        try:
            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "creating",
                    "progress": 0,
                    "source_video_id": source_video_id,
                    "source_filename": source_filename,
                    "input_reference_name": input_reference_name,
                    "has_input_reference": has_input_reference,
                })
                save_jobs_snapshot_unlocked()

            video = create_or_continue_video(item)

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "queued",
                    "video_id": video.id,
                    "progress": 0,
                })
                save_jobs_snapshot_unlocked()

            while True:
                video = client.videos.retrieve(video.id)
                progress = getattr(video, "progress", None)
                progress = progress if progress is not None else 0

                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": video.status,
                        "progress": progress,
                    })
                    save_jobs_snapshot_unlocked()

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
                input_reference_name=input_reference_name,
                input_reference_local_filename=item.get("input_reference_local_filename") or "",
                has_input_reference=has_input_reference,
            )
            generate_thumbnail_once(output_path.name)

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "completed",
                    "progress": 100,
                    "filename": output_path.name,
                    "video_id": video.id,
                    "source_video_id": source_video_id,
                    "source_filename": source_filename,
                    "input_reference_name": input_reference_name,
                    "has_input_reference": has_input_reference,
                })
                save_jobs_snapshot_unlocked()

        except Exception as e:
            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "failed",
                    "error": str(e),
                })
                save_jobs_snapshot_unlocked()

    with jobs_lock:
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_jobs_snapshot_unlocked()




def run_resume_standard_job(job_id):
    """
    Resume or retry a standard/remix/extend local job after the Flask server was restarted.

    If an item already has a persisted video_id, this reconnects to that OpenAI video job,
    polls it, downloads the mp4, writes metadata, and marks the item completed.

    If the server restarted before the video_id was saved, there is no remote handle to
    reconnect to. In that case this safely retries that item from the saved prompt/settings.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return

        job["status"] = "resuming"
        job["error"] = None
        job["started_at"] = job.get("started_at") or datetime.now().isoformat(timespec="seconds")
        items = job.get("items", [])

        for item in items:
            if item.get("filename"):
                item["status"] = "completed"
                item["progress"] = 100
            elif item.get("status") in ("interrupted", "failed"):
                item["status"] = "queued_resume"
                item["error"] = None

        save_jobs_snapshot_unlocked()

    try:
        for index, item in enumerate(items):
            if item.get("filename"):
                continue

            prompt = (item.get("prompt") or "").strip()
            if not prompt:
                with jobs_lock:
                    item["status"] = "failed"
                    item["error"] = "Cannot resume item because its prompt is missing."
                    item["progress"] = 100
                    save_jobs_snapshot_unlocked()
                continue

            model = item.get("model", "sora-2")
            seconds = str(item.get("seconds", "4"))
            size = item.get("size", "720x1280")
            mode = item.get("mode", "generate")
            source_video_id = item.get("source_video_id") or ""
            source_filename = item.get("source_filename") or ""
            input_reference_name = item.get("input_reference_name") or ""
            has_input_reference = bool(item.get("input_reference_image_url") or item.get("input_reference_local_filename") or item.get("has_input_reference"))

            if source_video_id and not source_filename:
                source_filename = find_filename_by_video_id(source_video_id) or ""

            try:
                video_id = item.get("video_id") or ""

                with jobs_lock:
                    item.update({
                        "status": "resuming" if video_id else "retrying",
                        "progress": item.get("progress") or 0,
                        "error": None,
                        "source_video_id": source_video_id,
                        "source_filename": source_filename,
                    })
                    save_jobs_snapshot_unlocked()

                if video_id:
                    video = client.videos.retrieve(video_id)
                else:
                    # No persisted video_id means the original local worker died before it
                    # saved the remote handle. Retrying is the only recoverable path.
                    video = create_or_continue_video(item)
                    video_id = video.id
                    with jobs_lock:
                        item.update({
                            "status": "queued",
                            "video_id": video_id,
                            "progress": 0,
                        })
                        save_jobs_snapshot_unlocked()

                while True:
                    video = client.videos.retrieve(video_id)
                    video_status = getattr(video, "status", "unknown")
                    progress = getattr(video, "progress", None)
                    progress = progress if progress is not None else 0

                    with jobs_lock:
                        item.update({
                            "status": video_status,
                            "progress": progress,
                            "video_id": video_id,
                        })
                        save_jobs_snapshot_unlocked()

                    if video_status == "completed":
                        break

                    if video_status == "failed":
                        error = getattr(video, "error", None)
                        message = getattr(error, "message", "Unknown video generation error")
                        raise RuntimeError(message)

                    time.sleep(5)

                existing_filename = find_filename_by_video_id(video_id)
                if existing_filename:
                    with jobs_lock:
                        item.update({
                            "status": "completed",
                            "progress": 100,
                            "filename": existing_filename,
                            "video_id": video_id,
                        })
                        save_jobs_snapshot_unlocked()
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = mode if mode in ("remix", "extend") else "sora"
                output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{video_id}.mp4"

                content = client.videos.download_content(video_id, variant="video")
                content.write_to_file(str(output_path))

                save_rich_metadata(
                    video_id=video_id,
                    output_path=output_path,
                    prompt=prompt,
                    model=model,
                    seconds=seconds,
                    size=size,
                    source_mode=mode if mode in ("remix", "extend") else "",
                    source_video_id=source_video_id,
                    source_filename=source_filename,
                    input_reference_name=input_reference_name,
                    input_reference_local_filename=item.get("input_reference_local_filename") or "",
                    has_input_reference=has_input_reference,
                )
                generate_thumbnail_once(output_path.name)

                with jobs_lock:
                    item.update({
                        "status": "completed",
                        "progress": 100,
                        "filename": output_path.name,
                        "video_id": video_id,
                        "source_video_id": source_video_id,
                        "source_filename": source_filename,
                    })
                    save_jobs_snapshot_unlocked()

            except Exception as exc:
                with jobs_lock:
                    item.update({
                        "status": "failed",
                        "progress": 100,
                        "error": str(exc),
                    })
                    save_jobs_snapshot_unlocked()

        with jobs_lock:
            failed_count = sum(1 for item in jobs[job_id].get("items", []) if item.get("status") == "failed")
            jobs[job_id]["status"] = "completed" if failed_count == 0 else "completed_with_errors"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
            save_jobs_snapshot_unlocked()

    except Exception as exc:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(exc)
                jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
                save_jobs_snapshot_unlocked()


def format_openai_error_object(error) -> str:
    if not error:
        return ""

    if isinstance(error, dict):
        parts = []
        for key in ("message", "code", "type", "param"):
            value = error.get(key)
            if value:
                parts.append(f"{key}: {value}")
        return " | ".join(parts) if parts else json.dumps(error, ensure_ascii=False)

    return str(error)


def batch_row_message(row: dict) -> str:
    error = row.get("error")
    response = row.get("response")

    direct_error = format_openai_error_object(error)
    if direct_error:
        return direct_error

    if isinstance(response, dict):
        body = response.get("body")
        status_code = response.get("status_code")

        if isinstance(body, dict):
            nested_error = format_openai_error_object(body.get("error"))
            if nested_error:
                return f"status_code: {status_code} | {nested_error}" if status_code else nested_error

            if body.get("message"):
                return f"status_code: {status_code} | message: {body.get('message')}" if status_code else str(body.get("message"))

        return json.dumps(response, ensure_ascii=False)

    if response:
        return str(response)

    return "OpenAI batch request failed."


def read_batch_error_rows(error_file_id):
    if not error_file_id:
        return [], ""

    try:
        error_text = read_file_content_text(error_file_id)
        return parse_batch_jsonl_text(error_text), ""
    except Exception as exc:
        return [], f"Could not read error file {error_file_id}: {exc}"


def batch_error_rows_by_index(items, rows):
    errors = {}

    for row in rows:
        custom_id = row.get("custom_id", "")
        match = re.match(r"item-(\d+)$", custom_id)
        if not match:
            continue

        index = int(match.group(1))
        if index < 0 or index >= len(items):
            continue

        errors[index] = {
            "custom_id": custom_id,
            "message": batch_row_message(row),
        }

    return errors


def apply_batch_error_file(job_id, items, error_file_id, fallback_message):
    rows, read_error = read_batch_error_rows(error_file_id)
    if read_error:
        fallback_message = f"{fallback_message} {read_error}"

    errors_by_index = batch_error_rows_by_index(items, rows)

    with jobs_lock:
        for index, item in enumerate(jobs[job_id]["items"]):
            if item.get("filename"):
                item.update({
                    "status": "completed",
                    "progress": 100,
                    "error": None,
                })
                continue

            item_error = errors_by_index.get(index)
            item.update({
                "status": "failed",
                "error": item_error["message"] if item_error else fallback_message,
                "progress": 100,
            })
            if item_error:
                item["batch_custom_id"] = item_error["custom_id"]

        jobs[job_id]["status"] = "completed_with_errors"
        jobs[job_id]["error"] = fallback_message
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_jobs_snapshot_unlocked()


def wait_for_batch_video_to_finish(job_id, index, video_id):
    """
    A completed OpenAI Batch request can return a video job ID before the video
    binary is ready for download. Poll the returned video job the same way the
    standard path does, then return the completed video object.
    """
    while True:
        video = client.videos.retrieve(video_id)
        video_status = getattr(video, "status", "unknown")
        progress = getattr(video, "progress", None)
        progress = progress if progress is not None else 0

        with jobs_lock:
            jobs[job_id]["items"][index].update({
                "status": video_status,
                "progress": progress,
                "video_id": video_id,
            })
            save_jobs_snapshot_unlocked()

        if video_status == "completed":
            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "ready_to_download",
                    "progress": 100,
                    "video_id": video_id,
                })
                save_jobs_snapshot_unlocked()
            return video

        if video_status == "failed":
            error = getattr(video, "error", None)
            message = getattr(error, "message", "Unknown video generation error")
            raise RuntimeError(message)

        time.sleep(5)


def run_openai_batch_job(job_id, items, existing_batch_id=None):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "resuming_batch" if existing_batch_id else "uploading_batch"
            jobs[job_id]["started_at"] = jobs[job_id].get("started_at") or datetime.now().isoformat(timespec="seconds")
            jobs[job_id]["error"] = None

            for item in jobs[job_id]["items"]:
                # Keep already downloaded items. Reset old broken rows that were
                # marked completed only because the Batch request completed, but
                # never received a local output filename.
                if item.get("status") == "completed" and item.get("filename"):
                    continue

                item["status"] = "waiting_for_batch"
                item["progress"] = item.get("progress") or 0
                item["error"] = None

            save_jobs_snapshot_unlocked()

        if existing_batch_id:
            batch = retrieve_openai_batch(existing_batch_id)
            batch_input_file = None
            jsonl_path = jobs[job_id].get("batch_jsonl") or ""
        else:
            with jobs_lock:
                image_reference_mode = jobs.get(job_id, {}).get("batch_image_reference_mode") or "file_id"
            batch, batch_input_file, jsonl_path = create_openai_video_batch(items, job_id, image_reference_mode=image_reference_mode)

        with jobs_lock:
            update_data = {
                "status": batch.status,
                "batch_id": batch.id,
                "batch_jsonl": str(jsonl_path),
                "progress": batch_progress_percent(batch),
                "request_counts": get_batch_counts(batch),
            }
            if batch_input_file is not None:
                update_data["batch_input_file_id"] = batch_input_file.id
            jobs[job_id].update(update_data)
            save_jobs_snapshot_unlocked()

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

                for item in jobs[job_id]["items"]:
                    if item.get("status") == "completed" and item.get("filename"):
                        continue
                    if item.get("status") not in ("failed",):
                        item["status"] = batch.status
                        item["progress"] = progress

                save_jobs_snapshot_unlocked()

            if batch.status in TERMINAL_BATCH_STATUSES:
                break

            time.sleep(20)

        error_file_id = getattr(batch, "error_file_id", None)

        if batch.status != "completed":
            apply_batch_error_file(
                job_id,
                items,
                error_file_id,
                f"OpenAI batch ended with status: {batch.status}.",
            )
            return

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            apply_batch_error_file(
                job_id,
                items,
                error_file_id,
                "OpenAI batch completed without an output file ID. The requests likely failed validation before any video jobs were created.",
            )
            return

        with jobs_lock:
            jobs[job_id]["status"] = "processing_batch_output"
            jobs[job_id]["progress"] = 100
            for item in jobs[job_id]["items"]:
                if item.get("status") == "completed" and item.get("filename"):
                    continue
                if item.get("status") not in ("failed",):
                    item["status"] = "processing_batch_output"
                    item["progress"] = 100
            save_jobs_snapshot_unlocked()

        output_text = read_file_content_text(output_file_id)
        rows = parse_batch_jsonl_text(output_text)

        # In partial-failure batches, OpenAI can provide both an output file
        # for successful requests and an error file for failed requests. Process
        # the successful output rows first so one bad request never prevents
        # completed videos from being downloaded. Then attach per-item errors
        # from the error file to only the failed items.
        error_rows, error_file_read_error = read_batch_error_rows(error_file_id)
        errors_by_index = batch_error_rows_by_index(items, error_rows)
        seen_indexes = set()

        for row in rows:
            custom_id = row.get("custom_id", "")
            match = re.match(r"item-(\d+)$", custom_id)
            if not match:
                continue

            index = int(match.group(1))
            if index < 0 or index >= len(items):
                continue

            seen_indexes.add(index)
            item = items[index]

            existing_filename = jobs[job_id]["items"][index].get("filename")
            if existing_filename:
                try:
                    if safe_output_path(existing_filename).exists():
                        with jobs_lock:
                            jobs[job_id]["items"][index]["status"] = "completed"
                            jobs[job_id]["items"][index]["progress"] = 100
                            save_jobs_snapshot_unlocked()
                        continue
                except Exception:
                    pass

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
                    save_jobs_snapshot_unlocked()
                continue

            video_id = extract_video_id_from_batch_row(row)
            if not video_id:
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "failed",
                        "error": "Batch output row did not include a video ID.",
                        "progress": 100,
                    })
                    save_jobs_snapshot_unlocked()
                continue

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "waiting_for_video",
                    "progress": 0,
                    "video_id": video_id,
                    "batch_custom_id": custom_id,
                })
                save_jobs_snapshot_unlocked()

            try:
                completed_video = wait_for_batch_video_to_finish(job_id, index, video_id)
            except Exception as video_error:
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "failed",
                        "error": f"Video job failed after batch completion: {video_error}",
                        "progress": 100,
                    })
                    save_jobs_snapshot_unlocked()
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"batch_{timestamp}_{video_id}.mp4"

            try:
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "downloading_video",
                        "progress": 100,
                        "video_id": video_id,
                    })
                    save_jobs_snapshot_unlocked()

                content = client.videos.download_content(completed_video.id, variant="video")
                content.write_to_file(str(output_path))
            except Exception as download_error:
                with jobs_lock:
                    jobs[job_id]["items"][index].update({
                        "status": "failed",
                        "error": f"Video completed, but local download failed: {download_error}",
                        "progress": 100,
                    })
                    save_jobs_snapshot_unlocked()
                continue

            save_rich_metadata(
                video_id=video_id,
                output_path=output_path,
                prompt=item["prompt"].strip(),
                model=item.get("model", "sora-2"),
                seconds=str(item.get("seconds", "4")),
                size=item.get("size", "720x1280"),
                batch_id=batch.id,
                batch_custom_id=custom_id,
                input_reference_name=item.get("input_reference_name") or "",
                input_reference_local_filename=item.get("input_reference_local_filename") or "",
                has_input_reference=bool(item.get("input_reference_file_id") or item.get("input_reference_image_url") or item.get("input_reference_local_filename") or item.get("has_input_reference")),
            )
            generate_thumbnail_once(output_path.name)

            with jobs_lock:
                jobs[job_id]["items"][index].update({
                    "status": "completed",
                    "progress": 100,
                    "filename": output_path.name,
                    "video_id": video_id,
                    "batch_custom_id": custom_id,
                    "input_reference_name": item.get("input_reference_name") or "",
                    "has_input_reference": bool(item.get("input_reference_file_id") or item.get("input_reference_image_url") or item.get("input_reference_local_filename") or item.get("has_input_reference")),
                    "error": None,
                })
                save_jobs_snapshot_unlocked()

        with jobs_lock:
            for index, item in enumerate(jobs[job_id]["items"]):
                if item.get("filename"):
                    item.update({
                        "status": "completed",
                        "progress": 100,
                        "error": None,
                    })
                    continue

                if index in errors_by_index:
                    item_error = errors_by_index[index]
                    item.update({
                        "status": "failed",
                        "error": item_error["message"],
                        "progress": 100,
                        "batch_custom_id": item_error["custom_id"],
                    })
                    continue

                if index not in seen_indexes and item.get("status") != "failed":
                    message = "No matching row was found in the OpenAI batch output file."
                    if error_file_read_error:
                        message = f"{message} {error_file_read_error}"
                    item.update({
                        "status": "failed",
                        "error": message,
                        "progress": 100,
                    })

            failed_count = sum(1 for item in jobs[job_id]["items"] if item.get("status") == "failed")
            jobs[job_id]["status"] = "completed" if failed_count == 0 else "completed_with_errors"
            jobs[job_id]["error"] = None if failed_count == 0 else "One or more batch items failed, but completed items were still fetched."
            jobs[job_id]["progress"] = 100
            jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
            save_jobs_snapshot_unlocked()

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
            for item in jobs[job_id]["items"]:
                if not item.get("filename"):
                    item["status"] = "failed"
                    item["error"] = str(e)
                    item["progress"] = 100
            save_jobs_snapshot_unlocked()

def clean_item_from_request(item):
    raw_prompt = item.get("raw_prompt") or item.get("prompt", "")
    raw_prompt = str(raw_prompt or "").strip()
    if not raw_prompt:
        return None

    preset_id = str(item.get("preset_id") or "").strip()
    preset_name = str(item.get("preset_name") or "").strip()
    preset_template = str(item.get("preset_template") or "").strip()

    # Trust the frozen preset template sent with a queued item. If only an ID is
    # supplied, resolve it from the current preset file.
    if preset_id and not preset_template:
        for preset in load_presets():
            if preset.get("id") == preset_id:
                preset_name = preset.get("name") or preset_name
                preset_template = preset.get("template") or ""
                break

    prompt = apply_prompt_preset(raw_prompt, preset_template)

    mode = item.get("mode", "generate")
    source_video_id = item.get("source_video_id") or ""
    source_filename = item.get("source_filename") or ""

    if source_video_id and not source_filename:
        source_filename = find_filename_by_video_id(source_video_id) or ""

    clean_item = {
        "prompt": prompt,
        "raw_prompt": raw_prompt,
        "preset_id": preset_id,
        "preset_name": preset_name,
        "preset_template": preset_template,
        "model": item.get("model", "sora-2"),
        "seconds": str(item.get("seconds", "4")).strip(),
        "size": item.get("size", "720x1280"),
        "mode": mode,
        "source_video_id": source_video_id,
        "source_filename": source_filename,
        "input_reference_image_url": item.get("input_reference_image_url") or "",
        "input_reference_file_id": item.get("input_reference_file_id") or "",
        "input_reference_local_filename": item.get("input_reference_local_filename") or "",
        "input_reference_mime_type": item.get("input_reference_mime_type") or "",
        "input_reference_name": item.get("input_reference_name") or "",
        "input_reference_size": item.get("input_reference_size") or 0,
        "input_reference_upload_error": item.get("input_reference_upload_error") or "",
        "has_input_reference": bool(item.get("input_reference_image_url") or item.get("input_reference_file_id") or item.get("input_reference_local_filename") or item.get("has_input_reference")),
        "status": "waiting",
        "progress": 0,
        "filename": None,
        "video_id": None,
        "error": None,
    }

    return prepare_input_reference_for_api(clean_item)




def clone_batch_item_for_retry(item):
    """
    Build a clean retry copy from a persisted OpenAI Batch item.
    The old job is left intact; retry creates a new local batch job for items
    that never produced a local output file.
    """
    retry_item = copy.deepcopy(item)

    retry_item["status"] = "waiting"
    retry_item["progress"] = 0
    retry_item["filename"] = None
    retry_item["video_id"] = None
    retry_item["batch_custom_id"] = ""
    retry_item["error"] = None

    # These fields belong to the previous OpenAI batch attempt, not the retry.
    retry_item.pop("output_file_id", None)
    retry_item.pop("error_file_id", None)

    # Older saved rows may only have raw_prompt, or only prompt. Preserve the
    # frozen preset wrapping behavior if available.
    raw_prompt = str(retry_item.get("raw_prompt") or "").strip()
    preset_template = str(retry_item.get("preset_template") or "").strip()
    prompt = str(retry_item.get("prompt") or "").strip()
    if raw_prompt:
        retry_item["prompt"] = apply_prompt_preset(raw_prompt, preset_template)
    elif prompt:
        retry_item["raw_prompt"] = prompt
        retry_item["prompt"] = prompt

    return retry_item

def create_local_job(items, job_type="standard", batch_image_reference_mode="file_id"):
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
            "batch_image_reference_mode": batch_image_reference_mode if job_type == "openai_batch" else "",
            "items": items,
        }
        save_jobs_snapshot_unlocked()

    return job_id



def prompt_matches_search(video: dict, query: str) -> bool:
    query = (query or "").strip().lower()
    if not query:
        return True

    haystack_parts = [
        video.get("prompt") or "",
        video.get("prompt_preview") or "",
        video.get("filename") or "",
        video.get("preset_name") or "",
        video.get("source_filename") or "",
        video.get("input_reference_name") or "",
    ]
    haystack = " ".join(str(part).lower() for part in haystack_parts if part)
    words = [word for word in re.split(r"\s+", query) if word]
    return all(word in haystack for word in words)


def start_thumbnail_cache_for_filenames(filenames):
    """Generate cached thumbnails only for the filenames currently needed by the UI.

    This avoids scanning/regenerating every output on each app start. Existing JPGs
    in outputs/thumbnails are reused forever unless manually deleted.
    """
    missing = []
    seen = set()

    for filename in filenames or []:
        if not filename or filename in seen:
            continue
        seen.add(filename)
        try:
            path = safe_output_path(filename)
        except ValueError:
            continue
        if not path.exists() or path.suffix.lower() != ".mp4":
            continue
        if not cached_thumbnail(filename):
            missing.append(filename)

    if not missing:
        return False

    def worker(values):
        with thumbnail_lock:
            thumbnail_state.update({
                "running": True,
                "total": len(values),
                "done": 0,
                "failed": 0,
                "generated": 0,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "completed_at": "",
                "mode": "page",
            })

        try:
            for filename in values:
                before = cached_thumbnail(filename)
                after = generate_thumbnail_once(filename)
                with thumbnail_lock:
                    thumbnail_state["done"] += 1
                    if after and not before:
                        thumbnail_state["generated"] += 1
                    elif not after:
                        thumbnail_state["failed"] += 1
        finally:
            with thumbnail_lock:
                thumbnail_state["running"] = False
                thumbnail_state["completed_at"] = datetime.now().isoformat(timespec="seconds")

    with thumbnail_lock:
        if thumbnail_state.get("running"):
            return False
        thumbnail_state["running"] = True

    threading.Thread(target=worker, args=(missing,), daemon=True).start()
    return True


def parse_gallery_filters(raw_filters: str = "", tab: str = "main"):
    filters = set()
    for value in re.split(r"[,\s]+", str(raw_filters or "")):
        value = value.strip().lower()
        if value:
            filters.add(value)

    legacy_tab = (tab or "main").strip().lower()
    if not filters and legacy_tab and legacy_tab != "main":
        filters.add(legacy_tab)

    aliases = {
        "favorite": "favorites",
        "fav": "favorites",
        "unwatched": "new",
        "remix": "remixes",
        "extension": "extensions",
        "extend": "extensions",
        "extends": "extensions",
        "archived": "archive",
    }
    return {aliases.get(value, value) for value in filters if value and value != "main"}


def filtered_videos_for_tab(tab: str, query: str = "", filters=None, tag: str = ""):
    active_filters = parse_gallery_filters(",".join(filters or []), tab) if filters is not None else parse_gallery_filters("", tab)
    tag = normalize_tag_name(tag)
    videos = get_video_index()

    values = []
    for video in videos:
        archived = bool(video.get("archived"))

        if "archive" in active_filters:
            if not archived:
                continue
        elif archived:
            continue

        if "favorites" in active_filters and not video.get("favorite"):
            continue
        if "new" in active_filters and not video.get("is_new"):
            continue
        if "remixes" in active_filters and video.get("source_mode") != "remix":
            continue
        if "extensions" in active_filters and video.get("source_mode") != "extend":
            continue
        if tag and tag not in (video.get("custom_tags") or []):
            continue

        values.append(video)

    query = (query or "").strip().lower()
    if query:
        words = [word for word in re.split(r"\s+", query) if word]
        values = [
            video for video in values
            if all(word in (video.get("_search_text", "") + " " + " ".join(video.get("custom_tags") or []).lower()) for word in words)
        ]

    values.sort(
        key=lambda video: (
            1 if video.get("pinned") else 0,
            video.get("mtime", 0),
        ),
        reverse=True,
    )

    return values


def child_videos_for_source(source_filename: str, query: str = ""):
    source_filename = (source_filename or "").strip()
    videos = get_video_index()

    try:
        source_video = read_metadata(source_filename)
        source_video_id = source_video.get("video_id") or extract_video_id(source_filename) or ""
    except Exception:
        source_video_id = extract_video_id(source_filename) or ""

    values = []
    for video in videos:
        if video.get("filename") == source_filename:
            continue

        is_child_by_filename = bool(source_filename and video.get("source_filename") == source_filename)
        is_child_by_id = bool(source_video_id and video.get("source_video_id") == source_video_id)
        is_derivative = video.get("source_mode") in {"remix", "extend"}

        if is_derivative and (is_child_by_filename or is_child_by_id):
            values.append(video)

    query = (query or "").strip().lower()
    if query:
        words = [word for word in re.split(r"\s+", query) if word]
        values = [
            video for video in values
            if all(word in video.get("_search_text", "") for word in words)
        ]

    values.sort(
        key=lambda video: (
            1 if video.get("pinned") else 0,
            video.get("mtime", 0),
        ),
        reverse=True,
    )

    return values


def public_gallery_video(video: dict):
    clean = dict(video)
    clean.pop("_search_text", None)
    clean.pop("prompt", None)
    return clean

def paginate_list(values, page=1, per_page=40):
    try:
        page = int(page)
    except Exception:
        page = 1

    try:
        per_page = int(per_page)
    except Exception:
        per_page = 40

    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    total = len(values)
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * per_page
    end = start + per_page
    return values[start:end], {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
    }


def job_has_missing_outputs(job: dict) -> bool:
    return any(isinstance(item, dict) and not item.get("filename") for item in job.get("items", []))


def job_has_error(job: dict) -> bool:
    if job.get("error"):
        return True
    if job.get("status") in {"failed", "completed_with_errors", "expired", "cancelled", "interrupted"}:
        return True
    return any(isinstance(item, dict) and (item.get("status") == "failed" or item.get("error")) for item in job.get("items", []))


def job_is_active(job: dict) -> bool:
    status = job.get("status") or ""
    if status in TERMINAL_JOB_STATUSES or status in {"completed_with_errors"}:
        return False
    return True


def notification_jobs():
    values = get_recent_jobs()
    return [job for job in values if job_is_active(job) or job_has_error(job)]


def auto_recover_jobs_once():
    started = []
    with jobs_lock:
        candidates = list(jobs.values())

    for job in candidates:
        job_id = job.get("job_id")
        if not job_id:
            continue
        if not job_has_missing_outputs(job):
            continue

        status = job.get("status")
        if status not in {"interrupted", "queued_resume", "failed", "completed_with_errors"}:
            continue

        if job.get("job_type") == "openai_batch" and job.get("batch_id"):
            with jobs_lock:
                live_job = jobs.get(job_id)
                if not live_job or live_job.get("status") not in {"interrupted", "failed", "completed_with_errors", "queued_resume"}:
                    continue
                live_job["status"] = "queued_resume"
                live_job["error"] = None
                save_jobs_snapshot_unlocked()
            thread = threading.Thread(target=run_openai_batch_job, args=(job_id, job.get("items", []), job.get("batch_id")), daemon=True)
            thread.start()
            started.append(job_id)
            continue

        if job.get("job_type") != "openai_batch":
            has_video_id = any(item.get("video_id") for item in job.get("items", []) if isinstance(item, dict) and not item.get("filename"))
            if has_video_id:
                with jobs_lock:
                    live_job = jobs.get(job_id)
                    if not live_job or live_job.get("status") not in {"interrupted", "failed", "completed_with_errors", "queued_resume"}:
                        continue
                    live_job["status"] = "queued_resume"
                    live_job["error"] = None
                    save_jobs_snapshot_unlocked()
                thread = threading.Thread(target=run_resume_standard_job, args=(job_id,), daemon=True)
                thread.start()
                started.append(job_id)

    return started


with jobs_lock:
    jobs.update(load_persisted_jobs())

# Load the last saved gallery index immediately and refresh it in the background.
# This keeps Flask startup and the first page load fast even with thousands of videos.
load_video_index_from_disk_once()
start_gallery_index_refresh(force=False)


@app.route("/")
def index():
    return render_template("index.html", videos=[])


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


@app.route("/api/presets")
def api_presets():
    return jsonify({"presets": load_presets()})


@app.route("/api/presets", methods=["POST"])
def api_save_preset():
    data = request.get_json(force=True)
    try:
        preset = upsert_preset(
            data.get("id") or "",
            data.get("name") or "",
            data.get("template") or "",
            bool(data.get("force_new")),
        )
        return jsonify({"preset": preset, "presets": load_presets()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/presets/<preset_id>", methods=["DELETE"])
def api_delete_preset(preset_id):
    deleted = delete_preset(preset_id)
    if not deleted:
        return jsonify({"error": "Preset not found."}), 404
    return jsonify({"deleted": True, "presets": load_presets()})


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

    batch_image_reference_mode = "base64" if data.get("batch_image_reference_mode") == "base64" or data.get("batch_use_base64_image_references") else "file_id"

    unsupported = [item for item in clean_items if item.get("mode") != "generate"]
    if unsupported:
        return jsonify({
            "error": "Real OpenAI Batch mode currently supports normal generate items only in this UI. Send remix/extend as standard jobs."
        }), 400

    if batch_image_reference_mode != "base64":
        missing_batch_refs = [item for item in clean_items if item.get("has_input_reference") and item.get("input_reference_local_filename") and not item.get("input_reference_file_id")]
        if missing_batch_refs:
            first_error = missing_batch_refs[0].get("input_reference_upload_error") or "OpenAI Files upload failed for the image reference."
            return jsonify({
                "error": "Image-guided OpenAI Batch requires the image to upload to OpenAI Files first. " + first_error
            }), 400

    models = set(item.get("model", "sora-2") for item in clean_items)
    if len(models) > 1:
        return jsonify({
            "error": "OpenAI Batch input files for this UI are restricted to one model at a time. Use one model per batch."
        }), 400

    job_id = create_local_job(clean_items, job_type="openai_batch", batch_image_reference_mode=batch_image_reference_mode)

    thread = threading.Thread(target=run_openai_batch_job, args=(job_id, clean_items), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/cancel-batch/<job_id>", methods=["POST"])
def api_cancel_batch(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404

        if job.get("job_type") != "openai_batch":
            return jsonify({"error": "Only OpenAI batch jobs can be cancelled."}), 400

        batch_id = job.get("batch_id")
        if not batch_id:
            return jsonify({"error": "This batch job does not have a batch ID yet."}), 400

    try:
        current_batch = retrieve_openai_batch(batch_id)
        current_status = getattr(current_batch, "status", "")

        if current_status in TERMINAL_BATCH_STATUSES:
            with jobs_lock:
                job = jobs.get(job_id)
                if job:
                    job.update({
                        "status": current_status,
                        "progress": batch_progress_percent(current_batch),
                        "request_counts": get_batch_counts(current_batch),
                        "output_file_id": getattr(current_batch, "output_file_id", None),
                        "error_file_id": getattr(current_batch, "error_file_id", None),
                        "error": "This OpenAI batch is already terminal and can no longer be aborted.",
                    })
                    save_jobs_snapshot_unlocked()

            return jsonify({"error": "This OpenAI batch is already terminal and can no longer be aborted."}), 400

        cancelled_batch = client.batches.cancel(batch_id)
        cancelled_status = getattr(cancelled_batch, "status", "cancelling")

        with jobs_lock:
            job = jobs.get(job_id)
            if job:
                job.update({
                    "status": cancelled_status,
                    "progress": batch_progress_percent(cancelled_batch),
                    "request_counts": get_batch_counts(cancelled_batch),
                    "output_file_id": getattr(cancelled_batch, "output_file_id", None),
                    "error_file_id": getattr(cancelled_batch, "error_file_id", None),
                    "error": "Batch cancellation requested. Completed items may still be charged and can still be fetched when OpenAI returns them.",
                })

                for item in job.get("items", []):
                    if item.get("filename") or item.get("status") == "failed":
                        continue
                    item["status"] = cancelled_status
                    item["error"] = item.get("error") or "Batch cancellation requested before this item was downloaded."

                save_jobs_snapshot_unlocked()

        return jsonify({"job_id": job_id, "batch_id": batch_id, "status": cancelled_status})

    except Exception as e:
        with jobs_lock:
            job = jobs.get(job_id)
            if job:
                job["error"] = f"Failed to cancel OpenAI batch: {e}"
                save_jobs_snapshot_unlocked()

        return jsonify({"error": str(e)}), 500



@app.route("/api/retry-batch/<job_id>", methods=["POST"])
def api_retry_batch(job_id):
    with jobs_lock:
        old_job = jobs.get(job_id)
        if not old_job:
            return jsonify({"error": "Job not found."}), 404

        if old_job.get("job_type") != "openai_batch":
            return jsonify({"error": "Only OpenAI batch jobs can be retried here."}), 400

        if old_job.get("status") not in TERMINAL_JOB_STATUSES:
            return jsonify({"error": "This batch job is still active. Abort it before retrying."}), 400

        retry_items = []
        for item in old_job.get("items", []):
            if item.get("filename"):
                continue
            if item.get("mode") != "generate":
                continue
            retry_items.append(clone_batch_item_for_retry(item))

    if not retry_items:
        return jsonify({"error": "No failed or missing-output batch items are available to retry."}), 400

    batch_image_reference_mode = old_job.get("batch_image_reference_mode") or "file_id"

    missing_batch_refs = [
        item for item in retry_items
        if item.get("has_input_reference")
        and item.get("input_reference_local_filename")
        and not item.get("input_reference_file_id")
    ]
    if missing_batch_refs and batch_image_reference_mode != "base64":
        first_error = missing_batch_refs[0].get("input_reference_upload_error") or "OpenAI Files upload failed for the image reference."
        return jsonify({
            "error": "Image-guided OpenAI Batch retry requires the image to upload to OpenAI Files first. " + first_error
        }), 400

    models = set(item.get("model", "sora-2") for item in retry_items)
    if len(models) > 1:
        return jsonify({
            "error": "OpenAI Batch input files for this UI are restricted to one model at a time. Retry one model group at a time."
        }), 400

    new_job_id = create_local_job(retry_items, job_type="openai_batch", batch_image_reference_mode=batch_image_reference_mode)

    with jobs_lock:
        new_job = jobs.get(new_job_id)
        if new_job:
            new_job["retry_of_job_id"] = job_id
            new_job["error"] = None
            save_jobs_snapshot_unlocked()

    thread = threading.Thread(target=run_openai_batch_job, args=(new_job_id, retry_items), daemon=True)
    thread.start()

    return jsonify({"job_id": new_job_id, "retry_of_job_id": job_id})


@app.route("/api/resume-batch/<job_id>", methods=["POST"])
def api_resume_batch(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404

        if job.get("job_type") != "openai_batch":
            return jsonify({"error": "Only OpenAI batch jobs can be resumed."}), 400

        batch_id = job.get("batch_id")
        if not batch_id:
            return jsonify({"error": "This job does not have a batch ID to resume."}), 400

        if job.get("status") not in TERMINAL_JOB_STATUSES:
            return jsonify({"error": "This job is already active."}), 400

        items = job.get("items", [])
        job["status"] = "queued_resume"
        job["error"] = None
        save_jobs_snapshot_unlocked()

    thread = threading.Thread(target=run_openai_batch_job, args=(job_id, items, batch_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/resume-standard/<job_id>", methods=["POST"])
def api_resume_standard(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404

        if job.get("job_type") == "openai_batch":
            return jsonify({"error": "Use batch resume for OpenAI batch jobs."}), 400

        if job.get("status") not in TERMINAL_JOB_STATUSES:
            return jsonify({"error": "This job is already active."}), 400

        items = job.get("items", [])
        needs_resume = any(item for item in items if not item.get("filename"))
        if not needs_resume:
            return jsonify({"error": "This job already has local output files."}), 400

        job["status"] = "queued_resume"
        job["error"] = None
        save_jobs_snapshot_unlocked()

    thread = threading.Thread(target=run_resume_standard_job, args=(job_id,), daemon=True)
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
    return jsonify({"jobs": notification_jobs(), "mode": "notifications"})


@app.route("/api/jobs/log")
def api_jobs_log():
    page = request.args.get("page", "1")
    per_page = request.args.get("per_page", "20")
    values, pagination = paginate_list(get_recent_jobs(), page, per_page)
    return jsonify({"jobs": values, "pagination": pagination})


@app.route("/api/jobs/recover", methods=["POST"])
def api_recover_jobs():
    started = auto_recover_jobs_once()
    return jsonify({"ok": True, "started": started, "count": len(started)})


@app.route("/api/videos")
def api_videos():
    tab = request.args.get("tab", "main")
    query = request.args.get("q", "")
    page = request.args.get("page", "1")
    per_page = request.args.get("per_page", "40")
    filters = parse_gallery_filters(request.args.get("filters", ""), tab)
    tag = request.args.get("tag", "")
    page_values, pagination = paginate_list(filtered_videos_for_tab(tab, query, filters=filters, tag=tag), page, per_page)
    start_thumbnail_cache_for_filenames([video.get("filename") for video in page_values])
    videos = [public_gallery_video(video) for video in page_values]
    return jsonify({"videos": videos, "pagination": pagination, "tab": tab, "filters": sorted(filters), "tag": normalize_tag_name(tag), "q": query, "index": gallery_index_status()})


@app.route("/api/video-children/<path:filename>")
def api_video_children(filename):
    query = request.args.get("q", "")
    page = request.args.get("page", "1")
    per_page = request.args.get("per_page", "24")
    page_values, pagination = paginate_list(child_videos_for_source(filename, query), page, per_page)
    start_thumbnail_cache_for_filenames([video.get("filename") for video in page_values])
    videos = [public_gallery_video(video) for video in page_values]
    return jsonify({"videos": videos, "pagination": pagination, "filename": filename, "q": query})


@app.route("/api/video-info/<path:filename>")
def api_video_info(filename):
    try:
        path = safe_output_path(filename)
    except ValueError:
        return jsonify({"error": "Invalid filename."}), 400

    if not path.exists() or path.suffix.lower() != ".mp4":
        return jsonify({"error": "Video not found."}), 404

    return jsonify({"video": read_metadata(filename)})


@app.route("/api/tags")
def api_tags():
    return jsonify({"tags": all_custom_tag_names()})


@app.route("/api/tags", methods=["POST"])
def api_create_tag():
    data = request.get_json(force=True)
    try:
        tag = ensure_custom_tag(data.get("tag") or data.get("name") or "")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"tag": tag, "tags": all_custom_tag_names()})




@app.route("/api/tags/rename", methods=["POST"])
def api_rename_tag():
    data = request.get_json(force=True)
    try:
        old_name = data.get("old_name") or data.get("old") or data.get("tag") or ""
        new_name = data.get("new_name") or data.get("new") or data.get("name") or ""
        renamed = rename_custom_tag(old_name, new_name)
    except KeyError:
        return jsonify({"error": "Tag not found."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"tag": renamed, "tags": all_custom_tag_names()})


@app.route("/api/tags/delete", methods=["POST", "DELETE"])
def api_delete_tag():
    data = request.get_json(force=True, silent=True) or {}
    try:
        removed = delete_custom_tag(data.get("tag") or data.get("name") or "")
    except KeyError:
        return jsonify({"error": "Tag not found."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"removed": removed, "tags": all_custom_tag_names()})


@app.route("/api/video-tags/<path:filename>")
def api_get_video_tags(filename):
    try:
        path = safe_output_path(filename)
    except ValueError:
        return jsonify({"error": "Invalid filename."}), 400
    if not path.exists() or path.suffix.lower() != ".mp4":
        return jsonify({"error": "Video not found."}), 404
    tags = load_custom_tags()
    return jsonify({"filename": filename, "tags": tags_for_filename(filename, tags), "all_tags": all_custom_tag_names()})


@app.route("/api/video-tags", methods=["POST"])
def api_set_video_tags():
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    tag_names = data.get("tags", [])
    if not filename:
        return jsonify({"error": "Missing filename."}), 400
    if not isinstance(tag_names, list):
        return jsonify({"error": "tags must be a list."}), 400
    try:
        selected = set_video_custom_tags(filename, tag_names)
    except FileNotFoundError:
        return jsonify({"error": "Video not found."}), 404
    except ValueError:
        return jsonify({"error": "Invalid filename or tag."}), 400
    return jsonify({"filename": filename, "tags": selected, "all_tags": all_custom_tag_names()})


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


@app.route("/api/favorites")
def api_favorites():
    return jsonify({"favorites": sorted(load_favorites())})


@app.route("/api/favorite", methods=["POST"])
def api_favorite():
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    favorite = bool(data.get("favorite", True))

    if not filename:
        return jsonify({"error": "Missing filename."}), 400

    try:
        final_state = set_favorite(filename, favorite)
    except FileNotFoundError:
        return jsonify({"error": "Video not found."}), 404
    except ValueError:
        return jsonify({"error": "Invalid filename."}), 400

    return jsonify({"filename": filename, "favorite": final_state})


@app.route("/api/archive", methods=["POST"])
def api_archive():
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    archived = bool(data.get("archived", True))

    if not filename:
        return jsonify({"error": "Missing filename."}), 400

    try:
        final_state = set_archived(filename, archived)
    except FileNotFoundError:
        return jsonify({"error": "Video not found."}), 404
    except ValueError:
        return jsonify({"error": "Invalid filename."}), 400

    return jsonify({"filename": filename, "archived": final_state})


@app.route("/api/budget", methods=["GET", "POST"])
def api_budget():
    if request.method == "GET":
        return jsonify(load_budget())

    data = request.get_json(force=True)
    remaining = data.get("remaining", "")
    return jsonify(save_budget(remaining))


@app.route("/outputs/thumbnails/<path:filename>")
def output_thumbnail(filename):
    return send_from_directory(THUMBNAIL_DIR, filename)


@app.route("/api/thumbnails/generate", methods=["POST"])
def api_generate_thumbnails():
    # Legacy endpoint kept for compatibility. Thumbnail generation is now page-scoped
    # and starts from /api/videos for the currently displayed paginated results.
    return jsonify({"ok": True, "started": False, "thumbnail_status": get_thumbnail_status()})


@app.route("/api/thumbnails/status")
def api_thumbnail_status():
    return jsonify({"thumbnail_status": get_thumbnail_status()})


@app.route("/api/thumbnails/missing")
def api_missing_thumbnails():
    try:
        limit = int(request.args.get("limit", "40"))
    except Exception:
        limit = 40

    limit = max(1, min(limit, 80))
    tab = request.args.get("tab", "main")
    query = request.args.get("q", "")
    page = request.args.get("page", "1")
    per_page = request.args.get("per_page", "40")
    filters = parse_gallery_filters(request.args.get("filters", ""), tab)
    tag = request.args.get("tag", "")

    # Browser fallback should only consider the current visible page. The old
    # behavior scanned global missing thumbnails, which made search/tab/page
    # changes feel slow once the library grew.
    page_values, _pagination = paginate_list(filtered_videos_for_tab(tab, query, filters=filters, tag=tag), page, per_page)
    missing = []
    for video in page_values:
        filename = video.get("filename")
        if filename and not cached_thumbnail(filename):
            missing.append(filename)
        if len(missing) >= limit:
            break

    return jsonify({"missing": missing, "thumbnail_status": get_thumbnail_status()})


@app.route("/api/thumbnails/browser-upload", methods=["POST"])
def api_browser_thumbnail_upload():
    data = request.get_json(force=True)
    filename = str(data.get("filename") or "").strip()
    image_data = str(data.get("image_data") or "")

    if not filename:
        return jsonify({"error": "Missing filename."}), 400

    try:
        thumb_name = save_browser_thumbnail(filename, image_data)
        return jsonify({"ok": True, "thumbnail": thumb_name, "thumbnail_status": get_thumbnail_status()})
    except Exception as exc:
        set_thumbnail_error(f"Browser thumbnail failed for {filename}: {exc}")
        return jsonify({"error": str(exc), "thumbnail_status": get_thumbnail_status()}), 400


@app.route("/api/thumbnail/<path:filename>")
def api_thumbnail(filename):
    thumb_name = cached_thumbnail(filename)
    if not thumb_name:
        return jsonify({"error": "Thumbnail not ready.", "thumbnail_status": get_thumbnail_status()}), 404
    return send_from_directory(THUMBNAIL_DIR, thumb_name)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
