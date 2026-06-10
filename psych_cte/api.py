"""Local web platform for audio CTE prediction."""

from __future__ import annotations

import csv
import gc
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from urllib.parse import urlparse

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .export_speaker_samples import export_speaker_audio, normalize_speaker_labels, write_rttm


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
WEB_DIR = ROOT_DIR / "web"
PLATFORM_DIR = ROOT_DIR / "outputs" / "platform"
UPLOAD_AUDIO_DIR = PLATFORM_DIR / "uploads" / "audio"
UPLOAD_MODEL_DIR = PLATFORM_DIR / "uploads" / "models"
UPLOAD_MANIFEST_DIR = PLATFORM_DIR / "uploads" / "diarization"
SPEAKER_PREVIEW_DIR = PLATFORM_DIR / "speaker_previews"
JOB_DIR = PLATFORM_DIR / "jobs"
MODEL_CACHE_ROOT = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic"


def cached_model_or_id(dirname, model_id):
    path = MODEL_CACHE_ROOT / dirname
    return str(path) if path.exists() else model_id


DEFAULT_ASR_MODEL = cached_model_or_id(
    "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
DEFAULT_VAD_MODEL = cached_model_or_id(
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
)
DEFAULT_PUNC_MODEL = cached_model_or_id(
    "punc_ct-transformer_cn-en-common-vocab471067-large",
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
)
DEFAULT_SPK_MODEL = cached_model_or_id(
    "speech_campplus_sv_zh-cn_16k-common",
    "iic/speech_campplus_sv_zh-cn_16k-common",
)

for folder in (UPLOAD_AUDIO_DIR, UPLOAD_MODEL_DIR, UPLOAD_MANIFEST_DIR, SPEAKER_PREVIEW_DIR, JOB_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def resolve_runner_python():
    configured = os.environ.get("PSYCH_CTE_PYTHON", r"D:\cuda\python.exe")
    candidate = Path(configured)
    if candidate.exists():
        return str(candidate)
    return sys.executable


def has_hf_token():
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


RUNNER_PYTHON = resolve_runner_python()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

app = FastAPI(title="Psych CTE Platform", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")
app.mount("/speaker-previews", StaticFiles(directory=str(SPEAKER_PREVIEW_DIR)), name="speaker-previews")

JOBS = {}
DIARIZATION_JOBS = {}
DIARIZATION_JOB_TOKENS = {}
MODEL_CACHE = {}


def is_under(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def clean_name(filename):
    suffix = Path(filename or "upload").suffix.lower()
    stem = Path(filename or "upload").stem
    stem = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", stem).strip("._-")
    return "{}{}".format(stem or "upload", suffix)


def save_upload(upload, folder):
    folder.mkdir(parents=True, exist_ok=True)
    safe = clean_name(upload.filename)
    path = folder / "{}_{}".format(uuid.uuid4().hex[:10], safe)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def rel_id(path):
    return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()


def job_path(job_id):
    return JOB_DIR / job_id


def save_job(job):
    folder = job_path(job["id"])
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "job.json", "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    JOBS[job["id"]] = job


def update_job(job_id, **changes):
    job = JOBS.get(job_id) or load_job(job_id)
    job.update(changes)
    job["updated_at"] = time.time()
    save_job(job)
    return job


def load_job(job_id):
    path = job_path(job_id) / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with open(path, "r", encoding="utf-8") as f:
        job = json.load(f)
    JOBS[job_id] = job
    return job


def public_job(job):
    data = dict(job)
    data.pop("command", None)
    data["downloads"] = {}
    for key, path in job.get("files", {}).items():
        if path and Path(path).exists():
            data["downloads"][key] = "/api/jobs/{}/download/{}".format(job["id"], key)
    return data


def diarization_job_path(preview_id):
    return SPEAKER_PREVIEW_DIR / preview_id / "diarization_job.json"


def save_diarization_job(job):
    folder = SPEAKER_PREVIEW_DIR / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "diarization_job.json", "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    DIARIZATION_JOBS[job["id"]] = job


def load_diarization_job(preview_id):
    folder = (SPEAKER_PREVIEW_DIR / preview_id).resolve()
    if not is_under(folder, SPEAKER_PREVIEW_DIR):
        raise HTTPException(status_code=400, detail="Invalid preview id")
    path = folder / "diarization_job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Diarization job not found")
    with open(path, "r", encoding="utf-8") as f:
        job = json.load(f)
    DIARIZATION_JOBS[preview_id] = job
    return job


def update_diarization_job(preview_id, **changes):
    job = DIARIZATION_JOBS.get(preview_id) or load_diarization_job(preview_id)
    job.update(changes)
    job["updated_at"] = time.time()
    save_diarization_job(job)
    return job


def public_diarization_job(job):
    files = job.get("files", {})
    manifest_path = Path(files.get("manifest", "")) if files.get("manifest") else None
    if manifest_path and manifest_path.exists() and job.get("status") != "completed":
        try:
            result = build_diarization_result(job["id"], manifest_path, Path(files.get("rttm", "")), Path(files.get("log", "")))
            job.update(
                {
                    "status": "completed",
                    "message": "说话人分离 JSON 已生成",
                    "result": result,
                    **result,
                }
            )
            save_diarization_job(job)
        except Exception:
            pass
    elif job.get("status") == "completed" and (not manifest_path or not manifest_path.exists()):
        job.update(
            {
                "status": "failed",
                "message": "生成结果异常：未找到说话人分离 JSON 文件",
                "failure_reason": "任务标记为完成，但 speaker_samples_manifest.json 不存在。",
            }
        )
        save_diarization_job(job)

    data = dict(job)
    data.pop("command", None)
    data["downloads"] = {}
    for key, path in job.get("files", {}).items():
        if key in ("audio", "output_root"):
            continue
        if path and Path(path).exists():
            data["downloads"][key] = "/api/diarization/{}/download/{}".format(job["id"], key)
    return data


def iter_model_files():
    for base in (CHECKPOINT_DIR, UPLOAD_MODEL_DIR):
        if not base.exists():
            continue
        for suffix in ("*.pt", "*.pth", "*.bin"):
            for path in base.rglob(suffix):
                if path.is_file():
                    yield path


def inspect_checkpoint(path):
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    cached = MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    model_type = "unknown"
    encoder = ""
    max_length = ""
    error = ""
    try:
        ckpt = torch.load(str(path), map_location="cpu")
        if isinstance(ckpt, dict):
            encoder = ckpt.get("encoder", "") or ""
            max_length = ckpt.get("max_length", "") or ""
            declared = ckpt.get("model_type", "") or ""
            state = ckpt.get("model_state", {})
            state_keys = list(state.keys()) if hasattr(state, "keys") else []
            if declared or any(k.startswith("segment_model.") or k.startswith("weight_net.") for k in state_keys):
                model_type = "audio"
            elif any(k.startswith("cte_head.") or k.startswith("empathy_head.") for k in state_keys):
                model_type = "segment"
            elif "audio" in path.name.lower():
                model_type = "audio"
            elif "segment" in path.name.lower():
                model_type = "segment"
        del ckpt
        gc.collect()
    except Exception as exc:  # pragma: no cover - depends on external model files
        error = str(exc)
        lowered = path.name.lower()
        if "audio" in lowered:
            model_type = "audio"
        elif "segment" in lowered:
            model_type = "segment"

    meta = {
        "id": rel_id(path),
        "name": path.name,
        "type": model_type,
        "path": str(path),
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "updated_at": stat.st_mtime,
        "encoder": encoder,
        "max_length": max_length,
        "source": "uploaded" if is_under(path, UPLOAD_MODEL_DIR) else "checkpoint",
        "error": error,
    }
    MODEL_CACHE[cache_key] = meta
    return meta


def resolve_platform_file(file_id, allowed_roots):
    if not file_id:
        return None
    path = (ROOT_DIR / file_id).resolve()
    if not any(is_under(path, root) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="File path is outside allowed folders")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found: {}".format(file_id))
    return path


def read_csv_preview(path, limit=20):
    if not path or not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append(row)
    return rows


def read_json_summary(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return {
        "audio_id": data.get("audio_id", ""),
        "role_map": data.get("role_map", {}),
        "analysis_summary": data.get("analysis_summary", {}),
        "audio_prediction": data.get("audio_prediction", {}),
        "audio_summary": data.get("audio_summary", {}),
    }


def load_diarization_segments(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="Diarization JSON must contain a segments list")

    normalized = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        speaker = seg.get("speaker", seg.get("source_speaker", seg.get("label", "")))
        if start is None or end is None or not speaker:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        normalized.append(
            {
                "source_speaker": str(seg.get("source_speaker", speaker)),
                "speaker": str(speaker),
                "start": start,
                "end": end,
            }
        )
    return normalized


def generate_funasr_diarization(audio_path, output_root, manifest_path, rttm_path, device):
    from .predict_from_audio import normalize_asr_segments, run_funasr

    args = SimpleNamespace(
        audio=str(audio_path),
        asr_model=DEFAULT_ASR_MODEL,
        vad_model=DEFAULT_VAD_MODEL,
        punc_model=DEFAULT_PUNC_MODEL,
        spk_model=DEFAULT_SPK_MODEL,
        asr_device=device,
        hub="ms",
        max_single_segment_time=60000,
        batch_size_s=300,
        hotword="",
    )
    asr_result = run_funasr(args)
    turns = normalize_asr_segments(asr_result)
    raw_segments = []
    for turn in turns:
        start = int(turn.get("start_ms", 0)) / 1000.0
        end = int(turn.get("end_ms", 0)) / 1000.0
        speaker = str(turn.get("speaker", "unknown"))
        if end <= start:
            continue
        raw_segments.append({"source_speaker": speaker, "start": start, "end": end})
    if not raw_segments:
        raise RuntimeError("FunASR did not return usable speaker segments.")

    segments, source_mapping = normalize_speaker_labels(raw_segments)
    sample_files = export_speaker_audio(
        audio_path,
        output_root,
        segments,
        sample_seconds=15.0,
        min_segment_seconds=0.8,
        gap_seconds=0.25,
    )
    write_rttm(rttm_path, Path(audio_path).stem, segments)
    speakers = sorted(set(seg["speaker"] for seg in segments))
    manifest = {
        "audio_id": Path(audio_path).stem,
        "audio_file": str(Path(audio_path).resolve()),
        "backend": "funasr",
        "source_speaker_mapping": source_mapping,
        "sample_files": sample_files,
        "rttm": str(Path(rttm_path).resolve()),
        "segments": segments,
        "role_assignment_template": {
            "client_speaker": speakers[0] if speakers else "",
            "therapist_speaker": speakers[1] if len(speakers) > 1 else "",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def audio_cache_keys(audio_path):
    stem = Path(audio_path).stem
    parts = stem.split("_")
    keys = ["_".join(parts[idx:]) for idx in range(len(parts))]
    return [key for key in dict.fromkeys(keys) if key]


def find_cached_turns_csv(audio_path):
    keys = audio_cache_keys(audio_path)
    candidates = []
    for path in JOB_DIR.glob("*/*_turns.csv"):
        name = path.stem
        for key in keys:
            if name == "{}_turns".format(key) or name.endswith("_{}_turns".format(key)):
                candidates.append(path)
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def generate_diarization_from_turns_csv(audio_path, turns_csv, output_root, manifest_path, rttm_path):
    raw_segments = []
    with open(turns_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = (row.get("speaker") or "").strip()
            if not speaker:
                continue
            try:
                start = int(float(row.get("start_ms", 0))) / 1000.0
                end = int(float(row.get("end_ms", 0))) / 1000.0
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            raw_segments.append({"source_speaker": speaker, "start": start, "end": end})

    if not raw_segments:
        raise RuntimeError("Cached turns CSV did not contain usable speaker segments.")

    segments, source_mapping = normalize_speaker_labels(raw_segments)
    sample_files = export_speaker_audio(
        audio_path,
        output_root,
        segments,
        sample_seconds=15.0,
        min_segment_seconds=0.8,
        gap_seconds=0.25,
    )
    write_rttm(rttm_path, Path(audio_path).stem, segments)
    speakers = sorted(set(seg["speaker"] for seg in segments))
    manifest = {
        "audio_id": Path(audio_path).stem,
        "audio_file": str(Path(audio_path).resolve()),
        "backend": "cached_turns",
        "source_turns_csv": str(Path(turns_csv).resolve()),
        "source_speaker_mapping": source_mapping,
        "sample_files": sample_files,
        "rttm": str(Path(rttm_path).resolve()),
        "segments": segments,
        "role_assignment_template": {
            "client_speaker": speakers[0] if speakers else "",
            "therapist_speaker": speakers[1] if len(speakers) > 1 else "",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def infer_speaker_for_role(path, role):
    if not path or not Path(path).exists():
        return ""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    role_map = data.get("role_map", {}) or {}
    for speaker, mapped_role in role_map.items():
        if mapped_role == role:
            return speaker
    return ""


def tail_log_text(path, limit=20000):
    if not path or not Path(path).exists():
        return ""
    data = Path(path).read_bytes()
    if len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace")


def clean_log_text(text):
    if not text:
        return ""
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")


def last_non_empty_line(text):
    lines = [line.strip() for line in clean_log_text(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def last_error_line(text):
    lines = [line.strip() for line in clean_log_text(text).splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("$ "):
            continue
        if line.lower().startswith(("warning:", "userwarning:", "futurewarning:")):
            continue
        return line
    return lines[-1] if lines else ""


def append_log_line(log_path, text):
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(text)
        if not text.endswith("\n"):
            log.write("\n")
        log.flush()


def redact_command(command):
    redacted = []
    hide_next = False
    for item in command:
        if hide_next:
            redacted.append("<hidden>")
            hide_next = False
            continue
        redacted.append(item)
        if item in ("--hf-token", "--token"):
            hide_next = True
    return redacted


def format_command_for_log(command):
    return " ".join('"{}"'.format(x) if " " in x else x for x in redact_command(command))


def run_diarization_subprocess(
    preview_id,
    command,
    env,
    log_path,
    running_message,
    idle_timeout=900,
    total_timeout=7200,
):
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    line_queue = queue.Queue()
    sentinel = object()

    def read_output():
        try:
            if process.stdout:
                for raw_line in iter(process.stdout.readline, ""):
                    line_queue.put(raw_line)
        finally:
            line_queue.put(sentinel)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    current_message = running_message
    started_at = time.time()
    last_output_at = started_at

    while True:
        got_line = False
        raw_line = None
        try:
            raw_line = line_queue.get(timeout=1)
            got_line = True
        except queue.Empty:
            pass

        if got_line and raw_line is sentinel:
            break

        if got_line and raw_line is not None:
            clean = raw_line.rstrip("\n")
            append_log_line(log_path, clean)
            stripped = clean_log_text(clean).strip()
            if stripped:
                last_output_at = time.time()
            if stripped and len(stripped) <= 320:
                current_message = stripped
                update_diarization_job(preview_id, message=stripped, log_tail=tail_log_text(log_path))

        if process.poll() is not None and line_queue.empty():
            break

        now = time.time()
        if process.poll() is None and now - last_output_at > idle_timeout:
            message = "Diarization subprocess produced no new log output for {} seconds; it was stopped automatically.".format(
                int(idle_timeout)
            )
            append_log_line(log_path, "[stage] {}".format(message))
            update_diarization_job(preview_id, message=message, log_tail=tail_log_text(log_path))
            process.kill()
            process.wait(timeout=10)
            raise TimeoutError(message)

        if process.poll() is None and now - started_at > total_timeout:
            message = "Diarization subprocess exceeded total runtime of {} seconds; it was stopped automatically.".format(
                int(total_timeout)
            )
            append_log_line(log_path, "[stage] {}".format(message))
            update_diarization_job(preview_id, message=message, log_tail=tail_log_text(log_path))
            process.kill()
            process.wait(timeout=10)
            raise TimeoutError(message)

    return_code = process.wait()
    return return_code, current_message


def start_diarization_watchdog(preview_id, process, log_path, idle_timeout=900, total_timeout=7200):
    log_path = Path(log_path)

    def watch():
        started_at = time.time()
        last_change_at = started_at
        try:
            last_mtime = log_path.stat().st_mtime
        except OSError:
            last_mtime = 0

        while process.poll() is None:
            time.sleep(5)
            now = time.time()
            try:
                current_mtime = log_path.stat().st_mtime
            except OSError:
                current_mtime = last_mtime

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                last_change_at = now

            if now - last_change_at > idle_timeout:
                message = "说话人分离子进程超过 {} 秒没有新的日志输出，已自动终止。".format(int(idle_timeout))
                append_log_line(log_path, "[stage] {}".format(message))
                update_diarization_job(preview_id, message=message, log_tail=tail_log_text(log_path))
                try:
                    process.kill()
                except OSError:
                    pass
                break

            if now - started_at > total_timeout:
                message = "说话人分离子进程超过 {} 秒总运行时间，已自动终止。".format(int(total_timeout))
                append_log_line(log_path, "[stage] {}".format(message))
                update_diarization_job(preview_id, message=message, log_tail=tail_log_text(log_path))
                try:
                    process.kill()
                except OSError:
                    pass
                break

    threading.Thread(target=watch, daemon=True).start()


def local_proxy_is_unavailable(proxy_url):
    try:
        parsed = urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port
    except Exception:
        return False
    if host not in ("127.0.0.1", "localhost", "::1") or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return False
    except OSError:
        return True


def drop_unavailable_local_proxy(env, log_path=None):
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    bad_values = {env.get(key) for key in proxy_keys if env.get(key) and local_proxy_is_unavailable(env.get(key))}
    if not bad_values:
        return env
    for key in proxy_keys:
        if env.get(key) in bad_values:
            env.pop(key, None)
    if log_path:
        append_log_line(
            log_path,
            "[stage] Detected unavailable local proxy {}; cleared proxy variables for this diarization run.".format(
                ", ".join(sorted(bad_values))
            ),
        )
    return env


class LogWriter:
    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def write(self, text):
        if not text:
            return 0
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(text)
            log.flush()
        return len(text)

    def flush(self):
        return None


def run_logged_command(job_id, command, log_path, running_message):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    append_log_line(log_path, "")
    append_log_line(log_path, "$ {}".format(format_command_for_log(command)))

    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except Exception:
        error_text = traceback.format_exc()
        append_log_line(log_path, error_text)
        update_job(
            job_id,
            status="failed",
            message="{}: {}".format(running_message, last_non_empty_line(error_text) or "启动失败"),
            failure_reason=error_text,
            log_tail=tail_log_text(log_path),
        )
        return 1

    current_message = running_message
    for raw_line in iter(process.stdout.readline, ""):
        if raw_line == "" and process.poll() is not None:
            break
        clean = raw_line.rstrip("\n")
        append_log_line(log_path, clean)
        stripped = clean_log_text(clean).strip()
        if stripped and len(stripped) <= 320:
            current_message = stripped
            update_job(job_id, message=stripped, log_tail=tail_log_text(log_path))

    return_code = process.wait()
    tail_text = tail_log_text(log_path)
    if return_code != 0:
        failure_reason = last_non_empty_line(tail_text) or "子进程返回码 {}".format(return_code)
        update_job(
            job_id,
            status="failed",
            message="预测失败：{}".format(failure_reason),
            failure_reason=failure_reason,
            log_tail=tail_text,
        )
    else:
        update_job(job_id, message=current_message, log_tail=tail_text)
    return return_code


def run_prediction_job(job_id):
    job = update_job(job_id, status="running", message="正在执行音频转写和预测", failure_reason="")
    files = job["files"]
    log_path = Path(files["log"])

    command = [
        RUNNER_PYTHON,
        "-u",
        "-m",
        "psych_cte.predict_from_audio",
        "--audio",
        files["audio"],
        "--segment-checkpoint",
        files["segment_model"],
        "--output-json",
        files["json"],
        "--output-csv",
        files["csv"],
        "--output-turns-csv",
        files["turns_csv"],
        "--device",
        job["options"]["device"],
        "--asr-device",
        job["options"]["asr_device"],
        "--asr-model",
        job["options"]["asr_model"],
        "--vad-model",
        job["options"]["vad_model"],
        "--punc-model",
        job["options"]["punc_model"],
        "--spk-model",
        job["options"]["spk_model"],
        "--hub",
        job["options"]["hub"],
        "--threshold",
        str(job["options"]["threshold"]),
        "--first-speaker-role",
        job["options"]["first_speaker_role"],
    ]
    if files.get("audio_model"):
        command.extend(["--audio-checkpoint", files["audio_model"]])
    if files.get("diarization_json"):
        command.extend(["--diarization-json", files["diarization_json"]])
    if job["options"].get("client_speaker"):
        command.extend(["--client-speaker", job["options"]["client_speaker"]])
    if job["options"].get("therapist_speaker"):
        command.extend(["--therapist-speaker", job["options"]["therapist_speaker"]])
    if job["options"].get("hotword"):
        command.extend(["--hotword", job["options"]["hotword"]])

    update_job(job_id, command=command)
    result = run_logged_command(job_id, command, log_path, "正在执行音频转写和预测")
    if result != 0:
        return

    therapist_speaker = job["options"].get("therapist_speaker", "")
    if job["options"].get("run_acoustics") and files.get("diarization_json") and not therapist_speaker:
        therapist_speaker = infer_speaker_for_role(files["json"], "therapist")
        if therapist_speaker:
            options = dict(job["options"])
            options["therapist_speaker"] = therapist_speaker
            job = update_job(job_id, options=options)
            append_log_line(
                log_path,
                "[stage] Inferred therapist speaker for acoustic features: {}".format(therapist_speaker),
            )

    if job["options"].get("run_acoustics") and files.get("diarization_json") and therapist_speaker:
        update_job(job_id, status="running", message="预测完成，正在提取咨询师声学特征")
        acoustic_command = [
            RUNNER_PYTHON,
            "-u",
            "-m",
            "psych_cte.extract_therapist_acoustics",
            "--audio",
            files["audio"],
            "--diarization-json",
            files["diarization_json"],
            "--therapist-speaker",
            therapist_speaker,
            "--turns-csv",
            files["turns_csv"],
            "--asr-model",
            job["options"]["asr_model"],
            "--vad-model",
            job["options"]["vad_model"],
            "--punc-model",
            job["options"]["punc_model"],
            "--spk-model",
            job["options"]["spk_model"],
            "--hub",
            job["options"]["hub"],
            "--output-csv",
            files["acoustics_csv"],
            "--output-text-csv",
            files["acoustics_text_csv"],
        ]
        acoustic_result = run_logged_command(job_id, acoustic_command, log_path, "正在提取咨询师声学特征")
        if acoustic_result != 0:
            return

    elif job["options"].get("run_acoustics"):
        append_log_line(
            log_path,
            "[stage] Acoustic features skipped: diarization JSON or therapist speaker was not available.",
        )

    preview = {
        "summary": read_json_summary(files["json"]),
        "segments": read_csv_preview(files["csv"], limit=20),
        "turns": read_csv_preview(files["turns_csv"], limit=20),
        "acoustics": read_csv_preview(files.get("acoustics_csv"), limit=5),
    }
    update_job(
        job_id,
        status="completed",
        message="预测完成",
        preview=preview,
        log_tail=tail_log_text(log_path),
    )

def build_diarization_result(preview_id, manifest_path, rttm_path, log_path):
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        manifest = json.load(f)

    counts = {}
    durations = {}
    for seg in manifest.get("segments", []):
        speaker = str(seg.get("speaker", ""))
        if not speaker:
            continue
        counts[speaker] = counts.get(speaker, 0) + 1
        durations[speaker] = durations.get(speaker, 0.0) + max(
            0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        )

    speakers = []
    for speaker, path in sorted((manifest.get("sample_files") or {}).items()):
        sample_path = Path(path)
        try:
            sample_url = "/speaker-previews/{}".format(
                sample_path.resolve().relative_to(SPEAKER_PREVIEW_DIR.resolve()).as_posix()
            )
        except ValueError:
            sample_url = ""
        speakers.append(
            {
                "speaker": speaker,
                "sample_url": sample_url,
                "segment_count": counts.get(speaker, 0),
                "duration_s": round(durations.get(speaker, 0.0), 3),
            }
        )

    return {
        "manifest_id": rel_id(manifest_path),
        "manifest_url": "/api/diarization/{}/download/manifest".format(preview_id),
        "rttm_url": "/api/diarization/{}/download/rttm".format(preview_id),
        "log_url": "/api/diarization/{}/download/log".format(preview_id),
        "speakers": speakers,
    }


def run_diarization_job(preview_id):
    job = update_diarization_job(preview_id, status="running", message="正在生成说话人分离 JSON", failure_reason="")
    files = job["files"]
    options = job["options"]
    audio_path = Path(files["audio"])
    output_root = Path(files["output_root"])
    manifest_path = Path(files["manifest"])
    rttm_path = Path(files["rttm"])
    log_path = Path(files["log"])
    backend = options["backend"]
    hf_token = DIARIZATION_JOB_TOKENS.get(preview_id, "")
    if not hf_token and options.get("hf_token") and not str(options.get("hf_token")).startswith("<"):
        hf_token = options.get("hf_token", "")

    try:
        append_log_line(log_path, "[stage] Speaker diarization job started.")
        if backend == "funasr":
            cached_turns_csv = find_cached_turns_csv(audio_path)
            if cached_turns_csv:
                append_log_line(
                    log_path,
                    "[stage] Reusing cached prediction turns for speaker preview: {}".format(cached_turns_csv),
                )
                generate_diarization_from_turns_csv(
                    audio_path,
                    cached_turns_csv,
                    output_root,
                    manifest_path,
                    rttm_path,
                )
                result = build_diarization_result(preview_id, manifest_path, rttm_path, log_path)
                update_diarization_job(
                    preview_id,
                    status="completed",
                    message="说话人分离 JSON 已生成",
                    result=result,
                    log_tail=tail_log_text(log_path),
                    **result,
                )
                return

            command = [
                RUNNER_PYTHON,
                "-u",
                "-m",
                "psych_cte.export_speaker_samples",
                "--audio",
                str(audio_path),
                "--backend",
                "funasr",
                "--output-dir",
                str(output_root),
                "--manifest",
                str(manifest_path),
                "--rttm",
                str(rttm_path),
                "--device",
                options["device"],
                "--asr-device",
                options["device"],
                "--asr-model",
                DEFAULT_ASR_MODEL,
                "--vad-model",
                DEFAULT_VAD_MODEL,
                "--punc-model",
                DEFAULT_PUNC_MODEL,
                "--spk-model",
                DEFAULT_SPK_MODEL,
                "--hub",
                "ms",
                "--num-speakers",
                str(int(options.get("num_speakers") or 2)),
            ]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            update_diarization_job(preview_id, command=redact_command(command))
            append_log_line(log_path, "$ {}".format(format_command_for_log(command)))
            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            start_diarization_watchdog(preview_id, process, log_path, idle_timeout=300)
            current_message = "正在生成说话人分离 JSON"
            for raw_line in iter(process.stdout.readline, ""):
                if raw_line == "" and process.poll() is not None:
                    break
                clean = raw_line.rstrip("\n")
                append_log_line(log_path, clean)
                stripped = clean_log_text(clean).strip()
                if stripped and len(stripped) <= 320:
                    current_message = stripped
                    update_diarization_job(preview_id, message=stripped, log_tail=tail_log_text(log_path))
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(last_error_line(tail_log_text(log_path)) or "speaker diarization failed")
            update_diarization_job(preview_id, message=current_message, log_tail=tail_log_text(log_path))
            if not manifest_path.exists():
                raise RuntimeError(last_error_line(tail_log_text(log_path)) or "speaker diarization failed")
            result = build_diarization_result(preview_id, manifest_path, rttm_path, log_path)
            update_diarization_job(
                preview_id,
                status="completed",
                message="说话人分离 JSON 已生成",
                result=result,
                log_tail=tail_log_text(log_path),
                **result,
            )
            return
        if backend == "funasr":
            append_log_line(log_path, "[stage] Running FunASR speaker diarization...")
            writer = LogWriter(log_path)
            with redirect_stdout(writer), redirect_stderr(writer):
                generate_funasr_diarization(
                    audio_path,
                    output_root,
                    manifest_path,
                    rttm_path,
                    options["device"],
                )
            append_log_line(log_path, "[stage] FunASR speaker diarization completed.")
        else:
            if not hf_token and not has_hf_token():
                message = (
                    "pyannote/whisperx diarization needs a Hugging Face token. "
                    "Fill the HF Token field in the page, or set HF_TOKEN before starting the backend. "
                    "No manifest, RTTM, or speaker sample files will be created until the token is available."
                )
                append_log_line(log_path, message)
                raise RuntimeError(message)

            command = [
                RUNNER_PYTHON,
                "-u",
                "-m",
                "psych_cte.export_speaker_samples",
                "--audio",
                str(audio_path),
                "--backend",
                backend,
                "--output-dir",
                str(output_root),
                "--manifest",
                str(manifest_path),
                "--rttm",
                str(rttm_path),
                "--device",
                options["device"],
                "--num-speakers",
                str(int(options.get("num_speakers") or 2)),
            ]
            if hf_token:
                command.extend(["--hf-token", hf_token])

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
            env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
            env = drop_unavailable_local_proxy(env, log_path)
            append_log_line(
                log_path,
                "[stage] Hugging Face timeouts: etag={}s, download={}s.".format(
                    env.get("HF_HUB_ETAG_TIMEOUT", "default"),
                    env.get("HF_HUB_DOWNLOAD_TIMEOUT", "default"),
                ),
            )
            update_diarization_job(preview_id, command=redact_command(command))
            append_log_line(log_path, "$ {}".format(format_command_for_log(command)))
            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            current_message = "正在生成说话人分离 JSON"
            for raw_line in iter(process.stdout.readline, ""):
                if raw_line == "" and process.poll() is not None:
                    break
                clean = raw_line.rstrip("\n")
                append_log_line(log_path, clean)
                stripped = clean_log_text(clean).strip()
                if stripped and len(stripped) <= 320:
                    current_message = stripped
                    update_diarization_job(preview_id, message=stripped, log_tail=tail_log_text(log_path))
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(last_error_line(tail_log_text(log_path)) or "speaker diarization failed")
            update_diarization_job(preview_id, message=current_message, log_tail=tail_log_text(log_path))

        if not manifest_path.exists():
            raise RuntimeError(last_error_line(tail_log_text(log_path)) or "speaker diarization failed")

        result = build_diarization_result(preview_id, manifest_path, rttm_path, log_path)
        update_diarization_job(
            preview_id,
            status="completed",
            message="说话人分离 JSON 已生成",
            result=result,
            log_tail=tail_log_text(log_path),
            **result,
        )
    except Exception:
        error_text = traceback.format_exc()
        append_log_line(log_path, error_text)
        reason = last_error_line(error_text) or last_error_line(tail_log_text(log_path)) or "speaker diarization failed"
        update_diarization_job(
            preview_id,
            status="failed",
            message="生成失败：{}".format(reason),
            failure_reason=reason,
            log_tail=tail_log_text(log_path),
        )


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Psych CTE Platform API"})


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "root": str(ROOT_DIR),
        "runner_python": RUNNER_PYTHON,
        "api_python": sys.executable,
        "hf_token_configured": has_hf_token(),
        "asr_defaults": {
            "asr_model": DEFAULT_ASR_MODEL,
            "vad_model": DEFAULT_VAD_MODEL,
            "punc_model": DEFAULT_PUNC_MODEL,
            "spk_model": DEFAULT_SPK_MODEL,
        },
    }


@app.get("/api/models")
def list_models():
    models = [inspect_checkpoint(path) for path in iter_model_files()]
    models.sort(key=lambda item: (item["type"], item["source"], item["name"]))
    return {"models": models}


@app.post("/api/models/upload")
def upload_model(model: UploadFile = File(...)):
    path = save_upload(model, UPLOAD_MODEL_DIR)
    return {"model": inspect_checkpoint(path)}


@app.post("/api/speaker-preview")
def create_speaker_preview(
    audio: UploadFile = File(...),
    diarization_json: UploadFile = File(...),
):
    audio_path = save_upload(audio, UPLOAD_AUDIO_DIR)
    diarization_path = save_upload(diarization_json, UPLOAD_MANIFEST_DIR)
    segments = load_diarization_segments(diarization_path)
    if not segments:
        raise HTTPException(status_code=400, detail="No usable speaker segments were found")

    preview_id = uuid.uuid4().hex
    output_dir = SPEAKER_PREVIEW_DIR / preview_id
    try:
        sample_files = export_speaker_audio(
            audio_path,
            output_dir,
            segments,
            sample_seconds=12.0,
            min_segment_seconds=0.7,
            gap_seconds=0.25,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not sample_files:
        raise HTTPException(status_code=400, detail="No speaker sample audio could be generated")

    counts = {}
    durations = {}
    for seg in segments:
        speaker = seg["speaker"]
        counts[speaker] = counts.get(speaker, 0) + 1
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, seg["end"] - seg["start"])

    speakers = []
    for speaker in sorted(sample_files):
        filename = Path(sample_files[speaker]).name
        speakers.append(
            {
                "speaker": speaker,
                "sample_url": "/speaker-previews/{}/{}".format(preview_id, filename),
                "segment_count": counts.get(speaker, 0),
                "duration_s": round(durations.get(speaker, 0.0), 3),
            }
        )
    return {"preview_id": preview_id, "speakers": speakers}


@app.post("/api/diarization")
def create_diarization(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    backend: str = Form("funasr"),
    device: str = Form("cpu"),
    num_speakers: int = Form(2),
    hf_token: str = Form(""),
):
    if backend not in ("funasr", "pyannote", "whisperx"):
        raise HTTPException(status_code=400, detail="backend must be funasr, pyannote or whisperx")

    audio_path = save_upload(audio, UPLOAD_AUDIO_DIR)
    preview_id = uuid.uuid4().hex
    output_root = SPEAKER_PREVIEW_DIR / preview_id
    manifest_path = output_root / "speaker_samples_manifest.json"
    rttm_path = output_root / "{}.rttm".format(audio_path.stem)
    log_path = output_root / "diarization.log"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("[stage] Diarization job queued.\n")

    hf_token_value = hf_token.strip()
    if hf_token_value:
        DIARIZATION_JOB_TOKENS[preview_id] = hf_token_value

    job = {
        "id": preview_id,
        "status": "queued",
        "message": "说话人分离任务已创建",
        "failure_reason": "",
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": {
            "audio": str(audio_path),
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "rttm": str(rttm_path),
            "log": str(log_path),
        },
        "input": {
            "audio_filename": audio.filename,
        },
        "options": {
            "backend": backend,
            "device": device.strip() or "cpu",
            "num_speakers": int(num_speakers or 2),
            "hf_token": "<provided>" if hf_token_value else "",
        },
        "speakers": [],
        "manifest_id": "",
        "manifest_url": "",
        "rttm_url": "",
        "log_url": "/api/diarization/{}/download/log".format(preview_id),
    }
    save_diarization_job(job)
    background_tasks.add_task(run_diarization_job, preview_id)
    return public_diarization_job(job)


@app.get("/api/diarization/{preview_id}")
def get_diarization_job(preview_id: str):
    return public_diarization_job(DIARIZATION_JOBS.get(preview_id) or load_diarization_job(preview_id))


@app.get("/api/diarization/{preview_id}/log")
def get_diarization_log(preview_id: str, tail: int = 200):
    job = DIARIZATION_JOBS.get(preview_id) or load_diarization_job(preview_id)
    log_path = job.get("files", {}).get("log")
    tail_text = tail_log_text(log_path, limit=max(4096, int(tail) * 200))
    display_text = clean_log_text(tail_text)
    tail_lines = [line for line in display_text.splitlines() if line.strip()]
    failure_reason = job.get("failure_reason", "")
    if job.get("status") == "failed" and not failure_reason:
        failure_reason = last_error_line(display_text)
    return {
        "job_id": preview_id,
        "status": job.get("status", ""),
        "message": job.get("message", ""),
        "failure_reason": failure_reason,
        "line_count": len(tail_lines),
        "tail_text": display_text,
    }


@app.get("/api/diarization/{preview_id}/download/{kind}")
def download_diarization_file(preview_id: str, kind: str):
    folder = (SPEAKER_PREVIEW_DIR / preview_id).resolve()
    if not is_under(folder, SPEAKER_PREVIEW_DIR):
        raise HTTPException(status_code=400, detail="Invalid preview id")
    files = {
        "manifest": folder / "speaker_samples_manifest.json",
        "log": folder / "diarization.log",
    }
    if kind == "rttm":
        matches = list(folder.glob("*.rttm"))
        path = matches[0] if matches else folder / "diarization.rttm"
    else:
        path = files.get(kind)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Requested file does not exist")
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs")
def list_jobs():
    for meta in JOB_DIR.glob("*/job.json"):
        try:
            load_job(meta.parent.name)
        except Exception:
            pass
    jobs = sorted(JOBS.values(), key=lambda item: item.get("created_at", 0), reverse=True)
    return {"jobs": [public_job(job) for job in jobs]}


@app.post("/api/predict")
def create_prediction_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    segment_model_id: str = Form(...),
    audio_model_id: str = Form(""),
    diarization_json: UploadFile = File(None),
    diarization_file_id: str = Form(""),
    client_speaker: str = Form(""),
    therapist_speaker: str = Form(""),
    first_speaker_role: str = Form("client"),
    threshold: float = Form(0.5),
    device: str = Form("cpu"),
    asr_device: str = Form("cpu"),
    asr_model: str = Form(DEFAULT_ASR_MODEL),
    vad_model: str = Form(DEFAULT_VAD_MODEL),
    punc_model: str = Form(DEFAULT_PUNC_MODEL),
    spk_model: str = Form(DEFAULT_SPK_MODEL),
    hub: str = Form("ms"),
    hotword: str = Form(""),
    run_acoustics: bool = Form(False),
):
    if first_speaker_role not in ("client", "therapist"):
        raise HTTPException(status_code=400, detail="first_speaker_role must be client or therapist")

    segment_model = resolve_platform_file(segment_model_id, [CHECKPOINT_DIR, UPLOAD_MODEL_DIR])
    audio_model = resolve_platform_file(audio_model_id, [CHECKPOINT_DIR, UPLOAD_MODEL_DIR]) if audio_model_id else None
    audio_path = save_upload(audio, UPLOAD_AUDIO_DIR)
    if diarization_json and diarization_json.filename:
        diarization_path = save_upload(diarization_json, UPLOAD_MANIFEST_DIR)
    elif diarization_file_id:
        diarization_path = resolve_platform_file(diarization_file_id, [UPLOAD_MANIFEST_DIR, SPEAKER_PREVIEW_DIR])
    else:
        diarization_path = None

    job_id = uuid.uuid4().hex
    folder = job_path(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    audio_id = audio_path.stem
    files = {
        "audio": str(audio_path),
        "segment_model": str(segment_model),
        "audio_model": str(audio_model) if audio_model else "",
        "diarization_json": str(diarization_path) if diarization_path else "",
        "json": str(folder / "{}_prediction.json".format(audio_id)),
        "csv": str(folder / "{}_prediction.csv".format(audio_id)),
        "turns_csv": str(folder / "{}_turns.csv".format(audio_id)),
        "acoustics_csv": str(folder / "{}_therapist_acoustics.csv".format(audio_id)),
        "acoustics_text_csv": str(folder / "{}_therapist_text_segments.csv".format(audio_id)),
        "log": str(folder / "run.log"),
    }
    job = {
        "id": job_id,
        "status": "queued",
        "message": "任务已创建",
        "failure_reason": "",
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": files,
        "input": {
            "audio_filename": audio.filename,
            "segment_model_id": segment_model_id,
            "audio_model_id": audio_model_id,
            "diarization_filename": diarization_json.filename if diarization_json else diarization_file_id,
        },
        "options": {
            "client_speaker": client_speaker.strip(),
            "therapist_speaker": therapist_speaker.strip(),
            "first_speaker_role": first_speaker_role,
            "threshold": float(threshold),
            "device": device.strip() or "cpu",
            "asr_device": asr_device.strip() or "cpu",
            "asr_model": asr_model.strip() or DEFAULT_ASR_MODEL,
            "vad_model": vad_model.strip() or DEFAULT_VAD_MODEL,
            "punc_model": punc_model.strip() or DEFAULT_PUNC_MODEL,
            "spk_model": spk_model.strip() or DEFAULT_SPK_MODEL,
            "hub": hub.strip() or "ms",
            "hotword": hotword.strip(),
            "run_acoustics": bool(run_acoustics),
        },
        "preview": {},
    }
    save_job(job)
    background_tasks.add_task(run_prediction_job, job_id)
    return public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return public_job(JOBS.get(job_id) or load_job(job_id))


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, tail: int = 200):
    job = JOBS.get(job_id) or load_job(job_id)
    log_path = job.get("files", {}).get("log")
    tail_text = tail_log_text(log_path, limit=max(4096, int(tail) * 200))
    display_text = clean_log_text(tail_text)
    tail_lines = [line for line in display_text.splitlines() if line.strip()]
    failure_reason = job.get("failure_reason", "")
    if job.get("status") == "failed" and not failure_reason:
        failure_reason = last_non_empty_line(display_text)
    return {
        "job_id": job_id,
        "status": job.get("status", ""),
        "message": job.get("message", ""),
        "failure_reason": failure_reason,
        "line_count": len(tail_lines),
        "tail_text": display_text,
    }


@app.get("/api/jobs/{job_id}/download/{kind}")
def download_job_file(job_id: str, kind: str):
    job = JOBS.get(job_id) or load_job(job_id)
    path = job.get("files", {}).get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Requested output does not exist")
    return FileResponse(path, filename=Path(path).name)
