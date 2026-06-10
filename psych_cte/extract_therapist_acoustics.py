"""Extract therapist acoustic features with openSMILE.

This module is independent from the existing CTE training and prediction
pipeline. It reads speaker diarization/role-assignment results, isolates the
therapist speech, extracts acoustic statistics, estimates therapist speech rate
from ASR text, and exports a merged CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, load_json


PITCH_HINTS = ("f0semitonefrom27.5hz",)
LOUDNESS_HINTS = ("loudness_sma3",)
PUNCTUATION_RE = re.compile(r"[\s\u3000\.,!?;:'\"`~，。！？；：、（）()【】\[\]《》<>“”‘’—…-]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")


def configure_funasr_torch_backend(device):
    if str(device or "").lower().startswith("cpu"):
        try:
            import torch

            torch.backends.mkldnn.enabled = False
            print("[stage] Disabled torch MKLDNN backend for FunASR CPU inference.", flush=True)
        except Exception:
            pass


def first_non_empty(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def load_segment_source(path):
    data = load_json(path)
    if isinstance(data, dict):
        segments = data.get("segments")
        if isinstance(segments, list):
            therapist = ""
            template = data.get("role_assignment_template")
            if isinstance(template, dict):
                therapist = str(first_non_empty(template.get("therapist_speaker"), data.get("therapist_speaker")))
            else:
                therapist = str(first_non_empty(data.get("therapist_speaker")))
            return data, segments, therapist
    if isinstance(data, list):
        return {}, data, ""
    raise SystemExit(
        "Unsupported diarization JSON format. Expected a manifest with a 'segments' list or a bare list of segments."
    )


def normalize_speaker(value):
    if value is None or value == "":
        return "unknown"
    if isinstance(value, int):
        return "spk{}".format(value)
    value = str(value).strip()
    if value.isdigit():
        return "spk{}".format(value)
    if value.startswith("SPEAKER_"):
        suffix = value.split("_", 1)[1].lstrip("0") or "0"
        return "speaker_{}".format(suffix)
    return value


def normalize_segments(records):
    normalized = []
    for item in records:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        speaker = first_non_empty(item.get("speaker"), item.get("source_speaker"), item.get("label"))
        normalized.append(
            {
                "start": float(start),
                "end": float(end),
                "speaker": normalize_speaker(speaker),
            }
        )
    return normalized


def select_therapist_segments(segments, therapist_speaker):
    therapist_speaker = normalize_speaker(str(therapist_speaker or "").strip())
    if not therapist_speaker:
        return []

    selected = []
    for seg in sorted(segments, key=lambda x: (x["start"], x["end"])):
        speaker = normalize_speaker(seg.get("speaker", ""))
        if speaker == therapist_speaker:
            selected.append(seg)
    return selected


def merge_overlaps(segments):
    merged = []
    for seg in sorted(segments, key=lambda x: (x["start"], x["end"])):
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        if merged and start < merged[-1]["end"]:
            start = merged[-1]["end"]
        if end <= start:
            continue
        merged.append({"start": start, "end": end, "speaker": normalize_speaker(seg.get("speaker", ""))})
    return merged


def build_smile():
    try:
        import opensmile
    except ImportError as exc:
        raise SystemExit(
            "The 'opensmile' package is not installed.\n"
            "Install it with:\n"
            "  pip install opensmile\n"
        ) from exc

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )


def find_column(columns, hints):
    lowered = {str(col).lower(): col for col in columns}
    for hint in hints:
        hint = hint.lower()
        for key, original in lowered.items():
            if hint in key:
                return original
    return ""


def extract_frames(audio_path, segments):
    smile = build_smile()
    frames = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        frame = smile.process_file(str(audio_path), start=start, end=end)
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def aggregate_features(frames):
    pitch_col = find_column(frames.columns, PITCH_HINTS)
    loudness_col = find_column(frames.columns, LOUDNESS_HINTS)
    if not pitch_col:
        raise RuntimeError("Could not find the openSMILE pitch column in the extracted features.")
    if not loudness_col:
        raise RuntimeError("Could not find the openSMILE loudness column in the extracted features.")

    pitch_series = pd.to_numeric(frames[pitch_col], errors="coerce").to_numpy(dtype=float)
    loudness_series = pd.to_numeric(frames[loudness_col], errors="coerce").to_numpy(dtype=float)

    pitch_series = pitch_series[np.isfinite(pitch_series)]
    loudness_series = loudness_series[np.isfinite(loudness_series)]

    voiced_pitch = pitch_series[pitch_series > 0]
    voiced_pitch_hz = 27.5 * np.power(2.0, voiced_pitch / 12.0) if voiced_pitch.size else np.array([])

    return {
        "pitch_frame_count": int(voiced_pitch_hz.size),
        "pitch_mean_hz": float(np.mean(voiced_pitch_hz)) if voiced_pitch_hz.size else float("nan"),
        "pitch_std_hz": float(np.std(voiced_pitch_hz, ddof=0)) if voiced_pitch_hz.size else float("nan"),
        "loudness_frame_count": int(loudness_series.size),
        "loudness_mean": float(np.mean(loudness_series)) if loudness_series.size else float("nan"),
        "loudness_std": float(np.std(loudness_series, ddof=0)) if loudness_series.size else float("nan"),
    }


def time_from_item(item):
    start = item.get("start")
    end = item.get("end")
    if start is not None and end is not None:
        return int(float(start)), int(float(end))

    timestamp = item.get("timestamp")
    if isinstance(timestamp, list) and timestamp:
        first = timestamp[0]
        last = timestamp[-1]
        if isinstance(first, (list, tuple)) and isinstance(last, (list, tuple)):
            return int(float(first[0])), int(float(last[-1]))
    return 0, 0


def normalize_asr_segments(asr_result):
    if isinstance(asr_result, list) and asr_result:
        asr_result = asr_result[0]
    if not isinstance(asr_result, dict):
        raise ValueError("Unexpected FunASR result format: {}".format(type(asr_result)))

    sentence_info = asr_result.get("sentence_info") or []
    turns = []
    if sentence_info:
        for idx, item in enumerate(sentence_info, 1):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            start, end = time_from_item(item)
            speaker = normalize_speaker(item.get("spk", item.get("speaker", item.get("spk_id"))))
            turns.append(
                {
                    "turn_id": idx,
                    "speaker": speaker,
                    "start_ms": start,
                    "end_ms": end,
                    "text": text,
                }
            )

    if not turns:
        text = (asr_result.get("text") or "").strip()
        if text:
            timestamp = asr_result.get("timestamp") or []
            start = int(timestamp[0][0]) if timestamp else 0
            end = int(timestamp[-1][-1]) if timestamp else 0
            turns.append(
                {
                    "turn_id": 1,
                    "speaker": "unknown",
                    "start_ms": start,
                    "end_ms": end,
                    "text": text,
                }
            )
    return turns


def run_funasr(args):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "The 'funasr' package is not installed, so speech rate cannot be computed from a new audio file.\n"
            "Install requirements or pass --turns-csv from an existing transcription."
        ) from exc

    configure_funasr_torch_backend(args.asr_device)
    model = AutoModel(
        model=args.asr_model,
        vad_model=args.vad_model,
        vad_kwargs={"max_single_segment_time": args.max_single_segment_time},
        punc_model=args.punc_model,
        device=args.asr_device,
        hub=args.hub,
        disable_update=True,
    )
    generate_kwargs = {
        "input": str(Path(args.audio).resolve()),
        "batch_size_s": args.batch_size_s,
    }
    if args.hotword:
        generate_kwargs["hotword"] = args.hotword
    return model.generate(**generate_kwargs)


def diarization_turns_from_segments(segments):
    turns = []
    for idx, seg in enumerate(segments, 1):
        turns.append(
            {
                "turn_id": idx,
                "speaker": normalize_speaker(seg.get("speaker", "")),
                "start_ms": int(float(seg.get("start", 0.0)) * 1000),
                "end_ms": int(float(seg.get("end", 0.0)) * 1000),
                "text": "",
                "source_turn_ids": [idx],
            }
        )
    return turns


def interval_distance(start_ms, end_ms, point_ms):
    if point_ms < start_ms:
        return start_ms - point_ms
    if point_ms > end_ms:
        return point_ms - end_ms
    return 0.0


def alignment_score(diarization_turn, asr_turn):
    a_start = asr_turn["start_ms"]
    a_end = asr_turn["end_ms"]
    a_mid = (a_start + a_end) / 2.0
    t_start = diarization_turn["start_ms"]
    t_end = diarization_turn["end_ms"]
    t_mid = (t_start + t_end) / 2.0
    overlap = max(0, min(t_end, a_end) - max(t_start, a_start))
    if overlap > 0:
        return (1, overlap, -abs(t_mid - a_mid), -(t_end - t_start), -t_start)
    return (0, -interval_distance(t_start, t_end, a_mid), -abs(t_mid - a_mid), -(t_end - t_start), -t_start)


def attach_asr_text_to_turns(diarization_turns, asr_turns):
    attached = [dict(turn) for turn in diarization_turns]
    pieces_by_turn = [[] for _ in diarization_turns]
    for asr_turn in asr_turns:
        text = (asr_turn.get("text") or "").strip()
        if not text:
            continue
        best_idx = None
        best_score = None
        for idx, turn in enumerate(diarization_turns):
            score = alignment_score(turn, asr_turn)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            pieces_by_turn[best_idx].append(text)
    for idx, turn in enumerate(attached):
        turn["text"] = "".join(pieces_by_turn[idx]).strip()
    return [turn for turn in attached if turn["text"]]


def load_turns_csv(path):
    turns = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            text = first_non_empty(row.get("text"), row.get("咨询师文本"), row.get("therapist_text"))
            speaker = first_non_empty(row.get("speaker"), row.get("therapist_speaker"))
            start_ms = first_non_empty(row.get("start_ms"), row.get("therapist_start_ms"))
            end_ms = first_non_empty(row.get("end_ms"), row.get("therapist_end_ms"))
            turns.append(
                {
                    "turn_id": first_non_empty(row.get("turn_id"), idx),
                    "role": row.get("role", ""),
                    "speaker": normalize_speaker(speaker),
                    "start_ms": int(float(start_ms)) if start_ms not in (None, "") else 0,
                    "end_ms": int(float(end_ms)) if end_ms not in (None, "") else 0,
                    "text": str(text or "").strip(),
                }
            )
    return [turn for turn in turns if turn.get("text")]


def build_text_turns(args, segments, therapist_speaker):
    if args.turns_csv:
        turns = load_turns_csv(args.turns_csv)
    else:
        asr_result = run_funasr(args)
        asr_turns = normalize_asr_segments(asr_result)
        diarization_turns = diarization_turns_from_segments(segments)
        turns = attach_asr_text_to_turns(diarization_turns, asr_turns)

    therapist_speaker = normalize_speaker(therapist_speaker)
    selected = []
    for turn in turns:
        role = str(turn.get("role", "")).strip().lower()
        speaker = normalize_speaker(turn.get("speaker", ""))
        if role == "therapist" or speaker == therapist_speaker:
            item = dict(turn)
            item["speaker"] = speaker
            selected.append(item)
    return selected


def count_speech_units(text, unit):
    text = str(text or "")
    if unit == "word":
        return len(TOKEN_RE.findall(text))
    cleaned = PUNCTUATION_RE.sub("", text)
    return len(cleaned)


def compute_speech_rate(text_turns, duration_s, unit):
    texts = [(turn.get("text") or "").strip() for turn in text_turns if (turn.get("text") or "").strip()]
    text = "".join(texts)
    count = count_speech_units(text, unit)
    rate_per_second = count / duration_s if duration_s > 0 else float("nan")
    return {
        "speech_rate_unit": unit,
        "speech_unit_count": int(count),
        "speech_rate_per_second": float(rate_per_second) if np.isfinite(rate_per_second) else float("nan"),
        "speech_rate_per_minute": float(rate_per_second * 60.0) if np.isfinite(rate_per_second) else float("nan"),
        "text_turn_count": len(text_turns),
        "therapist_text": text,
    }


def export_csv(path, row):
    path = ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def export_text_turns_csv(path, audio_id, turns):
    path = ensure_dir(path)
    rows = []
    for turn in turns:
        rows.append(
            {
                "audio_id": audio_id,
                "turn_id": turn.get("turn_id", ""),
                "speaker": turn.get("speaker", ""),
                "start_ms": turn.get("start_ms", ""),
                "end_ms": turn.get("end_ms", ""),
                "duration_s": round(max(0, int(turn.get("end_ms", 0)) - int(turn.get("start_ms", 0))) / 1000.0, 3),
                "text": turn.get("text", ""),
            }
        )
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["audio_id"])
        writer.writeheader()
        writer.writerows(rows)


def finite_or_blank(value):
    return round(value, 6) if np.isfinite(value) else ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract therapist acoustic features and speech rate, then export a merged CSV."
    )
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument(
        "--diarization-json",
        required=True,
        help="Speaker diarization manifest or segment list used to isolate the therapist",
    )
    parser.add_argument("--therapist-speaker", default="", help="Therapist speaker label, for example speaker_1")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output folder. Defaults to outputs/therapist_acoustic_features/<audio_id>",
    )
    parser.add_argument("--output-csv", default="", help="Output merged CSV path")
    parser.add_argument("--output-text-csv", default="", help="Optional therapist text-detail CSV path")
    parser.add_argument(
        "--turns-csv",
        default="",
        help="Optional existing turns CSV. If omitted, this script runs FunASR and aligns text to diarization.",
    )
    parser.add_argument(
        "--speech-rate-unit",
        choices=["char", "word"],
        default="char",
        help="Use Chinese character count or simple token count for speech rate.",
    )
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument("--spk-model", default="cam++")
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--max-single-segment-time", type=int, default=60000)
    parser.add_argument("--hotword", default="")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    audio_id = audio_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "therapist_acoustic_features" / audio_id
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "{}_therapist_acoustics.csv".format(audio_id)
    output_text_csv = Path(args.output_text_csv) if args.output_text_csv else output_dir / "{}_therapist_text_segments.csv".format(audio_id)

    manifest, records, manifest_therapist = load_segment_source(args.diarization_json)
    segments = normalize_segments(records)
    therapist_speaker = normalize_speaker(args.therapist_speaker.strip() or manifest_therapist)

    therapist_segments = merge_overlaps(select_therapist_segments(segments, therapist_speaker))
    if not therapist_segments:
        raise SystemExit("No therapist segments were found. Check --diarization-json and --therapist-speaker.")

    frames = extract_frames(audio_path, therapist_segments)
    if frames.empty:
        raise SystemExit("openSMILE did not return any usable frames for the therapist segments.")

    total_duration = sum(seg["end"] - seg["start"] for seg in therapist_segments)
    acoustic_features = aggregate_features(frames)
    text_turns = build_text_turns(args, segments, therapist_speaker)
    speech_rate = compute_speech_rate(text_turns, total_duration, args.speech_rate_unit)

    row = {
        "audio_id": manifest.get("audio_id", audio_id),
        "audio_file": str(audio_path.resolve()),
        "therapist_speaker": therapist_speaker,
        "therapist_segment_count": len(therapist_segments),
        "therapist_speech_duration_s": round(float(total_duration), 3),
        "speech_rate_unit": speech_rate["speech_rate_unit"],
        "speech_unit_count": speech_rate["speech_unit_count"],
        "speech_rate_per_second": finite_or_blank(speech_rate["speech_rate_per_second"]),
        "speech_rate_per_minute": finite_or_blank(speech_rate["speech_rate_per_minute"]),
        "speech_rate_text_turn_count": speech_rate["text_turn_count"],
        "pitch_mean_hz": finite_or_blank(acoustic_features["pitch_mean_hz"]),
        "pitch_std_hz": finite_or_blank(acoustic_features["pitch_std_hz"]),
        "loudness_mean": finite_or_blank(acoustic_features["loudness_mean"]),
        "loudness_std": finite_or_blank(acoustic_features["loudness_std"]),
        "pitch_frame_count": acoustic_features["pitch_frame_count"],
        "loudness_frame_count": acoustic_features["loudness_frame_count"],
        "therapist_text": speech_rate["therapist_text"],
        "source_diarization_json": str(Path(args.diarization_json).resolve()),
        "source_turns_csv": str(Path(args.turns_csv).resolve()) if args.turns_csv else "FunASR generated in this run",
    }

    export_csv(output_csv, row)
    export_text_turns_csv(output_text_csv, row["audio_id"], text_turns)
    print(
        json.dumps(
            {
                "output_csv": str(output_csv.resolve()),
                "output_text_csv": str(output_text_csv.resolve()),
                "audio_id": row["audio_id"],
                "therapist_speaker": therapist_speaker,
                "speech_rate_per_minute": row["speech_rate_per_minute"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
