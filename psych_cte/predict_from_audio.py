"""Predict empathy units directly from an audio file.

Pipeline:
1. Run FunASR ASR + VAD + punctuation + speaker diarization.
2. Map speakers to client/therapist roles.
3. Build empathy units from client turn(s) followed by therapist response(s).
4. Run the trained segment-level CTE model on each unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .constants import BLOCKING_TYPES, EMPATHY_LABELS
from .data import make_audio_windows
from .modeling import AudioCTEModel, LocalWeightedAudioCTEModel, SegmentCTEModel, resolve_encoder_path
from .rationale_utils import load_rationale_bank, rationale_bank_path, topk_rationales
from .utils import dump_json, ensure_dir


def configure_funasr_torch_backend(device):
    if str(device or "").lower().startswith("cpu"):
        try:
            torch.backends.mkldnn.enabled = False
            print("[stage] Disabled torch MKLDNN backend for FunASR CPU inference.", flush=True)
        except Exception:
            pass


EMPATHY_CN = {
    "emotion_reflection": "内容情感反映",
    "deep_meaning_understanding": "深层意义理解",
    "acceptance_confirmation": "接纳确认",
    "exploration_facilitation": "促进探索",
    "blocking_present": "共情阻碍",
}

BLOCKING_CN = {
    "none": "无",
    "premature_advice": "过早建议",
    "judgment_blame": "评价责备",
    "minimization": "淡化感受",
    "topic_shift": "转移话题",
    "vague_response": "空泛回应",
    "other": "其他",
}


def ms_to_time(ms):
    ms = int(ms or 0)
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return "{:02d}:{:02d}".format(minutes, seconds)


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
    segments = []
    if sentence_info:
        for idx, item in enumerate(sentence_info, 1):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            start, end = time_from_item(item)
            speaker = normalize_speaker(
                item.get("spk", item.get("speaker", item.get("spk_id")))
            )
            segments.append(
                {
                    "turn_id": idx,
                    "speaker": speaker,
                    "start_ms": start,
                    "end_ms": end,
                    "text": text,
                    "raw": item,
                }
            )

    if not segments:
        text = (asr_result.get("text") or "").strip()
        if text:
            timestamp = asr_result.get("timestamp") or []
            start = int(timestamp[0][0]) if timestamp else 0
            end = int(timestamp[-1][-1]) if timestamp else 0
            segments.append(
                {
                    "turn_id": 1,
                    "speaker": "unknown",
                    "start_ms": start,
                    "end_ms": end,
                    "text": text,
                    "raw": asr_result,
                }
            )
    return segments


def run_funasr(args):
    from funasr import AutoModel

    configure_funasr_torch_backend(args.asr_device)
    model_kwargs = {
        "model": args.asr_model,
        "vad_model": args.vad_model,
        "vad_kwargs": {"max_single_segment_time": args.max_single_segment_time},
        "punc_model": args.punc_model,
        "device": args.asr_device,
        "hub": args.hub,
        "disable_update": True,
    }
    if getattr(args, "diarization_json", ""):
        print("[stage] Imported diarization JSON detected; skipping FunASR speaker model.", flush=True)
    else:
        model_kwargs["spk_model"] = args.spk_model
    model = AutoModel(**model_kwargs)
    generate_kwargs = {
        "input": str(Path(args.audio).resolve()),
        "batch_size_s": args.batch_size_s,
    }
    if args.hotword:
        generate_kwargs["hotword"] = args.hotword
    return model.generate(**generate_kwargs)


def merge_adjacent_turns(turns, merge_gap_ms):
    turns = sorted(turns, key=lambda x: (x["start_ms"], x["end_ms"]))
    merged = []
    for turn in turns:
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and turn["start_ms"] - merged[-1]["end_ms"] <= merge_gap_ms
        ):
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], turn["end_ms"])
            merged[-1]["text"] = (merged[-1]["text"] + turn["text"]).strip()
            merged[-1]["source_turn_ids"].append(turn["turn_id"])
        else:
            item = dict(turn)
            item["source_turn_ids"] = [turn["turn_id"]]
            merged.append(item)

    for idx, turn in enumerate(merged, 1):
        turn["turn_id"] = idx
    return merged


def turns_from_diarization_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    turns = []
    for idx, seg in enumerate(segments, 1):
        turns.append(
            {
                "turn_id": idx,
                "speaker": normalize_speaker(seg.get("speaker", seg.get("source_speaker", ""))),
                "start_ms": int(float(seg.get("start", 0.0)) * 1000),
                "end_ms": int(float(seg.get("end", 0.0)) * 1000),
                "text": "",
                "source_turn_ids": [idx],
            }
        )
    return turns


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


def char_timeline_from_asr_turn(asr_turn):
    text = (asr_turn.get("text") or "").strip()
    raw = asr_turn.get("raw") or {}
    timestamps = raw.get("timestamp") if isinstance(raw, dict) else None
    if not text or not isinstance(timestamps, list):
        return []

    chars = [ch for ch in text if not ch.isspace()]
    pairs = []
    for item in timestamps:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and item[0] is not None
            and item[1] is not None
        ):
            pairs.append((int(float(item[0])), int(float(item[1]))))

    count = min(len(chars), len(pairs))
    if count == 0:
        return []
    return [
        {"text": chars[idx], "start_ms": pairs[idx][0], "end_ms": pairs[idx][1]}
        for idx in range(count)
    ]


def split_full_asr_turn_by_diarization(asr_turn, diarization_turns):
    timeline = char_timeline_from_asr_turn(asr_turn)
    if not timeline:
        return []

    split_turns = []
    for idx, diar_turn in enumerate(diarization_turns, 1):
        start_ms = diar_turn["start_ms"]
        end_ms = diar_turn["end_ms"]
        pieces = []
        piece_starts = []
        piece_ends = []
        for item in timeline:
            mid_ms = (item["start_ms"] + item["end_ms"]) / 2.0
            if start_ms <= mid_ms <= end_ms:
                pieces.append(item["text"])
                piece_starts.append(item["start_ms"])
                piece_ends.append(item["end_ms"])
        text = "".join(pieces).strip()
        if not text:
            continue
        split_turns.append(
            {
                "turn_id": idx,
                "speaker": diar_turn["speaker"],
                "start_ms": min(piece_starts) if piece_starts else start_ms,
                "end_ms": max(piece_ends) if piece_ends else end_ms,
                "text": text,
                "source_turn_ids": [diar_turn.get("turn_id", idx)],
            }
        )
    return split_turns


def overlap_ms(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


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
        return (
            1,
            overlap,
            -abs(t_mid - a_mid),
            -(t_end - t_start),
            -t_start,
        )
    return (
        0,
        -interval_distance(t_start, t_end, a_mid),
        -abs(t_mid - a_mid),
        -(t_end - t_start),
        -t_start,
    )


def infer_role_map(turns, client_speaker, therapist_speaker, first_speaker_role):
    speakers = []
    for turn in turns:
        spk = turn["speaker"]
        if spk not in speakers:
            speakers.append(spk)

    if client_speaker and therapist_speaker:
        return {client_speaker: "client", therapist_speaker: "therapist"}

    if len(speakers) < 2:
        raise ValueError(
            "FunASR only found one speaker. Please check diarization output or provide a better audio file."
        )

    first = speakers[0]
    second = speakers[1]
    if first_speaker_role == "therapist":
        return {first: "therapist", second: "client"}
    return {first: "client", second: "therapist"}


def assign_roles(turns, role_map):
    assigned = []
    for turn in turns:
        role = role_map.get(turn["speaker"], "other")
        item = dict(turn)
        item["role"] = role
        assigned.append(item)
    return assigned


def join_turn_text(turns):
    return "".join((turn.get("text") or "").strip() for turn in turns).strip()


def turn_ids(turns):
    ids = []
    for turn in turns:
        ids.extend(turn.get("source_turn_ids", [turn.get("turn_id")]))
    return [item for item in ids if item is not None]


def make_empathy_unit(audio_id, unit_index, client_turns, therapist_turns):
    client_text = join_turn_text(client_turns)
    therapist_text = join_turn_text(therapist_turns)
    client_start = min(turn["start_ms"] for turn in client_turns)
    client_end = max(turn["end_ms"] for turn in client_turns)
    therapist_start = min(turn["start_ms"] for turn in therapist_turns)
    therapist_end = max(turn["end_ms"] for turn in therapist_turns)
    start_ms = min(client_start, therapist_start)
    end_ms = max(client_end, therapist_end)

    return {
        "audio_id": audio_id,
        "segment_id": "unit_{:04d}".format(unit_index),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "共情单元起止时间": "{}-{}".format(ms_to_time(start_ms), ms_to_time(end_ms)),
        "client_start_ms": client_start,
        "client_end_ms": client_end,
        "therapist_start_ms": therapist_start,
        "therapist_end_ms": therapist_end,
        "来访者表达区间": "{}-{}".format(ms_to_time(client_start), ms_to_time(client_end)),
        "咨询师回应区间": "{}-{}".format(ms_to_time(therapist_start), ms_to_time(therapist_end)),
        "client_speaker": ",".join(sorted({turn["speaker"] for turn in client_turns})),
        "therapist_speaker": ",".join(sorted({turn["speaker"] for turn in therapist_turns})),
        "client_turn_ids": turn_ids(client_turns),
        "therapist_turn_ids": turn_ids(therapist_turns),
        "client_text": client_text,
        "therapist_text": therapist_text,
        "来访文本": client_text,
        "咨询师文本": therapist_text,
    }


def build_empathy_units(turns, audio_id):
    units = []
    client_turns = []
    therapist_turns = []

    for turn in turns:
        role = turn.get("role")
        if role == "client":
            if client_turns and therapist_turns:
                units.append(
                    make_empathy_unit(audio_id, len(units) + 1, client_turns, therapist_turns)
                )
                client_turns = []
                therapist_turns = []
            client_turns.append(dict(turn))
        elif role == "therapist":
            if client_turns:
                therapist_turns.append(dict(turn))
        elif client_turns and therapist_turns:
            units.append(make_empathy_unit(audio_id, len(units) + 1, client_turns, therapist_turns))
            client_turns = []
            therapist_turns = []

    if client_turns and therapist_turns:
        units.append(make_empathy_unit(audio_id, len(units) + 1, client_turns, therapist_turns))

    return units

    units = []
    current_client = None

    for turn in turns:
        role = turn.get("role")
        if role == "client":
            current_client = dict(turn)
        elif role == "therapist" and current_client is not None:
            unit_index = len(units) + 1
            start_ms = current_client["start_ms"]
            end_ms = turn["end_ms"]
            units.append(
                {
                    "audio_id": audio_id,
                    "segment_id": "unit_{:04d}".format(unit_index),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "共情单元起止时间": "{}-{}".format(ms_to_time(start_ms), ms_to_time(end_ms)),
                    "client_start_ms": current_client["start_ms"],
                    "client_end_ms": current_client["end_ms"],
                    "therapist_start_ms": turn["start_ms"],
                    "therapist_end_ms": turn["end_ms"],
                    "来访者表达区间": "{}-{}".format(
                        ms_to_time(current_client["start_ms"]), ms_to_time(current_client["end_ms"])
                    ),
                    "咨询师回应区间": "{}-{}".format(
                        ms_to_time(turn["start_ms"]), ms_to_time(turn["end_ms"])
                    ),
                    "client_speaker": current_client["speaker"],
                    "therapist_speaker": turn["speaker"],
                    "client_text": current_client["text"],
                    "therapist_text": turn["text"],
                    "来访文本": current_client["text"],
                    "咨询师文本": turn["text"],
                }
            )
            current_client = None

    return units


def load_segment_predictor(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_encoder_path(ckpt["encoder"]), use_fast=True, local_files_only=True
    )
    model = SegmentCTEModel(ckpt["encoder"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt.get("max_length", 256)


def predict_segment(model, tokenizer, unit, max_length, threshold, device, rationale_bank=None):
    enc = tokenizer(
        [unit["client_text"]],
        [unit["therapist_text"]],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)

    empathy_probs = torch.sigmoid(out["empathy_logits"]).squeeze(0).cpu().tolist()
    blocking_probs = torch.softmax(out["blocking_logits"], dim=-1).squeeze(0).cpu().tolist()
    blocking_idx = int(torch.argmax(out["blocking_logits"], dim=-1).item())
    blocking_type = BLOCKING_TYPES[blocking_idx]
    evidence = []
    if rationale_bank:
        evidence = topk_rationales(out["input_repr"].squeeze(0).detach().cpu().tolist(), rationale_bank, top_k=3)

    empathy = {}
    for idx, name in enumerate(EMPATHY_LABELS):
        empathy[EMPATHY_CN[name]] = {
            "prob": round(float(empathy_probs[idx]), 4),
            "present": bool(empathy_probs[idx] >= threshold),
        }

    return {
        "cte_score": round(float(out["cte_pred"].item()), 4),
        "empathy": empathy,
        "blocking_type": BLOCKING_CN[blocking_type],
        "blocking_probabilities": {
            BLOCKING_CN[name]: round(float(prob), 4)
            for name, prob in zip(BLOCKING_TYPES, blocking_probs)
        },
        "rationale_evidence": evidence,
    }


def predict_segments_batch(
    model,
    tokenizer,
    units,
    max_length,
    threshold,
    device,
    rationale_bank=None,
    batch_size=16,
):
    predictions = []
    batch_size = max(1, int(batch_size))
    total = len(units)
    for start in range(0, total, batch_size):
        batch_units = units[start : start + batch_size]
        end = start + len(batch_units)
        print("[stage] Predicting segments {}/{}...".format(end, total), flush=True)
        enc = tokenizer(
            [unit["client_text"] for unit in batch_units],
            [unit["therapist_text"] for unit in batch_units],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)

        empathy_probs = torch.sigmoid(out["empathy_logits"]).detach().cpu().tolist()
        blocking_probs = torch.softmax(out["blocking_logits"], dim=-1).detach().cpu().tolist()
        blocking_indices = torch.argmax(out["blocking_logits"], dim=-1).detach().cpu().tolist()
        cte_scores = out["cte_pred"].detach().cpu().view(-1).tolist()
        input_reprs = (
            out["input_repr"].detach().cpu().tolist()
            if rationale_bank and "input_repr" in out
            else [None] * len(batch_units)
        )

        for idx, probs in enumerate(empathy_probs):
            empathy = {}
            for label_idx, name in enumerate(EMPATHY_LABELS):
                prob = probs[label_idx]
                empathy[EMPATHY_CN[name]] = {
                    "prob": round(float(prob), 4),
                    "present": bool(float(prob) >= threshold),
                }

            blocking_idx = int(blocking_indices[idx])
            blocking_type = BLOCKING_TYPES[blocking_idx]
            evidence = []
            if rationale_bank and input_reprs[idx] is not None:
                evidence = topk_rationales(input_reprs[idx], rationale_bank, top_k=3)

            predictions.append(
                {
                    "cte_score": round(float(cte_scores[idx]), 4),
                    "empathy": empathy,
                    "blocking_type": BLOCKING_CN[blocking_type],
                    "blocking_probabilities": {
                        BLOCKING_CN[name]: round(float(prob), 4)
                        for name, prob in zip(BLOCKING_TYPES, blocking_probs[idx])
                    },
                    "rationale_evidence": evidence,
                }
            )
    return predictions


def predict_audio_cte(audio_checkpoint, record, device):
    ckpt = torch.load(audio_checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_encoder_path(ckpt["encoder"]), use_fast=True, local_files_only=True
    )
    if ckpt.get("model_type") == "local_weighted_audio":
        model = LocalWeightedAudioCTEModel(ckpt["encoder"], max_segments=ckpt.get("max_segments", 128))
    else:
        model = AudioCTEModel(ckpt["encoder"], max_segments=ckpt.get("max_segments", 128))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    window_stride = ckpt.get("window_stride", 0)
    if not window_stride:
        window_stride = max(1, ckpt.get("max_segments", 128) // 2)
    windows = make_audio_windows(record, ckpt.get("max_segments", 128), window_stride)
    if not windows:
        return {
            "audio_cte_score": None,
            "segment_weights": [],
            "local_cte_scores": [],
            "weighted_contributions": [],
            "window_predictions": [],
        }

    total_segments = len(record.get("segments", []))
    total_window_weight = float(sum(len(window.get("segments", [])) for window in windows))
    audio_pred_sum = 0.0
    audio_weight_sum = 0.0
    segment_weight_sum = [0.0] * total_segments
    local_score_sum = [0.0] * total_segments
    local_score_count = [0] * total_segments
    contribution_sum = [0.0] * total_segments
    window_predictions = []
    with torch.no_grad():
        for window_index, window in enumerate(windows, start=1):
            if window_index == 1 or window_index == len(windows) or window_index % 10 == 0:
                print(
                    "[stage] Audio-level window {}/{}...".format(window_index, len(windows)),
                    flush=True,
                )
            segments = window.get("segments", [])
            client_texts = [seg.get("client_text", "") for seg in segments]
            therapist_texts = [seg.get("therapist_text", "") for seg in segments]
            enc = tokenizer(
                client_texts,
                therapist_texts,
                padding=True,
                truncation=True,
                max_length=ckpt.get("max_length", 256),
                return_tensors="pt",
            )
            batch = {
                "input_ids": enc["input_ids"].unsqueeze(0).to(device),
                "attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
                "segment_mask": torch.ones((1, len(segments)), dtype=torch.bool).to(device),
            }
            if "token_type_ids" in enc:
                batch["token_type_ids"] = enc["token_type_ids"].unsqueeze(0).to(device)
            out = model(**batch)
            score = float(out["audio_pred"].item())
            weights = out["segment_weights"].squeeze(0).detach().cpu().tolist()
            local_scores = out.get("local_cte_pred")
            local_scores = local_scores.squeeze(0).detach().cpu().tolist() if local_scores is not None else []
            weight = float(len(segments))
            global_window_weight = weight / max(total_window_weight, 1.0)
            audio_pred_sum += score * weight
            audio_weight_sum += weight
            start = int(window.get("window_start", 0))
            for offset, value in enumerate(weights):
                seg_idx = start + offset
                if 0 <= seg_idx < total_segments:
                    global_weight = global_window_weight * float(value)
                    segment_weight_sum[seg_idx] += global_weight
                    if offset < len(local_scores):
                        local_score = float(local_scores[offset])
                        local_score_sum[seg_idx] += local_score
                        local_score_count[seg_idx] += 1
                        contribution_sum[seg_idx] += global_weight * local_score
            window_predictions.append(
                {
                    "window_id": window.get("window_id", ""),
                    "window_index": window.get("window_index", ""),
                    "window_start": start,
                    "window_end": int(window.get("window_end", start + len(segments))),
                    "segment_count": len(segments),
                    "audio_cte_score": round(score, 4),
                }
            )

    audio_cte_score = audio_pred_sum / max(audio_weight_sum, 1.0)
    weight_total = sum(segment_weight_sum)
    if weight_total > 0:
        segment_weight_sum = [value / weight_total for value in segment_weight_sum]
    segment_weights = [round(value, 4) for value in segment_weight_sum]
    local_cte_scores = [
        round(total / count, 4) if count else None
        for total, count in zip(local_score_sum, local_score_count)
    ]
    weighted_contributions = [round(value, 4) for value in contribution_sum]
    return {
        "audio_cte_score": round(float(audio_cte_score), 4),
        "segment_weights": segment_weights,
        "local_cte_scores": local_cte_scores,
        "weighted_contributions": weighted_contributions,
        "window_predictions": window_predictions,
    }


def first_present(record, keys, default=""):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def collect_local_cte_scores(record):
    items = []
    scores = []
    for unit in record.get("segments", []):
        pred = unit.get("prediction", {})
        score = pred.get("cte_score", None)
        if score is None:
            continue
        score = float(score)
        scores.append(score)
        items.append(
            {
                "segment_id": unit.get("segment_id", ""),
                "time": first_present(unit, ["共情单元起止时间", "鍏辨儏鍗曞厓璧锋鏃堕棿"]),
                "client_text": first_present(unit, ["来访文本", "鏉ヨ鏂囨湰", "client_text"]),
                "therapist_text": first_present(unit, ["咨询师文本", "鍜ㄨ甯堟枃鏈?", "therapist_text"]),
                "cte_score": round(score, 4),
                "blocking_type": pred.get("blocking_type", ""),
            }
        )
    return scores, items


def summarize_empathy_labels(record, threshold=0.5):
    sums = {name: 0.0 for name in EMPATHY_LABELS}
    counts = {name: 0 for name in EMPATHY_LABELS}

    for unit in record.get("segments", []):
        empathy = (unit.get("prediction", {}) or {}).get("empathy", {}) or {}
        for name in EMPATHY_LABELS:
            cn_name = EMPATHY_CN[name]
            item = empathy.get(cn_name, {})
            if not isinstance(item, dict):
                continue
            prob = item.get("prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            sums[name] += prob
            counts[name] += 1

    labels = {}
    for name in EMPATHY_LABELS:
        avg_prob = None
        if counts[name]:
            avg_prob = sums[name] / counts[name]
        if name == "blocking_present":
            status = (
                "存在明显阻碍"
                if avg_prob is not None and avg_prob > threshold
                else "未见明显阻碍"
            )
        else:
            status = (
                "已体现"
                if avg_prob is not None and avg_prob > threshold
                else "未明显体现"
            )
        labels[name] = {
            "label": EMPATHY_CN[name],
            "average_prob": round(float(avg_prob), 4) if avg_prob is not None else None,
            "count": counts[name],
            "threshold": float(threshold),
            "status": status,
            "present": bool(avg_prob is not None and avg_prob > threshold),
        }
    return labels


def summarize_local_cte_scores(record):
    scores, items = collect_local_cte_scores(record)
    if not scores:
        return {
            "count": 0,
            "scores": [],
            "items": [],
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "low_count": 0,
            "mid_count": 0,
            "high_count": 0,
        }

    mean_score = sum(scores) / len(scores)
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    low_count = sum(1 for score in scores if score < 0.4)
    mid_count = sum(1 for score in scores if 0.4 <= score < 0.7)
    high_count = sum(1 for score in scores if score >= 0.7)
    return {
        "count": len(scores),
        "scores": [round(score, 4) for score in scores],
        "items": items,
        "mean": round(mean_score, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(std_score, 4),
        "low_count": low_count,
        "mid_count": mid_count,
        "high_count": high_count,
    }


def build_audio_summary(record):
    audio_prediction = record.get("audio_prediction", {}) or {}
    local_summary = summarize_local_cte_scores(record)
    empathy_label_summary = summarize_empathy_labels(record)
    local_items = local_summary.get("items", [])
    local_scores = local_summary.get("scores", [])

    audio_score = audio_prediction.get("audio_cte_score", None)
    if audio_score is None and local_summary.get("mean") is not None:
        audio_score = local_summary["mean"]

    segment_weights = audio_prediction.get("segment_weights", []) or []
    weighted_contributions = audio_prediction.get("weighted_contributions", []) or []
    audio_local_scores = audio_prediction.get("local_cte_scores", []) or []
    weighted_units = []
    for idx, unit in enumerate(record.get("segments", [])):
        pred = unit.get("prediction", {})
        score = pred.get("cte_score", None)
        if score is None:
            continue
        weight = segment_weights[idx] if idx < len(segment_weights) else None
        contribution = weighted_contributions[idx] if idx < len(weighted_contributions) else None
        audio_local_score = audio_local_scores[idx] if idx < len(audio_local_scores) else None
        weighted_units.append(
            {
                "segment_id": unit.get("segment_id", ""),
                "time": first_present(unit, ["共情单元起止时间", "鍏辨儏鍗曞厓璧锋鏃堕棿"]),
                "cte_score": round(float(score), 4),
                "weight": round(float(weight), 4) if weight is not None else None,
                "audio_local_cte_score": round(float(audio_local_score), 4) if audio_local_score is not None else None,
                "weighted_contribution": round(float(contribution), 4) if contribution is not None else None,
                "client_text": first_present(unit, ["来访文本", "鏉ヨ鏂囨湰", "client_text"]),
                "therapist_text": first_present(unit, ["咨询师文本", "鍜ㄨ甯堟枃鏈?", "therapist_text"]),
                "blocking_type": pred.get("blocking_type", ""),
            }
        )

    if segment_weights:
        weighted_units.sort(
            key=lambda item: (
                item["weight"] is None,
                -(item["weight"] if item["weight"] is not None else -1.0),
                item["segment_id"],
            )
        )
    else:
        weighted_units.sort(key=lambda item: (-item["cte_score"], item["segment_id"]))

    top_units = weighted_units[:2]
    if audio_score is None:
        rationale = "未能生成整段CTE总分，因此只能返回局部分数汇总。"
    else:
        if audio_score >= 0.7:
            lead = "整段分数偏高，说明高权重片段里有较多明确的共情回应。"
        elif audio_score <= 0.4:
            lead = "整段分数偏低，说明关键片段里共情表达不足或更偏离来访者当下体验。"
        else:
            lead = "整段分数处于中间水平，说明有共情回应，但强度和稳定性不够一致。"
        details = []
        for item in top_units:
            weight_part = ""
            if item["weight"] is not None:
                weight_part = "，权重{:.4f}".format(item["weight"])
            details.append(
                "{}({}，局部分数{:.4f}{})".format(
                    item["segment_id"],
                    item["time"],
                    item["cte_score"],
                    weight_part,
                )
            )
        detail_text = "；重点片段是{}".format("、".join(details)) if details else ""
        rationale = "整段CTE总分{:.4f}。{}局部CTE分数分布为{}".format(
            float(audio_score),
            lead,
            "、".join("{:.4f}".format(score) for score in local_scores) if local_scores else "空",
        )
        if detail_text:
            rationale += detail_text

    return {
        "audio_cte_score": round(float(audio_score), 4) if audio_score is not None else None,
        "local_cte_scores": local_scores,
        "local_cte_items": local_items,
        "local_cte_summary": local_summary,
        "empathy_label_summary": empathy_label_summary,
        "audio_cte_rationale": rationale,
        "top_weighted_segments": top_units,
    }


def enrich_record_with_summary(record):
    summary = build_audio_summary(record)
    record["analysis_summary"] = dict(record.get("analysis_summary", {}))
    record["analysis_summary"].update(
        {
            "local_cte_scores": summary["local_cte_scores"],
            "local_cte_summary": summary["local_cte_summary"],
            "audio_cte_score": summary["audio_cte_score"],
            "audio_cte_rationale": summary["audio_cte_rationale"],
            "empathy_label_summary": summary["empathy_label_summary"],
            "top_weighted_segments": summary["top_weighted_segments"],
        }
    )
    record["audio_summary"] = summary
    return record


def export_units_csv(path, record):
    summary = record.get("audio_summary", {}) or {}
    audio_prediction = record.get("audio_prediction", {}) or {}
    segment_weights = audio_prediction.get("segment_weights", []) or []
    weighted_contributions = audio_prediction.get("weighted_contributions", []) or []
    audio_local_scores = audio_prediction.get("local_cte_scores", []) or []
    rows = []
    for idx, unit in enumerate(record.get("segments", [])):
        pred = unit.get("prediction", {})
        rows.append(
            {
                "audio_id": record.get("audio_id", ""),
                "segment_id": unit.get("segment_id", ""),
                "共情单元起止时间": first_present(unit, ["共情单元起止时间", "鍏辨儏鍗曞厓璧锋鏃堕棿"]),
                "来访者表达区间": first_present(unit, ["来访者表达区间", "鏉ヨ鑰呰〃杈惧尯闂?"]),
                "咨询师回应区间": first_present(unit, ["咨询师回应区间", "鍜ㄨ甯堝洖搴斿尯闂?"]),
                "来访文本": first_present(unit, ["来访文本", "鏉ヨ鏂囨湰", "client_text"]),
                "咨询师文本": first_present(unit, ["咨询师文本", "鍜ㄨ甯堟枃鏈?", "therapist_text"]),
                "预测CTE分数": pred.get("cte_score", ""),
                "预测阻碍类型": pred.get("blocking_type", ""),
                "预测共情标签": json.dumps(pred.get("empathy", {}), ensure_ascii=False),
                "预测依据": json.dumps(pred.get("rationale_evidence", []), ensure_ascii=False),
                "整段CTE总分": summary.get("audio_cte_score", ""),
                "整段CTE简短依据": summary.get("audio_cte_rationale", ""),
                "局部CTE分数汇总": json.dumps(summary.get("local_cte_scores", []), ensure_ascii=False),
                "局部CTE统计": json.dumps(summary.get("local_cte_summary", {}), ensure_ascii=False),
            }
        )

    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return

    rows = []
    for unit in record.get("segments", []):
        pred = unit.get("prediction", {})
        rows.append(
            {
                "audio_id": record.get("audio_id", ""),
                "segment_id": unit.get("segment_id", ""),
                "共情单元起止时间": unit.get("共情单元起止时间", ""),
                "来访者表达区间": unit.get("来访者表达区间", ""),
                "咨询师回应区间": unit.get("咨询师回应区间", ""),
                "来访文本": unit.get("来访文本", ""),
                "咨询师文本": unit.get("咨询师文本", ""),
                "预测CTE分数": pred.get("cte_score", ""),
                "预测阻碍类型": pred.get("blocking_type", ""),
                "预测共情标签": json.dumps(pred.get("empathy", {}), ensure_ascii=False),
            }
        )

    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def export_turns_csv(path, record):
    rows = []
    for turn in record.get("turns", []):
        rows.append(
            {
                "audio_id": record.get("audio_id", ""),
                "turn_id": turn.get("turn_id", ""),
                "role": turn.get("role", ""),
                "speaker": turn.get("speaker", ""),
                "start_time": ms_to_time(turn.get("start_ms", 0)),
                "end_time": ms_to_time(turn.get("end_ms", 0)),
                "start_ms": turn.get("start_ms", ""),
                "end_ms": turn.get("end_ms", ""),
                "text": turn.get("text", ""),
                "source_turn_ids": json.dumps(turn.get("source_turn_ids", []), ensure_ascii=False),
            }
        )

    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict CTE units directly from audio.")
    parser.add_argument("--audio", required=True, help="Input .wav/.mp3/.m4a audio file")
    parser.add_argument("--segment-checkpoint", required=True, help="Segment model checkpoint")
    parser.add_argument("--audio-checkpoint", default="", help="Optional audio-level CTE checkpoint")
    parser.add_argument("--rationale-bank", default="", help="Optional rationale bank JSON")
    parser.add_argument("--diarization-json", default="", help="Optional speaker sample manifest from export_speaker_samples")
    parser.add_argument("--output-json", default="", help="Output JSON path")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path")
    parser.add_argument("--output-turns-csv", default="", help="Optional full turn transcript CSV path")
    parser.add_argument("--print-json", action="store_true", help="Print the full prediction JSON to stdout")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asr-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asr-model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument("--spk-model", default="cam++")
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--segment-batch-size", type=int, default=16)
    parser.add_argument("--max-single-segment-time", type=int, default=60000)
    parser.add_argument("--hotword", default="")
    parser.add_argument("--merge-gap-ms", type=int, default=800)
    parser.add_argument("--client-speaker", default="", help="Optional speaker id, e.g. spk0")
    parser.add_argument("--therapist-speaker", default="", help="Optional speaker id, e.g. spk1")
    parser.add_argument(
        "--first-speaker-role",
        choices=["client", "therapist"],
        default="client",
        help="Role of the first detected speaker when speaker ids are not specified.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    audio_id = audio_path.stem
    output_json = args.output_json or str(Path("outputs") / "{}_prediction.json".format(audio_id))

    print("[stage] Running FunASR ASR and speaker diarization...", flush=True)
    asr_result = run_funasr(args)
    print("[stage] FunASR finished. Normalizing transcript turns...", flush=True)
    raw_turns = normalize_asr_segments(asr_result)
    if args.diarization_json:
        print("[stage] Aligning ASR text with imported speaker samples...", flush=True)
        diarization_turns = turns_from_diarization_json(args.diarization_json)
        if len(raw_turns) == 1:
            merged_turns = split_full_asr_turn_by_diarization(raw_turns[0], diarization_turns)
            if not merged_turns:
                merged_turns = attach_asr_text_to_turns(diarization_turns, raw_turns)
            print(
                "[stage] Split full ASR transcript into {} diarized turns by timestamp.".format(
                    len(merged_turns)
                ),
                flush=True,
            )
        else:
            merged_turns = attach_asr_text_to_turns(diarization_turns, raw_turns)
            print(
                "[stage] Attached {} ASR turns to imported diarization timeline.".format(
                    len(raw_turns)
                ),
                flush=True,
            )
        merged_turns = merge_adjacent_turns(merged_turns, args.merge_gap_ms)
    else:
        print("[stage] Merging adjacent ASR turns...", flush=True)
        merged_turns = merge_adjacent_turns(raw_turns, args.merge_gap_ms)
    print("[stage] Inferring client/therapist speaker roles...", flush=True)
    role_map = infer_role_map(
        merged_turns,
        normalize_speaker(args.client_speaker) if args.client_speaker else "",
        normalize_speaker(args.therapist_speaker) if args.therapist_speaker else "",
        args.first_speaker_role,
    )
    diarized_turns = assign_roles(merged_turns, role_map)
    units = build_empathy_units(diarized_turns, audio_id)
    print(
        "[stage] Built {} turns and {} empathy units.".format(len(diarized_turns), len(units)),
        flush=True,
    )
    if not units:
        raise SystemExit(
            "No empathy units were built. Check speaker roles or rerun with --client-speaker/--therapist-speaker."
        )

    device = torch.device(args.device)
    print("[stage] Loading segment CTE model...", flush=True)
    segment_model, segment_tokenizer, max_length = load_segment_predictor(
        args.segment_checkpoint, device
    )
    bank_path = Path(args.rationale_bank) if args.rationale_bank else rationale_bank_path(args.segment_checkpoint)
    rationale_bank = load_rationale_bank(bank_path)
    print("[stage] Running segment predictions...", flush=True)
    predictions = predict_segments_batch(
        segment_model,
        segment_tokenizer,
        units,
        max_length,
        args.threshold,
        device,
        rationale_bank=rationale_bank,
        batch_size=args.segment_batch_size,
    )
    for unit, prediction in zip(units, predictions):
        unit["prediction"] = prediction

    record = {
        "audio_id": audio_id,
        "audio_file": str(audio_path.resolve()),
        "role_map": role_map,
        "raw_asr_result": asr_result,
        "turns": diarized_turns,
        "segments": units,
        "analysis_summary": {
            "turn_count": len(diarized_turns),
            "unit_count": len(units),
            "client_turn_count": sum(1 for turn in diarized_turns if turn.get("role") == "client"),
            "therapist_turn_count": sum(1 for turn in diarized_turns if turn.get("role") == "therapist"),
            "other_turn_count": sum(1 for turn in diarized_turns if turn.get("role") == "other"),
        },
    }
    if args.audio_checkpoint:
        print("[stage] Running audio-level CTE prediction...", flush=True)
        record["audio_prediction"] = predict_audio_cte(args.audio_checkpoint, record, device)

    print("[stage] Building summaries and writing output files...", flush=True)
    enrich_record_with_summary(record)

    dump_json(output_json, record)
    if args.output_csv:
        export_units_csv(args.output_csv, record)
    output_turns_csv = args.output_turns_csv
    if not output_turns_csv and args.output_csv:
        csv_path = Path(args.output_csv)
        output_turns_csv = str(csv_path.with_name("{}_turns{}".format(csv_path.stem, csv_path.suffix)))
    if output_turns_csv:
        export_turns_csv(output_turns_csv, record)

    print("[stage] Prediction completed. Results were saved to:", flush=True)
    print("[output] json={}".format(output_json), flush=True)
    if args.output_csv:
        print("[output] csv={}".format(args.output_csv), flush=True)
    if output_turns_csv:
        print("[output] turns_csv={}".format(output_turns_csv), flush=True)
    if args.print_json:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
