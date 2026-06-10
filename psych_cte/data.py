"""Dataset utilities for segment-level and audio-level tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .constants import BLOCKING_TYPES, EMPATHY_LABELS, FIELD_ALIASES
from .utils import first_value, load_json


EMPATHY_KEY_ALIASES = {
    "emotion_reflection": ["emotion_reflection", "情感内容反应", "内容情感反映", "内容反映", "情感反映"],
    "deep_meaning_understanding": ["deep_meaning_understanding", "深层意义理解"],
    "acceptance_confirmation": ["acceptance_confirmation", "接纳确认"],
    "exploration_facilitation": ["exploration_facilitation", "促进探索"],
    "blocking_present": ["blocking_present", "阻碍类型", "阻碍存在", "共情阻碍"],
}

BLOCKING_ALIASES = {
    "none": ["none", "无", "正常", "0"],
    "premature_advice": ["premature_advice", "过早建议"],
    "judgment_blame": ["judgment_blame", "评价责备"],
    "minimization": ["minimization", "淡化感受"],
    "topic_shift": ["topic_shift", "转移话题"],
    "vague_response": ["vague_response", "空泛回应"],
    "other": ["other", "其他"],
}


def raw_labels(record):
    raw = record.get("raw_labels", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def record_value(record, keys, default=None):
    value = first_value(record, keys, None)
    if value not in (None, ""):
        return value
    return first_value(raw_labels(record), keys, default)


def flatten_records(data):
    if isinstance(data, list):
        for item in data:
            for flat in flatten_records(item):
                yield flat
    elif isinstance(data, dict):
        if "segments" in data:
            for seg in data.get("segments", []):
                flat = dict(seg)
                flat["audio_id"] = data.get("audio_id", flat.get("audio_id", ""))
                flat["audio_cte_score"] = first_value(
                    data, FIELD_ALIASES["audio_cte_score"], None
                )
                flat["audio_cte_rationale"] = first_value(
                    data, FIELD_ALIASES["audio_cte_rationale"], ""
                )
                yield flat
        else:
            yield data


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        data = load_json(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return list(flatten_records(data))


def sliding_window_spans(total_items, window_size, stride=None):
    total_items = int(total_items or 0)
    window_size = max(1, int(window_size or 1))
    if stride is None or stride <= 0:
        stride = max(1, window_size // 2)
    else:
        stride = max(1, int(stride))

    if total_items <= 0:
        return []
    if total_items <= window_size:
        return [(0, total_items)]

    spans = []
    start = 0
    last_start = max(0, total_items - window_size)
    while start < total_items:
        end = min(start + window_size, total_items)
        spans.append((start, end))
        if end >= total_items:
            break
        start += stride

    if spans[-1][1] < total_items:
        spans.append((last_start, total_items))

    deduped = []
    for span in spans:
        if not deduped or deduped[-1] != span:
            deduped.append(span)
    return deduped


def make_audio_windows(record, window_size, stride=None):
    segments = record.get("segments", []) or []
    spans = sliding_window_spans(len(segments), window_size, stride)
    if not spans:
        return []

    audio_id = first_value(record, ["audio_id"], "")
    audio_cte_score = first_value(record, ["audio_cte_score"], None)
    audio_cte_rationale = first_value(record, ["audio_cte_rationale"], "")
    windows = []
    for idx, (start, end) in enumerate(spans, 1):
        window = dict(record)
        window["audio_id"] = audio_id
        window["segments"] = [dict(seg) for seg in segments[start:end]]
        window["audio_cte_score"] = audio_cte_score
        window["audio_cte_rationale"] = audio_cte_rationale
        window["window_id"] = "{}_window_{:03d}".format(audio_id or "audio", idx)
        window["window_index"] = idx
        window["window_count"] = len(spans)
        window["window_start"] = start
        window["window_end"] = end
        window["window_size"] = window_size
        window["window_stride"] = stride if stride is not None else max(1, int(window_size) // 2)
        windows.append(window)
    return windows


def expand_audio_windows(records, window_size, stride=None):
    windows = []
    for record in records:
        windows.extend(make_audio_windows(record, window_size, stride))
    return windows


def split_indices_by_group(records, group_key="audio_id", val_ratio=0.2, seed=42):
    if not records:
        return [], []

    group_to_indices = {}
    for idx, record in enumerate(records):
        group_value = first_value(record, [group_key], "")
        if group_value in (None, ""):
            group_value = "__row_{}".format(idx)
        group_value = str(group_value)
        group_to_indices.setdefault(group_value, []).append(idx)

    groups = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val_groups = max(1, int(len(groups) * val_ratio))
    n_val_groups = min(n_val_groups, len(groups))
    val_groups = set(groups[:n_val_groups])

    train_idx = []
    val_idx = []
    for group, indices in group_to_indices.items():
        if group in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if not train_idx:
        train_idx = list(val_idx)
    return train_idx, val_idx


def label_to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    return value in ("1", "true", "yes", "y", "是", "有", "存在", "positive")


def normalize_blocking_type(value):
    if value is None:
        return "none"
    value = str(value).strip().lower()
    for canonical, aliases in BLOCKING_ALIASES.items():
        if value in [str(item).strip().lower() for item in aliases]:
            return canonical
    return "other"


def infer_blocking_type(record):
    direct = record_value(record, FIELD_ALIASES["blocking_type"], "")
    if direct not in (None, ""):
        return normalize_blocking_type(direct)

    evidence = str(first_value(raw_labels(record), ["共情标签依据", "blocking_evidence"], "")).lower()
    for canonical, aliases in BLOCKING_ALIASES.items():
        if canonical == "none":
            continue
        for alias in aliases:
            if str(alias).strip().lower() in evidence:
                return canonical
    return "none"


def normalize_empathy_labels(raw):
    labels = [0.0] * len(EMPATHY_LABELS)
    if isinstance(raw, list):
        for i, value in enumerate(raw[: len(labels)]):
            labels[i] = 1.0 if label_to_bool(value) else 0.0
        return labels
    if not isinstance(raw, dict):
        return labels
    for i, name in enumerate(EMPATHY_LABELS):
        aliases = EMPATHY_KEY_ALIASES[name]
        found = first_value(raw, aliases, 0)
        labels[i] = 1.0 if label_to_bool(found) else 0.0
    evidence = str(first_value(raw, ["共情标签依据", "empathy_evidence"], "")).lower()
    if "情感反映" in evidence or "情感反应" in evidence or "内容反映" in evidence:
        labels[EMPATHY_LABELS.index("emotion_reflection")] = 1.0
    if "深层意义" in evidence or "深层理解" in evidence:
        labels[EMPATHY_LABELS.index("deep_meaning_understanding")] = 1.0
    if "接纳" in evidence or "确认" in evidence:
        labels[EMPATHY_LABELS.index("acceptance_confirmation")] = 1.0
    if "促进探索" in evidence or "探索" in evidence:
        labels[EMPATHY_LABELS.index("exploration_facilitation")] = 1.0
    return labels


def canonical_text(record, keys, default=""):
    value = first_value(record, keys, default)
    return "" if value is None else str(value)


class SegmentDataset(Dataset):
    def __init__(self, path):
        self.records = []
        for record in load_records(path):
            client_text = canonical_text(record, FIELD_ALIASES["client_text"])
            therapist_text = canonical_text(record, FIELD_ALIASES["therapist_text"])
            if not client_text and not therapist_text:
                continue
            labels = raw_labels(record)
            if not labels:
                labels = record.get("empathy_labels", record.get("labels", {}))
            self.records.append(
                {
                    "audio_id": canonical_text(record, FIELD_ALIASES["audio_id"]),
                    "segment_id": canonical_text(record, FIELD_ALIASES["segment_id"]),
                    "client_text": client_text,
                    "therapist_text": therapist_text,
                    "local_cte_score": record_value(record, FIELD_ALIASES["local_cte_score"], None),
                    "local_cte_rationale": record_value(record, FIELD_ALIASES["local_cte_rationale"], ""),
                    "empathy_labels": normalize_empathy_labels(labels),
                    "blocking_type": infer_blocking_type(record),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def segment_collate(tokenizer, max_length=256):
    def _collate(batch):
        client_texts = [item["client_text"] for item in batch]
        therapist_texts = [item["therapist_text"] for item in batch]
        enc = tokenizer(
            client_texts,
            therapist_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        result = dict(enc)
        result["audio_ids"] = [item["audio_id"] for item in batch]
        result["segment_ids"] = [item["segment_id"] for item in batch]
        result["client_texts"] = client_texts
        result["therapist_texts"] = therapist_texts
        cte_scores = []
        has_cte = []
        empathy = []
        blocking = []
        for item in batch:
            cte = item["local_cte_score"]
            if cte is None or cte == "":
                cte_scores.append(0.0)
                has_cte.append(0)
            else:
                cte_scores.append(float(cte))
                has_cte.append(1)
            empathy.append(item["empathy_labels"])
            blocking.append(BLOCKING_TYPES.index(item["blocking_type"]) if item["blocking_type"] in BLOCKING_TYPES else BLOCKING_TYPES.index("other"))

        result["cte_scores"] = torch.tensor(cte_scores, dtype=torch.float32)
        result["has_cte"] = torch.tensor(has_cte, dtype=torch.float32)
        result["empathy_labels"] = torch.tensor(empathy, dtype=torch.float32)
        result["blocking_labels"] = torch.tensor(blocking, dtype=torch.long)
        result["rationale_texts"] = [item["local_cte_rationale"] for item in batch]
        result["has_rationale"] = torch.tensor(
            [1 if item["local_cte_rationale"] else 0 for item in batch], dtype=torch.bool
        )
        return result

    return _collate


class AudioDataset(Dataset):
    def __init__(self, path):
        self.records = []
        raw = load_json(path)
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict) and "audios" in raw:
            records = raw["audios"]
        elif isinstance(raw, dict) and "segments" in raw:
            records = [raw]
        else:
            records = [raw]

        for record in records:
            if "segments" not in record:
                continue
            segments = []
            for seg in record.get("segments", []):
                client_text = canonical_text(seg, FIELD_ALIASES["client_text"])
                therapist_text = canonical_text(seg, FIELD_ALIASES["therapist_text"])
                if not client_text and not therapist_text:
                    continue
                segments.append(
                    {
                        "segment_id": canonical_text(seg, FIELD_ALIASES["segment_id"]),
                        "client_text": client_text,
                        "therapist_text": therapist_text,
                    }
                )
            if not segments:
                continue
            self.records.append(
                {
                    "audio_id": canonical_text(record, FIELD_ALIASES["audio_id"]),
                    "segments": segments,
                    "audio_cte_score": record_value(record, FIELD_ALIASES["audio_cte_score"], None),
                    "audio_cte_rationale": record_value(record, FIELD_ALIASES["audio_cte_rationale"], ""),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def audio_collate(tokenizer, max_length=256):
    def _collate(batch):
        batch_size = len(batch)
        counts = [len(item["segments"]) for item in batch]
        max_segments = max(counts)
        flat_client = []
        flat_therapist = []
        for item in batch:
            for seg in item["segments"]:
                flat_client.append(seg["client_text"])
                flat_therapist.append(seg["therapist_text"])

        enc = tokenizer(
            flat_client,
            flat_therapist,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        seq_len = enc["input_ids"].shape[1]
        input_ids = enc["input_ids"].new_zeros((batch_size, max_segments, seq_len))
        attention_mask = enc["attention_mask"].new_zeros((batch_size, max_segments, seq_len))
        token_type_ids = None
        if "token_type_ids" in enc:
            token_type_ids = enc["token_type_ids"].new_zeros((batch_size, max_segments, seq_len))
        segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)

        offset = 0
        for i, count in enumerate(counts):
            input_ids[i, :count] = enc["input_ids"][offset : offset + count]
            attention_mask[i, :count] = enc["attention_mask"][offset : offset + count]
            if token_type_ids is not None:
                token_type_ids[i, :count] = enc["token_type_ids"][offset : offset + count]
            segment_mask[i, :count] = True
            offset += count

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_mask": segment_mask,
            "audio_ids": [item["audio_id"] for item in batch],
            "window_ids": [item.get("window_id", "") for item in batch],
            "window_starts": [item.get("window_start", None) for item in batch],
            "window_ends": [item.get("window_end", None) for item in batch],
            "window_counts": [item.get("window_count", None) for item in batch],
            "segment_counts": torch.tensor(counts, dtype=torch.long),
            "client_texts": flat_client,
            "therapist_texts": flat_therapist,
        }
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids

        targets = []
        has_target = []
        for item in batch:
            value = item["audio_cte_score"]
            if value is None or value == "":
                targets.append(0.0)
                has_target.append(0)
            else:
                targets.append(float(value))
                has_target.append(1)
        result["audio_cte_scores"] = torch.tensor(targets, dtype=torch.float32)
        result["has_audio_cte"] = torch.tensor(has_target, dtype=torch.float32)
        result["rationale_texts"] = [item["audio_cte_rationale"] for item in batch]
        result["has_rationale"] = torch.tensor(
            [1 if item["audio_cte_rationale"] else 0 for item in batch], dtype=torch.bool
        )
        return result

    return _collate
