"""Helpers for rationale-aware checkpoints and retrieval."""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import dump_json, load_json


def rationale_bank_path(checkpoint_path):
    path = Path(checkpoint_path)
    return path.with_suffix(".rationale_bank.json")


def save_rationale_bank(path, records):
    dump_json(path, {"items": records})


def load_rationale_bank(path):
    path = Path(path)
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    return []


def topk_rationales(query_repr, bank_items, top_k=3):
    if not bank_items:
        return []

    query = torch.tensor(query_repr, dtype=torch.float32)
    query = torch.nn.functional.normalize(query, dim=0)

    embeddings = []
    valid_items = []
    for item in bank_items:
        embedding = item.get("embedding")
        rationale = item.get("rationale")
        if not embedding or not rationale:
            continue
        embeddings.append(embedding)
        valid_items.append(item)

    if not embeddings:
        return []

    bank_tensor = torch.tensor(embeddings, dtype=torch.float32)
    bank_tensor = torch.nn.functional.normalize(bank_tensor, dim=-1)
    sims = bank_tensor @ query
    k = min(top_k, sims.numel())
    values, indices = torch.topk(sims, k=k)
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        item = valid_items[idx]
        results.append(
            {
                "similarity": round(float(score), 4),
                "audio_id": item.get("audio_id", ""),
                "segment_id": item.get("segment_id", ""),
                "cte_score": item.get("cte_score", None),
                "rationale": item.get("rationale", ""),
                "client_text": item.get("client_text", ""),
                "therapist_text": item.get("therapist_text", ""),
            }
        )
    return results
