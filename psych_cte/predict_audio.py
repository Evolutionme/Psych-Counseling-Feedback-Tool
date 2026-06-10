"""Run inference for the audio-level model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import make_audio_windows
from .modeling import AudioCTEModel, LocalWeightedAudioCTEModel, resolve_encoder_path
from .rationale_utils import load_rationale_bank, rationale_bank_path, topk_rationales


def load_inputs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "audios" in data:
        return data["audios"]
    return [data]


def build_batch(tokenizer, record, max_length, max_segments):
    segments = record.get("segments", [])
    if not segments:
        raise ValueError("Audio record has no segments.")
    if len(segments) > max_segments:
        segments = segments[:max_segments]
    client_texts = [seg.get("client_text", "") for seg in segments]
    therapist_texts = [seg.get("therapist_text", "") for seg in segments]
    enc = tokenizer(
        client_texts,
        therapist_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].unsqueeze(0),
        "attention_mask": enc["attention_mask"].unsqueeze(0),
        "token_type_ids": enc["token_type_ids"].unsqueeze(0) if "token_type_ids" in enc else None,
        "segment_mask": torch.ones((1, len(segments)), dtype=torch.bool),
    }


def predict_audio_with_windows(
    model,
    tokenizer,
    record,
    max_length,
    max_segments,
    window_stride,
    device,
    rationale_bank=None,
):
    windows = make_audio_windows(record, max_segments, window_stride)
    if not windows:
        raise ValueError("Audio record has no segments.")

    total_segments = len(record.get("segments", []))
    total_window_weight = float(sum(len(window.get("segments", [])) for window in windows))
    audio_pred_sum = 0.0
    audio_weight_sum = 0.0
    repr_sum = None
    segment_weight_sum = [0.0] * total_segments
    local_score_sum = [0.0] * total_segments
    local_score_count = [0] * total_segments
    contribution_sum = [0.0] * total_segments
    window_predictions = []

    for window in windows:
        batch = build_batch(tokenizer, window, max_length, max_segments)
        batch = {k: v.to(device) for k, v in batch.items() if v is not None}
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                segment_mask=batch["segment_mask"],
            )
        score = float(out["audio_pred"].item())
        weights = out["segment_weights"].squeeze(0).detach().cpu().tolist()
        local_scores = out.get("local_cte_pred")
        local_scores = local_scores.squeeze(0).detach().cpu().tolist() if local_scores is not None else []
        audio_repr = out["audio_repr"].squeeze(0).detach().cpu().tolist()
        weight = float(len(window["segments"]))
        global_window_weight = weight / max(total_window_weight, 1.0)
        audio_pred_sum += score * weight
        audio_weight_sum += weight
        if repr_sum is None:
            repr_sum = [0.0] * len(audio_repr)
        for idx, value in enumerate(audio_repr):
            repr_sum[idx] += float(value) * weight
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
                "window_end": int(window.get("window_end", start + len(window["segments"]))),
                "segment_count": len(window["segments"]),
                "audio_cte_score": round(score, 4),
            }
        )

    audio_cte_score = audio_pred_sum / max(audio_weight_sum, 1.0)
    audio_repr = [value / max(audio_weight_sum, 1.0) for value in (repr_sum or [])]
    weight_total = sum(segment_weight_sum)
    if weight_total > 0:
        segment_weight_sum = [value / weight_total for value in segment_weight_sum]
    segment_weights = [round(value, 4) for value in segment_weight_sum]
    local_cte_scores = [
        round(total / count, 4) if count else None
        for total, count in zip(local_score_sum, local_score_count)
    ]
    weighted_contributions = [round(value, 4) for value in contribution_sum]

    evidence = []
    if rationale_bank and audio_repr:
        evidence = topk_rationales(audio_repr, rationale_bank, top_k=3)

    return {
        "audio_cte_score": round(float(audio_cte_score), 4),
        "segment_weights": segment_weights,
        "local_cte_scores": local_cte_scores,
        "weighted_contributions": weighted_contributions,
        "window_predictions": window_predictions,
        "rationale_evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict audio-level CTE outputs.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--input-json", required=True, help="Input audio JSON")
    parser.add_argument("--rationale-bank", default="", help="Optional rationale bank JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_encoder_path(ckpt["encoder"]), use_fast=True, local_files_only=True
    )
    if ckpt.get("model_type") == "local_weighted_audio":
        model = LocalWeightedAudioCTEModel(ckpt["encoder"], max_segments=ckpt.get("max_segments", 128))
    else:
        model = AudioCTEModel(ckpt["encoder"], max_segments=ckpt.get("max_segments", 128))
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    bank_path = Path(args.rationale_bank) if args.rationale_bank else rationale_bank_path(args.checkpoint)
    rationale_bank = load_rationale_bank(bank_path)
    window_stride = ckpt.get("window_stride", 0)
    if not window_stride:
        window_stride = max(1, ckpt.get("max_segments", 128) // 2)

    outputs = []
    for record in load_inputs(args.input_json):
        result = predict_audio_with_windows(
            model,
            tokenizer,
            record,
            ckpt.get("max_length", 256),
            ckpt.get("max_segments", 128),
            window_stride,
            device,
            rationale_bank=rationale_bank,
        )
        outputs.append(
            {
                "audio_id": record.get("audio_id", ""),
                "audio_cte_score": result["audio_cte_score"],
                "segment_weights": result["segment_weights"],
                "local_cte_scores": result["local_cte_scores"],
                "weighted_contributions": result["weighted_contributions"],
                "window_predictions": result["window_predictions"],
                "rationale_evidence": result["rationale_evidence"],
            }
        )

    print(json.dumps(outputs if len(outputs) > 1 else outputs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
