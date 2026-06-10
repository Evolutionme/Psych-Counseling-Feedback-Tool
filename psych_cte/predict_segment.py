"""Run inference for the segment-level model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .constants import BLOCKING_TYPES, EMPATHY_LABELS
from .modeling import SegmentCTEModel, resolve_encoder_path
from .rationale_utils import load_rationale_bank, rationale_bank_path, topk_rationales


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


def first_value(record, keys, default=""):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return default


def load_inputs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        samples = []
        for seg in data["segments"]:
            item = dict(seg)
            item.setdefault("audio_id", data.get("audio_id", ""))
            samples.append(item)
        return samples
    return [data]


def predict_one(model, tokenizer, sample, max_length, threshold, device, rationale_bank=None):
    client_text = first_value(sample, ["client_text", "来访文本", "来访者文本"])
    therapist_text = first_value(sample, ["therapist_text", "咨询师文本"])
    enc = tokenizer(
        [client_text],
        [therapist_text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)

    cte_score = float(out["cte_pred"].item())
    empathy_probs = torch.sigmoid(out["empathy_logits"]).squeeze(0).cpu().tolist()
    blocking_probs = torch.softmax(out["blocking_logits"], dim=-1).squeeze(0).cpu().tolist()

    empathy = {}
    for idx, name in enumerate(EMPATHY_LABELS):
        empathy[EMPATHY_CN[name]] = {
            "prob": round(float(empathy_probs[idx]), 4),
            "present": bool(empathy_probs[idx] >= threshold),
        }

    blocking_idx = int(torch.argmax(out["blocking_logits"], dim=-1).item())
    blocking_type = BLOCKING_TYPES[blocking_idx]
    evidence = []
    if rationale_bank:
        evidence = topk_rationales(out["input_repr"].squeeze(0).detach().cpu().tolist(), rationale_bank, top_k=3)
    return {
        "audio_id": sample.get("audio_id", ""),
        "segment_id": sample.get("segment_id", sample.get("annotation_id", "")),
        "cte_score": round(cte_score, 4),
        "empathy": empathy,
        "blocking_type": BLOCKING_CN[blocking_type],
        "blocking_probabilities": {
            BLOCKING_CN[name]: round(float(prob), 4)
            for name, prob in zip(BLOCKING_TYPES, blocking_probs)
        },
        "rationale_evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict segment-level CTE outputs.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--input-json", default="", help="Input JSON file")
    parser.add_argument("--client-text", default="", help="Client utterance text")
    parser.add_argument("--therapist-text", default="", help="Therapist utterance text")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rationale-bank", default="", help="Optional rationale bank JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_encoder_path(ckpt["encoder"]), use_fast=True, local_files_only=True
    )
    model = SegmentCTEModel(ckpt["encoder"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    bank_path = Path(args.rationale_bank) if args.rationale_bank else rationale_bank_path(args.checkpoint)
    rationale_bank = load_rationale_bank(bank_path)

    if args.input_json:
        samples = load_inputs(args.input_json)
    else:
        samples = [{"client_text": args.client_text, "therapist_text": args.therapist_text}]

    outputs = [
        predict_one(
            model,
            tokenizer,
            sample,
            ckpt.get("max_length", 256),
            args.threshold,
            device,
            rationale_bank=rationale_bank,
        )
        for sample in samples
    ]
    print(json.dumps(outputs if len(outputs) > 1 else outputs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
