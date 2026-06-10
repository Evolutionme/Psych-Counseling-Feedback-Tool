"""Train the audio-level CTE model."""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import AudioDataset, audio_collate, expand_audio_windows, split_indices_by_group
from .modeling import LocalWeightedAudioCTEModel
from .rationale_utils import rationale_bank_path, save_rationale_bank


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def tokenize_rationales(tokenizer, batch, max_length):
    if tokenizer is None:
        return {}
    texts = batch.get("rationale_texts", [])
    if not texts:
        return {}
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc


def evaluate(model, loader, device, tokenizer, max_length, rationale_loss_weight):
    model.eval()
    total_loss = 0.0
    total_rationale_loss = 0.0
    total = 0
    window_se = 0.0
    window_count = 0
    audio_pred_sum = {}
    audio_weight_sum = {}
    audio_target = {}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            rationale_enc = tokenize_rationales(tokenizer, batch, max_length)
            rationale_enc = to_device(rationale_enc, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                segment_mask=batch["segment_mask"],
                audio_cte_scores=batch["audio_cte_scores"],
                audio_cte_mask=batch["has_audio_cte"],
                rationale_input_ids=rationale_enc.get("input_ids"),
                rationale_attention_mask=rationale_enc.get("attention_mask"),
                rationale_token_type_ids=rationale_enc.get("token_type_ids"),
                rationale_mask=batch.get("has_rationale"),
                rationale_loss_weight=rationale_loss_weight,
            )
            total_loss += float(outputs["loss"].item())
            total_rationale_loss += float(outputs.get("rationale_loss", 0.0))
            total += 1
            pred = outputs["audio_pred"].detach().cpu()
            target = batch["audio_cte_scores"].detach().cpu()
            mask = batch["has_audio_cte"].detach().cpu()
            counts = batch.get("segment_counts")
            counts = counts.detach().cpu().tolist() if torch.is_tensor(counts) else [1] * len(pred)
            window_se += float(torch.sum(((pred - target) ** 2) * mask).item())
            window_count += int(mask.sum().item())
            for idx, audio_id in enumerate(batch.get("audio_ids", [])):
                if not bool(mask[idx].item()):
                    continue
                weight = float(counts[idx] if idx < len(counts) else 1.0)
                audio_pred_sum[audio_id] = audio_pred_sum.get(audio_id, 0.0) + float(pred[idx].item()) * weight
                audio_weight_sum[audio_id] = audio_weight_sum.get(audio_id, 0.0) + weight
                audio_target[audio_id] = float(target[idx].item())

    audio_se = 0.0
    audio_count = 0
    for audio_id, target in audio_target.items():
        weight = audio_weight_sum.get(audio_id, 0.0)
        if weight <= 0:
            continue
        pred = audio_pred_sum.get(audio_id, 0.0) / weight
        audio_se += (pred - target) ** 2
        audio_count += 1

    window_rmse = math.sqrt(window_se / max(window_count, 1))
    audio_rmse = math.sqrt(audio_se / max(audio_count, 1))
    return {
        "loss": total_loss / max(total, 1),
        "window_rmse": window_rmse,
        "audio_rmse": audio_rmse,
        "rationale_loss": total_rationale_loss / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the audio-level CTE model.")
    parser.add_argument("--data", required=True, help="Path to audio-level JSON data")
    parser.add_argument("--val-data", default="", help="Optional validation data path")
    parser.add_argument(
        "--segment-checkpoint",
        required=True,
        help="Pretrained segment_cte.pt checkpoint used for local CTE predictions",
    )
    parser.add_argument("--encoder", default="", help="Deprecated; encoder is read from segment checkpoint")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-segments", type=int, default=128)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=0,
        help="Sliding-window stride in segments; 0 uses half-window overlap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true", help="Deprecated; segment model is frozen by default")
    parser.add_argument(
        "--finetune-segment",
        action="store_true",
        help="Allow audio-level training to update the loaded segment model",
    )
    parser.add_argument("--split-group-key", default="audio_id", help="Group key for train/val split")
    parser.add_argument(
        "--rationale-loss-weight",
        type=float,
        default=1.2,
        help="Weight for rationale contrastive supervision",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    segment_ckpt = torch.load(args.segment_checkpoint, map_location="cpu")
    encoder = segment_ckpt["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    dataset = AudioDataset(args.data)
    if len(dataset) == 0:
        raise SystemExit("Training data is empty.")

    window_stride = args.window_stride if args.window_stride > 0 else None
    effective_window_stride = window_stride if window_stride is not None else max(1, args.max_segments // 2)

    if args.val_data:
        train_records = dataset.records
        val_dataset = AudioDataset(args.val_data)
        val_records = val_dataset.records
    else:
        train_idx, val_idx = split_indices_by_group(
            dataset.records,
            group_key=args.split_group_key,
            val_ratio=0.2,
            seed=args.seed,
        )
        train_records = [dataset.records[idx] for idx in train_idx]
        val_records = [dataset.records[idx] for idx in val_idx]

    train_set = expand_audio_windows(train_records, args.max_segments, window_stride)
    val_set = expand_audio_windows(val_records, args.max_segments, window_stride)
    if not train_set:
        raise SystemExit("Training split produced no sliding windows.")
    if not val_set:
        raise SystemExit("Validation split produced no sliding windows.")

    collate = audio_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = LocalWeightedAudioCTEModel(encoder, max_segments=args.max_segments)
    model.segment_model.load_state_dict(segment_ckpt["model_state"])
    model.set_segment_trainable(args.finetune_segment)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    best_val = None
    best_state = None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            rationale_enc = tokenize_rationales(tokenizer, batch, args.max_length)
            rationale_enc = to_device(rationale_enc, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                segment_mask=batch["segment_mask"],
                audio_cte_scores=batch["audio_cte_scores"],
                audio_cte_mask=batch["has_audio_cte"],
                rationale_input_ids=rationale_enc.get("input_ids"),
                rationale_attention_mask=rationale_enc.get("attention_mask"),
                rationale_token_type_ids=rationale_enc.get("token_type_ids"),
                rationale_mask=batch.get("has_rationale"),
                rationale_loss_weight=args.rationale_loss_weight,
            )
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())

        stats = evaluate(model, val_loader, device, tokenizer, args.max_length, args.rationale_loss_weight)
        print(
            "epoch={} train_loss={:.4f} val_loss={:.4f} val_window_rmse={:.4f} val_audio_rmse={:.4f} val_rationale_loss={:.4f}".format(
                epoch,
                running / max(len(train_loader), 1),
                stats["loss"],
                stats["window_rmse"],
                stats["audio_rmse"],
                stats["rationale_loss"],
            )
        )
        if best_val is None or stats["audio_rmse"] < best_val:
            best_val = stats["audio_rmse"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "local_weighted_audio",
                    "encoder": encoder,
                    "max_length": args.max_length,
                    "max_segments": args.max_segments,
                    "window_stride": effective_window_stride,
                    "segment_checkpoint": str(Path(args.segment_checkpoint)),
                    "finetune_segment": bool(args.finetune_segment),
                },
                args.output,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    bank_records = train_set
    bank_loader = DataLoader(bank_records, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    bank = []
    model.eval()
    with torch.no_grad():
        for batch in bank_loader:
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                segment_mask=batch["segment_mask"],
            )
            reprs = outputs["audio_repr"].detach().cpu()
            local_cte = outputs["local_cte_pred"].detach().cpu()
            weights = outputs["segment_weights"].detach().cpu()
            targets = batch["audio_cte_scores"].detach().cpu().tolist()
            counts = batch.get("segment_counts")
            counts = counts.tolist() if torch.is_tensor(counts) else [1] * len(targets)
            flat_client = batch.get("client_texts", [])
            flat_therapist = batch.get("therapist_texts", [])
            offset = 0
            for idx, rationale in enumerate(batch.get("rationale_texts", [])):
                if not rationale:
                    offset += int(counts[idx]) if idx < len(counts) else 0
                    continue
                count = int(counts[idx]) if idx < len(counts) else 0
                client_text = "".join(flat_client[offset : offset + count]).strip()
                therapist_text = "".join(flat_therapist[offset : offset + count]).strip()
                bank.append(
                    {
                        "audio_id": batch["audio_ids"][idx],
                        "segment_id": batch.get("window_ids", [""])[idx] if batch.get("window_ids") else "",
                        "client_text": client_text,
                        "therapist_text": therapist_text,
                        "cte_score": targets[idx],
                        "rationale": rationale,
                        "local_cte_scores": [
                            round(float(x), 4) for x in local_cte[idx, :count].tolist()
                        ],
                        "segment_weights": [
                            round(float(x), 4) for x in weights[idx, :count].tolist()
                        ],
                        "embedding": [round(float(x), 6) for x in reprs[idx].tolist()],
                    }
                )
                offset += count

    save_rationale_bank(rationale_bank_path(args.output), bank)


if __name__ == "__main__":
    main()
