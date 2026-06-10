"""Train the segment-level multi-task model."""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import SegmentDataset, segment_collate, split_indices_by_group
from .modeling import SegmentCTEModel
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
    cte_se = 0.0
    cte_count = 0
    blocking_ok = 0
    blocking_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            rationale_enc = tokenize_rationales(tokenizer, batch, max_length)
            rationale_enc = to_device(rationale_enc, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                cte_scores=batch["cte_scores"],
                cte_mask=batch["has_cte"],
                empathy_labels=batch["empathy_labels"],
                blocking_labels=batch["blocking_labels"],
                rationale_input_ids=rationale_enc.get("input_ids"),
                rationale_attention_mask=rationale_enc.get("attention_mask"),
                rationale_token_type_ids=rationale_enc.get("token_type_ids"),
                rationale_mask=batch.get("has_rationale"),
                rationale_loss_weight=rationale_loss_weight,
            )
            loss = outputs["loss"]
            total_loss += float(loss.item())
            total_rationale_loss += float(outputs.get("rationale_loss", 0.0))
            total += 1
            pred = outputs["cte_pred"].detach().cpu()
            target = batch["cte_scores"].detach().cpu()
            mask = batch["has_cte"].detach().cpu()
            cte_se += float(torch.sum(((pred - target) ** 2) * mask).item())
            valid = int(mask.sum().item())
            cte_count += valid
            if valid:
                blocking_pred = torch.argmax(outputs["blocking_logits"], dim=-1).detach().cpu()
                blocking_true = batch["blocking_labels"].detach().cpu()
                blocking_ok += int((blocking_pred == blocking_true).sum().item())
                blocking_total += int(blocking_true.numel())

    rmse = math.sqrt(cte_se / max(cte_count, 1))
    acc = blocking_ok / max(blocking_total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "rmse": rmse,
        "blocking_acc": acc,
        "rationale_loss": total_rationale_loss / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the segment-level CTE model.")
    parser.add_argument("--data", required=True, help="Path to JSON/JSONL training data")
    parser.add_argument("--val-data", default="", help="Optional validation data path")
    parser.add_argument("--encoder", default="bert-base-chinese", help="HF encoder name")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--split-group-key", default="audio_id", help="Group key for train/val split")
    parser.add_argument(
        "--rationale-loss-weight",
        type=float,
        default=1.2,
        help="Weight for rationale contrastive supervision",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    seed_everything(args.seed)
    dataset = SegmentDataset(args.data)
    if len(dataset) == 0:
        raise SystemExit("Training data is empty.")

    if args.val_data:
        train_set = dataset
        val_set = SegmentDataset(args.val_data)
    else:
        train_idx, val_idx = split_indices_by_group(
            dataset.records,
            group_key=args.split_group_key,
            val_ratio=0.2,
            seed=args.seed,
        )
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

    collate = segment_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SegmentCTEModel(args.encoder)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
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
                cte_scores=batch["cte_scores"],
                cte_mask=batch["has_cte"],
                empathy_labels=batch["empathy_labels"],
                blocking_labels=batch["blocking_labels"],
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
        mean_train = running / max(len(train_loader), 1)
        print(
            "epoch={} train_loss={:.4f} val_loss={:.4f} val_rmse={:.4f} val_blocking_acc={:.4f} val_rationale_loss={:.4f}".format(
                epoch,
                mean_train,
                stats["loss"],
                stats["rmse"],
                stats["blocking_acc"],
                stats["rationale_loss"],
            )
        )
        if best_val is None or stats["loss"] < best_val:
            best_val = stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder": args.encoder,
                    "max_length": args.max_length,
                },
                args.output,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    bank_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    bank = []
    model.eval()
    with torch.no_grad():
        for batch in bank_loader:
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            reprs = outputs["input_repr"].detach().cpu()
            cte_scores = batch["cte_scores"].detach().cpu().tolist()
            for idx, rationale in enumerate(batch.get("rationale_texts", [])):
                if not rationale:
                    continue
                bank.append(
                    {
                        "audio_id": batch["audio_ids"][idx],
                        "segment_id": batch["segment_ids"][idx],
                        "client_text": batch["client_texts"][idx],
                        "therapist_text": batch["therapist_texts"][idx],
                        "cte_score": cte_scores[idx],
                        "rationale": rationale,
                        "embedding": [round(float(x), 6) for x in reprs[idx].tolist()],
                    }
                )

    save_rationale_bank(rationale_bank_path(args.output), bank)


if __name__ == "__main__":
    main()
