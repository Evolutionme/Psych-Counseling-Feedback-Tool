"""PyTorch models for segment and audio CTE prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from .constants import BLOCKING_TYPES, EMPATHY_LABELS


def resolve_encoder_path(encoder_name):
    path = Path(str(encoder_name))
    if path.exists():
        return str(path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + str(encoder_name).replace("/", "--"))
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = [item for item in snapshots_dir.iterdir() if item.is_dir()]
        if snapshots:
            snapshots.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            return str(snapshots[0])

    return str(encoder_name)


def load_encoder_model(encoder_name):
    encoder_path = resolve_encoder_path(encoder_name)
    return AutoModel.from_pretrained(encoder_path, local_files_only=True)


def pooled_output(model_output):
    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output
    return model_output.last_hidden_state[:, 0]


def contrastive_alignment_loss(left_repr, right_repr, mask=None, temperature=0.07):
    if mask is not None:
        mask = mask.bool()
        left_repr = left_repr[mask]
        right_repr = right_repr[mask]
    if left_repr.size(0) == 0:
        return left_repr.new_tensor(0.0)

    left_repr = nn.functional.normalize(left_repr, dim=-1)
    right_repr = nn.functional.normalize(right_repr, dim=-1)
    logits = left_repr @ right_repr.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_lr = nn.functional.cross_entropy(logits, targets)
    loss_rl = nn.functional.cross_entropy(logits.transpose(0, 1), targets)
    return (loss_lr + loss_rl) / 2.0


class SegmentCTEModel(nn.Module):
    def __init__(self, encoder_name="bert-base-chinese", dropout=0.1, rationale_temperature=0.07):
        super(SegmentCTEModel, self).__init__()
        self.encoder = load_encoder_model(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cte_head = nn.Linear(hidden, 1)
        self.empathy_head = nn.Linear(hidden, len(EMPATHY_LABELS))
        self.blocking_head = nn.Linear(hidden, len(BLOCKING_TYPES))
        self.rationale_temperature = rationale_temperature

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        cte_scores=None,
        cte_mask=None,
        empathy_labels=None,
        blocking_labels=None,
        rationale_input_ids=None,
        rationale_attention_mask=None,
        rationale_token_type_ids=None,
        rationale_mask=None,
        rationale_loss_weight=1.0,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        rep = self.dropout(pooled_output(out))
        cte_pred = self.cte_head(rep).squeeze(-1)
        empathy_logits = self.empathy_head(rep)
        blocking_logits = self.blocking_head(rep)

        loss = None
        rationale_loss = rep.new_tensor(0.0)
        if (
            cte_scores is not None
            or empathy_labels is not None
            or blocking_labels is not None
            or rationale_input_ids is not None
        ):
            loss = rep.new_tensor(0.0)
            if cte_scores is not None:
                mse = nn.functional.mse_loss(cte_pred, cte_scores, reduction="none")
                if cte_mask is not None:
                    denom = torch.clamp(cte_mask.sum(), min=1.0)
                    loss = loss + torch.sum(mse * cte_mask) / denom
                else:
                    loss = loss + mse.mean()
            if empathy_labels is not None:
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    empathy_logits, empathy_labels
                )
            if blocking_labels is not None:
                loss = loss + nn.functional.cross_entropy(blocking_logits, blocking_labels)
            if rationale_input_ids is not None:
                rationale_out = self.encoder(
                    input_ids=rationale_input_ids,
                    attention_mask=rationale_attention_mask,
                    token_type_ids=rationale_token_type_ids,
                )
                rationale_rep = self.dropout(pooled_output(rationale_out))
                rationale_loss = contrastive_alignment_loss(
                    rep, rationale_rep, mask=rationale_mask, temperature=self.rationale_temperature
                )
                loss = loss + float(rationale_loss_weight) * rationale_loss

        return {
            "loss": loss,
            "cte_pred": cte_pred,
            "empathy_logits": empathy_logits,
            "blocking_logits": blocking_logits,
            "input_repr": rep,
            "rationale_loss": rationale_loss,
        }


class AudioCTEModel(nn.Module):
    def __init__(self, encoder_name="bert-base-chinese", max_segments=128, dropout=0.1, rationale_temperature=0.07):
        super(AudioCTEModel, self).__init__()
        self.encoder = load_encoder_model(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.segment_proj = nn.Linear(hidden, hidden)
        self.segment_score = nn.Linear(hidden, 1)
        self.positional = nn.Embedding(max_segments, hidden)
        self.dropout = nn.Dropout(dropout)
        self.audio_head = nn.Linear(hidden, 1)
        self.rationale_temperature = rationale_temperature

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        segment_mask=None,
        audio_cte_scores=None,
        audio_cte_mask=None,
        rationale_input_ids=None,
        rationale_attention_mask=None,
        rationale_token_type_ids=None,
        rationale_mask=None,
        rationale_loss_weight=1.0,
    ):
        batch_size, max_segments, seq_len = input_ids.shape
        max_pos = self.positional.num_embeddings
        if max_segments > max_pos:
            input_ids = input_ids[:, :max_pos, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_pos, :]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :max_pos, :]
            if segment_mask is not None:
                segment_mask = segment_mask[:, :max_pos]
            max_segments = max_pos
        flat_ids = input_ids.view(batch_size * max_segments, seq_len)
        flat_mask = None
        flat_type_ids = None
        if attention_mask is not None:
            flat_mask = attention_mask.view(batch_size * max_segments, seq_len)
        if token_type_ids is not None:
            flat_type_ids = token_type_ids.view(batch_size * max_segments, seq_len)

        out = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            token_type_ids=flat_type_ids,
        )
        seg_rep = pooled_output(out).view(batch_size, max_segments, -1)
        seg_count = seg_rep.size(1)
        pos_ids = torch.arange(seg_count, device=seg_rep.device).unsqueeze(0).expand(batch_size, seg_count)
        seg_rep = seg_rep + self.positional(pos_ids)
        seg_rep = torch.tanh(self.segment_proj(self.dropout(seg_rep)))

        attn_logits = self.segment_score(seg_rep).squeeze(-1)
        if segment_mask is not None:
            attn_logits = attn_logits.masked_fill(~segment_mask, -1e9)
        weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * seg_rep, dim=1)
        audio_pred = self.audio_head(self.dropout(context)).squeeze(-1)

        loss = None
        rationale_loss = context.new_tensor(0.0)
        if audio_cte_scores is not None or rationale_input_ids is not None:
            loss = context.new_tensor(0.0)
        if audio_cte_scores is not None:
            mse = nn.functional.mse_loss(audio_pred, audio_cte_scores, reduction="none")
            if audio_cte_mask is not None:
                denom = torch.clamp(audio_cte_mask.sum(), min=1.0)
                loss = torch.sum(mse * audio_cte_mask) / denom
            else:
                loss = mse.mean()
        if rationale_input_ids is not None:
            rationale_out = self.encoder(
                input_ids=rationale_input_ids,
                attention_mask=rationale_attention_mask,
                token_type_ids=rationale_token_type_ids,
            )
            rationale_rep = self.dropout(pooled_output(rationale_out))
            rationale_loss = contrastive_alignment_loss(
                context, rationale_rep, mask=rationale_mask, temperature=self.rationale_temperature
            )
            loss = loss + float(rationale_loss_weight) * rationale_loss

        return {
            "loss": loss,
            "audio_pred": audio_pred,
            "segment_weights": weights,
            "audio_repr": context,
            "rationale_loss": rationale_loss,
        }


class LocalWeightedAudioCTEModel(nn.Module):
    def __init__(
        self,
        encoder_name="bert-base-chinese",
        max_segments=128,
        dropout=0.1,
        weight_hidden=128,
        rationale_temperature=0.07,
    ):
        super(LocalWeightedAudioCTEModel, self).__init__()
        self.segment_model = SegmentCTEModel(
            encoder_name=encoder_name,
            dropout=dropout,
            rationale_temperature=rationale_temperature,
        )
        hidden = self.segment_model.encoder.config.hidden_size
        feature_dim = hidden + 1 + len(EMPATHY_LABELS) + len(BLOCKING_TYPES) + 1
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, weight_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(weight_hidden, 1),
        )
        self.max_segments = int(max_segments)
        self.rationale_temperature = rationale_temperature
        self.segment_trainable = True

    def set_segment_trainable(self, trainable):
        self.segment_trainable = bool(trainable)
        for param in self.segment_model.parameters():
            param.requires_grad = bool(trainable)
        if not trainable:
            self.segment_model.eval()

    def train(self, mode=True):
        super(LocalWeightedAudioCTEModel, self).train(mode)
        if not self.segment_trainable:
            self.segment_model.eval()
        return self

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        segment_mask=None,
        audio_cte_scores=None,
        audio_cte_mask=None,
        rationale_input_ids=None,
        rationale_attention_mask=None,
        rationale_token_type_ids=None,
        rationale_mask=None,
        rationale_loss_weight=1.0,
    ):
        batch_size, max_segments, seq_len = input_ids.shape
        if max_segments > self.max_segments:
            input_ids = input_ids[:, : self.max_segments, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_segments, :]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, : self.max_segments, :]
            if segment_mask is not None:
                segment_mask = segment_mask[:, : self.max_segments]
            max_segments = self.max_segments

        flat_ids = input_ids.reshape(batch_size * max_segments, seq_len)
        flat_mask = attention_mask.reshape(batch_size * max_segments, seq_len) if attention_mask is not None else None
        flat_type_ids = token_type_ids.reshape(batch_size * max_segments, seq_len) if token_type_ids is not None else None

        if self.segment_trainable:
            seg_out = self.segment_model(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                token_type_ids=flat_type_ids,
            )
        else:
            with torch.no_grad():
                seg_out = self.segment_model(
                    input_ids=flat_ids,
                    attention_mask=flat_mask,
                    token_type_ids=flat_type_ids,
                )
        seg_repr = seg_out["input_repr"].view(batch_size, max_segments, -1)
        local_cte = seg_out["cte_pred"].view(batch_size, max_segments)
        empathy_probs = torch.sigmoid(seg_out["empathy_logits"]).view(
            batch_size, max_segments, len(EMPATHY_LABELS)
        )
        blocking_probs = torch.softmax(seg_out["blocking_logits"], dim=-1).view(
            batch_size, max_segments, len(BLOCKING_TYPES)
        )

        if segment_mask is None:
            segment_mask = torch.ones(
                (batch_size, max_segments), dtype=torch.bool, device=input_ids.device
            )

        denom = torch.clamp(segment_mask.float().sum(dim=1, keepdim=True) - 1.0, min=1.0)
        positions = torch.arange(max_segments, device=input_ids.device).float().view(1, max_segments, 1)
        positions = positions.expand(batch_size, max_segments, 1) / denom.unsqueeze(-1)

        features = torch.cat(
            [
                seg_repr,
                local_cte.unsqueeze(-1),
                empathy_probs,
                blocking_probs,
                positions,
            ],
            dim=-1,
        )
        attn_logits = self.weight_net(features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~segment_mask.bool(), -1e9)
        weights = torch.softmax(attn_logits, dim=1)
        audio_pred = torch.sum(weights * local_cte, dim=1)
        audio_repr = torch.sum(weights.unsqueeze(-1) * seg_repr, dim=1)

        loss = None
        rationale_loss = audio_repr.new_tensor(0.0)
        if audio_cte_scores is not None or rationale_input_ids is not None:
            loss = audio_repr.new_tensor(0.0)
        if audio_cte_scores is not None:
            mse = nn.functional.mse_loss(audio_pred, audio_cte_scores, reduction="none")
            if audio_cte_mask is not None:
                denom = torch.clamp(audio_cte_mask.sum(), min=1.0)
                loss = torch.sum(mse * audio_cte_mask) / denom
            else:
                loss = mse.mean()
        if rationale_input_ids is not None:
            if self.segment_trainable:
                rationale_out = self.segment_model.encoder(
                    input_ids=rationale_input_ids,
                    attention_mask=rationale_attention_mask,
                    token_type_ids=rationale_token_type_ids,
                )
            else:
                with torch.no_grad():
                    rationale_out = self.segment_model.encoder(
                        input_ids=rationale_input_ids,
                        attention_mask=rationale_attention_mask,
                        token_type_ids=rationale_token_type_ids,
                    )
            rationale_rep = self.segment_model.dropout(pooled_output(rationale_out))
            rationale_loss = contrastive_alignment_loss(
                audio_repr,
                rationale_rep,
                mask=rationale_mask,
                temperature=self.rationale_temperature,
            )
            loss = loss + float(rationale_loss_weight) * rationale_loss

        return {
            "loss": loss,
            "audio_pred": audio_pred,
            "segment_weights": weights,
            "audio_repr": audio_repr,
            "local_cte_pred": local_cte,
            "empathy_probs": empathy_probs,
            "blocking_probs": blocking_probs,
            "rationale_loss": rationale_loss,
        }
