"""Perplexity and top-k next-token accuracy."""

from __future__ import annotations

import math

import numpy as np
import torch

IGNORE_INDEX = -100


def make_logits_preprocessor(max_k: int):
    """Shrink logits to top-k indices *before* Trainer accumulates them.

    Without this, evaluation holds a (num_samples, seq_len, 50257) float tensor
    in memory, which is tens of gigabytes on WikiText-2 and will OOM long before
    the metrics are ever computed.
    """

    def preprocess(logits, labels):  # noqa: ARG001 - signature fixed by Trainer
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.topk(max_k, dim=-1).indices

    return preprocess


def make_compute_metrics(topk_values: list[int]):
    ks = sorted(set(topk_values))
    max_k = max(ks)

    def compute_metrics(eval_pred) -> dict[str, float]:
        topk_indices, labels = eval_pred.predictions, eval_pred.label_ids
        topk_indices = np.asarray(topk_indices)
        labels = np.asarray(labels)

        # Position i predicts token i+1, so drop the last prediction and the
        # first label to line them up.
        preds = topk_indices[:, :-1, :max_k]
        targets = labels[:, 1:]

        valid = targets != IGNORE_INDEX
        total = int(valid.sum())
        if total == 0:
            return {}

        matches = preds == targets[..., None]
        results: dict[str, float] = {}
        for k in ks:
            hits = matches[..., :k].any(axis=-1) & valid
            results[f"top{k}_accuracy"] = float(hits.sum()) / total
        return results

    return compute_metrics


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate_language_model(
    model,
    dataloader,
    device: torch.device,
    topk_values: list[int] | None = None,
) -> dict[str, float]:
    """Standalone eval loop: token-weighted loss, perplexity, and top-k accuracy.

    Used by evaluate.py so a saved checkpoint can be scored without spinning up
    a Trainer. Loss is weighted by token count rather than averaged per batch,
    which keeps the number exact when the final batch is short.
    """
    ks = sorted(set(topk_values or [1, 5, 10]))
    max_k = max(ks)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hits = {k: 0 for k in ks}

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        labels = batch["labels"]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        valid = shift_labels != IGNORE_INDEX
        n_tokens = int(valid.sum())
        total_loss += loss.item()
        total_tokens += n_tokens

        topk = shift_logits.topk(max_k, dim=-1).indices
        matches = topk == shift_labels.unsqueeze(-1)
        for k in ks:
            hits[k] += int((matches[..., :k].any(dim=-1) & valid).sum())

    mean_loss = total_loss / max(total_tokens, 1)
    results = {
        "loss": mean_loss,
        "perplexity": perplexity_from_loss(mean_loss),
        "eval_tokens": total_tokens,
    }
    for k in ks:
        results[f"top{k}_accuracy"] = hits[k] / max(total_tokens, 1)
    return results
