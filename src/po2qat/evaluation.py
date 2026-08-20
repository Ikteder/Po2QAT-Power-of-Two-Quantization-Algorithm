from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .data import sample_language_batch


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def classification_report(model: nn.Module, loader, device: torch.device, class_names: list[str] | None = None) -> dict[str, Any]:
    names = class_names or CIFAR10_CLASSES
    classes = len(names)
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    loss_sum, count, top5_correct = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
            predictions = logits.argmax(dim=1)
            indices = (targets.cpu() * classes + predictions.cpu()).to(torch.int64)
            confusion += torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)
            top5_correct += int((logits.topk(min(5, classes), dim=1).indices == targets[:, None]).any(dim=1).sum())
            count += targets.numel()

    true_positive = confusion.diag().to(torch.float64)
    predicted_count = confusion.sum(dim=0).to(torch.float64)
    actual_count = confusion.sum(dim=1).to(torch.float64)
    precision = torch.where(predicted_count > 0, true_positive / predicted_count, 0.0)
    recall = torch.where(actual_count > 0, true_positive / actual_count, 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    support = actual_count
    weights = support / max(float(support.sum()), 1.0)
    accuracy = float(true_positive.sum() / max(count, 1))
    per_class = [
        {
            "class_index": index,
            "class_name": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    ]
    return {
        "task": "classification",
        "summary": {
            "loss": loss_sum / max(count, 1),
            "accuracy": accuracy,
            "top5_accuracy": top5_correct / max(count, 1),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_precision": float((precision * weights).sum()),
            "weighted_recall": float((recall * weights).sum()),
            "weighted_f1": float((f1 * weights).sum()),
            "balanced_accuracy": float(recall.mean()),
            "examples": count,
        },
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": names,
    }


def language_report(
    model: nn.Module,
    encoded: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    batches: int = 10,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    loss_sum, token_count, top1_correct, top5_correct = 0.0, 0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_language_batch(encoded, block_size, batch_size, generator, device)
            logits = model(inputs)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            loss_sum += float(F.cross_entropy(flat_logits, flat_targets, reduction="sum"))
            top1_correct += int((flat_logits.argmax(dim=1) == flat_targets).sum())
            top5_correct += int(
                (flat_logits.topk(min(5, flat_logits.shape[-1]), dim=1).indices == flat_targets[:, None]).any(dim=1).sum()
            )
            token_count += flat_targets.numel()
    cross_entropy = loss_sum / max(token_count, 1)
    return {
        "task": "character_language_modeling",
        "summary": {
            "cross_entropy_loss": cross_entropy,
            "perplexity": math.exp(min(cross_entropy, 20)),
            "bits_per_character": cross_entropy / math.log(2),
            "next_token_top1_accuracy": top1_correct / max(token_count, 1),
            "next_token_top5_accuracy": top5_correct / max(token_count, 1),
            "evaluated_tokens": token_count,
            "evaluation_batches": batches,
            "context_length": block_size,
        },
    }

