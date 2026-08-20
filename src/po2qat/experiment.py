from __future__ import annotations

import copy
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .artifacts import export_run
from .data import CharacterTokenizer, load_text, sample_language_batch, vision_loaders
from .evaluation import classification_report, language_report
from .models import build_model, count_parameters
from .quantization import materialize_model, prepare_po2_qat


@dataclass
class ExperimentConfig:
    model: str
    profile: str = "quick"
    output_dir: Path = Path("runs")
    data_dir: Path = Path("data")
    seed: int = 42
    num_bits: int = 4
    batch_size: int | None = None
    learning_rate: float = 3e-3
    qat_learning_rate: float = 5e-4
    baseline_epochs: int | None = None
    qat_epochs: int | None = None
    baseline_steps: int | None = None
    qat_steps: int | None = None
    device: str = "auto"
    workers: int = 0


PROFILE_DEFAULTS = {
    "smoke": {"baseline_epochs": 1, "qat_epochs": 1, "baseline_steps": 2, "qat_steps": 2, "batch_size": 16},
    "quick": {"baseline_epochs": 3, "qat_epochs": 2, "baseline_steps": 300, "qat_steps": 150, "batch_size": 64},
    "strong": {"baseline_epochs": 10, "qat_epochs": 5, "baseline_steps": 600, "qat_steps": 300, "batch_size": 64},
    "full": {"baseline_epochs": 20, "qat_epochs": 8, "baseline_steps": 3000, "qat_steps": 1000, "batch_size": 128},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve(config: ExperimentConfig, key: str) -> int:
    explicit = getattr(config, key)
    return int(explicit if explicit is not None else PROFILE_DEFAULTS[config.profile][key])


def _classification_eval(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(1) == targets).sum())
            count += targets.numel()
    return loss_sum / count, correct / count


def _train_classification(
    model: nn.Module,
    loader,
    test_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    phase: str,
    history: list[dict[str, Any]],
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * targets.numel()
            seen += targets.numel()
        val_loss, val_accuracy = _classification_eval(model, test_loader, device)
        history.append(
            {"phase": phase, "step_or_epoch": epoch, "train_loss": running / seen, "val_loss": val_loss, "val_metric": val_accuracy}
        )


def _train_language(
    model: nn.Module,
    encoded: torch.Tensor,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    block_size: int,
    seed: int,
    phase: str,
    history: list[dict[str, Any]],
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    model.to(device).train()
    log_every = max(1, steps // 5)
    for step in range(1, steps + 1):
        inputs, targets = sample_language_batch(encoded, block_size, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            history.append(
                {
                    "phase": phase,
                    "step_or_epoch": step,
                    "train_loss": float(loss.detach()),
                    "val_loss": "",
                    "val_metric": "",
                }
            )


def run_experiment(config: ExperimentConfig) -> Path:
    if config.model not in {"cnn", "vit", "llm"}:
        raise ValueError("model must be cnn, vit, or llm")
    if config.profile not in PROFILE_DEFAULTS:
        raise ValueError("profile must be smoke, quick, strong, or full")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    batch_size = config.batch_size if config.batch_size is not None else _resolve(config, "batch_size")
    history: list[dict[str, Any]] = []
    started = time.time()

    if config.model in {"cnn", "vit"}:
        train_loader, test_loader, dataset_label = vision_loaders(
            config.data_dir, batch_size, config.profile, config.seed, config.workers
        )
        baseline = build_model(config.model)
        baseline_epochs = _resolve(config, "baseline_epochs")
        qat_epochs = _resolve(config, "qat_epochs")
        _train_classification(
            baseline, train_loader, test_loader, device, baseline_epochs, config.learning_rate, "baseline", history
        )
        initial_model = copy.deepcopy(baseline).cpu()
        initial_report = classification_report(initial_model.to(device), test_loader, device)
        qat_model, quantized_names = prepare_po2_qat(initial_model, config.num_bits)
        _train_classification(qat_model, train_loader, test_loader, device, qat_epochs, config.qat_learning_rate, "qat", history)
        master_model = materialize_model(qat_model, quantized=False).to(device)
        po2_model = materialize_model(qat_model, quantized=True).to(device)
        master_report = classification_report(master_model, test_loader, device)
        po2_report = classification_report(po2_model, test_loader, device)
    else:
        text, dataset_label = load_text(config.data_dir, config.profile)
        split = int(len(text) * 0.9)
        tokenizer = CharacterTokenizer(text)
        train_tokens = tokenizer.encode(text[:split])
        val_tokens = tokenizer.encode(text[split:])
        block_size = min(96, len(val_tokens) - 2)
        baseline = build_model("llm", vocab_size=tokenizer.vocab_size, block_size=block_size)
        baseline_steps = _resolve(config, "baseline_steps")
        qat_steps = _resolve(config, "qat_steps")
        _train_language(
            baseline, train_tokens, device, baseline_steps, batch_size, config.learning_rate, block_size, config.seed, "baseline", history
        )
        initial_model = copy.deepcopy(baseline).cpu()
        initial_report = language_report(
            initial_model.to(device), val_tokens, block_size, batch_size, device, config.seed + 10
        )
        qat_model, quantized_names = prepare_po2_qat(initial_model, config.num_bits)
        _train_language(
            qat_model, train_tokens, device, qat_steps, batch_size, config.qat_learning_rate, block_size, config.seed + 1, "qat", history
        )
        master_model = materialize_model(qat_model, quantized=False).to(device)
        po2_model = materialize_model(qat_model, quantized=True).to(device)
        master_report = language_report(master_model, val_tokens, block_size, batch_size, device, config.seed + 10)
        po2_report = language_report(po2_model, val_tokens, block_size, batch_size, device, config.seed + 10)

    run_dir = config.output_dir / config.model
    evaluations = {
        "initial_fp32": initial_report,
        "qat_master_fp32": master_report,
        "po2_quantized": po2_report,
    }
    metrics = {
        "model": config.model,
        "dataset": dataset_label,
        "device": str(device),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "parameters": count_parameters(initial_model),
        "quantized_modules": len(quantized_names),
        "elapsed_seconds": time.time() - started,
    }
    for state, report in evaluations.items():
        for metric_name, value in report["summary"].items():
            metrics[f"{state}_{metric_name}"] = value

    initial_summary = evaluations["initial_fp32"]["summary"]
    po2_summary = evaluations["po2_quantized"]["summary"]
    if config.model in {"cnn", "vit"}:
        quality_checks = {
            "accuracy_drop_at_most_0.02": po2_summary["accuracy"] >= initial_summary["accuracy"] - 0.02,
            "loss_increase_at_most_5_percent": po2_summary["loss"] <= initial_summary["loss"] * 1.05,
        }
    else:
        quality_checks = {
            "perplexity_increase_at_most_5_percent": po2_summary["perplexity"] <= initial_summary["perplexity"] * 1.05,
            "top1_accuracy_drop_at_most_0.01": po2_summary["next_token_top1_accuracy"]
            >= initial_summary["next_token_top1_accuracy"] - 0.01,
        }
    metrics["quality_gate_passed"] = all(quality_checks.values())
    metrics["quality_gate_checks"] = quality_checks
    payload = asdict(config)
    payload.update(
        {
            "resolved_batch_size": batch_size,
            "resolved_baseline_epochs": _resolve(config, "baseline_epochs"),
            "resolved_qat_epochs": _resolve(config, "qat_epochs"),
            "resolved_baseline_steps": _resolve(config, "baseline_steps"),
            "resolved_qat_steps": _resolve(config, "qat_steps"),
        }
    )
    export_run(run_dir, initial_model, qat_model, metrics, payload, history, evaluations)
    return run_dir
