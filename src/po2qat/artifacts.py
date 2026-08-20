from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .quantization import materialize_model, quantized_module_metadata


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _is_weight(name: str, value: torch.Tensor) -> bool:
    return name.endswith("weight") and value.ndim >= 2 and value.is_floating_point()


def _is_exact_po2(values: torch.Tensor) -> bool:
    nonzero = values.detach().cpu().abs()
    nonzero = nonzero[nonzero != 0]
    if nonzero.numel() == 0:
        return True
    logs = torch.log2(nonzero)
    return bool(torch.allclose(logs, torch.round(logs), atol=1e-6, rtol=0))


def export_run(
    run_dir: Path,
    initial_model: nn.Module,
    qat_model: nn.Module,
    metrics: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    initial = initial_model.cpu().state_dict()
    master_model = materialize_model(qat_model, quantized=False)
    po2_model = materialize_model(qat_model, quantized=True)
    master = master_model.state_dict()
    po2 = po2_model.state_dict()
    torch.save(initial, run_dir / "initial_fp32.pt")
    torch.save(master, run_dir / "qat_master_fp32.pt")
    torch.save(po2, run_dir / "po2_quantized.pt")

    module_meta = quantized_module_metadata(qat_model)
    save_json(run_dir / "quantization_metadata.json", module_meta)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "evaluation.json", evaluations)

    eligible_names = [f"{name}.weight" for name in module_meta]
    initial_flat = torch.cat([initial[name].detach().cpu().float().flatten() for name in eligible_names])
    master_flat = torch.cat([master[name].detach().cpu().float().flatten() for name in eligible_names])
    po2_flat = torch.cat([po2[name].detach().cpu().float().flatten() for name in eligible_names])

    def comparison(reference: torch.Tensor, projected: torch.Tensor, prefix: str) -> dict[str, float]:
        difference = reference - projected
        cosine = torch.nn.functional.cosine_similarity(reference, projected, dim=0)
        return {
            f"{prefix}_mae": float(difference.abs().mean()),
            f"{prefix}_rmse": float(torch.sqrt(torch.mean(difference.square()))),
            f"{prefix}_max_absolute_error": float(difference.abs().max()),
            f"{prefix}_cosine_similarity": float(cosine),
        }

    bit_values = {metadata["num_bits"] for metadata in module_meta.values()}
    uniform_bits = next(iter(bit_values)) if len(bit_values) == 1 else None
    encoded_bits = sum(initial[f"{name}.weight"].numel() * metadata["num_bits"] for name, metadata in module_meta.items())
    fp32_bits = initial_flat.numel() * 32
    nonzero = po2_flat[po2_flat != 0]
    quantization_metrics: dict[str, Any] = {
        "eligible_parameters": initial_flat.numel(),
        "configured_bits": uniform_bits,
        "po2_zero_fraction": float((po2_flat == 0).float().mean()),
        "distinct_nonzero_po2_values_across_model": int(torch.unique(nonzero).numel()),
        "fp32_eligible_storage_bits": fp32_bits,
        "theoretical_po2_storage_bits": encoded_bits,
        "theoretical_eligible_compression_ratio": fp32_bits / max(encoded_bits, 1),
        **comparison(initial_flat, po2_flat, "initial_to_po2"),
        **comparison(master_flat, po2_flat, "qat_master_to_po2"),
    }
    save_json(run_dir / "quantization_metrics.json", quantization_metrics)
    metrics.update({f"quantization_{key}": value for key, value in quantization_metrics.items()})
    save_json(run_dir / "metrics.json", metrics)

    metric_names = sorted({key for report in evaluations.values() for key in report["summary"]})
    with (run_dir / "metrics_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state", *metric_names])
        writer.writeheader()
        for state, report in evaluations.items():
            writer.writerow({"state": state, **report["summary"]})

    with (run_dir / "metric_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "metric",
            "initial_fp32",
            "qat_master_fp32",
            "po2_quantized",
            "qat_master_minus_initial",
            "po2_minus_initial",
            "po2_percent_change",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        common_metrics = set.intersection(*(set(report["summary"]) for report in evaluations.values()))
        for metric in sorted(common_metrics):
            initial_metric = evaluations["initial_fp32"]["summary"][metric]
            master_metric = evaluations["qat_master_fp32"]["summary"][metric]
            po2_metric = evaluations["po2_quantized"]["summary"][metric]
            if not all(isinstance(value, (int, float)) for value in (initial_metric, master_metric, po2_metric)):
                continue
            writer.writerow(
                {
                    "metric": metric,
                    "initial_fp32": initial_metric,
                    "qat_master_fp32": master_metric,
                    "po2_quantized": po2_metric,
                    "qat_master_minus_initial": master_metric - initial_metric,
                    "po2_minus_initial": po2_metric - initial_metric,
                    "po2_percent_change": "" if initial_metric == 0 else 100 * (po2_metric - initial_metric) / abs(initial_metric),
                }
            )

    for state, report in evaluations.items():
        if report.get("per_class"):
            per_class = report["per_class"]
            with (run_dir / f"per_class_metrics_{state}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
                writer.writeheader()
                writer.writerows(per_class)
        if report.get("confusion_matrix"):
            names = report["class_names"]
            with (run_dir / f"confusion_matrix_{state}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["actual\\predicted", *names])
                for name, row in zip(names, report["confusion_matrix"], strict=True):
                    writer.writerow([name, *row])

    if evaluations["initial_fp32"].get("per_class"):
        initial_rows = evaluations["initial_fp32"]["per_class"]
        po2_rows = evaluations["po2_quantized"]["per_class"]
        fields = ["class_index", "class_name", "precision_delta", "recall_delta", "f1_delta", "support"]
        with (run_dir / "per_class_delta_initial_to_po2.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for initial_row, po2_row in zip(initial_rows, po2_rows, strict=True):
                writer.writerow(
                    {
                        "class_index": initial_row["class_index"],
                        "class_name": initial_row["class_name"],
                        "precision_delta": po2_row["precision"] - initial_row["precision"],
                        "recall_delta": po2_row["recall"] - initial_row["recall"],
                        "f1_delta": po2_row["f1"] - initial_row["f1"],
                        "support": initial_row["support"],
                    }
                )

    if history:
        with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    comparison_fields = [
        "tensor", "flat_index", "initial_fp32", "qat_master_fp32", "po2_weight", "po2_expression", "quantized_tensor"
    ]
    summary_fields = [
        "tensor", "shape", "parameters", "quantized", "exp_min", "exp_max", "mae_initial_to_po2", "mae_master_to_po2", "exact_po2"
    ]
    comparison_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    packed: dict[str, np.ndarray] = {}
    for name, final_value in po2.items():
        if not _is_weight(name, final_value):
            continue
        module_name = name.rsplit(".", 1)[0]
        is_quantized = module_name in module_meta
        initial_value = initial[name].detach().cpu().float()
        master_value = master[name].detach().cpu().float()
        final_value = final_value.detach().cpu().float()
        meta = module_meta.get(module_name, {})
        exact = _is_exact_po2(final_value) if is_quantized else False
        summary_rows.append(
            {
                "tensor": name,
                "shape": "x".join(map(str, final_value.shape)),
                "parameters": final_value.numel(),
                "quantized": is_quantized,
                "exp_min": meta.get("exp_min", ""),
                "exp_max": meta.get("exp_max", ""),
                "mae_initial_to_po2": float((initial_value - final_value).abs().mean()),
                "mae_master_to_po2": float((master_value - final_value).abs().mean()),
                "exact_po2": exact,
            }
        )
        initial_flat, master_flat, final_flat = initial_value.flatten(), master_value.flatten(), final_value.flatten()
        sample_count = min(16, final_flat.numel())
        sample_indices = torch.linspace(0, final_flat.numel() - 1, steps=sample_count).long().unique()
        for index in sample_indices.tolist():
            value = float(final_flat[index])
            expression = "0" if value == 0 else f"{'-' if value < 0 else '+'}2^{int(round(np.log2(abs(value))))}"
            comparison_rows.append(
                {
                    "tensor": name,
                    "flat_index": index,
                    "initial_fp32": float(initial_flat[index]),
                    "qat_master_fp32": float(master_flat[index]),
                    "po2_weight": value,
                    "po2_expression": expression if is_quantized else "not quantized",
                    "quantized_tensor": is_quantized,
                }
            )
        if is_quantized:
            array = final_value.numpy()
            signs = np.sign(array).astype(np.int8)
            exponents = np.zeros(array.shape, dtype=np.int16)
            nonzero = signs != 0
            exponents[nonzero] = np.rint(np.log2(np.abs(array[nonzero]))).astype(np.int16)
            safe_name = name.replace(".", "__")
            packed[f"{safe_name}__sign"] = signs
            packed[f"{safe_name}__exponent"] = exponents

    with (run_dir / "weight_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (run_dir / "weight_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    np.savez_compressed(run_dir / "po2_sign_exponent.npz", **packed)


def read_summary(run_dir: Path) -> str:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lines = [f"Run complete: {run_dir}"]
    lines.append(f"  Model/data: {metrics.get('model')} / {metrics.get('dataset')}")
    lines.append(f"  Device: {metrics.get('device')} | Parameters: {metrics.get('parameters'):,}")
    if "initial_fp32_accuracy" in metrics:
        lines.append(
            "  Accuracy: "
            f"initial={metrics['initial_fp32_accuracy']:.2%} -> Po2={metrics['po2_quantized_accuracy']:.2%}"
        )
        lines.append(
            "  Loss: "
            f"initial={metrics['initial_fp32_loss']:.4f} -> Po2={metrics['po2_quantized_loss']:.4f}"
        )
        lines.append(
            "  Macro F1: "
            f"initial={metrics['initial_fp32_macro_f1']:.4f} -> Po2={metrics['po2_quantized_macro_f1']:.4f}"
        )
    else:
        lines.append(
            "  Perplexity: "
            f"initial={metrics['initial_fp32_perplexity']:.4f} -> Po2={metrics['po2_quantized_perplexity']:.4f}"
        )
        lines.append(
            "  Token accuracy: "
            f"initial={metrics['initial_fp32_next_token_top1_accuracy']:.2%} "
            f"-> Po2={metrics['po2_quantized_next_token_top1_accuracy']:.2%}"
        )
    gate = "PASS" if metrics.get("quality_gate_passed") else "WARNING"
    lines.append(f"  Quality gate: {gate}")
    lines.append(
        "  Theoretical eligible-weight compression: "
        f"{metrics.get('quantization_theoretical_eligible_compression_ratio', 0):.1f}x"
    )
    lines.append(f"  Full metrics: {run_dir / 'metrics_comparison.csv'}")
    lines.append(f"  Weight comparison: {run_dir / 'weight_comparison.csv'}")
    return "\n".join(lines)
