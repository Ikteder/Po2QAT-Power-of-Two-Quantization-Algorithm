import csv
import json

import numpy as np
import torch
from torch import nn

from po2qat.artifacts import export_run
from po2qat.quantization import prepare_po2_qat


def test_export_contains_reconstructable_weights(tmp_path):
    source = nn.Sequential(nn.Linear(4, 3))
    qat, _ = prepare_po2_qat(source)
    evaluations = {
        "initial_fp32": {"task": "test", "summary": {"loss": 1.0}},
        "qat_master_fp32": {"task": "test", "summary": {"loss": 0.9}},
        "po2_quantized": {"task": "test", "summary": {"loss": 1.1}},
    }
    export_run(tmp_path, source, qat, {"ok": True}, {"seed": 42}, [{"loss": 1.0}], evaluations)
    expected = {
        "initial_fp32.pt",
        "qat_master_fp32.pt",
        "po2_quantized.pt",
        "po2_sign_exponent.npz",
        "weight_comparison.csv",
        "weight_summary.csv",
        "metrics.json",
        "evaluation.json",
        "metrics_comparison.csv",
        "metric_deltas.csv",
        "quantization_metrics.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    rows = list(csv.DictReader((tmp_path / "weight_summary.csv").open()))
    assert rows[0]["exact_po2"] == "True"
    assert json.loads((tmp_path / "metrics.json").read_text())["ok"] is True
    assert len(list(csv.DictReader((tmp_path / "metrics_comparison.csv").open()))) == 3
    deltas = list(csv.DictReader((tmp_path / "metric_deltas.csv").open()))
    assert float(deltas[0]["po2_minus_initial"]) == 0.10000000000000009
    state = torch.load(tmp_path / "po2_quantized.pt", weights_only=True)
    packed = np.load(tmp_path / "po2_sign_exponent.npz")
    reconstructed = np.where(
        packed["0__weight__sign"] == 0,
        0.0,
        packed["0__weight__sign"] * np.power(2.0, packed["0__weight__exponent"]),
    )
    assert np.array_equal(reconstructed, state["0.weight"].numpy())
