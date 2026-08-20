import torch

from po2qat.experiment import PROFILE_DEFAULTS
from po2qat.models import build_model


def test_model_shapes():
    assert build_model("cnn")(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert build_model("vit")(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert build_model("llm", vocab_size=31, block_size=16)(torch.randint(0, 31, (2, 16))).shape == (2, 16, 31)


def test_measured_strong_profile_is_available():
    assert PROFILE_DEFAULTS["strong"] == {
        "baseline_epochs": 10,
        "qat_epochs": 5,
        "baseline_steps": 600,
        "qat_steps": 300,
        "batch_size": 64,
    }
