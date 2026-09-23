import torch
from torch import nn

from po2qat.experiment import PROFILE_DEFAULTS
from po2qat.models import build_model


def test_model_shapes():
    assert build_model("cnn")(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert build_model("vit")(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert build_model("llm", vocab_size=31, block_size=16)(torch.randint(0, 31, (2, 16))).shape == (2, 16, 31)


def test_large_vision_model_shapes_and_sizes():
    with torch.inference_mode():
        vgg19 = build_model("vgg19").eval()
        assert vgg19(torch.randn(1, 3, 32, 32)).shape == (1, 10)
        assert sum(parameter.numel() for parameter in vgg19.parameters()) > 20_000_000
        del vgg19

        resnet50 = build_model("resnet50").eval()
        assert resnet50(torch.randn(1, 3, 32, 32)).shape == (1, 10)
        assert sum(parameter.numel() for parameter in resnet50.parameters()) > 23_000_000


def test_large_models_expose_expected_quantizable_layers():
    expected = {"vgg19": 18, "resnet50": 54}
    for name, count in expected.items():
        model = build_model(name)
        eligible = sum(isinstance(module, (nn.Conv2d, nn.Linear)) for module in model.modules())
        assert eligible == count


def test_measured_strong_profile_is_available():
    assert PROFILE_DEFAULTS["strong"] == {
        "baseline_epochs": 10,
        "qat_epochs": 5,
        "baseline_steps": 600,
        "qat_steps": 300,
        "batch_size": 64,
    }
