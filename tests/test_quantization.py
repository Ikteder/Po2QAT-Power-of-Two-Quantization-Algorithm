import torch
from torch import nn

from po2qat.quantization import fake_quantize_po2, materialize_model, prepare_po2_qat, quantize_po2


def test_projection_uses_only_zero_or_exact_powers_of_two():
    values = torch.tensor([-0.3, -0.01, 0.0, 0.07, 0.49, 1.2])
    projected = quantize_po2(values, -3, 1)
    nonzero = projected.abs()[projected != 0]
    assert torch.allclose(torch.log2(nonzero), torch.round(torch.log2(nonzero)))


def test_ste_passes_gradient():
    values = torch.tensor([0.2, -0.7], requires_grad=True)
    fake_quantize_po2(values, -4, 0).sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_wrapped_and_materialized_models_have_matching_forward():
    torch.manual_seed(1)
    source = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    qat, names = prepare_po2_qat(source, 4)
    materialized = materialize_model(qat, quantized=True)
    inputs = torch.randn(3, 4)
    assert names == ["0", "2"]
    assert torch.allclose(qat(inputs), materialized(inputs), atol=1e-6)

