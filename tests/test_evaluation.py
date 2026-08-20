import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from po2qat.evaluation import classification_report, language_report


class FixedClassifier(nn.Module):
    def forward(self, inputs):
        return inputs


class FixedLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, tokens):
        return torch.nn.functional.one_hot(tokens, self.vocab_size).float()


def test_classification_report_has_aggregate_per_class_and_confusion_metrics():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [4.0, 1.0], [1.0, 4.0]])
    targets = torch.tensor([0, 1, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    report = classification_report(FixedClassifier(), loader, torch.device("cpu"), ["zero", "one"])
    assert report["summary"]["accuracy"] == 0.75
    assert len(report["per_class"]) == 2
    assert report["confusion_matrix"] == [[1, 0], [1, 2]]
    assert "macro_f1" in report["summary"]


def test_language_report_has_perplexity_bpc_and_token_accuracy():
    encoded = torch.arange(200) % 7
    report = language_report(FixedLanguageModel(7), encoded, 8, 2, torch.device("cpu"), seed=4, batches=2)
    summary = report["summary"]
    assert set(["cross_entropy_loss", "perplexity", "bits_per_character", "next_token_top1_accuracy"]).issubset(summary)
    assert summary["evaluated_tokens"] == 32
