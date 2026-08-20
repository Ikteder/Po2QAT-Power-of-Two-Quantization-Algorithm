from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _subset(dataset: Dataset, size: int, seed: int) -> Dataset:
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=generator)[:size].tolist())


def vision_loaders(
    data_dir: Path,
    batch_size: int,
    profile: str,
    seed: int,
    workers: int = 0,
) -> tuple[DataLoader, DataLoader, str]:
    if profile == "smoke":
        generator = torch.Generator().manual_seed(seed)
        train_x = torch.rand(64, 3, 32, 32, generator=generator)
        train_y = torch.randint(0, 10, (64,), generator=generator)
        test_x = torch.rand(32, 3, 32, 32, generator=generator)
        test_y = torch.randint(0, 10, (32,), generator=generator)
        train_set: Dataset = TensorDataset(train_x, train_y)
        test_set: Dataset = TensorDataset(test_x, test_y)
        label = "deterministic synthetic smoke data"
    else:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )
        test_transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        if profile in {"quick", "strong"}:
            train_set = _subset(train_set, 4096, seed)
            test_set = _subset(test_set, 1000, seed + 1)
        label = "CIFAR-10 fixed classroom subset" if profile in {"quick", "strong"} else "CIFAR-10 full dataset"
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train, test, label


SMOKE_TEXT = ("power of two quantization helps small models use shift based arithmetic. " * 700).strip()


def load_text(data_dir: Path, profile: str) -> tuple[str, str]:
    if profile == "smoke":
        return SMOKE_TEXT, "built-in deterministic smoke text"
    path = data_dir / "tinyshakespeare" / "input.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    text = path.read_text(encoding="utf-8")
    if profile in {"quick", "strong"}:
        text = text[:200_000]
    return text, "Tiny Shakespeare classroom slice" if profile in {"quick", "strong"} else "Tiny Shakespeare full text"


class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        self.characters = sorted(set(text))
        self.to_id = {character: index for index, character in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.to_id[character] for character in text], dtype=torch.long)


def sample_language_batch(
    encoded: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(encoded) - block_size - 1, (batch_size,), generator=generator)
    inputs = torch.stack([encoded[start : start + block_size] for start in starts])
    targets = torch.stack([encoded[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)
