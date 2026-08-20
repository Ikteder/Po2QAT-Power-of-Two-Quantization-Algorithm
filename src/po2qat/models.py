from __future__ import annotations

import math

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, groups: int = 1) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden, 1),
            ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        return inputs + output if self.use_residual else output


class MobileNetTiny(nn.Module):
    """A small MobileNet-style CNN sized for classroom CPU experiments."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, 3),
            InvertedResidual(24, 32, 1),
            InvertedResidual(32, 48, 2),
            InvertedResidual(48, 48, 1),
            InvertedResidual(48, 72, 2),
            InvertedResidual(72, 72, 1),
            ConvBNAct(72, 96, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, causal: bool = False) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = inputs.shape
        normalized = self.norm1(inputs)
        qkv = self.qkv(normalized).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        attention = attention.transpose(1, 2).reshape(batch, tokens, dim)
        inputs = inputs + self.proj(attention)
        return inputs + self.mlp(self.norm2(inputs))


class TinyViT(nn.Module):
    """A compact Vision Transformer for 32x32 CIFAR-10 images."""
    def __init__(self, num_classes: int = 10, dim: int = 96, depth: int = 3, heads: int = 3, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(inputs.shape[0], -1, -1)
        encoded = self.blocks(torch.cat((cls, patches), dim=1) + self.position)
        return self.head(self.norm(encoded[:, 0]))


class TinyGPT(nn.Module):
    """A decoder-only character language model; intentionally tiny, not a production LLM."""
    def __init__(self, vocab_size: int, block_size: int = 96, dim: int = 128, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, causal=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.lm_head(self.norm(self.blocks(hidden)))


def build_model(name: str, vocab_size: int | None = None, block_size: int = 96) -> nn.Module:
    if name == "cnn":
        return MobileNetTiny()
    if name == "vit":
        return TinyViT()
    if name == "llm":
        if vocab_size is None:
            raise ValueError("vocab_size is required for the LLM")
        return TinyGPT(vocab_size=vocab_size, block_size=block_size)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

