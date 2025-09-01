from typing import Any
import torch.nn as nn


def get_blocks(model: Any, arch: str):
    arch = arch.lower()
    if arch in {"llama", "mistral", "qwen"}:
        return model.model.layers
    if arch in {"gpt-neox", "gptneox"}:
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {arch}")


def slice_blocks(model: Any, arch: str, start: int, end: int) -> nn.Sequential:
    blocks = get_blocks(model, arch)
    return nn.Sequential(*blocks[start:end])
