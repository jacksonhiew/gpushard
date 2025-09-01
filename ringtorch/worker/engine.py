from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from . import model_slicer


@dataclass
class TorchSliceEngine:
    device: str = "cpu"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        self.model: Optional[nn.Module] = None
        self.start: int = 0
        self.end: int = 0
        self.hidden_size: int = 0

    def load(self, model_id: Optional[str], arch: str, start: int, end: int, hidden_size: int = 4096):
        self.start, self.end = start, end
        if model_id is None:
            layers = []
            # [dummy stack] Linear + LayerNorm repeating with fixed hidden size
            for _ in range(end - start):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                layers.append(nn.LayerNorm(hidden_size))
            self.hidden_size = hidden_size
            self.model = nn.Sequential(*layers).to(self.device, dtype=self.dtype)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            blocks = model_slicer.slice_blocks(model, arch, start, end)
            self.model = blocks.to(self.device, dtype=self.dtype)
            self.hidden_size = model.config.hidden_size
            try:
                del model
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return self.hidden_size

    def forward_slice(self, hidden: torch.Tensor, seq_id: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model slice not loaded")
        with torch.inference_mode():
            hidden = hidden.to(self.device, dtype=self.dtype)
            out = self.model(hidden)
        return out.detach()

    def reset(self, seq_id: int):
        # No persistent state in this MVP
        return True
