from typing import List, Dict
import requests
import torch

from ..common.tensor_io import tensor_to_b64, b64_to_tensor
from .planner import plan_contiguous


class RingExecutor:
    def __init__(self, workers: List[Dict], n_layers: int):
        # Enrich with live free_vram if missing
        enriched = []
        for w in workers:
            if "free_vram_mb" in w and isinstance(w["free_vram_mb"], int):
                enriched.append(w)
            else:
                try:
                    s = requests.get(f"{w['url']}/stats", timeout=3).json()
                    enriched.append({**w, "free_vram_mb": int(s.get("free_vram_mb", 0))})
                except Exception:
                    enriched.append({**w, "free_vram_mb": 0})
        self.workers = plan_contiguous(enriched, n_layers)

    def load_all(self, model_id, arch, dtype="fp16"):
        for w in self.workers:
            payload = {
                "model_id": model_id,
                "arch": arch,
                "layer_start": w["start"],
                "layer_end": w["end"],
                "dtype": dtype,
            }
            r = requests.post(f"{w['url']}/load", json=payload, timeout=30)
            r.raise_for_status()

    def forward_round(self, hidden: torch.Tensor, seq_id: int) -> torch.Tensor:
        b64, shape = tensor_to_b64(hidden)
        B, T, H = shape
        for w in self.workers:
            payload = {"hidden_b64": b64, "shape": shape, "seq_id": seq_id}
            r = requests.post(f"{w['url']}/forward", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            b64, shape = data["hidden_b64"], data["shape"]
            if shape != [B, T, H]:
                raise RuntimeError(f"Worker {w['url']} returned shape {shape}, expected {[B, T, H]}")
        return b64_to_tensor(b64, shape, device=hidden.device, dtype=hidden.dtype)

    def reset_all(self, seq_id: int):
        for w in self.workers:
            requests.post(f"{w['url']}/reset", json={"seq_id": seq_id}, timeout=5)
