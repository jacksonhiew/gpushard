from typing import List, Dict
import requests
from requests import Response, HTTPError
import torch

from ..common.tensor_io import tensor_to_b64, b64_to_tensor
from .planner import plan_contiguous


class RingExecutor:
    def __init__(self, workers: List[Dict], n_layers: int):
        enriched = []
        for w in workers:
            info = dict(w)
            try:
                r = requests.get(f"{w['url']}/stats", timeout=5)
                r.raise_for_status()
                info['free_vram_mb'] = r.json().get('free_vram_mb', w.get('free_vram_mb', 0))
            except Exception:
                info.setdefault('free_vram_mb', w.get('free_vram_mb', 0))
            enriched.append(info)
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
            r = requests.post(f"{w['url']}/load", json=payload, timeout=5)
            r.raise_for_status()

    def forward_round(self, hidden: torch.Tensor, seq_id: int) -> torch.Tensor:
        expected_shape = list(hidden.shape)
        if len(expected_shape) != 3:
            resp = Response(); resp.status_code = 422
            raise HTTPError(f"Expected [B,T,H] tensor, got {expected_shape}", response=resp)
        b64, shape = tensor_to_b64(hidden)
        for w in self.workers:
            payload = {"hidden_b64": b64, "shape": shape, "seq_id": seq_id}
            r = requests.post(f"{w['url']}/forward", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data["shape"] != expected_shape:
                resp = Response(); resp.status_code = 422
                raise HTTPError(f"Shape mismatch from {w['url']}: {data['shape']} != {expected_shape}", response=resp)
            b64, shape = data["hidden_b64"], data["shape"]
        return b64_to_tensor(b64, shape, device=hidden.device, dtype=hidden.dtype)

    def reset_all(self, seq_id: int):
        for w in self.workers:
            requests.post(f"{w['url']}/reset", json={"seq_id": seq_id}, timeout=5)
