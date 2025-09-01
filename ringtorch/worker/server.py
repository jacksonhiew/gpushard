import os
from fastapi import FastAPI, HTTPException
import torch

from .engine import TorchSliceEngine
from .schemas import (
    LoadRequest,
    LoadResponse,
    ForwardRequest,
    ForwardResponse,
    ResetRequest,
    ResetResponse,
    StatsResponse,
)
from ..common.tensor_io import tensor_to_b64, b64_to_tensor


def _normalize_device(dev: str) -> str:
    # PyTorch ROCm exposes AMD via 'cuda' device type too
    if dev.startswith("hip"):
        return dev.replace("hip", "cuda", 1)
    return dev


def get_free_vram_mb(device: str) -> int:
    dev = _normalize_device(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(dev)
        return int(free // (1024 * 1024))
    return 0


DEVICE = _normalize_device(os.environ.get("WORKER_DEVICE", "cpu"))
DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}.get(os.environ.get("WORKER_DTYPE", "fp16"), torch.float16)

engine = TorchSliceEngine(device=DEVICE, dtype=DTYPE)
app = FastAPI()

@app.post("/load", response_model=LoadResponse)
def load(req: LoadRequest):
    hidden_size = engine.load(req.model_id, req.arch, req.layer_start, req.layer_end)
    return LoadResponse(ok=True, hidden_size=hidden_size)

@app.post("/forward", response_model=ForwardResponse)
def forward(req: ForwardRequest):
    # Shape guard: [B,T,H] and H must match engine.hidden_size when known
    if len(req.shape) != 3:
        raise HTTPException(status_code=422, detail=f"Expected [B,T,H], got {req.shape}")
    if engine.hidden_size and req.shape[-1] != engine.hidden_size:
        raise HTTPException(status_code=422, detail=f"H mismatch: {req.shape[-1]} != {engine.hidden_size}")
    hidden = b64_to_tensor(req.hidden_b64, req.shape, engine.device, engine.dtype)
    out = engine.forward_slice(hidden, req.seq_id)
    b64, shape = tensor_to_b64(out)
    return ForwardResponse(hidden_b64=b64, shape=shape)

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    engine.reset(req.seq_id)
    return ResetResponse(ok=True)

@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(
        device=engine.device,
        start=engine.start,
        end=engine.end,
        free_vram_mb=get_free_vram_mb(engine.device),
    )
