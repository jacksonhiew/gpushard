from typing import List, Dict, Optional
from fastapi import FastAPI
import torch

from .ring import RingExecutor
from ..common.tensor_io import tensor_to_b64, b64_to_tensor
from .schemas import ForwardInput, ForwardOutput

app = FastAPI()
ring: Optional[RingExecutor] = None


@app.post("/init")
def init_ring(workers: List[Dict], n_layers: int):
    global ring
    ring = RingExecutor(workers, n_layers)
    return {"ok": True}


@app.post("/load")
def load(model_id: Optional[str], arch: str, dtype: str = "fp16"):
    if ring is None:
        raise RuntimeError("Ring not initialized")
    ring.load_all(model_id, arch, dtype)
    return {"ok": True}


@app.post("/forward", response_model=ForwardOutput)
def forward(req: ForwardInput):
    if ring is None:
        raise RuntimeError("Ring not initialized")
    hidden = b64_to_tensor(req.hidden_b64, req.shape, device="cpu", dtype=torch.float16)
    out = ring.forward_round(hidden, req.seq_id)
    b64, shape = tensor_to_b64(out)
    return ForwardOutput(hidden_b64=b64, shape=shape)
