from typing import List, Optional
from pydantic import BaseModel

class LoadRequest(BaseModel):
    model_id: Optional[str]
    arch: str
    layer_start: int
    layer_end: int
    dtype: str = "fp16"

class LoadResponse(BaseModel):
    ok: bool
    hidden_size: int

class ForwardRequest(BaseModel):
    hidden_b64: str
    shape: List[int]
    seq_id: int

class ForwardResponse(BaseModel):
    hidden_b64: str
    shape: List[int]

class ResetRequest(BaseModel):
    seq_id: int

class ResetResponse(BaseModel):
    ok: bool

class StatsResponse(BaseModel):
    device: str
    start: int
    end: int
    free_vram_mb: int
