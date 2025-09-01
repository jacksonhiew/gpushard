from typing import List
from pydantic import BaseModel

class ForwardInput(BaseModel):
    hidden_b64: str
    shape: List[int]
    seq_id: int

class ForwardOutput(BaseModel):
    hidden_b64: str
    shape: List[int]
