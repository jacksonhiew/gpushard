import base64
from typing import List, Tuple
import numpy as np
import torch

def tensor_to_b64(tensor: torch.Tensor) -> Tuple[str, List[int]]:
    """Serialize a tensor to base64 string along with shape metadata."""
    tensor = tensor.detach().to(dtype=torch.float16).contiguous().cpu()
    b64 = base64.b64encode(tensor.numpy().tobytes()).decode("ascii")
    shape = list(tensor.shape)
    return b64, shape

def b64_to_tensor(b64: str, shape: List[int], device: str, dtype: torch.dtype) -> torch.Tensor:
    """Deserialize tensor from base64 string."""
    data = base64.b64decode(b64.encode("ascii"))
    arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype).contiguous()
    return tensor
