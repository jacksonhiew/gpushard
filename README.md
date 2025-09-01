# gpushard
GPU layering and sharding over network.

## Quickstart

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # from repo root
   pip install -e .
   ```

2. Launch workers (adjust ports/devices as needed):
   ```bash
   CUDA_VISIBLE_DEVICES=0 ringtorch/scripts/launch_worker_cuda.sh
   HIP_VISIBLE_DEVICES=0 ringtorch/scripts/launch_worker_hip.sh  # ROCm build; uses WORKER_DEVICE=cuda:0
   ```

   > **ROCm note:** PyTorch’s ROCm build exposes AMD GPUs via the `cuda` device type.
   > Keep `HIP_VISIBLE_DEVICES` for card selection, but set `WORKER_DEVICE=cuda:0`.
   > Using `hip:0` as a torch device will not work.

3. Launch coordinator:
   ```bash
   ringtorch/scripts/launch_coordinator.sh
   ```

4. Run fake tensor ring test:
   ```bash
   pytest -q tests/test_ring_fake_tensor.py
   ```

### One-shot forward-round test (no pytest)
```bash
python - <<'PY'
import torch, time
from ringtorch.coordinator.ring import RingExecutor
workers=[{"url":"http://127.0.0.1:9000"},{"url":"http://127.0.0.1:9001"}]
ring=RingExecutor(workers, n_layers=32)
ring.load_all(model_id=None, arch="dummy")
torch.manual_seed(0)
x=torch.randn(1,16,4096,dtype=torch.float16)
t=time.time(); y=ring.forward_round(x, seq_id=0)
print("latency:", round(time.time()-t,3),"s","shape:",tuple(y.shape),"changed:",not torch.allclose(x,y))
PY
```
