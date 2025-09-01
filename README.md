# gpushard
GPU layering and sharding over network.

## Quickstart

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ringtorch
   ```

2. Launch workers (adjust ports/devices as needed):
   ```bash
   CUDA_VISIBLE_DEVICES=0 ringtorch/scripts/launch_worker_cuda.sh
   HIP_VISIBLE_DEVICES=0 ringtorch/scripts/launch_worker_hip.sh   # or set WORKER_DEVICE=cpu
   ```

3. Launch coordinator:
   ```bash
   ringtorch/scripts/launch_coordinator.sh
   ```

4. Run fake tensor ring test:
   ```bash
   pytest -q tests/test_ring_fake_tensor.py
   ```
