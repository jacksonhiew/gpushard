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
   HIP_VISIBLE_DEVICES=0 WORKER_DEVICE=cuda:0 ringtorch/scripts/launch_worker_hip.sh   # or set WORKER_DEVICE=cpu
   ```

   > **ROCm note:** PyTorch with ROCm still uses the `cuda` device type. Run AMD workers with `WORKER_DEVICE=cuda:0`. Using `hip:0` will not work.

3. Launch coordinator:
   ```bash
   ringtorch/scripts/launch_coordinator.sh
   ```

4. Run fake tensor ring test:
   ```bash
   pytest -q tests/test_ring_fake_tensor.py
   ```
