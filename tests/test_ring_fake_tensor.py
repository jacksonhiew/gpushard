import os
import time
import multiprocessing
import requests
import torch
import numpy as np

from ringtorch.coordinator.ring import RingExecutor


def run_worker(port: int, device: str):
    os.environ["WORKER_DEVICE"] = device
    import uvicorn
    uvicorn.run("ringtorch.worker.server:app", host="127.0.0.1", port=port, log_level="warning")


def start_worker(port: int, device: str):
    p = multiprocessing.Process(target=run_worker, args=(port, device))
    p.start()
    # wait for server
    for _ in range(50):
        try:
            requests.get(f"http://127.0.0.1:{port}/stats", timeout=0.2)
            break
        except Exception:
            time.sleep(0.1)
    return p


def test_ring_fake_tensor():
    torch.manual_seed(0)
    np.random.seed(0)
    p1 = start_worker(9000, "cpu")
    p2 = start_worker(9001, "cpu")
    try:
        r1 = requests.get("http://127.0.0.1:9000/stats").json()
        r2 = requests.get("http://127.0.0.1:9001/stats").json()
        assert r1["device"] == "cpu" and r2["device"] == "cpu"

        workers = [
            {"url": "http://127.0.0.1:9000", "free_vram_mb": 12000},
            {"url": "http://127.0.0.1:9001", "free_vram_mb": 8000},
        ]
        ring = RingExecutor(workers, n_layers=32)
        ring.load_all(model_id=None, arch="dummy")

        # ensure contiguous plan
        assert ring.workers[0]["start"] == 0
        assert ring.workers[-1]["end"] == 32
        assert ring.workers[0]["end"] == ring.workers[1]["start"]

        hidden = torch.randn(1, 16, 4096, dtype=torch.float16)
        t0 = time.time()
        out = ring.forward_round(hidden, seq_id=0)
        t1 = time.time()
        assert out.shape == hidden.shape
        assert not torch.allclose(out, hidden)
        assert (t1 - t0) > 0

        ring.reset_all(seq_id=0)
    finally:
        p1.terminate(); p1.join()
        p2.terminate(); p2.join()
