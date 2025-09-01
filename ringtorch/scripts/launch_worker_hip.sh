#!/usr/bin/env bash
export HIP_VISIBLE_DEVICES=0
# PyTorch ROCm uses 'cuda' device type; HIP env selects the AMD card
export WORKER_DEVICE=${WORKER_DEVICE:-cuda:0}
uvicorn ringtorch.worker.server:app --host 0.0.0.0 --port 9001
