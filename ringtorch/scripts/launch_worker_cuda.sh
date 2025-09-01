#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export WORKER_DEVICE=${WORKER_DEVICE:-cuda:0}
uvicorn ringtorch.worker.server:app --host 0.0.0.0 --port 9000
