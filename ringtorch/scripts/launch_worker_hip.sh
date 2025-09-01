#!/usr/bin/env bash
export HIP_VISIBLE_DEVICES=0
export WORKER_DEVICE=${WORKER_DEVICE:-hip:0}
uvicorn ringtorch.worker.server:app --host 0.0.0.0 --port 9001
