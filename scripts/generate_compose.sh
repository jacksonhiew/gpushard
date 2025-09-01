#!/usr/bin/env bash
set -euo pipefail

# Generates docker-compose.override.yaml with one worker per detected GPU
# - NVIDIA GPUs via nvidia-smi
# - AMD GPUs via rocminfo
# Also injects RING_WORKERS env for coordinator

OUT=docker-compose.override.yaml

detect_nvidia() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
  else
    echo 0
  fi
}

detect_amd() {
  if command -v rocminfo >/dev/null 2>&1; then
    # Count GPU agents (simple heuristic)
    rocminfo 2>/dev/null | awk '/Agent.*gpu/ {c++} END {print c+0}' || echo 0
  else
    echo 0
  fi
}

N_NVIDIA=$(detect_nvidia)
N_AMD=$(detect_amd)

echo "Detected NVIDIA GPUs: $N_NVIDIA, AMD GPUs: $N_AMD"

PORT_BASE_CUDA=9000
PORT_BASE_ROCM=9100

cat > "$OUT" <<YAML
version: "3.9"

services:
YAML

WORKER_URLS=()

# NVIDIA workers
for i in $(seq 0 $((N_NVIDIA-1))); do
  PORT=$((PORT_BASE_CUDA + i))
  NAME="worker_cuda_${i}"
  cat >> "$OUT" <<YAML
  ${NAME}:
    extends:
      file: docker-compose.yaml
      service: worker_cuda
    container_name: ${NAME}
    environment:
      - NVIDIA_VISIBLE_DEVICES=${i}
      - WORKER_DEVICE=cuda:0
    ports:
      - "${PORT}:9000"
YAML
  WORKER_URLS+=("http://127.0.0.1:${PORT}")
done

# AMD workers
for i in $(seq 0 $((N_AMD-1))); do
  PORT=$((PORT_BASE_ROCM + i))
  NAME="worker_rocm_${i}"
  cat >> "$OUT" <<YAML
  ${NAME}:
    extends:
      file: docker-compose.yaml
      service: worker_rocm
    container_name: ${NAME}
    environment:
      - HIP_VISIBLE_DEVICES=${i}
      - WORKER_DEVICE=cuda:0
    ports:
      - "${PORT}:9001"
YAML
  WORKER_URLS+=("http://127.0.0.1:${PORT}")
done

# Coordinator with RING_WORKERS env (informational; server does not auto-init)
if (( ${#WORKER_URLS[@]} > 0 )); then
  # shellcheck disable=SC2016
  JSON=$(python - <<PY
import json, os
urls = os.environ.get('URLS','').split(',') if os.environ.get('URLS') else []
print(json.dumps([{"url": u} for u in urls]))
PY
  )
  URLS=$(IFS=,; echo "${WORKER_URLS[*]}") URLS="$URLS" python - <<'PY' URLS="$URLS" OUT="$OUT"
import os, json
urls = os.environ['URLS'].split(',')
workers = json.dumps([{"url": u} for u in urls])
with open(os.environ['OUT'], 'a') as f:
    f.write('  coordinator:\n')
    f.write('    environment:\n')
    f.write(f'      - RING_WORKERS={workers}\n')
PY
fi

echo "Wrote $OUT"

