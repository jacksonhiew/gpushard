#!/usr/bin/env bash
uvicorn ringtorch.coordinator.server:app --host 0.0.0.0 --port 8787
