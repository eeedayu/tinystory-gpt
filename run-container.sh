#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-tinystory-gpt:0.1}"
host="${HOST_BIND:-127.0.0.1}"
port="${PORT:-8000}"
gpu_args=()

if ! docker run --rm --entrypoint python "$image" -c \
    'from pathlib import Path; import os; p=Path("/model/model.safetensors"); ok=p.is_file() and os.access(p, os.R_OK); print(f"Model weights missing or unreadable: {p}") if not ok else None; raise SystemExit(not ok)' \
    >/dev/null 2>&1; then
    echo "The image '$image' does not contain readable /model/model.safetensors." >&2
    echo "From the project root, rebuild it with:" >&2
    echo "  sudo docker build --no-cache -t $image ." >&2
    exit 1
fi

if docker run --rm --gpus all --entrypoint python "$image" \
    -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1; then
    gpu_args=(--gpus all)
    echo "NVIDIA GPU available to Docker; starting with GPU support."
else
    echo "No usable NVIDIA GPU exposed to Docker; starting on CPU."
fi

exec docker run --rm \
    "${gpu_args[@]}" \
    -p "${host}:${port}:8000" \
    --read-only \
    --tmpfs /tmp \
    --memory 4g \
    --cpus 2 \
    "$image"
