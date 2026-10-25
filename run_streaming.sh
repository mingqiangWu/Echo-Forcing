#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    decoder_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F, '$1 != 0 && $2 > free {free = $2; gpu = $1} END {if (free >= 16000) print gpu}')
    if [[ -z "$decoder_gpu" ]]; then
        echo "[Echo] No second GPU has 16 GiB free for VAE decoding." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="0,$decoder_gpu"
fi
if [[ -z "${ECHO_VAE_DEVICE:-}" && "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    export ECHO_VAE_DEVICE=cuda:1
fi
echo "[Echo] Generator / decoder GPUs: $CUDA_VISIBLE_DEVICES (VAE: ${ECHO_VAE_DEVICE:-same GPU})"
export GRADIO_ANALYTICS_ENABLED=False
export ECHO_HOST="${ECHO_HOST:-127.0.0.1}"
export ECHO_PORT="${ECHO_PORT:-1325}"
PYTHON="${ECHO_PYTHON:-python}"
exec "$PYTHON" -u gradio_demo.py "$@"
