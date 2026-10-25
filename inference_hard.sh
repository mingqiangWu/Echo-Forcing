#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python_bin="${ECHO_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
timestamp=$(date +%Y%m%d_%H%M%S)
output_folder="./output/hard/H_$timestamp"

echo "[Echo] Project: $SCRIPT_DIR"
echo "[Echo] Python: $python_bin | GPU: $CUDA_VISIBLE_DEVICES | verbose: ${ECHO_VERBOSE:-0}"
exec "$python_bin" -u inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder "$output_folder" \
    --data_path ./prompts/prompts_hard.txt \
    --use_ema \
    --save_with_index \
    --start_idx 0 \
    --end_idx 1 "$@"
