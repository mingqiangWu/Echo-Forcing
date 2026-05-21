#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
timestamp=$(date +%Y%m%d_%H%M%S)
output_folder="./output/recall/R_$timestamp"

python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder "$output_folder" \
    --data_path ./prompts/prompts_recall.txt \
    --use_ema \
    --save_with_index \
    --start_idx 0 \
    --end_idx 1
