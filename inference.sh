#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
timestamp=$(date +%Y%m%d_%H%M%S)
output_folder="./output/K1_$timestamp"

python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder "$output_folder" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path ./prompts/moviegenbench_128_refined.txt \
    --num_output_frames 672 \
    --use_ema \
    --save_with_index \
    --seed 0 \
    --start_idx 0 \
    --end_idx 1
