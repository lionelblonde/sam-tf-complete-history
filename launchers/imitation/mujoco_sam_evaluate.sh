#!/usr/bin/env bash
# Example: ./mujoco_sam_evaluate.sh <env_id> <sam_pol_ckpt_dir_path>

cd ../..

python -m imitation.imitation_algorithms.run_sam \
    --note="" \
    --env_id=$1 \
    --no-from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="evaluate_sam_policy" \
    --actorcritic_hid_widths 64 64 \
    --d_hid_widths 64 64 \
    --hid_nonlin="leaky_relu" \
    --num_trajs=20 \
    --no-render \
    --model_ckpt_dir=$2
