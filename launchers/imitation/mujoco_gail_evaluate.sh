#!/usr/bin/env bash
# Example: ./mujoco_gail_evaluate.sh <env_id> <gail_pol_ckpt_dir_path>

cd ../..

python -m imitation.imitation_algorithms.run_gail \
    --note="" \
    --env_id=$1 \
    --no-from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="evaluate_gail_policy" \
    --pol_hid_widths 100 100 \
    --d_hid_widths 100 100 \
    --hid_nonlin="tanh" \
    --num_trajs=20 \
    --sample_or_mode \
    --no-render \
    --model_ckpt_dir=$2
