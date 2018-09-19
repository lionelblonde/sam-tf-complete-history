#!/usr/bin/env bash
# Example: ./atari_evaluate_xpo_expert.sh <env_id> <xpo_pol_ckpt_dir_path>

cd ../..

python -m imitation.expert_algorithms.run_xpo_expert \
    --note="" \
    --env_id="BreakoutNoFrameskip-v4" \
    --from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="evaluate_xpo_expert" \
    --nums_filters 8 16 \
    --filter_shapes 8 4 \
    --stride_shapes 4 2 \
    --hid_widths 128 \
    --hid_nonlin="leaky_relu" \
    --num_trajs=20 \
    --no-sample_or_mode \
    --no-render \
    --model_ckpt_dir=$2
