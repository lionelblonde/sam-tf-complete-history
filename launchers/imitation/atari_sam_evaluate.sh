#!/usr/bin/env bash
# Example: ./atari_sam_evaluate.sh <env_id> <sam_pol_ckpt_dir_path>

cd ../..

python -m imitation.imitation_algorithms.run_sam \
    --note="" \
    --env_id=$1 \
    --from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="evaluate_sam_policy" \
    --actorcritic_nums_filters 8 16 \
    --actorcritic_filter_shapes 8 4 \
    --actorcritic_stride_shapes 4 2 \
    --actorcritic_hid_widths 128 \
    --d_nums_filters 8 16 \
    --d_filter_shapes 8 4 \
    --d_stride_shapes 4 2 \
    --d_hid_widths 128 \
    --hid_nonlin="leaky_relu" \
    --noise_type="none" \
    --num_trajs=20 \
    --no-render \
    --model_ckpt_dir=$2
