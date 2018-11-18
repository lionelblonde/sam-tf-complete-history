#!/usr/bin/env bash
# Example: ./atari_gail_imitate.sh <num_mpi_workers> <env_id> <expert_demos_path> <num_demos>

cd ../..

opts="-np $1"
unamestr="$(uname)"
if [[ ! "$unamestr" == "Darwin" ]]; then
    opts="$opts --bind-to core"
    nlogicores="$(nproc)"
    echo "non-Darwin platform: binding processes to cores"
else
    nlogicores="$(sysctl -n hw.ncpu)"
    echo "Darwin platform: not binding processes to cores"
fi
echo "$1 process(es), $nlogicores total logical cores"
echo "mpi options: $opts"

mpirun $opts python -m imitation.imitation_algorithms.run_gail \
    --note="" \
    --env_id=$2 \
    --from_raw_pixels \
    --seed=0 \
    --checkpoint_dir="data/imitation_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="imitate_via_gail" \
    --expert_path=$3 \
    --no-rmsify_obs \
    --save_frequency=100 \
    --num_timesteps=1000000000 \
    --timesteps_per_batch=1024 \
    --batch_size=128 \
    --sample_or_mode \
    --num_demos=$4 \
    --g_steps=3 \
    --d_steps=1 \
    --no-non_satur_grad \
    --pol_nums_filters 8 16 \
    --pol_filter_shapes 8 4 \
    --pol_stride_shapes 4 2 \
    --pol_hid_widths 128 \
    --d_nums_filters 8 16 \
    --d_filter_shapes 8 4 \
    --d_stride_shapes 4 2 \
    --d_hid_widths 128 \
    --hid_nonlin="tanh" \
    --hid_w_init="xavier_normal" \
    --gaussian_fixed_var \
    --no-with_layernorm \
    --max_kl=0.01 \
    --pol_ent_reg_scale=0. \
    --d_ent_reg_scale=0. \
    --label_smoothing \
    --one_sided_label_smoothing \
    --vf_lr=3e-4 \
    --d_lr=3e-4 \
    --gamma=0.995 \
    --gae_lambda=0.99
