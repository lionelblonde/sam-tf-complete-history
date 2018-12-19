#!/usr/bin/env bash
# Example: ./atari_train_trpo_expert.sh <num_mpi_workers> <env_id>

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

mpirun $opts python -m imitation.expert_algorithms.run_xpo_expert \
    --note="" \
    --env_id=$2 \
    --from_raw_pixels \
    --seed=0 \
    --checkpoint_dir="data/expert_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="train_xpo_expert" \
    --algo="trpo" \
    --no-rmsify_obs \
    --save_frequency=10 \
    --num_timesteps=1000000000 \
    --timesteps_per_batch=1024 \
    --batch_size=64 \
    --sample_or_mode \
    --nums_filters 8 16 \
    --filter_shapes 8 4 \
    --stride_shapes 4 2 \
    --hid_widths 128 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --gaussian_fixed_var \
    --no-with_layernorm \
    --max_kl=0.01 \
    --ent_reg_scale=0. \
    --cg_iters=10 \
    --cg_damping=0.1 \
    --vf_iters=5 \
    --vf_lr=1e-3 \
    --gamma=0.99 \
    --gae_lambda=0.98
