#!/usr/bin/env bash
# Example: ./atari_sam_imitate.sh <num_mpi_workers> <env_id> <expert_demos_path> <num_demos>

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

mpirun $opts python -m imitation.imitation_algorithms.run_sam \
    --note="" \
    --env_id=$2 \
    --from_raw_pixels \
    --seed=0 \
    --checkpoint_dir="data/imitation_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="imitate_via_sam" \
    --expert_path=$3 \
    --no-rmsify_obs \
    --save_frequency=100 \
    --num_timesteps=10000000 \
    --training_steps_per_iter=25 \
    --eval_steps_per_iter=10 \
    --no-render \
    --timesteps_per_batch=4 \
    --batch_size=64 \
    --num_demos=$4 \
    --g_steps=3 \
    --d_steps=1 \
    --no-non_satur_grad \
    --actorcritic_nums_filters 8 16 \
    --actorcritic_filter_shapes 8 4 \
    --actorcritic_stride_shapes 4 2 \
    --actorcritic_hid_widths 128 \
    --d_nums_filters 8 16 \
    --d_filter_shapes 8 4 \
    --d_stride_shapes 4 2 \
    --d_hid_widths 128 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --tau=0.01 \
    --with_layernorm \
    --ac_branch_in=1 \
    --d_ent_reg_scale=0. \
    --label_smoothing \
    --one_sided_label_smoothing \
    --reward_scale=1. \
    --rmsify_rets \
    --enable_popart \
    --actor_lr=1e-4 \
    --critic_lr=1e-3 \
    --d_lr=3e-4 \
    --clip_norm=5. \
    --noise_type="adaptive-param_0.2" \
    --rew_aug_coeff=0. \
    --param_noise_adaption_frequency=40 \
    --gamma=0.99 \
    --mem_size=100000 \
    --no-prioritized_replay \
    --alpha=0.3 \
    --beta=1. \
    --no-ranked \
    --reset_with_demos \
    --no-add_demos_to_mem \
    --no-unreal \
    --q_loss_scale=1. \
    --td_loss_1_scale=1. \
    --td_loss_n_scale=1. \
    --wd_scale=0.001 \
    --n_step_returns \
    --n=96
