#!/usr/bin/env bash
#SBATCH --job-name=hpsearch4_sam_HalfCheetah
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:59
#SBATCH --mem=20000
#SBATCH --constraint=gpu

module load daint-gpu

cd ../..

mpirun python -m imitation.imitation_algorithms.run_sam --no-from_raw_pixels --seed=0 --checkpoint_dir=data/imitation_checkpoints --summary_dir=data/summaries --log_dir=data/logs --task=imitate_via_sam --rmsify_obs --save_frequency=100 --num_timesteps=10000000 --training_steps_per_iter=8 --eval_steps_per_iter=10 --no-render --timesteps_per_batch=4 --batch_size=16 --num_demos=16 --g_steps=3 --d_steps=1 --no-non_satur_grad --actorcritic_hid_widths 64 64 --d_hid_widths 64 64 --hid_nonlin=leaky_relu --hid_w_init=he_normal --tau=0.01 --with_layernorm --ac_branch_in=2 --d_ent_reg_scale=0.0 --label_smoothing --one_sided_label_smoothing --reward_scale=10.0 --rmsify_rets --no-enable_popart --actor_lr=0.0001 --critic_lr=0.001 --d_lr=0.0003 --clip_norm=5.0 --noise_type=adaptive-param_0.2, ou_0.2, normal_0.2 --param_noise_adaption_frequency=40 --gamma=0.98 --mem_size=100000 --no-prioritized_replay --alpha=0.3 --beta=1.0 --no-ranked --no-add_demos_to_mem --no-unreal --q_loss_scale=1.0 --td_loss_1_scale=1.0 --td_loss_n_scale=1.0 --wd_scale=0.01 --n_step_returns --n=20 --env_id=HalfCheetah-v2 --expert_path=ph