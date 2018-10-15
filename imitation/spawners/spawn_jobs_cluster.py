"""Example launch
    python -m imitation.spawners.spawn_jobs_cluster \
        --script_dir=tmpscripts \
        --task=sam \
        --benchmark=mujoco \
        --cluster=cscs \
        --device=gpu \
        --demos_dir=/code/sam-tf/DEMOS \
        --no-mpi \
        --num_workers=4 \
        --partition=shared-gpu \
        --time=12:00:00 \
        --max_seed=5 \
        --docker \
        --no-call \
        --no-rand

partitions for cscs: debug, normal
partitions for baobab: debug, shared, shared-gpu, kalousis-gpu
"""

import argparse
import os.path as osp
import numpy as np
from subprocess import call
from copy import copy

from imitation.common.misc_util import flatten_lists, zipsame, boolean_flag
from imitation.common.experiment_initializer import rand_id


parser = argparse.ArgumentParser(description='SAM Job Orchestrator + HP Search')
parser.add_argument('--task', type=str, choices=['ppo', 'gail', 'sam'], default='ppo',
                    help="whether to train an expert with PPO or an imitator with GAIL or SAM")
parser.add_argument('--benchmark', type=str, choices=['atari', 'mujoco'], default='mujoco',
                    help="benchmark on which to run experiments")
parser.add_argument('--cluster', type=str, choices=['baobab', 'cscs', 'gengar'], default='gengar',
                    help="cluster on which the experiments will be launched")
parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                    help="type de processing unit to use")
parser.add_argument('--num_rand_trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
boolean_flag(parser, 'mpi', default=False, help="whether to use mpi")
parser.add_argument('--num_workers', type=int, default=1,
                    help="number of parallel mpi workers (actors) to use for each job")
parser.add_argument('--partition', type=str, default=None, help="partition to launch jobs on")
parser.add_argument('--time', type=str, default=None, help="duration of the jobs")
parser.add_argument('--max_seed', type=int, default=5,
                    help="amount of seeds across which jobs are replicated ('range(max_seed)'')")
boolean_flag(parser, 'docker', default=True, help="whether to run through docker containers")
boolean_flag(parser, 'call', default=False, help="whether to launch the jobs once created")
boolean_flag(parser, 'rand', default=False, help="whether to perform hyperparameter search")
args = parser.parse_args()

NUM_DEMOS_SET = [4, 16, 32]
MUJOCO_ENVS_SET = ['InvertedPendulum-v2',
                   'InvertedDoublePendulum-v2',
                   'Reacher-v2',
                   'Hopper-v2',
                   'HalfCheetah-v2',
                   'Walker2d-v2',
                   'Ant-v2']
ATARI_ENVS_SET = ['BreakoutNoFrameskip-v4']
MUJOCO_EXPERT_DEMOS = ['InvertedPendulum-v2_s0_mode_d32.npz',
                       'InvertedDoublePendulum-v2_s0_mode_d32.npz',
                       'Reacher-v2_s0_mode_d32.npz',
                       'Hopper-v2_s0_mode_d32.npz',
                       'HalfCheetah-v2_s0_mode_d32.npz',
                       'Walker2d-v2_s0_mode_d32.npz',
                       'Ant-v2_s0_mode_d32.npz']
ATARI_EXPERT_DEMOS = ['BreakoutNoFrameskip-v4_s0_mode_d32.npz']
# Note 1: the orders must correspond, otherwise `zipsame` will return an error
# Note 2: `zipsame` returns a single-use iterator, that's why we don't define the pairs here


def fmt_path(args, meta, dir_):
    """Transform as relative path into an absolute path"""
    relative_path = osp.join("data/{}".format(meta), dir_)
    if args.cluster == 'cscs' and args.docker:
        return osp.join("/code/sam-tf", relative_path)
    else:
        return relative_path


def dup_hps_for_env(hpmap, env):
    """Return a separate copy of the HP map after adding extra key-value pair
    for the key 'env_id'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    return hpmap_


def dup_hps_for_env_w_demos(args, hpmap, env, demos):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the keys 'env_id' and 'expert_path'
    """
    demos = osp.join("DEMOS", demos)
    if args.cluster == 'cscs' and args.docker:
        # Prepend full container path to the demos arxis
        demos = osp.join("/code/sam-tf", demos)
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    hpmap_.update({'expert_path': demos})
    return hpmap_


def dup_hps_for_seed(hpmap, seed):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the key 'seed'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'seed': seed})
    return hpmap_


def dup_hps_for_num_demos(hpmap, num_demos):
    """Return a separate copy of the HP map after adding extra key-value pairs
    for the key 'num_demos'
    """
    hpmap_ = copy(hpmap)
    hpmap_.update({'num_demos': num_demos})
    return hpmap_


def rand_tuple_from_list(list_):
    """Return a random tuple from a list of tuples
    (Function created because `np.random.choice` does not work on lists of tuples)
    """
    assert all(isinstance(v, tuple) for v in list_), "not a list of tuples"
    return list_[np.random.randint(low=0, high=len(list_))]


def get_rand_hps(args, meta):
    """Return a list of maps of hyperparameters selected by random search
    Example of hyperparameter dictionary:
        {'hid_widths': rand_tuple_from_list([(64, 64)]),  # list of tuples
         'hid_nonlin': np.random.choice(['relu', 'leaky_relu']),
         'hid_w_init': np.random.choice(['he_normal', 'he_uniform']),
         'tau': np.random.choice([0.001, 0.01]),
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    # Atari
    if args.benchmark == 'atari':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 0,
                     'save_frequency': 10,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'nums_filters': (8, 16),
                     'filter_shapes': (8, 4),
                     'stride_shapes': (4, 2),
                     'hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            return [dup_hps_for_env(hpmap, env)
                    for env in ATARI_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_nums_filters': (8, 16),
                     'pol_filter_shapes': (8, 4),
                     'pol_stride_shapes': (4, 2),
                     'pol_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            return [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                    for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': np.random.choice([10, 20, 40]),
                     'eval_steps_per_iter': 10,
                     'render': 0,
                     'timesteps_per_batch': np.random.choice([2, 4, 8, 16, 32]),
                     'batch_size': np.random.choice([32, 64, 128, 256]),
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_nums_filters': (8, 16),
                     'actorcritic_filter_shapes': (8, 4),
                     'actorcritic_stride_shapes': (4, 2),
                     'actorcritic_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'tau': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': np.random.choice([0, 1]),
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': np.random.choice(['adaptive-param_0.1', 'adaptive-param_0.2']),
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': np.random.choice([0.98, 0.99, 0.995]),
                     'mem_size': int(1e5),
                     'prioritized_replay': np.random.choice([0, 1]),
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'q_loss_scale': 1.,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': np.random.choice([0.01, 0.001]),
                     'n_step_returns': 1,
                     'n': np.random.choice([36, 72, 96])}
            return [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                    for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
    # MuJoCo
    elif args.benchmark == 'mujoco':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 1,
                     'save_frequency': 10,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            return [dup_hps_for_env(hpmap, env)
                    for env in MUJOCO_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_hid_widths': (100, 100),
                     'd_hid_widths': (100, 100),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            return [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                    for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': np.random.choice([10, 20, 40]),
                     'eval_steps_per_iter': 10,
                     'render': 0,
                     'timesteps_per_batch': np.random.choice([2, 4, 8, 16, 32]),
                     'batch_size': np.random.choice([16, 32, 64]),
                     'num_demos': 16,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_hid_widths': (64, 64),
                     'd_hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'tau': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 2,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': np.random.choice([0, 1]),
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': np.random.choice(['"adaptive-param_0.2"',
                                                     '"adaptive-param_0.2, ou_0.2"']),
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': np.random.choice([0.98, 0.99, 0.995]),
                     'mem_size': int(1e5),
                     'prioritized_replay': np.random.choice([0, 1]),
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'q_loss_scale': 1.,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': np.random.choice([0.01, 0.001]),
                     'n_step_returns': 1,
                     'n': np.random.choice([36, 72, 96])}
            return [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                    for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
    else:
        raise RuntimeError("unknown benchmark, check what's available in 'spawn_jobs_cscs.py'")


def get_spectrum_hps(args, meta, max_seed):
    """Return a list of maps of hyperparameters selected deterministically
    and spanning the specified range of seeds
    Example of hyperparameter dictionary:
        {'hid_widths': rand_tuple_from_list([(64, 64)]),  # list of tuples
         'hid_nonlin': np.random.choice(['relu', 'leaky_relu']),
         'hid_w_init': np.random.choice(['he_normal', 'he_uniform']),
         'tau': np.random.choice([0.001, 0.01]),
         'with_layernorm': 1,
         'ent_reg_scale': 0.}
    """
    # Atari
    if args.benchmark == 'atari':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 0,
                     'save_frequency': 10,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'nums_filters': (8, 16),
                     'filter_shapes': (8, 4),
                     'stride_shapes': (4, 2),
                     'hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env) for env in ATARI_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_nums_filters': (8, 16),
                     'pol_filter_shapes': (8, 4),
                     'pol_stride_shapes': (4, 2),
                     'pol_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 1,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': 40,
                     'eval_steps_per_iter': 10,
                     'render': 0,
                     'timesteps_per_batch': 4,
                     'batch_size': 256,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_nums_filters': (8, 16),
                     'actorcritic_filter_shapes': (8, 4),
                     'actorcritic_stride_shapes': (4, 2),
                     'actorcritic_hid_widths': (128,),
                     'd_nums_filters': (8, 16),
                     'd_filter_shapes': (8, 4),
                     'd_stride_shapes': (4, 2),
                     'd_hid_widths': (128,),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'tau': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 1,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': 1,
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': 'adaptive-param_0.4',
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': 0.99,
                     'mem_size': int(1e5),
                     'prioritized_replay': 0,
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'q_loss_scale': 1.,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': 0.001,
                     'n_step_returns': 1,
                     'n': 96}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
    # MuJoCo
    elif args.benchmark == 'mujoco':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'expert_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'train_xpo_expert',
                     'algo': 'ppo',
                     'rmsify_obs': 1,
                     'save_frequency': 10,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': 'constant'}
            hpmaps = [dup_hps_for_env(hpmap, env) for env in MUJOCO_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_gail',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 128,
                     'sample_or_mode': 1,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'pol_hid_widths': (100, 100),
                     'd_hid_widths': (100, 100),
                     'hid_nonlin': 'tanh',
                     'hid_w_init': 'xavier_normal',
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'max_kl': 0.01,
                     'pol_ent_reg_scale': 0.,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'vf_lr': 3e-4,
                     'd_lr': 3e-4,
                     'gamma': 0.995,
                     'gae_lambda': 0.99}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 0,
                     'checkpoint_dir': fmt_path(args, meta, 'imitation_checkpoints'),
                     'summary_dir': fmt_path(args, meta, 'summaries'),
                     'log_dir': fmt_path(args, meta, 'logs'),
                     'task': 'imitate_via_sam',
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': 40,
                     'eval_steps_per_iter': 10,
                     'render': 0,
                     'timesteps_per_batch': 4,
                     'batch_size': 256,
                     'g_steps': 3,
                     'd_steps': 1,
                     'non_satur_grad': 0,
                     'actorcritic_hid_widths': (64, 64),
                     'd_hid_widths': (64, 64),
                     'hid_nonlin': 'leaky_relu',
                     'hid_w_init': 'he_normal',
                     'tau': 0.01,
                     'with_layernorm': 1,
                     'ac_branch_in': 2,
                     'd_ent_reg_scale': 0.,
                     'label_smoothing': 1,
                     'one_sided_label_smoothing': 1,
                     'reward_scale': 1.,
                     'rmsify_rets': 1,
                     'enable_popart': 1,
                     'actor_lr': 1e-4,
                     'critic_lr': 1e-3,
                     'd_lr': 3e-4,
                     'clip_norm': 5.,
                     'noise_type': '"adaptive-param_0.2, ou_0.2"',
                     'rew_aug_coeff': 0.,
                     'param_noise_adaption_frequency': 40,
                     'gamma': 0.99,
                     'mem_size': int(1e5),
                     'prioritized_replay': 0,
                     'alpha': 0.3,
                     'beta': 1.,
                     'ranked': 0,
                     'add_demos_to_mem': 0,
                     'unreal': 0,
                     'q_loss_scale': 1.,
                     'td_loss_1_scale': 1.,
                     'td_loss_n_scale': 1.,
                     'wd_scale': 0.001,
                     'n_step_returns': 1,
                     'n': 96}
            hpmaps = [dup_hps_for_env_w_demos(args, hpmap, env, demos)
                      for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
    else:
        raise RuntimeError("unknown benchmark, check what's available in 'spawn_jobs_cscs.py'")

    if args.task in ['gail', 'sam']:
        # Duplicate every hyperparameter map of the list to span the range of seeds
        output = [dup_hps_for_seed(hpmap, seed)
                  for seed in range(max_seed)
                  for hpmap in hpmaps]
        # Duplicate every hyperparameter map of the list to span the range of num of demos
        output = [dup_hps_for_num_demos(hpmap, num_demos)
                  for num_demos in NUM_DEMOS_SET
                  for hpmap in output]
    return output


def unroll_options(hpmap):
    """Transform the dictionary of hyperparameters into a string of bash options"""
    base_str = ""
    no_value_keys = ['from_raw_pixels',
                     'rmsify_obs',
                     'sample_or_mode',
                     'gaussian_fixed_var',
                     'render',
                     'non_satur_grad',
                     'with_layernorm',
                     'rmsify_rets',
                     'enable_popart',
                     'label_smoothing',
                     'one_sided_label_smoothing',
                     'prioritized_replay',
                     'ranked',
                     'add_demos_to_mem',
                     'unreal',
                     'n_step_returns']
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            base_str += " --no-{}".format(k)
            continue
        elif k in no_value_keys:
            base_str += " --{}".format(k)
            continue
        if isinstance(v, tuple):
            base_str += " --{} ".format(k) + " ".join(str(v_) for v_ in v)
            continue
        base_str += " --{}={}".format(k, v)
    return base_str


def format_job_str(args, job_map, run_str):
    """Build the batch script that launches a job"""
    message = "MuJoCo bugged for Singularity/Shifter (looks for bins + license in host's ~) "
    message += "-> https://github.com/openai/mujoco-py/issues/295"
    assert not (args.benchmark == 'mujoco' and args.docker), message
    message = "MPICH install in container, necessary for inter-container comm "
    message += "in Singularity/Shifter, breaks mpi4py in the python code. "
    message += "PITA to solve, for little gains: inter-container MPI comms not supported. "
    assert not (args.mpi and args.docker), message
    if args.cluster == 'baobab':
        assert args.docker, "Baobab's everything is old. Docker usage forced."
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --mem=32000\n')
        if args.device == 'gpu':
            contraint = "COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
            bash_script_str += ('#SBATCH --gres=gpu:1\n'
                                '#SBATCH --constraint="{}"\n'.format(contraint))
        bash_script_str += ('\n')
        # Load modules
        bash_script_str += ('module load GCC/6.3.0-2.27\n'
                            'module load Singularity/2.4.2\n')
        if args.device == 'gpu':
            bash_script_str += ('module load CUDA\n')
        bash_script_str += ('\n')
        # Launch command
        if args.mpi:
            bash_script_str += ('mpirun ')
        else:
            bash_script_str += ('srun ')
        bash_script_str += ('singularity --debug exec ')
        if args.device == 'gpu':
            bash_script_str += ('--nv ')
            bash_script_str += ('/home/blonde0/docker_images/docker-sam-tf-gpu ')
        elif args.device == 'cpu':
            bash_script_str += ('/home/blonde0/docker_images/docker-sam-tf-cpu ')
        bash_script_str += ('{}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)

    elif args.cluster == 'cscs':
        # Set sbatch config
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('#SBATCH --job-name={}\n'
                            '#SBATCH --partition={}\n'
                            '#SBATCH --ntasks={}\n'
                            '#SBATCH --cpus-per-task=1\n'
                            '#SBATCH --time={}\n'
                            '#SBATCH --constraint=gpu\n\n')
        # Load modules
        bash_script_str += ('module load daint-gpu\n')
        if args.docker:
            bash_script_str += ('module load shifter-ng\n')
        bash_script_str += ('\n')
        # Launch command
        if args.docker:
            bash_script_str += ('srun shifter --debug ')
            if args.mpi:

                bash_script_str += ('--mpi ')
            bash_script_str += ('run ')
            bash_script_str += ('--mount='
                                'type=bind,'
                                'source=/users/lblonde/Code/seil-tf/imitation,'
                                'destination=/code/sam-tf/imitation ')
            bash_script_str += ('--mount='
                                'type=bind,'
                                'source=/users/lblonde/Code/seil-tf/launchers,'
                                'destination=/code/sam-tf/launchers ')
            bash_script_str += ('--mount='
                                'type=bind,'
                                'source=/users/lblonde/Code/seil-tf/data,'
                                'destination=/code/sam-tf/data ')
            bash_script_str += ('--mount='
                                'type=bind,'
                                'source=/users/lblonde/Code/seil-tf/DEMOS,'
                                'destination=/code/sam-tf/DEMOS ')
            if args.device == 'gpu':
                bash_script_str += ('lionelblonde/docker-sam-tf-gpu:latest ')
            elif args.device == 'cpu':
                bash_script_str += ('lionelblonde/docker-sam-tf-cpu:latest ')
            bash_script_str += ('bash -c "{}"')
        else:
            assert args.benchmark == 'mujoco', "Can only go the non-docker way with MuJoCo."
            assert args.device == 'cpu', "Continuous control always faster on CPU. \
                                          GPU usage prevented."
            if args.mpi:
                bash_script_str += ('mpirun {}')
            else:
                bash_script_str += ('srun {}')

        return bash_script_str.format(job_map['job-name'],
                                      job_map['partition'],
                                      job_map['ntasks'],
                                      job_map['time'],
                                      run_str)

    elif args.cluster == 'gengar':
        # Set header
        bash_script_str = ('#!/usr/bin/env bash\n\n')
        bash_script_str += ('# job name: {}\n\n')
        # Launch command
        bash_script_str += ('mpirun -np {} {}')
        return bash_script_str.format(job_map['job-name'],
                                      job_map['ntasks'],
                                      run_str)


def format_exp_str(args, hpmap):
    """Build the experiment name"""
    hpmap_str = unroll_options(hpmap)
    # Parse task name
    if args.task == 'ppo':
        script = "imitation.expert_algorithms.run_xpo_expert"
    elif args.task == 'gail':
        script = "imitation.imitation_algorithms.run_gail"
    elif args.task == 'sam':
        script = "imitation.imitation_algorithms.run_sam"

    pre = "cd /code/sam-tf && " if args.cluster == 'cscs' and args.docker else ''

    return "{}python -m {}{}".format(pre, script, hpmap_str)


def get_job_map(args, meta, i, env, seed, num_demos, type_exp):
    if not args.mpi:
        # Override the number of parallel workers to 1 when mpi is off
        args.num_workers = 1
    return {'ntasks': args.num_workers,
            'partition': args.partition,
            'time': args.time,
            'job-name': "{}_{}{}_{}_{}_s{}_d{}".format(meta,
                                                       type_exp,
                                                       i,
                                                       args.task,
                                                       env.split('-')[0],
                                                       seed,
                                                       num_demos)}


def run(args):
    """Spawn jobs"""
    # Create meta-experiment identifier
    meta = rand_id()
    # Define experiment type
    if args.rand:
        type_exp = 'hpsearch'
    else:
        type_exp = 'sweep'
    # Get hyperparameter configurations
    if args.rand:
        # Get a number of random hyperparameter configurations
        hpmaps = [get_rand_hps(args, meta) for _ in range(args.num_rand_trials)]
        # Flatten into a 1-dim list
        hpmaps = flatten_lists(hpmaps)
    else:
        # Get the deterministic spectrum of specified hyperparameters
        hpmaps = get_spectrum_hps(args, meta, args.max_seed)
    # Create associated task strings
    exp_strs = [format_exp_str(args, hpmap) for hpmap in hpmaps]
    if not len(exp_strs) == len(set(exp_strs)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again :)")
    # Create the job maps
    job_maps = [get_job_map(args,
                            meta,
                            i,
                            hpmap['env_id'],
                            hpmap['seed'],
                            '0' if args.task == 'ppo' else hpmap['num_demos'],
                            type_exp)
                for i, hpmap in enumerate(hpmaps)]
    # Finally get all the required job strings
    job_strs = [format_job_str(args, jm, es) for jm, es in zipsame(job_maps, exp_strs)]
    # Spawn the jobs
    for i, (jm, js) in enumerate(zipsame(job_maps, job_strs)):
        print('-' * 10 + "> job #{} launcher content:".format(i))
        print(js + "\n")
        job_name = "{}.sh".format(jm['job-name'])
        with open(job_name, 'w') as f:
            f.write(js)
        if args.call:
            # Spawn the job!
            call(["sbatch", "./{}".format(job_name)])
    # Summarize the number of jobs spawned
    print("total num job (successfully) spawned: {}".format(len(job_strs)))


if __name__ == "__main__":
    # Create (and optionally launch) the jobs!
    run(args)
