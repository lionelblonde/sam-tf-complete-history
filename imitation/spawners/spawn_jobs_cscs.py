import argparse
import os.path as osp
import numpy as np
from subprocess import call
from copy import copy

from imitation.common.misc_util import flatten_lists, zipsame


parser = argparse.ArgumentParser(description='SAM Job Orchestrator + HP Search')
parser.add_argument('--script_dir', type=str, default="scripts/cscs",
                    help="where to store the generated scripts")
parser.add_argument('--task', type=str, choices=['ppo', 'gail', 'sam'], default='ppo',
                    help="whether to train an expert with PPO or an imitator with GAIL or SAM")
parser.add_argument('--benchmark', type=str, choices=['atari', 'mujoco'], default='mujoco',
                    help="benchmark on which to run experiments")
parser.add_argument('--num_trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
parser.add_argument('--demos_dir', type=str, default=None,
                    help="location of the expert demos arxivs")
parser.add_argument('--num_workers', type=int, default=32,
                    help="number of parallel mpi workers to use for each job")
parser.add_argument('--call', type=int, choices=[0, 1], default=0,
                    help="whether to directly launch the jobs once created")
args = parser.parse_args()

MUJOCO_ENVS_SET = ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2']
ATARI_ENVS_SET = ['FrostbiteNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SeaquestNoFrameskip-v4']
MUJOCO_EXPERT_DEMOS = ['ph', 'ph', 'ph']  # TODO fill
ATARI_EXPERT_DEMOS = ['ph', 'ph', 'ph']  # TODO fill
# Note 1: the orders must correspond, otherwise `zipsame` will return an error
# Note 2: `zipsame` returns a single-use iterator, that's why we don't define the pairs here


def dup_hps_for_env(hpmap, env):
    """Return a separate copy of the HP map after adding extra key-value pairs"""
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    return hpmap_


def dup_hps_for_env_w_demos(hpmap, env, demos):
    """Return a separate copy of the HP map after adding extra key-value pairs"""
    hpmap_ = copy(hpmap)
    hpmap_.update({'env_id': env})
    hpmap_.update({'expert_path': demos})
    return hpmap_

def dup_hps_for_seed(hpmap, seed):



def rand_tuple_from_list(list_):
    """Return a random tuple from a list of tuples
    (Function created because `np.random.choice` does not work on lists of tuples)
    """
    assert all(isinstance(v, tuple) for v in list_), "not a list of tuples"
    return list_[np.random.randint(low=0, high=len(list_))]


def get_rand_hyperparameters(args):
    """Return a map of hyperparameters
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
                     'checkpoint_dir': "data/expert_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "train_xpo_expert",
                     'algo': "ppo",
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'nums_filters': (8, 16),
                     'filter_shapes': (8, 4),
                     'stride_shapes': (4, 2),
                     'hid_widths': (128,),
                     'hid_nonlin': "leaky_relu",
                     'hid_w_init': "he_normal",
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': "constant"}
            return [dup_hps_for_env(hpmap, env)
                    for env in ATARI_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': "data/imitation_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "imitate_via_gail",
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 64,
                     'sample_or_mode': 1,
                     'num_demos': 16,
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
            return [dup_hps_for_env_w_demos(hpmap, env, demos)
                    for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 1,
                     'seed': 0,
                     'checkpoint_dir': "data/imitation_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "imitate_via_sam",
                     'rmsify_obs': 0,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': np.random.choice([2, 4, 8, 16, 32]),
                     'eval_steps_per_iter': 10,
                     'render': 0,
                     'timesteps_per_batch': np.random.choice([2, 4, 8, 16, 32]),
                     'batch_size': np.random.choice([16, 32, 64]),
                     'num_demos': 16,
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
                     'tau': np.random.choice([0.001, 0.01]),
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
                     'n': np.random.choice([10, 20, 40, 60])}
            return [dup_hps_for_env_w_demos(hpmap, env, demos)
                    for env, demos in zipsame(ATARI_ENVS_SET, ATARI_EXPERT_DEMOS)]
    # MuJoCo
    elif args.benchmark == 'mujoco':
        # Expert
        if args.task == 'ppo':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': "data/expert_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "train_xpo_expert",
                     'algo': "ppo",
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 2048,
                     'batch_size': 64,
                     'optim_epochs_per_iter': 10,
                     'sample_or_mode': 1,
                     'hid_widths': (64, 64),
                     'hid_nonlin': "leaky_relu",
                     'hid_w_init': "he_normal",
                     'gaussian_fixed_var': 1,
                     'with_layernorm': 0,
                     'ent_reg_scale': 0.,
                     'clipping_eps': 0.2,
                     'lr': 3e-4,
                     'gamma': 0.99,
                     'gae_lambda': 0.98,
                     'schedule': "constant"}
            return [dup_hps_for_env(hpmap, env)
                    for env in MUJOCO_ENVS_SET]
        # Imitator
        elif args.task == 'gail':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': "data/imitation_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "imitate_via_gail",
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'timesteps_per_batch': 1024,
                     'batch_size': 64,
                     'sample_or_mode': 1,
                     'num_demos': 16,
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
            return [dup_hps_for_env_w_demos(hpmap, env, demos)
                    for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
        elif args.task == 'sam':
            hpmap = {'from_raw_pixels': 0,
                     'seed': 0,
                     'checkpoint_dir': "data/imitation_checkpoints",
                     'summary_dir': "data/summaries",
                     'log_dir': "data/logs",
                     'task': "imitate_via_sam",
                     'rmsify_obs': 1,
                     'save_frequency': 100,
                     'num_timesteps': int(1e7),
                     'training_steps_per_iter': np.random.choice([2, 4, 8, 16, 32]),
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
                     'tau': np.random.choice([0.001, 0.01]),
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
                     'noise_type': np.random.choice(['adaptive-param_0.2',
                                                     'adaptive-param_0.2, ou_0.2',
                                                     'adaptive-param_0.2, ou_0.2, normal_0.2']),
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
                     'n': np.random.choice([10, 20, 40, 60])}
            return [dup_hps_for_env_w_demos(hpmap, env, demos)
                    for env, demos in zipsame(MUJOCO_ENVS_SET, MUJOCO_EXPERT_DEMOS)]
    else:
        raise RuntimeError("unknown benchmark, check what's available in 'spawn_jobs_cscs.py'")


def format_job_str(job_map, run_str):
    """Build the batch script that launches a job"""
    bash_script_str = ('#!/usr/bin/env bash\n'
                       '#SBATCH --job-name={}\n'
                       '#SBATCH --ntasks={}\n'
                       '#SBATCH --cpus-per-task=1\n'
                       '#SBATCH --time={}\n'
                       '#SBATCH --mem=20000\n'
                       '#SBATCH --constraint=gpu\n\n'
                       'module load daint-gpu\n\n'
                       'cd ../..\n\n'
                       'mpirun {}')
    return bash_script_str.format(job_map['job-name'],
                                  job_map['ntasks'],
                                  job_map['time'],
                                  run_str)


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


def format_exp_str(hpmap, task):
    """Build the experiment name"""
    hpmap_str = unroll_options(hpmap)
    # Parse task name
    if task == 'ppo':
        script = "run_xpo_expert"
    elif task == 'gail':
        script = "run_gail"
    elif task == 'sam':
        script = "run_sam"
    return """python -m imitation.imitation_algorithms.{}{}""".format(script, hpmap_str)


def get_job_map(idx, env, task, num_workers):
    return {'ntasks': num_workers,
            'time': '23:59:59',
            'job-name': "hpsearch{}_{}_{}".format(idx, task, env.split('-')[0])}


def run(args):
    """Spawn jobs"""
    # Grab some random hps
    hpmaps = [get_rand_hyperparameters(args) for _ in range(args.num_trials)]
    # Flatten into a 1-dim list
    hpmaps = flatten_lists(hpmaps)
    # Create associated task strings
    exp_strs = [format_exp_str(hpmap, args.task) for hpmap in hpmaps]
    if not len(exp_strs) == len(set(exp_strs)):
        # Terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> Try again :)")
    # Create the job maps
    job_maps = [get_job_map(i, hpmap['env_id'], args.task, args.num_workers)
                for i, hpmap in enumerate(hpmaps)]
    # Finally get all the required job strings
    job_strs = [format_job_str(jm, es) for jm, es in zipsame(job_maps, exp_strs)]
    # Spawn the jobs
    for i, js in enumerate(set(job_strs)):
        print('-' * 10 + "> job #{} launcher content:".format(i))
        print(js + "\n")
        job_name = "hp_search_{}.sh".format(i)
        with open(osp.join(args.script_dir, job_name), 'w') as f:
            f.write(js)
        if args.call:
            # Spawn the job!
            call(["sbatch", "./{}".format(job_name)])
    # Summarize the number of jobs spawned
    print("total num job (successfully) spawned: {}".format(len(job_strs)))


if __name__ == "__main__":
    # Prepend the full path to the demos arxis
    MUJOCO_EXPERT_DEMOS = [osp.join(args.demos_dir, arxiv) for arxiv in MUJOCO_EXPERT_DEMOS]
    ATARI_EXPERT_DEMOS = [osp.join(args.demos_dir, arxiv) for arxiv in ATARI_EXPERT_DEMOS]
    # Create (and optionally launch) the jobs!
    run(args)
