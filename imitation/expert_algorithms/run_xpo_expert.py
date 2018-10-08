import os.path as osp

import gym.spaces  # noqa

from imitation.common import tf_util as U
from imitation.common.argparsers import xpo_expert_argparser
from imitation.common.experiment_initializer import ExperimentInitializer
from imitation.common.env_makers import make_env
from imitation.expert_algorithms.xpo_agent import XPOAgent
from imitation.expert_algorithms import ppo, trpo, xpo_util


def train_xpo_expert(args):
    """Train a XPO expert policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, comm=comm)
    experiment.configure()

    # Create environment
    rank = comm.Get_rank()
    worker_seed = args.seed + 10000 * rank
    name = "{}.worker_{}".format(args.task, rank)
    env = make_env(args)(args.env_id, worker_seed, name, args.horizon)

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Create experiment name
    experiment_name = experiment.get_long_name()

    # Train XPO expert policy
    if args.algo == 'ppo':
        ppo.learn(comm=comm,
                  env=env,
                  xpo_agent_wrapper=xpo_agent_wrapper,
                  sample_or_mode=args.sample_or_mode,
                  gamma=args.gamma,
                  save_frequency=args.save_frequency,
                  ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                  summary_dir=osp.join(args.summary_dir, experiment_name),
                  timesteps_per_batch=args.timesteps_per_batch,
                  batch_size=args.batch_size,
                  optim_epochs_per_iter=args.optim_epochs_per_iter,
                  lr=args.lr,
                  experiment_name=experiment_name,
                  ent_reg_scale=args.ent_reg_scale,
                  clipping_eps=args.clipping_eps,
                  gae_lambda=args.gae_lambda,
                  schedule=args.schedule,
                  max_timesteps=args.num_timesteps)
    elif args.algo == 'trpo':
        trpo.learn(comm=comm,
                   env=env,
                   xpo_agent_wrapper=xpo_agent_wrapper,
                   experiment_name=experiment_name,
                   ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
                   summary_dir=osp.join(args.summary_dir, experiment_name),
                   sample_or_mode=args.sample_or_mode,
                   gamma=args.gamma,
                   max_kl=args.max_kl,
                   save_frequency=args.save_frequency,
                   timesteps_per_batch=args.timesteps_per_batch,
                   batch_size=args.batch_size,
                   ent_reg_scale=args.ent_reg_scale,
                   gae_lambda=args.gae_lambda,
                   cg_iters=args.cg_iters,
                   cg_damping=args.cg_damping,
                   vf_iters=args.vf_iters,
                   vf_lr=args.vf_lr,
                   max_timesteps=args.num_timesteps)
    else:
        raise RuntimeError("unknown algorithm")

    # Close environment
    env.close()


def evaluate_xpo_expert(args):
    """Evaluate a trained XPO expert policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure()

    # Create environment
    env = make_env(args)(args.env_id, args.seed, args.task, args.horizon)

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Evaluate trained XPO expert policy
    xpo_util.evaluate(env=env,
                      xpo_agent_wrapper=xpo_agent_wrapper,
                      num_trajs=args.num_trajs,
                      sample_or_mode=args.sample_or_mode,
                      render=args.render,
                      exact_model_path=args.exact_model_path,
                      model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


def gather_xpo_expert(args):
    """Gather trajectories from a trained XPO expert policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure()

    # Create environment
    env = make_env(args)(args.env_id, args.seed, args.task, args.horizon)

    def xpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=args)

    # Prepare trajectories destination
    expert_arxiv_name = experiment.get_expert_arxiv_name()

    # Gather trajectories from a trained XPO expert policy
    xpo_util.gather_trajectories(env=env,
                                 xpo_agent_wrapper=xpo_agent_wrapper,
                                 num_trajs=args.num_trajs,
                                 sample_or_mode=args.sample_or_mode,
                                 render=args.render,
                                 exact_model_path=args.exact_model_path,
                                 model_ckpt_dir=args.model_ckpt_dir,
                                 demos_dir=args.demos_dir,
                                 expert_arxiv_name=expert_arxiv_name)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = xpo_expert_argparser().parse_args()
    if _args.task == 'train_xpo_expert':
        train_xpo_expert(_args)
    elif _args.task == 'evaluate_xpo_expert':
        evaluate_xpo_expert(_args)
    elif _args.task == 'gather_xpo_expert':
        gather_xpo_expert(_args)
    else:
        raise NotImplementedError
