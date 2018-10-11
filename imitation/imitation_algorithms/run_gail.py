import os.path as osp

import gym.spaces  # noqa

from imitation.common import tf_util as U
from imitation.common.argparsers import gail_argparser, disambiguate
from imitation.common.experiment_initializer import ExperimentInitializer
from imitation.common.env_makers import make_env
from imitation.common.misc_util import set_global_seeds
from imitation.expert_algorithms.xpo_agent import XPOAgent
from imitation.imitation_algorithms.discriminator import Discriminator
from imitation.imitation_algorithms import gail
from imitation.imitation_algorithms.demo_dataset import DemoDataset


def imitate_via_gail(args):
    """Train a GAIL imitation policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args, comm=comm)
    experiment.configure_logging()

    # Seedify
    rank = comm.Get_rank()
    worker_seed = args.seed + 1000000 * rank
    set_global_seeds(worker_seed)
    # Create environment
    name = "{}.worker_{}".format(args.task, rank)
    env = make_env(args)(args.env_id, worker_seed, name, args.horizon)

    # Refine hps to avoid ambiguities
    pol_hps, d_hps = disambiguate(kvs=args, tokens=['pol', 'd'])

    def trpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=pol_hps)

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps)

    # Create experiment name
    experiment_name = experiment.get_long_name()

    # Create the expert demonstrations dataset from expert trajectories
    dataset = DemoDataset(expert_arxiv=args.expert_path, size=args.num_demos)

    comm.Barrier()

    # Train GAIL imitation policy
    gail.learn(comm=comm,
               env=env,
               trpo_agent_wrapper=trpo_agent_wrapper,
               discriminator_wrapper=discriminator_wrapper,
               experiment_name=experiment_name,
               ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
               summary_dir=osp.join(args.summary_dir, experiment_name),
               expert_dataset=dataset,
               g_steps=args.g_steps,
               d_steps=args.d_steps,
               sample_or_mode=args.sample_or_mode,
               ent_reg_scale=args.pol_ent_reg_scale,
               gamma=args.gamma,
               gae_lambda=args.gae_lambda,
               max_kl=args.max_kl,
               save_frequency=args.save_frequency,
               timesteps_per_batch=args.timesteps_per_batch,
               batch_size=args.batch_size,
               cg_iters=args.cg_iters,
               cg_damping=args.cg_damping,
               vf_iters=args.vf_iters,
               vf_lr=args.vf_lr,
               d_lr=args.d_lr,
               max_timesteps=args.num_timesteps)

    # Close environment
    env.close()


def evaluate_gail_policy(args):
    """Evaluate a trained GAIL imitation policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()

    # Seedify
    set_global_seeds(args.seed)
    # Create environment
    env = make_env(args)(args.env_id, args.seed, args.task, args.horizon)

    # Refine hps to avoid ambiguities
    pol_hps, d_hps = disambiguate(kvs=args, tokens=['pol', 'd'])

    def trpo_agent_wrapper(name):
        return XPOAgent(name=name, env=env, hps=pol_hps)

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps)

    # Evaluate trained imitation agent
    gail.evaluate(env=env,
                  trpo_agent_wrapper=trpo_agent_wrapper,
                  discriminator_wrapper=discriminator_wrapper,
                  num_trajs=args.num_trajs,
                  sample_or_mode=args.sample_or_mode,
                  render=args.render,
                  exact_model_path=args.exact_model_path,
                  model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = gail_argparser().parse_args()
    if _args.task == 'imitate_via_gail':
        imitate_via_gail(_args)
    elif _args.task == 'evaluate_gail_policy':
        evaluate_gail_policy(_args)
    else:
        raise NotImplementedError
