import os.path as osp

import gym.spaces  # noqa

from imitation.common import tf_util as U
from imitation.common.argparsers import sam_argparser, disambiguate
from imitation.common.experiment_initializer import ExperimentInitializer
from imitation.common.env_makers import make_env
from imitation.imitation_algorithms.sam_agent import SAMAgent
from imitation.imitation_algorithms.discriminator import Discriminator
from imitation.imitation_algorithms import sam
from imitation.imitation_algorithms.demo_dataset import DemoDataset


def imitate_via_sam(args):
    """Train a SAM imitation policy"""
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

    # Refine hps to avoid ambiguities
    actorcritic_hps, d_hps = disambiguate(kvs=args, tokens=['actorcritic', 'd'])

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps)

    # Create a sam agent wrapper (note the second input)
    def sam_agent_wrapper(name, d):
        return SAMAgent(name=name, comm=comm, env=env, hps=actorcritic_hps, d=d)

    # Create experiment name
    experiment_name = experiment.get_long_name()

    # Create the expert demonstrations dataset from expert trajectories
    dataset = DemoDataset(expert_arxiv=args.expert_path, size=args.num_demos)

    # Create an evaluation environment not to mess up with training rollouts
    eval_env = None
    if rank == 0:
        eval_env = make_env(args)(args.env_id, args.seed, "eval", args.horizon)

    comm.Barrier()

    # Train SAM imitation policy
    sam.learn(comm=comm,
              env=env,
              eval_env=eval_env,
              discriminator_wrapper=discriminator_wrapper,
              sam_agent_wrapper=sam_agent_wrapper,
              experiment_name=experiment_name,
              ckpt_dir=osp.join(args.checkpoint_dir, experiment_name),
              summary_dir=osp.join(args.summary_dir, experiment_name),
              expert_dataset=dataset,
              add_demos_to_mem=args.add_demos_to_mem,
              save_frequency=args.save_frequency,
              d_lr=args.d_lr,
              rew_aug_coeff=args.rew_aug_coeff,
              param_noise_adaption_frequency=args.param_noise_adaption_frequency,
              timesteps_per_batch=args.timesteps_per_batch,
              batch_size=args.batch_size,
              g_steps=args.g_steps,
              d_steps=args.d_steps,
              training_steps_per_iter=args.training_steps_per_iter,
              eval_steps_per_iter=args.eval_steps_per_iter,
              render=args.render,
              max_timesteps=args.num_timesteps)

    # Close environment
    env.close()

    # Close the eval env
    if eval_env is not None:
        assert rank == 0
        eval_env.close()


def evaluate_sam_policy(args):
    """Evaluate a trained SAM imitation policy"""
    # Create a single-threaded session
    U.single_threaded_session().__enter__()

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure()

    # Create environment
    env = make_env(args)(args.env_id, args.seed, args.task, args.horizon)

    # Refine hps to avoid ambiguities
    actorcritic_hps, d_hps = disambiguate(kvs=args, tokens=['actorcritic', 'd'])

    def discriminator_wrapper(name):
        return Discriminator(name=name, env=env, hps=d_hps)

    # Create a sam agent wrapper (note the second input)
    def sam_agent_wrapper(name, d):
        return SAMAgent(name=name, env=env, hps=actorcritic_hps, d=d)

    # Evaluate TRPO agent trained via SAM
    sam.evaluate(env=env,
                 discriminator_wrapper=discriminator_wrapper,
                 sam_agent_wrapper=sam_agent_wrapper,
                 num_trajs=args.num_trajs,
                 render=args.render,
                 exact_model_path=args.exact_model_path,
                 model_ckpt_dir=args.model_ckpt_dir)

    # Close environment
    env.close()


if __name__ == '__main__':
    _args = sam_argparser().parse_args()
    if _args.task == 'imitate_via_sam':
        imitate_via_sam(_args)
    elif _args.task == 'evaluate_sam_policy':
        evaluate_sam_policy(_args)
    else:
        raise NotImplementedError
