import time
import os.path as osp
from collections import deque

import tensorflow as tf
import numpy as np

from imitation.common import tf_util as U
from imitation.common import logger
from imitation.common.feeder import Feeder
from imitation.common.misc_util import zipsame, flatten_lists, prettify_time
from imitation.common.math_util import explained_variance
from imitation.common.math_util import augment_segment_gae_stats
from imitation.common.console_util import columnize, timed_cm_wrapper, pretty_iter, pretty_elapsed
from imitation.common.mpi_adam import MpiAdam
from imitation.common.summary_util import CustomSummary
from imitation.common.mpi_moments import mpi_mean_like, mpi_mean_reduce
from imitation.expert_algorithms.xpo_util import traj_segment_generator


def learn(comm,
          env,
          xpo_agent_wrapper,
          sample_or_mode,
          gamma,
          save_frequency,
          ckpt_dir,
          summary_dir,
          timesteps_per_batch,
          batch_size,
          optim_epochs_per_iter,
          lr,
          experiment_name,
          ent_reg_scale,
          clipping_eps,
          gae_lambda,
          schedule,
          max_timesteps=0,
          max_episodes=0,
          max_iters=0):

    rank = comm.Get_rank()

    # Create policies
    pi = xpo_agent_wrapper('pi')
    old_pi = xpo_agent_wrapper('old_pi')

    # Create and retrieve already-existing placeholders
    ob = U.get_placeholder_cached(name='ob')
    ac = pi.pd_type.sample_placeholder([None])
    adv = U.get_placeholder(name='adv', dtype=tf.float32, shape=[None])  # advantage
    ret = U.get_placeholder(name='ret', dtype=tf.float32, shape=[None])  # return
    # Adaptive learning rate multiplier, updated with schedule
    lr_mult = U.get_placeholder(name='lr_mult', dtype=tf.float32, shape=[])

    # Build graphs
    kl_mean = tf.reduce_mean(old_pi.pd_pred.kl(pi.pd_pred))
    ent_mean = tf.reduce_mean(pi.pd_pred.entropy())
    ent_pen = (-ent_reg_scale) * ent_mean
    vf_err = tf.reduce_mean(tf.square(pi.v_pred - ret))  # MC error
    # The surrogate objective is defined as: advantage * pnew / pold
    ratio = tf.exp(pi.pd_pred.logp(ac) - old_pi.pd_pred.logp(ac))  # IS
    surr_gain = ratio * adv  # surrogate objective (CPI)
    # Annealed clipping parameter epsilon
    clipping_eps = clipping_eps * lr_mult
    surr_gain_w_clipping = tf.clip_by_value(ratio,
                                            1.0 - clipping_eps,
                                            1.0 + clipping_eps) * adv
    # PPO's pessimistic surrogate (L^CLIP in paper)
    surr_loss = -tf.reduce_mean(tf.minimum(surr_gain, surr_gain_w_clipping))
    # Assemble losses (including the value function loss)
    total_loss = surr_loss + ent_pen + vf_err

    losses = [kl_mean, ent_mean, ent_pen, surr_loss, vf_err]
    loss_names = ["kl_mean", "ent_mean", "ent_pen", "surr_loss", "vf_err"]
    loss_names = ["pol_" + e for e in loss_names]

    # Make the current `pi` become the next `old_pi`
    zipped = zipsame(old_pi.vars, pi.vars)
    updates_op = []
    for k, v in zipped:
        # Populate list of assignment operations
        logger.info("assignment: {} <- {}".format(k, v))
        assign_op = tf.assign(k, v)
        updates_op.append(assign_op)
    assert len(updates_op) == len(pi.vars)
    # Create Theano-like op that performs the update
    assign_old_eq_new = U.function([], [], updates=updates_op)

    # Create Theano-like ops
    compute_lossandgrad = U.function([ob, ac, adv, ret, lr_mult],
                                     losses + [U.flatgrad(total_loss, pi.trainable_vars)])

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm=comm, logger=logger,
                             color_message='magenta', color_elapsed_time='cyan')

    # Create mpi adam optimizer
    adam = MpiAdam(pi.trainable_vars)

    U.initialize()
    adam.sync()

    if rank == 0:
        # Create summary writer
        writer = U.file_writer(summary_dir)
        ep_stats_names = ["ep_len", "ep_env_ret"]
        _names = ep_stats_names + loss_names
        _summary = CustomSummary(scalar_keys=_names, family="ppo")

    seg_gen = traj_segment_generator(env, pi, timesteps_per_batch, sample_or_mode)

    eps_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for recent stats aggregation
    maxlen = 100
    len_buffer = deque(maxlen=maxlen)
    env_ret_buffer = deque(maxlen=maxlen)

    # Only one of those three parameters can be set by the user (all three are zero by default)
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    while True:

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and eps_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Manage lr multiplier schedule
        if schedule == 'constant':
            current_lr_mult = 1.0
        elif schedule == 'linear':
            current_lr_mult = max(1.0 - float(timesteps_per_batch) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # Save the model
        if rank == 0 and iters_so_far % save_frequency == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            U.save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        with timed("sampling mini-batch"):
            seg = seg_gen.__next__()

        augment_segment_gae_stats(seg, gamma, gae_lambda, rew_key="env_rews")

        # Extract rollout data
        obs = seg['obs']
        acs = seg['acs']
        advs = seg['advs']
        # Standardize advantage function estimate
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        vs = seg['vs']
        td_lam_rets = seg['td_lam_rets']

        # Update running mean and std
        if hasattr(pi, 'obs_rms'):
            with timed("normalizing obs via rms"):
                pi.obs_rms.update(obs, comm)

        assign_old_eq_new()

        # Create Feeder object to iterate over (ob, ac, adv, td_lam_ret) tuples
        data_map = dict(obs=obs, acs=acs, advs=advs, td_lam_rets=td_lam_rets)
        feeder = Feeder(data_map=data_map, enable_shuffle=True)
        # Update policy and state-value function
        with timed("updating policy and value function"):
            losses = []
            for _ in range(optim_epochs_per_iter):
                for minibatch in feeder.get_feed(batch_size=batch_size):
                    args = (minibatch['obs'], minibatch['acs'],
                            minibatch['advs'], minibatch['td_lam_rets'])
                    *pi_losses, g = compute_lossandgrad(*args, current_lr_mult)
                    adam.update(g, lr * current_lr_mult)
                    losses.append(pi_losses)
        # Log policy update statistics
        logger.info("logging training losses (log)")
        pi_losses_np_mean = np.mean(losses, axis=0)
        pi_losses_mpi_mean = mpi_mean_reduce(losses, comm, axis=0)
        zipped_losses = zipsame(loss_names, pi_losses_np_mean, pi_losses_mpi_mean)
        logger.info(columnize(names=['name', 'local', 'global'],
                              tuples=zipped_losses,
                              widths=[20, 16, 16]))

        # Log statistics

        logger.info("logging misc training stats (log + csv)")
        # Gather statistics across workers
        local_lens_rets = (seg['ep_lens'], seg['ep_env_rets'])
        gathered_lens_rets = comm.allgather(local_lens_rets)
        lens, env_rets = map(flatten_lists, zip(*gathered_lens_rets))
        # Extend the deques of recorded statistics
        len_buffer.extend(lens)
        env_ret_buffer.extend(env_rets)
        ep_len_mpi_mean = np.mean(len_buffer)
        ep_env_ret_mpi_mean = np.mean(env_ret_buffer)
        logger.record_tabular('ep_len_mpi_mean', ep_len_mpi_mean)
        logger.record_tabular('ep_env_ret_mpi_mean', ep_env_ret_mpi_mean)
        eps_this_iter = len(lens)
        timesteps_this_iter = sum(lens)
        eps_so_far += eps_this_iter
        timesteps_so_far += timesteps_this_iter
        eps_this_iter_mpi_mean = mpi_mean_like(eps_this_iter, comm)
        timesteps_this_iter_mpi_mean = mpi_mean_like(timesteps_this_iter, comm)
        eps_so_far_mpi_mean = mpi_mean_like(eps_so_far, comm)
        timesteps_so_far_mpi_mean = mpi_mean_like(timesteps_so_far, comm)
        logger.record_tabular('eps_this_iter_mpi_mean', eps_this_iter_mpi_mean)
        logger.record_tabular('timesteps_this_iter_mpi_mean', timesteps_this_iter_mpi_mean)
        logger.record_tabular('eps_so_far_mpi_mean', eps_so_far_mpi_mean)
        logger.record_tabular('timesteps_so_far_mpi_mean', timesteps_so_far_mpi_mean)
        logger.record_tabular('elapsed time', prettify_time(time.time() - tstart))  # no mpi mean
        logger.record_tabular('ev_td_lam_before', explained_variance(vs, td_lam_rets))
        iters_so_far += 1

        if rank == 0:
            logger.dump_tabular()

        # Prepare losses to be dumped in summaries
        # This block must be visible by all workers
        ep_stats_summary = [ep_len_mpi_mean, ep_env_ret_mpi_mean]
        agent_loss_summary = [*pi_losses_mpi_mean]
        trpo_stats_summary = ep_stats_summary + agent_loss_summary

        if rank == 0:
            _summary.add_all_summaries(writer, trpo_stats_summary, iters_so_far)
