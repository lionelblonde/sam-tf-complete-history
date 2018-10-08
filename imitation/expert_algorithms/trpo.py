import time
import os.path as osp
from collections import deque

import tensorflow as tf
import numpy as np

from imitation.common import tf_util as U
from imitation.common import logger
from imitation.common.feeder import Feeder
from imitation.common.misc_util import zipsame, flatten_lists, prettify_time
from imitation.common.math_util import explained_variance, conjugate_gradient
from imitation.common.math_util import augment_segment_gae_stats
from imitation.common.console_util import columnize, timed_cm_wrapper, pretty_iter, pretty_elapsed
from imitation.common.mpi_adam import MpiAdam
from imitation.common.summary_util import CustomSummary
from imitation.common.mpi_moments import mpi_mean_like
from imitation.expert_algorithms.xpo_util import traj_segment_generator


def learn(comm,
          env,
          xpo_agent_wrapper,
          sample_or_mode,
          gamma,
          max_kl,
          save_frequency,
          ckpt_dir,
          summary_dir,
          timesteps_per_batch,
          batch_size,
          experiment_name,
          ent_reg_scale,
          gae_lambda,
          cg_iters,
          cg_damping,
          vf_iters,
          vf_lr,
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
    flat_tangent = tf.placeholder(name='flat_tan', dtype=tf.float32, shape=[None])  # natural grad

    # Build graphs
    kl_mean = tf.reduce_mean(old_pi.pd_pred.kl(pi.pd_pred))
    ent_mean = tf.reduce_mean(pi.pd_pred.entropy())
    ent_bonus = ent_reg_scale * ent_mean
    vf_err = tf.reduce_mean(tf.square(pi.v_pred - ret))  # MC error
    # The surrogate objective is defined as: advantage * pnew / pold
    ratio = tf.exp(pi.pd_pred.logp(ac) - old_pi.pd_pred.logp(ac))  # IS
    surr_gain = tf.reduce_mean(ratio * adv)  # surrogate objective (CPI)
    # Add entropy bonus
    optim_gain = surr_gain + ent_bonus
    # Assemble losses
    losses = [kl_mean, ent_mean, ent_bonus, surr_gain, optim_gain, vf_err]
    loss_names = ["kl_mean", "ent_mean", "ent_bonus", "surr_gain", "optim_gain", "vf_err"]
    loss_names = ["pol_" + e for e in loss_names]

    # Extract trainable variables
    gp_trainable_vars = pi.gp_trainable_vars
    vf_trainable_vars = pi.vf_trainable_vars

    # Build natural gradient material
    get_flat = U.GetFlat(gp_trainable_vars)
    set_from_flat = U.SetFromFlat(gp_trainable_vars)
    kl_grads = tf.gradients(kl_mean, gp_trainable_vars)
    shapes = [var.get_shape().as_list() for var in gp_trainable_vars]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    # Create the gradient vector product
    gvp = tf.add_n([tf.reduce_sum(g * tangent)
                    for (g, tangent) in zipsame(kl_grads, tangents)])
    # Create the Fisher vector product
    fvp = U.flatgrad(gvp, gp_trainable_vars)

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
    compute_losses = U.function([ob, ac, adv, ret], losses)
    compute_lossandgrad = U.function([ob, ac, adv, ret],
                                     losses + [U.flatgrad(optim_gain, gp_trainable_vars)])
    compute_fvp = U.function([flat_tangent, ob, ac, adv], fvp)
    compute_vf_grad = U.function([ob, ret], U.flatgrad(vf_err, vf_trainable_vars))

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm=comm, logger=logger,
                             color_message='magenta', color_elapsed_time='cyan')

    # Create mpi adam optimizer
    vf_adam = MpiAdam(vf_trainable_vars)

    U.initialize()
    # Initialize MPI sync
    theta_init = get_flat()
    comm.Bcast(theta_init, root=0)
    set_from_flat(theta_init)
    vf_adam.sync()

    if rank == 0:
        # Create summary writer
        writer = U.file_writer(summary_dir)
        ep_stats_names = ["ep_len", "ep_env_ret"]
        _names = ep_stats_names + loss_names
        _summary = CustomSummary(scalar_keys=_names, family="trpo")

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

        def fisher_vector_product(p):
            computed_fvp = compute_fvp(p, obs, acs, advs)
            return mpi_mean_like(computed_fvp, comm) + cg_damping * p

        assign_old_eq_new()

        # Compute gradients
        with timed("computing gradients"):
            *loss_before, g = compute_lossandgrad(obs, acs, advs, td_lam_rets)
        loss_before = mpi_mean_like(loss_before, comm)
        g = mpi_mean_like(g, comm)
        if np.allclose(g, 0):
            logger.info("got zero gradient -> not updating")
        else:
            with timed("performing conjugate gradient procedure"):
                step_direction = conjugate_gradient(fisher_vector_product, g, cg_iters,
                                                    verbose=(rank == 0))
            assert np.isfinite(step_direction).all()
            shs = 0.5 * step_direction.dot(fisher_vector_product(step_direction))
            # shs is (1/2)*s^T*A*s in the paper
            lm = np.sqrt(shs / max_kl)
            # lm is 1/beta in the paper (max_kl is user-specified delta)
            full_step = step_direction / lm  # beta*s
            expected_improve = g.dot(full_step)  # project s on g
            surr_before = loss_before[4]  # 5-th in loss list
            step_size = 1.0
            theta_before = get_flat()
            with timed("updating policy"):
                for _ in range(10):  # trying (10 times max) until the stepsize is OK
                    # Update the policy parameters
                    theta_new = theta_before + full_step * step_size
                    set_from_flat(theta_new)
                    pi_losses = compute_losses(obs, acs, advs, td_lam_rets)
                    pi_losses_mpi_mean = mpi_mean_like(pi_losses, comm)
                    # Extract specific losses
                    surr = pi_losses_mpi_mean[4]
                    kl = pi_losses_mpi_mean[0]
                    actual_improve = surr - surr_before
                    logger.info("  expected: {:.3f} | actual: {:.3f}".format(expected_improve,
                                                                             actual_improve))
                    if not np.isfinite(pi_losses_mpi_mean).all():
                        logger.info("  got non-finite value of losses :(")
                    elif kl > max_kl * 1.5:
                        logger.info("  violated KL constraint -> shrinking step.")
                    elif actual_improve < 0:
                        logger.info("  surrogate didn't improve -> shrinking step.")
                    else:
                        logger.info("  stepsize fine :)")
                        break
                    step_size *= 0.5  # backtracking when the step size is deemed inappropriate
                else:
                    logger.info("  couldn't compute a good step")
                    set_from_flat(theta_before)

        # Create Feeder object to iterate over (ob, ret) pairs
        feeder = Feeder(data_map=dict(obs=obs, td_lam_rets=td_lam_rets), enable_shuffle=True)
        # Update state-value function
        with timed("updating value function"):
            for _ in range(vf_iters):
                for minibatch in feeder.get_feed(batch_size=batch_size):
                    g = compute_vf_grad(minibatch['obs'], minibatch['td_lam_rets'])
                    vf_adam.update(g, vf_lr)
        # Log policy update statistics
        logger.info("logging training losses (log)")
        zipped_losses = zipsame(loss_names, pi_losses, pi_losses_mpi_mean)
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
