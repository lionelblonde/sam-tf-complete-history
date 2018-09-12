import time
import copy
import os.path as osp
from collections import deque

import numpy as np

from imitation.common import tf_util as U
from imitation.common import logger
from imitation.common.misc_util import zipsame
from imitation.common.math_util import meanv
from imitation.common.console_util import timed_cm_wrapper, pretty_iter, pretty_elapsed, columnize
from imitation.common.mpi_adam import MpiAdam
from imitation.common.summary_util import CustomSummary
from imitation.common.mpi_moments import mpi_mean_like, mpi_mean_reduce, mpi_moments


def traj_segment_generator(env, mu, d, timesteps_per_batch, comm):
    t = 0
    ac = env.action_space.sample()
    done = True
    syn_rew = 0.0
    env_rew = 0.0
    mu.reset_noise()
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_syn_ret = 0
    cur_ep_env_ret = 0
    ep_lens = []
    ep_syn_rets = []
    ep_env_rets = []
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    qs = np.zeros(timesteps_per_batch, 'float32')
    syn_rews = np.zeros(timesteps_per_batch, 'float32')
    env_rews = np.zeros(timesteps_per_batch, 'float32')
    dones = np.zeros(timesteps_per_batch, 'int32')

    while True:
        ac, q_pred = mu.predict(ob, apply_noise=True, compute_q=True)
        if t > 0 and t % timesteps_per_batch == 0:
            yield {"obs": obs,
                   "acs": acs,
                   "qs": qs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "dones": dones,
                   "ep_lens": ep_lens,
                   "ep_syn_rets": ep_syn_rets,
                   "ep_env_rets": ep_env_rets}
            _, q_pred = mu.predict(ob, apply_noise=True, compute_q=True)
            ep_lens = []
            ep_syn_rets = []
            ep_env_rets = []
        i = t % timesteps_per_batch
        obs[i] = ob
        acs[i] = ac
        qs[i] = q_pred
        dones[i] = done
        syn_rew = d.get_reward(ob, ac)
        new_ob, env_rew, done, _ = env.step(ac)
        syn_rews[i] = syn_rew
        env_rews[i] = env_rew
        cur_ep_len += 1
        cur_ep_syn_ret += syn_rew
        cur_ep_env_ret += env_rew
        mu.store_transition(ob, ac, syn_rew, new_ob, done, comm)
        ob = copy.copy(new_ob)
        if done:
            ep_lens.append(cur_ep_len)
            ep_syn_rets.append(cur_ep_syn_ret)
            ep_env_rets.append(cur_ep_env_ret)
            cur_ep_len = 0
            cur_ep_syn_ret = 0
            cur_ep_env_ret = 0
            mu.reset_noise()
            ob = env.reset()
        t += 1


def traj_ep_generator(env, mu, d, render):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """
    ob = env.reset()
    cur_ep_len = 0
    cur_ep_syn_ret = 0
    cur_ep_env_ret = 0
    obs = []
    acs = []
    qs = []
    syn_rews = []
    env_rews = []

    while True:
        ac, q = mu.predict(ob, apply_noise=False, compute_q=True)
        obs.append(ob)
        acs.append(ac)
        qs.append(q)
        if render:
            env.render()
        syn_rew = d.get_reward(ob, ac)  # must be before the `step` in environment
        new_ob, env_rew, done, _ = env.step(ac)
        syn_rews.append(syn_rew)
        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_syn_ret += syn_rew
        cur_ep_env_ret += env_rew
        if done:
            obs = np.array(obs)
            acs = np.array(acs)
            syn_rews = np.array(syn_rews)
            env_rews = np.array(env_rews)
            yield {"obs": obs,
                   "acs": acs,
                   "qs": qs,
                   "syn_rews": syn_rews,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_syn_ret": cur_ep_syn_ret,
                   "ep_env_ret": cur_ep_env_ret}
            mu.reset_noise()
            ob = env.reset()
            cur_ep_len = 0
            cur_ep_syn_ret = 0
            cur_ep_env_ret = 0
            obs = []
            acs = []
            syn_rews = []
            env_rews = []


def evaluate(env,
             discriminator_wrapper,
             sam_agent_wrapper,
             num_trajs,
             render,
             exact_model_path=None,
             model_ckpt_dir=None):
    """Evaluate a trained SAM agent"""

    # Only one of the two arguments can be provided
    assert sum([exact_model_path is None, model_ckpt_dir is None]) == 1

    # Rebuild the computational graph
    # Create discriminator
    d = discriminator_wrapper('d')
    # Create a sam agent, taking `d` as input
    mu = sam_agent_wrapper('mu', d)
    # Create episode generator
    traj_gen = traj_ep_generator(env, mu, d, render)
    # Initialize and load the previously learned weights into the freshly re-built graph
    U.initialize()
    mu.initialize()
    if exact_model_path is not None:
        U.load_model(exact_model_path)
        logger.info("model loaded from exact path:\n  {}".format(exact_model_path))
    else:  # `exact_model_path` is None -> `model_ckpt_dir` is not None
        U.load_latest_checkpoint(model_ckpt_dir)
        logger.info("model loaded from ckpt dir:\n  {}".format(model_ckpt_dir))
    # Initialize the history data structures
    ep_lens = []
    ep_syn_rets = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, num_trajs))
        traj = traj_gen.__next__()
        ep_len, ep_syn_ret, ep_env_ret = traj['ep_len'], traj['ep_syn_ret'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_syn_rets.append(ep_syn_ret)
        ep_env_rets.append(ep_env_ret)
    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_syn_ret_mean = np.mean(ep_syn_rets)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_syn_ret_mean", ep_syn_ret_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(comm,
          env,
          eval_env,
          discriminator_wrapper,
          sam_agent_wrapper,
          experiment_name,
          ckpt_dir,
          summary_dir,
          expert_dataset,
          add_demos_to_mem,
          pretrained_model_path,
          save_frequency,
          d_lr,
          param_noise_adaption_frequency,
          timesteps_per_batch,
          batch_size,
          g_steps,
          d_steps,
          training_steps_per_iter,
          eval_steps_per_iter,
          render,
          max_timesteps=0,
          max_episodes=0,
          max_iters=0):

    rank = comm.Get_rank()

    # Create discriminator
    d = discriminator_wrapper('d')
    # Create a sam agent, taking `d` as input
    mu = sam_agent_wrapper('mu', d)

    d_adam = MpiAdam(d.trainable_vars)

    if add_demos_to_mem:
        # Add demonstrations to memory
        mu.replay_buffer.add_demo_transitions_to_mem(expert_dataset)

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(comm=comm, logger=logger,
                             color_message='magenta', color_elapsed_time='cyan')

    # Initialize
    U.initialize()
    mu.initialize()
    d_adam.sync()

    if rank == 0:
        # Create summary writer
        writer = U.file_writer(summary_dir)
        # Create the summary
        _names = []
        ep_stats_names = ['ep_length', 'ep_syn_ret', 'ep_env_ret']
        _names.extend(ep_stats_names)
        if mu.param_noise is not None:
            pn_names = ['pn_cur_std', 'pn_dist']
            _names.extend(pn_names)
        mu_l_names = [*mu.actor_names, *mu.critic_names]
        _names.extend(mu_l_names)
        _names.extend(d.loss_names)
        _summary = CustomSummary(scalar_keys=_names, family="sam")

    # Create segment generator for training the agent
    seg_gen = traj_segment_generator(env, mu, d, timesteps_per_batch, comm)
    if eval_env is not None:
        # Create episode generator for evaluating the agent
        eval_ep_gen = traj_ep_generator(eval_env, mu, d, render)

    eps_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # Define rolling buffers for experiental data collection
    maxlen = 100
    ac_buffer = deque(maxlen=maxlen)
    q_buffer = deque(maxlen=maxlen)
    len_buffer = deque(maxlen=maxlen)
    syn_ret_buffer = deque(maxlen=maxlen)
    env_ret_buffer = deque(maxlen=maxlen)
    actor_grads_buffer = deque(maxlen=maxlen)
    actor_losses_buffer = deque(maxlen=maxlen)
    critic_grads_buffer = deque(maxlen=maxlen)
    critic_losses_buffer = deque(maxlen=maxlen)
    d_grads_buffer = deque(maxlen=maxlen)
    d_losses_buffer = deque(maxlen=maxlen)
    if mu.param_noise is not None:
        pn_dist_buffer = deque(maxlen=maxlen)
        pn_cur_std_buffer = deque(maxlen=maxlen)
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        eval_ac_buffer = deque(maxlen=maxlen)
        eval_q_buffer = deque(maxlen=maxlen)
        eval_len_buffer = deque(maxlen=maxlen)
        eval_syn_ret_buffer = deque(maxlen=maxlen)
        eval_env_ret_buffer = deque(maxlen=maxlen)

    # Only one of those three parameters can be set by the user (all three are zero by default)
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    # If pretrained weights are provided
    if pretrained_model_path is not None:
        U.load_model(pretrained_model_path, var_list=mu.actor.vars)
        logger.info("actor model loaded from: {}".format(pretrained_model_path))
        U.load_model(pretrained_model_path, var_list=mu.critic.vars)
        logger.info("critic model loaded from: {}".format(pretrained_model_path))

    while True:

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and eps_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        if hasattr(mu, 'eps_greedy_sched'):
            # Adapt the param noise threshold
            mu.adapt_eps_greedy(timesteps_so_far)

        # Save the model
        if rank == 0 and iters_so_far % save_frequency == 0 and ckpt_dir is not None:
            model_path = osp.join(ckpt_dir, experiment_name)
            U.save_state(model_path, iters_so_far=iters_so_far)
            logger.info("saving model")
            logger.info("  @: {}".format(model_path))

        pretty_iter(logger, iters_so_far)
        pretty_elapsed(logger, tstart)

        # Sample mini-batch in env w/ perturbed actor and store transitions
        with timed("sampling mini-batch"):
            seg = seg_gen.__next__()
        # Extend deques with collected experiental data
        acs, qs = seg['acs'], seg['qs']
        lens, syn_rets, env_rets = seg['ep_lens'], seg['ep_syn_rets'], seg['ep_env_rets']
        ac_buffer.extend(acs)
        q_buffer.extend(qs)
        len_buffer.extend(lens)
        syn_ret_buffer.extend(syn_rets)
        env_ret_buffer.extend(env_rets)

        for training_step in range(training_steps_per_iter):

            logger.info("training [{}/{}]".format(training_step + 1, training_steps_per_iter))

            for d_step in range(d_steps):
                # Update discriminator w/ samples from replay buffer & expert dataset
                logger.info("  updating d [{}/{}]".format(d_step + 1, d_steps))
                # Collect generated data from experience buffer
                xp_batch = mu.replay_buffer.sample(batch_size=batch_size)
                ob_mu, ac_mu = xp_batch['obs0'], xp_batch['acs']

                # Collect expert data w/ identical batch size (GAN's equal mixture)
                ob_expert, ac_expert = expert_dataset.get_next_p_batch(batch_size=batch_size)

                # Update running mean and std
                if hasattr(d, 'obs_rms'):
                    d.obs_rms.update(np.concatenate((ob_mu, ob_expert), axis=0))

                # Compute losses and gradients
                *new_losses, grads = d.lossandgrad(ob_mu, ac_mu, ob_expert, ac_expert)
                # Use the retrieved local gradient to make an Adam optimization update
                d_adam.update(grads, d_lr)
                # Store the losses and gradients in their respective deques
                d_grads_buffer.append(grads)
                d_losses_buffer.append(new_losses)
                # Assess consistency of accuracies
                assert d.assert_acc_consistency(ob_mu, ac_mu, ob_expert, ac_expert)

            if mu.param_noise is not None:
                if training_step % param_noise_adaption_frequency == 0:
                    logger.info("  adapting param noise")
                    # Adapt parameter noise
                    mu.adapt_param_noise(comm)
                    # Store the action-space distance between perturbed and non-perturbed actors
                    pn_dist_buffer.append(mu.pn_dist)
                    # Store the new std resulting from the adaption
                    pn_cur_std_buffer.append(mu.pn_cur_std)

            for g_step in range(g_steps):
                # Update agent w/ samples from replay buffer
                logger.info("  updating g [{}/{}]".format(g_step + 1, g_steps))
                # Train the actor-critic architecture
                actor_grads, actor_loss, critic_grads, critic_loss = mu.train()
                # Store the losses and gradients in their respective deques
                actor_grads_buffer.append(actor_grads)
                actor_losses_buffer.append(actor_loss)
                critic_grads_buffer.append(critic_grads)
                critic_losses_buffer.append(critic_loss)
                # Update the target networks
                mu.update_target_net()

        if eval_env is not None:  # `eval_env` not None iff rank = 0
            assert rank == 0, "non-zero rank mpi worker forbidden here"
            for eval_step in range(eval_steps_per_iter):
                logger.info("evaluating [{}/{}]".format(eval_step + 1, eval_steps_per_iter))
                # Sample an episode w/ non-perturbed actor w/o storing anything
                eval_ep = eval_ep_gen.__next__()
                # Unpack collected episodic data
                eval_acs, eval_qs = eval_ep['acs'], eval_ep['qs']
                eval_len = eval_ep['ep_len']
                eval_syn_ret = eval_ep['ep_syn_ret']
                eval_env_ret = eval_ep['ep_env_ret']
                # Aggregate data collected during the evaluation to the buffers
                eval_ac_buffer.extend(eval_acs)
                eval_q_buffer.extend(eval_qs)
                eval_len_buffer.append(eval_len)
                eval_syn_ret_buffer.append(eval_syn_ret)
                eval_env_ret_buffer.append(eval_env_ret)

        # Make non-zero-rank workers wait for rank zero to finish the eval
        comm.Barrier()

        # Log statistics

        # Specify a width for each column of the two following tables
        column_widths = [20, 16, 16]

        logger.info("logging training losses (log)")
        # Assemble agent losses
        actor_avg_l = mpi_mean_reduce(list(actor_losses_buffer), comm)
        critic_avg_l = mpi_mean_reduce(list(critic_losses_buffer), comm)
        # Assemble discriminator losses
        d_avg_l = mpi_mean_reduce(list(d_losses_buffer), comm)
        # Reorganize for table formatting
        all_agg_l_names = [mu.actor_names[-1]] + [mu.critic_names[-1]]
        all_agg_l = [actor_avg_l[-1]] + [critic_avg_l[-1]]
        all_sub_l_names = mu.actor_names[:-1] + mu.critic_names[:-1]
        actor_sub_us_l = list(actor_avg_l[:(len(actor_avg_l) - 1) // 2])
        critic_sub_us_l = list(critic_avg_l[:(len(critic_avg_l) - 1) // 2])
        all_sub_us_l = actor_sub_us_l + critic_sub_us_l  # unscaled
        actor_sub_s_l = list(actor_avg_l[(len(actor_avg_l) - 1) // 2:-1])
        critic_sub_s_l = list(critic_avg_l[(len(critic_avg_l) - 1) // 2:-1])
        all_sub_s_l = actor_sub_s_l + critic_sub_s_l  # scaled

        zipped_losses = zipsame(all_sub_l_names + all_agg_l_names + d.loss_names,
                                all_sub_us_l + all_agg_l + list(d_avg_l),
                                all_sub_s_l + ['N.A.'] * 2 + ['N.A.'] * 7)  # to maintain symmetry
        logger.info(columnize(names=['name', 'unscaled', 'scaled'],
                              tuples=zipped_losses,
                              widths=column_widths))

        logger.info("logging misc training stats (log)")
        # Initialize lists
        _stats_n = []  # names
        _stats_l = []  # local stats
        _stats_g = []  # global stats
        # Add min, max and mean of the components of the average action
        ac_np_mean = np.mean(ac_buffer, axis=0)  # vector
        ac_mpi_mean, _, _ = mpi_moments(list(ac_buffer), comm)  # vector
        _stats_n.append('min_ac_comp')
        _stats_l.append(np.amin(ac_np_mean))
        _stats_g.append(np.amin(ac_mpi_mean))
        _stats_n.append('max_ac_comp')
        _stats_l.append(np.amax(ac_np_mean))
        _stats_g.append(np.amax(ac_mpi_mean))
        _stats_n.append('mean_ac_comp')
        _stats_l.append(np.mean(ac_np_mean))
        _stats_g.append(np.mean(ac_mpi_mean))
        # Add Q values mean and std
        q_mpi_mean, q_mpi_std, _ = mpi_moments(list(q_buffer), comm)  # scalars
        _stats_n.append('q_value')
        _stats_l.append(np.mean(q_buffer))
        _stats_g.append(q_mpi_mean)
        _stats_n.append('q_deviation')
        _stats_l.append(np.std(q_buffer))
        _stats_g.append(q_mpi_std)
        # Add episodic stats (use custom mean for verbosity)
        _stats_n.append('ep_len')
        _stats_l.append(meanv(len_buffer))
        _stats_g.append(mpi_mean_reduce(list(len_buffer), comm))
        _stats_n.append('ep_syn_ret')
        _stats_l.append(meanv(syn_ret_buffer))
        _stats_g.append(mpi_mean_reduce(list(syn_ret_buffer), comm))
        _stats_n.append('ep_env_ret')
        _stats_l.append(meanv(env_ret_buffer))
        _stats_g.append(mpi_mean_reduce(list(env_ret_buffer), comm))
        # Add misc time-related stats
        eps_this_iter = len(lens)
        timesteps_this_iter = sum(lens)
        eps_so_far += eps_this_iter
        timesteps_so_far += timesteps_this_iter
        _stats_n.append('ep_this_iter')
        _stats_l.append(np.mean(eps_this_iter))
        _stats_g.append(mpi_mean_like(eps_this_iter, comm))
        _stats_n.append('ts_this_iter')
        _stats_l.append(np.mean(timesteps_this_iter))
        _stats_g.append(mpi_mean_like(timesteps_this_iter, comm))
        _stats_n.append('eps_so_far')
        _stats_l.append(np.mean(eps_so_far))
        _stats_g.append(mpi_mean_like(eps_so_far, comm))
        _stats_n.append('ts_so_far')
        _stats_l.append(np.mean(timesteps_so_far))
        _stats_g.append(mpi_mean_like(timesteps_so_far, comm))
        # Add gradient norms
        _stats_n.append('actor_grad_norm')
        _stats_l.append(np.linalg.norm(np.mean(actor_grads_buffer, axis=0)))
        _stats_g.append(np.linalg.norm(mpi_mean_reduce(list(actor_grads_buffer), comm)))
        _stats_n.append('critic_grad_norm')
        _stats_l.append(np.linalg.norm(np.mean(critic_grads_buffer, axis=0)))
        _stats_g.append(np.linalg.norm(mpi_mean_reduce(list(critic_grads_buffer), comm)))
        _stats_n.append('d_grad_norm')
        _stats_l.append(np.linalg.norm(np.mean(d_grads_buffer, axis=0)))
        _stats_g.append(np.linalg.norm(mpi_mean_reduce(list(d_grads_buffer), comm)))
        if mu.param_noise is not None:
            # Add parameter noise current std
            pn_cur_std_mpi_mean = mpi_mean_reduce(list(pn_cur_std_buffer), comm)
            _stats_n.append('pn_cur_std')
            _stats_l.append(np.mean(pn_cur_std_buffer))
            _stats_g.append(pn_cur_std_mpi_mean)
            # Add parameter noise distance
            pn_dist_mpi_mean = mpi_mean_reduce(list(pn_dist_buffer), comm)
            _stats_n.append('pn_dist')
            _stats_l.append(np.mean(pn_dist_buffer))
            _stats_g.append(pn_dist_mpi_mean)
        # Add replay buffer num entries
        _stats_n.append('mem_num_entries')
        _stats_l.append(np.mean(mu.replay_buffer.num_entries))
        _stats_g.append(mpi_mean_like(mu.replay_buffer.num_entries, comm))
        # Zip names, local and global stats
        zipped_nlg = zipsame(_stats_n, _stats_l, _stats_g)
        # Log the dictionaries into one table
        logger.info(columnize(names=['name', 'local', 'global'],
                              tuples=zipped_nlg,
                              widths=column_widths))

        if eval_env is not None:
            # Use the logger object to log the eval stats (will appear in `progress{}.csv`)
            assert rank == 0, "non-zero rank mpi worker forbidden here"
            logger.info("logging misc eval stats (log + csv)")
            # Add min, max and mean of the components of the average action
            ac_np_mean = np.mean(eval_ac_buffer, axis=0)  # vector
            logger.record_tabular('min_ac_comp', np.amin(ac_np_mean))
            logger.record_tabular('max_ac_comp', np.amax(ac_np_mean))
            logger.record_tabular('mean_ac_comp', np.mean(ac_np_mean))
            # Add Q values mean and std
            logger.record_tabular('q_value', np.mean(eval_q_buffer))
            logger.record_tabular('q_deviation', np.std(eval_q_buffer))
            # Add episodic stats
            logger.record_tabular('ep_len', np.mean(eval_len_buffer))
            logger.record_tabular('ep_syn_ret', np.mean(eval_syn_ret_buffer))
            logger.record_tabular('ep_env_ret', np.mean(eval_env_ret_buffer))
            logger.dump_tabular()

        # Mark the end of the iter in the logs
        logger.info('')

        iters_so_far += 1

        # Prepare losses to be dumped in summaries
        # Note: this block must be visible by all workers
        all_summaries = []
        ep_stats_summary = []
        if eval_env is not None:
            # Add misc episodic stats from eval time
            assert rank == 0, "non-zero rank mpi worker forbidden here"
            ep_stats_summary = [np.mean(eval_len_buffer),
                                np.mean(eval_syn_ret_buffer),
                                np.mean(eval_env_ret_buffer)]
        all_summaries.extend(ep_stats_summary)
        # Add stats from training time
        if mu.param_noise is not None:
            # Add param noise stats to summary
            pn_summary = [pn_cur_std_mpi_mean, pn_dist_mpi_mean]
            all_summaries.extend(pn_summary)
        # Add all the scaled sublosses and the aggregated losses
        mu_avg_l_summary = (actor_sub_s_l + [actor_avg_l[-1]] +
                            critic_sub_s_l + [critic_avg_l[-1]])
        all_summaries.extend(mu_avg_l_summary)
        d_avg_l_summary = [*d_avg_l]
        all_summaries.extend(d_avg_l_summary)

        if rank == 0:
            assert len(_names) == len(all_summaries), "mismatch in list lengths"
            _summary.add_all_summaries(writer, all_summaries, iters_so_far)
