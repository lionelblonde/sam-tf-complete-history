import numpy as np
import tensorflow as tf

from gym import spaces

from imitation.common import tf_util as U
from imitation.common import abstract_module as my
from imitation.common.sonnet_util import RewardNN
from imitation.common.mpi_running_mean_std import RunningMeanStd


class Discriminator(my.AbstractModule):

    def __init__(self, name, env, hps):
        super(Discriminator, self).__init__(name=name)
        # Define everything in a specific scope
        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._init(env=env, hps=hps)

    def _init(self, env, hps):
        # Parameters
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        if "NoFrameskip" in env.spec.id:
            # Expand the dimension for Atari
            self.ac_shape = (1,) + self.ac_shape
        self.hps = hps
        assert self.hps.ent_reg_scale >= 0, "'ent_reg_scale' must be non-negative"

        # Assemble clipping functions
        unlimited_range = (-np.infty, np.infty)
        if isinstance(self.ac_space, spaces.Box):
            self.clip_obs = U.clip((-5, 5))
        elif isinstance(self.ac_space, spaces.Discrete):
            self.clip_obs = U.clip(unlimited_range)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        # Define the synthetic reward network
        self.reward_nn = RewardNN(scope=self.scope, name='sr', hps=self.hps)

        # Create inputs
        p_obs = tf.placeholder(name='p_obs', dtype=tf.float32, shape=(None,) + self.ob_shape)
        p_acs = tf.placeholder(name='p_acs', dtype=tf.float32, shape=(None,) + self.ac_shape)
        e_obs = tf.placeholder(name='e_obs', dtype=tf.float32, shape=(None,) + self.ob_shape)
        e_acs = tf.placeholder(name='e_acs', dtype=tf.float32, shape=(None,) + self.ac_shape)

        # Rescale observations
        if self.hps.from_raw_pixels:
            # Scale de pixel values
            p_obz = p_obs / 255.0
            e_obz = e_obs / 255.0
        else:
            if self.hps.rmsify_obs:
                # Smooth out observations using running statistics and clip
                with tf.variable_scope("apply_obs_rms"):
                    self.obs_rms = RunningMeanStd(shape=self.ob_shape)
                p_obz = self.clip_obs(self.rmsify(p_obs, self.obs_rms))
                e_obz = self.clip_obs(self.rmsify(e_obs, self.obs_rms))
            else:
                p_obz = p_obs
                e_obz = e_obs

        # Build graphs
        self.p_scores = self.reward_nn(p_obz, p_acs)
        self.e_scores = self.reward_nn(e_obz, e_acs)
        self.scores = tf.concat([self.p_scores, self.e_scores], axis=0)
        # `scores` define the conditional distribution D(s,a) := p(label|(state,action))

        # Create entropy loss
        self.ent_mean = tf.reduce_mean(U.logit_bernoulli_entropy(logits=self.scores))
        self.ent_loss = -self.hps.ent_reg_scale * self.ent_mean

        # Create labels
        self.fake_labels = tf.zeros_like(self.p_scores)
        self.real_labels = tf.ones_like(self.e_scores)
        if self.hps.label_smoothing:
            # Label smoothing, suggested in 'Improved Techniques for Training GANs',
            # Salimans 2016, https://arxiv.org/abs/1606.03498
            # The paper advises on the use of one-sided label smoothing (positive targets side)
            # Extra comment explanation: https://github.com/openai/improved-gan/blob/
            # 9ff96a7e9e5ac4346796985ddbb9af3239c6eed1/imagenet/build_model.py#L88-L121
            if not self.hps.one_sided_label_smoothing:
                # Fake labels (negative targets)
                soft_fake_u_b = 0.0  # standard, hyperparameterization not needed
                soft_fake_l_b = 0.3  # standard, hyperparameterization not needed
                self.fake_labels = tf.random_uniform(shape=tf.shape(self.fake_labels),
                                                     name="fake_labels_smoothing",
                                                     minval=soft_fake_l_b, maxval=soft_fake_u_b)
            # Real labels (positive targets)
            soft_real_u_b = 0.7  # standard, hyperparameterization not needed
            soft_real_l_b = 1.2  # standard, hyperparameterization not needed
            self.real_labels = tf.random_uniform(shape=tf.shape(self.real_labels),
                                                 name="real_labels_smoothing",
                                                 minval=soft_real_l_b, maxval=soft_real_u_b)
        self.labels = tf.concat([self.fake_labels, self.real_labels], axis=0)

        # Build accuracies
        weights = [1.0 * tf.ones_like(self.p_scores) / U.batch_size(self.p_scores),
                   1.0 * tf.ones_like(self.e_scores) / U.batch_size(self.e_scores)]
        # HAXX: multiply by 1.0 to cast to float
        self.weights = tf.concat(weights, axis=0)
        classification_0_1_scores = tf.to_float((self.scores < 0) == (self.labels == 0))
        self.accuracy = 0.5 * tf.reduce_sum(self.weights * classification_0_1_scores)
        self.p_acc = tf.reduce_mean(tf.sigmoid(self.p_scores))
        self.e_acc = tf.reduce_mean(tf.sigmoid(self.e_scores))
        self.consistency = tf.norm(self.accuracy - (0.5 * (self.p_acc + self.e_acc))) < 1e-8
        self.assert_acc_consistency = U.function([p_obs, p_acs, e_obs, e_acs],
                                                 [self.consistency])

        # Build binary classification losses
        self.p_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.p_scores,
                                                              labels=self.fake_labels)
        self.p_loss_mean = tf.reduce_mean(self.p_loss)
        self.e_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.e_scores,
                                                              labels=self.real_labels)
        self.e_loss_mean = tf.reduce_mean(self.e_loss)

        # Add a gradient penalty (motivation from WGANs (Gulrajani),
        # but empirically useful in JS-GANs(Lucic et al. 2017))
        shape_obz = (tf.to_int64(U.batch_size(p_obz)),) + self.ob_shape
        eps_obz = tf.random_uniform(shape=shape_obz, minval=0.0, maxval=1.0)
        obz_interp = eps_obz * p_obz + (1. - eps_obz) * e_obz
        shape_acs = (tf.to_int64(U.batch_size(p_acs)),) + self.ac_shape
        eps_acs = tf.random_uniform(shape=shape_acs, minval=0.0, maxval=1.0)
        acs_interp = eps_acs * p_acs + (1. - eps_acs) * e_acs
        self.interp_scores = self.reward_nn(obz_interp, acs_interp)
        grads = tf.gradients(self.interp_scores, [obz_interp, acs_interp], name="interp_grads")[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads)))
        self.grad_pen = tf.reduce_mean(tf.square(grad_l2 - 1.0))

        # Assemble previous elements into the losses ops
        self.losses = [self.p_loss_mean,
                       self.e_loss_mean,
                       self.ent_mean,
                       self.ent_loss,
                       self.p_acc,
                       self.e_acc,
                       self.grad_pen]
        self.loss_names = ["policy_loss", "expert_loss", "ent_mean", "ent_loss",
                           "policy_acc", "expert_acc", "pd_grad_pen"]
        self.loss_names = ["d_" + e for e in self.loss_names]
        p_e_losses = tf.concat([self.p_loss, self.e_loss], axis=0)
        self.loss = tf.reduce_sum(self.weights * p_e_losses) + self.ent_loss + self.grad_pen
        # Add coeff in front of gradient penalty

        # Create Theano-like op that computes the discriminator losses and gradients
        self.lossandgrad = U.function([p_obs, p_acs, e_obs, e_acs],
                                      self.losses + [U.flatgrad(self.loss,
                                                                self.trainable_vars,
                                                                self.hps.clip_norm)])

        # Create Theano-like op that compute the synthetic reward
        if self.hps.non_satur_grad:
            # Recommended in the original GAN paper and later in Fedus et al. 2017 (Many Paths...)
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            self.reward = tf.log_sigmoid(self.p_scores)
        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            self.reward = -tf.log(1. - tf.sigmoid(self.p_scores) + 1e-8)  # HAXX: avoids log(0)
        self.compute_reward = U.function([p_obs, p_acs], self.reward)

        # Summarize module information in logs
        self.log_module_info(self.reward_nn)

    def get_reward(self, ob, ac):
        """Compute synthetic reward from a single observation
        The network is able to process observations and actions in minibatches, but the RL
        paradigm enforces the agent to see observations and perform actions in sequences,
        therefore seeing only one at a time.
        `ob` and `ac` are structured as np.array([a, b, c, ...]). Since the network work with
        minibatches, we have to construct a minibatch of size 1, e.g. by using `ob[None]`
        (do not use `[ob]`!) and `ac[None]` which are structured as np.array([[a, b, c, ...]]).
        The network outputs a single synthetic_reward per single observation-action pair in the
        joined input minibatch: np.array([[d]])
        Since a minibatch will later be sequentially construted out of the outputs, we extract
        the output from the returned minibatch of size 1. The extraction can be done by taking
        the first element (one or several times) with `.[0]` or by collapsing the returned array
        into one dimension with numpy's `flatten()` function:
            `synthetic_reward`: np.array([[d]]) -> np.array([d])
        Note that we only manipulate numpy arrays and not lists. We therefore do not need to
        extract the scalar `d` from `np.array([d])`, as for arithmetic operations a numpy array
        of size 1 is equivalent to a scalar (`np.array([1]) + np.array([2]) = np.array([3])` but
        `[1] + [2] = [1, 2]`).
        For safety reason, the scalar is still extracted from the singleton numpy array with
        `np.asscalar`.
        """
        ob_expanded = ob[None]
        ac_expanded = ac[None]
        synthetic_reward = self.compute_reward(ob_expanded, ac_expanded)
        return np.asscalar(synthetic_reward.flatten())

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
