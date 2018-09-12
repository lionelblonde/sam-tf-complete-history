import tensorflow as tf
import sonnet as snt

from gym import spaces

from imitation.common import tf_util as U
from imitation.common.misc_util import zipsame


def parse_nonlin(nonlin_key):
    """Parse the activation function"""
    nonlin_map = dict(relu=tf.nn.relu,
                      leaky_relu=U.leaky_relu,
                      prelu=U.prelu,
                      elu=tf.nn.elu,
                      selu=U.selu,
                      tanh=tf.nn.tanh,
                      identity=tf.identity)
    if nonlin_key in nonlin_map.keys():
        return nonlin_map[nonlin_key]
    else:
        raise RuntimeError("unknown nonlinearity: '{}'".format(nonlin_key))


def parse_initializer(hid_w_init_key):
    """Parse the weight initializer"""
    init_map = dict(he_normal=U.he_normal_init(),
                    he_uniform=U.he_uniform_init(),
                    xavier_normal=U.xavier_normal_init(),
                    xavier_uniform=U.xavier_uniform_init())
    if hid_w_init_key in init_map.keys():
        return init_map[hid_w_init_key]
    else:
        raise RuntimeError("unknown weight init: '{}'".format(hid_w_init_key))


class AbstractNN(object):

    def __init__(self):
        pass

    def stack_conv2d_layers(self, embedding):
        """Stack the Conv2D layers"""
        conv2dz = zipsame(self.hps.nums_filters, self.hps.filter_shapes, self.hps.stride_shapes)
        for conv2d_layer_index, zipped_conv2d in enumerate(conv2dz, start=1):
            conv2d_layer_id = "conv2d{}".format(conv2d_layer_index)
            num_filters, filter_shape, stride_shape = zipped_conv2d  # unpack
            # Add cond2d hidden layer
            embedding = snt.Conv2D(output_channels=num_filters,
                                   kernel_shape=filter_shape,
                                   stride=stride_shape,
                                   padding=snt.VALID,
                                   name=conv2d_layer_id,
                                   initializers=self.hid_initializers,
                                   regularizers=self.hid_regularizers)(embedding)
            # Add non-linearity
            embedding = parse_nonlin(self.hps.hid_nonlin)(embedding)

        # Flatten between conv2d layers and fully-connected layers
        embedding = snt.BatchFlatten(name='flatten')(embedding)
        return embedding

    def stack_fc_layers(self, embedding):
        """Stack the fully-connected layers
        Note that according to the paper 'Parameter Space Noise for Exploration', layer
        normalization should only be used for the fully-connected part of the network.
        """
        for hid_layer_index, hid_width in enumerate(self.hps.hid_widths, start=1):
            hid_layer_id = "fc{}".format(hid_layer_index)
            # Add hidden layer
            embedding = snt.Linear(output_size=hid_width, name=hid_layer_id,
                                   initializers=self.hid_initializers,
                                   regularizers=self.hid_regularizers)(embedding)
            # Add non-linearity
            embedding = parse_nonlin(self.hps.hid_nonlin)(embedding)
            if self.hps.with_layernorm:
                # Add layer normalization
                layer_norm = snt.LayerNorm(name="layernorm{}".format(hid_layer_index))
                embedding = layer_norm(embedding)
        return embedding

    def set_hid_initializers(self):
        self.hid_initializers = {'w': parse_initializer(self.hps.hid_w_init),
                                 'b': tf.zeros_initializer()}

    def set_out_initializers(self):
        self.out_initializers = {'w': tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                 'b': tf.zeros_initializer()}

    def set_hid_regularizer(self):
        self.hid_regularizers = {'w': U.weight_decay_regularizer(scale=0.)}

    def add_out_layer(self, embedding):
        pass

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope + "/" + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.scope + "/" + self.name)


class PolicyNN(snt.AbstractModule, AbstractNN):

    def __init__(self, scope, name, ac_space, hps):
        super(PolicyNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.ac_space = ac_space
        self.hps = hps
        self.set_pd_type()

    def _build(self, ob):
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = ob
        if self.hps.from_raw_pixels:
            embedding = self.stack_conv2d_layers(embedding)
        embedding = self.stack_fc_layers(embedding)
        pd_params = self.add_out_layer(embedding)
        # Return the probability distribution over actions
        pd = self.pd_type.pdfromflat(pd_params)
        return pd

    def set_pd_type(self):
        """Create `pd_type` based on the action space.
        Covers gaussian policies for continuous action spaces (e.g. MuJoCo)
        and categorical policies for discrete action spaces (e.g. ALE)"""
        from imitation.common.distributions import DiagGaussianPdType, CategoricalPdType
        if isinstance(self.ac_space, spaces.Box):
            self.ac_dim = self.ac_space.shape[-1]  # num dims
            self.pd_type = DiagGaussianPdType(self.ac_dim)
        elif isinstance(self.ac_space, spaces.Discrete):
            self.ac_dim = self.ac_space.n  # num ac choices
            self.pd_type = CategoricalPdType(self.ac_dim)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

    def set_out_initializers(self):
        """Set output initializers"""
        self.out_initializers = {'w': tf.truncated_normal_initializer(stddev=0.01),
                                 'b': tf.zeros_initializer()}

    def add_out_layer(self, embedding):
        """Add the output layer"""
        if isinstance(self.ac_space, spaces.Box) and self.hps.gaussian_fixed_var:
            half_pd_param_len = self.pd_type.param_shape()[0] // 2
            mean = snt.Linear(output_size=half_pd_param_len, name='final',
                              initializers=self.out_initializers)(embedding)
            log_std = tf.get_variable(shape=[1, half_pd_param_len], name='log_std',
                                      initializer=tf.zeros_initializer())
            # Concat mean and std
            # "What is what?"" in the params is done in the pd sampling function, not here
            pd_params = tf.concat([mean, mean * 0.0 + log_std], axis=1)
            # HAXX: w/o `mean * 0.0 +` tf does not accept as mean is [None, ...] >< [1, ...]
            # -> broadcasting haxx
        else:
            pd_params = snt.Linear(output_size=self.pd_type.param_shape()[0], name='final',
                                   initializers=self.out_initializers)(embedding)
        return pd_params


class ValueNN(snt.AbstractModule, AbstractNN):

    def __init__(self, scope, name, hps):
        super(ValueNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.hps = hps

    def _build(self, ob):
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = ob
        if self.hps.from_raw_pixels:
            embedding = self.stack_conv2d_layers(embedding)
        embedding = self.stack_fc_layers(embedding)
        v = self.add_out_layer(embedding)
        return v

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = snt.Linear(output_size=1, name='final',
                               initializers=self.out_initializers)(embedding)
        return embedding


class RewardNN(snt.AbstractModule, AbstractNN):

    def __init__(self, scope, name, hps):
        super(RewardNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.hps = hps

    def _build(self, obs, acs):
        # Concatenate the observations and actions placeholders to form a pair
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = obs
        if self.hps.from_raw_pixels:
            embedding = self.stack_conv2d_layers(embedding)
        embedding = tf.concat([embedding, acs], axis=-1)  # preserves batch size
        embedding = self.stack_fc_layers(embedding)
        scores = self.add_out_layer(embedding)
        return scores

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = snt.Linear(output_size=1, name='final',
                               initializers=self.out_initializers)(embedding)
        return embedding


class ActorNN(snt.AbstractModule, AbstractNN):

    def __init__(self, scope, name, ac_space, hps):
        super(ActorNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.ac_space = ac_space
        self.hps = hps

    def _build(self, ob):
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = ob
        if self.hps.from_raw_pixels:
            embedding = self.stack_conv2d_layers(embedding)
        embedding = self.stack_fc_layers(embedding)
        ac = self.add_out_layer(embedding)
        return ac

    def add_out_layer(self, embedding):
        """Add the output layer"""
        if isinstance(self.ac_space, spaces.Box):
            self.ac_dim = self.ac_space.shape[-1]  # num dims
            embedding = snt.Linear(output_size=self.ac_dim, name='final',
                                   initializers=self.out_initializers)(embedding)
            # Apply tanh as output nonlinearity
            embedding = tf.nn.tanh(embedding)
            # Scale with maximum ac value
            embedding = self.ac_space.high * embedding
        elif isinstance(self.ac_space, spaces.Discrete):
            self.ac_dim = self.ac_space.n  # num ac choices
            embedding = snt.Linear(output_size=self.ac_dim, name='final',
                                   initializers=self.out_initializers)(embedding)
            # Apply softmax as output nonlinearity (prob of playing each discrete action)
            embedding = tf.nn.softmax(embedding, axis=-1)
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")
        return embedding

    @property
    def perturbable_vars(self):
        """Following the paper 'Parameter Space Noise for Exploration', we do not
        perturb the conv2d layers, only the fully-connected part of the network.
        Additionally, the extra variables introduced by layer normalization should remain
        unperturbed as they do not play any role in exploration. The only variables that
        we want to perturb are the weights and biases of the fully-connected layers.
        """
        return [var for var in self.trainable_vars if ('layernorm' not in var.name and
                                                       'conv2d' not in var.name)]


class CriticNN(snt.AbstractModule, AbstractNN):

    def __init__(self, scope, name, hps):
        super(CriticNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.hps = hps

    def _build(self, ob, ac):
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = ob
        if self.hps.from_raw_pixels:
            embedding = self.stack_conv2d_layers(embedding)
        embedding = self.stack_fc_layers(embedding, ac)
        q = self.add_out_layer(embedding)
        return q

    def stack_fc_layers(self, embedding, ac):
        """Stack the fully-connected layers"""
        for hid_layer_index, hid_width in enumerate(self.hps.hid_widths, start=1):
            hid_layer_id = "fc{}".format(hid_layer_index)
            if hid_layer_index == self.hps.ac_branch_in:
                # Concat ac to features extracted from ob
                embedding = tf.concat([embedding, ac], axis=-1)  # preserves batch size
            # Add hidden layer
            embedding = snt.Linear(output_size=hid_width, name=hid_layer_id,
                                   initializers=self.hid_initializers,
                                   regularizers=self.hid_regularizers)(embedding)
            # Add non-linearity
            embedding = parse_nonlin(self.hps.hid_nonlin)(embedding)
            if self.hps.with_layernorm:
                # Add layer normalization
                layer_norm = snt.LayerNorm(name="layernorm{}".format(hid_layer_index))
                embedding = layer_norm(embedding)
        return embedding

    def set_hid_regularizer(self):
        self.hid_regularizers = {'w': U.weight_decay_regularizer(scale=self.hps.wd_scale)}

    def add_out_layer(self, embedding):
        """Add the output layer"""
        embedding = snt.Linear(output_size=1, name='final',
                               initializers=self.out_initializers)(embedding)
        return embedding

    @property
    def decayable_vars(self):
        """Variables on which weight decay can be applied"""
        return [var for var in self.trainable_vars if ('w' in var.name and
                                                       'layernorm' not in var.name and
                                                       'conv2d' not in var.name and
                                                       'final' not in var.name)]

    @property
    def regularization_losses(self):
        return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 scope=self.scope + "/" + self.name)

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'final' in var.name]
        return output_vars
