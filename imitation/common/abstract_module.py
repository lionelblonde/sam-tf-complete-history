import tensorflow as tf

from imitation.common import tf_util as U
from imitation.common.misc_util import zipsame
from imitation.common import logger
from imitation.common.console_util import columnize


class AbstractModule(object):

    def __init__(self, name):
        self.name = name

    def rmsify(self, x, x_rms):
        """Normalize `x` with running statistics"""
        assert x.dtype == tf.float32, "must be a tensor of the right dtype"
        rmsed_x = (x - x_rms.mean) / x_rms.std
        return rmsed_x

    def dermsify(self, x, x_rms):
        """Denormalize `x` with running statistics"""
        assert x.dtype == tf.float32, "must be a tensor of the right dtype"
        dermsed_x = (x * x_rms.std) + x_rms.mean
        return dermsed_x

    def log_module_info(self, *components):
        assert len(components) > 0, "components list is empty"
        for component in components:
            logger.info("logging {}/{} specs".format(self.name, component.name))
            names = [var.name for var in component.trainable_vars]
            shapes = [U.var_shape(var) for var in component.trainable_vars]
            num_paramss = [U.numel(var) for var in component.trainable_vars]
            zipped_info = zipsame(names, shapes, num_paramss)
            logger.info(columnize(names=['name', 'shape', 'num_params'],
                                  tuples=zipped_info,
                                  widths=[36, 16, 10]))
            logger.info("  total num params: {}".format(sum(num_paramss)))
