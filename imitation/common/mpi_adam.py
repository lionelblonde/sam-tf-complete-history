import numpy as np
import tensorflow as tf
from mpi4py import MPI

import imitation.common.tf_util as U
from imitation.common.mpi_moments import mpi_mean_like


class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0:  # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)


class MpiAdamOptimizer(tf.train.AdamOptimizer):

    def __init__(self, comm, clip_norm=None, **kwargs):
        """Entension of the ADAM optimizer that performs parallel SGD
        consisting in averaging the gradients across mpi processes.
        """
        self.comm = comm
        assert self.comm is not None, "define 'comm' before"
        self.clip_norm = clip_norm
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        """Override the ADAM optimizer standard function"""
        _grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(grad_, var_) if grad_ is not None else (tf.zeros_like(var_), var_)
                          for (grad_, var_) in _grads_and_vars]
        flat_grad = tf.concat([tf.reshape(grad_, [U.numel(var_)])
                               for (grad_, var_) in grads_and_vars], axis=0)
        numels = [U.numel(var_) for _, var_ in grads_and_vars]
        # Wraps a python function and uses it as a TensorFlow op
        mean_flat_grad = tf.py_func(lambda x: mpi_mean_like(x, self.comm),
                                    [flat_grad], Tout=tf.float32)
        mean_flat_grad.set_shape(flat_grad.shape)
        mean_grads = tf.split(mean_flat_grad, numels, axis=0)
        if self.clip_norm is not None:
            # Clip the gradients by the ratio of the sum of their norms
            mean_grads, _ = tf.clip_by_global_norm(mean_grads, clip_norm=self.clip_norm)
        mean_grads_and_vars = [(tf.reshape(mean_grad_, var_.shape), var_)
                               for mean_grad_, (_, var_) in zip(mean_grads, grads_and_vars)]
        return mean_grads_and_vars

    def sync_from_root(self, var_list):
        """Send the root node parameters to every mpi worker"""
        rank = self.comm.Get_rank()
        # Reminder: for a Tensor t, calling t.eval() is equivalent to calling
        # tf.get_default_session().run(t)
        for var_ in var_list:
            if rank == 0:
                # Run the graph to get the value of the variable
                _var = var_.eval()
                # Broadcast the params from rank 0
                self.comm.Bcast(_var, root=0)
            else:
                _var_pulled = np.empty(var_.shape, dtype=np.float32)
                self.comm.Bcast(_var_pulled, root=0)
                tf.get_default_session().run(tf.assign(var_, _var_pulled))

    def check_synced(self, var_list):
        """Assert whether the workers' params have not strayed"""
        rank = self.comm.Get_rank()
        # Reminder: for a Tensor t, calling t.eval() is equivalent to calling
        # tf.get_default_session().run(t)
        for var_ in var_list:
            if rank == 0:
                # Run the graph to get the value of the variable
                _var = var_.eval()
                # Broadcast the params from rank 0
                self.comm.Bcast(_var, root=0)
            else:
                _var_local = var_.eval()
                _var_root = np.empty_like(_var_local)
                self.comm.Bcast(_var_root, root=0)
                assert (_var_root == _var_local).all(), "mismatch {}\n{}".format(_var_root,
                                                                                 _var_local)
