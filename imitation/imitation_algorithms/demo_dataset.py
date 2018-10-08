import numpy as np

from imitation.common import logger


class PairDataset(object):
    """Dataset containing (state, action) pairs"""

    def __init__(self, obs0, acs, randomize):
        self.obs0 = obs0
        self.acs = acs
        self.num_entries = len(self.obs0)
        assert all(len(x) == self.num_entries for x in [self.obs0, self.acs])
        self.randomize = randomize
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            # Create vector of indices
            indices = np.arange(self.num_entries)
            # Shuffle the indices
            np.random.shuffle(indices)
            # Rearrange the pairs according to the indices
            self.obs0 = self.obs0[indices, :]
            self.acs = self.acs[indices, :]

    def get_next_batch(self, batch_size):
        """Returns a batch of pairs"""
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.obs0, self.acs
        if self.pointer + batch_size >= self.num_entries:
            # if there are not enough pairs left after the pointer,
            # reset the pointer, which shuffles the dataset if `randomize` is True
            self.init_pointer()
        end = self.pointer + batch_size
        obs0 = self.obs0[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        self.pointer = end
        return obs0, acs


class TransitionDataset(object):
    """Data containing (state, action, reward, end of episode, next state) tuples"""

    def __init__(self, obs0, acs, env_rews, dones1, obs1, randomize):
        self.obs0 = obs0
        self.acs = acs
        self.env_rews = env_rews
        self.dones1 = dones1
        self.obs1 = obs1
        self.num_entries = len(self.obs0)
        assert all(len(x) == self.num_entries for x in [self.obs0, self.acs, self.env_rews,
                                                        self.dones1, self.obs1])
        self.randomize = randomize
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            # Create vector of indices
            indices = np.arange(self.num_entries)
            # Shuffle the indices
            np.random.shuffle(indices)
            # Rearrange the transitions according to the indices
            self.obs0 = self.obs0[indices, :]
            self.acs = self.acs[indices, :]
            self.env_rews = self.env_rews[indices, :]
            self.dones1 = self.dones1[indices, :]
            self.obs1 = self.obs1[indices, :]

    def get_next_batch(self, batch_size):
        """Returns a batch of transitions"""
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.obs0, self.acs, self.env_rews, self.dones1, self.obs1
        if self.pointer + batch_size >= self.num_entries:
            # if there are not enough pairs left after the pointer, reset the pointer,
            # which shuffles the dataset if `randomize` is True
            self.init_pointer()
        end = self.pointer + batch_size
        obs0 = self.obs0[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        rews = self.env_rews[self.pointer:end, :]
        dones1 = self.dones1[self.pointer:end, :]
        obs1 = self.obs1[self.pointer:end, :]
        self.pointer = end
        return obs0, acs, rews, dones1, obs1


class DemoDataset(object):

    def __init__(self, expert_arxiv, size, train_fraction=None, randomize=True):
        """Create a dataset given the `expert_path` expert demonstration trajectories archive.
        Data structure of the archive in .npz format:
        the transitions are saved in python dictionary format with keys:
        'obs0', 'acs', 'rews', 'dones1', 'obs1', 'ep_rets',
        the values of each item is a list storing the expert trajectory sequentially.
        Note that 'ep_rets' is stored solely for monitoring purposes, and w/o 'ep_rets',
        a transition corrsponds exactly to the format of transitions stored in memory.
        """
        logger.info("loading expert demonstration trajectories from archive")
        # Load the .npz archive file
        self.traj_data = np.load(expert_arxiv)

        # Establish the num of trajectories to work w/: `self.size`
        self.size = size
        if self.size is None or self.size > self.total_num_trajs or self.size <= 0:
            # Override `num_demo_trajs` w/ the max available num of trajs, i.e. all of them
            self.size = self.total_num_trajs
        # Unpack
        #   1. Slice the desired quantity of trajectories
        #   2. Flatten the list of trajectories into a list of transitions
        #   Unpacking in done separately for each atom
        self.obs0 = np.array(flatten(self.traj_data['obs0'][:self.size]))
        self.acs = np.array(flatten(self.traj_data['acs'][:self.size]))
        self.env_rews = np.array(flatten(self.traj_data['env_rews'][:self.size]))
        self.dones1 = np.array(flatten(self.traj_data['dones1'][:self.size]))
        self.obs1 = np.array(flatten(self.traj_data['obs1'][:self.size]))

        self.ep_rets = self.traj_data['ep_env_rets'][:self.size]
        self.ep_lens = self.traj_data['ep_lens'][:self.size]

        # Compute dataset statistics
        self.ret_mean = np.mean(np.array(self.ep_rets))
        self.ret_std = np.std(np.array(self.ep_rets))
        self.len_mean = np.mean(np.array(self.ep_lens))
        self.len_std = np.std(np.array(self.ep_lens))

        # Create (obs0,acs) dataset
        self.randomize = randomize
        self.pair_dset = PairDataset(self.obs0, self.acs, self.randomize)

        if train_fraction is not None:
            # Split dataset into train and test datasets (used in BC)
            t_t_frontier = int(self.extracted_num_transitions * train_fraction)
            self.pair_train_set = PairDataset(self.obs0[:t_t_frontier, :],
                                              self.acs[:t_t_frontier, :],
                                              self.randomize)
            self.pair_val_set = PairDataset(self.obs0[t_t_frontier:, :],
                                            self.acs[t_t_frontier:, :],
                                            self.randomize)

        # Log message upon successful trajectory dataset initialization
        self.log_info()

    def log_info(self):
        logger.info("successfully initialized (obs0,acs) dataset, w/ statitics:")
        logger.info("  extracted num trajectories: {}".format(self.size))
        logger.info("  extracted num transitions: {}".format(self.extracted_num_transitions))
        logger.info("  trajectory return mean: {}".format(self.ret_mean))
        logger.info("  trajectory return std: {}".format(self.ret_std))
        logger.info("  trajectory length mean: {}".format(self.len_mean))
        logger.info("  trajectory length std: {}".format(self.len_std))

    def get_next_p_batch(self, batch_size, split=None):
        """Returns a batch of pairs"""
        if split is None:
            return self.pair_dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.pair_train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.pair_val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def get_next_t_batch(self, batch_size, split=None):
        """Returns a batch of transitions"""
        if split is None:
            return self.transition_dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.transition_train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.transition_val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.ep_rets, bins=10, density=True, facecolor='b', alpha=0.75)
        plt.xlabel('Return')
        plt.title("Histogram of Episodic Returns")
        plt.grid(True)
        plt.show()
        plt.close()

    @property
    def total_num_trajs(self):
        """Get the num of trajectories
        traj_data['w/e'] is a list of lists of w/e, each sub-list corresponding to a traj
        len(traj_data['w/e']) therefore is the total num of trajs in the archive
        Since the structure is similar for every atomic element of a transition,
        we arbitrarily picked 'obs0'.
        """
        return len(self.traj_data['obs0'])

    @property
    def extracted_num_transitions(self):
        # `self.obs0` picked arbitrarily
        return len(self.obs0)


def flatten(x):
    """Flatten list a trajectories, themselves lists of transitions
    into a list of transitions, getting rid of one embedding depth level.
    Example of flattening:
        input: 2 trajectories of 3 transitions each, w/ atom dimension of 4
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        output: 6 transitions, each, w/ atom dimension of 4
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    Note that each atom is treated separately; 'transition' here means 'one of the atoms'
    """
    if len(x[0].shape) == 1:
        atom_dim = 1
    elif len(x[0].shape) == 4:  # for raw pixel inputs
        atom_dim = (x[0].shape)[1:]
    else:
        atom_dim = (x[0].shape)[1]
    # print("atom_dim: {}".format(atom_dim))

    # Extract trajectories respective lengths, corresponding to the number of
    # transitions in each of the trajectories (in example, `traj_lens` is [3, 3])
    traj_lens = [len(i) for i in x]
    # Compute the total number of transitions (in example, `total_num_transitions` is 6)
    total_num_transitions = sum(traj_lens)
    # Create x's new shape (total num transitions, atom dim)
    # (in example, x's new shape is (6, 4))
    if np.isscalar(atom_dim):
        new_shape = (total_num_transitions, atom_dim)
    else:
        new_shape = (total_num_transitions, *atom_dim)
    # Reshape the array to have the desired shape. Could use `np.reshape(x, new_shape)`
    # but this does not work w/ embedded np arrays, only w/ purely embedded py lists
    # Initialize array w/ the correct size
    x_flat = np.empty(new_shape)
    # Fill the array w/ NaNs, which we will use to check that the array has been fully overriden
    x_flat.fill(np.nan)

    start_index = 0
    # Iterate over the trajectories
    for traj, traj_len in zip(x, traj_lens):
        # Insert `traj` at the right spot in `x_flat`
        if len(x[0].shape) == 1:
            # necessary for rewards (scalars) and categorical actions for flattening
            traj = np.expand_dims(traj, axis=1)
        x_flat[start_index:start_index + traj_len] = traj
        # Update start index
        start_index += traj_len
    assert start_index == total_num_transitions
    # Verify that all the NaNs have been overridden
    assert not np.any(np.isnan(x_flat)), "array contains NaNs"
    return x_flat


def test(expert_arxiv, size, plot):
    """Test the dataset creation
    Example usage: execute the following command from the project root
        python -m imitation.imitation_utils.demo_dataset \
            --expert_arxiv=data/expert_demonstrations/sehee_zerroo_koocer.gather_trajectories. \
            model_phushe_feefa_werhoo.num_trajs_100.Hopper.seed_0.npz \
            --num_demos=40 --plot=True
    """
    dset = DemoDataset(expert_arxiv, size=size)
    if plot:
        dset.plot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_arxiv", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_arxiv, args.num_demos, args.plot)
