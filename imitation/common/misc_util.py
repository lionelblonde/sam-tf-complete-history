import random

import gym
import numpy as np


def fl32(x):
    """Cast any castable entity to type 'float32'
    `astype` is a numpy function
    """
    return x.astype('float32')


def flatten_lists(listoflists):
    """Flatten a list of lists"""
    return [el for list_ in listoflists for el in list_]


def zipsame(*seqs):
    """Verify that all the sequences in `seqs` are the same length, then zip them together"""
    assert seqs, "empty input sequence"
    ref_len = len(seqs[0])
    assert all(len(seq) == ref_len for seq in seqs[1:])
    return zip(*seqs)


def unpack(seq, sizes):
    """Unpack `seq` into a sequence of lists, with lengths specified by `sizes`.
    `None` in `sizes` means just one bare element, not a list.

    Example:
    unpack([1, 2, 3, 4, 5, 6], [3, None, 2]) -> ([1, 2, 3], 4, [5, 6])
    Technically `upack` returns a generator object, i.e. an iterator over ([1, 2, 3], 4, [5, 6])
    """
    seq = list(seq)
    it = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), \
        "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield it.__next__()
        else:
            li = []
            for _ in range(size):
                li.append(it.__next__())
            yield li


def set_global_seeds(i):
    """Set global seeds"""
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def prettify_time(seconds):
    """Print the number of seconds in human-readable format.
    Examples: '2 days', '2 hours and 37 minutes', 'less than a minute'.
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        return "{} {}{}".format(str(count), name, ('s' if count > 1 else ''))

    # Display only the two greater units (days and hours, hours and minutes, minutes and seconds)
    if days > 0:
        message = helper(days, 'day')
        if hours > 0:
            message += ' and ' + helper(hours, 'hour')
        return message
    if hours > 0:
        message = helper(hours, 'hour')
        if minutes > 0:
            message += ' and ' + helper(minutes, 'minute')
        return message
    if minutes > 0:
        return helper(minutes, 'minute')
    # Finally, if none of the previous conditions is valid
    return 'less than a minute'


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    # Parameters:
        - parser: argparse.Parser
            parser to add the flag to
        - name: str
            --<name> will enable the flag, while --no-<name> will disable it
        - default: bool or None
            default value of the flag
        - help: str
            help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def get_wrapper_by_name(env, classname):
    """Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    # Parameters:
        - env: gym.Env of gym.Wrapper
            gym environment
        - classname: str
            name of the wrapper

    # Returns:
        - wrapper: gym.Wrapper
            wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
