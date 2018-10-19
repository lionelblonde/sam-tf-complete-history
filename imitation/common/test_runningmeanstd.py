import numpy as np

from imitation.common.running_mean_std import RunningMeanStd


def test_runningmeanstd():
    triplet_list = [(np.random.randn(3), np.random.randn(4), np.random.randn(5)),
                    (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2))]
    for (x1, x2, x3) in triplet_list:
        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])
        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]
        assert np.allclose(ms1, ms2)


if __name__ == "__main__":
    test_runningmeanstd()
