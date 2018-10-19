import numpy as np

from imitation.common.math_util import discount_with_boundaries


def test_discount_with_boundaries():
    gamma = 0.9
    x = np.array([1.0, 2.0, 3.0, 4.0], 'float32')
    starts = [1.0, 0.0, 0.0, 1.0]
    y = discount_with_boundaries(x, starts, gamma)
    assert np.allclose(y, [
                       1 + gamma * 2 + gamma**2 * 3,
                       2 + gamma * 3,
                       3,
                       4])


if __name__ == "__main__":
    test_discount_with_boundaries()
