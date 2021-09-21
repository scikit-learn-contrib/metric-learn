from metric_learn.oasis import OASIS
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_toy_distance():
    # Random generalized test for 2 points
    d = 100

    u = np.random.rand(d)
    v = np.random.rand(d)

    mixin = OASIS()
    mixin.fit([u, v], [0, 0])  # Dummy fit

    # The distances must match, whether calc with get_metric() or score_pairs()
    dist1 = mixin.score_pairs([[u, v], [v, u]])
    dist2 = [mixin.get_metric()(u, v), mixin.get_metric()(v, u)]

    u_v = (np.dot(np.dot(u.T, mixin.get_bilinear_matrix()), v))
    v_u = (np.dot(np.dot(v.T, mixin.get_bilinear_matrix()), u))
    desired = [u_v, v_u]

    assert_array_almost_equal(dist1, desired)
    assert_array_almost_equal(dist2, desired)

    # Handmade example
    u = np.array([0, 1, 2])
    v = np.array([3, 4, 5])

    mixin.components_ = np.array([[2, 4, 6], [6, 4, 2], [1, 2, 3]])
    dists = mixin.score_pairs([[u, v], [v, u]])
    assert_array_almost_equal(dists, [96, 120])

    # Symetric example
    u = np.array([0, 1, 2])
    v = np.array([3, 4, 5])

    mixin.components_ = np.identity(3)  # Identity matrix
    dists = mixin.score_pairs([[u, v], [v, u]])
    assert_array_almost_equal(dists, [14, 14])


test_toy_distance()
