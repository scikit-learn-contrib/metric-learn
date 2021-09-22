from metric_learn.oasis import OASIS
import numpy as np
from numpy.testing import assert_array_almost_equal
from timeit import default_timer as timer


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


def test_bilinar_properties():
    d = 100

    u = np.random.rand(d)
    v = np.random.rand(d)

    mixin = OASIS()
    mixin.fit([u, v], [0, 0])  # Dummy fit

    dist1 = mixin.score_pairs([[u, u], [v, v], [u, v], [v, u]])

    print(dist1)


def test_performace():

    features = int(1e4)
    samples = int(1e4)

    a = [np.random.rand(features) for i in range(samples)]
    b = [np.random.rand(features) for i in range(samples)]
    pairs = np.array([(aa, bb) for aa, bb in zip(a, b)])
    components = np.identity(features)

    def op_1(pairs, components):
        return np.diagonal(np.dot(
            np.dot(pairs[:, 0, :], components),
            pairs[:, 1, :].T))

    def op_2(pairs, components):
        return np.array([np.dot(np.dot(u.T, components), v)
                         for u, v in zip(pairs[:, 0, :], pairs[:, 1, :])])

    def op_3(pairs, components):
        return np.sum(np.dot(pairs[:, 0, :], components) * pairs[:, 1, :],
                      axis=-1)

    # Test first method
    start = timer()
    op_1(pairs, components)
    end = timer()
    print(f'First method took {end - start}')

    # Test second method
    start = timer()
    op_2(pairs, components)
    end = timer()
    print(f'Second method took {end - start}')
    # Test second method
    start = timer()
    op_3(pairs, components)
    end = timer()
    print(f'Third method took {end - start}')
