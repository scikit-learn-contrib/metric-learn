from metric_learn.oasis import OASIS
import numpy as np
from numpy.testing import assert_array_almost_equal

def test_toy_distance():
    d = 100

    u = np.random.rand(d)
    v = np.random.rand(d)

    mixin = OASIS()
    mixin.fit([u, v], [0, 0]) # Dummy fit

    # The distances must match, whether calc with get_metric() or score_pairs()
    dist1 = mixin.score_pairs([[u, v], [v, u]])
    dist2 = [mixin.get_metric()(u, v), mixin.get_metric()(v, u)]
    
    u_v = (np.dot(np.dot(u.T, mixin.get_bilinear_matrix()), v))
    v_u = (np.dot(np.dot(v.T, mixin.get_bilinear_matrix()), u))
    desired = [u_v, v_u]
    
    assert_array_almost_equal(dist1, desired)
    assert_array_almost_equal(dist2, desired)

test_toy_distance()