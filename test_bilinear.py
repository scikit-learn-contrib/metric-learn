from metric_learn.oasis import OASIS
import numpy as np

def test_toy_distance():
    u = np.array([0, 1, 2])
    v = np.array([3, 4, 5])

    mixin = OASIS()
    mixin.fit([u, v], [0, 0])
    #mixin.components_ = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

    dist = mixin.score_pairs([[u, v],[v, u]])
    print(dist)

test_toy_distance()