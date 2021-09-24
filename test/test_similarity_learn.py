from metric_learn.oasis import OASIS
import numpy as np


def test_not_broken():
  triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                       [[2, 1], [0, 1], [2, 0]],
                       [[0, 0], [2, 0], [0, 1]],
                       [[2, 0], [0, 0], [2, 1]]])
  oasis = OASIS(max_iter=2, c=0.24, random_state=33)
  oasis.fit(triplets)

  new_triplets = np.array([[[0, 1], [4, 5], [0, 0]],
                          [[2, 0], [4, 7], [2, 0]]])

  oasis.partial_fit(new_triplets)
