from metric_learn.oasis import OASIS
import numpy as np
from sklearn.utils import check_random_state


RNG = check_random_state(0)


def test_sanity_check():
  triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                       [[2, 1], [0, 1], [2, 0]],
                       [[0, 0], [2, 0], [0, 1]],
                       [[2, 0], [0, 0], [2, 1]]])

  # Baseline, no M = Identity
  oasis1 = OASIS(max_iter=0, c=0.24, random_state=RNG)
  oasis1.fit(triplets)
  a1 = oasis1.score(triplets)

  # See 2/4 triplets
  oasis2 = OASIS(max_iter=2, c=0.24, random_state=RNG)
  oasis2.fit(triplets)
  a2 = oasis2.score(triplets)

  # See 3/4 triplets
  oasis3 = OASIS(max_iter=3, c=0.24, random_state=RNG)
  oasis3.fit(triplets)
  a3 = oasis3.score(triplets)

  # See 5/4 triplets, one is seen again
  oasis4 = OASIS(max_iter=6, c=0.24, random_state=RNG)
  oasis4.fit(triplets)
  a4 = oasis4.score(triplets)

  assert a2 >= a1
  assert a3 >= a2
  assert a4 >= a3
