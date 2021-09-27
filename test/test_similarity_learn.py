from metric_learn.oasis import OASIS
import numpy as np
from sklearn.utils import check_random_state
import pytest

RNG = check_random_state(0)


def test_sanity_check():
  """
  With M=I init. As the algorithm sees more triplets,
  the score(triplet) should increse or maintain.

  A warning might show up regarding division by 0. See
  test_divide_zero for further research.
  """
  triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                       [[2, 1], [0, 1], [2, 0]],
                       [[0, 0], [2, 0], [0, 1]],
                       [[2, 0], [0, 0], [2, 1]]])

  # Baseline, no M = Identity
  oasis1 = OASIS(n_iter=0, c=0.24, random_state=RNG)
  oasis1.fit(triplets)
  a1 = oasis1.score(triplets)

  msg = "divide by zero encountered in double_scalars"
  with pytest.warns(RuntimeWarning) as raised_warning:
  # See 2/4 triplets
    oasis2 = OASIS(n_iter=2, c=0.24, random_state=RNG)
    oasis2.fit(triplets)
    a2 = oasis2.score(triplets)

    # See 3/4 triplets
    oasis3 = OASIS(n_iter=3, c=0.24, random_state=RNG)
    oasis3.fit(triplets)
    a3 = oasis3.score(triplets)

    # See 5/4 triplets, one is seen again
    oasis4 = OASIS(n_iter=6, c=0.24, random_state=RNG)
    oasis4.fit(triplets)
    a4 = oasis4.score(triplets)

    assert a2 >= a1
    assert a3 >= a2
    assert a4 >= a3
  assert msg == raised_warning[0].message.args[0]

def test_score_zero():
  """
  The third triplet will give similarity 0, then the prediction
  will be 0. But predict() must give results in {+1, -1}. This
  tests forcing prediction 0 to be -1.
  """
  triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                       [[2, 1], [0, 1], [2, 0]],
                       [[0, 0], [2, 0], [0, 1]],
                       [[2, 0], [0, 0], [2, 1]]])

  # Baseline, no M = Identity
  oasis1 = OASIS(n_iter=0, c=0.24, random_state=RNG)
  oasis1.fit(triplets)
  predictions = oasis1.predict(triplets)
  not_valid = [e for e in predictions if e not in [-1, 1]]
  assert len(not_valid) == 0


def test_divide_zero():
  """
  The thrid triplet willl force norm(V_i) to be zero, and
  force a division by 0 when calculating tau = loss / norm(V_i).
  No error should be experienced. A warning should show up.
  """
  triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                       [[2, 1], [0, 1], [2, 0]],
                       [[0, 0], [2, 0], [0, 1]],
                       [[2, 0], [0, 0], [2, 1]]])

  # Baseline, no M = Identity
  oasis1 = OASIS(n_iter=20, c=0.24, random_state=RNG)
  msg = "divide by zero encountered in double_scalars"
  with pytest.warns(RuntimeWarning) as raised_warning:
    oasis1.fit(triplets)
  assert msg == raised_warning[0].message.args[0]
