from metric_learn.oasis import OASIS, OASIS_Supervised
import numpy as np
from sklearn.utils import check_random_state
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from test.test_utils import build_triplets


SEED = 33
RNG = check_random_state(SEED)


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
                       [[2, 0], [0, 0], [2, 1]],
                       [[2, 1], [-1, -1], [33, 21]]])

  # Baseline, no M = Identity
  oasis = OASIS(n_iter=1, c=0.24, random_state=RNG, init='identity')
  # See 1/5 triplets
  oasis.fit(triplets[:1])
  a1 = oasis.score(triplets)

  msg = "divide by zero encountered in double_scalars"
  with pytest.warns(RuntimeWarning) as raised_warning:
    # See 2/5 triplets
    oasis.partial_fit(triplets[1:2], n_iter=2)
    a2 = oasis.score(triplets)

    # See 4/5 triplets
    oasis.partial_fit(triplets[2:4], n_iter=3)
    a3 = oasis.score(triplets)

    # See 5/5 triplets, one is seen again
    oasis.partial_fit(triplets[4:5], n_iter=1)
    a4 = oasis.score(triplets)

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
  with pytest.raises(ValueError):
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


def class_separation(X, labels, callable_metric):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in range(len(unique_labels)):
    Xc = X[label_inds == li]
    Xnc = X[label_inds != li]
    aux = pairwise_distances(Xc, metric=callable_metric).mean()
    ratio += aux / pairwise_distances(Xc, Xnc, metric=callable_metric).mean()
  return ratio / len(unique_labels)


def test_iris_supervised():
  """
  Test a real use case: Using class separation as evaluation metric,
  and the Iris dataset, this tests verifies that points of the same
  class are closer now, using the learnt bilinear similarity at OASIS.

  In contrast with Mahalanobis tests, we cant use transform(X) and
  then use euclidean metric. Instead, we need to pass pairwise_distances
  method from sklearn an explicit callable metric. Then we use
  get_metric() for that purpose.
  """

  # Default bilinear similarity uses M = Identity
  def bilinear_identity(u, v):
    return - np.dot(np.dot(u.T, np.identity(np.shape(u)[0])), v)

  X, y = load_iris(return_X_y=True)
  prev = class_separation(X, y, bilinear_identity)

  oasis = OASIS_Supervised(random_state=33, c=0.38)
  oasis.fit(X, y)
  now = class_separation(X, y, oasis.get_metric())
  assert now < prev  # -0.0407866 vs 1.08 !


@pytest.mark.parametrize('init', ['random', 'random_spd',
                         'covariance', 'identity'])
@pytest.mark.parametrize('random_state', [33, 69, 112])
def test_random_state_in_suffling(init, random_state):
  """
  Tests that many instances of OASIS, with the same random_state,
  produce the same shuffling on the triplets given.

  Test that many instances of OASIS, with different random_state,
  produce different shuffling on the trilpets given.

  The triplets are produced with the Iris dataset.

  Tested with all possible init.
  """
  triplets, _, _, _ = build_triplets()

  # Test same random_state, then same shuffling
  oasis_a = OASIS(random_state=random_state, init=init)
  oasis_a.fit(triplets)
  shuffle_a = oasis_a.indices

  oasis_b = OASIS(random_state=random_state, init=init)
  oasis_b.fit(triplets)
  shuffle_b = oasis_b.indices

  assert_array_equal(shuffle_a, shuffle_b)

  # Test different random states
  last_suffle = shuffle_b
  for i in range(3, 5):
    oasis_a = OASIS(random_state=random_state+i, init=init)
    oasis_a.fit(triplets)
    shuffle_a = oasis_a.indices

    with pytest.raises(AssertionError):
      assert_array_equal(last_suffle, shuffle_a)

    last_suffle = shuffle_a


@pytest.mark.parametrize('init', ['random', 'random_spd',
                         'covariance', 'identity'])
@pytest.mark.parametrize('random_state', [33, 69, 112])
def test_general_results_random_state(init, random_state):
  """
  With fixed triplets and random_state, two instances of OASIS
  should produce the same output (matrix W)
  """
  triplets, _, _, _ = build_triplets()
  oasis_a = OASIS(random_state=random_state, init=init)
  oasis_a.fit(triplets)
  matrix_a = oasis_a.get_bilinear_matrix()

  oasis_b = OASIS(random_state=random_state, init=init)
  oasis_b.fit(triplets)
  matrix_b = oasis_b.get_bilinear_matrix()

  assert_array_equal(matrix_a, matrix_b)
