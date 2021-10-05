from metric_learn.oasis import OASIS, OASIS_Supervised
import numpy as np
from sklearn.utils import check_random_state
import pytest
from numpy.testing import assert_array_equal, assert_raises
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from metric_learn.constraints import Constraints

SEED = 33
RNG = check_random_state(SEED)


def gen_iris_triplets():
  X, y = load_iris(return_X_y=True)
  constraints = Constraints(y)
  k_geniuine = 3
  k_impostor = 10
  triplets = constraints.generate_knntriplets(X, k_geniuine, k_impostor)
  return X[triplets]


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
  with pytest.raises(ValueError):
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


@pytest.mark.parametrize(('n_triplets', 'n_iter'),
                         [(10, 10), (33, 70), (100, 67),
                         (10000, 20000)])
def test_indices_funct(n_triplets, n_iter):
  """
  This test verifies the behaviour of _get_random_indices. The
  method used inside OASIS that defines the order in which the
  triplets are given to the algorithm, in an online manner.
  """
  oasis = OASIS(random_state=SEED)
  # Not random cases
  base = np.arange(n_triplets)

  # n_iter = n_triplets
  if n_iter == n_triplets:
    r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                  shuffle=False, random=False)
    assert_array_equal(r, base)  # No shuffle
    assert len(r) == len(base)  # Same lenght

    # Shuffle
    r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                  shuffle=True, random=False)
    with assert_raises(AssertionError):  # Should be different
      assert_array_equal(r, base)
    # But contain the same elements
    assert_array_equal(np.unique(r), np.unique(base))
    assert len(r) == len(base)  # Same lenght

  # n_iter > n_triplets
  if n_iter > n_triplets:
    r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                  shuffle=False, random=False)
    assert_array_equal(r[:n_triplets], base)  # First n_triplets must match
    assert len(r) == n_iter  # Expected lenght

    # Next n_iter-n_triplets must be in range(n_triplets)
    sample = r[n_triplets:]
    for i in range(n_iter - n_triplets):
      if sample[i] not in base:
        raise AssertionError("Sampling has values out of range")

    # Shuffle
    r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                  shuffle=True, random=False)
    assert len(r) == n_iter  # Expected lenght

    # Each triplet must be at least one time
    assert_array_equal(np.unique(r), np.unique(base))
    with assert_raises(AssertionError):  # First n_triplets should be different
      assert_array_equal(r[:n_triplets], base)

    # Each index should appear at least ceil(n_iter/n_triplets) - 1 times
    # But no more than ceil(n_iter/n_triplets)
    min_times = int(np.ceil(n_iter / n_triplets)) - 1
    _, counts = np.unique(r, return_counts=True)
    a = len(counts[counts >= min_times])
    b = len(counts[counts <= min_times + 1])
    assert len(np.unique(r)) == a
    assert n_triplets == b

  # n_iter < n_triplets
  if n_iter < n_triplets:
    r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                  shuffle=False, random=False)
    assert len(r) == n_iter  # Expected lenght
    u = np.unique(r)
    assert len(u) == len(r)  # No duplicates
    # Final array must cointain only elements in range(n_triplets)
    for i in range(n_iter):
      if r[i] not in base:
        raise AssertionError("Sampling has values out of range")

    # Shuffle must only sort elements
    # It takes two instances with same random_state, to show that only
    # the final order is mixed
    def is_sorted(a):
      return np.all(a[:-1] <= a[1:])

    oasis_a = OASIS(random_state=SEED)
    r_a = oasis_a._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                      shuffle=False, random=False)
    assert is_sorted(r_a)  # Its not shuffled
    values_r_a, counts_r_a = np.unique(r_a, return_counts=True)

    oasis_b = OASIS(random_state=SEED)
    r_b = oasis_b._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                      shuffle=True, random=False)

    with assert_raises(AssertionError):
      assert is_sorted(r_b)  # This one should not besorted, but shuffled
    values_r_b, counts_r_b = np.unique(r_b, return_counts=True)

    assert_array_equal(values_r_a, values_r_b)  # Same elements
    assert_array_equal(counts_r_a, counts_r_b)  # Same counts
    with assert_raises(AssertionError):
      assert_array_equal(r_a, r_b)  # Diferent order

  # Random case
  r = oasis._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                random=True)
  assert len(r) == n_iter  # Expected lenght
  for i in range(n_iter):
    if r[i] not in base:
      raise AssertionError("Sampling has values out of range")
  # Shuffle has no effect
  oasis_a = OASIS(random_state=SEED)
  r_a = oasis_a._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                    shuffle=False, random=True)

  oasis_b = OASIS(random_state=SEED)
  r_b = oasis_b._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                    shuffle=True, random=True)
  assert_array_equal(r_a, r_b)

  # n_triplets and n_iter cannot be 0
  msg = ("n_triplets cannot be 0")
  with pytest.raises(ValueError) as raised_error:
    oasis._get_random_indices(n_triplets=0, n_iter=n_iter, random=True)
  assert msg == raised_error.value.args[0]

  msg = ("n_iter cannot be 0")
  with pytest.raises(ValueError) as raised_error:
    oasis._get_random_indices(n_triplets=n_triplets, n_iter=0, random=True)
  assert msg == raised_error.value.args[0]


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


@pytest.mark.parametrize('custom_M', ["identity", "random", "spd"])
@pytest.mark.parametrize('random_state', [33, 69, 112])
def test_random_state_in_suffling(custom_M, random_state):
  """
  Tests that many instances of OASIS, with the same random_state,
  produce the same shuffling on the triplets given.

  Test that many instances of OASIS, with different random_state,
  produce different shuffling on the trilpets given.

  The triplets are produced with the Iris dataset.

  Tested with all possible custom_M.
  """
  triplets = gen_iris_triplets()

  # Test same random_state, then same shuffling
  oasis_a = OASIS(random_state=random_state, custom_M=custom_M)
  oasis_a.fit(triplets)
  shuffle_a = oasis_a.get_indices()

  oasis_b = OASIS(random_state=random_state, custom_M=custom_M)
  oasis_b.fit(triplets)
  shuffle_b = oasis_b.get_indices()

  assert_array_equal(shuffle_a, shuffle_b)

  # Test different random states
  last_suffle = shuffle_b
  for i in range(3, 5):
    oasis_a = OASIS(random_state=random_state+i, custom_M=custom_M)
    oasis_a.fit(triplets)
    shuffle_a = oasis_a.get_indices()

    with pytest.raises(AssertionError):
      assert_array_equal(last_suffle, shuffle_a)

    last_suffle = shuffle_a


@pytest.mark.parametrize('custom_M', ["identity", "random", "spd"])
@pytest.mark.parametrize('random_state', [33, 69, 112])
def test_general_results_random_state(custom_M, random_state):
  """
  With fixed triplets and random_state, two instances of OASIS
  should produce the same output (matrix W)
  """
  triplets = gen_iris_triplets()
  oasis_a = OASIS(random_state=random_state, custom_M=custom_M)
  oasis_a.fit(triplets)
  matrix_a = oasis_a.get_bilinear_matrix()

  oasis_b = OASIS(random_state=random_state, custom_M=custom_M)
  oasis_b.fit(triplets)
  matrix_b = oasis_b.get_bilinear_matrix()

  assert_array_equal(matrix_a, matrix_b)


@pytest.mark.parametrize('custom_M', ["random", "spd"])
@pytest.mark.parametrize('random_state', [6, 42])
@pytest.mark.parametrize('d', [23, 27])
def test_random_state_random_base_M(custom_M, random_state, d):
  """
  Tests that the function _check_M outputs the same matrix,
  given the same random_state to OASIS instace, with a fixed d.
  """
  oasis_a = OASIS(random_state=random_state)
  oasis_a.d = d
  matrix_a = oasis_a._check_M(custom_M=custom_M)

  oasis_b = OASIS(random_state=random_state)
  oasis_b.d = d
  matrix_b = oasis_b._check_M(custom_M=custom_M)

  assert_array_equal(matrix_a, matrix_b)
