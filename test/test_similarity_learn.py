from metric_learn.oasis import OASIS
import numpy as np
from sklearn.utils import check_random_state
import pytest
from numpy.testing import assert_array_equal, assert_raises

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

    # Shuffle must have no efect
    oasis_a = OASIS(random_state=SEED)
    r_a = oasis_a._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                      shuffle=False, random=False)

    oasis_b = OASIS(random_state=SEED)
    r_b = oasis_b._get_random_indices(n_triplets=n_triplets, n_iter=n_iter,
                                      shuffle=True, random=False)
    assert_array_equal(r_a, r_b)

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
