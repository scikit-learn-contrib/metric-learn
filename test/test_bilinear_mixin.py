from itertools import product
from metric_learn.base_metric import BilinearMixin
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from metric_learn._util import make_context
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state

RNG = check_random_state(0)


class IdentityBilinearMixin(BilinearMixin):
  """A simple Identity bilinear mixin that returns an identity matrix
  M as learned. Can change M for a random matrix specifying random=True
  at fit(). Class for testing purposes.
  """
  def __init__(self, preprocessor=None):
    super().__init__(preprocessor=preprocessor)

  def fit(self, X, y, random=False):
    """
    Checks input's format. If random=False, sets M matrix to
    identity of shape (d,d) where d is the dimension of the input.
    Otherwise, a random (d,d) matrix is set.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d = np.shape(X[0])[-1]
    if random:
      self.components_ = np.random.rand(self.d, self.d)
    else:
      self.components_ = np.identity(self.d)
    return self


def identity_fit(d=100, n=100, n_pairs=None, random=False):
  """
  Creates 'n' d-dimentional arrays. Also generates 'n_pairs'
  sampled from the 'n' arrays. Fits an IdentityBilinearMixin()
  and then returns the arrays, the pairs and the mixin. Only
  generates the pairs if n_pairs is not None. If random=True,
  the matrix M fitted will be random.
  """
  X = np.array([np.random.rand(d) for _ in range(n)])
  mixin = IdentityBilinearMixin()
  mixin.fit(X, [0 for _ in range(n)], random=random)
  if n_pairs is not None:
    random_pairs = [[X[RNG.randint(0, n)], X[RNG.randint(0, n)]]
                    for _ in range(n_pairs)]
  else:
    random_pairs = None
  return X, random_pairs, mixin


def test_same_similarity_with_two_methods():
  """"
  Tests that score_pairs() and get_metric() give consistent results.
  In both cases, the results must match for the same input.
  Tests it for 'n_pairs' sampled from 'n' d-dimentional arrays.
  """
  d, n, n_pairs = 100, 100, 1000
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.score_pairs(random_pairs)
  dist2 = [mixin.get_metric()(p[0], p[1]) for p in random_pairs]

  assert_array_almost_equal(dist1, dist2)


def test_check_correctness_similarity():
  """
  Tests the correctness of the results made from socre_paris() and
  get_metric(). Results are compared with the real bilinear similarity
  calculated in-place.
  """
  d, n, n_pairs = 100, 100, 1000
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.score_pairs(random_pairs)
  dist2 = [mixin.get_metric()(p[0], p[1]) for p in random_pairs]
  desired = [np.dot(np.dot(p[0].T, mixin.components_), p[1])
             for p in random_pairs]

  assert_array_almost_equal(dist1, desired)  # score_pairs
  assert_array_almost_equal(dist2, desired)  # get_metric


def test_check_handmade_example():
  """
  Checks that score_pairs() result is correct comparing it with a
  handmade example.
  """
  u = np.array([0, 1, 2])
  v = np.array([3, 4, 5])
  mixin = IdentityBilinearMixin()
  mixin.fit([u, v], [0, 0])  # Identity fit
  c = np.array([[2, 4, 6], [6, 4, 2], [1, 2, 3]])
  mixin.components_ = c  # Force components_
  dists = mixin.score_pairs([[u, v], [v, u]])
  assert_array_almost_equal(dists, [96, 120])


def test_check_handmade_symmetric_example():
  """
  When the Bilinear matrix is the identity. The similarity
  between two arrays must be equal: S(u,v) = S(v,u). Also
  checks the random case: when the matrix is spd and symetric.
  """
  # Random pairs for M = Identity
  d, n, n_pairs = 100, 100, 1000
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs)
  pairs_reverse = [[p[1], p[0]] for p in random_pairs]
  dist1 = mixin.score_pairs(random_pairs)
  dist2 = mixin.score_pairs(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)

  # Random pairs for M = spd Matrix
  spd_matrix = make_spd_matrix(d, random_state=RNG)
  mixin.components_ = spd_matrix
  dist1 = mixin.score_pairs(random_pairs)
  dist2 = mixin.score_pairs(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)


def test_score_pairs_finite():
  """
  Checks for 'n' score_pairs() of 'd' dimentions, that all
  similarities are finite numbers: not NaN, +inf or -inf.
  Considers a random M for bilinear similarity.
  """
  d, n, n_pairs = 100, 100, 1000
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.score_pairs(random_pairs)
  assert np.isfinite(dist1).all()


def test_score_pairs_dim():
  """
  Scoring of 3D arrays should return 1D array (several tuples),
  and scoring of 2D arrays (one tuple) should return an error (like
  scikit-learn's error when scoring 1D arrays)
  """
  d, n = 100, 100
  X, _, mixin = identity_fit(d=d, n=n, n_pairs=None, random=True)
  tuples = np.array(list(product(X, X)))
  assert mixin.score_pairs(tuples).shape == (tuples.shape[0],)
  context = make_context(mixin)
  msg = ("3D array of formed tuples expected{}. Found 2D array "
         "instead:\ninput={}. Reshape your data and/or use a preprocessor.\n"
         .format(context, tuples[1]))
  with pytest.raises(ValueError) as raised_error:
    mixin.score_pairs(tuples[1])
  assert str(raised_error.value) == msg


def test_check_scikitlearn_compatibility():
  """Check that the similarity returned by get_metric() is compatible with
  scikit-learn's algorithms using a custom metric, DBSCAN for instance"""
  d, n = 100, 100
  X, _, mixin = identity_fit(d=d, n=n, n_pairs=None, random=True)
  clustering = DBSCAN(metric=mixin.get_metric())
  clustering.fit(X)
