from itertools import product
from metric_learn.base_metric import BilinearMixin
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from metric_learn._util import make_context
from sklearn.cluster import DBSCAN


class IdentityBilinearMixin(BilinearMixin):
  """A simple Identity bilinear mixin that returns an identity matrix
  M as learned. Can change M for a random matrix calling random_M.
  Class for testing purposes.
  """
  def __init__(self, preprocessor=None):
    super().__init__(preprocessor=preprocessor)

  def fit(self, X, y):
    """
    Checks input's format. Sets M matrix to identity of shape (d,d)
    where d is the dimension of the input.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d = np.shape(X[0])[-1]
    self.components_ = np.identity(self.d)
    return self

  def random_M(self):
    """
    Changes the matrix M for a random one of shape (d,d)
    """
    self.components_ = np.random.rand(self.d, self.d)


def identity_fit(d=100):
  """
  Creates two d-dimentional arrays. Fits an IdentityBilinearMixin()
  and then returns the two arrays and the mixin. Testing purposes
  """
  d = 100
  u = np.random.rand(d)
  v = np.random.rand(d)
  mixin = IdentityBilinearMixin()
  mixin.fit([u, v], [0, 0])
  return u, v, mixin


def test_same_similarity_with_two_methods():
  """"
  Tests that score_pairs() and get_metric() give consistent results.
  In both cases, the results must match for the same input.
  """
  u, v, mixin = identity_fit()
  mixin.random_M()  # Dummy fit
  # The distances must match, whether calc with get_metric() or score_pairs()
  dist1 = mixin.score_pairs([[u, v], [v, u]])
  dist2 = [mixin.get_metric()(u, v), mixin.get_metric()(v, u)]

  assert_array_almost_equal(dist1, dist2)


def test_check_correctness_similarity():
  """
  Tests the correctness of the results made from socre_paris() and
  get_metric(). Results are compared with the real bilinear similarity
  calculated in-place.
  """
  d = 100
  u, v, mixin = identity_fit(d)
  dist1 = mixin.score_pairs([[u, v], [v, u]])
  dist2 = [mixin.get_metric()(u, v), mixin.get_metric()(v, u)]

  u_v = np.dot(np.dot(u.T, np.identity(d)), v)
  v_u = np.dot(np.dot(v.T, np.identity(d)), u)
  desired = [u_v, v_u]
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
  checks the random case: when the matrix is pd and symetric.
  """
  u = np.array([0, 1, 2])
  v = np.array([3, 4, 5])
  mixin = IdentityBilinearMixin()
  mixin.fit([u, v], [0, 0])   # Identity fit
  dists = mixin.score_pairs([[u, v], [v, u]])
  assert_array_almost_equal(dists, [14, 14])


def test_score_pairs_finite():
  """
  Checks for 'n' score_pairs() of 'd' dimentions, that all
  similarities are finite numbers, not NaN, +inf or -inf.
  Considering a random M for bilinear similarity.
  """
  d = 100
  u, v, mixin = identity_fit(d)
  mixin.random_M()  # Dummy fit
  n = 100
  X = np.array([np.random.rand(d) for i in range(n)])
  pairs = np.array(list(product(X, X)))
  assert np.isfinite(mixin.score_pairs(pairs)).all()


def test_score_pairs_dim():
  """
  Scoring of 3D arrays should return 1D array (several tuples),
  and scoring of 2D arrays (one tuple) should return an error (like
  scikit-learn's error when scoring 1D arrays)
  """
  d = 100
  u, v, mixin = identity_fit()
  mixin.random_M()  # Dummy fit
  n = 100
  X = np.array([np.random.rand(d) for i in range(n)])
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
  d = 100
  u, v, mixin = identity_fit(d)
  mixin.random_M()  # Dummy fit

  n = 100
  X = np.array([np.random.rand(d) for i in range(n)])
  clustering = DBSCAN(metric=mixin.get_metric())
  clustering.fit(X)
