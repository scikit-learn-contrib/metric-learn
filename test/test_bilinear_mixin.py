"""
Tests all functionality for Bilinear learners. Correctness, use cases,
warnings, etc.
"""
from itertools import product
from metric_learn.base_metric import BilinearMixin
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from metric_learn._util import make_context
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state
from metric_learn.base_metric import _PairsClassifierMixin, \
  _TripletsClassifierMixin, _QuadrupletsClassifierMixin

RNG = check_random_state(0)


class RandomBilinearLearner(BilinearMixin):
  """A simple Random bilinear mixin that returns an random matrix
  M as learned. Class for testing purposes.
  """
  def __init__(self, preprocessor=None, random_state=33):
    super().__init__(preprocessor=preprocessor)
    self.random_state = random_state

  def fit(self, X, y):
    """
    Checks input's format. A random (d,d) matrix is set.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d_ = np.shape(X[0])[-1]
    rng = check_random_state(self.random_state)
    self.components_ = rng.rand(self.d_, self.d_)
    return self


class IdentityBilinearLearner(BilinearMixin):
  """A simple Identity bilinear mixin that returns an identity matrix
  M as learned. Class for testing purposes.
  """
  def __init__(self, preprocessor=None):
    super().__init__(preprocessor=preprocessor)

  def fit(self, X, y):
    """
    Checks input's format. Sets M matrix to identity of shape (d,d)
    where d is the dimension of the input.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.d_ = np.shape(X[0])[-1]
    self.components_ = np.identity(self.d_)
    return self


class MockPairIdentityBilinearLearner(BilinearMixin,
                                      _PairsClassifierMixin):

  def __init__(self, preprocessor=None):
      super().__init__(preprocessor=preprocessor)

  def fit(self, pairs, y, calibration_params=None):
    calibration_params = (calibration_params if calibration_params is not
                          None else dict())
    self._validate_calibration_params(**calibration_params)
    pairs = self._prepare_inputs(pairs, type_of_inputs='tuples')
    self.d_ = np.shape(pairs[0][0])[-1]
    self.components_ = np.identity(self.d_)
    self.calibrate_threshold(pairs, y, **calibration_params)
    return self


class MockTripletsIdentityBilinearLearner(BilinearMixin,
                                          _TripletsClassifierMixin):

  def __init__(self, preprocessor=None):
      super().__init__(preprocessor=preprocessor)

  def fit(self, triplets):
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')
    self.d_ = np.shape(triplets[0][0])[-1]
    self.components_ = np.identity(self.d_)
    return self


class MockQuadrpletsIdentityBilinearLearner(BilinearMixin,
                                            _QuadrupletsClassifierMixin):

  def __init__(self, preprocessor=None):
      super().__init__(preprocessor=preprocessor)

  def fit(self, quadruplets):
    quadruplets = self._prepare_inputs(quadruplets, type_of_inputs='tuples')
    self.d_ = np.shape(quadruplets[0][0])[-1]
    self.components_ = np.identity(self.d_)
    return self


def identity_fit(d=100, n=100, n_pairs=None, random=False):
  """
  Creates 'n' d-dimentional arrays. Also generates 'n_pairs'
  sampled from the 'n' arrays. Fits an IdentityBilinearLearner()
  and then returns the arrays, the pairs and the mixin. Only
  generates the pairs if n_pairs is not None. If random=True,
  the matrix M fitted will be random.
  """
  X = np.array([np.random.rand(d) for _ in range(n)])
  mixin = IdentityBilinearLearner()
  mixin.fit(X, [0 for _ in range(n)])
  if n_pairs is not None:
    random_pairs = [[X[RNG.randint(0, n)], X[RNG.randint(0, n)]]
                    for _ in range(n_pairs)]
  else:
    random_pairs = None
  return X, random_pairs, mixin


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_same_similarity_with_two_methods(d, n, n_pairs):
  """"
  Tests that pair_score() and get_metric() give consistent results.
  In both cases, the results must match for the same input.
  Tests it for 'n_pairs' sampled from 'n' d-dimentional arrays.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.pair_score(random_pairs)
  dist2 = [mixin.get_metric()(p[0], p[1]) for p in random_pairs]

  assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_check_correctness_similarity(d, n, n_pairs):
  """
  Tests the correctness of the results made from socre_paris(),
  get_metric() and get_bilinear_matrix. Results are compared with
  the real bilinear similarity calculated in-place.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.pair_score(random_pairs)
  dist2 = [mixin.get_metric()(p[0], p[1]) for p in random_pairs]
  dist3 = [np.dot(np.dot(p[0].T, mixin.get_bilinear_matrix()), p[1])
           for p in random_pairs]
  desired = [np.dot(np.dot(p[0].T, mixin.components_), p[1])
             for p in random_pairs]

  assert_array_almost_equal(dist1, desired)  # pair_score
  assert_array_almost_equal(dist2, desired)  # get_metric
  assert_array_almost_equal(dist3, desired)  # get_metric


def test_check_handmade_example():
  """
  Checks that pair_score() result is correct comparing it with a
  handmade example.
  """
  u = np.array([0, 1, 2])
  v = np.array([3, 4, 5])
  mixin = IdentityBilinearLearner()
  mixin.fit([u, v], [0, 0])  # Identity fit
  c = np.array([[2, 4, 6], [6, 4, 2], [1, 2, 3]])
  mixin.components_ = c  # Force components_
  dists = mixin.pair_score([[u, v], [v, u]])
  assert_array_almost_equal(dists, [96, 120])


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_check_handmade_symmetric_example(d, n, n_pairs):
  """
  When the Bilinear matrix is the identity. The similarity
  between two arrays must be equal: S(u,v) = S(v,u). Also
  checks the random case: when the matrix is spd and symetric.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs)
  pairs_reverse = [[p[1], p[0]] for p in random_pairs]
  dist1 = mixin.pair_score(random_pairs)
  dist2 = mixin.pair_score(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)

  # Random pairs for M = spd Matrix
  spd_matrix = make_spd_matrix(d, random_state=RNG)
  mixin.components_ = spd_matrix
  dist1 = mixin.pair_score(random_pairs)
  dist2 = mixin.pair_score(pairs_reverse)
  assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_pair_score_finite(d, n, n_pairs):
  """
  Checks for 'n' pair_score() of 'd' dimentions, that all
  similarities are finite numbers: not NaN, +inf or -inf.
  Considers a random M for bilinear similarity.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dist1 = mixin.pair_score(random_pairs)
  assert np.isfinite(dist1).all()


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
def test_pair_score_dim(d, n):
  """
  Scoring of 3D arrays should return 1D array (several tuples),
  and scoring of 2D arrays (one tuple) should return an error (like
  scikit-learn's error when scoring 1D arrays)
  """
  X, _, mixin = identity_fit(d=d, n=n, n_pairs=None, random=True)
  tuples = np.array(list(product(X, X)))
  assert mixin.pair_score(tuples).shape == (tuples.shape[0],)
  context = make_context(mixin)
  msg = ("3D array of formed tuples expected{}. Found 2D array "
         "instead:\ninput={}. Reshape your data and/or use a preprocessor.\n"
         .format(context, tuples[1]))
  with pytest.raises(ValueError) as raised_error:
    mixin.pair_score(tuples[1])
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
def test_check_scikitlearn_compatibility(d, n):
  """
  Check that the similarity returned by get_metric() is compatible with
  scikit-learn's algorithms using a custom metric, DBSCAN for instance
  """
  X, _, mixin = identity_fit(d=d, n=n, n_pairs=None, random=True)
  clustering = DBSCAN(metric=mixin.get_metric())
  clustering.fit(X)


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_check_score_pairs_deprecation_and_output(d, n, n_pairs):
  """
  Check that calling score_pairs shows a warning of deprecation, and also
  that the output of score_pairs matches calling pair_score.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  dpr_msg = ("score_pairs will be deprecated in release 0.7.0. "
             "Use pair_score to compute similarity scores, or "
             "pair_distances to compute distances.")
  with pytest.warns(FutureWarning) as raised_warnings:
    s1 = mixin.score_pairs(random_pairs)
    s2 = mixin.pair_score(random_pairs)
    assert_array_almost_equal(s1, s2)
  assert any(str(w.message) == dpr_msg for w in raised_warnings)


@pytest.mark.parametrize('d', [10, 300])
@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('n_pairs', [100, 1000])
def test_check_error_with_pair_distance(d, n, n_pairs):
  """
  Check that calling pair_distance is not possible with a Bilinear learner.
  An Exception must be shown instead.
  """
  _, random_pairs, mixin = identity_fit(d=d, n=n, n_pairs=n_pairs, random=True)
  msg = ("This learner doesn't learn a distance, thus ",
         "this method is not implemented. Use pair_score instead")
  with pytest.raises(Exception) as e:
    _ = mixin.pair_distance(random_pairs)
  assert e.value.args[0] == msg
