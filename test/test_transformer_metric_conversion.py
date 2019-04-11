import unittest
import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.stats import ortho_group
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal, assert_allclose
from sklearn.utils.testing import ignore_warnings

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)
from metric_learn._util import transformer_from_metric


class TestTransformerMetricConversion(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.X = iris_data['data']
    self.y = iris_data['target']

  def test_cov(self):
    cov = Covariance()
    cov.fit(self.X)
    L = cov.transformer_
    assert_array_almost_equal(L.T.dot(L), cov.get_mahalanobis_matrix())

  def test_lsml_supervised(self):
    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.X, self.y, random_state=seed)
    L = lsml.transformer_
    assert_array_almost_equal(L.T.dot(L), lsml.get_mahalanobis_matrix())

  def test_itml_supervised(self):
    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.X, self.y, random_state=seed)
    L = itml.transformer_
    assert_array_almost_equal(L.T.dot(L), itml.get_mahalanobis_matrix())

  def test_lmnn(self):
    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    L = lmnn.transformer_
    assert_array_almost_equal(L.T.dot(L), lmnn.get_mahalanobis_matrix())

  def test_sdml_supervised(self):
    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500, use_cov=False,
                           balance_param=1e-5)
    sdml.fit(self.X, self.y, random_state=seed)
    L = sdml.transformer_
    assert_array_almost_equal(L.T.dot(L), sdml.get_mahalanobis_matrix())

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000//n))
    nca.fit(self.X, self.y)
    L = nca.transformer_
    assert_array_almost_equal(L.T.dot(L), nca.get_mahalanobis_matrix())

  def test_lfda(self):
    lfda = LFDA(k=2, num_dims=2)
    lfda.fit(self.X, self.y)
    L = lfda.transformer_
    assert_array_almost_equal(L.T.dot(L), lfda.get_mahalanobis_matrix())

  def test_rca_supervised(self):
    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(num_dims=2, num_chunks=30, chunk_size=2)
    rca.fit(self.X, self.y, random_state=seed)
    L = rca.transformer_
    assert_array_almost_equal(L.T.dot(L), rca.get_mahalanobis_matrix())

  def test_mlkr(self):
    mlkr = MLKR(num_dims=2)
    mlkr.fit(self.X, self.y)
    L = mlkr.transformer_
    assert_array_almost_equal(L.T.dot(L), mlkr.get_mahalanobis_matrix())

  @ignore_warnings
  def test_transformer_from_metric_edge_cases(self):
    """Test that transformer_from_metric returns the right result in various
    edge cases"""
    rng = np.random.RandomState(42)

    # an orthonormal matrix useful for creating matrices with given
    # eigenvalues:
    P = ortho_group.rvs(7, random_state=rng)

    # matrix with all its coefficients very low (to check that the algorithm
    # does not consider it as a diagonal matrix)(non regression test for
    # https://github.com/metric-learn/metric-learn/issues/175)
    M = np.diag([1e-15, 2e-16, 3e-15, 4e-16, 5e-15, 6e-16, 7e-15])
    M = P.dot(M).dot(P.T)
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # diagonal matrix
    M = np.diag(np.abs(rng.randn(5)))
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # low-rank matrix (with zeros)
    M = np.zeros((7, 7))
    small_random = rng.randn(3, 3)
    M[:3, :3] = small_random.T.dot(small_random)
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # low-rank matrix (without necessarily zeros)
    R = np.abs(rng.randn(7, 7))
    M = R.dot(np.diag([1, 5, 3, 2, 0, 0, 0])).dot(R.T)
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # matrix with a determinant still high but which should be considered as a
    # non-definite matrix (to check we don't test the definiteness with the
    # determinant which is a bad strategy)
    M = np.diag([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e-20])
    M = P.dot(M).dot(P.T)
    assert np.abs(np.linalg.det(M)) > 10
    assert np.linalg.slogdet(M)[1] > 1  # (just to show that the computed
    # determinant is far from null)
    with pytest.raises(LinAlgError) as err_msg:
      np.linalg.cholesky(M)
    assert str(err_msg.value) == 'Matrix is not positive definite'
    # (just to show that this case is indeed considered by numpy as an
    # indefinite case)
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # matrix with lots of small nonzeros that make a big zero when multiplied
    M = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

    # full rank matrix
    M = rng.randn(10, 10)
    M = M.T.dot(M)
    assert np.linalg.matrix_rank(M) == 10
    L = transformer_from_metric(M)
    assert_allclose(L.T.dot(L), M)

  def test_non_symmetric_matrix_raises(self):
    """Checks that if a non symmetric matrix is given to
    transformer_from_metric, an error is thrown"""
    rng = np.random.RandomState(42)
    M = rng.randn(10, 10)
    with pytest.raises(ValueError) as raised_error:
      transformer_from_metric(M)
    assert str(raised_error.value) == "The input metric should be symmetric."

  def test_non_psd_warns(self):
    """Checks that if the matrix is not PSD it will raise a warning saying
    that we will do the eigendecomposition"""
    rng = np.random.RandomState(42)
    R = np.abs(rng.randn(7, 7))
    M = R.dot(np.diag([1, 5, 3, 2, 0, 0, 0])).dot(R.T)
    msg = ("The Cholesky decomposition returned the following "
           "error: 'Matrix is not positive definite'. Using the "
           "eigendecomposition instead.")
    with pytest.warns(None) as raised_warning:
      L = transformer_from_metric(M)
    assert str(raised_warning[0].message) == msg
    assert_allclose(L.T.dot(L), M)


if __name__ == '__main__':
  unittest.main()
