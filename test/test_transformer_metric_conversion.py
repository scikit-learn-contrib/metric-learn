import unittest
import numpy as np
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)


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
    sdml = SDML_Supervised(num_constraints=1500)
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


if __name__ == '__main__':
  unittest.main()
