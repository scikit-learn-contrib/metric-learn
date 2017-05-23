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
    L = cov.transformer()
    assert_array_almost_equal(L.T.dot(L), cov.metric())

  def test_lsml_supervised(self):
    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.X, self.y, random_state=seed)
    L = lsml.transformer()
    assert_array_almost_equal(L.T.dot(L), lsml.metric())

  def test_itml_supervised(self):
    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.X, self.y, random_state=seed)
    L = itml.transformer()
    assert_array_almost_equal(L.T.dot(L), itml.metric())

  def test_lmnn(self):
    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    L = lmnn.transformer()
    assert_array_almost_equal(L.T.dot(L), lmnn.metric())

  def test_sdml_supervised(self):
    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500)
    sdml.fit(self.X, self.y, random_state=seed)
    L = sdml.transformer()
    assert_array_almost_equal(L.T.dot(L), sdml.metric())

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    nca.fit(self.X, self.y)
    L = nca.transformer()
    assert_array_almost_equal(L.T.dot(L), nca.metric())

  def test_lfda(self):
    lfda = LFDA(k=2, num_dims=2)
    lfda.fit(self.X, self.y)
    L = lfda.transformer()
    assert_array_almost_equal(L.T.dot(L), lfda.metric())

  def test_rca_supervised(self):
    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(num_dims=2, num_chunks=30, chunk_size=2)
    rca.fit(self.X, self.y, random_state=seed)
    L = rca.transformer()
    assert_array_almost_equal(L.T.dot(L), rca.metric())

  def test_mlkr(self):
    mlkr = MLKR(num_dims=2)
    mlkr.fit(self.X, self.y)
    L = mlkr.transformer()
    assert_array_almost_equal(L.T.dot(L), mlkr.metric())


if __name__ == '__main__':
  unittest.main()
