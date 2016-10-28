import unittest
import numpy as np
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)


class TestFitTransform(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.X = iris_data['data']
    self.y = iris_data['target']

  def test_cov(self):
    cov = Covariance()
    cov.fit(self.X)
    res_1 = cov.transform()

    cov = Covariance()
    res_2 = cov.fit_transform(self.X)
    # deterministic result
    assert_array_almost_equal(res_1, res_2)

  def test_lsml_supervised(self):
    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.X, self.y, random_state=seed)
    res_1 = lsml.transform()

    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    res_2 = lsml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_itml_supervised(self):
    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.X, self.y, random_state=seed)
    res_1 = itml.transform()

    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    res_2 = itml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_lmnn(self):
    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.X, self.y)
    res_1 = lmnn.transform()

    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    res_2 = lmnn.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_sdml_supervised(self):
    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500)
    sdml.fit(self.X, self.y, random_state=seed)
    res_1 = sdml.transform()

    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500)
    res_2 = sdml.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_nca(self):
    n = self.X.shape[0]
    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    nca.fit(self.X, self.y)
    res_1 = nca.transform()

    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    res_2 = nca.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)

  def test_lfda(self):
    lfda = LFDA(k=2, dim=2)
    lfda.fit(self.X, self.y)
    res_1 = lfda.transform()

    lfda = LFDA(k=2, dim=2)
    res_2 = lfda.fit_transform(self.X, self.y)

    # signs may be flipped, that's okay
    if np.sign(res_1[0,0]) != np.sign(res_2[0,0]):
        res_2 *= -1
    assert_array_almost_equal(res_1, res_2)

  def test_rca_supervised(self):
    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(dim=2, num_chunks=30, chunk_size=2)
    rca.fit(self.X, self.y, random_state=seed)
    res_1 = rca.transform()

    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(dim=2, num_chunks=30, chunk_size=2)
    res_2 = rca.fit_transform(self.X, self.y, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

  def test_mlkr(self):
    mlkr = MLKR(num_dims=2)
    mlkr.fit(self.X, self.y)
    res_1 = mlkr.transform()

    mlkr = MLKR(num_dims=2)
    res_2 = mlkr.fit_transform(self.X, self.y)

    assert_array_almost_equal(res_1, res_2)


if __name__ == '__main__':
  unittest.main()
