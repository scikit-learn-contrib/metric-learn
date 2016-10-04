import unittest
import numpy as np
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN, NCA, LFDA, Covariance,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)



class MetricTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.iris_points = iris_data['data']
    self.iris_labels = iris_data['target']


class TestCovariance(MetricTestCase):
  def test_cov(self):
    cov = Covariance()
    cov.fit(self.iris_points)
    res_1 = cov.transform()

    cov = Covariance()
    res_2 = cov.fit_transform(self.iris_points)
    # deterministic result
    assert_array_almost_equal(res_1, res_2)


class TestLSML(MetricTestCase):
  def test_lsml(self):

    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels, random_state=seed)
    res_1 = lsml.transform()

    seed = np.random.RandomState(1234)
    lsml = LSML_Supervised(num_constraints=200)
    res_2 = lsml.fit_transform(self.iris_points, self.iris_labels, random_state=seed)
    
    assert_array_almost_equal(res_1, res_2)

class TestITML(MetricTestCase):
  def test_itml(self):

    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.iris_points, self.iris_labels, random_state=seed)
    res_1 = itml.transform()

    seed = np.random.RandomState(1234)
    itml = ITML_Supervised(num_constraints=200)
    res_2 = itml.fit_transform(self.iris_points, self.iris_labels, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

class TestLMNN(MetricTestCase):
  def test_lmnn(self):

    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    lmnn.fit(self.iris_points, self.iris_labels)
    res_1 = lmnn.transform()

    lmnn = LMNN(k=5, learn_rate=1e-6, verbose=False)
    res_2 = lmnn.fit_transform(self.iris_points, self.iris_labels)    

    assert_array_almost_equal(res_1, res_2)

class TestSDML(MetricTestCase):
  def test_sdml(self):

    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500)
    sdml.fit(self.iris_points, self.iris_labels, random_state=seed)
    res_1 = sdml.transform()

    seed = np.random.RandomState(1234)
    sdml = SDML_Supervised(num_constraints=1500)
    res_2 = sdml.fit_transform(self.iris_points, self.iris_labels, random_state=seed)

    assert_array_almost_equal(res_1, res_2)

class TestNCA(MetricTestCase):
  def test_nca(self):
    
    n = self.iris_points.shape[0]
    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    nca.fit(self.iris_points, self.iris_labels)
    res_1 = nca.transform()

    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    res_2 = nca.fit_transform(self.iris_points, self.iris_labels)

    assert_array_almost_equal(res_1, res_2)

class TestLFDA(MetricTestCase):
  def test_lfda(self):
    
    lfda = LFDA(k=2, dim=2)
    lfda.fit(self.iris_points, self.iris_labels)
    res_1 = lfda.transform()

    lfda = LFDA(k=2, dim=2)
    res_2 = lfda.fit_transform(self.iris_points, self.iris_labels)

    assert_array_almost_equal(res_1, -(res_2))

class TestRCA(MetricTestCase):
  def test_rca(self):

    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(dim=2, num_chunks=30, chunk_size=2)
    rca.fit(self.iris_points, self.iris_labels, random_state=seed)
    res_1 = rca.transform()

    seed = np.random.RandomState(1234)
    rca = RCA_Supervised(dim=2, num_chunks=30, chunk_size=2)
    res_2 = rca.fit_transform(self.iris_points, self.iris_labels, random_state=seed)

    assert_array_almost_equal(res_1, res_2)


if __name__ == '__main__':
  unittest.main()
