import unittest
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris
from numpy.testing import assert_array_almost_equal

from metric_learn import (
    LMNN, NCA, LFDA,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)
# Import this specially for testing.
from metric_learn.lmnn import python_LMNN


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in xrange(len(unique_labels)):
    Xc = X[label_inds==li]
    Xnc = X[label_inds!=li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
  return ratio / len(unique_labels)


class MetricTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.iris_points = iris_data['data']
    self.iris_labels = iris_data['target']
    np.random.seed(1234)


class TestLSML(MetricTestCase):
  def test_iris(self):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lsml.transform(), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible


class TestITML(MetricTestCase):
  def test_iris(self):
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(itml.transform(), self.iris_labels)
    self.assertLess(csep, 0.4)  # it's not great


class TestLMNN(MetricTestCase):
  def test_iris(self):
    # Test both impls, if available.
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=5, learn_rate=1e-6, verbose=False)
      lmnn.fit(self.iris_points, self.iris_labels)

      csep = class_separation(lmnn.transform(), self.iris_labels)
      self.assertLess(csep, 0.25)


class TestSDML(MetricTestCase):
  def test_iris(self):
    # Note: this is a flaky test, which fails for certain seeds.
    # TODO: un-flake it!
    np.random.seed(5555)

    sdml = SDML_Supervised(num_constraints=1500)
    sdml.fit(self.iris_points, self.iris_labels)
    csep = class_separation(sdml.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)


class TestNCA(MetricTestCase):
  def test_iris(self):
    n = self.iris_points.shape[0]
    nca = NCA(max_iter=(100000//n), learning_rate=0.01)
    nca.fit(self.iris_points, self.iris_labels)

    # Result copied from Iris example at
    # https://github.com/vomjom/nca/blob/master/README.mkd
    expected = [[-0.09935, -0.2215,  0.3383,  0.443],
                [+0.2532,   0.5835, -0.8461, -0.8915],
                [-0.729,   -0.6386,  1.767,   1.832],
                [-0.9405,  -0.8461,  2.281,   2.794]]
    assert_array_almost_equal(expected, nca.transformer(), decimal=3)


class TestLFDA(MetricTestCase):
  def test_iris(self):
    lfda = LFDA(k=2, dim=2)
    lfda.fit(self.iris_points, self.iris_labels)
    csep = class_separation(lfda.transform(), self.iris_labels)
    self.assertLess(csep, 0.15)


class TestRCA(MetricTestCase):
  def test_iris(self):
    rca = RCA_Supervised(dim=2, num_chunks=30, chunk_size=2, seed=1234)
    rca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(rca.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)


if __name__ == '__main__':
  unittest.main()
