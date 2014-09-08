import unittest
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris

from metric_learn import LSML, ITML, LMNN, SDML
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


class TestLSML(MetricTestCase):
  def test_iris(self):
    num_constraints = 200

    C = LSML.prepare_constraints(self.iris_labels, num_constraints)
    lsml = LSML(self.iris_points, C).fit(verbose=False)

    csep = class_separation(lsml.transform(), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible


class TestITML(MetricTestCase):
  def test_iris(self):
    num_constraints = 200

    n = self.iris_points.shape[0]
    C = ITML.prepare_constraints(self.iris_labels, n, num_constraints)
    itml = ITML().fit(self.iris_points, C, verbose=False)

    csep = class_separation(itml.transform(), self.iris_labels)
    self.assertLess(csep, 0.4)  # it's not great


class TestLMNN(MetricTestCase):
  def test_iris(self):
    k = 5

    # Test both impls, if available.
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=k, learn_rate=1e-6)
      lmnn.fit(self.iris_points, self.iris_labels, verbose=False)

      csep = class_separation(lmnn.transform(), self.iris_labels)
      self.assertLess(csep, 0.25)


class TestSDML(MetricTestCase):
  def test_iris(self):
    num_constraints = 1500

    n = self.iris_points.shape[0]
    W = SDML.prepare_constraints(self.iris_labels, n, num_constraints)
    sdml = SDML(self.iris_points, W).fit()

    csep = class_separation(sdml.transform(), self.iris_labels)
    self.assertLess(csep, 0.25)


if __name__ == '__main__':
  unittest.main()
