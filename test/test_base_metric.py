import pytest
import unittest
import metric_learn
import numpy as np
from sklearn import clone
from sklearn.utils.testing import set_random_state

from test.test_utils import ids_metric_learners, metric_learners


class TestStringRepr(unittest.TestCase):

  def test_covariance(self):
    self.assertEqual(str(metric_learn.Covariance()),
                     "Covariance(preprocessor=None)")

  def test_lmnn(self):
    self.assertRegexpMatches(
        str(metric_learn.LMNN()),
        r"(python_)?LMNN\(convergence_tol=0.001, k=3, learn_rate=1e-07, "
        r"max_iter=1000,\n      min_iter=50, preprocessor=None, "
        r"regularization=0.5, use_pca=True,\n      verbose=False\)")

  def test_nca(self):
    self.assertEqual(str(metric_learn.NCA()),
                     "NCA(max_iter=100, num_dims=None, preprocessor=None, "
                     "tol=None, verbose=False)")

  def test_lfda(self):
    self.assertEqual(str(metric_learn.LFDA()),
                     "LFDA(embedding_type='weighted', k=None, num_dims=None, "
                     "preprocessor=None)")

  def test_itml(self):
    self.assertEqual(str(metric_learn.ITML()), """
ITML(A0=None, convergence_threshold=0.001, gamma=1.0, max_iter=1000,
   preprocessor=None, verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.ITML_Supervised()), """
ITML_Supervised(A0=None, bounds='deprecated', convergence_threshold=0.001,
        gamma=1.0, max_iter=1000, num_constraints=None,
        num_labeled='deprecated', preprocessor=None, verbose=False)
""".strip('\n'))

  def test_lsml(self):
    self.assertEqual(
        str(metric_learn.LSML()),
        "LSML(max_iter=1000, preprocessor=None, prior=None, tol=0.001, "
        "verbose=False)")
    self.assertEqual(str(metric_learn.LSML_Supervised()), """
LSML_Supervised(max_iter=1000, num_constraints=None, num_labeled='deprecated',
        preprocessor=None, prior=None, tol=0.001, verbose=False,
        weights=None)
""".strip('\n'))

  def test_sdml(self):
    self.assertEqual(str(metric_learn.SDML()),
                     "SDML(balance_param=0.5, preprocessor=None, "
                     "sparsity_param=0.01, use_cov=True,\n   "
                     "verbose=False)")
    self.assertEqual(str(metric_learn.SDML_Supervised()), """
SDML_Supervised(balance_param=0.5, num_constraints=None,
        num_labeled='deprecated', preprocessor=None, sparsity_param=0.01,
        use_cov=True, verbose=False)
""".strip('\n'))

  def test_rca(self):
    self.assertEqual(str(metric_learn.RCA()),
                     "RCA(num_dims=None, pca_comps=None, preprocessor=None)")
    self.assertEqual(str(metric_learn.RCA_Supervised()),
                     "RCA_Supervised(chunk_size=2, num_chunks=100, "
                     "num_dims=None, pca_comps=None,\n        "
                     "preprocessor=None)")

  def test_mlkr(self):
    self.assertEqual(str(metric_learn.MLKR()),
                     "MLKR(A0=None, max_iter=1000, num_dims=None, "
                     "preprocessor=None, tol=None,\n   verbose=False)")

  def test_mmc(self):
    self.assertEqual(str(metric_learn.MMC()), """
MMC(A0=None, convergence_threshold=0.001, diagonal=False, diagonal_c=1.0,
  max_iter=100, max_proj=10000, preprocessor=None, verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.MMC_Supervised()), """
MMC_Supervised(A0=None, convergence_threshold=1e-06, diagonal=False,
        diagonal_c=1.0, max_iter=100, max_proj=10000, num_constraints=None,
        num_labeled='deprecated', preprocessor=None, verbose=False)
""".strip('\n'))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_is_independent_from_metric_learner(estimator,
                                                       build_dataset):
  """Tests that the get_metric method returns a function that is independent
  from the original metric learner"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)

  # we fit the metric learner on it and then we compute the metric on some
  # points
  model.fit(input_data, labels)
  metric = model.get_metric()
  score = metric(X[0], X[1])

  # then we refit the estimator on another dataset
  model.fit(np.sin(input_data), labels)

  # we recompute the distance between the two points: it should be the same
  score_bis = metric(X[0], X[1])
  assert score_bis == score


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_raises_error(estimator, build_dataset):
  """Tests that the metric returned by get_metric raises errors similar to
  the distance functions in scipy.spatial.distance"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  metric = model.get_metric()

  list_test_get_metric_raises = [(X[0].tolist() + [5.2], X[1]),  # vectors with
                                 # different dimensions
                                 (X[0:4], X[1:5]),  # 2D vectors
                                 (X[0].tolist() + [5.2], X[1] + [7.2])]
  # vectors of same dimension but incompatible with what the metric learner
  # was trained on

  for u, v in list_test_get_metric_raises:
    with pytest.raises(ValueError):
      metric(u, v)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_works_does_not_raise(estimator, build_dataset):
  """Tests that the metric returned by get_metric does not raise errors (or
  warnings) similarly to the distance functions in scipy.spatial.distance"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  metric = model.get_metric()

  list_test_get_metric_doesnt_raise = [(X[0], X[1]),
                                       (X[0].tolist(), X[1].tolist()),
                                       (X[0][None], X[1][None])]

  for u, v in list_test_get_metric_doesnt_raise:
    with pytest.warns(None) as record:
      metric(u, v)
    assert len(record) == 0

  # Test that the scalar case works
  model.transformer_ = np.array([3.1])
  metric = model.get_metric()
  for u, v in [(5, 6.7), ([5], [6.7]), ([[5]], [[6.7]])]:
    with pytest.warns(None) as record:
      metric(u, v)
    assert len(record) == 0


if __name__ == '__main__':
  unittest.main()
