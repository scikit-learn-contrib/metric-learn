import pytest
import re
import unittest
import metric_learn
import numpy as np
from sklearn import clone
from sklearn.utils.testing import set_random_state
from test.test_utils import ids_metric_learners, metric_learners, remove_y


def remove_spaces(s):
  return re.sub(r'\s+', '', s)


class TestStringRepr(unittest.TestCase):

  def test_covariance(self):
    self.assertEqual(remove_spaces(str(metric_learn.Covariance())),
                     remove_spaces("Covariance()"))

  def test_lmnn(self):
    self.assertEqual(
        remove_spaces(str(metric_learn.LMNN(convergence_tol=0.01, k=6))),
        remove_spaces("LMNN(convergence_tol=0.01, k=6)"))

  def test_nca(self):
    self.assertEqual(remove_spaces(str(metric_learn.NCA(max_iter=42))),
                     remove_spaces("NCA(max_iter=42)"))

  def test_lfda(self):
    self.assertEqual(remove_spaces(str(metric_learn.LFDA(k=2))),
                     remove_spaces("LFDA(k=2)"))

  def test_itml(self):
    self.assertEqual(remove_spaces(str(metric_learn.ITML(gamma=0.5))),
                     remove_spaces("ITML(gamma=0.5)"))
    self.assertEqual(
        remove_spaces(str(metric_learn.ITML_Supervised(num_constraints=7))),
        remove_spaces("ITML_Supervised(num_constraints=7)"))

  def test_lsml(self):
    self.assertEqual(remove_spaces(str(metric_learn.LSML(tol=0.1))),
                     remove_spaces("LSML(tol=0.1)"))
    self.assertEqual(
        remove_spaces(str(metric_learn.LSML_Supervised(verbose=True))),
        remove_spaces("LSML_Supervised(verbose=True)"))

  def test_sdml(self):
    self.assertEqual(remove_spaces(str(metric_learn.SDML(verbose=True))),
                     remove_spaces("SDML(verbose=True)"))
    self.assertEqual(
        remove_spaces(str(metric_learn.SDML_Supervised(sparsity_param=0.5))),
        remove_spaces("SDML_Supervised(sparsity_param=0.5)"))

  def test_rca(self):
    self.assertEqual(remove_spaces(str(metric_learn.RCA(n_components=3))),
                     remove_spaces("RCA(n_components=3)"))
    self.assertEqual(
        remove_spaces(str(metric_learn.RCA_Supervised(num_chunks=5))),
        remove_spaces("RCA_Supervised(num_chunks=5)"))

  def test_mlkr(self):
    self.assertEqual(remove_spaces(str(metric_learn.MLKR(max_iter=777))),
                     remove_spaces("MLKR(max_iter=777)"))

  def test_mmc(self):
    self.assertEqual(remove_spaces(str(metric_learn.MMC(diagonal=True))),
                     remove_spaces("MMC(diagonal=True)"))
    self.assertEqual(
        remove_spaces(str(metric_learn.MMC_Supervised(max_iter=1))),
        remove_spaces("MMC_Supervised(max_iter=1)"))


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
  model.fit(*remove_y(model, input_data, labels))
  metric = model.get_metric()
  score = metric(X[0], X[1])

  # then we refit the estimator on another dataset
  model.fit(*remove_y(model, np.sin(input_data), labels))

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
  model.fit(*remove_y(model, input_data, labels))
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
  model.fit(*remove_y(model, input_data, labels))
  metric = model.get_metric()

  list_test_get_metric_doesnt_raise = [(X[0], X[1]),
                                       (X[0].tolist(), X[1].tolist()),
                                       (X[0][None], X[1][None])]

  for u, v in list_test_get_metric_doesnt_raise:
    with pytest.warns(None) as record:
      metric(u, v)
    assert len(record) == 0

  # Test that the scalar case works
  model.components_ = np.array([3.1])
  metric = model.get_metric()
  for u, v in [(5, 6.7), ([5], [6.7]), ([[5]], [[6.7]])]:
    with pytest.warns(None) as record:
      metric(u, v)
    assert len(record) == 0


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_n_components(estimator, build_dataset):
  """Check that estimators that have a n_components parameters can use it
  and that it actually works as expected"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)

  if hasattr(model, 'n_components'):
    set_random_state(model)
    model.set_params(n_components=None)
    model.fit(*remove_y(model, input_data, labels))
    assert model.components_.shape == (X.shape[1], X.shape[1])

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=X.shape[1] - 1)
    model.fit(*remove_y(model, input_data, labels))
    assert model.components_.shape == (X.shape[1] - 1, X.shape[1])

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=X.shape[1] + 1)
    with pytest.raises(ValueError) as expected_err:
      model.fit(*remove_y(model, input_data, labels))
    assert (str(expected_err.value) ==
            'Invalid n_components, must be in [1, {}]'.format(X.shape[1]))

    model = clone(estimator)
    set_random_state(model)
    model.set_params(n_components=0)
    with pytest.raises(ValueError) as expected_err:
      model.fit(*remove_y(model, input_data, labels))
    assert (str(expected_err.value) ==
            'Invalid n_components, must be in [1, {}]'.format(X.shape[1]))


if __name__ == '__main__':
  unittest.main()
