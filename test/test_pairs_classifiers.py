from functools import partial

import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.distance import euclidean

from metric_learn.base_metric import _PairsClassifierMixin, MahalanobisMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (f1_score, accuracy_score, fbeta_score,
                             precision_score)
from sklearn.model_selection import train_test_split

from test.test_utils import pairs_learners, ids_pairs_learners
from sklearn.utils.testing import set_random_state
from sklearn import clone
import numpy as np
from itertools import product


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_predict_only_one_or_minus_one(estimator, build_dataset,
                                       with_preprocessor):
  """Test that all predicted values are either +1 or -1"""
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  pairs_train, pairs_test, y_train, y_test = train_test_split(input_data,
                                                              labels)
  estimator.fit(pairs_train, y_train)
  predictions = estimator.predict(pairs_test)
  not_valid = [e for e in predictions if e not in [-1, 1]]
  assert len(not_valid) == 0


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_predict_monotonous(estimator, build_dataset,
                            with_preprocessor):
  """Test that there is a threshold distance separating points labeled as
  similar and points labeled as dissimilar """
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  pairs_train, pairs_test, y_train, y_test = train_test_split(input_data,
                                                              labels)
  estimator.fit(pairs_train, y_train)
  distances = estimator.score_pairs(pairs_test)
  predictions = estimator.predict(pairs_test)
  min_dissimilar = np.min(distances[predictions == -1])
  max_similar = np.max(distances[predictions == 1])
  assert max_similar <= min_dissimilar
  separator = np.mean([min_dissimilar, max_similar])
  assert (predictions[distances > separator] == -1).all()
  assert (predictions[distances < separator] == 1).all()


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_raise_not_fitted_error_if_not_fitted(estimator, build_dataset,
                                              with_preprocessor):
  """Test that a NotFittedError is raised if someone tries to use
  score_pairs, decision_function, get_metric, transform or
  get_mahalanobis_matrix on input data and the metric learner
  has not been fitted."""
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  with pytest.raises(NotFittedError):
    estimator.score_pairs(input_data)
  with pytest.raises(NotFittedError):
    estimator.decision_function(input_data)
  with pytest.raises(NotFittedError):
    estimator.get_metric()
  with pytest.raises(NotFittedError):
    estimator.transform(input_data)
  with pytest.raises(NotFittedError):
    estimator.get_mahalanobis_matrix()
  with pytest.raises(NotFittedError):
    estimator.calibrate_threshold(input_data, labels)

  with pytest.raises(NotFittedError):
    estimator.set_threshold(0.5)
  with pytest.raises(NotFittedError):
    estimator.predict(input_data)


@pytest.mark.parametrize('calibration_params',
                         [None, {}, dict(), {'strategy': 'accuracy'}] +
                         [{'strategy': strategy, 'min_rate': min_rate}
                          for (strategy, min_rate) in product(
                              ['max_tpr', 'max_tnr'], [0., 0.2, 0.8, 1.])] +
                         [{'strategy': 'f_beta', 'beta': beta}
                          for beta in [0., 0.1, 0.2, 1., 5.]]
                         )
@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_fit_with_valid_threshold_params(estimator, build_dataset,
                                         with_preprocessor,
                                         calibration_params):
  """Tests that fitting `calibration_params` with appropriate parameters works
  as expected"""
  pairs, y, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  estimator.fit(pairs, y, calibration_params=calibration_params)
  estimator.predict(pairs)


@pytest.mark.parametrize('kwargs',
                         [{'strategy': 'accuracy'}] +
                         [{'strategy': strategy, 'min_rate': min_rate}
                          for (strategy, min_rate) in product(
                              ['max_tpr', 'max_tnr'], [0., 0.2, 0.8, 1.])] +
                         [{'strategy': 'f_beta', 'beta': beta}
                          for beta in [0., 0.1, 0.2, 1., 5.]]
                         )
@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_threshold_different_scores_is_finite(estimator, build_dataset,
                                              with_preprocessor, kwargs):
  # test that calibrating the threshold works for every metric learner
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  estimator.fit(input_data, labels)
  with pytest.warns(None) as record:
    estimator.calibrate_threshold(input_data, labels, **kwargs)
  assert len(record) == 0


class IdentityPairsClassifier(MahalanobisMixin, _PairsClassifierMixin):
  """A simple pairs classifier for testing purposes, that will just have
  identity as components_, and a string threshold so that it returns an
  error if not explicitely set.
  """
  def fit(self, pairs, y):
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')
    self.components_ = np.atleast_2d(np.identity(pairs.shape[2]))
    # self.threshold_ is not set.
    return self


def test_unset_threshold():
  """Tests that the "threshold is unset" error is raised when using predict
  (performs binary classification on pairs) with an unset threshold."""
  identity_pairs_classifier = IdentityPairsClassifier()
  pairs = np.array([[[0.], [1.]], [[1.], [3.]], [[2.], [5.]], [[3.], [7.]]])
  y = np.array([1, 1, -1, -1])
  identity_pairs_classifier.fit(pairs, y)
  with pytest.raises(AttributeError) as e:
    identity_pairs_classifier.predict(pairs)

  expected_msg = ("A threshold for this estimator has not been set, "
                  "call its set_threshold or calibrate_threshold method.")

  assert str(e.value) == expected_msg


def test_set_threshold():
  # test that set_threshold indeed sets the threshold
  identity_pairs_classifier = IdentityPairsClassifier()
  pairs = np.array([[[0.], [1.]], [[1.], [3.]], [[2.], [5.]], [[3.], [7.]]])
  y = np.array([1, 1, -1, -1])
  identity_pairs_classifier.fit(pairs, y)
  identity_pairs_classifier.set_threshold(0.5)
  assert identity_pairs_classifier.threshold_ == 0.5


def test_f_beta_1_is_f_1():
  # test that putting beta to 1 indeed finds the best threshold to optimize
  # the f1_score
  rng = np.random.RandomState(42)
  n_samples = 100
  pairs, y = rng.randn(n_samples, 2, 5), rng.choice([-1, 1], size=n_samples)
  pairs_learner = IdentityPairsClassifier()
  pairs_learner.fit(pairs, y)
  pairs_learner.calibrate_threshold(pairs, y, strategy='f_beta', beta=1)
  best_f1_score = f1_score(y, pairs_learner.predict(pairs))
  for threshold in - pairs_learner.decision_function(pairs):
    pairs_learner.set_threshold(threshold)
    assert f1_score(y, pairs_learner.predict(pairs)) <= best_f1_score


def true_pos_true_neg_rates(y_true, y_pred):
  """A function that returns the true positive rates and the true negatives
  rate. For testing purposes (optimized for readability not performance)."""
  assert y_pred.shape[0] == y_true.shape[0]
  tp = np.sum((y_pred == 1) * (y_true == 1))
  tn = np.sum((y_pred == -1) * (y_true == -1))
  fn = np.sum((y_pred == -1) * (y_true == 1))
  fp = np.sum((y_pred == 1) * (y_true == -1))
  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  tpr = tpr if not np.isnan(tpr) else 0.
  tnr = tnr if not np.isnan(tnr) else 0.
  return tpr, tnr


def tpr_threshold(y_true, y_pred, tnr_threshold=0.):
  """A function that returns the true positive rate if the true negative
  rate is higher or equal than `threshold`, and -1 otherwise. For testing
  purposes"""
  tpr, tnr = true_pos_true_neg_rates(y_true, y_pred)
  if tnr < tnr_threshold:
    return -1
  else:
    return tpr


def tnr_threshold(y_true, y_pred, tpr_threshold=0.):
  """A function that returns the true negative rate if the true positive
  rate is higher or equal than `threshold`, and -1 otherwise. For testing
  purposes"""
  tpr, tnr = true_pos_true_neg_rates(y_true, y_pred)
  if tpr < tpr_threshold:
    return -1
  else:
    return tnr


@pytest.mark.parametrize('kwargs, scoring',
                         [({'strategy': 'accuracy'}, accuracy_score)] +
                         [({'strategy': 'f_beta', 'beta': b},
                           partial(fbeta_score, beta=b))
                          for b in [0.1, 0.5, 1.]] +
                         [({'strategy': 'f_beta', 'beta': 0},
                           precision_score)] +
                         [({'strategy': 'max_tpr', 'min_rate': t},
                           partial(tpr_threshold, tnr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]] +
                         [({'strategy': 'max_tnr', 'min_rate': t},
                           partial(tnr_threshold, tpr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]],
                         )
def test_found_score_is_best_score(kwargs, scoring):
  # test that when we use calibrate threshold, it will indeed be the
  # threshold that have the best score
  rng = np.random.RandomState(42)
  n_samples = 50
  pairs, y = rng.randn(n_samples, 2, 5), rng.choice([-1, 1], size=n_samples)
  pairs_learner = IdentityPairsClassifier()
  pairs_learner.fit(pairs, y)
  pairs_learner.calibrate_threshold(pairs, y, **kwargs)
  best_score = scoring(y, pairs_learner.predict(pairs))
  scores = []
  predicted_scores = pairs_learner.decision_function(pairs)
  predicted_scores = np.hstack([[np.min(predicted_scores) - 1],
                                predicted_scores,
                                [np.max(predicted_scores) + 1]])
  for threshold in - predicted_scores:
    pairs_learner.set_threshold(threshold)
    score = scoring(y, pairs_learner.predict(pairs))
    assert score <= best_score
    scores.append(score)
  assert len(set(scores)) > 1  # assert that we didn't always have the same
  # value for the score (which could be a hint for some bug, but would still
  # silently pass the test))


@pytest.mark.parametrize('kwargs, scoring',
                         [({'strategy': 'accuracy'}, accuracy_score)] +
                         [({'strategy': 'f_beta', 'beta': b},
                           partial(fbeta_score, beta=b))
                          for b in [0.1, 0.5, 1.]] +
                         [({'strategy': 'f_beta', 'beta': 0},
                           precision_score)] +
                         [({'strategy': 'max_tpr', 'min_rate': t},
                           partial(tpr_threshold, tnr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]] +
                         [({'strategy': 'max_tnr', 'min_rate': t},
                           partial(tnr_threshold, tpr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]]
                         )
def test_found_score_is_best_score_duplicates(kwargs, scoring):
  # test that when we use calibrate threshold, it will indeed be the
  # threshold that have the best score. It's the same as the previous test
  # except this time we test that the scores are coherent even if there are
  # duplicates (i.e. points that have the same score returned by
  # `decision_function`).
  rng = np.random.RandomState(42)
  n_samples = 50
  pairs, y = rng.randn(n_samples, 2, 5), rng.choice([-1, 1], size=n_samples)
  # we create some duplicates points, which will also have the same score
  # predicted
  pairs[6:10] = pairs[10:14]
  y[6:10] = y[10:14]
  pairs_learner = IdentityPairsClassifier()
  pairs_learner.fit(pairs, y)
  pairs_learner.calibrate_threshold(pairs, y, **kwargs)
  best_score = scoring(y, pairs_learner.predict(pairs))
  scores = []
  predicted_scores = pairs_learner.decision_function(pairs)
  predicted_scores = np.hstack([[np.min(predicted_scores) - 1],
                                predicted_scores,
                                [np.max(predicted_scores) + 1]])
  for threshold in - predicted_scores:
    pairs_learner.set_threshold(threshold)
    score = scoring(y, pairs_learner.predict(pairs))
    assert score <= best_score
    scores.append(score)
  assert len(set(scores)) > 1  # assert that we didn't always have the same
  # value for the score (which could be a hint for some bug, but would still
  # silently pass the test))


@pytest.mark.parametrize('invalid_args, expected_msg',
                         [({'strategy': 'weird'},
                           ('Strategy can either be "accuracy", "f_beta" or '
                            '"max_tpr" or "max_tnr". Got "weird" instead.'))] +
                         [({'strategy': strategy, 'min_rate': min_rate},
                           'Parameter min_rate must be a number in'
                           '[0, 1]. Got {} instead.'.format(min_rate))
                          for (strategy, min_rate) in product(
                             ['max_tpr', 'max_tnr'],
                             [None, 'weird', -0.2, 1.2, 3 + 2j])] +
                         [({'strategy': 'f_beta', 'beta': beta},
                           'Parameter beta must be a real number. '
                           'Got {} instead.'.format(type(beta)))
                          for beta in [None, 'weird', 3 + 2j]]
                         )
def test_calibrate_threshold_invalid_parameters_right_error(invalid_args,
                                                            expected_msg):
  # test that the right error message is returned if invalid arguments are
  # given to calibrate_threshold
  rng = np.random.RandomState(42)
  pairs, y = rng.randn(20, 2, 5), rng.choice([-1, 1], size=20)
  pairs_learner = IdentityPairsClassifier()
  pairs_learner.fit(pairs, y)
  with pytest.raises(ValueError) as raised_error:
    pairs_learner.calibrate_threshold(pairs, y, **invalid_args)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('valid_args',
                         [{'strategy': 'accuracy'}] +
                         [{'strategy': strategy, 'min_rate': min_rate}
                          for (strategy, min_rate) in product(
                             ['max_tpr', 'max_tnr'],
                             [0., 0.2, 0.8, 1.])] +
                         [{'strategy': 'f_beta', 'beta': beta}
                          for beta in [-5., -1., 0., 0.1, 0.2, 1., 5.]]
                         # Note that we authorize beta < 0 (even if
                         # in fact it will be squared, so it would be useless
                         # to do that)
                         )
def test_calibrate_threshold_valid_parameters(valid_args):
  # test that no warning message is returned if valid arguments are given to
  # calibrate threshold
  rng = np.random.RandomState(42)
  pairs, y = rng.randn(20, 2, 5), rng.choice([-1, 1], size=20)
  pairs_learner = IdentityPairsClassifier()
  pairs_learner.fit(pairs, y)
  with pytest.warns(None) as record:
    pairs_learner.calibrate_threshold(pairs, y, **valid_args)
  assert len(record) == 0


def test_calibrate_threshold_extreme():
  """Test that in the (rare) case where we should accept all points or
  reject all points, this is effectively what
  is done"""

  class MockBadPairsClassifier(MahalanobisMixin, _PairsClassifierMixin):
    """A pairs classifier that returns bad scores (i.e. in the inverse order
    of what we would expect from a good pairs classifier
    """

    def fit(self, pairs, y, calibration_params=None):
      self.preprocessor_ = 'not used'
      self.components_ = 'not used'
      self.calibrate_threshold(pairs, y, **(calibration_params if
                                            calibration_params is not None else
                                            dict()))
      return self

    def decision_function(self, pairs):
      return np.arange(pairs.shape[0], dtype=float)

  rng = np.random.RandomState(42)
  pairs = rng.randn(7, 2, 5)  # the info in X is not used, it's just for the
  # API

  y = [1., 1., 1., -1., -1., -1., -1.]
  mock_clf = MockBadPairsClassifier()
  # case of bad scoring with more negative than positives. In
  # this case, when:
  # optimizing for accuracy we should reject all points
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'accuracy'})
  assert_array_equal(mock_clf.predict(pairs), - np.ones(7))

  # optimizing for max_tpr we should accept all points if min_rate == 0. (
  # because by convention then tnr=0/0=0)
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'max_tpr',
                                             'min_rate': 0.})
  assert_array_equal(mock_clf.predict(pairs), np.ones(7))
  # optimizing for max_tnr we should reject all points if min_rate = 0. (
  # because by convention then tpr=0/0=0)
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'max_tnr',
                                             'min_rate': 0.})
  assert_array_equal(mock_clf.predict(pairs), - np.ones(7))

  y = [1., 1., 1., 1., -1., -1., -1.]
  # case of bad scoring with more positives than negatives. In
  # this case, when:
  # optimizing for accuracy we should accept all points
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'accuracy'})
  assert_array_equal(mock_clf.predict(pairs), np.ones(7))
  # optimizing for max_tpr we should accept all points if min_rate == 0. (
  # because by convention then tnr=0/0=0)
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'max_tpr',
                                             'min_rate': 0.})
  assert_array_equal(mock_clf.predict(pairs), np.ones(7))
  # optimizing for max_tnr we should reject all points if min_rate = 0. (
  # because by convention then tpr=0/0=0)
  mock_clf.fit(pairs, y, calibration_params={'strategy': 'max_tnr',
                                             'min_rate': 0.})
  assert_array_equal(mock_clf.predict(pairs), - np.ones(7))

  # Note: we'll never find a case where we would reject all points for
  # maximizing tpr (we can always accept more points), and accept all
  # points for maximizing tnr (we can always reject more points)

  # case of alternated scores: for optimizing the f_1 score we should accept
  # all points (because this way we have max recall (1) and max precision (
  # here: 0.5))
  y = [1., -1., 1., -1., 1., -1.]
  mock_clf.fit(pairs[:6], y, calibration_params={'strategy': 'f_beta',
                                                 'beta': 1.})
  assert_array_equal(mock_clf.predict(pairs[:6]), np.ones(6))

  # Note: for optimizing f_1 score, we will never find an optimal case where we
  # reject all points because in this case we would have 0 precision (by
  # convention, because it's 0/0), and 0 recall (and we could always decrease
  # the threshold to increase the recall, and we couldn't do worse for
  # precision so it would be better)


@pytest.mark.parametrize('estimator, _',
                         pairs_learners + [(IdentityPairsClassifier(), None),
                                           (_PairsClassifierMixin, None)],
                         ids=ids_pairs_learners + ['mock', 'class'])
@pytest.mark.parametrize('invalid_args, expected_msg',
                         [({'strategy': 'weird'},
                           ('Strategy can either be "accuracy", "f_beta" or '
                            '"max_tpr" or "max_tnr". Got "weird" instead.'))] +
                         [({'strategy': strategy, 'min_rate': min_rate},
                           'Parameter min_rate must be a number in'
                           '[0, 1]. Got {} instead.'.format(min_rate))
                          for (strategy, min_rate) in product(
                             ['max_tpr', 'max_tnr'],
                             [None, 'weird', -0.2, 1.2, 3 + 2j])] +
                         [({'strategy': 'f_beta', 'beta': beta},
                           'Parameter beta must be a real number. '
                           'Got {} instead.'.format(type(beta)))
                          for beta in [None, 'weird', 3 + 2j]]
                         )
def test_validate_calibration_params_invalid_parameters_right_error(
        estimator, _, invalid_args, expected_msg):
  # test that the right error message is returned if invalid arguments are
  # given to _validate_calibration_params, for all pairs metric learners as
  # well as a mocking general identity pairs classifier and the class itself
  with pytest.raises(ValueError) as raised_error:
    estimator._validate_calibration_params(**invalid_args)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator, _',
                         pairs_learners + [(IdentityPairsClassifier(), None),
                                           (_PairsClassifierMixin, None)],
                         ids=ids_pairs_learners + ['mock', 'class'])
@pytest.mark.parametrize('valid_args',
                         [{}, {'strategy': 'accuracy'}] +
                         [{'strategy': strategy, 'min_rate': min_rate}
                          for (strategy, min_rate) in product(
                             ['max_tpr', 'max_tnr'],
                             [0., 0.2, 0.8, 1.])] +
                         [{'strategy': 'f_beta', 'beta': beta}
                          for beta in [-5., -1., 0., 0.1, 0.2, 1., 5.]]
                         # Note that we authorize beta < 0 (even if
                         # in fact it will be squared, so it would be useless
                         # to do that)
                         )
def test_validate_calibration_params_valid_parameters(
        estimator, _, valid_args):
  # test that no warning message is returned if valid arguments are given to
  # _validate_calibration_params for all pairs metric learners, as well as
  # a mocking example, and the class itself
  with pytest.warns(None) as record:
    estimator._validate_calibration_params(**valid_args)
  assert len(record) == 0


@pytest.mark.parametrize('estimator, build_dataset',
                         pairs_learners,
                         ids=ids_pairs_learners)
def test_validate_calibration_params_invalid_parameters_error_before__fit(
        estimator, build_dataset):
  """For all pairs metric learners (which currently all have a _fit method),
  make sure that calibration parameters are validated before fitting"""
  estimator = clone(estimator)
  input_data, labels, _, _ = build_dataset()

  def breaking_fun(**args):  # a function that fails so that we will miss
    # the calibration at the end and therefore the right error message from
    # validating params should be thrown before
    raise RuntimeError('Game over.')
  estimator._fit = breaking_fun
  expected_msg = ('Strategy can either be "accuracy", "f_beta" or '
                  '"max_tpr" or "max_tnr". Got "weird" instead.')
  with pytest.raises(ValueError) as raised_error:
    estimator.fit(input_data, labels, calibration_params={'strategy': 'weird'})
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_accuracy_toy_example(estimator, build_dataset):
  """Test that the accuracy works on some toy example (hence that the
  prediction is OK)"""
  input_data, labels, preprocessor, X = build_dataset(with_preprocessor=False)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  estimator.fit(input_data, labels)
  # we force the transformation to be identity so that we control what it does
  estimator.components_ = np.eye(X.shape[1])
  # the threshold for similar or dissimilar pairs is half of the distance
  # between X[0] and X[1]
  estimator.set_threshold(euclidean(X[0], X[1]) / 2)
  # We take the two first points and we build 4 regularly spaced points on the
  # line they define, so that it's easy to build quadruplets of different
  # similarities.
  X_test = X[0] + np.arange(4)[:, np.newaxis] * (X[0] - X[1]) / 4
  pairs_test = np.array(
      [[X_test[0], X_test[1]],  # similar
       [X_test[0], X_test[3]],  # dissimilar
       [X_test[1], X_test[2]],  # similar
       [X_test[2], X_test[3]]])  # similar
  y = np.array([-1, 1, 1, -1])  # [F, F, T, F]
  assert accuracy_score(estimator.predict(pairs_test), y) == 0.25
