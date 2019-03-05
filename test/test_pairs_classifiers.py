from __future__ import division

from functools import partial

import pytest
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
  """Test that a NotFittedError is raised if someone tries to predict and
  the metric learner has not been fitted."""
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  with pytest.raises(NotFittedError):
    estimator.predict(input_data)


@pytest.mark.parametrize('kwargs',
                         [{'strategy': 'accuracy'}] +
                         [{'strategy': strategy, 'threshold': threshold}
                          for (strategy, threshold) in product(
                              ['max_tpr', 'max_tnr'], [0., 0.2, 0.8, 1.])] +
                         [{'strategy': 'f_beta', 'beta': beta}
                          for beta in [0., 0.1, 0.2, 1., 5.]]
                         )
@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_threshold_different_scores_is_finite(estimator, build_dataset,
                                              with_preprocessor, kwargs):
  # test that the score returned is finite for every metric learner
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
  identity as transformer_, and a string threshold so that it returns an
  error if not explicitely set.
  """
  def fit(self, pairs, y):
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')
    self.transformer_ = np.atleast_2d(np.identity(pairs.shape[2]))
    self.threshold_ = 'I am not set.'
    return self


def test_set_threshold():
  # test that set_threshold indeed sets the threshold
  identity_pairs_classifier = IdentityPairsClassifier()
  pairs = np.array([[[0.], [1.]], [[1.], [3.]], [[2.], [5.]], [[3.], [7.]]])
  y = np.array([1, 1, -1, -1])
  identity_pairs_classifier.fit(pairs, y)
  identity_pairs_classifier.set_threshold(0.5)
  assert identity_pairs_classifier.threshold_ == 0.5


def test_set_default_threshold_toy_example():
  # test that the default threshold has the right value on a toy example
  identity_pairs_classifier = IdentityPairsClassifier()
  pairs = np.array([[[0.], [1.]], [[1.], [3.]], [[2.], [5.]], [[3.], [7.]]])
  y = np.array([1, 1, -1, -1])
  identity_pairs_classifier.fit(pairs, y)
  identity_pairs_classifier._set_default_threshold(pairs, y)
  assert identity_pairs_classifier.threshold_ == 2.5


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
                         [({'strategy': 'max_tpr', 'threshold': t},
                           partial(tpr_threshold, tnr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]] +
                         [({'strategy': 'max_tnr', 'threshold': t},
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
                         [({'strategy': 'max_tpr', 'threshold': t},
                           partial(tpr_threshold, tnr_threshold=t))
                          for t in [0., 0.1, 0.5, 0.8, 1.]] +
                         [({'strategy': 'max_tnr', 'threshold': t},
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
                         [({'strategy': strategy, 'threshold': threshold},
                           'Parameter threshold must be a number in'
                           '[0, 1]. Got {} instead.'.format(threshold))
                          for (strategy, threshold) in product(
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
                         [{'strategy': strategy, 'threshold': threshold}
                          for (strategy, threshold) in product(
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
