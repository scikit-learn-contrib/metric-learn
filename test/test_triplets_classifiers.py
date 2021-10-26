"""
Tests all functionality for TripletsClassifiers. Methods, warrnings,
correctness, use cases, etc.
"""
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
import metric_learn

from test.test_utils import triplets_learners, ids_triplets_learners
from metric_learn.sklearn_shims import set_random_state
from sklearn import clone
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', triplets_learners,
                         ids=ids_triplets_learners)
def test_predict_only_one_or_minus_one(estimator, build_dataset,
                                       with_preprocessor):
  """Test that all predicted values are either +1 or -1"""
  input_data, _, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  triplets_train, triplets_test = train_test_split(input_data)
  if isinstance(estimator, metric_learn.SCML):
    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      estimator.fit(triplets_train)
    assert msg in str(raised_warning[0].message)
  else:
    estimator.fit(triplets_train)
  predictions = estimator.predict(triplets_test)

  not_valid = [e for e in predictions if e not in [-1, 1]]
  assert len(not_valid) == 0


@pytest.mark.parametrize('estimator, build_dataset', triplets_learners,
                         ids=ids_triplets_learners)
def test_no_zero_prediction(estimator, build_dataset):
  """
  Test that all predicted values are not zero, even when the
  distance d(x,y) and d(x,z) is the same for a triplet of the
  form (x, y, z). i.e border cases.
  """
  triplets, _, _, X = build_dataset(with_preprocessor=False)
  # Force 3 dimentions only, to use cross product and get easy orthogonal vec.
  triplets = np.array([[t[0][:3], t[1][:3], t[2][:3]] for t in triplets])
  X = X[:, :3]
  # Dummy fit
  estimator = clone(estimator)
  set_random_state(estimator)
  if isinstance(estimator, metric_learn.SCML):
    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      estimator.fit(triplets)
    assert msg in str(raised_warning[0].message)
  else:
    estimator.fit(triplets)
  # We force the transformation to be identity, to force euclidean distance
  estimator.components_ = np.eye(X.shape[1])

  # Get two orthogonal vectors in respect to X[1]
  k = X[1] / np.linalg.norm(X[1])  # Normalize first vector
  x = X[2] - X[2].dot(k) * k  # Get random orthogonal vector
  x /= np.linalg.norm(x)  # Normalize
  y = np.cross(k, x)  # Get orthogonal vector to x
  # Assert these orthogonal vectors are different
  with pytest.raises(AssertionError):
    assert_array_equal(X[1], x)
  with pytest.raises(AssertionError):
    assert_array_equal(X[1], y)
  # Assert the distance is the same for both
  assert estimator.get_metric()(X[1], x) == estimator.get_metric()(X[1], y)

  # Form the three scenarios where predict() gives 0 with numpy.sign
  triplets_test = np.array(  # Critical examples
    [[X[0], X[2], X[2]],
     [X[1], X[1], X[1]],
     [X[1], x, y]])
  # Predict
  predictions = estimator.predict(triplets_test)
  # Check there are no zero values
  assert np.sum(predictions == 0) == 0


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', triplets_learners,
                         ids=ids_triplets_learners)
def test_raise_not_fitted_error_if_not_fitted(estimator, build_dataset,
                                              with_preprocessor):
  """Test that a NotFittedError is raised if someone tries to use the
  methods: predict, decision_function and score when the metric learner
  has not been fitted."""
  input_data, _, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  with pytest.raises(NotFittedError):
    estimator.predict(input_data)
  with pytest.raises(NotFittedError):
    estimator.decision_function(input_data)
  with pytest.raises(NotFittedError):
    estimator.score(input_data)


@pytest.mark.parametrize('estimator, build_dataset', triplets_learners,
                         ids=ids_triplets_learners)
def test_accuracy_toy_example(estimator, build_dataset):
  """Test that the default scoring for triplets (accuracy) works on some
  toy example"""
  triplets, _, _, X = build_dataset(with_preprocessor=False)
  estimator = clone(estimator)
  set_random_state(estimator)
  if isinstance(estimator, metric_learn.SCML):
    msg = "As no value for `n_basis` was selected, "
    with pytest.warns(UserWarning) as raised_warning:
      estimator.fit(triplets)
    assert msg in str(raised_warning[0].message)
  else:
    estimator.fit(triplets)
  # We take the two first points and we build 4 regularly spaced points on the
  # line they define, so that it's easy to build triplets of different
  # similarities.
  X_test = X[0] + np.arange(4)[:, np.newaxis] * (X[0] - X[1]) / 4

  triplets_test = np.array(
      [[X_test[0], X_test[2], X_test[1]],
       [X_test[1], X_test[3], X_test[0]],
       [X_test[1], X_test[2], X_test[3]],
       [X_test[3], X_test[0], X_test[2]]])
  # we force the transformation to be identity so that we control what it does
  estimator.components_ = np.eye(X.shape[1])
  assert estimator.score(triplets_test) == 0.25
