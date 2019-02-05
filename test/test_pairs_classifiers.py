import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from test.test_utils import pairs_learners, ids_pairs_learners
from sklearn.utils.testing import set_random_state
from sklearn import clone
import numpy as np


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', pairs_learners,
                         ids=ids_pairs_learners)
def test_predict_monotonous(estimator, build_dataset,
                                            with_preprocessor):
  """Test that all predicted values are either +1 or -1"""
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  pairs_train, pairs_test, y_train, y_test = train_test_split(input_data,
                                                              labels)
  estimator.fit(pairs_train, y_train)
  predictions = estimator.predict(pairs_test, y_test)
  assert np.isin(predictions, [-1, 1]).all()


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

