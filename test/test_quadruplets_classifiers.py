import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from test.test_utils import quadruplets_learners, ids_quadruplets_learners
from sklearn.utils.testing import set_random_state
from sklearn import clone
import numpy as np


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', quadruplets_learners,
                         ids=ids_quadruplets_learners)
def test_predict_only_one_or_minus_one(estimator, build_dataset,
                                       with_preprocessor):
  """Test that all predicted values are either +1 or -1"""
  input_data, labels, preprocessor, _ = build_dataset(with_preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  set_random_state(estimator)
  (quadruplets_train,
   quadruplets_test, y_train, y_test) = train_test_split(input_data, labels)
  estimator.fit(quadruplets_train)
  predictions = estimator.predict(quadruplets_test)
  not_valid = [e for e in predictions if e not in [-1, 1]]
  assert len(not_valid) == 0


@pytest.mark.parametrize('with_preprocessor', [True, False])
@pytest.mark.parametrize('estimator, build_dataset', quadruplets_learners,
                         ids=ids_quadruplets_learners)
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

