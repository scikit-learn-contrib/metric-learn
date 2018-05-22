import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle, check_random_state
from sklearn.utils.estimator_checks import is_public_parameter
from sklearn.utils.testing import (assert_allclose_dense_sparse,
                                   set_random_state)

from metric_learn import ITML, MMC, SDML, LSML
from metric_learn.constraints import wrap_pairs, Constraints
from sklearn import clone
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split


def build_data():
  RNG = check_random_state(0)
  dataset = load_iris()
  X, y = shuffle(dataset.data, dataset.target, random_state=RNG)
  num_constraints = 20
  constraints = Constraints.random_subset(y)
  pairs = constraints.positive_negative_pairs(num_constraints,
                                              same_length=True,
                                              random_state=RNG)
  return X, pairs


def build_pairs():
  # test that you can do cross validation on a ConstrainedDataset with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  X_constrained, y = wrap_pairs(X, pairs)
  X_constrained, y = shuffle(X_constrained, y)
  (X_constrained_train, X_constrained_test, y_train,
   y_test) = train_test_split(X_constrained, y)
  return (X_constrained, y, X_constrained_train, X_constrained_test,
          y_train, y_test)


def build_quadruplets():
  # test that you can do cross validation on a ConstrainedDataset with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  c = np.column_stack(pairs)
  X_constrained = X[c]
  X_constrained = shuffle(X_constrained)
  y = y_train = y_test = None
  X_constrained_train, X_constrained_test = train_test_split(X_constrained)
  return (X_constrained, y, X_constrained_train, X_constrained_test,
          y_train, y_test)


list_estimators = [(ITML(), build_pairs),
                   (LSML(), build_quadruplets),
                   (MMC(), build_pairs),
                   (SDML(), build_pairs)
                   ]

ids_estimators = ['itml',
                  'lsml',
                  'mmc',
                  'sdml',
                  ]


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_cross_validation(estimator, build_dataset):
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  set_random_state(estimator)

  assert np.isfinite(cross_val_score(estimator, X_constrained, y)).all()


def check_score(estimator, X_constrained, y):
  score = estimator.score(X_constrained, y)
  assert np.isfinite(score)


def check_predict(estimator, X_constrained):
  y_predicted = estimator.predict(X_constrained)
  assert len(y_predicted), len(X_constrained)


def check_transform(estimator, X_constrained):
  X_transformed = estimator.transform(X_constrained)
  assert len(X_transformed), len(X_constrained.X)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_simple_estimator(estimator, build_dataset):
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  set_random_state(estimator)

  estimator.fit(X_constrained_train, y_train)
  check_score(estimator, X_constrained_test, y_test)
  check_predict(estimator, X_constrained_test)
  check_transform(estimator, X_constrained_test)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_pipelining_with_transformer(estimator, build_dataset):
  """
  Test that weakly supervised estimators fit well into pipelines
  """
  # test in a pipeline with KMeans
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  set_random_state(estimator)

  pipe = make_pipeline(estimator, KMeans())
  pipe.fit(X_constrained_train, y_train)
  check_score(pipe, X_constrained_test, y_test)
  check_transform(pipe, X_constrained_test)
  # we cannot use check_predict because in this case the shape of the
  # output is the shape of X_constrained.X, not X_constrained
  y_predicted = pipe.predict(X_constrained)
  assert len(y_predicted) == len(X_constrained.X)

  # test in a pipeline with PCA
  estimator = clone(estimator)
  pipe = make_pipeline(estimator, PCA())
  pipe.fit(X_constrained_train, y_train)
  check_transform(pipe, X_constrained_test)


@pytest.mark.parametrize('estimator', [est[0] for est in list_estimators],
                         ids=ids_estimators)
def test_no_fit_attributes_set_in_init(estimator):
  """Check that Estimator.__init__ doesn't set trailing-_ attributes."""
  # From scikit-learn
  estimator = clone(estimator)
  for attr in dir(estimator):
    if attr.endswith("_") and not attr.startswith("__"):
      # This check is for properties, they can be listed in dir
      # while at the same time have hasattr return False as long
      # as the property getter raises an AttributeError
      assert hasattr(estimator, attr), \
          ("By convention, attributes ending with '_' are "
           "estimated from data in scikit-learn. Consequently they "
           "should not be initialized in the constructor of an "
           "estimator but in the fit method. Attribute {!r} "
           "was found in estimator {}".format(
             attr, type(estimator).__name__))


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_estimators_fit_returns_self(estimator, build_dataset):
  """Check if self is returned when calling fit"""
  # From scikit-learn
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  assert estimator.fit(X_constrained, y) is estimator


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_pipeline_consistency(estimator, build_dataset):
  # From scikit learn
  # check that make_pipeline(est) gives same score as est
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  pipeline = make_pipeline(estimator)
  estimator.fit(X_constrained, y)
  pipeline.fit(X_constrained, y)

  funcs = ["score", "fit_transform"]

  for func_name in funcs:
    func = getattr(estimator, func_name, None)
    if func is not None:
      func_pipeline = getattr(pipeline, func_name)
      result = func(X_constrained, y)
      result_pipe = func_pipeline(X_constrained, y)
      assert_allclose_dense_sparse(result, result_pipe)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_dict_unchanged(estimator, build_dataset):
  # From scikit-learn
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  if hasattr(estimator, "n_components"):
    estimator.n_components = 1
  estimator.fit(X_constrained, y)
  for method in ["predict", "transform", "decision_function",
                 "predict_proba"]:
    if hasattr(estimator, method):
      dict_before = estimator.__dict__.copy()
      getattr(estimator, method)(X_constrained)
      assert estimator.__dict__ == dict_before, \
          ("Estimator changes __dict__ during %s"
           % method)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_dont_overwrite_parameters(estimator, build_dataset):
  # From scikit-learn
  # check that fit method only changes or sets private attributes
  (X_constrained, y, X_constrained_train, X_constrained_test,
   y_train, y_test) = build_dataset()
  estimator = clone(estimator)
  if hasattr(estimator, "n_components"):
    estimator.n_components = 1
  dict_before_fit = estimator.__dict__.copy()

  estimator.fit(X_constrained, y)
  dict_after_fit = estimator.__dict__

  public_keys_after_fit = [key for key in dict_after_fit.keys()
                           if is_public_parameter(key)]

  attrs_added_by_fit = [key for key in public_keys_after_fit
                        if key not in dict_before_fit.keys()]

  # check that fit doesn't add any public attribute
  assert not attrs_added_by_fit, \
    ("Estimator adds public attribute(s) during"
     " the fit method."
     " Estimators are only allowed to add private "
     "attributes"
     " either started with _ or ended"
     " with _ but %s added" % ', '.join(attrs_added_by_fit))

  # check that fit doesn't change any public attribute
  attrs_changed_by_fit = [key for key in public_keys_after_fit
                          if (dict_before_fit[key]
                              is not dict_after_fit[key])]

  assert not attrs_changed_by_fit, \
    ("Estimator changes public attribute(s) during"
     " the fit method. Estimators are only allowed"
     " to change attributes started"
     " or ended with _, but"
     " %s changed" % ', '.join(attrs_changed_by_fit))
