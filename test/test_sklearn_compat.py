import pytest
import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import TransformerMixin
from sklearn.datasets import load_iris, make_regression, make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle, check_random_state
from sklearn.utils.estimator_checks import is_public_parameter
from sklearn.utils.testing import (assert_allclose_dense_sparse,
                                   set_random_state)
from sklearn.utils.fixes import signature

from metric_learn import (Covariance, ITML, LFDA, LMNN, LSML, MLKR, MMC, NCA,
                          RCA, SDML, ITML_Supervised, LSML_Supervised,
                          MMC_Supervised, RCA_Supervised, SDML_Supervised)
from metric_learn.constraints import wrap_pairs, Constraints
from sklearn import clone
import numpy as np
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split)
from sklearn.utils.testing import _get_args


# Wrap the _Supervised methods with a deterministic wrapper for testing.
class deterministic_mixin(object):
  def fit(self, X, y):
    rs = np.random.RandomState(1234)
    return super(deterministic_mixin, self).fit(X, y, random_state=rs)


class dLSML(deterministic_mixin, LSML_Supervised):
  pass


class dITML(deterministic_mixin, ITML_Supervised):
  pass


class dMMC(deterministic_mixin, MMC_Supervised):
  pass


class dSDML(deterministic_mixin, SDML_Supervised):
  pass


class dRCA(deterministic_mixin, RCA_Supervised):
  pass


class TestSklearnCompat(unittest.TestCase):
  def test_covariance(self):
    check_estimator(Covariance)

  def test_lmnn(self):
    check_estimator(LMNN)

  def test_lfda(self):
    check_estimator(LFDA)

  def test_mlkr(self):
    check_estimator(MLKR)

  def test_nca(self):
    check_estimator(NCA)

  def test_lsml(self):
    check_estimator(dLSML)

  def test_itml(self):
    check_estimator(dITML)

  def test_mmc(self):
    check_estimator(dMMC)

  # This fails due to a FloatingPointError
  # def test_sdml(self):
  #   check_estimator(dSDML)

  # This fails because the default num_chunks isn't data-dependent.
  # def test_rca(self):
  #   check_estimator(RCA_Supervised)


RNG = check_random_state(0)


# ---------------------- Test scikit-learn compatibility ----------------------


def build_data():
  dataset = load_iris()
  X, y = shuffle(dataset.data, dataset.target, random_state=RNG)
  num_constraints = 50
  constraints = Constraints.random_subset(y, random_state=RNG)
  pairs = constraints.positive_negative_pairs(num_constraints,
                                              same_length=True,
                                              random_state=RNG)
  return X, pairs


def build_classification(preprocessor):
  # builds a toy classification problem
  X, y = shuffle(*make_blobs(), random_state=RNG)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RNG)
  return (X, X, y, X_train, X_test, y_train, y_test, preprocessor)


def build_regression(preprocessor):
  # builds a toy regression problem
  X, y = shuffle(*make_regression(n_samples=100, n_features=10),
                 random_state=RNG)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RNG)
  return (X, X, y, X_train, X_test, y_train, y_test, preprocessor)


def build_pairs(preprocessor):
  # builds a toy pairs problem
  X, indices = build_data()
  if preprocessor is not None:
    # if preprocessor, we build a 2D array of pairs of indices
    _, y = wrap_pairs(X, indices)
    pairs = np.vstack([np.column_stack(indices[:2]),
                       np.column_stack(indices[2:])])
  else:
    # if not, we build a 3D array of pairs of samples
    pairs, y = wrap_pairs(X, indices)
  pairs, y = shuffle(pairs, y, random_state=RNG)
  (pairs_train, pairs_test, y_train,
   y_test) = train_test_split(pairs, y, random_state=RNG)
  return (X, pairs, y, pairs_train, pairs_test,
          y_train, y_test, preprocessor)


def build_quadruplets(preprocessor):
  # builds a toy quadruplets problem
  X, indices = build_data()
  c = np.column_stack(indices)
  if preprocessor is not None:
    # if preprocessor, we build a 2D array of quadruplets of indices
    quadruplets = c
  else:
    # if not, we build a 3D array of quadruplets of samples
    quadruplets = X[c]
  quadruplets = shuffle(quadruplets, random_state=RNG)
  y = y_train = y_test = None
  quadruplets_train, quadruplets_test = train_test_split(quadruplets,
                                                         random_state=RNG)
  return (X, quadruplets, y, quadruplets_train, quadruplets_test,
          y_train, y_test, preprocessor)


list_estimators = [(Covariance(), build_classification),
                   (ITML(), build_pairs),
                   (LFDA(), build_classification),
                   (LMNN(), build_classification),
                   (LSML(), build_quadruplets),
                   (MLKR(), build_regression),
                   (MMC(max_iter=2), build_pairs),  # max_iter=2 for faster
                   # testing
                   (NCA(), build_classification),
                   (RCA(), build_classification),
                   (SDML(), build_pairs),
                   (ITML_Supervised(), build_classification),
                   (LSML_Supervised(), build_classification),
                   (MMC_Supervised(), build_classification),
                   (RCA_Supervised(num_chunks=10), build_classification),
                   (SDML_Supervised(), build_classification)
                   ]

ids_estimators = ['covariance',
                  'itml',
                  'lfda',
                  'lmnn',
                  'lsml',
                  'mlkr',
                  'mmc',
                  'nca',
                  'rca',
                  'sdml',
                  'itml_supervised',
                  'lsml_supervised',
                  'mmc_supervised',
                  'rca_supervised',
                  'sdml_supervised'
                  ]


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_cross_validation(estimator, build_dataset, preprocessor):
  """Tests that you can do cross validation on metric-learn estimators
  """
  if any(hasattr(estimator, method) for method in ["predict", "score"]):
    (X, tuples, y, tuples_train, tuples_test,
     y_train, y_test, preprocessor) = build_dataset(preprocessor)
    estimator = clone(estimator)
    estimator.set_params(preprocessor=preprocessor)
    set_random_state(estimator)
    if hasattr(estimator, "score"):
      assert np.isfinite(cross_val_score(estimator, tuples, y)).all()
    if hasattr(estimator, "predict"):
      assert np.isfinite(cross_val_predict(estimator, tuples, y)).all()


def check_score(estimator, tuples, y):
  if hasattr(estimator, "score"):
    score = estimator.score(tuples, y)
    assert np.isfinite(score)


def check_predict(estimator, tuples):
  if hasattr(estimator, "predict"):
    y_predicted = estimator.predict(tuples)
    assert len(y_predicted), len(tuples)


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_simple_estimator(estimator, build_dataset, preprocessor):
  """Tests that fit, predict and scoring works.
  """
  if any(hasattr(estimator, method) for method in ["predict", "score"]):
    (X, tuples, y, tuples_train, tuples_test,
     y_train, y_test, preprocessor) = build_dataset(preprocessor)
    estimator = clone(estimator)
    estimator.set_params(preprocessor=preprocessor)
    set_random_state(estimator)

    estimator.fit(tuples_train, y_train)
    check_score(estimator, tuples_test, y_test)
    check_predict(estimator, tuples_test)


@pytest.mark.parametrize('estimator', [est[0] for est in list_estimators],
                         ids=ids_estimators)
@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
def test_no_attributes_set_in_init(estimator, preprocessor):
  """Check setting during init. Adapted from scikit-learn."""
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(type(estimator).__init__, "deprecated_original"):
      return

  init_params = _get_args(type(estimator).__init__)
  parents_init_params = [param for params_parent in
                         (_get_args(parent) for parent in
                          type(estimator).__mro__)
                         for param in params_parent]

  # Test for no setting apart from parameters during init
  invalid_attr = (set(vars(estimator)) - set(init_params) -
                  set(parents_init_params))
  assert not invalid_attr, \
      ("Estimator %s should not set any attribute apart"
       " from parameters during init. Found attributes %s."
       % (type(estimator).__name__, sorted(invalid_attr)))
  # Ensure that each parameter is set in init
  invalid_attr = (set(init_params) - set(vars(estimator)) -
                  set(["self"]))
  assert not invalid_attr, \
      ("Estimator %s should store all parameters"
       " as an attribute during init. Did not find "
       "attributes %s." % (type(estimator).__name__, sorted(invalid_attr)))


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_estimators_fit_returns_self(estimator, build_dataset, preprocessor):
  """Check if self is returned when calling fit"""
  # Adapted from scikit-learn
  (X, tuples, y, tuples_train, tuples_test,
   y_train, y_test, preprocessor) = build_dataset(preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  assert estimator.fit(tuples, y) is estimator


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_pipeline_consistency(estimator, build_dataset, preprocessor):
  # Adapted from scikit learn
  # check that make_pipeline(est) gives same score as est
  (_, input_data, y, _, _, _, _, preprocessor) = build_dataset(preprocessor)

  def make_random_state(estimator, in_pipeline):
    rs = {}
    name_estimator = estimator.__class__.__name__
    if name_estimator[-11:] == '_Supervised':
      name_param = 'random_state'
      if in_pipeline:
          name_param = name_estimator.lower() + '__' + name_param
      rs[name_param] = check_random_state(0)
    return rs

  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  pipeline = make_pipeline(estimator)
  estimator.fit(input_data, y, **make_random_state(estimator, False))
  pipeline.fit(input_data, y, **make_random_state(estimator, True))

  if hasattr(estimator, 'score'):
    result = estimator.score(input_data, y)
    result_pipe = pipeline.score(input_data, y)
    assert_allclose_dense_sparse(result, result_pipe)

  if hasattr(estimator, 'predict'):
    result = estimator.predict(input_data)
    result_pipe = pipeline.predict(input_data)
    assert_allclose_dense_sparse(result, result_pipe)

  if issubclass(estimator.__class__, TransformerMixin):
    if hasattr(estimator, 'transform'):
      result = estimator.transform(input_data)
      result_pipe = pipeline.transform(input_data)
      assert_allclose_dense_sparse(result, result_pipe)


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_dict_unchanged(estimator, build_dataset, preprocessor):
  # Adapted from scikit-learn
  (X, tuples, y, tuples_train, tuples_test,
   y_train, y_test, preprocessor) = build_dataset(preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "num_dims"):
    estimator.num_dims = 1
  estimator.fit(tuples, y)

  def check_dict():
    assert estimator.__dict__ == dict_before, (
        "Estimator changes __dict__ during %s" % method)
  for method in ["predict", "decision_function", "predict_proba"]:
    if hasattr(estimator, method):
      dict_before = estimator.__dict__.copy()
      getattr(estimator, method)(tuples)
      check_dict()
  if hasattr(estimator, "transform"):
    dict_before = estimator.__dict__.copy()
    # we transform only 2D arrays (dataset of points)
    estimator.transform(X)
    check_dict()


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_dont_overwrite_parameters(estimator, build_dataset, preprocessor):
  # Adapted from scikit-learn
  # check that fit method only changes or sets private attributes
  (X, tuples, y, tuples_train, tuples_test,
   y_train, y_test, preprocessor) = build_dataset(preprocessor)
  estimator = clone(estimator)
  estimator.set_params(preprocessor=preprocessor)
  if hasattr(estimator, "num_dims"):
    estimator.num_dims = 1
  dict_before_fit = estimator.__dict__.copy()

  estimator.fit(tuples, y)
  dict_after_fit = estimator.__dict__

  public_keys_after_fit = [key for key in dict_after_fit.keys()
                           if is_public_parameter(key)]

  attrs_added_by_fit = [key for key in public_keys_after_fit
                        if key not in dict_before_fit.keys()]

  # check that fit doesn't add any public attribute
  assert not attrs_added_by_fit, (
      "Estimator adds public attribute(s) during"
      " the fit method."
      " Estimators are only allowed to add private "
      "attributes"
      " either started with _ or ended"
      " with _ but %s added" % ', '.join(attrs_added_by_fit))

  # check that fit doesn't change any public attribute
  attrs_changed_by_fit = [key for key in public_keys_after_fit
                          if (dict_before_fit[key]
                              is not dict_after_fit[key])]

  assert not attrs_changed_by_fit, (
      "Estimator changes public attribute(s) during"
      " the fit method. Estimators are only allowed"
      " to change attributes started"
      " or ended with _, but"
      " %s changed" % ', '.join(attrs_changed_by_fit))


if __name__ == '__main__':
  unittest.main()
