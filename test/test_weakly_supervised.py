import pytest
from sklearn.base import TransformerMixin
from sklearn.datasets import load_iris, make_regression, make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle, check_random_state
from sklearn.utils.estimator_checks import is_public_parameter
from sklearn.utils.testing import (assert_allclose_dense_sparse,
                                   set_random_state)
from sklearn.utils.fixes import signature

from metric_learn import (ITML, LFDA, LMNN, LSML, MLKR, MMC, NCA, RCA, SDML,
                          ITML_Supervised, LSML_Supervised, MMC_Supervised,
                          SDML_Supervised)
from metric_learn.constraints import wrap_pairs, Constraints
from sklearn import clone
import numpy as np
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split)

RNG = check_random_state(0)


def mock_preprocessor(indices):
  """A preprocessor for testing purposes that returns an all ones 3D array
  """
  return np.ones((indices.shape[0], 3))


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
  # test that you can do cross validation on tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, y = shuffle(*make_blobs(), random_state=RNG)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RNG)
  return (X, X, y, X_train, X_test, y_train, y_test, preprocessor)


def build_regression(preprocessor):
  # test that you can do cross validation on tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, y = shuffle(*make_regression(n_samples=100, n_features=10),
                  random_state=RNG)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RNG)
  return (X, X, y, X_train, X_test, y_train, y_test, preprocessor)


def build_pairs(preprocessor):
  # test that you can do cross validation on tuples of points with
  #  a WeaklySupervisedMetricLearner
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
  # test that you can do cross validation on a tuples of points with
  #  a WeaklySupervisedMetricLearner
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


list_estimators = [(ITML(), build_pairs),
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
                   (SDML_Supervised(), build_classification)
                   ]

ids_estimators = ['itml',
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
                  'sdml_supervised'
                  ]


@pytest.mark.parametrize('preprocessor', [None, build_data()[0]])
@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_cross_validation(estimator, build_dataset, preprocessor):
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
  (_, inputs, y, _, _,  _, _, preprocessor) = build_dataset(preprocessor)

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
  estimator.fit(inputs, y, **make_random_state(estimator, False))
  pipeline.fit(inputs, y, **make_random_state(estimator, True))

  if hasattr(estimator, 'score'):
    result = estimator.score(inputs, y)
    result_pipe = pipeline.score(inputs, y)
    assert_allclose_dense_sparse(result, result_pipe)

  if hasattr(estimator, 'predict'):
    result = estimator.predict(inputs)
    result_pipe = pipeline.predict(inputs)
    assert_allclose_dense_sparse(result, result_pipe)

  if issubclass(estimator.__class__, TransformerMixin):
    if hasattr(estimator, 'transform'):
      result = estimator.transform(inputs)
      result_pipe = pipeline.transform(inputs)
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
  for method in ["predict", "decision_function", "predict_proba"]:
    if hasattr(estimator, method):
      dict_before = estimator.__dict__.copy()
      getattr(estimator, method)(tuples)
      assert estimator.__dict__ == dict_before, \
          ("Estimator changes __dict__ during %s"
           % method)
  for method in ["transform"]:
    if hasattr(estimator, method):
      dict_before = estimator.__dict__.copy()
      # we transform only 2D arrays (dataset of points)
      getattr(estimator, method)(X)
      assert estimator.__dict__ == dict_before, \
          ("Estimator changes __dict__ during %s"
           % method)


@pytest.mark.parametrize('estimator, build_dataset',
                         [(ITML(), build_pairs),
                          (LSML(), build_quadruplets),
                          (MMC(max_iter=2), build_pairs),
                          (SDML(), build_pairs)],
                         ids=['itml', 'lsml', 'mmc', 'sdml'])
def test_same_result_with_or_without_preprocessor(estimator, build_dataset):
  (X, tuples, y, tuples_train, tuples_test, y_train,
   y_test, _) = build_dataset(preprocessor=mock_preprocessor)
  formed_tuples_train = X[tuples_train]
  formed_tuples_test = X[tuples_test]

  estimator_with_preprocessor = clone(estimator)
  set_random_state(estimator_with_preprocessor)
  estimator_with_preprocessor.set_params(preprocessor=X)
  if estimator.__class__.__name__ == 'LSML':
    estimator_with_preprocessor.fit(tuples_train)
  else:
    estimator_with_preprocessor.fit(tuples_train, y_train)

  estimator_without_preprocessor = clone(estimator)
  set_random_state(estimator_without_preprocessor)
  estimator_without_preprocessor.set_params(preprocessor=None)
  if estimator.__class__.__name__ == 'LSML':
    estimator_without_preprocessor.fit(formed_tuples_train)
  else:
    estimator_without_preprocessor.fit(formed_tuples_train, y_train)

  estimator_with_prep_formed = clone(estimator)
  set_random_state(estimator_with_prep_formed)
  estimator_with_prep_formed.set_params(preprocessor=X)
  if estimator.__class__.__name__ == 'LSML':
    estimator_with_prep_formed.fit(tuples_train)
  else:
    estimator_with_prep_formed.fit(tuples_train, y_train)

  # test prediction methods
  for method in ["predict", "decision_function"]:
    if hasattr(estimator, method):
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(tuples_test)
      output_without_prep = getattr(estimator_without_preprocessor,
                                    method)(formed_tuples_test)
      assert np.array(output_with_prep == output_without_prep).all()
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(tuples_test)
      output_with_prep_formed = getattr(estimator_with_prep_formed,
                                        method)(formed_tuples_test)
      assert np.array(output_with_prep == output_with_prep_formed).all()

  # test score_pairs
  output_with_prep = estimator_with_preprocessor.score_pairs(
      tuples_test[:, :2])
  output_without_prep = estimator_without_preprocessor.score_pairs(
      formed_tuples_test[:, :2])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.score_pairs(
      tuples_test[:, :2])
  output_without_prep = estimator_with_prep_formed.score_pairs(
      formed_tuples_test[:, :2])
  assert np.array(output_with_prep == output_without_prep).all()

  # test transform
  output_with_prep = estimator_with_preprocessor.transform(
      tuples_test[:, 0])
  output_without_prep = estimator_without_preprocessor.transform(
      formed_tuples_test[:, 0])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.transform(
      tuples_test[:, 0])
  output_without_prep = estimator_with_prep_formed.transform(
      formed_tuples_test[:, 0])
  assert np.array(output_with_prep == output_without_prep).all()


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


def _get_args(function, varargs=False):
    """Helper to get function arguments"""

    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args
