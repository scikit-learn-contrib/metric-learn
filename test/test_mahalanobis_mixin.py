from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.distance import pdist, squareform
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.testing import set_random_state

from metric_learn import (Constraints, ITML, LSML, MMC, SDML, Covariance, LFDA,
                          LMNN, MLKR, NCA, RCA)
from metric_learn.constraints import wrap_pairs
from functools import partial

RNG = check_random_state(0)

def build_data():
  dataset = load_iris()
  X, y = shuffle(dataset.data, dataset.target, random_state=RNG)
  num_constraints = 20
  constraints = Constraints.random_subset(y, random_state=RNG)
  pairs = constraints.positive_negative_pairs(num_constraints,
                                              same_length=True,
                                              random_state=RNG)
  return X, pairs


def build_pairs():
  # test that you can do cross validation on tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  pairs, y = wrap_pairs(X, pairs)
  pairs, y = shuffle(pairs, y, random_state=RNG)
  return pairs, y


def build_quadruplets():
  # test that you can do cross validation on a tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  c = np.column_stack(pairs)
  quadruplets = X[c]
  quadruplets = shuffle(quadruplets, random_state=RNG)
  return quadruplets, None


list_estimators = [(Covariance(), build_data),
                   (ITML(), build_pairs),
                   (LFDA(), partial(load_iris, return_X_y=True)),
                   (LMNN(), partial(load_iris, return_X_y=True)),
                   (LSML(), build_quadruplets),
                   (MLKR(), partial(load_iris, return_X_y=True)),
                   (MMC(), build_pairs),
                   (NCA(), partial(load_iris, return_X_y=True)),
                   (RCA(), partial(load_iris, return_X_y=True)),
                   (SDML(), build_pairs)
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
                  ]


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_pairs_pairwise(estimator, build_dataset):
  # Computing pairwise scores should return a euclidean distance matrix.
  inputs, labels = build_dataset()
  X, _ = load_iris(return_X_y=True)
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)

  pairwise = model.score_pairs(np.array(list(product(X, X))))\
      .reshape(n_samples, n_samples)

  check_is_distance_matrix(pairwise)

  # a necessary condition for euclidean distance matrices: (see
  # https://en.wikipedia.org/wiki/Euclidean_distance_matrix)
  assert np.linalg.matrix_rank(pairwise**2) <= min(X.shape) + 2

  # assert that this distance is coherent with pdist on embeddings
  assert_array_almost_equal(squareform(pairwise), pdist(model.transform(X)))


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_pairs_toy_example(estimator, build_dataset):
    # Checks that score_pairs works on a toy example
    inputs, labels = build_dataset()
    X, _ = load_iris(return_X_y=True)
    n_samples = 20
    X = X[:n_samples]
    model = clone(estimator)
    set_random_state(model)
    model.fit(inputs, labels)
    pairs = np.stack([X[:10], X[10:20]], axis=1)
    embedded_pairs = pairs.dot(model.transformer_.T)
    distances = np.sqrt(np.sum((embedded_pairs[:, 1] -
                               embedded_pairs[:, 0])**2,
                               axis=-1))
    assert_array_almost_equal(model.score_pairs(pairs), distances)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_pairs_finite(estimator, build_dataset):
  # tests that the score is finite
  inputs, labels = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  pairs = np.array(list(product(X, X)))
  assert np.isfinite(model.score_pairs(pairs)).all()


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_pairs_dim(estimator, build_dataset):
  # scoring of 3D arrays should return 1D array (several tuples),
  # and scoring of 2D arrays (one tuple) should return an error (like
  # scikit-learn's error when scoring 1D arrays)
  inputs, labels = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  tuples = np.array(list(product(X, X)))
  assert model.score_pairs(tuples).shape == (tuples.shape[0],)
  msg = ("Expected 3D array, got 2D array instead:\ntuples={}.\n"
         "Reshape your data either using tuples.reshape(-1, {}, 1) if "
         "your data has a single feature or tuples.reshape(1, {}, -1) "
         "if it contains a single tuple.".format(tuples, tuples.shape[1],
                                                 tuples.shape[0]))
  with pytest.raises(ValueError) as raised_error:
    model.score_pairs(tuples[1])
  assert str(raised_error.value) == msg


def check_is_distance_matrix(pairwise):
  assert (pairwise >= 0).all()  # positivity
  assert np.array_equal(pairwise, pairwise.T)  # symmetry
  assert (pairwise.diagonal() == 0).all()  # identity
  # triangular inequality
  tol = 1e-15
  assert (pairwise <= pairwise[:, :, np.newaxis]
          + pairwise[:, np.newaxis, :] + tol).all()


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_embed_toy_example(estimator, build_dataset):
    # Checks that embed works on a toy example
    inputs, labels = build_dataset()
    X, _ = load_iris(return_X_y=True)
    n_samples = 20
    X = X[:n_samples]
    model = clone(estimator)
    set_random_state(model)
    model.fit(inputs, labels)
    embedded_points = X.dot(model.transformer_.T)
    assert_array_almost_equal(model.transform(X), embedded_points)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_embed_dim(estimator, build_dataset):
  # Checks that the the dimension of the output space is as expected
  inputs, labels = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  assert model.transform(X).shape == X.shape

  # assert that ValueError is thrown if input shape is 1D
  err_msg = ("Expected 2D array, got 1D array instead:\narray={}.\n"
             "Reshape your data either using array.reshape(-1, 1) if "
             "your data has a single feature or array.reshape(1, -1) "
             "if it contains a single sample.".format(X))
  with pytest.raises(ValueError) as raised_error:
    model.score_pairs(model.transform(X[0, :]))
  assert str(raised_error.value) == err_msg
  # we test that the shape is also OK when doing dimensionality reduction
  if type(model).__name__ in {'LFDA', 'MLKR', 'NCA', 'RCA'}:
    model.set_params(num_dims=2)
    model.fit(inputs, labels)
    assert model.transform(X).shape == (X.shape[0], 2)
    # assert that ValueError is thrown if input shape is 1D
    with pytest.raises(ValueError) as raised_error:
        model.transform(model.transform(X[0, :]))
    assert str(raised_error.value) == err_msg


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_embed_finite(estimator, build_dataset):
  # Checks that embed returns vectors with finite values
  inputs, labels = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  assert np.isfinite(model.transform(X)).all()


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_embed_is_linear(estimator, build_dataset):
  # Checks that the embedding is linear
  inputs, labels = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  assert_array_almost_equal(model.transform(X[:10] + X[10:20]),
                            model.transform(X[:10]) +
                            model.transform(X[10:20]))
  assert_array_almost_equal(model.transform(5 * X[:10]),
                            5 * model.transform(X[:10]))
