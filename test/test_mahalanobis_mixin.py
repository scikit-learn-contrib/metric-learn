from itertools import product

import pytest
import numpy as np
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state, shuffle

from metric_learn import (Constraints, ITML, LSML, MMC, SDML, Covariance, LFDA,
                          LMNN, MLKR, NCA, RCA)
from metric_learn.constraints import wrap_pairs
from functools import partial


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
  # test that you can do cross validation on tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  pairs, y = wrap_pairs(X, pairs)
  pairs, y = shuffle(pairs, y)
  return (pairs, y)


def build_quadruplets():
  # test that you can do cross validation on a tuples of points with
  #  a WeaklySupervisedMetricLearner
  X, pairs = build_data()
  c = np.column_stack(pairs)
  quadruplets = X[c]
  quadruplets = shuffle(quadruplets)
  return (quadruplets, None)


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
def test_score_pairwise(estimator, build_dataset):
  # Computing pairwise scores should return an euclidean distance matrix.
  inputs, labels = build_dataset()
  X, _ = load_iris(return_X_y=True)
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  model.fit(inputs, labels)

  pairwise = model.score_pairs(np.array(list(product(X, X))))\
      .reshape(n_samples, n_samples)

  check_is_distance_matrix(pairwise)

  # a necessary condition for euclidean distances matrix: (see
  # https://en.wikipedia.org/wiki/Euclidean_distance_matrix)
  assert np.linalg.matrix_rank(pairwise**2) <= min(X.shape) + 2


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_toy_example(estimator, build_dataset):
    # Checks that score_pairs works on a toy example
    inputs, labels = build_dataset()
    X, _ = load_iris(return_X_y=True)
    n_samples = 20
    X = X[:n_samples]
    model = clone(estimator)
    model.fit(inputs, labels)
    pairs = np.stack([X[:10], X[10:20]], axis=1)
    embedded_pairs = pairs.dot(model.transformer_.T)
    distances = np.sqrt(np.sum((embedded_pairs[:, 1] -
                               embedded_pairs[:, 0])**2,
                               axis=1))
    np.array_equal(model.score_pairs(pairs), distances)


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def test_score_finite(estimator, build_dataset):
  # tests that the score is finite
  inputs, labels = build_dataset()
  model = clone(estimator)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  pairs = np.array(list(product(X, X)))
  assert np.isfinite(model.score_pairs(pairs)).all()


@pytest.mark.parametrize('estimator, build_dataset', list_estimators,
                         ids=ids_estimators)
def tests_score_dim(estimator, build_dataset):
  # scoring of 3D arrays should return 1D array (several pairs),
  # and scoring of 2D arrays (one pair) should return a scalar (0D array).
  inputs, labels = build_dataset()
  model = clone(estimator)
  model.fit(inputs, labels)
  X, _ = load_iris(return_X_y=True)
  pairs = np.array(list(product(X, X)))
  assert model.score_pairs(pairs).shape == (pairs.shape[0],)
  assert np.isscalar(model.score_pairs(pairs[1]))


def check_is_distance_matrix(pairwise):
  assert (pairwise >= 0).all()  # positivity
  assert np.array_equal(pairwise, pairwise.T)  # symmetry
  assert (pairwise.diagonal() == 0).all()  # identity
  # triangular inequality
  for i in range(pairwise.shape[1]):
    for j in range(pairwise.shape[1]):
      for k in range(pairwise.shape[1]):
        assert (pairwise[i, j] - (pairwise[i, k] + pairwise[k, j]) <= 0 +
                1e-3).all()
