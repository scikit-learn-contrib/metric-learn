from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy.spatial.distance import pdist, squareform, mahalanobis
from sklearn import clone
from sklearn.cluster import DBSCAN
from sklearn.utils import check_random_state
from sklearn.utils.testing import set_random_state

from metric_learn._util import make_context

from test.test_utils import ids_metric_learners, metric_learners

RNG = check_random_state(0)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_score_pairs_pairwise(estimator, build_dataset):
  # Computing pairwise scores should return a euclidean distance matrix.
  input_data, labels, _, X = build_dataset()
  n_samples = 20
  X = X[:n_samples]
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)

  pairwise = model.score_pairs(np.array(list(product(X, X))))\
      .reshape(n_samples, n_samples)

  check_is_distance_matrix(pairwise)

  # a necessary condition for euclidean distance matrices: (see
  # https://en.wikipedia.org/wiki/Euclidean_distance_matrix)
  assert np.linalg.matrix_rank(pairwise**2) <= min(X.shape) + 2

  # assert that this distance is coherent with pdist on embeddings
  assert_array_almost_equal(squareform(pairwise), pdist(model.transform(X)))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_score_pairs_toy_example(estimator, build_dataset):
    # Checks that score_pairs works on a toy example
    input_data, labels, _, X = build_dataset()
    n_samples = 20
    X = X[:n_samples]
    model = clone(estimator)
    set_random_state(model)
    model.fit(input_data, labels)
    pairs = np.stack([X[:10], X[10:20]], axis=1)
    embedded_pairs = pairs.dot(model.transformer_.T)
    distances = np.sqrt(np.sum((embedded_pairs[:, 1] -
                               embedded_pairs[:, 0])**2,
                               axis=-1))
    assert_array_almost_equal(model.score_pairs(pairs), distances)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_score_pairs_finite(estimator, build_dataset):
  # tests that the score is finite
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  pairs = np.array(list(product(X, X)))
  assert np.isfinite(model.score_pairs(pairs)).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_score_pairs_dim(estimator, build_dataset):
  # scoring of 3D arrays should return 1D array (several tuples),
  # and scoring of 2D arrays (one tuple) should return an error (like
  # scikit-learn's error when scoring 1D arrays)
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  tuples = np.array(list(product(X, X)))
  assert model.score_pairs(tuples).shape == (tuples.shape[0],)
  context = make_context(estimator)
  msg = ("3D array of formed tuples expected{}. Found 2D array "
         "instead:\ninput={}. Reshape your data and/or use a preprocessor.\n"
         .format(context, tuples[1]))
  with pytest.raises(ValueError) as raised_error:
    model.score_pairs(tuples[1])
  assert str(raised_error.value) == msg


def check_is_distance_matrix(pairwise):
  assert (pairwise >= 0).all()  # positivity
  assert np.array_equal(pairwise, pairwise.T)  # symmetry
  assert (pairwise.diagonal() == 0).all()  # identity
  # triangular inequality
  tol = 1e-12
  assert (pairwise <= pairwise[:, :, np.newaxis] +
          pairwise[:, np.newaxis, :] + tol).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_toy_example(estimator, build_dataset):
    # Checks that embed works on a toy example
    input_data, labels, _, X = build_dataset()
    n_samples = 20
    X = X[:n_samples]
    model = clone(estimator)
    set_random_state(model)
    model.fit(input_data, labels)
    embedded_points = X.dot(model.transformer_.T)
    assert_array_almost_equal(model.transform(X), embedded_points)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_dim(estimator, build_dataset):
  # Checks that the the dimension of the output space is as expected
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  assert model.transform(X).shape == X.shape

  # assert that ValueError is thrown if input shape is 1D
  context = make_context(estimator)
  err_msg = ("2D array of formed points expected{}. Found 1D array "
             "instead:\ninput={}. Reshape your data and/or use a "
             "preprocessor.\n".format(context, X[0]))
  with pytest.raises(ValueError) as raised_error:
    model.score_pairs(model.transform(X[0, :]))
  assert str(raised_error.value) == err_msg
  # we test that the shape is also OK when doing dimensionality reduction
  if type(model).__name__ in {'LFDA', 'MLKR', 'NCA', 'RCA'}:
    model.set_params(num_dims=2)
    model.fit(input_data, labels)
    assert model.transform(X).shape == (X.shape[0], 2)
    # assert that ValueError is thrown if input shape is 1D
    with pytest.raises(ValueError) as raised_error:
        model.transform(model.transform(X[0, :]))
    assert str(raised_error.value) == err_msg


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_finite(estimator, build_dataset):
  # Checks that embed returns vectors with finite values
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  assert np.isfinite(model.transform(X)).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_is_linear(estimator, build_dataset):
  # Checks that the embedding is linear
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  assert_array_almost_equal(model.transform(X[:10] + X[10:20]),
                            model.transform(X[:10]) +
                            model.transform(X[10:20]))
  assert_array_almost_equal(model.transform(5 * X[:10]),
                            5 * model.transform(X[:10]))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_equivalent_to_explicit_mahalanobis(estimator,
                                                       build_dataset):
  """Tests that using the get_metric method of mahalanobis metric learners is
  equivalent to explicitely calling scipy's mahalanobis metric
  """
  rng = np.random.RandomState(42)
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  metric = model.get_metric()
  n_features = X.shape[1]
  a, b = (rng.randn(n_features), rng.randn(n_features))
  expected_dist = mahalanobis(a[None], b[None],
                              VI=model.get_mahalanobis_matrix())
  assert_allclose(metric(a, b), expected_dist, rtol=1e-15)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_is_pseudo_metric(estimator, build_dataset):
  """Tests that the get_metric method of mahalanobis metric learners returns a
  pseudo-metric (metric but without one side of the equivalence of
  the identity of indiscernables property)
  """
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  metric = model.get_metric()

  n_features = X.shape[1]
  for seed in range(10):
    rng = np.random.RandomState(seed)
    a, b, c = (rng.randn(n_features) for _ in range(3))
    assert metric(a, b) >= 0  # positivity
    assert metric(a, b) == metric(b, a)  # symmetry
    # one side of identity indiscernables: x == y => d(x, y) == 0. The other
    # side of the equivalence is not always true for Mahalanobis distances.
    assert metric(a, a) == 0
    # triangular inequality
    assert (metric(a, c) < metric(a, b) + metric(b, c) or
            np.isclose(metric(a, c), metric(a, b) + metric(b, c), rtol=1e-20))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_metric_raises_deprecation_warning(estimator, build_dataset):
  """assert that a deprecation warning is raised if someones wants to call
  the `metric` function"""
  # TODO: remove this method in version 0.6.0
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)

  with pytest.warns(DeprecationWarning) as raised_warning:
    model.metric()
  assert (str(raised_warning[0].message) ==
          ("`metric` is deprecated since version 0.5.0 and will be removed "
           "in 0.6.0. Use `get_mahalanobis_matrix` instead."))


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_metric_compatible_with_scikit_learn(estimator, build_dataset):
  """Check that the metric returned by get_metric is compatible with
  scikit-learn's algorithms using a custom metric, DBSCAN for instance"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  clustering = DBSCAN(metric=model.get_metric())
  clustering.fit(X)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_get_squared_metric(estimator, build_dataset):
  """Test that the squared metric returned is indeed the square of the
  metric"""
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(input_data, labels)
  metric = model.get_metric()

  n_features = X.shape[1]
  for seed in range(10):
    rng = np.random.RandomState(seed)
    a, b = (rng.randn(n_features) for _ in range(2))
    assert_allclose(metric(a, b, squared=True),
                    metric(a, b, squared=False)**2,
                    rtol=1e-15)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_transformer_is_2D(estimator, build_dataset):
  """Tests that the transformer of metric learners is 2D"""
  # TODO: remove this check when SDML has become robust to 1D elements,
  #  or when the 1D case is dealt with separately
  if not str(estimator).startswith('SDML'):
    input_data, labels, _, X = build_dataset()
    model = clone(estimator)
    set_random_state(model)
    # test that it works for X.shape[1] features
    model.fit(input_data, labels)
    assert model.transformer_.shape == (X.shape[1], X.shape[1])

    # test that it works for 1 feature
    trunc_data = input_data[..., :1]
    model.fit(trunc_data, labels)
    assert model.transformer_.shape == (1, 1)  # the transformer must be 2D
