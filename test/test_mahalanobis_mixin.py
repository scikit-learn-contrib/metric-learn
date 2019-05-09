from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy.spatial.distance import pdist, squareform, mahalanobis
from sklearn import clone
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.testing import set_random_state

from metric_learn._util import make_context
from metric_learn.base_metric import (_QuadrupletsClassifierMixin,
                                      _PairsClassifierMixin)

from test.test_utils import (ids_metric_learners, metric_learners,
                             remove_y_quadruplets, ids_regressors,
                             ids_supervised_learners, supervised_learners,
                             ids_classifiers)

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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))

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
    model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
    model.fit(*remove_y_quadruplets(estimator, input_data, labels))
    embedded_points = X.dot(model.transformer_.T)
    assert_array_almost_equal(model.transform(X), embedded_points)


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_dim(estimator, build_dataset):
  # Checks that the the dimension of the output space is as expected
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
    model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
  assert np.isfinite(model.transform(X)).all()


@pytest.mark.parametrize('estimator, build_dataset', metric_learners,
                         ids=ids_metric_learners)
def test_embed_is_linear(estimator, build_dataset):
  # Checks that the embedding is linear
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))

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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
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
  input_data, labels, _, X = build_dataset()
  model = clone(estimator)
  set_random_state(model)
  # test that it works for X.shape[1] features
  model.fit(*remove_y_quadruplets(estimator, input_data, labels))
  assert model.transformer_.shape == (X.shape[1], X.shape[1])

  # test that it works for 1 feature
  trunc_data = input_data[..., :1]
  # we drop duplicates that might have been formed, i.e. of the form
  # aabc or abcc or aabb for quadruplets, and aa for pairs.
  if isinstance(estimator, _QuadrupletsClassifierMixin):
    for slice_idx in [slice(0, 2), slice(2, 4)]:
      pairs = trunc_data[:, slice_idx, :]
      diffs = pairs[:, 1, :] - pairs[:, 0, :]
      to_keep = np.where(np.abs(diffs.ravel()) > 1e-9)
      trunc_data = trunc_data[to_keep]
      labels = labels[to_keep]
  elif isinstance(estimator, _PairsClassifierMixin):
    diffs = trunc_data[:, 1, :] - trunc_data[:, 0, :]
    to_keep = np.where(np.abs(diffs.ravel()) > 1e-9)
    trunc_data = trunc_data[to_keep]
    labels = labels[to_keep]
  model.fit(*remove_y_quadruplets(estimator, trunc_data, labels))
  assert model.transformer_.shape == (1, 1)  # the transformer must be 2D


@pytest.mark.parametrize('estimator, build_dataset',
                         [(ml, bd) for idml, (ml, bd)
                          in zip(ids_metric_learners,
                                 metric_learners)
                          if hasattr(ml, 'num_dims') and
                          hasattr(ml, 'init')],
                         ids=[idml for idml, (ml, _)
                              in zip(ids_metric_learners,
                                     metric_learners)
                              if hasattr(ml, 'num_dims') and
                              hasattr(ml, 'init')])
def test_init_transformation(estimator, build_dataset):
    input_data, labels, _, X = build_dataset()
    is_classification = (type_of_target(labels) in ['multiclass', 'binary'])
    model = clone(estimator)
    rng = np.random.RandomState(42)

    # Start learning from scratch
    model.set_params(init='identity')
    model.fit(input_data, labels)

    # Initialize with random
    model.set_params(init='random')
    model.fit(input_data, labels)

    # Initialize with auto
    model.set_params(init='auto')
    model.fit(input_data, labels)

    # Initialize with PCA
    model.set_params(init='pca')
    model.fit(input_data, labels)

    # Initialize with LDA
    if is_classification:
      model.set_params(init='lda')
      model.fit(input_data, labels)

    # Initialize with a numpy array
    init = rng.rand(X.shape[1], X.shape[1])
    model.set_params(init=init)
    model.fit(input_data, labels)

    # init.shape[1] must match X.shape[1]
    init = rng.rand(X.shape[1], X.shape[1] + 1)
    model.set_params(init=init)
    msg = ('The input dimensionality ({}) of the given '
           'linear transformation `init` must match the '
           'dimensionality of the given inputs `X` ({}).'
           .format(init.shape[1], X.shape[1]))
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg

    # init.shape[0] must be <= init.shape[1]
    init = rng.rand(X.shape[1] + 1, X.shape[1])
    model.set_params(init=init)
    msg = ('The output dimensionality ({}) of the given '
           'linear transformation `init` cannot be '
           'greater than its input dimensionality ({}).'
           .format(init.shape[0], init.shape[1]))
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg

    # init.shape[0] must match num_dims
    init = rng.rand(X.shape[1], X.shape[1])
    num_dims = X.shape[1] - 1
    model.set_params(init=init, num_dims=num_dims)
    msg = ('The preferred dimensionality of the '
           'projected space `num_dims` ({}) does not match '
           'the output dimensionality of the given '
           'linear transformation `init` ({})!'
           .format(num_dims, init.shape[0]))
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg

    # init must be as specified in the docstring
    model.set_params(init=1)
    msg = ("`init` must be 'auto', 'pca', 'identity', "
           "'random'{} or a numpy array of shape "
           "(num_dims, n_features)."
           .format(", 'lda'" if is_classification else ''))
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg


@pytest.mark.parametrize('n_samples', [3, 5, 7, 11])
@pytest.mark.parametrize('n_features', [3, 5, 7, 11])
@pytest.mark.parametrize('n_classes', [5, 7, 11])
@pytest.mark.parametrize('num_dims', [3, 5, 7, 11])
@pytest.mark.parametrize('estimator, build_dataset',
                         [(ml, bd) for idml, (ml, bd)
                          in zip(ids_metric_learners,
                                 metric_learners)
                          if hasattr(ml, 'num_dims') and
                          hasattr(ml, 'init')],
                         ids=[idml for idml, (ml, _)
                              in zip(ids_metric_learners,
                                     metric_learners)
                              if hasattr(ml, 'num_dims') and
                              hasattr(ml, 'init')])
def test_auto_init_transformation(n_samples, n_features, n_classes, num_dims,
                                  estimator, build_dataset):
  # Test that auto choose the init transformation as expected with every
  # configuration of order of n_samples, n_features, n_classes and num_dims,
  # for all metric learners that learn a transformation.
  if n_classes >= n_samples:
    pass
    # n_classes > n_samples is impossible, and n_classes == n_samples
    # throws an error from lda but is an absurd case
  else:
    input_data, labels, _, X = build_dataset()
    model_base = clone(estimator)
    rng = np.random.RandomState(42)
    model_base.set_params(init='auto',
                          num_dims=num_dims,
                          random_state=rng)
    # To make the test work for LMNN:
    if 'LMNN' in model_base.__class__.__name__:
      model_base.set_params(k=1)
    # To make the test faster for estimators that have a max_iter:
    if hasattr(model_base, 'max_iter'):
      model_base.set_params(max_iter=1)
    if num_dims > n_features:
      # this would return a ValueError, which is tested in
      # test_init_transformation
      pass
    else:
      # We need to build a dataset of the right shape:
      num_to_pad_n_samples = ((n_samples // input_data.shape[0] + 1))
      num_to_pad_n_features = ((n_samples // input_data.shape[-1] + 1))
      if input_data.ndim == 3:
        input_data = np.tile(input_data,
                             (num_to_pad_n_samples, input_data.shape[1],
                              num_to_pad_n_features))
      else:
        input_data = np.tile(input_data,
                             (num_to_pad_n_samples, num_to_pad_n_features))
      input_data = input_data[:n_samples, ..., :n_features]
      has_classes = model_base.__class__.__name__ in ids_classifiers
      if has_classes:
        labels = np.tile(range(n_classes), n_samples //
                          n_classes + 1)[:n_samples]
      else:
        labels = np.tile(labels, n_samples // labels.shape[0] + 1)[:n_samples]
      model = clone(model_base)
      model.fit(input_data, labels)
      if num_dims <= min(n_classes - 1, n_features) and has_classes:
        model_other = clone(model_base).set_params(init='lda')
      elif num_dims < min(n_features, n_samples):
        model_other = clone(model_base).set_params(init='pca')
      else:
        model_other = clone(model_base).set_params(init='identity')
      model_other.fit(input_data, labels)
      assert_array_almost_equal(model.transformer_,
                                model_other.transformer_)


@pytest.mark.parametrize('estimator, build_dataset',
                         [(ml, bd) for idml, (ml, bd)
                          in zip(ids_metric_learners,
                                 metric_learners)
                          if not hasattr(ml, 'num_dims') and
                          hasattr(ml, 'init')],
                         ids=[idml for idml, (ml, _)
                              in zip(ids_metric_learners,
                                     metric_learners)
                              if not hasattr(ml, 'num_dims') and
                              hasattr(ml, 'init')])
def test_init_mahalanobis(estimator, build_dataset):
    """Tests that for estimators that learn a mahalanobis matrix
    instead of a transformer, i.e. those that are mahalanobis metric learners
    where we can change the init, but not choose the num_dims, (TODO: be more
    explicit on this characterization, for instance with safe_flags like in
    scikit-learn) that the init has an expected behaviour.
    """
    input_data, labels, _, X = build_dataset()
    model = clone(estimator)
    rng = np.random.RandomState(42)

    # Start learning from scratch
    model.set_params(init='identity')
    model.fit(input_data, labels)

    # Initialize with random
    model.set_params(init='random')
    model.fit(input_data, labels)

    # Initialize with covariance
    model.set_params(init='covariance')
    model.fit(input_data, labels)

    # Initialize with a random spd matrix
    init = make_spd_matrix(X.shape[1], random_state=rng)
    model.set_params(init=init)
    model.fit(input_data, labels)

    # init.shape[1] must match X.shape[1]
    init = make_spd_matrix(X.shape[1] + 1, X.shape[1] + 1)
    model.set_params(init=init)
    msg = ('The input dimensionality {} of the given '
           'mahalanobis matrix `init` must match the '
           'dimensionality of the given inputs ({}).'
           .format(init.shape, input_data.shape[-1]))
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg

    # The input matrix must be symmetric
    init = rng.rand(X.shape[1], X.shape[1])
    model.set_params(init=init)
    msg = ("The initialization matrix should be semi-definite "
           "positive (SPD). It is not, since it appears not to be "
           "symmetric.")
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg

    # init must be as specified in the docstring
    model.set_params(init=1)
    msg = ("`init` must be 'identity', 'covariance', "
           "'random' or a numpy array of shape "
           "(n_features, n_features).")
    with pytest.raises(ValueError) as raised_error:
      model.fit(input_data, labels)
    assert str(raised_error.value) == msg
