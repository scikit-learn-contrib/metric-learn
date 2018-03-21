from numpy.linalg import cholesky
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class TransformerMixin(object):
  """Mixin class for all transformers in metric-learn. Same as the one in
  scikit-learn, but the documentation is changed: this Transformer is
  allowed to take as y a non array-like input"""

  def __init__(self):
    raise NotImplementedError('TransformerMixin should not be instantiated')

  def fit_transform(self, X, y=None, **fit_params):
    """Fit to data, then transform it.

    Fits transformer to X and y with optional parameters fit_params
    and returns a transformed version of X.

    Parameters
    ----------
    X : numpy array of shape [n_samples, n_features]
        Training set.

    y : numpy array of shape [n_samples] or 4-tuple of arrays
        Target values, or constraints (a, b, c, d) indices into X, with
        (a, b) specifying similar and (c,d) dissimilar pairs).

    Returns
    -------
    X_new : numpy array of shape [n_samples, n_features_new]
        Transformed array.

    """
    # non-optimized default implementation; override when a better
    # method is possible for a given clustering algorithm
    if y is None:
      # fit method of arity 1 (unsupervised transformation)
      return self.fit(X, **fit_params).transform(X)
    else:
      # fit method of arity 2 (supervised transformation)
      return self.fit(X, y, **fit_params).transform(X)

class BaseMetricLearner(BaseEstimator, TransformerMixin):

  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = cholesky(M).T

    Returns
    -------
    L : upper triangular (d x d) matrix
    """
    return cholesky(self.metric()).T

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    else:
      X = check_array(X, accept_sparse=True)
    L = self.transformer()
    return X.dot(L.T)


class SupervisedMixin(object):

  def __init__(self):
    raise NotImplementedError('UnsupervisedMixin should not be instantiated')

  def fit(self, X, y):
    return NotImplementedError


class UnsupervisedMixin(object):

  def __init__(self):
    raise NotImplementedError('UnsupervisedMixin should not be instantiated')

  def fit(self, X, y=None):
    return NotImplementedError


class WeaklySupervisedMixin(object):

  def __init__(self):
    raise NotImplementedError('WeaklySupervisedMixin should not be '
                              'instantiated')

  def fit(self, X, constraints, **kwargs):
    return self._fit(X, constraints, **kwargs)


class PairsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('PairsMixin should not be instantiated')
  # TODO: introduce specific scoring functions etc


class TripletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('TripletsMixin should not be '
                              'instantiated')
  # TODO: introduce specific scoring functions etc


class QuadrupletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('QuadrupletsMixin should not be '
                              'instantiated')
  # TODO: introduce specific scoring functions etc

  def fit(self, X, constraints=None, **kwargs):
    return self._fit(X, **kwargs)

