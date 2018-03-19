from sklearn.metrics import roc_auc_score

from metric_learn.constraints import ConstrainedDataset
from numpy.linalg import cholesky
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
import numpy as np

class BaseMetricLearner(BaseEstimator):

  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')


  def fit_transform(self, X, y=None, **fit_params):
    """Fit to data, then transform it.

    Fits transformer to X and y with optional parameters fit_params
    and returns a transformed version of X.

    Parameters
    ----------
    X : array-like of shape [n_samples, n_features], or ConstrainedDataset
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
    X : (n x d) matrix or ConstrainedDataset, optional
        Data to transform. If not supplied, the training data will be used.
        In the case of a ConstrainedDataset, X_constrained.X is used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    elif type(X) is ConstrainedDataset:
      X = X.X
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

  def decision_function(self, X_constrained):
      return self.predict(X_constrained)


class PairsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('PairsMixin should not be instantiated')

  def predict(self, X_constrained):
    # TODO: provide better implementation
    pairwise_diffs = (X_constrained.X[X_constrained.c[:, 0]] -
                      X_constrained.X[X_constrained.c[:, 1]])
    return np.sqrt(np.sum(pairwise_diffs.dot(self.metric()) * pairwise_diffs,
                                  axis=1))

  def score(self, X_constrained, y):
      return roc_auc_score(y, self.decision_function(X_constrained))


class TripletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('TripletsMixin should not be '
                              'instantiated')

  def predict(self, X_constrained):
    # TODO: provide better implementation
    similar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                    X_constrained.X[X_constrained.c[:, 1]]
    dissimilar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                       X_constrained.X[X_constrained.c[:, 2]]
    return np.sqrt(np.sum(similar_diffs.dot(self.metric()) *
                          similar_diffs, axis=1)) - \
           np.sqrt(np.sum(dissimilar_diffs.dot(self.metric()) *
                          dissimilar_diffs, axis=1))

  def score(self, X_constrained, y=None):
    predicted_sign = self.decision_function(X_constrained) < 0
    return np.sum(predicted_sign) / predicted_sign.shape[0]



class QuadrupletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('QuadrupletsMixin should not be '
                              'instantiated')

  def fit(self, X, constraints=None, **kwargs):
    return self._fit(X, **kwargs)

  def predict(self, X_constrained):
    similar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                    X_constrained.X[X_constrained.c[:, 1]]
    dissimilar_diffs = X_constrained.X[X_constrained.c[:, 2]] - \
                       X_constrained.X[X_constrained.c[:, 3]]
    return np.sqrt(np.sum(similar_diffs.dot(self.metric()) *
                          similar_diffs, axis=1)) - \
           np.sqrt(np.sum(dissimilar_diffs.dot(self.metric()) *
                          dissimilar_diffs, axis=1))

  def decision_fuction(self, X_constrained):
      return self.predict(X_constrained)

  def score(self, X_constrained, y=None):
    predicted_sign = self.decision_function(X_constrained) < 0
    return np.sum(predicted_sign) / predicted_sign.shape[0]