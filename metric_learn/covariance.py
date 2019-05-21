"""
Covariance metric (baseline method)

This method does not "learn" anything, rather it calculates
the covariance matrix of the input data.

This is a simple baseline method first introduced in
On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936
"""

from __future__ import absolute_import
import numpy as np
from sklearn.base import TransformerMixin

from .base_metric import MahalanobisMixin
from ._util import transformer_from_metric


class Covariance(MahalanobisMixin, TransformerMixin):
  """Covariance metric (baseline method)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)
  """

  def __init__(self, preprocessor=None):
    super(Covariance, self).__init__(preprocessor)

  def fit(self, X, y=None):
    """
    X : data matrix, (n x d)
    y : unused
    """
    X = self._prepare_inputs(X, ensure_min_samples=2)
    M = np.atleast2d(np.cov(X, rowvar = False))
    if len(M) == 1:
      M = 1./M
    else:
      M = np.linalg.pinvh(M)

    self.transformer_ = transformer_from_metric(np.atleast_2d(M))
    return self
