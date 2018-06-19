"""
Covariance metric (baseline method)

This method does not "learn" anything, rather it calculates
the covariance matrix of the input data.

This is a simple baseline method first introduced in
On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936
"""

from __future__ import absolute_import
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from .base_metric import (BaseMetricLearner, MahalanobisMixin,
                          MetricTransformer)


class Covariance(BaseMetricLearner, MetricTransformer,
                 MahalanobisMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    """
    X : data matrix, (n x d)
    y : unused
    """
    self.X_ = check_array(X, ensure_min_samples=2)
    self.M_ = np.cov(self.X_, rowvar = False)
    if self.M_.ndim == 0:
      self.M_ = 1./self.M_
    else:
      self.M_ = np.linalg.inv(self.M_)

    self.transformer_ = self.transformer_from_metric(check_array(self.M_))
    return self
