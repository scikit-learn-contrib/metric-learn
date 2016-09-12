"""
Covariance metric (baseline method)

This method does not "learn" anything, rather it calculates
the covariance matrix of the input data.

This is a simple baseline method first introduced in
On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936
"""

from __future__ import absolute_import
import numpy as np

from .base_metric import BaseMetricLearner


class Covariance(BaseMetricLearner):
  def __init__(self):
    self.params = {}

  def metric(self):
    return self.M

  def fit(self, X, y=None):
    """
    X: data matrix, (n x d)
    y: unused, optional
    """
    self.X = X
    self.M = np.cov(X.T)
    return self
