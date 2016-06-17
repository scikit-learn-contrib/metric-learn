"""
Simple Covariance metric (learner)
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
    y: labels (optional)
    """
    self.M = np.cov(X.T)
    return self
