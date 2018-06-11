"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y, check_is_fitted

from .base_metric import (BaseMetricLearner, MahalanobisMixin,
                         MetricTransformer)

EPS = np.finfo(float).eps


class NCA(BaseMetricLearner, MahalanobisMixin,
          MetricTransformer):
  def __init__(self, num_dims=None, max_iter=100, learning_rate=0.01):
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.learning_rate = learning_rate

  def transformer(self):
    return self.A_

  @property
  def metric_(self):
    check_is_fitted(self, 'A_')
    return self.A_.T.dot(self.A_)

  def fit(self, X, y):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    X, labels = check_X_y(X, y)
    n, d = X.shape
    num_dims = self.num_dims
    if num_dims is None:
        num_dims = d
    # Initialize A to a scaling matrix
    A = np.zeros((num_dims, d))
    np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0)-X.min(axis=0), EPS)))

    # Run NCA
    dX = X[:,None] - X[None]  # shape (n, n, d)
    tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)
    masks = labels[:,None] == labels[None]
    for it in xrange(self.max_iter):
      for i, label in enumerate(labels):
        mask = masks[i]
        Ax = A.dot(X.T).T  # shape (n, num_dims)

        softmax = np.exp(-((Ax[i] - Ax)**2).sum(axis=1))  # shape (n)
        softmax[i] = 0
        softmax /= softmax.sum()

        t = softmax[:, None, None] * tmp[i]  # shape (n, d, d)
        d = softmax[mask].sum() * t.sum(axis=0) - t[mask].sum(axis=0)
        A += self.learning_rate * A.dot(d)

    self.X_ = X
    self.A_ = A
    self.n_iter_ = it
    return self
