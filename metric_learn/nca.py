"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

import numpy as np
from base_metric import BaseMetricLearner


class NCA(BaseMetricLearner):
  def __init__(self, max_iter=100, learning_rate=0.01):
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.A = None

  def transformer(self):
    return self.A

  def fit(self, X, labels):
    """
    X: data matrix, (n x d)
    labels: scalar labels, (n)
    """
    n, d = X.shape
    # Initialize A to a scaling matrix
    A = np.zeros((d, d))
    np.fill_diagonal(A, 1./(X.max(axis=0)-X.min(axis=0)))

    # Run NCA
    dX = X[:,None] - X[None]  # shape (n, n, d)
    tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)
    for it in xrange(self.max_iter):
      for i, label in enumerate(labels):
        mask = labels == label
        Ax = A.dot(X.T).T  # shape (n, d)

        softmax = np.exp(-((Ax[i] - Ax)**2).sum(axis=1))  # shape (n)
        softmax[i] = 0
        softmax /= softmax.sum()

        t = softmax[:, None, None] * tmp[i]  # shape (n, d, d)
        first_term = softmax[mask].sum() * t.sum(axis=0)  # shape (d, d)
        second_term = t[mask].sum(axis=0)  # shape (d, d)
        A += self.learning_rate * A.dot(first_term - second_term)

    self.X = X
    self.A = A
    return self
