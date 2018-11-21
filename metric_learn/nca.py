"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.base import TransformerMixin

from .base_metric import MahalanobisMixin

EPS = np.finfo(float).eps


class NCA(MahalanobisMixin, TransformerMixin):
  """Neighborhood Components Analysis (NCA)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims=None, max_iter=100, learning_rate=0.01,
               preprocessor=None):
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    super(NCA, self).__init__(preprocessor)

  def fit(self, X, y):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    X, labels = self._prepare_inputs(X, y, ensure_min_samples=2)
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
    self.transformer_ = A
    self.n_iter_ = it
    return self
