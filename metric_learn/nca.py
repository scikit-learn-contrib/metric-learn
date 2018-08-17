"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

from __future__ import absolute_import

import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y
from numpy.linalg import multi_dot

try:  # scipy.misc.logsumexp is deprecated in scipy 1.0.0
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .base_metric import BaseMetricLearner

EPS = np.finfo(float).eps


class NCA(BaseMetricLearner):
  def __init__(self, num_dims=None, max_iter=100, learning_rate='deprecated',
               tol=None):
    """Neighborhood Components Analysis

    Parameters
    ----------
    num_dims : int, optional (default=None)
      Embedding dimensionality. If None, will be set to ``n_features``
      (``d``) at fit time.

    max_iter : int, optional (default=100)
      Maximum number of iterations done by the optimization algorithm.

    learning_rate : Not used

      .. deprecated:: 0.4.0
        `learning_rate` was deprecated in version 0.4.0 and will
        be removed in 0.5.0. The current optimization algorithm does not need
        to fix a learning rate.

    tol : float, optional (default=None)
        Convergence tolerance for the optimization.
    """
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.learning_rate = learning_rate  # TODO: remove in v.0.5.0
    self.tol = tol

  def transformer(self):
    return self.A_

  def fit(self, X, y):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    if self.learning_rate != 'deprecated':
      warnings.warn('"learning_rate" parameter is not used.'
                    ' It has been deprecated in version 0.4 and will be'
                    'removed in 0.5', DeprecationWarning)

    X, labels = check_X_y(X, y)
    n, d = X.shape
    num_dims = self.num_dims
    if num_dims is None:
        num_dims = d

    # Initialize A to a scaling matrix
    A = np.zeros((num_dims, d))
    np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0)-X.min(axis=0), EPS)))

    # Run NCA
    mask = labels[:, np.newaxis] == labels[np.newaxis, :]
    optimizer_params = {'method': 'L-BFGS-B',
                        'fun': self._loss_grad_lbfgs,
                        'args': (X, mask, -1.0),
                        'jac': True,
                        'x0': A.ravel(),
                        'options': dict(maxiter=self.max_iter),
                        'tol': self.tol
                        }

    # Call the optimizer
    opt_result = minimize(**optimizer_params)

    self.X_ = X
    self.A_ = opt_result.x.reshape(-1, X.shape[1])
    self.n_iter_ = opt_result.nit
    return self

  @staticmethod
  def _loss_grad_lbfgs(A, X, mask, sign=1.0):
    A = A.reshape(-1, X.shape[1])
    X_embedded = np.dot(X, A.T)  # (n_samples, num_dims)
    # Compute softmax distances
    p_ij = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(p_ij, np.inf)
    p_ij = np.exp(-p_ij - logsumexp(-p_ij, axis=1)[:, np.newaxis])
    # (n_samples, n_samples)

    # Compute loss
    masked_p_ij = p_ij * mask
    p = masked_p_ij.sum(axis=1, keepdims=True)  # (n_samples, 1)
    loss = p.sum()

    # Compute gradient of loss w.r.t. `transform`
    weighted_p_ij = masked_p_ij - p_ij * p
    weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
    np.fill_diagonal(weighted_p_ij_sym, - weighted_p_ij.sum(axis=0))
    gradient = 2 * (X_embedded.T.dot(weighted_p_ij_sym)).dot(X)
    return sign * loss, sign * gradient.ravel()
