"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

from __future__ import absolute_import

import sys
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y, check_random_state

try:  # scipy.misc.logsumexp is deprecated in scipy 1.0.0
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .base_metric import BaseMetricLearner

EPS = np.finfo(float).eps


class NCA(BaseMetricLearner):
  def __init__(self, num_dims=None, max_iter=100, learning_rate='deprecated',
               init='auto', random_state=0, verbose=False):
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.learning_rate = learning_rate  # TODO: remove in v.0.5.0
    self.init = init
    self.random_state = 0
    self.verbose = 0

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

    # Initialize the random generator
    self.random_state_ = check_random_state(self.random_state)

    X, labels = check_X_y(X, y)
    n, d = X.shape
    num_dims = self.num_dims
    if num_dims is None:
        num_dims = d

    A = self._initialize(X, y, self.init)

    # Run NCA
    mask = y[:, np.newaxis] == y[np.newaxis, :]
    optimizer_params = {'method': 'L-BFGS-B',
                        'fun': self._loss_grad_lbfgs,
                        'args': (X, mask, -1.0),
                        'jac': True,
                        'x0': A.ravel(),
                        'options': dict(maxiter=self.max_iter),
                        }

    # Call the optimizer
    opt_result = minimize(**optimizer_params)

    self.X_ = X
    self.A_ = opt_result.x.reshape(num_dims, -1)
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
    p = np.sum(masked_p_ij, axis=1, keepdims=True)  # (n_samples, 1)
    loss = np.sum(p)

    # Compute gradient of loss w.r.t. `transform`
    weighted_p_ij = masked_p_ij - p_ij * p
    gradient = 2 * (X_embedded.T.dot(weighted_p_ij + weighted_p_ij.T) -
                    X_embedded.T * np.sum(weighted_p_ij, axis=0)).dot(X)
    return sign * loss, sign * gradient.ravel()

  def _initialize(self, X, y, init):
    """Initialize the transformation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training samples.

    y : array-like, shape (n_samples,)
        The training labels.

    init : string or numpy array of shape (n_features_a, n_features_b)
        The validated initialization of the linear transformation.

    Returns
    -------
    transformation : array, shape (n_components, n_features)
        The initialized linear transformation.

    """

    transformation = init

    if isinstance(init, np.ndarray):
      pass
    else:
      n_samples, n_features = X.shape
      n_components = self.num_dims or n_features
      if init == 'auto':
        n_classes = len(np.unique(y))
        if num_dims <= n_classes:
          init = 'lda'
        elif num_dims < min(n_features, n_samples):
          init = 'pca'
        else:
          init = 'identity'
      if init == 'identity':
        transformation = np.eye(num_dims, X.shape[1])
      elif init == 'random':
        transformation = self.random_state_.randn(num_dims,
                                                  X.shape[1])
      elif init in {'pca', 'lda'}:
        init_time = time.time()
        if init == 'pca':
          pca = PCA(n_components=num_dims,
                    random_state=self.random_state_)
          if self.verbose:
            print('Finding principal components... ', end='')
            sys.stdout.flush()
          pca.fit(X)
          transformation = pca.components_
        elif init == 'lda':
          from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis)
          lda = LinearDiscriminantAnalysis(n_components=n_components)
          if self.verbose:
            print('Finding most discriminative components... ',
                  end='')
            sys.stdout.flush()
          lda.fit(X, y)
          transformation = lda.scalings_.T[:n_components]
        if self.verbose:
          print('done in {:5.2f}s'.format(time.time() - init_time))
    return transformation