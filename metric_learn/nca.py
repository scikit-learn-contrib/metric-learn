"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

from __future__ import absolute_import
import warnings
import time
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.fixes import logsumexp
from sklearn.base import TransformerMixin

from ._util import _check_n_components
from .base_metric import MahalanobisMixin

EPS = np.finfo(float).eps


class NCA(MahalanobisMixin, TransformerMixin):
  """Neighborhood Components Analysis (NCA)

  Attributes
  ----------
  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, n_components=None, num_dims='deprecated', max_iter=100,
               tol=None, verbose=False, preprocessor=None):
    """Neighborhood Components Analysis

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Dimensionality of reduced space (if None, defaults to dimension of X).

    num_dims : Not used

        .. deprecated:: 0.5.0
          `num_dims` was deprecated in version 0.5.0 and will
          be removed in 0.6.0. Use `n_components` instead.

    max_iter : int, optional (default=100)
      Maximum number of iterations done by the optimization algorithm.

    tol : float, optional (default=None)
        Convergence tolerance for the optimization.

    verbose : bool, optional (default=False)
      Whether to print progress messages or not.
    """
    self.n_components = n_components
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    super(NCA, self).__init__(preprocessor)

  def fit(self, X, y):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    if self.num_dims != 'deprecated':
      warnings.warn('"num_dims" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    'removed in 0.6.0. Use "n_components" instead',
                    DeprecationWarning)
    X, labels = self._prepare_inputs(X, y, ensure_min_samples=2)
    n, d = X.shape
    n_components = _check_n_components(d, self.n_components)

    # Measure the total training time
    train_time = time.time()

    # Initialize A to a scaling matrix
    A = np.zeros((n_components, d))
    np.fill_diagonal(A, 1. / (np.maximum(X.max(axis=0) - X.min(axis=0), EPS)))

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
    self.n_iter_ = 0
    opt_result = minimize(**optimizer_params)

    self.transformer_ = opt_result.x.reshape(-1, X.shape[1])
    self.n_iter_ = opt_result.nit

    # Stop timer
    train_time = time.time() - train_time
    if self.verbose:
      cls_name = self.__class__.__name__

      # Warn the user if the algorithm did not converge
      if not opt_result.success:
        warnings.warn('[{}] NCA did not converge: {}'.format(
            cls_name, opt_result.message), ConvergenceWarning)

      print('[{}] Training took {:8.2f}s.'.format(cls_name, train_time))

    return self

  def _loss_grad_lbfgs(self, A, X, mask, sign=1.0):

    if self.n_iter_ == 0 and self.verbose:
      header_fields = ['Iteration', 'Objective Value', 'Time(s)']
      header_fmt = '{:>10} {:>20} {:>10}'
      header = header_fmt.format(*header_fields)
      cls_name = self.__class__.__name__
      print('[{cls}]'.format(cls=cls_name))
      print('[{cls}] {header}\n[{cls}] {sep}'.format(cls=cls_name,
                                                     header=header,
                                                     sep='-' * len(header)))

    start_time = time.time()

    A = A.reshape(-1, X.shape[1])
    X_embedded = np.dot(X, A.T)  # (n_samples, n_components)
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

    if self.verbose:
        start_time = time.time() - start_time
        values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
        print(values_fmt.format(cls=self.__class__.__name__,
                                n_iter=self.n_iter_, loss=loss,
                                start_time=start_time))
        sys.stdout.flush()

    self.n_iter_ += 1
    return sign * loss, sign * gradient.ravel()
