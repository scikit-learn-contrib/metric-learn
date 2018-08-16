"""
Metric Learning for Kernel Regression (MLKR), Weinberger et al.,

MLKR is an algorithm for supervised metric learning, which learns a distance
function by directly minimising the leave-one-out regression error. This
algorithm can also be viewed as a supervised variation of PCA and can be used
for dimensionality reduction and high dimensional data visualization.
"""
from __future__ import division, print_function
import time
import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning

from .base_metric import BaseMetricLearner

EPS = np.finfo(float).eps


class MLKR(BaseMetricLearner):
  """Metric Learning for Kernel Regression (MLKR)"""
  def __init__(self, num_dims=None, A0=None, epsilon=0.01, alpha=0.0001,
               max_iter=1000, verbose=False):
    """
    Initialize MLKR.

    Parameters
    ----------
    num_dims : int, optional
        Dimensionality of reduced space (defaults to dimension of X)

    A0: array-like, optional
        Initialization of transformation matrix. Defaults to PCA loadings.

    epsilon: float, optional
        Step size for congujate gradient descent.

    alpha: float, optional
        Stopping criterion for congujate gradient descent.

    max_iter: int, optional
        Cap on number of congugate gradient iterations.

    verbose : bool, optional (default=False)
        Whether to print progress messages or not.
    """
    self.num_dims = num_dims
    self.A0 = A0
    self.epsilon = epsilon
    self.alpha = alpha
    self.max_iter = max_iter
    self.verbose = verbose

  def _process_inputs(self, X, y):
      self.X_, y = check_X_y(X, y)
      n, d = self.X_.shape
      if y.shape[0] != n:
          raise ValueError('Data and label lengths mismatch: %d != %d'
                           % (n, y.shape[0]))

      A = self.A0
      m = self.num_dims
      if m is None:
          m = d
      if A is None:
          # initialize to PCA transformation matrix
          # note: not the same as n_components=m !
          A = PCA().fit(X).components_.T[:m]
      elif A.shape != (m, d):
          raise ValueError('A0 needs shape (%d,%d) but got %s' % (
              m, d, A.shape))
      return self.X_, y, A

  def fit(self, X, y):
      """
      Fit MLKR model

      Parameters
      ----------
      X : (n x d) array of samples
      y : (n) data labels
      """
      X, y, A = self._process_inputs(X, y)

      # Measure the total training time
      train_time = time.time()

      # note: this line takes (n*n*d) memory!
      # for larger datasets, we'll need to compute dX as we go
      dX = (X[None] - X[:, None]).reshape((-1, X.shape[1]))

      self.n_iter_ = 0
      res = minimize(self._loss, A.ravel(), (X, y, dX), method='L-BFGS-B',
                     jac=True, tol=self.alpha,
                     options=dict(maxiter=self.max_iter, eps=self.epsilon))
      self.transformer_ = res.x.reshape(A.shape)

      # Stop timer
      train_time = time.time() - train_time
      if self.verbose:
          cls_name = self.__class__.__name__
          # Warn the user if the algorithm did not converge
          if not res.success:
              warnings.warn('[{}] MLKR did not converge: {}'
                            .format(cls_name, res.message), ConvergenceWarning)
          print('[{}] Training took {:8.2f}s.'.format(cls_name, train_time))

      return self

  def transformer(self):
      return self.transformer_

  def _loss(self, flatA, X, y, dX):

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

    A = flatA.reshape((-1, X.shape[1]))
    dist = pdist(X, metric='mahalanobis', VI=A.T.dot(A))
    K = squareform(np.exp(-dist**2))
    denom = np.maximum(K.sum(axis=0), EPS)
    yhat = K.dot(y) / denom
    ydiff = yhat - y
    cost = (ydiff**2).sum()

    # also compute the gradient
    np.fill_diagonal(K, 1)
    W = 2 * K * (np.outer(ydiff, ydiff) / denom)
    # note: this is the part that the matlab impl drops to C for
    M = (dX.T * W.ravel()).dot(dX)
    grad = 2 * A.dot(M)

    if self.verbose:
      start_time = time.time() - start_time
      values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
      print(values_fmt.format(cls=self.__class__.__name__,
                              n_iter=self.n_iter_, loss=cost,
                              start_time=start_time))
      sys.stdout.flush()

    self.n_iter_ += 1

    return cost, grad.ravel()
