"""
Metric Learning for Kernel Regression (MLKR), Weinberger et al.,

MLKR is an algorithm for supervised metric learning, which learns a distance
function by directly minimising the leave-one-out regression error. This
algorithm can also be viewed as a supervised variation of PCA and can be used
for dimensionality reduction and high dimensional data visualization.
"""
from __future__ import division, print_function
import numpy as np
from sklearn.utils.fixes import logsumexp
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y
from numpy.linalg import multi_dot

from .base_metric import BaseMetricLearner

EPS = np.finfo(float).eps


class MLKR(BaseMetricLearner):
  """Metric Learning for Kernel Regression (MLKR)"""
  def __init__(self, num_dims=None, A0=None, epsilon=0.01, alpha=0.0001,
               max_iter=1000):
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
    """
    self.num_dims = num_dims
    self.A0 = A0
    self.epsilon = epsilon
    self.alpha = alpha
    self.max_iter = max_iter

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

      res = minimize(_loss, A.ravel(), (X, y), method='CG', jac=True,
                     tol=self.alpha,
                     options=dict(maxiter=self.max_iter, eps=self.epsilon))
      self.transformer_ = res.x.reshape(A.shape)
      self.n_iter_ = res.nit
      return self

  def transformer(self):
      return self.transformer_


def _loss(flatA, X, y):
  A = flatA.reshape((-1, X.shape[1]))
  dist = pdist(X, metric='mahalanobis', VI=A.T.dot(A))
  dist = squareform(dist ** 2)
  np.fill_diagonal(dist, np.inf)
  softmax = np.exp(- dist - logsumexp(- dist, axis=1)[:, np.newaxis])
  yhat = softmax.dot(y)
  ydiff = yhat - y
  cost = (ydiff**2).sum()

  # also compute the gradient
  W = softmax * ydiff[:, np.newaxis] * (y - yhat[:, np.newaxis])
  W_sym = W + W.T
  np.fill_diagonal(W_sym, - W.sum(axis=0))
  grad = 4 * multi_dot([A, X.T, W_sym, X])
  return cost, grad.ravel()
