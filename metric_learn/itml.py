"""
Information Theoretic Metric Learning, Kulis et al., ICML 2007
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from base_metric import BaseMetricLearner


class ITML(BaseMetricLearner):
  """
  Information Theoretic Metric Learning (ITML)
  """
  def __init__(self, gamma=1., max_iters=1000, A0=None,
               convergence_threshold=1e-3):
    """
    gamma: value for slack variables
    A0: [optional] (d x d) initial regularization matrix, defaults to identity
    """
    self.gamma = gamma
    self.A = A0
    self.max_iters = max_iters
    self.convergence_threshold = convergence_threshold

  def _process_inputs(self, X, constraints, bounds):
    self.X = X
    # check to make sure that no two constrained vectors are identical
    a,b,c,d = constraints
    ident = _vector_norm(self.X[a] - self.X[b]) > 1e-9
    a, b = a[ident], b[ident]
    ident = _vector_norm(self.X[c] - self.X[d]) > 1e-9
    c, d = c[ident], d[ident]
    # init bounds
    if bounds is None:
      self.bounds = np.percentile(pairwise_distances(X), (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds = bounds
    # init metric
    if self.A is None:
      self.A = np.identity(X.shape[1])
    return a,b,c,d

  def fit(self, X, constraints, bounds=None, verbose=False):
    """
    X: (n x d) data matrix - each row corresponds to a single instance
    constraints: tuple of arrays: (a,b,c,d) indices into X, such that:
      d(X[a],X[b]) < d(X[c],X[d])
    bounds: (pos,neg) pair of bounds on similarity, such that:
      d(X[a],X[b]) < pos
      d(X[c],X[d]) > neg
    """
    a,b,c,d = self._process_inputs(X, constraints, bounds)
    gamma = self.gamma
    num_pos = len(a)
    num_neg = len(c)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds[0]
    neg_bhat = np.zeros(num_neg) + self.bounds[1]
    A = self.A

    for it in xrange(self.max_iters):
      # update positives
      vv = self.X[a] - self.X[b]
      for i,v in enumerate(vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        A += beta * A.dot(np.outer(v,v)).dot(A)

      # update negatives
      vv = self.X[c] - self.X[d]
      for i,v in enumerate(vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i+num_pos],gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        A += beta * A.dot(np.outer(v,v)).dot(A)

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < self.convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if verbose:
        print 'itml iter: %d, conv = %f' % (it, conv)
    if verbose:
      print 'itml converged at iter: %d, conv = %f' % (it, conv)
    return self

  def metric(self):
    return self.A

  @classmethod
  def prepare_constraints(self, labels, num_points, num_constraints):
    ac,bd = np.random.randint(num_points, size=(2,num_constraints))
    pos = labels[ac] == labels[bd]
    a,c = ac[pos], ac[~pos]
    b,d = bd[pos], bd[~pos]
    return a,b,c,d

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def _vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def _vector_norm(X):
    return np.linalg.norm(X, axis=1)
