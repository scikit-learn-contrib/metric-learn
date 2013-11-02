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
  def __init__(self, X, constraints, bounds=None, A0=None):
    """
    X: (n x d) data matrix - each row corresponds to a single instance
    A0: [optional] (d x d) initial regularization matrix, defaults to identity
    constraints: tuple of arrays: (a,b,c,d) indices into X, such that:
      d(X[a],X[b]) < d(X[c],X[d])
    bounds: (pos,neg) pair of bounds on similarity, such that:
      d(X[a],X[b]) < pos
      d(X[c],X[d]) > neg
    """
    self.X = X
    # check to make sure that no two constrained vectors are identical
    a,b,c,d = constraints
    ident = np.linalg.norm(self.X[a] - self.X[b], axis=1) > 1e-9
    a, b = a[ident], b[ident]
    ident = np.linalg.norm(self.X[c] - self.X[d], axis=1) > 1e-9
    c, d = c[ident], d[ident]
    self.C = a,b,c,d
    # init bounds
    if bounds is None:
      self.bounds = np.percentile(pairwise_distances(X), (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds = bounds
    # intialize metric
    if A0 is None:
      self.A = np.identity(self.X.shape[1])
    else:
      self.A = A0

  def fit(self, convergence_threshold=1e-3, gamma=1., max_iters=1000, verbose=False):
    """
    gamma: value for slack variables
    """
    a,b,c,d = self.C
    num_pos = len(a)
    num_neg = len(c)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds[0]
    neg_bhat = np.zeros(num_neg) + self.bounds[1]
    A = self.A

    for it in xrange(max_iters):
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
      if conv < convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if verbose:
        print 'itml iter: %d, conv = %f' % (it, conv)
    if verbose:
      print 'itml converged at iter: %d, conv = %f' % (it, conv)

  def metric(self):
    return self.A

  @classmethod
  def prepare_constraints(self, labels, num_points, num_constraints):
    ac,bd = np.random.randint(num_points, size=(2,num_constraints))
    pos = labels[ac] == labels[bd]
    a,c = ac[pos], ac[~pos]
    b,d = bd[pos], bd[~pos]
    return a,b,c,d
