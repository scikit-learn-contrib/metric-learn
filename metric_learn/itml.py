"""
Information Theoretic Metric Learning, Kulis et al., ICML 2007

ITML minimizes the differential relative entropy between two multivariate
Gaussians under constraints on the distance function,
which can be formulated into a Bregman optimization problem by minimizing the
LogDet divergence subject to linear constraints.
This algorithm can handle a wide variety of constraints and can optionally
incorporate a prior on the distance function.
Unlike some other methods, ITML does not rely on an eigenvalue computation
or semi-definite programming.

Adapted from Matlab code at http://www.cs.utexas.edu/users/pjain/itml/
"""

from __future__ import print_function, absolute_import
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances

from .base_metric import BaseMetricLearner
from .constraints import Constraints


class ITML(BaseMetricLearner):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iters=1000, convergence_threshold=1e-3,
               verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iters : int, optional
    convergence_threshold : float, optional
    verbose : bool, optional
        if True, prints information while learning
    """
    self.params = {
      'gamma': gamma,
      'max_iters': max_iters,
      'convergence_threshold': convergence_threshold,
      'verbose': verbose,
    }

  def _process_inputs(self, X, constraints, bounds, A0):
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
    if A0 is None:
      self.A = np.identity(X.shape[1])
    else:
      self.A = A0
    return a,b,c,d

  def fit(self, X, constraints, bounds=None, A0=None):
    """Learn the ITML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, such that d(X[a],X[b]) < d(X[c],X[d])
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity
    """
    verbose = self.params['verbose']
    a,b,c,d = self._process_inputs(X, constraints, bounds, A0)
    gamma = self.params['gamma']
    conv_thresh = self.params['convergence_threshold']
    num_pos = len(a)
    num_neg = len(c)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds[0]
    neg_bhat = np.zeros(num_neg) + self.bounds[1]
    A = self.A

    for it in xrange(self.params['max_iters']):
      # update positives
      vv = self.X[a] - self.X[b]
      for i,v in enumerate(vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        Av = A.dot(v)
        A += beta * np.outer(Av, Av)

      # update negatives
      vv = self.X[c] - self.X[d]
      for i,v in enumerate(vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i+num_pos],gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        Av = A.dot(v)
        A += beta * np.outer(Av, Av)

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < conv_thresh:
        break
      lambdaold = _lambda.copy()
      if verbose:
        print('itml iter: %d, conv = %f' % (it, conv))
    if verbose:
      print('itml converged at iter: %d, conv = %f' % (it, conv))
    return self

  def metric(self):
    return self.A

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def _vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def _vector_norm(X):
    return np.linalg.norm(X, axis=1)


class ITML_Supervised(ITML):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iters=1000, convergence_threshold=1e-3,
               num_labeled=np.inf, num_constraints=None, bounds=None, A0=None,
               verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iters : int, optional
    convergence_threshold : float, optional
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    verbose : bool, optional
        if True, prints information while learning
    """
    ITML.__init__(self, gamma=gamma, max_iters=max_iters,
                  convergence_threshold=convergence_threshold, verbose=verbose)
    self.params.update(num_labeled=num_labeled, num_constraints=num_constraints,
                       bounds=bounds, A0=A0)

  def fit(self, X, labels, random_state=np.random):
    """Create constraints from labels and learn the ITML model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    labels : (n) data labels
    random_state : a numpy random.seed object to fix the random_state if needed.
    """
    num_constraints = self.params['num_constraints']
    if num_constraints is None:
      num_classes = np.unique(labels)
      num_constraints = 20*(len(num_classes))**2

    c = Constraints.random_subset(labels, self.params['num_labeled'],
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    return ITML.fit(self, X, pos_neg, bounds=self.params['bounds'],
                    A0=self.params['A0'])
