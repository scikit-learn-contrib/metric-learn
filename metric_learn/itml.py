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
from sklearn.utils.validation import check_array, check_X_y

from .base_metric import BaseMetricLearner
from .constraints import Constraints
from ._util import vector_norm


class ITML(BaseMetricLearner):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               A0=None, verbose=False):
    """Initialize ITML.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables

    max_iter : int, optional

    convergence_threshold : float, optional

    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity

    verbose : bool, optional
        if True, prints information while learning
    """
    self.gamma = gamma
    self.max_iter = max_iter
    self.convergence_threshold = convergence_threshold
    self.A0 = A0
    self.verbose = verbose

  def _process_inputs(self, X, constraints, bounds):
    self.X_ = X = check_array(X)
    # check to make sure that no two constrained vectors are identical
    a,b,c,d = constraints
    no_ident = vector_norm(X[a] - X[b]) > 1e-9
    a, b = a[no_ident], b[no_ident]
    no_ident = vector_norm(X[c] - X[d]) > 1e-9
    c, d = c[no_ident], d[no_ident]
    # init bounds
    if bounds is None:
      self.bounds_ = np.percentile(pairwise_distances(X), (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds_ = bounds
    self.bounds_[self.bounds_==0] = 1e-9
    # init metric
    if self.A0 is None:
      self.A_ = np.identity(X.shape[1])
    else:
      self.A_ = check_array(self.A0)
    return a,b,c,d

  def fit(self, X, constraints, bounds=None):
    """Learn the ITML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, with (a,b) specifying positive and (c,d)
        negative pairs
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    """
    a,b,c,d = self._process_inputs(X, constraints, bounds)
    gamma = self.gamma
    num_pos = len(a)
    num_neg = len(c)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds_[0]
    neg_bhat = np.zeros(num_neg) + self.bounds_[1]
    pos_vv = self.X_[a] - self.X_[b]
    neg_vv = self.X_[c] - self.X_[d]
    A = self.A_

    for it in xrange(self.max_iter):
      # update positives
      for i,v in enumerate(pos_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      # update negatives
      for i,v in enumerate(neg_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i+num_pos], gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < self.convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if self.verbose:
        print('itml iter: %d, conv = %f' % (it, conv))

    if self.verbose:
      print('itml converged at iter: %d, conv = %f' % (it, conv))
    self.n_iter_ = it
    return self

  def metric(self):
    return self.A_


class ITML_Supervised(ITML):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               num_constraints=None, bounds=None, A0=None, verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iter : int, optional
    convergence_threshold : float, optional
    num_constraints: int, optional
        number of constraints to generate
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity
    verbose : bool, optional
        if True, prints information while learning
    """
    ITML.__init__(self, gamma=gamma, max_iter=max_iter,
                  convergence_threshold=convergence_threshold,
                  A0=A0, verbose=verbose)
    self.num_constraints = num_constraints
    self.bounds = bounds

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the ITML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.
    """
    X, y = check_X_y(X, y)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    return ITML.fit(self, X, pos_neg, bounds=self.bounds)
