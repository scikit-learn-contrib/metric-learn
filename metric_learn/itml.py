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
from sklearn.utils.validation import check_array
from sklearn.base import TransformerMixin
from .base_metric import _PairsClassifierMixin, MahalanobisMixin
from .constraints import Constraints, wrap_pairs
from ._util import vector_norm


class _BaseITML(MahalanobisMixin):
  """Information Theoretic Metric Learning (ITML)"""

  _tuple_size = 2  # constraints are pairs

  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               A0=None, verbose=False, preprocessor=None):
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

    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    self.gamma = gamma
    self.max_iter = max_iter
    self.convergence_threshold = convergence_threshold
    self.A0 = A0
    self.verbose = verbose
    super(_BaseITML, self).__init__(preprocessor)

  def _process_pairs(self, pairs, y, bounds):
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')

    # check to make sure that no two constrained vectors are identical
    pos_pairs, neg_pairs = pairs[y == 1], pairs[y == -1]
    pos_no_ident = vector_norm(pos_pairs[:, 0, :] - pos_pairs[:, 1, :]) > 1e-9
    pos_pairs = pos_pairs[pos_no_ident]
    neg_no_ident = vector_norm(neg_pairs[:, 0, :] - neg_pairs[:, 1, :]) > 1e-9
    neg_pairs = neg_pairs[neg_no_ident]
    # init bounds
    if bounds is None:
      X = np.vstack({tuple(row) for row in pairs.reshape(-1, pairs.shape[2])})
      self.bounds_ = np.percentile(pairwise_distances(X), (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds_ = bounds
    self.bounds_[self.bounds_==0] = 1e-9
    # init metric
    if self.A0 is None:
      self.A_ = np.identity(pairs.shape[2])
    else:
      self.A_ = check_array(self.A0)
    pairs = np.vstack([pos_pairs, neg_pairs])
    y = np.hstack([np.ones(len(pos_pairs)), - np.ones(len(neg_pairs))])
    return pairs, y

  def _fit(self, pairs, y, bounds=None):
    pairs, y = self._process_pairs(pairs, y, bounds)
    gamma = self.gamma
    pos_pairs, neg_pairs = pairs[y == 1], pairs[y == -1]
    num_pos = len(pos_pairs)
    num_neg = len(neg_pairs)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds_[0]
    neg_bhat = np.zeros(num_neg) + self.bounds_[1]
    pos_vv = pos_pairs[:, 0, :] - pos_pairs[:, 1, :]
    neg_vv = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]
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

    self.transformer_ = self.transformer_from_metric(self.A_)
    return self


class ITML(_BaseITML, _PairsClassifierMixin):
  """Information Theoretic Metric Learning (ITML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See :meth:`transformer_from_metric`.)
  """

  def fit(self, pairs, y, bounds=None):
    """Learn the ITML model.

    Parameters
    ----------
    pairs: array-like, shape=(n_constraints, 2, n_features) or
           (n_constraints, 2)
        3D Array of pairs with each row corresponding to two points,
        or 2D array of indices of pairs if the metric learner uses a
        preprocessor.
    y: array-like, of shape (n_constraints,)
        Labels of constraints. Should be -1 for dissimilar pair, 1 for similar.
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg

    Returns
    -------
    self : object
        Returns the instance.
    """
    return self._fit(pairs, y, bounds=bounds)


class ITML_Supervised(_BaseITML, TransformerMixin):
  """Supervised version of Information Theoretic Metric Learning (ITML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See `transformer_from_metric`.)
  """

  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               num_labeled=np.inf, num_constraints=None, bounds=None, A0=None,
               verbose=False, preprocessor=None):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iter : int, optional
    convergence_threshold : float, optional
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    _BaseITML.__init__(self, gamma=gamma, max_iter=max_iter,
                       convergence_threshold=convergence_threshold,
                       A0=A0, verbose=verbose, preprocessor=preprocessor)
    self.num_labeled = num_labeled
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
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseITML._fit(self, pairs, y, bounds=self.bounds)
