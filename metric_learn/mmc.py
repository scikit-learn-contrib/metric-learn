"""
Mahalanobis Metric Learning with Application for Clustering with Side-Information, Xing et al., NIPS 2002

MMC minimizes the sum of squared distances between similar examples,
while enforcing the sum of distances between dissimilar examples to be
greater than a certain margin.
This leads to a convex and, thus, local-minima-free optimization problem
that can be solved efficiently.
However, the algorithm involves the computation of eigenvalues, which is the
main speed-bottleneck.
Since it has initially been designed for clustering applications, one of the
implicit assumptions of MMC is that all classes form a compact set, i.e.,
follow a unimodal distribution, which restricts the possible use-cases of
this method. However, it is one of the earliest and a still often cited technique.

Adapted from Matlab code at http://www.cs.cmu.edu/%7Eepxing/papers/Old_papers/code_Metric_online.tar.gz
"""

from __future__ import print_function, absolute_import, division
import warnings
import numpy as np
from six.moves import xrange
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array, assert_all_finite

from .base_metric import _PairsClassifierMixin, MahalanobisMixin
from .constraints import Constraints, wrap_pairs
from ._util import vector_norm, transformer_from_metric


class _BaseMMC(MahalanobisMixin):
  """Mahalanobis Metric for Clustering (MMC)"""

  _tuple_size = 2  # constraints are pairs

  def __init__(self, max_iter=100, max_proj=10000, convergence_threshold=1e-3,
               A0=None, diagonal=False, diagonal_c=1.0, verbose=False,
               preprocessor=None):
    """Initialize MMC.
    Parameters
    ----------
    max_iter : int, optional
    max_proj : int, optional
    convergence_threshold : float, optional
    A0 : (d x d) matrix, optional
        initial metric, defaults to identity
        only the main diagonal is taken if `diagonal == True`
    diagonal : bool, optional
        if True, a diagonal metric will be learned,
        i.e., a simple scaling of dimensions
    diagonal_c : float, optional
        weight of the dissimilarity constraint for diagonal
        metric learning
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be gotten like this: X[indices].
    """
    self.max_iter = max_iter
    self.max_proj = max_proj
    self.convergence_threshold = convergence_threshold
    self.A0 = A0
    self.diagonal = diagonal
    self.diagonal_c = diagonal_c
    self.verbose = verbose
    super(_BaseMMC, self).__init__(preprocessor)

  def _fit(self, pairs, y):
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')

    # init metric
    if self.A0 is None:
      self.A_ = np.identity(pairs.shape[2])
      if not self.diagonal:
        # Don't know why division by 10... it's in the original code
        # and seems to affect the overall scale of the learned metric.
        self.A_ /= 10.0
    else:
      self.A_ = check_array(self.A0)

    if self.diagonal:
      return self._fit_diag(pairs, y)
    else:
      return self._fit_full(pairs, y)

  def _fit_full(self, pairs, y):
    """Learn full metric using MMC.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, with (a,b) specifying similar and (c,d)
        dissimilar pairs
    """
    num_dim = pairs.shape[2]

    error1 = error2 = 1e10
    eps = 0.01        # error-bound of iterative projection on C1 and C2
    A = self.A_

    pos_pairs, neg_pairs = pairs[y == 1], pairs[y == -1]

    # Create weight vector from similar samples
    pos_diff = pos_pairs[:, 0, :] - pos_pairs[:, 1, :]
    w = np.einsum('ij,ik->jk', pos_diff, pos_diff).ravel()
    # `w` is the sum of all outer products of the rows in `pos_diff`.
    # The above `einsum` is equivalent to the much more inefficient:
    # w = np.apply_along_axis(
    #         lambda x: np.outer(x,x).ravel(),
    #         1,
    #         X[a] - X[b]
    #     ).sum(axis = 0)
    t = w.dot(A.ravel()) / 100.0

    w_norm = np.linalg.norm(w)
    w1 = w / w_norm  # make `w` a unit vector
    t1 = t / w_norm  # distance from origin to `w^T*x=t` plane

    cycle = 1
    alpha = 0.1  # initial step size along gradient
    grad1 = self._fS1(pos_pairs, A)            # gradient of similarity
    # constraint function
    grad2 = self._fD1(neg_pairs, A)            # gradient of dissimilarity
    # constraint function
    M = self._grad_projection(grad1, grad2)  # gradient of fD1 orthogonal to fS1

    A_old = A.copy()

    for cycle in xrange(self.max_iter):

      # projection of constraints C1 and C2
      satisfy = False

      for it in xrange(self.max_proj):

        # First constraint:
        # f(A) = \sum_{i,j \in S} d_ij' A d_ij <= t              (1)
        # (1) can be rewritten as a linear constraint: w^T x = t,
        # where x is the unrolled matrix of A,
        # w is also an unrolled matrix of W where
        # W_{kl}= \sum_{i,j \in S}d_ij^k * d_ij^l
        x0 = A.ravel()
        if w.dot(x0) <= t:
          x = x0
        else:
          x = x0 + (t1 - w1.dot(x0)) * w1
          A[:] = x.reshape(num_dim, num_dim)

        # Second constraint:
        # PSD constraint A >= 0
        # project A onto domain A>0
        l, V = np.linalg.eigh((A + A.T) / 2)
        A[:] = np.dot(V * np.maximum(0, l[None,:]), V.T)

        fDC2 = w.dot(A.ravel())
        error2 = (fDC2 - t) / t
        if error2 < eps:
          satisfy = True
          break

      # third constraint: gradient ascent
      # max: g(A) >= 1
      # here we suppose g(A) = fD(A) = \sum_{I,J \in D} sqrt(d_ij' A d_ij)

      obj_previous = self._fD(neg_pairs, A_old)  # g(A_old)
      obj = self._fD(neg_pairs, A)               # g(A)

      if satisfy and (obj > obj_previous or cycle == 0):

        # If projection of 1 and 2 is successful, and such projection
        # improves objective function, slightly increase learning rate
        # and update from the current A.
        alpha *= 1.05
        A_old[:] = A
        grad2 = self._fS1(pos_pairs, A)
        grad1 = self._fD1(neg_pairs, A)
        M = self._grad_projection(grad1, grad2)
        A += alpha * M

      else:

        # If projection of 1 and 2 failed, or obj <= obj_previous due
        # to projection of 1 and 2, shrink learning rate and re-update
        # from the previous A.
        alpha /= 2
        A[:] = A_old + alpha * M

      delta = np.linalg.norm(alpha * M) / np.linalg.norm(A_old)
      if delta < self.convergence_threshold:
        break
      if self.verbose:
        print('mmc iter: %d, conv = %f, projections = %d' % (cycle, delta, it+1))

    if delta > self.convergence_threshold:
      self.converged_ = False
      if self.verbose:
        print('mmc did not converge, conv = %f' % (delta,))
    else:
      self.converged_ = True
      if self.verbose:
        print('mmc converged at iter %d, conv = %f' % (cycle, delta))
    self.A_[:] = A_old
    self.n_iter_ = cycle

    self.transformer_ = transformer_from_metric(self.A_)
    return self

  def _fit_diag(self, pairs, y):
    """Learn diagonal metric using MMC.
    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, with (a,b) specifying similar and (c,d)
        dissimilar pairs
    """
    num_dim = pairs.shape[2]
    pos_pairs, neg_pairs = pairs[y == 1], pairs[y == -1]
    s_sum = np.sum((pos_pairs[:, 0, :] - pos_pairs[:, 1, :]) ** 2, axis=0)

    it = 0
    error = 1.0
    eps = 1e-6
    reduction = 2.0
    w = np.diag(self.A_).copy()

    while error > self.convergence_threshold and it < self.max_iter:

      fD0, fD_1st_d, fD_2nd_d = self._D_constraint(neg_pairs, w)
      obj_initial = np.dot(s_sum, w) + self.diagonal_c * fD0
      fS_1st_d = s_sum  # first derivative of the similarity constraints

      gradient = fS_1st_d - self.diagonal_c * fD_1st_d               # gradient of the objective
      hessian = -self.diagonal_c * fD_2nd_d + eps * np.eye(num_dim)  # Hessian of the objective
      step = np.dot(np.linalg.inv(hessian), gradient)

      # Newton-Rapshon update
      # search over optimal lambda
      lambd = 1  # initial step-size
      w_tmp = np.maximum(0, w - lambd * step)
      obj = (np.dot(s_sum, w_tmp) + self.diagonal_c *
             self._D_objective(neg_pairs, w_tmp))
      assert_all_finite(obj)
      obj_previous = obj + 1  # just to get the while-loop started

      inner_it = 0
      while obj < obj_previous:
        obj_previous = obj
        w_previous = w_tmp.copy()
        lambd /= reduction
        w_tmp = np.maximum(0, w - lambd * step)
        obj = (np.dot(s_sum, w_tmp) + self.diagonal_c *
               self._D_objective(neg_pairs, w_tmp))
        inner_it += 1
        assert_all_finite(obj)

      w[:] = w_previous
      error = np.abs((obj_previous - obj_initial) / obj_previous)
      if self.verbose:
        print('mmc iter: %d, conv = %f' % (it, error))
      it += 1

    self.A_ = np.diag(w)

    self.transformer_ = transformer_from_metric(self.A_)
    return self

  def _fD(self, neg_pairs, A):
    """The value of the dissimilarity constraint function.

    f = f(\sum_{ij \in D} distance(x_i, x_j))
    i.e. distance can be L1:  \sqrt{(x_i-x_j)A(x_i-x_j)'}
    """
    diff = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]
    return np.log(np.sum(np.sqrt(np.sum(np.dot(diff, A) * diff, axis=1))) + 1e-6)

  def _fD1(self, neg_pairs, A):
    """The gradient of the dissimilarity constraint function w.r.t. A.

    For example, let distance by L1 norm:
    f = f(\sum_{ij \in D} \sqrt{(x_i-x_j)A(x_i-x_j)'})
    df/dA_{kl} = f'* d(\sum_{ij \in D} \sqrt{(x_i-x_j)^k*(x_i-x_j)^l})/dA_{kl}

    Note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
    so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij
        df/dA = f'(\sum_{ij \in D} \sqrt{tr(d_ij'*d_ij*A)})
                * 0.5*(\sum_{ij \in D} (1/sqrt{tr(d_ij'*d_ij*A)})*(d_ij'*d_ij))
    """
    dim = neg_pairs.shape[2]
    diff = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]
    # outer products of all rows in `diff`
    M = np.einsum('ij,ik->ijk', diff, diff)
    # faster version of: dist = np.sqrt(np.sum(M * A[None,:,:], axis=(1,2)))
    dist = np.sqrt(np.einsum('ijk,jk', M, A))
    # faster version of: sum_deri = np.sum(M / (2 * (dist[:,None,None] + 1e-6)), axis=0)
    sum_deri = np.einsum('ijk,i->jk', M, 0.5 / (dist + 1e-6))
    sum_dist = dist.sum()
    return sum_deri / (sum_dist + 1e-6)

  def _fS1(self, pos_pairs, A):
    """The gradient of the similarity constraint function w.r.t. A.

    f = \sum_{ij}(x_i-x_j)A(x_i-x_j)' = \sum_{ij}d_ij*A*d_ij'
    df/dA = d(d_ij*A*d_ij')/dA

    Note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
    so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij
    """
    dim = pos_pairs.shape[2]
    diff = pos_pairs[:, 0, :] - pos_pairs[:, 1, :]
    return np.einsum('ij,ik->jk', diff, diff)  # sum of outer products of all rows in `diff`

  def _grad_projection(self, grad1, grad2):
    grad2 = grad2 / np.linalg.norm(grad2)
    gtemp = grad1 - np.sum(grad1 * grad2) * grad2
    gtemp /= np.linalg.norm(gtemp)
    return gtemp

  def _D_objective(self, neg_pairs, w):
    return np.log(np.sum(np.sqrt(np.sum(((neg_pairs[:, 0, :] -
                                          neg_pairs[:, 1, :]) ** 2) *
                                        w[None,:], axis=1) + 1e-6)))

  def _D_constraint(self, neg_pairs, w):
    """Compute the value, 1st derivative, second derivative (Hessian) of
    a dissimilarity constraint function gF(sum_ij distance(d_ij A d_ij))
    where A is a diagonal matrix (in the form of a column vector 'w').
    """
    diff = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]
    diff_sq = diff * diff
    dist = np.sqrt(diff_sq.dot(w))
    sum_deri1 = np.einsum('ij,i', diff_sq, 0.5 / np.maximum(dist, 1e-6))
    sum_deri2 = np.einsum(
        'ij,ik->jk',
        diff_sq,
        diff_sq / (-4 * np.maximum(1e-6, dist**3))[:,None]
    )
    sum_dist = dist.sum()
    return (
      np.log(sum_dist),
      sum_deri1 / sum_dist,
      sum_deri2 / sum_dist - np.outer(sum_deri1, sum_deri1) / (sum_dist * sum_dist)
    )


class MMC(_BaseMMC, _PairsClassifierMixin):
  """Mahalanobis Metric for Clustering (MMC)

  Attributes
  ----------
  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)

  threshold_ : `float`
      If the distance metric between two points is lower than this threshold,
      points will be classified as similar, otherwise they will be
      classified as dissimilar.

  classes_ : `list`
      The possible labels of the pairs `MMC` can fit on. `classes_ = [-1, 1]`,
      where -1 means points in a pair are dissimilar (negative label), and 1
      means they are similar (positive label).
  """

  def fit(self, pairs, y):
    """Learn the MMC model.

    Parameters
    ----------
    pairs: array-like, shape=(n_constraints, 2, n_features) or
           (n_constraints, 2)
        3D Array of pairs with each row corresponding to two points,
        or 2D array of indices of pairs if the metric learner uses a
        preprocessor.
    y: array-like, of shape (n_constraints,)
        Labels of constraints. Should be -1 for dissimilar pair, 1 for similar.

    Returns
    -------
    self : object
        Returns the instance.
    """
    self._fit(pairs, y)
    self._set_default_threshold(pairs, y)
    return self


class MMC_Supervised(_BaseMMC, TransformerMixin):
  """Supervised version of Mahalanobis Metric for Clustering (MMC)

  Attributes
  ----------
  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)
  """

  def __init__(self, max_iter=100, max_proj=10000, convergence_threshold=1e-6,
               num_labeled='deprecated', num_constraints=None, A0=None,
               diagonal=False, diagonal_c=1.0, verbose=False,
               preprocessor=None):
    """Initialize the supervised version of `MMC`.

    `MMC_Supervised` creates pairs of similar sample by taking same class
    samples, and pairs of dissimilar samples by taking different class
    samples. It then passes these pairs to `MMC` for training.

    Parameters
    ----------
    max_iter : int, optional
    max_proj : int, optional
    convergence_threshold : float, optional
    num_labeled : Not used
      .. deprecated:: 0.5.0
         `num_labeled` was deprecated in version 0.5.0 and will
         be removed in 0.6.0.
    num_constraints: int, optional
        number of constraints to generate
    A0 : (d x d) matrix, optional
        initial metric, defaults to identity
        only the main diagonal is taken if `diagonal == True`
    diagonal : bool, optional
        if True, a diagonal metric will be learned,
        i.e., a simple scaling of dimensions
    diagonal_c : float, optional
        weight of the dissimilarity constraint for diagonal
        metric learning
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    _BaseMMC.__init__(self, max_iter=max_iter, max_proj=max_proj,
                      convergence_threshold=convergence_threshold,
                      A0=A0, diagonal=diagonal, diagonal_c=diagonal_c,
                      verbose=verbose, preprocessor=preprocessor)
    self.num_labeled = num_labeled
    self.num_constraints = num_constraints

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the MMC model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.
    y : (n) array-like
        Data labels.
    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.
    """
    if self.num_labeled != 'deprecated':
      warnings.warn('"num_labeled" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    'removed in 0.6.0', DeprecationWarning)
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseMMC._fit(self, pairs, y)
