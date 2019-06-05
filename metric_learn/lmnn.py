r"""
Large Margin Nearest Neighbor Metric learning(LMNN)

LMNN learns a Mahalanobis distance metric in the kNN classification
setting. The learned metric attempts to keep close k-nearest neighbors
from the same class, while keeping examples from different classes
separated by a large margin. This algorithm makes no assumptions about
the distribution of the data.

Read more in the :ref:`User Guide <lmnn>`.

"""
#TODO: periodic recalculation of impostors, PCA initialization

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from collections import Counter
from six.moves import xrange
from sklearn.metrics import euclidean_distances
from sklearn.base import TransformerMixin

from metric_learn._util import _initialize_transformer
from .base_metric import MahalanobisMixin


# commonality between LMNN implementations
class _base_LMNN(MahalanobisMixin, TransformerMixin):
  def __init__(self, init='auto', k=3, min_iter=50, max_iter=1000,
               learn_rate=1e-7, regularization=0.5, convergence_tol=0.001,
               use_pca=True, num_dims=None,
               verbose=False, preprocessor=None, random_state=None):
    """Initialize the LMNN object.

    Parameters
    ----------
    init : string or numpy array, optional (default='auto')
        Initialization of the linear transformation. Possible options are
        'auto', 'pca', 'lda', 'identity', 'random', and a numpy array of shape
        (n_features_a, n_features_b).

        'auto'
            Depending on ``num_dims``, the most reasonable initialization
            will be chosen. If ``num_dims <= n_classes`` we use 'lda', as
            it uses labels information. If not, but
            ``num_dims < min(n_features, n_samples)``, we use 'pca', as
            it projects data in meaningful directions (those of higher
            variance). Otherwise, we just use 'identity'.

        'pca'
            ``num_dims`` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See `sklearn.decomposition.PCA`)

        'lda'
            ``min(num_dims, n_classes)`` most discriminative
            components of the inputs passed to :meth:`fit` will be used to
            initialize the transformation. (If ``num_dims > n_classes``,
            the rest of the components will be zero.) (See
            `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

        'identity'
            If ``num_dims`` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first ``num_dims`` rows.

        'random'
            The initial transformation will be a random array of shape
            `(num_dims, n_features)`. Each value is sampled from the
            standard normal distribution.

        numpy array
            n_features_b must match the dimensionality of the inputs passed to
            :meth:`fit` and n_features_a must be less than or equal to that.
            If ``num_dims`` is not None, n_features_a must match it.

    k : int, optional
        Number of neighbors to consider, not including self-edges.

    regularization: float, optional
        Weighting of pull and push terms, with 0.5 meaning equal weight.

    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].

    random_state : int or numpy.RandomState or None, optional (default=None)
        A pseudo random number generator object or a seed for it if int. If
        ``init='random'``, ``random_state`` is used to initialize the random
        transformation. If ``init='pca'``, ``random_state`` is passed as an
        argument to PCA when initializing the transformation.
    """
    self.init = init
    self.k = k
    self.min_iter = min_iter
    self.max_iter = max_iter
    self.learn_rate = learn_rate
    self.regularization = regularization
    self.convergence_tol = convergence_tol
    self.use_pca = use_pca
    self.num_dims = num_dims  # FIXME Tmp fix waiting for #167 to be merged:
    self.verbose = verbose
    self.random_state = random_state
    super(_base_LMNN, self).__init__(preprocessor)


# slower Python version
class python_LMNN(_base_LMNN):

  def fit(self, X, y):
    k = self.k
    reg = self.regularization
    learn_rate = self.learn_rate

    X, y = self._prepare_inputs(X, y, dtype=float,
                                ensure_min_samples=2)
    num_pts, num_dims = X.shape
    # FIXME Tmp fix waiting for #167 to be merged:
    n_dims = self.num_dims if self.num_dims is not None else num_dims
    unique_labels, label_inds = np.unique(y, return_inverse=True)
    if len(label_inds) != num_pts:
      raise ValueError('Must have one label per point.')
    self.labels_ = np.arange(len(unique_labels))
    self.transformer_ = _initialize_transformer(n_dims, X, y, self.init,
                                                self.verbose,
                                                self.random_state)
    required_k = np.bincount(label_inds).min()
    if self.k > required_k:
      raise ValueError('not enough class labels for specified k'
                       ' (smallest class has %d)' % required_k)

    target_neighbors = self._select_targets(X, label_inds)
    impostors = self._find_impostors(target_neighbors[:, -1], X, label_inds)
    if len(impostors) == 0:
        # L has already been initialized to an identity matrix
        return

    # sum outer products
    dfG = _sum_outer_products(X, target_neighbors.flatten(),
                              np.repeat(np.arange(X.shape[0]), k))
    df = np.zeros_like(dfG)

    # storage
    a1 = [None]*k
    a2 = [None]*k
    for nn_idx in xrange(k):
      a1[nn_idx] = np.array([])
      a2[nn_idx] = np.array([])

    # initialize L
    L = self.transformer_

    # first iteration: we compute variables (including objective and gradient)
    #  at initialization point
    G, objective, total_active, df, a1, a2 = (
        self._loss_grad(X, L, dfG, impostors, 1, k, reg, target_neighbors, df,
                        a1, a2))

    it = 1  # we already made one iteration

    # main loop
    for it in xrange(2, self.max_iter):
      # then at each iteration, we try to find a value of L that has better
      # objective than the previous L, following the gradient:
      while True:
        # the next point next_L to try out is found by a gradient step
        L_next = L - learn_rate * G
        # we compute the objective at next point
        # we copy variables that can be modified by _loss_grad, because if we
        # retry we don t want to modify them several times
        (G_next, objective_next, total_active_next, df_next, a1_next,
         a2_next) = (
            self._loss_grad(X, L_next, dfG, impostors, it, k, reg,
                            target_neighbors, df.copy(), list(a1), list(a2)))
        assert not np.isnan(objective)
        delta_obj = objective_next - objective
        if delta_obj > 0:
          # if we did not find a better objective, we retry with an L closer to
          # the starting point, by decreasing the learning rate (making the
          # gradient step smaller)
          learn_rate /= 2
        else:
          # otherwise, if we indeed found a better obj, we get out of the loop
          break
      # when the better L is found (and the related variables), we set the
      # old variables to these new ones before next iteration and we
      # slightly increase the learning rate
      L = L_next
      G, df, objective, total_active, a1, a2 = (
          G_next, df_next, objective_next, total_active_next, a1_next, a2_next)
      learn_rate *= 1.01

      if self.verbose:
        print(it, objective, delta_obj, total_active, learn_rate)

      # check for convergence
      if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
        if self.verbose:
          print("LMNN converged with objective", objective)
        break
    else:
      if self.verbose:
        print("LMNN didn't converge in %d steps." % self.max_iter)

    # store the last L
    self.transformer_ = L
    self.n_iter_ = it
    return self

  def _loss_grad(self, X, L, dfG, impostors, it, k, reg, target_neighbors, df,
                 a1, a2):
    # Compute pairwise distances under current metric
    Lx = L.dot(X.T).T
    g0 = _inplace_paired_L2(*Lx[impostors])
    Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:, None, :])
    g1, g2 = Ni[impostors]
    # compute the gradient
    total_active = 0
    for nn_idx in reversed(xrange(k)):
      act1 = g0 < g1[:, nn_idx]
      act2 = g0 < g2[:, nn_idx]
      total_active += act1.sum() + act2.sum()

      if it > 1:
        plus1 = act1 & ~a1[nn_idx]
        minus1 = a1[nn_idx] & ~act1
        plus2 = act2 & ~a2[nn_idx]
        minus2 = a2[nn_idx] & ~act2
      else:
        plus1 = act1
        plus2 = act2
        minus1 = np.zeros(0, dtype=int)
        minus2 = np.zeros(0, dtype=int)

      targets = target_neighbors[:, nn_idx]
      PLUS, pweight = _count_edges(plus1, plus2, impostors, targets)
      df += _sum_outer_products(X, PLUS[:, 0], PLUS[:, 1], pweight)
      MINUS, mweight = _count_edges(minus1, minus2, impostors, targets)
      df -= _sum_outer_products(X, MINUS[:, 0], MINUS[:, 1], mweight)

      in_imp, out_imp = impostors
      df += _sum_outer_products(X, in_imp[minus1], out_imp[minus1])
      df += _sum_outer_products(X, in_imp[minus2], out_imp[minus2])

      df -= _sum_outer_products(X, in_imp[plus1], out_imp[plus1])
      df -= _sum_outer_products(X, in_imp[plus2], out_imp[plus2])

      a1[nn_idx] = act1
      a2[nn_idx] = act2
    # do the gradient update
    assert not np.isnan(df).any()
    G = dfG * reg + df * (1 - reg)
    G = L.dot(G)
    # compute the objective function
    objective = total_active * (1 - reg)
    objective += G.flatten().dot(L.flatten())
    return 2 * G, objective, total_active, df, a1, a2

  def _select_targets(self, X, label_inds):
    target_neighbors = np.empty((X.shape[0], self.k), dtype=int)
    for label in self.labels_:
      inds, = np.nonzero(label_inds == label)
      dd = euclidean_distances(X[inds], squared=True)
      np.fill_diagonal(dd, np.inf)
      nn = np.argsort(dd)[..., :self.k]
      target_neighbors[inds] = inds[nn]
    return target_neighbors

  def _find_impostors(self, furthest_neighbors, X, label_inds):
    Lx = self.transform(X)
    margin_radii = 1 + _inplace_paired_L2(Lx[furthest_neighbors], Lx)
    impostors = []
    for label in self.labels_[:-1]:
      in_inds, = np.nonzero(label_inds == label)
      out_inds, = np.nonzero(label_inds > label)
      dist = euclidean_distances(Lx[out_inds], Lx[in_inds], squared=True)
      i1,j1 = np.nonzero(dist < margin_radii[out_inds][:,None])
      i2,j2 = np.nonzero(dist < margin_radii[in_inds])
      i = np.hstack((i1,i2))
      j = np.hstack((j1,j2))
      if i.size > 0:
        # get unique (i,j) pairs using index trickery
        shape = (i.max()+1, j.max()+1)
        tmp = np.ravel_multi_index((i,j), shape)
        i,j = np.unravel_index(np.unique(tmp), shape)
      impostors.append(np.vstack((in_inds[j], out_inds[i])))
    if len(impostors) == 0:
        # No impostors detected
        return impostors
    return np.hstack(impostors)


def _inplace_paired_L2(A, B):
  '''Equivalent to ((A-B)**2).sum(axis=-1), but modifies A in place.'''
  A -= B
  return np.einsum('...ij,...ij->...i', A, A)


def _count_edges(act1, act2, impostors, targets):
  imp = impostors[0,act1]
  c = Counter(zip(imp, targets[imp]))
  imp = impostors[1,act2]
  c.update(zip(imp, targets[imp]))
  if c:
    active_pairs = np.array(list(c.keys()))
  else:
    active_pairs = np.empty((0,2), dtype=int)
  return active_pairs, np.array(list(c.values()))


def _sum_outer_products(data, a_inds, b_inds, weights=None):
  Xab = data[a_inds] - data[b_inds]
  if weights is not None:
    return np.dot(Xab.T, Xab * weights[:,None])
  return np.dot(Xab.T, Xab)


try:
  # use the fast C++ version, if available
  from modshogun import LMNN as shogun_LMNN
  from modshogun import RealFeatures, MulticlassLabels

  class LMNN(_base_LMNN):
    """Large Margin Nearest Neighbor (LMNN)

    Attributes
    ----------
    n_iter_ : `int`
        The number of iterations the solver has run.

    transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
        The learned linear transformation ``L``.
    """

    def fit(self, X, y):
      X, y = self._prepare_inputs(X, y, dtype=float,
                                  ensure_min_samples=2)
      labels = MulticlassLabels(y)
      self._lmnn = shogun_LMNN(RealFeatures(X.T), labels, self.k)
      self._lmnn.set_maxiter(self.max_iter)
      self._lmnn.set_obj_threshold(self.convergence_tol)
      self._lmnn.set_regularization(self.regularization)
      self._lmnn.set_stepsize(self.learn_rate)
      if self.use_pca:
        self._lmnn.train()
      else:
        self._lmnn.train(np.eye(X.shape[1]))
      self.transformer_ = self._lmnn.get_linear_transform(X)
      return self

except ImportError:
  LMNN = python_LMNN
