"""
Large-margin nearest neighbor metric learning. (Weinberger 2005)

LMNN learns a Mahanalobis distance metric in the kNN classification setting
using semidefinite programming.
The learned metric attempts to keep k-nearest neighbors in the same class,
while keeping examples from different classes separated by a large margin.
This algorithm makes no assumptions about the distribution of the data.
"""
#TODO: periodic recalculation of impostors, PCA initialization

from __future__ import print_function, absolute_import
import numpy as np
import warnings
from collections import Counter
from six.moves import xrange
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import euclidean_distances

from .base_metric import BaseMetricLearner, SupervisedMixin


# commonality between LMNN implementations
class _base_LMNN(BaseMetricLearner):
  def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,
               regularization=0.5, convergence_tol=0.001, use_pca=True,
               verbose=False):
    """Initialize the LMNN object.

    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider, not including self-edges.

    regularization: float, optional
        Weighting of pull and push terms, with 0.5 meaning equal weight.
    """
    self.k = k
    self.min_iter = min_iter
    self.max_iter = max_iter
    self.learn_rate = learn_rate
    self.regularization = regularization
    self.convergence_tol = convergence_tol
    self.use_pca = use_pca
    self.verbose = verbose

  def transformer(self):
    return self.L_


# slower Python version
class _python_LMNN(_base_LMNN):

  def _process_inputs(self, X, labels):
    self.X_ = check_array(X, dtype=float)
    num_pts, num_dims = self.X_.shape
    unique_labels, self.label_inds_ = np.unique(labels, return_inverse=True)
    if len(self.label_inds_) != num_pts:
      raise ValueError('Must have one label per point.')
    self.labels_ = np.arange(len(unique_labels))
    if self.use_pca:
      warnings.warn('use_pca does nothing for the python_LMNN implementation')
    self.L_ = np.eye(num_dims)
    required_k = np.bincount(self.label_inds_).min()
    if self.k > required_k:
      raise ValueError('not enough class labels for specified k'
                       ' (smallest class has %d)' % required_k)

  def _fit(self, X, y):
    k = self.k
    reg = self.regularization
    learn_rate = self.learn_rate
    self._process_inputs(X, y)

    target_neighbors = self._select_targets()
    impostors = self._find_impostors(target_neighbors[:,-1])
    if len(impostors) == 0:
        # L has already been initialized to an identity matrix
        return

    # sum outer products
    dfG = _sum_outer_products(self.X_, target_neighbors.flatten(),
                              np.repeat(np.arange(self.X_.shape[0]), k))
    df = np.zeros_like(dfG)

    # storage
    a1 = [None]*k
    a2 = [None]*k
    for nn_idx in xrange(k):
      a1[nn_idx] = np.array([])
      a2[nn_idx] = np.array([])

    # initialize gradient and L
    G = dfG * reg + df * (1-reg)
    L = self.L_
    objective = np.inf

    # main loop
    for it in xrange(1, self.max_iter):
      df_old = df.copy()
      a1_old = [a.copy() for a in a1]
      a2_old = [a.copy() for a in a2]
      objective_old = objective
      # Compute pairwise distances under current metric
      Lx = L.dot(self.X_.T).T
      g0 = _inplace_paired_L2(*Lx[impostors])
      Ni = 1 + _inplace_paired_L2(Lx[target_neighbors], Lx[:,None,:])
      g1,g2 = Ni[impostors]

      # compute the gradient
      total_active = 0
      for nn_idx in reversed(xrange(k)):
        act1 = g0 < g1[:,nn_idx]
        act2 = g0 < g2[:,nn_idx]
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

        targets = target_neighbors[:,nn_idx]
        PLUS, pweight = _count_edges(plus1, plus2, impostors, targets)
        df += _sum_outer_products(self.X_, PLUS[:,0], PLUS[:,1], pweight)
        MINUS, mweight = _count_edges(minus1, minus2, impostors, targets)
        df -= _sum_outer_products(self.X_, MINUS[:,0], MINUS[:,1], mweight)

        in_imp, out_imp = impostors
        df += _sum_outer_products(self.X_, in_imp[minus1], out_imp[minus1])
        df += _sum_outer_products(self.X_, in_imp[minus2], out_imp[minus2])

        df -= _sum_outer_products(self.X_, in_imp[plus1], out_imp[plus1])
        df -= _sum_outer_products(self.X_, in_imp[plus2], out_imp[plus2])

        a1[nn_idx] = act1
        a2[nn_idx] = act2

      # do the gradient update
      assert not np.isnan(df).any()
      G = dfG * reg + df * (1-reg)

      # compute the objective function
      objective = total_active * (1-reg)
      objective += G.flatten().dot(L.T.dot(L).flatten())
      assert not np.isnan(objective)
      delta_obj = objective - objective_old

      if self.verbose:
        print(it, objective, delta_obj, total_active, learn_rate)

      # update step size
      if delta_obj > 0:
        # we're getting worse... roll back!
        learn_rate /= 2.0
        df = df_old
        a1 = a1_old
        a2 = a2_old
        objective = objective_old
      else:
        # update L
        L -= learn_rate * 2 * L.dot(G)
        learn_rate *= 1.01

      # check for convergence
      if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
        if self.verbose:
          print("LMNN converged with objective", objective)
        break
    else:
      if self.verbose:
        print("LMNN didn't converge in %d steps." % self.max_iter)

    # store the last L
    self.L_ = L
    self.n_iter_ = it
    return self

  def _select_targets(self):
    target_neighbors = np.empty((self.X_.shape[0], self.k), dtype=int)
    for label in self.labels_:
      inds, = np.nonzero(self.label_inds_ == label)
      dd = euclidean_distances(self.X_[inds], squared=True)
      np.fill_diagonal(dd, np.inf)
      nn = np.argsort(dd)[..., :self.k]
      target_neighbors[inds] = inds[nn]
    return target_neighbors

  def _find_impostors(self, furthest_neighbors):
    Lx = self.transform()
    margin_radii = 1 + _inplace_paired_L2(Lx[furthest_neighbors], Lx)
    impostors = []
    for label in self.labels_[:-1]:
      in_inds, = np.nonzero(self.label_inds_ == label)
      out_inds, = np.nonzero(self.label_inds_ > label)
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

  class _LMNN(_base_LMNN):

    def _fit(self, X, y):
      self.X_, y = check_X_y(X, y, dtype=float)
      labels = MulticlassLabels(y)
      self._lmnn = shogun_LMNN(RealFeatures(self.X_.T), labels, self.k)
      self._lmnn.set_maxiter(self.max_iter)
      self._lmnn.set_obj_threshold(self.convergence_tol)
      self._lmnn.set_regularization(self.regularization)
      self._lmnn.set_stepsize(self.learn_rate)
      if self.use_pca:
        self._lmnn.train()
      else:
        self._lmnn.train(np.eye(X.shape[1]))
      self.L_ = self._lmnn.get_linear_transform()
      return self

except ImportError:
  _LMNN = _python_LMNN

class LMNN(_LMNN, SupervisedMixin):

  def fit(self, X, y):
    return self._fit(X, y)