"""
Large-margin nearest neighbor metric learning. (Weinberger 2005)

TODO: periodic recalculation of impostors, PCA initialization
"""

import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances
from base_metric import BaseMetricLearner


# commonality between LMNN implementations
class _base_LMNN(BaseMetricLearner):
  def __init__(self, **kwargs):
    self.params = kwargs

  def transformer(self):
    return self.L


# slower Python version
class python_LMNN(_base_LMNN):
  """
   LMNN Learns a metric using large-margin nearest neighbor metric learning.
     LMNN(X, labels, k).fit()
   Learn a metric on X (NxD matrix) and labels (Nx1 vector).
   k: number of neighbors to consider, (does not include self-edges)
   regularization: weighting of pull and push terms
  """
  def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,
               regularization=0.5, convergence_tol=0.001):
    _base_LMNN.__init__(self, k=k, min_iter=min_iter, max_iter=max_iter,
                        learn_rate=learn_rate, regularization=regularization,
                        convergence_tol=convergence_tol)

  def _process_inputs(self, X, labels):
    num_pts = X.shape[0]
    assert len(labels) == num_pts
    unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
    self.labels = np.arange(len(unique_labels))
    self.X = X
    self.L = np.eye(X.shape[1])
    required_k = np.bincount(self.label_inds).min()
    k = self.params['k']
    assert k <= required_k, ('not enough class labels for specified k' +
                             ' (smallest class has %d)' % required_k)

  def fit(self, X, labels, verbose=False):
    k = self.params['k']
    reg = self.params['regularization']
    learn_rate = self.params['learn_rate']
    convergence_tol = self.params['convergence_tol']
    min_iter = self.params['min_iter']
    self._process_inputs(X, labels)

    target_neighbors = self._select_targets()
    impostors = self._find_impostors(target_neighbors[:,-1])

    # sum outer products
    dfG = _sum_outer_products(self.X, target_neighbors.flatten(),
                              np.repeat(np.arange(self.X.shape[0]), k))
    df = np.zeros_like(dfG)

    # storage
    a1 = [None]*k
    a2 = [None]*k
    for nn_idx in xrange(k):
      a1[nn_idx] = np.array([])
      a2[nn_idx] = np.array([])

    # initialize gradient and L
    G = dfG * reg + df * (1-reg)
    L = self.L
    objective = np.inf

    # main loop
    for it in xrange(1, self.params['max_iter']):
      df_old = df.copy()
      a1_old = [a.copy() for a in a1]
      a2_old = [a.copy() for a in a2]
      objective_old = objective
      # Compute pairwise distances under current metric
      Lx = L.dot(self.X.T).T
      g0 = _pairwise_L2(*Lx[impostors])
      Ni = ((Lx[:,None,:] - Lx[target_neighbors])**2).sum(axis=2) + 1
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
        df += _sum_outer_products(self.X, PLUS[:,0], PLUS[:,1], pweight)
        MINUS, mweight = _count_edges(minus1, minus2, impostors, targets)
        df -= _sum_outer_products(self.X, MINUS[:,0], MINUS[:,1], mweight)

        in_imp, out_imp = impostors
        df += _sum_outer_products(self.X, in_imp[minus1], out_imp[minus1])
        df += _sum_outer_products(self.X, in_imp[minus2], out_imp[minus2])

        df -= _sum_outer_products(self.X, in_imp[plus1], out_imp[plus1])
        df -= _sum_outer_products(self.X, in_imp[plus2], out_imp[plus2])

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

      if verbose:
        print it, objective, delta_obj, total_active, learn_rate

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
      if it > min_iter and abs(delta_obj) < convergence_tol:
        if verbose:
          print "LMNN converged with objective", objective
        break
    else:
      if verbose:
        print "LMNN didn't converge in %(max_iter)d steps." % self.params

    # store the last L
    self.L = L

  def metric(self):
    return self.L.T.dot(self.L)

  def transform(self, X=None):
    if X is None:
      X = self.X
    return self.L.dot(X.T).T

  def _select_targets(self):
    k = self.params['k']
    target_neighbors = np.empty((self.X.shape[0], k), dtype=int)
    for label in self.labels:
      inds, = np.nonzero(self.label_inds == label)
      dd = pairwise_distances(self.X[inds])
      np.fill_diagonal(dd, np.inf)
      nn = np.argsort(dd)[...,:k]
      target_neighbors[inds] = inds[nn]
    return target_neighbors

  def _find_impostors(self, furthest_neighbors):
    Lx = self.transform()
    margin_radii = _pairwise_L2(Lx, Lx[furthest_neighbors]) + 1
    impostors = []
    for label in self.labels[:-1]:
      in_inds, = np.nonzero(self.label_inds == label)
      out_inds, = np.nonzero(self.label_inds > label)
      dist = pairwise_distances(Lx[out_inds], Lx[in_inds])
      i1,j1 = np.nonzero(dist < margin_radii[out_inds][:,None])
      i2,j2 = np.nonzero(dist < margin_radii[in_inds])
      i = np.hstack((i1,i2))
      j = np.hstack((j1,j2))
      ind = np.vstack((i,j)).T
      if ind.size > 0:
        # gross: get unique rows
        ind = np.array(list(set(map(tuple,ind))))
      i,j = np.atleast_2d(ind).T
      impostors.append(np.vstack((in_inds[j], out_inds[i])))
    return np.hstack(impostors)


def _pairwise_L2(A, B):
  return ((A-B)**2).sum(axis=1)


def _count_edges(act1, act2, impostors, targets):
  imp = impostors[0,act1]
  c = Counter(zip(imp, targets[imp]))
  imp = impostors[1,act2]
  c.update(zip(imp, targets[imp]))
  if c:
    active_pairs = np.array(c.keys())
  else:
    active_pairs = np.empty((0,2), dtype=int)
  return active_pairs, np.array(c.values())


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
    def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,
                 regularization=0.5, convergence_tol=0.001, use_pca=True):
      _base_LMNN.__init__(self, k=k, min_iter=min_iter, max_iter=max_iter,
                          learn_rate=learn_rate, regularization=regularization,
                          convergence_tol=convergence_tol, use_pca=use_pca)

    def fit(self, X, labels, verbose=False):
      self.X = X
      self.L = np.eye(X.shape[1])
      labels = MulticlassLabels(labels.astype(np.float64))
      self._lmnn = shogun_LMNN(RealFeatures(X.T), labels, self.params['k'])
      self._lmnn.set_maxiter(self.params['max_iter'])
      self._lmnn.set_obj_threshold(self.params['convergence_tol'])
      self._lmnn.set_regularization(self.params['regularization'])
      self._lmnn.set_stepsize(self.params['learn_rate'])
      if self.params['use_pca']:
        self._lmnn.train()
      else:
        self._lmnn.train(self.L)
      self.L = self._lmnn.get_linear_transform()

except ImportError:
  LMNN = python_LMNN
