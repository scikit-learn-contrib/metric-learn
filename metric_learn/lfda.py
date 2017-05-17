"""
Local Fisher Discriminant Analysis (LFDA)

Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction
Sugiyama, ICML 2006

LFDA is a linear supervised dimensionality reduction method.
It is particularly useful when dealing with multimodality,
where one ore more classes consist of separate clusters in input space.
The core optimization problem of LFDA is solved as a generalized
eigenvalue problem.
"""
from __future__ import division, absolute_import
import numpy as np
import scipy
import warnings
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y

from .base_metric import BaseMetricLearner


class LFDA(BaseMetricLearner):
  '''
  Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction
  Sugiyama, ICML 2006
  '''
  def __init__(self, num_dims=None, k=None, embedding_type='weighted'):
    '''
    Initialize LFDA.

    Parameters
    ----------
    num_dims : int, optional
        Dimensionality of reduced space (defaults to dimension of X)

    k : int, optional
        Number of nearest neighbors used in local scaling method.
        Defaults to min(7, num_dims - 1).

    embedding_type : str, optional
        Type of metric in the embedding space (default: 'weighted')
          'weighted'        - weighted eigenvectors
          'orthonormalized' - orthonormalized
          'plain'           - raw eigenvectors
    '''
    if embedding_type not in ('weighted', 'orthonormalized', 'plain'):
      raise ValueError('Invalid embedding_type: %r' % embedding_type)
    self.num_dims = num_dims
    self.embedding_type = embedding_type
    self.k = k

  def transformer(self):
    return self.transformer_

  def _process_inputs(self, X, y):
    unique_classes, y = np.unique(y, return_inverse=True)
    self.X_, y = check_X_y(X, y)
    n, d = self.X_.shape
    num_classes = len(unique_classes)

    if self.num_dims is None:
      dim = d
    else:
      if not 0 < self.num_dims <= d:
        raise ValueError('Invalid num_dims, must be in [1,%d]' % d)
      dim = self.num_dims

    if self.k is None:
      k = min(7, d - 1)
    elif self.k >= d:
      warnings.warn('Chosen k (%d) too large, using %d instead.' % (self.k,d-1))
      k = d - 1
    else:
      k = int(self.k)

    return self.X_, y, num_classes, n, d, dim, k

  def fit(self, X, y):
    '''Fit the LFDA model.

    Parameters
    ----------
    X : (n, d) array-like
        Input data.

    y : (n,) array-like
        Class labels, one per point of data.
    '''
    X, y, num_classes, n, d, dim, k_ = self._process_inputs(X, y)
    tSb = np.zeros((d,d))
    tSw = np.zeros((d,d))

    for c in xrange(num_classes):
      Xc = X[y==c]
      nc = Xc.shape[0]

      # classwise affinity matrix
      dist = pairwise_distances(Xc, metric='l2', squared=True)
      # distances to k-th nearest neighbor
      k = min(k_, nc-1)
      sigma = np.sqrt(np.partition(dist, k, axis=0)[:,k])

      local_scale = np.outer(sigma, sigma)
      with np.errstate(divide='ignore', invalid='ignore'):
        A = np.exp(-dist/local_scale)
        A[local_scale==0] = 0

      G = Xc.T.dot(A.sum(axis=0)[:,None] * Xc) - Xc.T.dot(A).dot(Xc)
      tSb += G/n + (1-nc/n)*Xc.T.dot(Xc) + _sum_outer(Xc)/n
      tSw += G/nc

    tSb -= _sum_outer(X)/n - tSw

    # symmetrize
    tSb = (tSb + tSb.T) / 2
    tSw = (tSw + tSw.T) / 2

    vals, vecs = _eigh(tSb, tSw, dim)
    order = np.argsort(-vals)[:dim]
    vals = vals[order].real
    vecs = vecs[:,order]

    if self.embedding_type == 'weighted':
       vecs *= np.sqrt(vals)
    elif self.embedding_type == 'orthonormalized':
       vecs, _ = np.linalg.qr(vecs)

    self.transformer_ = vecs.T
    return self


def _sum_outer(x):
  s = x.sum(axis=0)
  return np.outer(s, s)


def _eigh(a, b, dim):
  try:
    return scipy.sparse.linalg.eigsh(a, k=dim, M=b, which='LA')
  except (ValueError, scipy.sparse.linalg.ArpackNoConvergence):
    pass
  try:
    return scipy.linalg.eigh(a, b)
  except np.linalg.LinAlgError:
    pass
  return scipy.linalg.eig(a, b)
