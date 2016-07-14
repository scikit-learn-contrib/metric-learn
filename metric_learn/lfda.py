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
from six.moves import xrange
from sklearn.metrics import pairwise_distances

from .base_metric import BaseMetricLearner


class LFDA(BaseMetricLearner):
  '''
  Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction
  Sugiyama, ICML 2006
  '''
  def __init__(self, dim=None, k=7, metric='weighted'):
    '''
    dim : dimensionality of reduced space (defaults to dimension of X)
    k : nearest neighbor used in local scaling method (default: 7)
    metric : type of metric in the embedding space (default: 'weighted')
      'weighted'        - weighted eigenvectors
      'orthonormalized' - orthonormalized
      'plain'           - raw eigenvectors
    '''
    if metric not in ('weighted', 'orthonormalized', 'plain'):
      raise ValueError('Invalid metric: %r' % metric)

    self.params = {
      'dim': dim,
      'metric': metric,
      'k': k,
    }

  def transformer(self):
    return self._transformer

  def _process_inputs(self, X, Y):
    X = np.asanyarray(X)
    self.X = X
    n, d = X.shape
    unique_classes, Y = np.unique(Y, return_inverse=True)
    num_classes = len(unique_classes)

    if self.params['dim'] is None:
      self.params['dim'] = d
    elif not 0 < self.params['dim'] <= d:
      raise ValueError('Invalid embedding dimension, must be in [1,%d]' % d)

    if not 0 < self.params['k'] < d:
      raise ValueError('Invalid k, must be in [0,%d]' % (d-1))

    return X, Y, num_classes, n, d

  def fit(self, X, Y):
    '''
     X: (n, d) array-like of samples
     Y: (n,) array-like of class labels
    '''
    X, Y, num_classes, n, d = self._process_inputs(X, Y)
    tSb = np.zeros((d,d))
    tSw = np.zeros((d,d))

    for c in xrange(num_classes):
      Xc = X[Y==c]
      nc = Xc.shape[0]

      # classwise affinity matrix
      dist = pairwise_distances(Xc, metric='l2', squared=True)
      # distances to k-th nearest neighbor
      k = min(self.params['k'], nc-1)
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

    if self.params['dim'] == d:
      vals, vecs = scipy.linalg.eigh(tSb, tSw)
    else:
      vals, vecs = scipy.sparse.linalg.eigsh(tSb, k=self.params['dim'], M=tSw,
                                             which='LA')

    order = np.argsort(-vals)[:self.params['dim']]
    vals = vals[order]
    vecs = vecs[:,order]

    if self.params['metric'] == 'weighted':
       vecs *= np.sqrt(vals)
    elif self.params['metric'] == 'orthonormalized':
       vecs, _ = np.linalg.qr(vecs)

    self._transformer = vecs.T
    return self


def _sum_outer(x):
  s = x.sum(axis=0)
  return np.outer(s, s)
