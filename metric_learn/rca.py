import numpy as np
from base_metric import BaseMetricLearner


class RCA(BaseMetricLearner):
  '''Relevant Components Analysis (RCA)
  'Learning distance functions using equivalence relations', ICML 2003
  '''

  def __init__(self, dim=None):
    '''
    dim : embedding dimension (default: original dimension of data)
    '''
    self.dim = dim

  def transformer(self):
    return self._transformer

  def _process_inputs(self, X, Y):
    X = np.asanyarray(X)
    self.X = X
    n, d = X.shape

    if self.dim is None:
      self.dim = d
    elif not 0 < self.dim <= d:
      raise ValueError('Invalid embedding dimension, must be in [1,%d]' % d)

    Y = np.asanyarray(Y)
    num_chunks = Y.max() + 1

    return X, Y, num_chunks, d

  def fit(self, data, chunks):
    '''
    data : (n,d) array-like, input data
    chunks : (n,) array-like
      chunks[i] == -1  -> point i doesn't belong to any chunklet
      chunks[i] == j   -> point i belongs to chunklet j
    '''
    data, chunks, num_chunks, d = self._process_inputs(data, chunks)

    # mean center
    data -= data.mean(axis=0)

    # mean center each chunklet separately
    chunk_mask = chunks != -1
    chunk_data = data[chunk_mask]
    chunk_labels = chunks[chunk_mask]
    for c in xrange(num_chunks):
      mask = chunk_labels == c
      chunk_data[mask] -= chunk_data[mask].mean(axis=0)

    # "inner" covariance of chunk deviations
    inner_cov = np.cov(chunk_data, rowvar=0, bias=1)

    # Fisher Linear Discriminant projection
    if self.dim < d:
      total_cov = np.cov(data[chunk_mask], rowvar=0)
      tmp = np.linalg.lstsq(total_cov, inner_cov)[0]
      vals, vecs = np.linalg.eig(tmp)
      inds = np.argsort(vals)[:self.dim]
      A = vecs[:,inds]
      inner_cov = A.T.dot(inner_cov).dot(A)
      self._transformer = _inv_sqrtm(inner_cov).dot(A.T)
    else:
      self._transformer = _inv_sqrtm(inner_cov).T


def _inv_sqrtm(x):
  '''Computes x^(-1/2)'''
  vals, vecs = np.linalg.eigh(x)
  return (vecs / np.sqrt(vals)).dot(vecs.T)
