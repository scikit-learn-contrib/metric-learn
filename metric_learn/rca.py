"""Relative Components Analysis (RCA)

RCA learns a full rank Mahalanobis distance metric based on a
weighted sum of in-class covariance matrices.
It applies a global linear transformation to assign large weights to
relevant dimensions and low weights to irrelevant dimensions.
Those relevant dimensions are estimated using "chunklets",
subsets of points that are known to belong to the same class.

'Learning distance functions using equivalence relations', ICML 2003
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange

from .base_metric import BaseMetricLearner
from .constraints import Constraints


class RCA(BaseMetricLearner):
  """Relevant Components Analysis (RCA)"""
  def __init__(self, dim=None):
    """Initialize the learner.

    Parameters
    ----------
    dim : int, optional
        embedding dimension (default: original dimension of data)
    """
    self.params = {
      'dim': dim,
    }

  def transformer(self):
    return self._transformer

  def _process_inputs(self, X, Y):
    X = np.asanyarray(X)
    self.X = X
    n, d = X.shape

    if self.params['dim'] is None:
      self.params['dim'] = d
    elif not 0 < self.params['dim'] <= d:
      raise ValueError('Invalid embedding dimension, must be in [1,%d]' % d)

    Y = np.asanyarray(Y)
    num_chunks = Y.max() + 1

    return X, Y, num_chunks, d

  def fit(self, data, chunks):
    """Learn the RCA model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    chunks : (n,) array of ints
        when ``chunks[i] == -1``, point i doesn't belong to any chunklet,
        when ``chunks[i] == j``, point i belongs to chunklet j.
    """
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
    if self.params['dim'] < d:
      total_cov = np.cov(data[chunk_mask], rowvar=0)
      tmp = np.linalg.lstsq(total_cov, inner_cov)[0]
      vals, vecs = np.linalg.eig(tmp)
      inds = np.argsort(vals)[:self.params['dim']]
      A = vecs[:,inds]
      inner_cov = A.T.dot(inner_cov).dot(A)
      self._transformer = _inv_sqrtm(inner_cov).dot(A.T)
    else:
      self._transformer = _inv_sqrtm(inner_cov).T

    return self


def _inv_sqrtm(x):
  '''Computes x^(-1/2)'''
  vals, vecs = np.linalg.eigh(x)
  return (vecs / np.sqrt(vals)).dot(vecs.T)


class RCA_Supervised(RCA):
  def __init__(self, dim=None, num_chunks=100, chunk_size=2):
    """Initialize the learner.

    Parameters
    ----------
    dim : int, optional
        embedding dimension (default: original dimension of data)
    num_chunks: int, optional
    chunk_size: int, optional
    """
    RCA.__init__(self, dim=dim)
    self.params.update(num_chunks=num_chunks, chunk_size=chunk_size)

  def fit(self, X, labels):
    """Create constraints from labels and learn the LSML model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    labels : (n) data labels
    """
    chunks = Constraints(labels).chunks(num_chunks=self.params['num_chunks'],
                                        chunk_size=self.params['chunk_size'])
    return RCA.fit(self, X, chunks)
