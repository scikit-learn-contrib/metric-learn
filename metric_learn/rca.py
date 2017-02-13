"""Relative Components Analysis (RCA)

RCA learns a full rank Mahalanobis distance metric based on a
weighted sum of in-class covariance matrices.
It applies a global linear transformation to assign large weights to
relevant dimensions and low weights to irrelevant dimensions.
Those relevant dimensions are estimated using "chunklets",
subsets of points that are known to belong to the same class.

'Learning distance functions using equivalence relations', ICML 2003
'Learning a Mahalanobis metric from equivalence constraints', JMLR 2005
"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn import decomposition
import warnings

from .base_metric import BaseMetricLearner
from .constraints import Constraints


# mean center each chunklet separately
def _chunk_mean_centering(data, chunks, num_chunks):
  chunk_mask = chunks != -1
  chunk_data = data[chunk_mask]
  chunk_labels = chunks[chunk_mask]
  for c in xrange(num_chunks):
    mask = chunk_labels == c
    chunk_data[mask] -= chunk_data[mask].mean(axis=0)

  return chunk_mask, chunk_data


class RCA(BaseMetricLearner):
  """Relevant Components Analysis (RCA)"""
  def __init__(self, dim=None, pca_comps=None):
    """Initialize the learner.

    Parameters
    ----------
    dim : int, optional
        embedding dimension (default: original dimension of data)
    pca_comps : int, float, None or string
        Number of components to keep during PCA preprocessing.
        If None (default), does not perform PCA.
        If ``0 < pca_comps < 1``, it is used as the minimum explained variance ratio.
        See sklearn.decomposition.PCA for more details.
    """
    self.params = {'dim': dim, 'pca_comps': pca_comps}

  def transformer(self):
    return self._transformer

  def _process_data(self, data):
    data = np.asanyarray(data)
    self.X = data
    n, d = data.shape
    return data, d

  def _process_chunks(self, data, chunks):
    chunks = np.asanyarray(chunks)
    num_chunks = chunks.max() + 1
    return _chunk_mean_centering(data, chunks, num_chunks)

  def _process_parameters(self, d):
    if self.params['dim'] is None:
      self.params['dim'] = d
    elif not self.params['dim'] > 0:
      raise ValueError('Invalid embedding dimension, dim must be greater than 0.')
    elif self.params['dim'] > d:
      self.params['dim'] = d
      warnings.warn('dim must be smaller than the data dimension. ' +
                    'dim is set to %d.' % (d))
    return self.params['dim']

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

    data, d = self._process_data(data)

    # PCA projection to remove noise and redundant information.
    M_pca = None
    if self.params['pca_comps'] is not None:
      pca = decomposition.PCA(n_components=self.params['pca_comps'],
                              svd_solver='full')
      data = pca.fit_transform(data)
      d = data.shape[1]
      M_pca = pca.components_
    else:
      data -= data.mean(axis=0)

    chunk_mask, chunk_data = self._process_chunks(data, chunks)
    inner_cov = np.cov(chunk_data, rowvar=0, bias=1)
    rank = np.linalg.matrix_rank(inner_cov)

    if rank < d:
      warnings.warn('The inner covariance matrix is not invertible, '
                    'so the transformation matrix may contain Nan values. '
                    'You should adjust pca_comps to remove noise and '
                    'redundant information.')

    # Fisher Linear Discriminant projection
    dim = self._process_parameters(d)
    if dim < d:
      total_cov = np.cov(data[chunk_mask], rowvar=0)
      tmp = np.linalg.lstsq(total_cov, inner_cov)[0]
      vals, vecs = np.linalg.eig(tmp)
      inds = np.argsort(vals)[:dim]
      A = vecs[:, inds]
      inner_cov = A.T.dot(inner_cov).dot(A)
      self._transformer = _inv_sqrtm(inner_cov).dot(A.T)
    else:
      self._transformer = _inv_sqrtm(inner_cov).T

    if M_pca is not None:
        self._transformer = self._transformer.dot(M_pca)

    return self


def _inv_sqrtm(x):
  '''Computes x^(-1/2)'''
  vals, vecs = np.linalg.eigh(x)
  return (vecs / np.sqrt(vals)).dot(vecs.T)


class RCA_Supervised(RCA):
  def __init__(self, dim=None, pca_comps=None, num_chunks=100, chunk_size=2):
    """Initialize the learner.

    Parameters
    ----------
    dim : int, optional
        embedding dimension (default: original dimension of data)
    num_chunks: int, optional
    chunk_size: int, optional
    """
    RCA.__init__(self, dim=dim, pca_comps=pca_comps)
    self.params.update(num_chunks=num_chunks, chunk_size=chunk_size)

  def fit(self, X, labels, random_state=np.random):
    """Create constraints from labels and learn the RCA model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    labels : (n) data labels
    random_state : a random.seed object to fix the random_state if needed.
    """
    chunks = Constraints(labels).chunks(num_chunks=self.params['num_chunks'],
                                        chunk_size=self.params['chunk_size'],
                                        random_state=random_state)
    return RCA.fit(self, X, chunks)
