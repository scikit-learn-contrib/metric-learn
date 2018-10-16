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
import warnings
from six.moves import xrange
from sklearn import decomposition
from sklearn.base import TransformerMixin

from metric_learn._util import check_input
from .base_metric import MahalanobisMixin
from .constraints import Constraints


# mean center each chunklet separately
def _chunk_mean_centering(data, chunks):
  num_chunks = chunks.max() + 1
  chunk_mask = chunks != -1
  chunk_data = data[chunk_mask]
  chunk_labels = chunks[chunk_mask]
  for c in xrange(num_chunks):
    mask = chunk_labels == c
    chunk_data[mask] -= chunk_data[mask].mean(axis=0)

  return chunk_mask, chunk_data


class RCA(MahalanobisMixin, TransformerMixin):
  """Relevant Components Analysis (RCA)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims=None, pca_comps=None, preprocessor=None):
    """Initialize the learner.

    Parameters
    ----------
    num_dims : int, optional
        embedding dimension (default: original dimension of data)

    pca_comps : int, float, None or string
        Number of components to keep during PCA preprocessing.
        If None (default), does not perform PCA.
        If ``0 < pca_comps < 1``, it is used as
        the minimum explained variance ratio.
        See sklearn.decomposition.PCA for more details.
    """
    self.num_dims = num_dims
    self.pca_comps = pca_comps
    super(RCA, self).__init__(preprocessor)

  def _process_data(self, X):
    self.check_preprocessor()
    X = check_input(X, type_of_inputs='classic',
                    preprocessor=self.preprocessor_,
                    estimator=self)

    # PCA projection to remove noise and redundant information.
    if self.pca_comps is not None:
      pca = decomposition.PCA(n_components=self.pca_comps)
      X_transformed = pca.fit_transform(X)
      M_pca = pca.components_
    else:
      X_transformed = X - X.mean(axis=0)
      M_pca = None

    return X_transformed, M_pca

  def _check_dimension(self, rank, X):
    d = X.shape[1]
    if rank < d:
      warnings.warn('The inner covariance matrix is not invertible, '
                    'so the transformation matrix may contain Nan values. '
                    'You should adjust pca_comps to remove noise and '
                    'redundant information.')

    if self.num_dims is None:
      dim = d
    elif self.num_dims <= 0:
      raise ValueError('Invalid embedding dimension: must be greater than 0.')
    elif self.num_dims > d:
      dim = d
      warnings.warn('num_dims (%d) must be smaller than '
                    'the data dimension (%d)' % (self.num_dims, d))
    else:
      dim = self.num_dims
    return dim

  def fit(self, data, chunks):
    """Learn the RCA model.

    Parameters
    ----------
    data : (n x d) data matrix
        Each row corresponds to a single instance
    chunks : (n,) array of ints
        When ``chunks[i] == -1``, point i doesn't belong to any chunklet.
        When ``chunks[i] == j``, point i belongs to chunklet j.
    """

    data, M_pca = self._process_data(data)

    chunks = np.asanyarray(chunks, dtype=int)
    chunk_mask, chunked_data = _chunk_mean_centering(data, chunks)

    inner_cov = np.cov(chunked_data, rowvar=0, bias=1)
    dim = self._check_dimension(np.linalg.matrix_rank(inner_cov), data)

    # Fisher Linear Discriminant projection
    if dim < data.shape[1]:
      total_cov = np.cov(data[chunk_mask], rowvar=0)
      tmp = np.linalg.lstsq(total_cov, inner_cov)[0]
      vals, vecs = np.linalg.eig(tmp)
      inds = np.argsort(vals)[:dim]
      A = vecs[:, inds]
      inner_cov = A.T.dot(inner_cov).dot(A)
      self.transformer_ = _inv_sqrtm(inner_cov).dot(A.T)
    else:
      self.transformer_ = _inv_sqrtm(inner_cov).T

    if M_pca is not None:
        self.transformer_ = self.transformer_.dot(M_pca)

    return self


def _inv_sqrtm(x):
  '''Computes x^(-1/2)'''
  vals, vecs = np.linalg.eigh(x)
  return (vecs / np.sqrt(vals)).dot(vecs.T)


class RCA_Supervised(RCA):
  """Supervised version of Relevant Components Analysis (RCA)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims=None, pca_comps=None, num_chunks=100,
               chunk_size=2, preprocessor=None):
    """Initialize the learner.

    Parameters
    ----------
    num_dims : int, optional
        embedding dimension (default: original dimension of data)
    num_chunks: int, optional
    chunk_size: int, optional
    """
    RCA.__init__(self, num_dims=num_dims, pca_comps=pca_comps,
                 preprocessor=preprocessor)
    self.num_chunks = num_chunks
    self.chunk_size = chunk_size

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the RCA model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    y : (n) data labels
    random_state : a random.seed object to fix the random_state if needed.
    """
    self.check_preprocessor()
    X, y = check_input(X, y, type_of_inputs='classic', estimator=self,
                       preprocessor=self.preprocessor_)
    chunks = Constraints(y).chunks(num_chunks=self.num_chunks,
                                   chunk_size=self.chunk_size,
                                   random_state=random_state)
    return RCA.fit(self, X, chunks)
