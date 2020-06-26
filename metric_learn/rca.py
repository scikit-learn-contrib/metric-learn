"""
Relative Components Analysis (RCA)
"""

import numpy as np
import warnings
from sklearn.base import TransformerMixin

from ._util import _check_n_components
from .base_metric import MahalanobisMixin
from .constraints import Constraints


# mean center each chunklet separately
def _chunk_mean_centering(data, chunks):
  num_chunks = chunks.max() + 1
  chunk_mask = chunks != -1
  # We need to ensure the data is float so that we can substract the
  # mean on it
  chunk_data = data[chunk_mask].astype(float, copy=False)
  chunk_labels = chunks[chunk_mask]
  for c in range(num_chunks):
    mask = chunk_labels == c
    chunk_data[mask] -= chunk_data[mask].mean(axis=0)

  return chunk_mask, chunk_data


class RCA(MahalanobisMixin, TransformerMixin):
  """Relevant Components Analysis (RCA)

  RCA learns a full rank Mahalanobis distance metric based on a weighted sum of
  in-chunklets covariance matrices. It applies a global linear transformation
  to assign large weights to relevant dimensions and low weights to irrelevant
  dimensions. Those relevant dimensions are estimated using "chunklets",
  subsets of points that are known to belong to the same class.

  Read more in the :ref:`User Guide <rca>`.

  Parameters
  ----------
  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  Examples
  --------
  >>> from metric_learn import RCA
  >>> X = [[-0.05,  3.0],[0.05, -3.0],
  >>>     [0.1, -3.55],[-0.1, 3.55],
  >>>     [-0.95, -0.05],[0.95, 0.05],
  >>>     [0.4,  0.05],[-0.4, -0.05]]
  >>> chunks = [0, 0, 1, 1, 2, 2, 3, 3]
  >>> rca = RCA()
  >>> rca.fit(X, chunks)

  References
  ------------------
  .. [1] Noam Shental, et al. `Adjustment learning and relevant component
         analysis <http://citeseerx.ist.\
         psu.edu/viewdoc/download?doi=10.1.1.19.2871&rep=rep1&type=pdf>`_ .
         ECCV 2002.


  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.
  """

  def __init__(self, n_components=None, preprocessor=None):
    self.n_components = n_components
    super(RCA, self).__init__(preprocessor)

  def _check_dimension(self, rank, X):
    d = X.shape[1]

    if rank < d:
      warnings.warn('The inner covariance matrix is not invertible, '
                    'so the transformation matrix may contain Nan values. '
                    'You should remove any linearly dependent features and/or '
                    'reduce the dimensionality of your input, '
                    'for instance using `sklearn.decomposition.PCA` as a '
                    'preprocessing step.')

    dim = _check_n_components(d, self.n_components)
    return dim

  def fit(self, X, chunks):
    """Learn the RCA model.

    Parameters
    ----------
    data : (n x d) data matrix
      Each row corresponds to a single instance

    chunks : (n,) array of ints
      When ``chunks[i] == -1``, point i doesn't belong to any chunklet.
      When ``chunks[i] == j``, point i belongs to chunklet j.
    """
    X, chunks = self._prepare_inputs(X, chunks, ensure_min_samples=2)

    chunks = np.asanyarray(chunks, dtype=int)
    chunk_mask, chunked_data = _chunk_mean_centering(X, chunks)

    inner_cov = np.atleast_2d(np.cov(chunked_data, rowvar=0, bias=1))
    dim = self._check_dimension(np.linalg.matrix_rank(inner_cov), X)

    # Fisher Linear Discriminant projection
    if dim < X.shape[1]:
      total_cov = np.cov(X[chunk_mask], rowvar=0)
      tmp = np.linalg.lstsq(total_cov, inner_cov)[0]
      vals, vecs = np.linalg.eig(tmp)
      inds = np.argsort(vals)[:dim]
      A = vecs[:, inds]
      inner_cov = np.atleast_2d(A.T.dot(inner_cov).dot(A))
      self.components_ = _inv_sqrtm(inner_cov).dot(A.T)
    else:
      self.components_ = _inv_sqrtm(inner_cov).T

    return self


def _inv_sqrtm(x):
  '''Computes x^(-1/2)'''
  vals, vecs = np.linalg.eigh(x)
  return (vecs / np.sqrt(vals)).dot(vecs.T)


class RCA_Supervised(RCA):
  """Supervised version of Relevant Components Analysis (RCA)

  `RCA_Supervised` creates chunks of similar points by first sampling a
  class, taking `chunk_size` elements in it, and repeating the process
  `num_chunks` times.

  Parameters
  ----------
  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  num_chunks: int, optional (default=100)
    Number of chunks to generate.

  chunk_size: int, optional (default=2)
    Number of points per chunk.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.
    It is used to randomly sample constraints from labels.

  Examples
  --------
  >>> from metric_learn import RCA_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> rca = RCA_Supervised(num_chunks=30, chunk_size=2)
  >>> rca.fit(X, Y)

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.
  """

  def __init__(self, n_components=None, num_chunks=100, chunk_size=2,
               preprocessor=None, random_state=None):
    """Initialize the supervised version of `RCA`."""
    RCA.__init__(self, n_components=n_components, preprocessor=preprocessor)
    self.num_chunks = num_chunks
    self.chunk_size = chunk_size
    self.random_state = random_state

  def fit(self, X, y):
    """Create constraints from labels and learn the RCA model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
      each row corresponds to a single instance

    y : (n) data labels
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    chunks = Constraints(y).chunks(num_chunks=self.num_chunks,
                                   chunk_size=self.chunk_size,
                                   random_state=self.random_state)

    if self.num_chunks * (self.chunk_size - 1) < X.shape[1]:
      warnings.warn('Due to the parameters of RCA_Supervised, '
                    'the inner covariance matrix is not invertible, '
                    'so the transformation matrix will contain Nan values. '
                    'Increase the number or size of the chunks to correct '
                    'this problem.'
                    )

    return RCA.fit(self, X, chunks)
