"""
Relative Components Analysis (RCA)
"""

from __future__ import absolute_import
import numpy as np
import warnings
from six.moves import xrange
from sklearn.base import TransformerMixin
from sklearn.exceptions import ChangedBehaviorWarning

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
  for c in xrange(num_chunks):
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

  num_dims : Not used

      .. deprecated:: 0.5.0
        `num_dims` was deprecated in version 0.5.0 and will
        be removed in 0.6.0. Use `n_components` instead.

  pca_comps : Not used
      .. deprecated:: 0.5.0
      `pca_comps` was deprecated in version 0.5.0 and will
      be removed in 0.6.0.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be formed like this: X[indices].

  Examples
  --------
  >>> from metric_learn import RCA_SemiSupervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> rca = RCA_Supervised(num_chunks=30, chunk_size=2)
  >>> rca.fit(X, Y)

  References
  ------------------
  .. [1] `Adjustment learning and relevant component analysis
         <http://citeseerx.ist.\
psu.edu/viewdoc/download?doi=10.1.1.19.2871&rep=rep1&type=pdf>`_ Noam
         Shental, et al.


  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, n_components=None, num_dims='deprecated',
               pca_comps='deprecated', preprocessor=None):
    self.n_components = n_components
    self.num_dims = num_dims
    self.pca_comps = pca_comps
    super(RCA, self).__init__(preprocessor)

  def _check_dimension(self, rank, X):
    d = X.shape[1]
    if rank < d:
      warnings.warn('The inner covariance matrix is not invertible, '
                    'so the transformation matrix may contain Nan values. '
                    'You should reduce the dimensionality of your input,'
                    'for instance using `sklearn.decomposition.PCA` as a '
                    'preprocessing step.')

    dim = _check_n_components(d, self.n_components)
    return dim

  def fit(self, X, chunks):
    """Learn the RCA model.

    Parameters
    ----------
    X : (n x d) data matrix
        Each row corresponds to a single instance
    chunks : (n,) array of ints
        When ``chunks[i] == -1``, point i doesn't belong to any chunklet.
        When ``chunks[i] == j``, point i belongs to chunklet j.
    """
    if self.num_dims != 'deprecated':
      warnings.warn('"num_dims" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    ' removed in 0.6.0. Use "n_components" instead',
                    DeprecationWarning)

    if self.pca_comps != 'deprecated':
      warnings.warn(
          '"pca_comps" parameter is not used. '
          'It has been deprecated in version 0.5.0 and will be'
          'removed in 0.6.0. RCA will not do PCA preprocessing anymore. If '
          'you still want to do it, you could use '
          '`sklearn.decomposition.PCA` and an `sklearn.pipeline.Pipeline`.',
          DeprecationWarning)

    X, chunks = self._prepare_inputs(X, chunks, ensure_min_samples=2)

    warnings.warn(
        "RCA will no longer center the data before training. If you want "
        "to do some preprocessing, you should do it manually (you can also "
        "use an `sklearn.pipeline.Pipeline` for instance). This warning "
        "will disappear in version 0.6.0.", ChangedBehaviorWarning)

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

  num_dims : Not used

      .. deprecated:: 0.5.0
        `num_dims` was deprecated in version 0.5.0 and will
        be removed in 0.6.0. Use `n_components` instead.

  num_chunks: int, optional

  chunk_size: int, optional

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.
      It is used to randomly sample constraints from labels.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims='deprecated', n_components=None,
               pca_comps='deprecated', num_chunks=100, chunk_size=2,
               preprocessor=None, random_state=None):
    """Initialize the supervised version of `RCA`."""
    RCA.__init__(self, num_dims=num_dims, n_components=n_components,
                 pca_comps=pca_comps, preprocessor=preprocessor)
    self.num_chunks = num_chunks
    self.chunk_size = chunk_size
    self.random_state = random_state

  def fit(self, X, y, random_state='deprecated'):
    """Create constraints from labels and learn the RCA model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    y : (n) data labels
    random_state : Not used
      .. deprecated:: 0.5.0
        `random_state` in the `fit` function was deprecated in version 0.5.0
        and will be removed in 0.6.0. Set `random_state` at initialization
        instead (when instantiating a new `RCA_Supervised` object).
    """
    if random_state != 'deprecated':
      warnings.warn('"random_state" parameter in the `fit` function is '
                    'deprecated. Set `random_state` at initialization '
                    'instead (when instantiating a new `RCA_Supervised` '
                    'object).', DeprecationWarning)
    else:
      warnings.warn('As of v0.5.0, `RCA_Supervised` now uses the '
                    '`random_state` given at initialization to sample '
                    'constraints, not the default `np.random` from the `fit` '
                    'method, since this argument is now deprecated. '
                    'This warning will disappear in v0.6.0.',
                    ChangedBehaviorWarning)
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    chunks = Constraints(y).chunks(num_chunks=self.num_chunks,
                                   chunk_size=self.chunk_size,
                                   random_state=self.random_state)
    return RCA.fit(self, X, chunks)


class RCA_SemiSupervised(RCA):
  """Semi-Supervised version of Relevant Components Analysis (RCA)

  `RCA_SemiSupervised` combines data in the form of chunks with
  data in the form of labeled points that goes through the same
  process as in `RCA_SemiSupervised`.

  Parameters
  ----------
  n_components : int or None, optional (default=None)
      Dimensionality of reduced space (if None, defaults to dimension of X).

  num_dims : Not used

      .. deprecated:: 0.5.0
        `num_dims` was deprecated in version 0.5.0 and will
        be removed in 0.6.0. Use `n_components` instead.

  num_chunks: int, optional

  chunk_size: int, optional

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.
      It is used to randomly sample constraints from labels.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.
  """

  def __init__(self, num_dims='deprecated', n_components=None,
               pca_comps='deprecated', num_chunks=100, chunk_size=2,
               preprocessor=None, random_state=None):
    """Initialize the supervised version of `RCA`."""
    RCA.__init__(self, num_dims=num_dims, n_components=n_components,
                 pca_comps=pca_comps, preprocessor=preprocessor)
    self.num_chunks = num_chunks
    self.chunk_size = chunk_size
    self.random_state = random_state

  def fit(self, X, y, X_u, chunks,
          random_state='deprecated'):
    """Create constraints from labels and learn the RCA model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) labeled data matrix
        each row corresponds to a single instance
    y : (n) data labels
    X_u : (n x d) unlabeled data matrix
    chunks : (n,) array of ints
        When ``chunks[i] == -1``, point i doesn't belong to any chunklet.
        When ``chunks[i] == j``, point i belongs to chunklet j.
    random_state : Not used
      .. deprecated:: 0.5.0
        `random_state` in the `fit` function was deprecated in version 0.5.0
        and will be removed in 0.6.0. Set `random_state` at initialization
        instead (when instantiating a new `RCA_SemiSupervised` object).
    """
    if random_state != 'deprecated':
      warnings.warn('"random_state" parameter in the `fit` function is '
                    'deprecated. Set `random_state` at initialization '
                    'instead (when instantiating a new `RCA_SemiSupervised` '
                    'object).', DeprecationWarning)
    else:
      warnings.warn('As of v0.5.0, `RCA_SemiSupervised` now uses the '
                    '`random_state` given at initialization to sample '
                    'constraints, not the default `np.random` from the `fit` '
                    'method, since this argument is now deprecated. '
                    'This warning will disappear in v0.6.0.',
                    ChangedBehaviorWarning)
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    sup_chunks = Constraints(y).chunks(num_chunks=self.num_chunks,
                                       chunk_size=self.chunk_size,
                                       random_state=self.random_state)
    X_tot = np.concatenate([X, X_u])
    chunks_tot = np.concatenate([sup_chunks, chunks])

    return RCA.fit(self, X_tot, chunks_tot)
