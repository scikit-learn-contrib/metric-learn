"""
Local Fisher Discriminant Analysis (LFDA)
"""
import numpy as np
import scipy
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.base import TransformerMixin

from ._util import _check_n_components
from .base_metric import MahalanobisMixin


class LFDA(MahalanobisMixin, TransformerMixin):
  '''
  Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction

  LFDA is a linear supervised dimensionality reduction method. It is
  particularly useful when dealing with multimodality, where one ore more
  classes consist of separate clusters in input space. The core optimization
  problem of LFDA is solved as a generalized eigenvalue problem.

  Read more in the :ref:`User Guide <lfda>`.

  Parameters
  ----------
  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  k : int, optional (default=None)
    Number of nearest neighbors used in local scaling method. If None,
    defaults to min(7, n_features - 1).

  embedding_type : str, optional (default: 'weighted')
    Type of metric in the embedding space.

    'weighted'
      weighted eigenvectors

    'orthonormalized'
      orthonormalized

    'plain'
      raw eigenvectors

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``L``.

  Examples
  --------

  >>> import numpy as np
  >>> from metric_learn import LFDA
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> lfda = LFDA(k=2, dim=2)
  >>> lfda.fit(X, Y)

  References
  ------------------
  .. [1] Masashi Sugiyama. `Dimensionality Reduction of Multimodal Labeled
         Data by Local Fisher Discriminant Analysis
         <http://www.ms.k.u-tokyo.ac.jp/2007/LFDA.pdf>`_. JMLR 2007.

  .. [2] Yuan Tang. `Local Fisher Discriminant Analysis on Beer Style
        Clustering
        <https://gastrograph.com/resources/whitepapers/local-fisher\
        -discriminant-analysis-on-beer-style-clustering.html#>`_.
  '''

  def __init__(self, n_components=None,
               k=None, embedding_type='weighted', preprocessor=None):
    if embedding_type not in ('weighted', 'orthonormalized', 'plain'):
      raise ValueError('Invalid embedding_type: %r' % embedding_type)
    self.n_components = n_components
    self.embedding_type = embedding_type
    self.k = k
    super(LFDA, self).__init__(preprocessor)

  def fit(self, X, y):
    '''Fit the LFDA model.

    Parameters
    ----------
    X : (n, d) array-like
        Input data.

    y : (n,) array-like
        Class labels, one per point of data.
    '''
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    unique_classes, y = np.unique(y, return_inverse=True)
    n, d = X.shape
    num_classes = len(unique_classes)

    dim = _check_n_components(d, self.n_components)

    if self.k is None:
      k = min(7, d - 1)
    elif self.k >= d:
      warnings.warn('Chosen k (%d) too large, using %d instead.'
                    % (self.k, d - 1))
      k = d - 1
    else:
      k = int(self.k)
    tSb = np.zeros((d, d))
    tSw = np.zeros((d, d))

    for c in range(num_classes):
      Xc = X[y == c]
      nc = Xc.shape[0]

      # classwise affinity matrix
      dist = pairwise_distances(Xc, metric='l2', squared=True)
      # distances to k-th nearest neighbor
      k = min(k, nc - 1)
      sigma = np.sqrt(np.partition(dist, k, axis=0)[:, k])

      local_scale = np.outer(sigma, sigma)
      with np.errstate(divide='ignore', invalid='ignore'):
        A = np.exp(-dist / local_scale)
        A[local_scale == 0] = 0

      G = Xc.T.dot(A.sum(axis=0)[:, None] * Xc) - Xc.T.dot(A).dot(Xc)
      tSb += G / n + (1 - nc / n) * Xc.T.dot(Xc) + _sum_outer(Xc) / n
      tSw += G / nc

    tSb -= _sum_outer(X) / n - tSw

    # symmetrize
    tSb = (tSb + tSb.T) / 2
    tSw = (tSw + tSw.T) / 2

    vals, vecs = _eigh(tSb, tSw, dim)
    order = np.argsort(-vals)[:dim]
    vals = vals[order].real
    vecs = vecs[:, order]

    if self.embedding_type == 'weighted':
       vecs *= np.sqrt(vals)
    elif self.embedding_type == 'orthonormalized':
       vecs, _ = np.linalg.qr(vecs)

    self.components_ = vecs.T
    return self


def _sum_outer(x):
  s = x.sum(axis=0)
  return np.outer(s, s)


def _eigh(a, b, dim):
  try:
    return scipy.sparse.linalg.eigsh(a, k=dim, M=b, which='LA')
  except np.linalg.LinAlgError:
    pass  # scipy already tried eigh for us
  except (ValueError, scipy.sparse.linalg.ArpackNoConvergence):
    try:
      return scipy.linalg.eigh(a, b)
    except np.linalg.LinAlgError:
      pass
  return scipy.linalg.eig(a, b)
