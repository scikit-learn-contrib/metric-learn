from numpy.linalg import inv, cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from abc import ABCMeta, abstractmethod
import six


class BaseMetricLearner(BaseEstimator, TransformerMixin):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = cholesky(M).T

    Returns
    -------
    L : upper triangular (d x d) matrix
    """
    return cholesky(self.metric_).T

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    else:
      X = check_array(X, accept_sparse=True)
    L = self.transformer()
    return X.dot(L.T)


class MahalanobisMixin(six.with_metaclass(ABCMeta)):
  """Mahalanobis metric learning algorithms.

  Algorithm that learns a Mahalanobis (pseudo) distance :math:`d_M(x, x')`,
  defined between two column vectors :math:`x` and :math:`x'` by:
  :math:`d_M(x, x') = \sqrt{(x-x')^T M (x-x')}`, where :math:`M` is the
  learned square matrix.

  Attributes
  ----------
  metric_: `np.ndarray`, shape=(n_features, n_features)
      The learned Mahalanobis matrix.
  """

  @property
  @abstractmethod
  def metric_(self):
    pass
