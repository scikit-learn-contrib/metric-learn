"""
Covariance metric (baseline method)
"""

import numpy as np
import scipy
from sklearn.base import TransformerMixin

from .base_metric import MahalanobisMixin
from ._util import components_from_metric


class Covariance(MahalanobisMixin, TransformerMixin):
  """Covariance metric (baseline method)

  This method does not "learn" anything, rather it calculates
  the covariance matrix of the input data.

  This is a simple baseline method first introduced in
  On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936

  Read more in the :ref:`User Guide <covariance>`.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `components_from_metric`.)

  Methods
  -------
  fit:
    Calculates the covariance matrix of the input data.

  fit_transform:
    Fit to data, then transform it.

  get_mahalanobis_matrix:
    Returns a copy of the Mahalanobis matrix learned by the metric learner.

  get_metric:
    Returns a function that takes as input two 1D arrays and outputs the
    learned metric score on these two points.

  get_params:
    Get parameters for this estimator.

  pair_distance:
      Returns the (pseudo) distance between pairs, when available.

  pair_score:
    Returns the similarity score between pairs of points.

  score_pairs:
    Deprecated. Returns the learned Mahalanobis distance between pairs.

  set_params:
    Set the parameters of this estimator.

  transform:
    Embeds data points in the learned linear embedding space.

  Examples
  --------
  >>> from metric_learn import Covariance
  >>> from sklearn.datasets import load_iris
  >>> iris = load_iris()['data']
  >>> cov = Covariance().fit(iris)
  >>> x = cov.transform(iris)

  """

  def __init__(self, preprocessor=None):
    super(Covariance, self).__init__(preprocessor)

  def fit(self, X, y=None):
    """
    Calculates the covariance matrix of the input data.

    Parameters
    ----------
    X : data matrix, (n x d)
    y : unused
    """
    X = self._prepare_inputs(X, ensure_min_samples=2)
    M = np.atleast_2d(np.cov(X, rowvar=False))
    if M.size == 1:
      M = 1. / M
    else:
      M = scipy.linalg.pinvh(M)

    self.components_ = components_from_metric(np.atleast_2d(M))
    return self
