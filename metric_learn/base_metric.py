from numpy.linalg import cholesky
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.metrics import roc_auc_score
import numpy as np
from abc import ABCMeta, abstractmethod
import six
from ._util import check_tuples


class BaseMetricLearner(BaseEstimator):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  @abstractmethod
  def score_pairs(self, pairs):
    """Returns the score between pairs
    (can be a similarity, or a distance/metric depending on the algorithm)

    Parameters
    ----------
    pairs : `numpy.ndarray`, shape=(n_samples, 2, n_features)
      3D array of pairs.

    Returns
    -------
    scores: `numpy.ndarray` of shape=(n_pairs,)
      The score of every pair.
    """


class MetricTransformer():

  @abstractmethod
  def transform(self, X):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix
        Data to transform.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """


class MahalanobisMixin(six.with_metaclass(ABCMeta, BaseMetricLearner,
                                          MetricTransformer)):
  """Mahalanobis metric learning algorithms.

  Algorithm that learns a Mahalanobis (pseudo) distance :math:`d_M(x, x')`,
  defined between two column vectors :math:`x` and :math:`x'` by: :math:`d_M(x,
  x') = \sqrt{(x-x')^T M (x-x')}`, where :math:`M` is a learned symmetric
  positive semi-definite (PSD) matrix. The metric between points can then be
  expressed as the euclidean distance between points embedded in a new space
  through a linear transformation. Indeed, the above matrix can be decomposed
  into the product of two transpose matrices (through SVD or Cholesky
  decomposition): :math:`d_M(x, x')^2 = (x-x')^T M (x-x') = (x-x')^T L^T L
  (x-x') = (L x - L x')^T (L x- L x')`

  Attributes
  ----------
  transformer_ : `np.ndarray`, shape=(num_dims, n_features)
    The learned linear transformation ``L``.
  """

  def score_pairs(self, pairs):
    """Returns the learned Mahalanobis distance between pairs.

    This distance is defined as: :math:`d_M(x, x') = \sqrt{(x-x')^T M (x-x')}`
    where ``M`` is the learned Mahalanobis matrix, for every pair of points
    ``x`` and ``x'``. This corresponds to the euclidean distance between
    embeddings of the points in a new space, obtained through a linear
    transformation. Indeed, we have also: :math:`d_M(x, x') = \sqrt{(x_e -
    x_e')^T (x_e- x_e')}`, with :math:`x_e = L x` (See
    :class:`MahalanobisMixin`).

    Parameters
    ----------
    pairs : `numpy.ndarray`, shape=(n_samples, 2, n_features)
      3D array of pairs, or 2D array of one pair.

    Returns
    -------
    scores: `numpy.ndarray` of shape=(n_pairs,)
      The learned Mahalanobis distance for every pair.
    """
    pairs = check_tuples(pairs)
    pairwise_diffs = self.transform(pairs[:, 1, :] - pairs[:, 0, :])
    # (for MahalanobisMixin, the embedding is linear so we can just embed the
    # difference)
    return np.sqrt(np.sum(pairwise_diffs**2, axis=-1))

  def transform(self, X):
    """Embeds data points in the learned linear embedding space.

    Transforms samples in ``X`` into ``X_embedded``, samples inside a new
    embedding space such that: ``X_embedded = X.dot(L.T)``, where ``L`` is
    the learned linear transformation (See :class:`MahalanobisMixin`).

    Parameters
    ----------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
      The data points to embed.

    Returns
    -------
    X_embedded : `numpy.ndarray`, shape=(n_samples, num_dims)
      The embedded data points.
    """
    X_checked = check_array(X, accept_sparse=True)
    return X_checked.dot(self.transformer_.T)

  def metric(self):
    return self.transformer_.T.dot(self.transformer_)

  def _transformer_from_metric(self, metric):
    """Computes the transformation matrix from the Mahalanobis matrix.

    Since by definition the metric `M` is positive semi-definite (PSD), it
    admits a Cholesky decomposition: L = cholesky(M).T. However, currently the
    computation of the Cholesky decomposition used does not support
    non-definite matrices. If the metric is not definite, this method will
    return L = V.T w^( -1/2), with M = V*w*V.T being the eigenvector
    decomposition of M with the eigenvalues in the diagonal matrix w and the
    columns of V being the eigenvectors. If M is diagonal, this method will
    just return its elementwise square root (since the diagonalization of
    the matrix is itself).

    Returns
    -------
    L : (d x d) matrix
    """

    if np.allclose(metric, np.diag(np.diag(metric))):
      return np.sqrt(metric)
    elif not np.isclose(np.linalg.det(metric), 0):
      return cholesky(metric).T
    else:
      w, V = np.linalg.eigh(metric)
      return V.T * np.sqrt(np.maximum(0, w[:, None]))


class _PairsClassifierMixin(BaseMetricLearner):

  def predict(self, pairs):
    """Predicts the learned metric between input pairs.

    Returns the learned metric value between samples in every pair. It should
    ideally be low for similar samples and high for dissimilar samples.

    Parameters
    ----------
    pairs : array-like, shape=(n_constraints, 2, n_features)
      Input pairs.

    Returns
    -------
    y_predicted : `numpy.ndarray` of floats, shape=(n_constraints,)
      The predicted learned metric value between samples in every pair.
    """
    pairs = check_tuples(pairs)
    return self.score_pairs(pairs)

  def decision_function(self, pairs):
    pairs = check_tuples(pairs)
    return self.predict(pairs)

  def score(self, pairs, y):
    """Computes score of pairs similarity prediction.

    Returns the ``roc_auc`` score of the fitted metric learner. It is
    computed in the following way: for every value of a threshold
    ``t`` we classify all pairs of samples where the predicted distance is
    inferior to ``t`` as belonging to the "similar" class, and the other as
    belonging to the "dissimilar" class, and we count false positive and
    true positives as in a classical ``roc_auc`` curve.

    Parameters
    ----------
    pairs : array-like, shape=(n_constraints, 2, n_features)
      Input Pairs.

    y : array-like, shape=(n_constraints,)
      The corresponding labels.

    Returns
    -------
    score : float
      The ``roc_auc`` score.
    """
    pairs = check_tuples(pairs)
    return roc_auc_score(y, self.decision_function(pairs))


class _QuadrupletsClassifierMixin(BaseMetricLearner):

  def predict(self, quadruplets):
    """Predicts the ordering between sample distances in input quadruplets.

    For each quadruplet, returns 1 if the quadruplet is in the right order (
    first pair is more similar than second pair), and -1 if not.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features)
      Input quadruplets.

    Returns
    -------
    prediction : `numpy.ndarray` of floats, shape=(n_constraints,)
      Predictions of the ordering of pairs, for each quadruplet.
    """
    quadruplets = check_tuples(quadruplets)
    return np.sign(self.decision_function(quadruplets))

  def decision_function(self, quadruplets):
    """Predicts differences between sample distances in input quadruplets.

    For each quadruplet of samples, computes the difference between the learned
    metric of the first pair minus the learned metric of the second pair.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features)
      Input quadruplets.

    Returns
    -------
    decision_function : `numpy.ndarray` of floats, shape=(n_constraints,)
      Metric differences.
    """
    quadruplets = check_tuples(quadruplets)
    return (self.score_pairs(quadruplets[:, :2, :]) -
            self.score_pairs(quadruplets[:, 2:, :]))

  def score(self, quadruplets, y=None):
    """Computes score on input quadruplets

    Returns the accuracy score of the following classification task: a record
    is correctly classified if the predicted similarity between the first two
    samples is higher than that of the last two.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features)
      Input quadruplets.

    y : Ignored, for scikit-learn compatibility.

    Returns
    -------
    score : float
      The quadruplets score.
    """
    quadruplets = check_tuples(quadruplets)
    return - np.mean(self.predict(quadruplets))
