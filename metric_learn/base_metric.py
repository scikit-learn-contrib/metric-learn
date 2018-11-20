from numpy.linalg import cholesky
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _is_arraylike
from sklearn.metrics import roc_auc_score
import numpy as np
from abc import ABCMeta, abstractmethod
import six
from ._util import ArrayIndexer, check_input


class BaseMetricLearner(six.with_metaclass(ABCMeta, BaseEstimator)):

  def __init__(self, preprocessor=None):
    """

    Parameters
    ----------
    preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be gotten like this: X[indices].
    """
    self.preprocessor = preprocessor

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

  def check_preprocessor(self):
    """Initializes the preprocessor"""
    if _is_arraylike(self.preprocessor):
      self.preprocessor_ = ArrayIndexer(self.preprocessor)
    else:
      self.preprocessor_ = self.preprocessor

  def _prepare_inputs(self, X, y=None, type_of_inputs='classic',
                      **kwargs):
    """Initializes the preprocessor and processes inputs. See `check_input`
    for more details.

    Parameters
    ----------
    input: array-like
      The input data array to check.

    y : array-like
      The input labels array to check.

    type_of_inputs: `str` {'classic', 'tuples'}
      The type of inputs to check. If 'classic', the input should be
      a 2D array-like of points or a 1D array like of indicators of points. If
      'tuples', the input should be a 3D array-like of tuples or a 2D
      array-like of indicators of tuples.

    **kwargs: dict
      Arguments to pass to check_input.

    Returns
    -------
    X : `numpy.ndarray`
      The checked input data array.

    y: `numpy.ndarray` (optional)
      The checked input labels array.
    """
    self.check_preprocessor()
    return check_input(X, y,
                       type_of_inputs=type_of_inputs,
                       preprocessor=self.preprocessor_,
                       estimator=self,
                       tuple_size=getattr(self, '_tuple_size', None),
                       **kwargs)


class MetricTransformer(six.with_metaclass(ABCMeta)):

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
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
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
    pairs : array-like, shape=(n_pairs, 2, n_features) or (n_pairs, 2)
      3D Array of pairs to score, with each row corresponding to two points,
      for 2D array of indices of pairs if the metric learner uses a
      preprocessor.

    Returns
    -------
    scores: `numpy.ndarray` of shape=(n_pairs,)
      The learned Mahalanobis distance for every pair.
    """
    pairs = check_input(pairs, type_of_inputs='tuples',
                        preprocessor=self.preprocessor_,
                        estimator=self, tuple_size=2)
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
    X_checked = check_input(X, type_of_inputs='classic', estimator=self,
                             preprocessor=self.preprocessor_,
                             accept_sparse=True)
    return X_checked.dot(self.transformer_.T)

  def metric(self):
    return self.transformer_.T.dot(self.transformer_)

  def transformer_from_metric(self, metric):
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

  _tuple_size = 2  # number of points in a tuple, 2 for pairs

  def predict(self, pairs):
    """Predicts the learned metric between input pairs.

    Returns the learned metric value between samples in every pair. It should
    ideally be low for similar samples and high for dissimilar samples.

    Parameters
    ----------
    pairs : array-like, shape=(n_pairs, 2, n_features) or (n_pairs, 2)
      3D Array of pairs to predict, with each row corresponding to two
      points, or 2D array of indices of pairs if the metric learner uses a
      preprocessor.

    Returns
    -------
    y_predicted : `numpy.ndarray` of floats, shape=(n_constraints,)
      The predicted learned metric value between samples in every pair.
    """
    pairs = check_input(pairs, type_of_inputs='tuples',
                        preprocessor=self.preprocessor_,
                        estimator=self, tuple_size=self._tuple_size)
    return self.score_pairs(pairs)

  def decision_function(self, pairs):
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
    pairs : array-like, shape=(n_pairs, 2, n_features) or (n_pairs, 2)
      3D Array of pairs, with each row corresponding to two points,
      or 2D array of indices of pairs if the metric learner uses a
      preprocessor.

    y : array-like, shape=(n_constraints,)
      The corresponding labels.

    Returns
    -------
    score : float
      The ``roc_auc`` score.
    """
    return roc_auc_score(y, self.decision_function(pairs))


class _QuadrupletsClassifierMixin(BaseMetricLearner):

  _tuple_size = 4  # number of points in a tuple, 4 for quadruplets

  def predict(self, quadruplets):
    """Predicts the ordering between sample distances in input quadruplets.

    For each quadruplet, returns 1 if the quadruplet is in the right order (
    first pair is more similar than second pair), and -1 if not.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_quadruplets, 4, n_features) or
                  (n_quadruplets, 4)
      3D Array of quadruplets to predict, with each row corresponding to four
      points, or 2D array of indices of quadruplets if the metric learner
      uses a preprocessor.

    Returns
    -------
    prediction : `numpy.ndarray` of floats, shape=(n_constraints,)
      Predictions of the ordering of pairs, for each quadruplet.
    """
    quadruplets = check_input(quadruplets, type_of_inputs='tuples',
                              preprocessor=self.preprocessor_,
                              estimator=self, tuple_size=self._tuple_size)
    # we broadcast with ... because here we allow quadruplets to be
    # either a 3D array of points or 2D array of indices
    return np.sign(self.decision_function(quadruplets))

  def decision_function(self, quadruplets):
    # we broadcast with ... because here we allow quadruplets to be
    # either a 3D array of points or 2D array of indices
    return (self.score_pairs(quadruplets[:, :2, ...]) -
            self.score_pairs(quadruplets[:, 2:, ...]))

  def score(self, quadruplets, y=None):
    """Computes score on input quadruplets

    Returns the accuracy score of the following classification task: a record
    is correctly classified if the predicted similarity between the first two
    samples is higher than that of the last two.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_quadruplets, 4, n_features) or
                  (n_quadruplets, 4)
      3D Array of quadruplets to score, with each row corresponding to four
      points, or 2D array of indices of quadruplets if the metric learner
      uses a preprocessor.

    y : Ignored, for scikit-learn compatibility.

    Returns
    -------
    score : float
      The quadruplets score.
    """
    return -np.mean(self.predict(quadruplets))
