from numpy.linalg import cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import roc_auc_score
import numpy as np


class BaseMetricLearner(BaseEstimator):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = cholesky(M).T

    Returns
    -------
    L : upper triangular (d x d) matrix
    """
    return cholesky(self.metric()).T


class MetricTransformer(TransformerMixin):

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


class _PairsClassifierMixin:

  def predict(self, pairs):
    """Predicts the learned similarity between input pairs.

    Returns the learned metric value between samples in every pair. It should
    ideally be low for similar samples and high for dissimilar samples.

    Parameters
    ----------
    pairs : array-like, shape=(n_constraints, 2, n_features)
      A constrained dataset of paired samples.

    Returns
    -------
    y_predicted : `numpy.ndarray` of floats, shape=(n_constraints,)
      The predicted learned metric value between samples in every pair.
    """
    pairwise_diffs = pairs[:, 0, :] - pairs[:, 1, :]
    return np.sqrt(np.sum(pairwise_diffs.dot(self.metric()) * pairwise_diffs,
                          axis=1))

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
    pairs : array-like, shape=(n_constraints, 2, n_features)
      Input Pairs.

    y : array-like, shape=(n_constraints,)
      The corresponding labels.

    Returns
    -------
    score : float
      The ``roc_auc`` score.
    """
    return roc_auc_score(y, self.decision_function(pairs))


class _QuadrupletsClassifierMixin:

  def predict(self, quadruplets):
    """Predicts differences between sample similarities in input quadruplets.

    For each quadruplet of samples, computes the difference between the learned
    metric of the first pair minus the learned metric of the second pair.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features)
      Input quadruplets.

    Returns
    -------
    prediction : np.ndarray of floats, shape=(n_constraints,)
      Metric differences.
    """
    similar_diffs = quadruplets[:, 0, :] - quadruplets[:, 1, :]
    dissimilar_diffs = quadruplets[:, 2, :] - quadruplets[:, 3, :]
    return (np.sqrt(np.sum(similar_diffs.dot(self.metric()) *
                           similar_diffs, axis=1)) -
            np.sqrt(np.sum(dissimilar_diffs.dot(self.metric()) *
                           dissimilar_diffs, axis=1)))

  def decision_function(self, quadruplets):
    return self.predict(quadruplets)

  def score(self, quadruplets, y=None):
    """Computes score on an input constrained dataset

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
    predicted_sign = self.decision_function(quadruplets) < 0
    return np.sum(predicted_sign) / predicted_sign.shape[0]
