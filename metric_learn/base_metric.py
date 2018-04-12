from sklearn.metrics import roc_auc_score

from metric_learn.constraints import ConstrainedDataset
from numpy.linalg import cholesky
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
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

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix or `ConstrainedDataset` , optional
        Data to transform. If not supplied, the training data will be used.
        In the case of a ConstrainedDataset, X_constrained.X is used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X_
    elif type(X) is ConstrainedDataset:
      X = X.X
    else:
      X = check_array(X, accept_sparse=True)
    L = self.transformer()
    return X.dot(L.T)


class SupervisedMixin(TransformerMixin):

  def __init__(self):
    raise NotImplementedError('UnsupervisedMixin should not be instantiated')

  def fit(self, X, y):
    return NotImplementedError


class UnsupervisedMixin(TransformerMixin):

  def __init__(self):
    raise NotImplementedError('UnsupervisedMixin should not be instantiated')

  def fit(self, X, y=None):
    return NotImplementedError


class WeaklySupervisedMixin(object):

  def __init__(self):
    raise NotImplementedError('WeaklySupervisedMixin should not be '
                              'instantiated')

  def fit_transform(self, X_constrained, y=None, **fit_params):
    """Fit to data, then transform it.

    Fits transformer to X and y with optional parameters fit_params
    and returns a transformed version of X.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, t, n_features)
        Training set of ``n_constraints`` tuples of samples.
    y : None, or numpy array of shape [n_constraints]
        Constraints labels.
    """
    if y is None:
      # fit method of arity 1 (unsupervised transformation)
      return self.fit(X_constrained, **fit_params).transform(X_constrained)
    else:
      # fit method of arity 2 (supervised transformation)
      return self.fit(X_constrained, y, **fit_params).transform(X_constrained)

  def decision_function(self, X_constrained):
      return self.predict(X_constrained)


class PairsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('PairsMixin should not be instantiated')

  def fit(self, X_constrained, y_constraints, **kwargs):
    """Fit a pairs based metric learner.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 2, n_features)
      Training `ConstrainedDataset`.

    y_constraints : array-like, shape=(n_constraints,)
      Labels of constraints (0 for similar pairs, 1 for dissimilar).

    kwargs : Any
      Algorithm specific parameters.

    Returns
    -------
    self : The fitted estimator.
    """
    return self._fit(X_constrained, y_constraints, **kwargs)

  def predict(self, X_constrained):
    """Predicts the learned similarity between input pairs

    Returns the learned metric value between samples in every pair. It should
    ideally be low for similar samples and high for dissimilar samples.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 2, n_features)
      A constrained dataset of paired samples.

    Returns
    -------
    y_predicted : `numpy.ndarray` of floats, shape=(n_constraints,)
      The predicted learned metric value between samples in every pair.
    """
    # TODO: provide better implementation
    pairwise_diffs = (X_constrained.X[X_constrained.c[:, 0]] -
                      X_constrained.X[X_constrained.c[:, 1]])
    return np.sqrt(np.sum(pairwise_diffs.dot(self.metric()) * pairwise_diffs,
                                  axis=1))

  def score(self, X_constrained, y_constraints):
    """Computes score of pairs similarity prediction.

    Returns the ``roc_auc`` score of the fitted metric learner. It is
    computed in the following way: for every value of a threshold
    ``t`` we classify all pairs of samples where the predicted distance is
    inferior to ``t`` as belonging to the "similar" class, and the other as
    belonging to the "dissimilar" class, and we count false positive and
    true positives as in a classical ``roc_auc`` curve.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 2, n_features)
      Constrained dataset of paired samples.

    y_constraints : array-like, shape=(n_constraints,)
      The corresponding labels.

    Returns
    -------
    score : float
      The ``roc_auc`` score.
    """
    return roc_auc_score(y_constraints, self.decision_function(X_constrained))


class TripletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('TripletsMixin should not be '
                              'instantiated')

  def fit(self, X_constrained, y=None, **kwargs):
    """Fit a triplets based metric learner.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 3, n_features)
      Training `ConstrainedDataset`. To give the right supervision to the
      algorithm, the first two points should be more similar than the first
      and the third.

    y : Ignored, for scikit-learn compatibility.

    kwargs : Any
      Algorithm specific parameters.

    Returns
    -------
    self : The fitted estimator.
    """
    return self._fit(X_constrained, **kwargs)


  def predict(self, X_constrained):
    """Predict the difference between samples similarities in input triplets.

    For each triplet of samples in ``X_constrained``, returns the
    difference between the learned similarity between the first and the
    second point, minus the similarity between the first and the third point.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 3, n_features)
      Input constrained dataset.

    Returns
    -------
    prediction : `numpy.ndarray` of floats, shape=(n_constraints,)
      Predictions for each triplet.
    """
    # TODO: provide better implementation
    similar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                    X_constrained.X[X_constrained.c[:, 1]]
    dissimilar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                       X_constrained.X[X_constrained.c[:, 2]]
    return np.sqrt(np.sum(similar_diffs.dot(self.metric()) *
                          similar_diffs, axis=1)) - \
           np.sqrt(np.sum(dissimilar_diffs.dot(self.metric()) *
                          dissimilar_diffs, axis=1))

  def score(self, X_constrained, y=None):
    """Computes score of triplets similarity prediction.

    Returns the accuracy score of the following classification task: a record
    is correctly classified if the predicted similarity between the first two
    samples is higher than that between the first and the third.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 3, n_features)
      Constrained dataset of triplets of samples.

    y: Ignored (for scikit-learn compatibility).

    Returns
    -------
    score: float
      The triplets score.
    """
    predicted_sign = self.decision_function(X_constrained) < 0
    return np.sum(predicted_sign) / predicted_sign.shape[0]



class QuadrupletsMixin(WeaklySupervisedMixin):

  def __init__(self):
    raise NotImplementedError('QuadrupletsMixin should not be '
                              'instantiated')

  def fit(self, X_constrained, y=None, **kwargs):
    """Fit a quadruplets based metric learner.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 4, n_features)
      Training `ConstrainedDataset`. To give the right supervision to the
      algorithm, the first two points should be more similar than the last two.

    y : Ignored, for scikit-learn compatibility.

    kwargs : Any
      Algorithm specific parameters.

    Returns
    -------
    self : The fitted estimator.
    """
    return self._fit(X_constrained, **kwargs)

  def predict(self, X_constrained):
    """Predicts differences between sample similarities in input quadruplets.

    For each quadruplet of samples in ``X_constrained``, computes the
    difference between the learned metric of the first pair minus the learned
    metric of the second pair.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 4, n_features)
      Input constrained dataset.

    Returns
    -------
    prediction : np.ndarray of floats, shape=(n_constraints,)
      Metric differences.
    """
    similar_diffs = X_constrained.X[X_constrained.c[:, 0]] - \
                    X_constrained.X[X_constrained.c[:, 1]]
    dissimilar_diffs = X_constrained.X[X_constrained.c[:, 2]] - \
                       X_constrained.X[X_constrained.c[:, 3]]
    return np.sqrt(np.sum(similar_diffs.dot(self.metric()) *
                          similar_diffs, axis=1)) - \
           np.sqrt(np.sum(dissimilar_diffs.dot(self.metric()) *
                          dissimilar_diffs, axis=1))

  def decision_fuction(self, X_constrained):
      return self.predict(X_constrained)

  def score(self, X_constrained, y=None):
    """Computes score on an input constrained dataset

    Returns the accuracy score of the following classification task: a record
    is correctly classified if the predicted similarity between the first two
    samples is higher than that of the last two.

    Parameters
    ----------
    X_constrained : `ConstrainedDataset`, shape=(n_constraints, 4, n_features)
      Input constrained dataset.

    y : Ignored, for scikit-learn compatibility.

    Returns
    -------
    score : float
      The quadruplets score.
    """
    predicted_sign = self.decision_function(X_constrained) < 0
    return np.sum(predicted_sign) / predicted_sign.shape[0]