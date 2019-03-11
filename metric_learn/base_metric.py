from sklearn.base import BaseEstimator
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import _is_arraylike, check_is_fitted
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import numpy as np
from abc import ABCMeta, abstractmethod
import six
from ._util import ArrayIndexer, check_input, validate_vector
import warnings


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

    See Also
    --------
    get_metric : a method that returns a function to compute the metric between
      two points. The difference with `score_pairs` is that it works on two 1D
      arrays and cannot use a preprocessor. Besides, the returned function is
      independent of the metric learner and hence is not modified if the metric
      learner is.
    """

  def check_preprocessor(self):
    """Initializes the preprocessor"""
    if _is_arraylike(self.preprocessor):
      self.preprocessor_ = ArrayIndexer(self.preprocessor)
    elif callable(self.preprocessor) or self.preprocessor is None:
      self.preprocessor_ = self.preprocessor
    else:
      raise ValueError("Invalid type for the preprocessor: {}. You should "
                       "provide either None, an array-like object, "
                       "or a callable.".format(type(self.preprocessor)))

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

  @abstractmethod
  def get_metric(self):
    """Returns a function that takes as input two 1D arrays and outputs the
    learned metric score on these two points.

    This function will be independent from the metric learner that learned it
    (it will not be modified if the initial metric learner is modified),
    and it can be directly plugged into the `metric` argument of
    scikit-learn's estimators.

    Returns
    -------
    metric_fun : function
      The function described above.


    Examples
    --------
    .. doctest::

      >>> from metric_learn import NCA
      >>> from sklearn.datasets import make_classification
      >>> from sklearn.neighbors import KNeighborsClassifier
      >>> nca = NCA()
      >>> X, y = make_classification()
      >>> nca.fit(X, y)
      >>> knn = KNeighborsClassifier(metric=nca.get_metric())
      >>> knn.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
      KNeighborsClassifier(algorithm='auto', leaf_size=30,
        metric=<function MahalanobisMixin.get_metric.<locals>.metric_fun
                at 0x...>,
        metric_params=None, n_jobs=None, n_neighbors=5, p=2,
        weights='uniform')

    See Also
    --------
    score_pairs : a method that returns the metric score between several pairs
      of points. Unlike `get_metric`, this is a method of the metric learner
      and therefore can change if the metric learner changes. Besides, it can
      use the metric learner's preprocessor, and works on concatenated arrays.
    """


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

    See Also
    --------
    get_metric : a method that returns a function to compute the metric between
      two points. The difference with `score_pairs` is that it works on two 1D
      arrays and cannot use a preprocessor. Besides, the returned function is
      independent of the metric learner and hence is not modified if the metric
      learner is.

    :ref:`mahalanobis_distances` : The section of the project documentation
      that describes Mahalanobis Distances.
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

  def get_metric(self):
    transformer_T = self.transformer_.T.copy()

    def metric_fun(u, v, squared=False):
      """This function computes the metric between u and v, according to the
      previously learned metric.

      Parameters
      ----------
      u : array-like, shape=(n_features,)
        The first point involved in the distance computation.

      v : array-like, shape=(n_features,)
        The second point involved in the distance computation.

      squared : `bool`
        If True, the function will return the squared metric between u and
        v, which is faster to compute.

      Returns
      -------
      distance: float
        The distance between u and v according to the new metric.
      """
      u = validate_vector(u)
      v = validate_vector(v)
      transformed_diff = (u - v).dot(transformer_T)
      dist = np.dot(transformed_diff, transformed_diff.T)
      if not squared:
        dist = np.sqrt(dist)
      return dist

    return metric_fun

  get_metric.__doc__ = BaseMetricLearner.get_metric.__doc__

  def metric(self):
    # TODO: remove this method in version 0.6.0
    warnings.warn(("`metric` is deprecated since version 0.5.0 and will be "
                   "removed in 0.6.0. Use `get_mahalanobis_matrix` instead."),
                  DeprecationWarning)
    return self.get_mahalanobis_matrix()

  def get_mahalanobis_matrix(self):
    """Returns a copy of the Mahalanobis matrix learned by the metric learner.

    Returns
    -------
    M : `numpy.ndarray`, shape=(n_components, n_features)
      The copy of the learned Mahalanobis matrix.
    """
    return self.transformer_.T.dot(self.transformer_)


class _PairsClassifierMixin(BaseMetricLearner):
  """
  Attributes
  ----------
  threshold_ : `float`
      If the distance metric between two points is lower than this threshold,
      points will be classified as similar, otherwise they will be
      classified as dissimilar.

  classes_ : `list`
      The possible labels of the pairs `MMC` can fit on. `classes_ = [-1, 1]`,
      where -1 means points in a pair are dissimilar (negative label), and 1
      means they are similar (positive label).
  """

  classes_ = [-1, 1]
  _tuple_size = 2  # number of points in a tuple, 2 for pairs

  def predict(self, pairs):
    """Predicts the learned metric between input pairs. (For now it just
    calls decision function).

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
    check_is_fitted(self, ['threshold_', 'transformer_'])
    return 2 * (self.decision_function(pairs) >= - self.threshold_) - 1

  def decision_function(self, pairs):
    """Returns the decision function used to classify the pairs.

    Returns the opposite of the learned metric value between samples in every
    pair. Hence it should ideally be low for dissimilar samples and high for
    similar samples. This is the decision function that is used to classify
    pairs as similar (+1), or dissimilar (-1).

    Parameters
    ----------
    pairs : array-like, shape=(n_pairs, 2, n_features) or (n_pairs, 2)
      3D Array of pairs to predict, with each row corresponding to two
      points, or 2D array of indices of pairs if the metric learner uses a
      preprocessor.

    Returns
    -------
    y_predicted : `numpy.ndarray` of floats, shape=(n_constraints,)
      The predicted decision function value for each pair.
    """
    pairs = check_input(pairs, type_of_inputs='tuples',
                        preprocessor=self.preprocessor_,
                        estimator=self, tuple_size=self._tuple_size)
    return - self.score_pairs(pairs)

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

  def set_default_threshold(self, pairs, y):
    """Sets the default threshold on the given dataset.

    Returns a threshold that is the mean between the similar
    metrics mean and the dissimilar metrics mean.

    See more in the :ref:`User Guide <calibration>`.
    """
    similar_threshold = np.mean(self.score_pairs(
        pairs[(y == 1).ravel()]))
    dissimilar_threshold = np.mean(self.score_pairs(
        pairs[(y == -1).ravel()]))
    self.threshold_ = np.mean([similar_threshold, dissimilar_threshold])

  def set_threshold(self, threshold):
    """Sets the threshold of the metric learner to the given value `threshold

    See more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    threshold : float
      The threshold value we want to set. It's a distance metric with
      respect to which the predicted distance metric for test pairs will be
      compared to. If they are superior to the threshold they will be
      classified as similar (+1), and dissimilar (-1) if not.

    Returns
    -------
    self : `_PairsClassifier`
      The pairs classifier with the new threshold set.
    """
    self.threshold_ = threshold
    return self

  def calibrate_threshold(self, pairs_valid, y_valid, strategy='accuracy',
                          threshold=None, beta=None):
    """Decision threshold calibration for binary classification

    Method that calibrates the decision threshold (cutoff point) of the metric
    learner. This threshold will then be used when calling the method
    `predict`. The methods for picking cutoff points make use of traditional
    binary classification evaluation statistics such as the true positive and
    true negative rates and F-scores. The threshold will be found to maximize
    the chosen score on the validation set `(pairs_valid, y_valid)`.

    See more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    strategy : str, optional (default='roc')
      The strategy to use for choosing the cutoff point

      'accuracy'
          selects a decision threshold that maximizes the accuracy
      'f_beta'
          selects a decision threshold that maximizes the f_beta score
      'max_tpr'
          selects the point that yields the highest true positive rate with
          true negative rate at least equal to the value of the parameter
          threshold
      'max_tnr'
          selects the point that yields the highest true negative rate with
          true positive rate at least equal to the value of the parameter
          threshold

    beta : float in [0, 1], optional (default=None)
      beta value to be used in case strategy == 'f_beta'

    threshold : float in [0, 1] or None, (default=None)
      In case strategy is 'max_tpr' or 'max_tnr' this parameter must be set
      to specify the threshold for the true negative rate or true positive
      rate respectively that needs to be achieved

    pairs_valid : array-like, shape=(n_pairs_valid, 2, n_features)
      The validation set of pairs to use to set the threshold.

    y_valid : array-like, shape=(n_pairs_valid,)
      The labels of the pairs of the validation set to use to set the
      threshold.

    References
    ----------
    .. [1] Receiver-operating characteristic (ROC) plots: a fundamental
           evaluation tool in clinical medicine, MH Zweig, G Campbell -
           Clinical chemistry, 1993

    .. [2] most of the code of this function is from scikit-learn's PR #10117

    See Also
    --------
    sklearn.calibration : scikit-learn's module for calibrating classifiers
    """

    if strategy not in ('accuracy', 'f_beta', 'max_tpr',
                        'max_tnr'):
      raise ValueError('Strategy can either be "accuracy", "f_beta" or '
                       '"max_tpr" or "max_tnr". Got "{}" instead.'
                       .format(strategy))

    if strategy == 'max_tpr' or strategy == 'max_tnr':
      if (threshold is None or not isinstance(threshold, (int, float)) or
          not threshold >= 0 or not threshold <= 1):
        raise ValueError('Parameter threshold must be a number in'
                         '[0, 1]. '
                         'Got {} instead.'.format(threshold))

    if strategy == 'f_beta':
      if beta is None or not isinstance(beta, (int, float)):
        raise ValueError('Parameter beta must be a real number. '
                         'Got {} instead.'.format(type(beta)))

    pairs_valid, y_valid = self._prepare_inputs(pairs_valid, y_valid,
                                                type_of_inputs='tuples')

    n_samples = pairs_valid.shape[0]
    if strategy == 'accuracy':
      scores = self.decision_function(pairs_valid)
      scores_sorted_idces = np.argsort(scores)[::-1]
      scores_sorted = scores[scores_sorted_idces]
      # true labels ordered by decision_function value: (higher first)
      y_ordered = y_valid[scores_sorted_idces]
      # finds the threshold that maximizes the accuracy:
      cum_tp = stable_cumsum(y_ordered == 1)  # cumulative number of true
      # positives
      cum_tn_inverted = stable_cumsum(y_ordered[::-1] == -1)
      cum_tn = np.concatenate([[0], cum_tn_inverted[:-1]])[::-1]
      cum_accuracy = (cum_tp + cum_tn) / n_samples
      max_i = np.argmax(cum_accuracy)
      # note: we want a positive threshold (distance), so we take - threshold
      self.threshold_ = - scores_sorted[max_i]
      return self

    if strategy == 'f_beta':
      precision, recall, thresholds = precision_recall_curve(
          y_valid, self.decision_function(pairs_valid), pos_label=1)
      with np.errstate(divide='ignore', invalid='ignore'):
        f_beta = ((1 + beta**2) * (precision * recall) /
                  (beta**2 * precision + recall))
      f_beta[np.isnan(f_beta)] = 0.
      imax = np.argmax(f_beta)
      self.threshold_ = - thresholds[imax]
      return self

    fpr, tpr, thresholds = roc_curve(y_valid,
                                     self.decision_function(pairs_valid),
                                     pos_label=1)
    fpr, tpr, thresholds = fpr, tpr, thresholds

    if strategy == 'max_tpr':
      indices = np.where(1 - fpr >= threshold)[0]
      max_tpr_index = np.argmax(tpr[indices])
      # note: we want a positive threshold (distance), so we take - threshold
      self.threshold_ = - thresholds[indices[max_tpr_index]]

    if strategy == 'max_tnr':
      indices = np.where(tpr >= threshold)[0]
      max_tnr_index = np.argmax(1 - fpr[indices])
      # note: we want a positive threshold (distance), so we take - threshold
      self.threshold_ = - thresholds[indices[max_tnr_index]]
    return self


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
    check_is_fitted(self, 'transformer_')
    quadruplets = check_input(quadruplets, type_of_inputs='tuples',
                              preprocessor=self.preprocessor_,
                              estimator=self, tuple_size=self._tuple_size)
    return np.sign(self.decision_function(quadruplets))

  def decision_function(self, quadruplets):
    """Predicts differences between sample distances in input quadruplets.

    For each quadruplet in the samples, computes the difference between the
    learned metric of the second pair minus the learned metric of the first
    pair. The higher it is, the more probable it is that the pairs in the
    quadruplet are presented in the right order, i.e. that the label of the
    quadruplet is 1. The lower it is, the more probable it is that the label of
    the quadruplet is -1.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_quadruplets, 4, n_features) or
                  (n_quadruplets, 4)
      3D Array of quadruplets to predict, with each row corresponding to four
      points, or 2D array of indices of quadruplets if the metric learner
      uses a preprocessor.

    Returns
    -------
    decision_function : `numpy.ndarray` of floats, shape=(n_constraints,)
      Metric differences.
    """
    return (self.score_pairs(quadruplets[:, 2:]) -
            self.score_pairs(quadruplets[:, :2]))

  def score(self, quadruplets):
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

    Returns
    -------
    score : float
      The quadruplets score.
    """
    return - np.mean(self.predict(quadruplets))
