"""
Information Theoretic Metric Learning (ITML)
"""

from __future__ import print_function, absolute_import
import warnings
import numpy as np
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array
from sklearn.base import TransformerMixin
from .base_metric import _PairsClassifierMixin, MahalanobisMixin
from .constraints import Constraints, wrap_pairs
from ._util import transformer_from_metric, _initialize_metric_mahalanobis


class _BaseITML(MahalanobisMixin):
  """Information Theoretic Metric Learning (ITML)"""

  _tuple_size = 2  # constraints are pairs

  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               prior='identity', A0='deprecated', verbose=False,
               preprocessor=None, random_state=None):
    self.gamma = gamma
    self.max_iter = max_iter
    self.convergence_threshold = convergence_threshold
    self.prior = prior
    self.A0 = A0
    self.verbose = verbose
    self.random_state = random_state
    super(_BaseITML, self).__init__(preprocessor)

  def _fit(self, pairs, y, bounds=None):
    if self.A0 != 'deprecated':
      warnings.warn('"A0" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    'removed in 0.6.0. Use "prior" instead.',
                    DeprecationWarning)
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')
    # init bounds
    if bounds is None:
      X = np.vstack({tuple(row) for row in pairs.reshape(-1, pairs.shape[2])})
      self.bounds_ = np.percentile(pairwise_distances(X), (5, 95))
    else:
      bounds = check_array(bounds, allow_nd=False, ensure_min_samples=0,
                           ensure_2d=False)
      bounds = bounds.ravel()
      if bounds.size != 2:
        raise ValueError("`bounds` should be an array-like of two elements.")
      self.bounds_ = bounds
    self.bounds_[self.bounds_ == 0] = 1e-9
    # set the prior
    # pairs will be deduplicated into X two times, TODO: avoid that
    A = _initialize_metric_mahalanobis(pairs, self.prior, self.random_state,
                                       strict_pd=True,
                                       matrix_name='prior')
    gamma = self.gamma
    pos_pairs, neg_pairs = pairs[y == 1], pairs[y == -1]
    num_pos = len(pos_pairs)
    num_neg = len(neg_pairs)
    _lambda = np.zeros(num_pos + num_neg)
    lambdaold = np.zeros_like(_lambda)
    gamma_proj = 1. if gamma is np.inf else gamma/(gamma+1.)
    pos_bhat = np.zeros(num_pos) + self.bounds_[0]
    neg_bhat = np.zeros(num_neg) + self.bounds_[1]
    pos_vv = pos_pairs[:, 0, :] - pos_pairs[:, 1, :]
    neg_vv = neg_pairs[:, 0, :] - neg_pairs[:, 1, :]

    for it in xrange(self.max_iter):
      # update positives
      for i,v in enumerate(pos_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      # update negatives
      for i,v in enumerate(neg_vv):
        wtw = v.dot(A).dot(v)  # scalar
        alpha = min(_lambda[i+num_pos], gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        Av = A.dot(v)
        A += np.outer(Av, Av * beta)

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < self.convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if self.verbose:
        print('itml iter: %d, conv = %f' % (it, conv))

    if self.verbose:
      print('itml converged at iter: %d, conv = %f' % (it, conv))
    self.n_iter_ = it

    self.transformer_ = transformer_from_metric(A)
    return self


class ITML(_BaseITML, _PairsClassifierMixin):
  """Information Theoretic Metric Learning (ITML)

  `ITML` minimizes the (differential) relative entropy, aka Kullback-Leibler
  divergence, between two multivariate Gaussians subject to constraints on the
  associated Mahalanobis distance, which can be formulated into a Bregman
  optimization problem by minimizing the LogDet divergence subject to
  linear constraints. This algorithm can handle a wide variety of constraints
  and can optionally incorporate a prior on the distance function. Unlike some
  other methods, `ITML` does not rely on an eigenvalue computation or
  semi-definite programming.

  Read more in the :ref:`User Guide <itml>`.

  Parameters
  ----------
  gamma : float, optional (default=1.)
      Value for slack variables

  max_iter : int, optional (default=1000)
      Maximum number of iteration of the optimization procedure.

  convergence_threshold : float, optional (default=1e-3)
      Convergence tolerance.

  prior : string or numpy array, optional (default='identity')
      The Mahalanobis matrix to use as a prior. Possible options are
      'identity', 'covariance', 'random', and a numpy array of shape
      (n_features, n_features). For ITML, the prior should be strictly
      positive definite (PD).

      'identity'
          An identity matrix of shape (n_features, n_features).

      'covariance'
          The inverse covariance matrix.

      'random'
          The prior will be a random SPD matrix of shape
          `(n_features, n_features)`, generated using
          `sklearn.datasets.make_spd_matrix`.

      numpy array
          A positive definite (PD) matrix of shape
          (n_features, n_features), that will be used as such to set the
          prior.

  A0 : Not used
      .. deprecated:: 0.5.0
          `A0` was deprecated in version 0.5.0 and will
          be removed in 0.6.0. Use 'prior' instead.

  verbose : bool, optional (default=False)
      If True, prints information while learning

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int. If
      ``prior='random'``, ``random_state`` is used to set the prior.

  Attributes
  ----------
  bounds_ : `numpy.ndarray`, shape=(2,)
      Bounds on similarity, aside slack variables, s.t.
      ``d(a, b) < bounds_[0]`` for all given pairs of similar points ``a``
      and ``b``, and ``d(c, d) > bounds_[1]`` for all given pairs of
      dissimilar points ``c`` and ``d``, with ``d`` the learned distance. If
      not provided at initialization, bounds_[0] and bounds_[1] are set at
      train time to the 5th and 95th percentile of the pairwise distances among
      all points present in the input `pairs`.

  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)

  threshold_ : `float`
      If the distance metric between two points is lower than this threshold,
      points will be classified as similar, otherwise they will be
      classified as dissimilar.

  Examples
  --------
  >>> from metric_learn import ITML_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> itml = ITML_Supervised(num_constraints=200)
  >>> itml.fit(X, Y)

  References
  ----------
  .. [1] `Information-theoretic Metric Learning
         <http://machinelearning.wustl.edu/mlpapers/paper_files\
/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis, et al.
  """

  def fit(self, pairs, y, bounds=None, calibration_params=None):
    """Learn the ITML model.

    The threshold will be calibrated on the trainset using the parameters
    `calibration_params`.

    Parameters
    ----------
    pairs: array-like, shape=(n_constraints, 2, n_features) or \
           (n_constraints, 2)
        3D Array of pairs with each row corresponding to two points,
        or 2D array of indices of pairs if the metric learner uses a
        preprocessor.
    y: array-like, of shape (n_constraints,)
        Labels of constraints. Should be -1 for dissimilar pair, 1 for similar.
    bounds : array-like of two numbers
        Bounds on similarity, aside slack variables, s.t.
        ``d(a, b) < bounds_[0]`` for all given pairs of similar points ``a``
        and ``b``, and ``d(c, d) > bounds_[1]`` for all given pairs of
        dissimilar points ``c`` and ``d``, with ``d`` the learned distance.
        If not provided at initialization, bounds_[0] and bounds_[1] will be
        set to the 5th and 95th percentile of the pairwise distances among all
        points present in the input `pairs`.
    calibration_params : `dict` or `None`
        Dictionary of parameters to give to `calibrate_threshold` for the
        threshold calibration step done at the end of `fit`. If `None` is
        given, `calibrate_threshold` will use the default parameters.

    Returns
    -------
    self : object
        Returns the instance.
    """
    calibration_params = (calibration_params if calibration_params is not
                          None else dict())
    self._validate_calibration_params(**calibration_params)
    self._fit(pairs, y, bounds=bounds)
    self.calibrate_threshold(pairs, y, **calibration_params)
    return self


class ITML_Supervised(_BaseITML, TransformerMixin):
  """Supervised version of Information Theoretic Metric Learning (ITML)

  `ITML_Supervised` creates pairs of similar sample by taking same class
  samples, and pairs of dissimilar samples by taking different class
  samples. It then passes these pairs to `ITML` for training.

  Parameters
  ----------
  gamma : float, optional
      value for slack variables
  max_iter : int, optional
  convergence_threshold : float, optional
  num_labeled : Not used
        .. deprecated:: 0.5.0
           `num_labeled` was deprecated in version 0.5.0 and will
           be removed in 0.6.0.
  num_constraints: int, optional
      number of constraints to generate
  bounds : Not used
         .. deprecated:: 0.5.0
        `bounds` was deprecated in version 0.5.0 and will
        be removed in 0.6.0. Set `bounds` at fit time instead :
        `itml_supervised.fit(X, y, bounds=...)`

  prior : string or numpy array, optional (default='identity')
       Initialization of the Mahalanobis matrix. Possible options are
       'identity', 'covariance', 'random', and a numpy array of shape
       (n_features, n_features). For ITML, the prior should be strictly
       positive definite (PD).

       'identity'
          An identity matrix of shape (n_features, n_features).

       'covariance'
          The inverse covariance matrix.

       'random'
          The prior will be a random SPD matrix of shape
          `(n_features, n_features)`, generated using
          `sklearn.datasets.make_spd_matrix`.

       numpy array
           A positive definite (PD) matrix of shape
           (n_features, n_features), that will be used as such to set the
           prior.

  A0 : Not used
    .. deprecated:: 0.5.0
       `A0` was deprecated in version 0.5.0 and will
       be removed in 0.6.0. Use 'prior' instead.
  verbose : bool, optional
      if True, prints information while learning
  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get tuples from indices. If array-like,
      tuples will be formed like this: X[indices].
  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int. If
      ``prior='random'``, ``random_state`` is used to set the prior.


  Attributes
  ----------
  bounds_ : `numpy.ndarray`, shape=(2,)
      Bounds on similarity, aside slack variables, s.t.
      ``d(a, b) < bounds_[0]`` for all given pairs of similar points ``a``
      and ``b``, and ``d(c, d) > bounds_[1]`` for all given pairs of
      dissimilar points ``c`` and ``d``, with ``d`` the learned distance.
      If not provided at initialization, bounds_[0] and bounds_[1] are set at
      train time to the 5th and 95th percentile of the pairwise distances
      among all points in the training data `X`.

  n_iter_ : `int`
      The number of iterations the solver has run.

  transformer_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)

  See Also
  --------
  metric_learn.ITML : The original weakly-supervised algorithm
  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               num_labeled='deprecated', num_constraints=None,
               bounds='deprecated', prior='identity', A0='deprecated',
               verbose=False, preprocessor=None, random_state=None):
    _BaseITML.__init__(self, gamma=gamma, max_iter=max_iter,
                       convergence_threshold=convergence_threshold,
                       A0=A0, prior=prior, verbose=verbose,
                       preprocessor=preprocessor, random_state=random_state)
    self.num_labeled = num_labeled
    self.num_constraints = num_constraints
    self.bounds = bounds

  def fit(self, X, y, random_state=np.random, bounds=None):
    """Create constraints from labels and learn the ITML model.


    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.

    bounds : array-like of two numbers
        Bounds on similarity, aside slack variables, s.t.
        ``d(a, b) < bounds_[0]`` for all given pairs of similar points ``a``
        and ``b``, and ``d(c, d) > bounds_[1]`` for all given pairs of
        dissimilar points ``c`` and ``d``, with ``d`` the learned distance.
        If not provided at initialization, bounds_[0] and bounds_[1] will be
        set to the 5th and 95th percentile of the pairwise distances among all
        points in the training data `X`.
    """
    # TODO: remove these in v0.6.0
    if self.num_labeled != 'deprecated':
      warnings.warn('"num_labeled" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    ' removed in 0.6.0', DeprecationWarning)
    if self.bounds != 'deprecated':
      warnings.warn('"bounds" parameter from initialization is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    ' removed in 0.6.0. Use the "bounds" parameter of this '
                    'fit method instead.', DeprecationWarning)
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseITML._fit(self, pairs, y, bounds=bounds)
