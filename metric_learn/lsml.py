"""
Metric Learning from Relative Comparisons by Minimizing Squared Residual (LSML)
"""

import numpy as np
import scipy.linalg
from sklearn.base import TransformerMixin

from .base_metric import _QuadrupletsClassifierMixin, MahalanobisMixin
from .constraints import Constraints
from ._util import components_from_metric, _initialize_metric_mahalanobis


class _BaseLSML(MahalanobisMixin):

  _tuple_size = 4  # constraints are quadruplets

  def __init__(self, tol=1e-3, max_iter=1000, prior='identity',
               verbose=False, preprocessor=None, random_state=None):
    self.prior = prior
    self.tol = tol
    self.max_iter = max_iter
    self.verbose = verbose
    self.random_state = random_state
    super(_BaseLSML, self).__init__(preprocessor)

  def _fit(self, quadruplets, weights=None):
    quadruplets = self._prepare_inputs(quadruplets,
                                       type_of_inputs='tuples')

    # check to make sure that no two constrained vectors are identical
    vab = quadruplets[:, 0, :] - quadruplets[:, 1, :]
    vcd = quadruplets[:, 2, :] - quadruplets[:, 3, :]
    if vab.shape != vcd.shape:
      raise ValueError('Constraints must have same length')
    if weights is None:
      self.w_ = np.ones(vab.shape[0])
    else:
      self.w_ = weights
    self.w_ /= self.w_.sum()  # weights must sum to 1
    M, prior_inv = _initialize_metric_mahalanobis(
        quadruplets, self.prior,
        return_inverse=True, strict_pd=True, matrix_name='prior',
        random_state=self.random_state)

    step_sizes = np.logspace(-10, 0, 10)
    # Keep track of the best step size and the loss at that step.
    l_best = 0
    s_best = self._total_loss(M, vab, vcd, prior_inv)
    if self.verbose:
      print('initial loss', s_best)
    for it in range(1, self.max_iter + 1):
      grad = self._gradient(M, vab, vcd, prior_inv)
      grad_norm = scipy.linalg.norm(grad)
      if grad_norm < self.tol:
        break
      if self.verbose:
        print('gradient norm', grad_norm)
      M_best = None
      for step_size in step_sizes:
        step_size /= grad_norm
        new_metric = M - step_size * grad
        w, v = scipy.linalg.eigh(new_metric)
        new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
        cur_s = self._total_loss(new_metric, vab, vcd, prior_inv)
        if cur_s < s_best:
          l_best = step_size
          s_best = cur_s
          M_best = new_metric
      if self.verbose:
        print('iter', it, 'cost', s_best, 'best step', l_best * grad_norm)
      if M_best is None:
        break
      M = M_best
    else:
      if self.verbose:
        print("Didn't converge after", it, "iterations. Final loss:", s_best)
    self.n_iter_ = it

    self.components_ = components_from_metric(M)
    return self

  def _comparison_loss(self, metric, vab, vcd):
    dab = np.sum(vab.dot(metric) * vab, axis=1)
    dcd = np.sum(vcd.dot(metric) * vcd, axis=1)
    violations = dab > dcd
    return self.w_[violations].dot((np.sqrt(dab[violations]) -
                                    np.sqrt(dcd[violations]))**2)

  def _total_loss(self, metric, vab, vcd, prior_inv):
    # Regularization loss
    sign, logdet = np.linalg.slogdet(metric)
    reg_loss = np.sum(metric * prior_inv) - sign * logdet
    return self._comparison_loss(metric, vab, vcd) + reg_loss

  def _gradient(self, metric, vab, vcd, prior_inv):
    dMetric = prior_inv - np.linalg.inv(metric)
    dabs = np.sum(vab.dot(metric) * vab, axis=1)
    dcds = np.sum(vcd.dot(metric) * vcd, axis=1)
    violations = dabs > dcds
    # TODO: vectorize
    for vab, dab, vcd, dcd in zip(vab[violations], dabs[violations],
                                  vcd[violations], dcds[violations]):
      dMetric += ((1 - np.sqrt(dcd / dab)) * np.outer(vab, vab) +
                  (1 - np.sqrt(dab / dcd)) * np.outer(vcd, vcd))
    return dMetric


class LSML(_BaseLSML, _QuadrupletsClassifierMixin):
  """Least Squared-residual Metric Learning (LSML)

  `LSML` proposes a simple, yet effective, algorithm that minimizes a convex
  objective function corresponding to the sum of squared residuals of
  constraints. This algorithm uses the constraints in the form of the
  relative distance comparisons, such method is especially useful where
  pairwise constraints are not natural to obtain, thus pairwise constraints
  based algorithms become infeasible to be deployed. Furthermore, its sparsity
  extension leads to more stable estimation when the dimension is high and
  only a small amount of constraints is given.

  Read more in the :ref:`User Guide <lsml>`.

  Parameters
  ----------
  prior : string or numpy array, optional (default='identity')
    Prior to set for the metric. Possible options are
    'identity', 'covariance', 'random', and a numpy array of
    shape (n_features, n_features). For LSML, the prior should be strictly
    positive definite (PD).

    'identity'
      An identity matrix of shape (n_features, n_features).

    'covariance'
      The inverse covariance matrix.

    'random'
      The initial Mahalanobis matrix will be a random positive definite
      (PD) matrix of shape `(n_features, n_features)`, generated using
      `sklearn.datasets.make_spd_matrix`.

    numpy array
      A positive definite (PD) matrix of shape
      (n_features, n_features), that will be used as such to set the
      prior.

  tol : float, optional (default=1e-3)
    Convergence tolerance of the optimization procedure.

  max_iter : int, optional (default=1000)
    Maximum number of iteration of the optimization procedure.

  verbose : bool, optional (default=False)
    If True, prints information while learning

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to set the random
    prior.

  Attributes
  ----------
  n_iter_ : `int`
    The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `components_from_metric`.)

  Examples
  --------
  >>> from metric_learn import LSML
  >>> quadruplets = [[[1.2, 7.5], [1.3, 1.5], [6.4, 2.6], [6.2, 9.7]],
  >>>                [[1.3, 4.5], [3.2, 4.6], [6.2, 5.5], [5.4, 5.4]],
  >>>                [[3.2, 7.5], [3.3, 1.5], [8.4, 2.6], [8.2, 9.7]],
  >>>                [[3.3, 4.5], [5.2, 4.6], [8.2, 5.5], [7.4, 5.4]]]
  >>> # we want to make closer points where the first feature is close, and
  >>> # further if the second feature is close
  >>> lsml = LSML()
  >>> lsml.fit(quadruplets)

  References
  ----------
  .. [1] Liu et al. `Metric Learning from Relative Comparisons by Minimizing
         Squared Residual
         <http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf>`_. ICDM 2012.

  .. [2] Code adapted from https://gist.github.com/kcarnold/5439917

  See Also
  --------
  metric_learn.LSML : The original weakly-supervised algorithm

  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def fit(self, quadruplets, weights=None):
    """Learn the LSML model.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features) or \
                  (n_constraints, 4)
      3D array-like of quadruplets of points or 2D array of quadruplets of
      indicators. In order to supervise the algorithm in the right way, we
      should have the four samples ordered in a way such that:
      d(pairs[i, 0],X[i, 1]) < d(X[i, 2], X[i, 3]) for all 0 <= i <
      n_constraints.

    weights : (n_constraints,) array of floats, optional
      scale factor for each constraint

    Returns
    -------
    self : object
      Returns the instance.
    """
    return self._fit(quadruplets, weights=weights)


class LSML_Supervised(_BaseLSML, TransformerMixin):
  """Supervised version of Least Squared-residual Metric Learning (LSML)

  `LSML_Supervised` creates quadruplets from labeled samples by taking two
  samples from the same class, and two samples from different classes.
  This way it builds quadruplets where the two first points must be more
  similar than the two last points.

  Parameters
  ----------
  tol : float, optional (default=1e-3)
    Convergence tolerance of the optimization procedure.

  max_iter : int, optional (default=1000)
    Number of maximum iterations of the optimization procedure.

  prior : string or numpy array, optional (default='identity')
    Prior to set for the metric. Possible options are
    'identity', 'covariance', 'random', and a numpy array of
    shape (n_features, n_features). For LSML, the prior should be strictly
    positive definite (PD).

    'identity'
      An identity matrix of shape (n_features, n_features).

    'covariance'
      The inverse covariance matrix.

    'random'
      The initial Mahalanobis matrix will be a random positive definite
      (PD) matrix of shape `(n_features, n_features)`, generated using
      `sklearn.datasets.make_spd_matrix`.

    numpy array
      A positive definite (PD) matrix of shape
      (n_features, n_features), that will be used as such to set the
      prior.

  num_constraints: int, optional (default=None)
    Number of constraints to generate. If None, default to `20 *
    num_classes**2`.

  weights : (num_constraints,) array of floats, optional (default=None)
    Relative weight given to each constraint. If None, defaults to uniform
    weights.

  verbose : bool, optional (default=False)
    If True, prints information while learning

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to set the random
    prior. In any case, `random_state` is also used to randomly sample
    constraints from labels.

  Examples
  --------
  >>> from metric_learn import LSML_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> lsml = LSML_Supervised(num_constraints=200)
  >>> lsml.fit(X, Y)

  Attributes
  ----------
  n_iter_ : `int`
    The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `components_from_metric`.)
  """

  def __init__(self, tol=1e-3, max_iter=1000, prior='identity',
               num_constraints=None, weights=None,
               verbose=False, preprocessor=None, random_state=None):
    _BaseLSML.__init__(self, tol=tol, max_iter=max_iter, prior=prior,
                       verbose=verbose, preprocessor=preprocessor,
                       random_state=random_state)
    self.num_constraints = num_constraints
    self.weights = weights

  def fit(self, X, y):
    """Create constraints from labels and learn the LSML model.

    Parameters
    ----------
    X : (n x d) matrix
      Input data, where each row corresponds to a single instance.

    y : (n) array-like
      Data labels.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints, same_length=True,
                                        random_state=self.random_state)
    return _BaseLSML._fit(self, X[np.column_stack(pos_neg)],
                          weights=self.weights)
