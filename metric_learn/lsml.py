"""
Liu et al.
"Metric Learning from Relative Comparisons by Minimizing Squared Residual".
ICDM 2012.

Adapted from https://gist.github.com/kcarnold/5439917
Paper: http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf
"""

from __future__ import print_function, absolute_import, division
import numpy as np
import scipy.linalg
from six.moves import xrange
from sklearn.base import TransformerMixin

from .base_metric import _QuadrupletsClassifierMixin, MahalanobisMixin
from .constraints import Constraints


class _BaseLSML(MahalanobisMixin):

  _t = 4  # constraints are quadruplets

  def __init__(self, tol=1e-3, max_iter=1000, prior=None, verbose=False,
               preprocessor=None):
    """Initialize LSML.

    Parameters
    ----------
    tol : float, optional
    max_iter : int, optional
    prior : (d x d) matrix, optional
        guess at a metric [default: inv(covariance(X))]
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    self.prior = prior
    self.tol = tol
    self.max_iter = max_iter
    self.verbose = verbose
    super(_BaseLSML, self).__init__(preprocessor)

  def _prepare_quadruplets(self, quadruplets, weights):
    quadruplets = self.initialize_and_check_inputs(quadruplets,
                                                   type_of_inputs='tuples')

    # check to make sure that no two constrained vectors are identical
    self.vab_ = quadruplets[:, 0, :] - quadruplets[:, 1, :]
    self.vcd_ = quadruplets[:, 2, :] - quadruplets[:, 3, :]
    if self.vab_.shape != self.vcd_.shape:
      raise ValueError('Constraints must have same length')
    if weights is None:
      self.w_ = np.ones(self.vab_.shape[0])
    else:
      self.w_ = weights
    self.w_ /= self.w_.sum()  # weights must sum to 1
    if self.prior is None:
      X = np.vstack({tuple(row) for row in
                     quadruplets.reshape(-1, quadruplets.shape[2])})
      self.prior_inv_ = np.atleast_2d(np.cov(X, rowvar=False))
      self.M_ = np.linalg.inv(self.prior_inv_)
    else:
      self.M_ = self.prior
      self.prior_inv_ = np.linalg.inv(self.prior)

  def _fit(self, quadruplets, weights=None):
    self._prepare_quadruplets(quadruplets, weights)
    step_sizes = np.logspace(-10, 0, 10)
    # Keep track of the best step size and the loss at that step.
    l_best = 0
    s_best = self._total_loss(self.M_)
    if self.verbose:
      print('initial loss', s_best)
    for it in xrange(1, self.max_iter+1):
      grad = self._gradient(self.M_)
      grad_norm = scipy.linalg.norm(grad)
      if grad_norm < self.tol:
        break
      if self.verbose:
        print('gradient norm', grad_norm)
      M_best = None
      for step_size in step_sizes:
        step_size /= grad_norm
        new_metric = self.M_ - step_size * grad
        w, v = scipy.linalg.eigh(new_metric)
        new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
        cur_s = self._total_loss(new_metric)
        if cur_s < s_best:
          l_best = step_size
          s_best = cur_s
          M_best = new_metric
      if self.verbose:
        print('iter', it, 'cost', s_best, 'best step', l_best * grad_norm)
      if M_best is None:
        break
      self.M_ = M_best
    else:
      if self.verbose:
        print("Didn't converge after", it, "iterations. Final loss:", s_best)
    self.n_iter_ = it

    self.transformer_ = self.transformer_from_metric(self.M_)
    return self

  def _comparison_loss(self, metric):
    dab = np.sum(self.vab_.dot(metric) * self.vab_, axis=1)
    dcd = np.sum(self.vcd_.dot(metric) * self.vcd_, axis=1)
    violations = dab > dcd
    return self.w_[violations].dot((np.sqrt(dab[violations]) -
                                    np.sqrt(dcd[violations]))**2)

  def _total_loss(self, metric):
    # Regularization loss
    sign, logdet = np.linalg.slogdet(metric)
    reg_loss = np.sum(metric * self.prior_inv_) - sign * logdet
    return self._comparison_loss(metric) + reg_loss

  def _gradient(self, metric):
    dMetric = self.prior_inv_ - np.linalg.inv(metric)
    dabs = np.sum(self.vab_.dot(metric) * self.vab_, axis=1)
    dcds = np.sum(self.vcd_.dot(metric) * self.vcd_, axis=1)
    violations = dabs > dcds
    # TODO: vectorize
    for vab, dab, vcd, dcd in zip(self.vab_[violations], dabs[violations],
                                  self.vcd_[violations], dcds[violations]):
      dMetric += ((1-np.sqrt(dcd/dab))*np.outer(vab, vab) +
                  (1-np.sqrt(dab/dcd))*np.outer(vcd, vcd))
    return dMetric


class LSML(_BaseLSML, _QuadrupletsClassifierMixin):
  """Least Squared-residual Metric Learning (LSML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See :meth:`transformer_from_metric`.)
  """

  def fit(self, quadruplets, weights=None):
    """Learn the LSML model.

    Parameters
    ----------
    quadruplets : array-like, shape=(n_constraints, 4, n_features) or
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

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See :meth:`transformer_from_metric`.)
  """

  def __init__(self, tol=1e-3, max_iter=1000, prior=None, num_labeled=np.inf,
               num_constraints=None, weights=None, verbose=False,
               preprocessor=None):
    """Initialize the learner.

    Parameters
    ----------
    tol : float, optional
    max_iter : int, optional
    prior : (d x d) matrix, optional
        guess at a metric [default: covariance(X)]
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    weights : (m,) array of floats, optional
        scale factor for each constraint
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    _BaseLSML.__init__(self, tol=tol, max_iter=max_iter, prior=prior,
                       verbose=verbose, preprocessor=preprocessor)
    self.num_labeled = num_labeled
    self.num_constraints = num_constraints
    self.weights = weights

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the LSML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.
    """
    X, y = self.initialize_and_check_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints, same_length=True,
                                        random_state=random_state)
    return _BaseLSML._fit(self, X[np.column_stack(pos_neg)],
                          weights=self.weights)
