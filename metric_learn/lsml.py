"""
Liu et al.
"Metric Learning from Relative Comparisons by Minimizing Squared Residual".
ICDM 2012.

Adapted from https://gist.github.com/kcarnold/5439917
Paper: http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf
"""

from __future__ import print_function, absolute_import
import numpy as np
import scipy.linalg
from six.moves import xrange

from .base_metric import BaseMetricLearner
from .constraints import Constraints


class LSML(BaseMetricLearner):
  def __init__(self, tol=1e-3, max_iter=1000, verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    tol : float, optional
    max_iter : int, optional
    verbose : bool, optional
        if True, prints information while learning
    """
    self.params = {
      'tol': tol,
      'max_iter': max_iter,
      'verbose': verbose,
    }

  def _prepare_inputs(self, X, constraints, weights, prior):
    self.X = X
    a,b,c,d = constraints
    self.vab = X[a] - X[b]
    self.vcd = X[c] - X[d]
    assert self.vab.shape == self.vcd.shape, 'Constraints must have same length'
    if weights is None:
      self.w = np.ones(self.vab.shape[0])
    else:
      self.w = weights
    self.w /= self.w.sum()  # weights must sum to 1
    if prior is None:
      self.M = np.cov(X.T)
    else:
      self.M = prior

  def metric(self):
    return self.M

  def fit(self, X, constraints, weights=None, prior=None):
    """Learn the LSML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, such that d(X[a],X[b]) < d(X[c],X[d])
    weights : (m,) array of floats, optional
        scale factor for each constraint
    prior : (d x d) matrix, optional
        guess at a metric [default: covariance(X)]
    """
    verbose = self.params['verbose']
    self._prepare_inputs(X, constraints, weights, prior)
    prior_inv = scipy.linalg.inv(self.M)
    s_best = self._total_loss(self.M, prior_inv)
    step_sizes = np.logspace(-10, 0, 10)
    if verbose:
      print('initial loss', s_best)
    tol = self.params['tol']
    for it in xrange(1, self.params['max_iter']+1):
      grad = self._gradient(self.M, prior_inv)
      grad_norm = scipy.linalg.norm(grad)
      if grad_norm < tol:
        break
      if verbose:
        print('gradient norm', grad_norm)
      M_best = None
      for step_size in step_sizes:
        step_size /= grad_norm
        new_metric = self.M - step_size * grad
        w, v = scipy.linalg.eigh(new_metric)
        new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
        cur_s = self._total_loss(new_metric, prior_inv)
        if cur_s < s_best:
          l_best = step_size
          s_best = cur_s
          M_best = new_metric
      if verbose:
        print('iter', it, 'cost', s_best, 'best step', l_best * grad_norm)
      if M_best is None:
        break
      self.M = M_best
    else:
      if verbose:
        print("Didn't converge after", it, "iterations. Final loss:", s_best)
    return self

  def _comparison_loss(self, metric):
    dab = np.sum(self.vab.dot(metric) * self.vab, axis=1)
    dcd = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
    violations = dab > dcd
    return self.w[violations].dot((np.sqrt(dab[violations]) -
                                   np.sqrt(dcd[violations]))**2)

  def _total_loss(self, metric, prior_inv):
    return (self._comparison_loss(metric) +
            _regularization_loss(metric, prior_inv))

  def _gradient(self, metric, prior_inv):
    dMetric = prior_inv - scipy.linalg.inv(metric)
    dabs = np.sum(self.vab.dot(metric) * self.vab, axis=1)
    dcds = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
    violations = dabs > dcds
    # TODO: vectorize
    for vab, dab, vcd, dcd in zip(self.vab[violations], dabs[violations],
                                  self.vcd[violations], dcds[violations]):
      dMetric += ((1-np.sqrt(dcd/dab))*np.outer(vab, vab) +
                  (1-np.sqrt(dab/dcd))*np.outer(vcd, vcd))
    return dMetric


def _regularization_loss(metric, prior_inv):
  sign, logdet = np.linalg.slogdet(metric)
  return np.sum(metric * prior_inv) - sign * logdet


class LSML_Supervised(LSML):
  def __init__(self, tol=1e-3, max_iter=1000, prior=None, num_labeled=np.inf,
               num_constraints=None, weights=None, verbose=False):
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
    """
    LSML.__init__(self, tol=tol, max_iter=max_iter, verbose=verbose)
    self.params.update(prior=prior, num_labeled=num_labeled,
                       num_constraints=num_constraints, weights=weights)

  def fit(self, X, labels, random_state=np.random):
    """Create constraints from labels and learn the LSML model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    labels : (n) data labels
    random_state : a numpy random.seed object to fix the random_state if needed.
    """
    num_constraints = self.params['num_constraints']
    if num_constraints is None:
      num_classes = np.unique(labels)
      num_constraints = 20*(len(num_classes))**2

    c = Constraints.random_subset(labels, self.params['num_labeled'],
                                  random_state=random_state)
    pairs = c.positive_negative_pairs(num_constraints, same_length=True,
                                      random_state=random_state)
    return LSML.fit(self, X, pairs, weights=self.params['weights'],
                    prior=self.params['prior'])
