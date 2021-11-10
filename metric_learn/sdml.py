"""
Sparse High-Dimensional Metric Learning (SDML)
"""

import warnings
import numpy as np
from sklearn.base import TransformerMixin
from scipy.linalg import pinvh
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from .base_metric import MahalanobisMixin, _PairsClassifierMixin
from .constraints import Constraints, wrap_pairs
from ._util import components_from_metric, _initialize_metric_mahalanobis
try:
  from inverse_covariance import quic
except ImportError:
  HAS_SKGGM = False
else:
  HAS_SKGGM = True


class _BaseSDML(MahalanobisMixin):

  _tuple_size = 2  # constraints are pairs

  def __init__(self, balance_param=0.5, sparsity_param=0.01, prior='identity',
               verbose=False, preprocessor=None,
               random_state=None):
    self.balance_param = balance_param
    self.sparsity_param = sparsity_param
    self.prior = prior
    self.verbose = verbose
    self.random_state = random_state
    super(_BaseSDML, self).__init__(preprocessor)

  def _fit(self, pairs, y):
    if not HAS_SKGGM:
      if self.verbose:
        print("SDML will use scikit-learn's graphical lasso solver.")
    else:
      if self.verbose:
        print("SDML will use skggm's graphical lasso solver.")
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')

    # set up (the inverse of) the prior M
    # if the prior is the default (None), we raise a warning
    _, prior_inv = _initialize_metric_mahalanobis(
        pairs, self.prior,
        return_inverse=True, strict_pd=True, matrix_name='prior',
        random_state=self.random_state)
    diff = pairs[:, 0] - pairs[:, 1]
    loss_matrix = (diff.T * y).dot(diff)
    emp_cov = prior_inv + self.balance_param * loss_matrix

    # our initialization will be the matrix with emp_cov's eigenvalues,
    # with a constant added so that they are all positive (plus an epsilon
    # to ensure definiteness). This is empirical.
    w, V = np.linalg.eigh(emp_cov)
    min_eigval = np.min(w)
    if min_eigval < 0.:
      warnings.warn("Warning, the input matrix of graphical lasso is not "
                    "positive semi-definite (PSD). The algorithm may diverge, "
                    "and lead to degenerate solutions. "
                    "To prevent that, try to decrease the balance parameter "
                    "`balance_param` and/or to set prior='identity'.",
                    ConvergenceWarning)
      w -= min_eigval  # we translate the eigenvalues to make them all positive
    w += 1e-10  # we add a small offset to avoid definiteness problems
    sigma0 = (V * w).dot(V.T)
    try:
      if HAS_SKGGM:
        theta0 = pinvh(sigma0)
        M, _, _, _, _, _ = quic(emp_cov, lam=self.sparsity_param,
                                msg=self.verbose,
                                Theta0=theta0, Sigma0=sigma0)
      else:
        _, M = graphical_lasso(emp_cov, alpha=self.sparsity_param,
                               verbose=self.verbose,
                               cov_init=sigma0)
      raised_error = None
      w_mahalanobis, _ = np.linalg.eigh(M)
      not_spd = any(w_mahalanobis < 0.)
      not_finite = not np.isfinite(M).all()
    except Exception as e:
      raised_error = e
      not_spd = False  # not_spd not applicable here so we set to False
      not_finite = False  # not_finite not applicable here so we set to False
    if raised_error is not None or not_spd or not_finite:
      msg = ("There was a problem in SDML when using {}'s graphical "
             "lasso solver.").format("skggm" if HAS_SKGGM else "scikit-learn")
      if not HAS_SKGGM:
        skggm_advice = (" skggm's graphical lasso can sometimes converge "
                        "on non SPD cases where scikit-learn's graphical "
                        "lasso fails to converge. Try to install skggm and "
                        "rerun the algorithm (see the README.md for the "
                        "right version of skggm).")
        msg += skggm_advice
      if raised_error is not None:
        msg += " The following error message was thrown: {}.".format(
            raised_error)
      raise RuntimeError(msg)

    self.components_ = components_from_metric(np.atleast_2d(M))
    return self


class SDML(_BaseSDML, _PairsClassifierMixin):
  r"""Sparse Distance Metric Learning (SDML)

  SDML is an efficient sparse metric learning in high-dimensional space via
  double regularization: an L1-penalization on the off-diagonal elements of the
  Mahalanobis matrix :math:`\mathbf{M}`, and a log-determinant divergence
  between :math:`\mathbf{M}` and :math:`\mathbf{M_0}` (set as either
  :math:`\mathbf{I}` or :math:`\mathbf{\Omega}^{-1}`, where
  :math:`\mathbf{\Omega}` is the covariance matrix).

  Read more in the :ref:`User Guide <sdml>`.

  Parameters
  ----------
  balance_param : float, optional (default=0.5)
    Trade off between sparsity and M0 prior.

  sparsity_param : float, optional  (default=0.01)
    Trade off between optimizer and sparseness (see graph_lasso).

  prior : string or numpy array, optional (default='identity')
    Prior to set for the metric. Possible options are
    'identity', 'covariance', 'random', and a numpy array of
    shape (n_features, n_features). For SDML, the prior should be strictly
    positive definite (PD).

    'identity'
      An identity matrix of shape (n_features, n_features).

    'covariance'
      The inverse covariance matrix.

    'random'
      The prior will be a random positive definite (PD) matrix of shape
      `(n_features, n_features)`, generated using
      `sklearn.datasets.make_spd_matrix`.

    numpy array
      A positive definite (PD) matrix of shape
      (n_features, n_features), that will be used as such to set the
      prior.

  verbose : bool, optional (default=False)
    If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be gotten like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``prior='random'``, ``random_state`` is used to set the prior.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `components_from_metric`.)

  threshold_ : `float`
    If the distance metric between two points is lower than this threshold,
    points will be classified as similar, otherwise they will be
    classified as dissimilar.

  Examples
  --------
  >>> from metric_learn import SDML_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> sdml = SDML_Supervised(num_constraints=200)
  >>> sdml.fit(X, Y)

  References
  ----------
  .. [1] Qi et al. `An efficient sparse metric learning in high-dimensional
         space via L1-penalized log-determinant regularization
         <http://www.machinelearning.org/archive/icml2009/papers/46.pdf>`_.
         ICML 2009.

  .. [2] Code adapted from https://gist.github.com/kcarnold/5439945
  """

  def fit(self, pairs, y, calibration_params=None):
    """Learn the SDML model.

    The threshold will be calibrated on the trainset using the parameters
    `calibration_params`.

    Parameters
    ----------
    pairs : array-like, shape=(n_constraints, 2, n_features) or \
           (n_constraints, 2)
      3D Array of pairs with each row corresponding to two points,
      or 2D array of indices of pairs if the metric learner uses a
      preprocessor.

    y : array-like, of shape (n_constraints,)
      Labels of constraints. Should be -1 for dissimilar pair, 1 for similar.

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
    self._fit(pairs, y)
    self.calibrate_threshold(pairs, y, **calibration_params)
    return self


class SDML_Supervised(_BaseSDML, TransformerMixin):
  """Supervised version of Sparse Distance Metric Learning (SDML)

  `SDML_Supervised` creates pairs of similar sample by taking same class
  samples, and pairs of dissimilar samples by taking different class
  samples. It then passes these pairs to `SDML` for training.

  Parameters
  ----------
  balance_param : float, optional (default=0.5)
    Trade off between sparsity and M0 prior.

  sparsity_param : float, optional (default=0.01)
    Trade off between optimizer and sparseness (see graph_lasso).

  prior : string or numpy array, optional (default='identity')
    Prior to set for the metric. Possible options are
    'identity', 'covariance', 'random', and a numpy array of
    shape (n_features, n_features). For SDML, the prior should be strictly
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

  num_constraints : int, optional (default=None)
    Number of constraints to generate. If None, defaults to `20 *
    num_classes**2`.

  verbose : bool, optional (default=False)
    If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to set the random
    prior. In any case, `random_state` is also used to randomly sample
    constraints from labels.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The linear transformation ``L`` deduced from the learned Mahalanobis
    metric (See function `components_from_metric`.)

  See Also
  --------
  metric_learn.SDML : The original weakly-supervised algorithm
  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def __init__(self, balance_param=0.5, sparsity_param=0.01, prior='identity',
               num_constraints=None, verbose=False, preprocessor=None,
               random_state=None):
    _BaseSDML.__init__(self, balance_param=balance_param,
                       sparsity_param=sparsity_param, prior=prior,
                       verbose=verbose,
                       preprocessor=preprocessor, random_state=random_state)
    self.num_constraints = num_constraints

  def fit(self, X, y):
    """Create constraints from labels and learn the SDML model.

    Parameters
    ----------
    X : array-like, shape (n, d)
      data matrix, where each row corresponds to a single instance

    y : array-like, shape (n,)
      data labels, one for each instance

    Returns
    -------
    self : object
      Returns the instance.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=self.random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseSDML._fit(self, pairs, y)
