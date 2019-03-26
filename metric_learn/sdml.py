"""
Qi et al.
An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization.
ICML 2009

Adapted from https://gist.github.com/kcarnold/5439945
Paper: http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-guojun.pdf
"""

from __future__ import absolute_import
import warnings
import numpy as np
from sklearn.base import TransformerMixin
from scipy.linalg import pinvh
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from .base_metric import MahalanobisMixin, _PairsClassifierMixin
from .constraints import Constraints, wrap_pairs
from ._util import transformer_from_metric
try:
  from inverse_covariance import quic
except ImportError:
  HAS_SKGGM = False
else:
  HAS_SKGGM = True


class _BaseSDML(MahalanobisMixin):

  _tuple_size = 2  # constraints are pairs

  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               verbose=False, preprocessor=None):
    """
    Parameters
    ----------
    balance_param : float, optional
        trade off between sparsity and M0 prior

    sparsity_param : float, optional
        trade off between optimizer and sparseness (see graph_lasso)

    use_cov : bool, optional
        controls prior matrix, will use the identity if use_cov=False

    verbose : bool, optional
        if True, prints information while learning

    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be gotten like this: X[indices].
    """
    self.balance_param = balance_param
    self.sparsity_param = sparsity_param
    self.use_cov = use_cov
    self.verbose = verbose
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
    if self.use_cov:
      X = np.vstack({tuple(row) for row in pairs.reshape(-1, pairs.shape[2])})
      prior_inv = np.atleast_2d(np.cov(X, rowvar=False))
    else:
      prior_inv = np.identity(pairs.shape[2])
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
                    "`balance_param` and/or to set use_cov=False.",
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

    self.transformer_ = transformer_from_metric(np.atleast_2d(M))
    return self


class SDML(_BaseSDML, _PairsClassifierMixin):
  """Sparse Distance Metric Learning (SDML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)
  """

  def fit(self, pairs, y):
    """Learn the SDML model.

    Parameters
    ----------
    pairs: array-like, shape=(n_constraints, 2, n_features) or
           (n_constraints, 2)
        3D Array of pairs with each row corresponding to two points,
        or 2D array of indices of pairs if the metric learner uses a
        preprocessor.
    y: array-like, of shape (n_constraints,)
        Labels of constraints. Should be -1 for dissimilar pair, 1 for similar.

    Returns
    -------
    self : object
        Returns the instance.
    """
    return self._fit(pairs, y)


class SDML_Supervised(_BaseSDML, TransformerMixin):
  """Supervised version of Sparse Distance Metric Learning (SDML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `transformer_from_metric`.)
  """

  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               num_labeled='deprecated', num_constraints=None, verbose=False,
               preprocessor=None):
    """Initialize the supervised version of `SDML`.

    `SDML_Supervised` creates pairs of similar sample by taking same class
    samples, and pairs of dissimilar samples by taking different class
    samples. It then passes these pairs to `SDML` for training.
    Parameters
    ----------
    balance_param : float, optional
        trade off between sparsity and M0 prior
    sparsity_param : float, optional
        trade off between optimizer and sparseness (see graph_lasso)
    use_cov : bool, optional
        controls prior matrix, will use the identity if use_cov=False
    num_labeled : Not used
      .. deprecated:: 0.5.0
         `num_labeled` was deprecated in version 0.5.0 and will
         be removed in 0.6.0.
    num_constraints : int, optional
        number of constraints to generate
    verbose : bool, optional
        if True, prints information while learning
    preprocessor : array-like, shape=(n_samples, n_features) or callable
        The preprocessor to call to get tuples from indices. If array-like,
        tuples will be formed like this: X[indices].
    """
    _BaseSDML.__init__(self, balance_param=balance_param,
                       sparsity_param=sparsity_param, use_cov=use_cov,
                       verbose=verbose, preprocessor=preprocessor)
    self.num_labeled = num_labeled
    self.num_constraints = num_constraints

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the SDML model.

    Parameters
    ----------
    X : array-like, shape (n, d)
        data matrix, where each row corresponds to a single instance
    y : array-like, shape (n,)
        data labels, one for each instance
    random_state : {numpy.random.RandomState, int}, optional
        Random number generator or random seed. If not given, the singleton
        numpy.random will be used.

    Returns
    -------
    self : object
        Returns the instance.
    """
    if self.num_labeled != 'deprecated':
      warnings.warn('"num_labeled" parameter is not used.'
                    ' It has been deprecated in version 0.5.0 and will be'
                    'removed in 0.6.0', DeprecationWarning)
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints(y)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseSDML._fit(self, pairs, y)
