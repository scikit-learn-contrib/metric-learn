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
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh

from .base_metric import MahalanobisMixin, _PairsClassifierMixin
from .constraints import Constraints, wrap_pairs
from ._util import transformer_from_metric


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
    pairs, y = self._prepare_inputs(pairs, y,
                                    type_of_inputs='tuples')

    # set up prior M
    if self.use_cov:
      X = np.vstack({tuple(row) for row in pairs.reshape(-1, pairs.shape[2])})
      prior = pinvh(np.cov(X, rowvar = False))
    else:
      prior = np.identity(pairs.shape[2])
    diff = pairs[:, 0] - pairs[:, 1]
    loss_matrix = (diff.T * y).dot(diff)
    emp_cov = pinvh(prior) + self.balance_param * loss_matrix
    _, self.M_ = graph_lasso(emp_cov, self.sparsity_param, verbose=self.verbose)

    self.transformer_ = transformer_from_metric(self.M_)
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
