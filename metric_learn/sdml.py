"""
Qi et al.
An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization.
ICML 2009

Adapted from https://gist.github.com/kcarnold/5439945
Paper: http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-guojun.pdf
"""

from __future__ import absolute_import
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh
from metric_learn._util import check_input

from .base_metric import MahalanobisMixin, _PairsClassifierMixin
from .constraints import Constraints, wrap_pairs


class _BaseSDML(MahalanobisMixin):

  _t = 2  # constraints are pairs

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

  def _prepare_pairs(self, pairs, y):
    pairs, y = self.initialize_and_check_inputs(pairs, y,
                                                type_of_inputs='tuples')

    # set up prior M
    if self.use_cov:
      X = np.vstack({tuple(row) for row in pairs.reshape(-1, pairs.shape[2])})
      self.M_ = pinvh(np.cov(X, rowvar = False))
    else:
      self.M_ = np.identity(pairs.shape[2])
    diff = pairs[:, 0] - pairs[:, 1]
    return (diff.T * y).dot(diff)

  def _fit(self, pairs, y):
    loss_matrix = self._prepare_pairs(pairs, y)
    P = self.M_ + self.balance_param * loss_matrix
    emp_cov = pinvh(P)
    # hack: ensure positive semidefinite
    emp_cov = emp_cov.T.dot(emp_cov)
    _, self.M_ = graph_lasso(emp_cov, self.sparsity_param, verbose=self.verbose)

    self.transformer_ = self.transformer_from_metric(self.M_)
    return self


class SDML(_BaseSDML, _PairsClassifierMixin):
  """Sparse Distance Metric Learning (SDML)

  Attributes
  ----------
  transformer_ : `numpy.ndarray`, shape=(num_dims, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See :meth:`transformer_from_metric`.)
  """

  def fit(self, pairs, y):
    """Learn the SDML model.

    Parameters
    ----------
    pairs : array-like, shape=(n_constraints, 2, n_features) or
           (n_constraints, 2)
        3D Array of pairs with each row corresponding to two points,
        or 2D array of indices of pairs if the metric learner uses a
        preprocessor.
    y : array-like, of shape (n_constraints,)
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
      metric (See :meth:`transformer_from_metric`.)
  """

  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               num_labeled=np.inf, num_constraints=None, verbose=False,
               preprocessor=None):
    """
    Parameters
    ----------
    balance_param : float, optional
        trade off between sparsity and M0 prior
    sparsity_param : float, optional
        trade off between optimizer and sparseness (see graph_lasso)
    use_cov : bool, optional
        controls prior matrix, will use the identity if use_cov=False
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints : int, optional
        number of constraints to generate
    verbose : bool, optional
        if True, prints information while learning
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
    X, y = self.initialize_and_check_inputs(X, y)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    pairs, y = wrap_pairs(X, pos_neg)
    return _BaseSDML._fit(self, pairs, y)
