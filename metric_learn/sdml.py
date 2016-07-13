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
from scipy.sparse.csgraph import laplacian
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh

from .base_metric import BaseMetricLearner
from .constraints import Constraints


class SDML(BaseMetricLearner):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               verbose=False):
    '''
    balance_param: float, optional
        trade off between sparsity and M0 prior
    sparsity_param: float, optional
        trade off between optimizer and sparseness (see graph_lasso)
    use_cov: bool, optional
        controls prior matrix, will use the identity if use_cov=False
    verbose : bool, optional
        if True, prints information while learning
    '''
    self.params = {
      'balance_param': balance_param,
      'sparsity_param': sparsity_param,
      'use_cov': use_cov,
      'verbose': verbose,
    }

  def _prepare_inputs(self, X, W):
    self.X = X
    # set up prior M
    if self.params['use_cov']:
      self.M = np.cov(X.T)
    else:
      self.M = np.identity(X.shape[1])
    L = laplacian(W, normed=False)
    self.loss_matrix = self.X.T.dot(L.dot(self.X))

  def metric(self):
    return self.M

  def fit(self, X, W):
    """
    X: data matrix, (n x d)
        each row corresponds to a single instance
    W: connectivity graph, (n x n)
        +1 for positive pairs, -1 for negative.
    """
    self._prepare_inputs(X, W)
    P = pinvh(self.M) + self.params['balance_param'] * self.loss_matrix
    emp_cov = pinvh(P)
    # hack: ensure positive semidefinite
    emp_cov = emp_cov.T.dot(emp_cov)
    self.M, _ = graph_lasso(emp_cov, self.params['sparsity_param'],
                            verbose=self.params['verbose'])
    return self


class SDML_Supervised(SDML):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True,
               num_labeled=np.inf, num_constraints=None, verbose=False):
    SDML.__init__(self, balance_param=balance_param,
                  sparsity_param=sparsity_param, use_cov=use_cov,
                  verbose=verbose)
    '''
    balance_param: float, optional
        trade off between sparsity and M0 prior
    sparsity_param: float, optional
        trade off between optimizer and sparseness (see graph_lasso)
    use_cov: bool, optional
        controls prior matrix, will use the identity if use_cov=False
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    verbose : bool, optional
        if True, prints information while learning
    '''
    self.params.update(num_labeled=num_labeled, num_constraints=num_constraints)

  def fit(self, X, labels):
    """Create constraints from labels and learn the SDML model.

    Parameters
    ----------
    X: data matrix, (n x d)
        each row corresponds to a single instance
    labels: data labels, (n,) array-like
    """
    num_constraints = self.params['num_constraints']
    if num_constraints is None:
      num_classes = np.unique(labels)
      num_constraints = 20*(len(num_classes))**2

    c = Constraints.random_subset(labels, self.params['num_labeled'])
    return SDML.fit(self, X, c.adjacency_matrix(num_constraints))
