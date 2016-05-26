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
from random import choice
from scipy.sparse.csgraph import laplacian
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh
from .base_metric import BaseMetricLearner


class SDML(BaseMetricLearner):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True, num_constraints=None, verbose=False):
    '''
    balance_param: trade off between sparsity and M0 prior
    sparsity_param: trade off between optimizer and sparseness (see graph_lasso)
    use_cov: controls prior matrix, will use the identity if use_cov=False
    num_constraints: int, needed for .fit()
    verbose: bool
    '''
    self.params = {
      'balance_param': balance_param,
      'sparsity_param': sparsity_param,
      'use_cov': use_cov,
      'num_constraints': num_constraints,
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

  def fit(self, X, labels):
    """Create constraints from labels and learn the SDML model.
    Needs num_constraints specified in constructor.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    labels : (n) data labels
    """
    num_constraints = self.params['num_constraints']
    if num_constraints is None:
      raise ValueError('You need to specify `num_constraints` before using .fit()')

    W = self.prepare_constraints(labels, X.shape[0], num_constraints)
    return self.fit_constraints(X, W, verbose=self.params['verbose'])

  def fit_constraints(self, X, W, verbose=False):
    """
    X: data matrix, (n x d)
    W: connectivity graph, (n x n). +1 for positive pairs, -1 for negative.
    """
    self._prepare_inputs(X, W)
    P = pinvh(self.M) + self.params['balance_param'] * self.loss_matrix
    emp_cov = pinvh(P)
    # hack: ensure positive semidefinite
    emp_cov = emp_cov.T.dot(emp_cov)
    self.M, _ = graph_lasso(emp_cov, self.params['sparsity_param'],
                            verbose=verbose)
    return self

  @classmethod
  def prepare_constraints(self, labels, num_points, num_constraints):
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    b, d = np.empty((2, num_constraints), dtype=int)
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      b[i] = choice(np.nonzero(labels == al)[0])
      d[i] = choice(np.nonzero(labels != cl)[0])
    W = np.zeros((num_points,num_points))
    W[a,b] = 1
    W[c,d] = -1
    # make W symmetric
    W[b,a] = 1
    W[d,c] = -1
    return W
