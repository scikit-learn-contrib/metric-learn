"""
Neighborhood Components Analysis (NCA)
"""

import warnings
import time
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from ._util import _initialize_components, _check_n_components
from .base_metric import MahalanobisMixin

EPS = np.finfo(float).eps


class NCA(MahalanobisMixin, TransformerMixin):
  """Neighborhood Components Analysis (NCA)

  NCA is a distance metric learning algorithm which aims to improve the
  accuracy of nearest neighbors classification compared to the standard
  Euclidean distance. The algorithm directly maximizes a stochastic variant
  of the leave-one-out k-nearest neighbors(KNN) score on the training set.
  It can also learn a low-dimensional linear transformation of data that can
  be used for data visualization and fast classification.

  Read more in the :ref:`User Guide <nca>`.

  Parameters
  ----------
  init : string or numpy array, optional (default='auto')
    Initialization of the linear transformation. Possible options are
    'auto', 'pca', 'identity', 'random', and a numpy array of shape
    (n_features_a, n_features_b).

    'auto'
      Depending on ``n_components``, the most reasonable initialization
      will be chosen. If ``n_components <= n_classes`` we use 'lda', as
      it uses labels information. If not, but
      ``n_components < min(n_features, n_samples)``, we use 'pca', as
      it projects data in meaningful directions (those of higher
      variance). Otherwise, we just use 'identity'.

    'pca'
      ``n_components`` principal components of the inputs passed
      to :meth:`fit` will be used to initialize the transformation.
      (See `sklearn.decomposition.PCA`)

    'lda'
      ``min(n_components, n_classes)`` most discriminative
      components of the inputs passed to :meth:`fit` will be used to
      initialize the transformation. (If ``n_components > n_classes``,
      the rest of the components will be zero.) (See
      `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

    'identity'
      If ``n_components`` is strictly smaller than the
      dimensionality of the inputs passed to :meth:`fit`, the identity
      matrix will be truncated to the first ``n_components`` rows.

    'random'
      The initial transformation will be a random array of shape
      `(n_components, n_features)`. Each value is sampled from the
      standard normal distribution.

    numpy array
      n_features_b must match the dimensionality of the inputs passed to
      :meth:`fit` and n_features_a must be less than or equal to that.
      If ``n_components`` is not None, n_features_a must match it.

  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

  max_iter : int, optional (default=100)
    Maximum number of iterations done by the optimization algorithm.

  tol : float, optional (default=None)
    Convergence tolerance for the optimization.

  verbose : bool, optional (default=False)
    Whether to print progress messages or not.

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to initialize the random
    transformation. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the transformation.

  Examples
  --------

  >>> import numpy as np
  >>> from metric_learn import NCA
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> nca = NCA(max_iter=1000)
  >>> nca.fit(X, Y)

  Attributes
  ----------
  n_iter_ : `int`
    The number of iterations the solver has run.

  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.

  References
  ----------
  .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov. `Neighbourhood
         Components Analysis
         <http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf>`_.
         NIPS 2005.

  .. [2] Wikipedia entry on `Neighborhood Components Analysis
         <https://en.wikipedia.org/wiki/Neighbourhood_components_analysis>`_
  """

  def __init__(self, init='auto', n_components=None,
               max_iter=100, tol=None, verbose=False, preprocessor=None,
               random_state=None):
    self.n_components = n_components
    self.init = init
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.random_state = random_state
    super(NCA, self).__init__(preprocessor)

  def fit(self, X, y):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    X, labels = self._prepare_inputs(X, y, ensure_min_samples=2)
    n, d = X.shape
    n_components = _check_n_components(d, self.n_components)

    # Measure the total training time
    train_time = time.time()

    # Initialize A
    A = _initialize_components(n_components, X, labels, self.init,
                               self.verbose, self.random_state)

    # Run NCA
    mask = labels[:, np.newaxis] == labels[np.newaxis, :]
    optimizer_params = {'method': 'L-BFGS-B',
                        'fun': self._loss_grad_lbfgs,
                        'args': (X, mask, -1.0),
                        'jac': True,
                        'x0': A.ravel(),
                        'options': dict(maxiter=self.max_iter),
                        'tol': self.tol
                        }

    # Call the optimizer
    self.n_iter_ = 0
    opt_result = minimize(**optimizer_params)

    self.components_ = opt_result.x.reshape(-1, X.shape[1])
    self.n_iter_ = opt_result.nit

    # Stop timer
    train_time = time.time() - train_time
    if self.verbose:
      cls_name = self.__class__.__name__

      # Warn the user if the algorithm did not converge
      if not opt_result.success:
        warnings.warn('[{}] NCA did not converge: {}'.format(
            cls_name, opt_result.message), ConvergenceWarning)

      print('[{}] Training took {:8.2f}s.'.format(cls_name, train_time))

    return self

  def _loss_grad_lbfgs(self, A, X, mask, sign=1.0):

    if self.n_iter_ == 0 and self.verbose:
      header_fields = ['Iteration', 'Objective Value', 'Time(s)']
      header_fmt = '{:>10} {:>20} {:>10}'
      header = header_fmt.format(*header_fields)
      cls_name = self.__class__.__name__
      print('[{cls}]'.format(cls=cls_name))
      print('[{cls}] {header}\n[{cls}] {sep}'.format(cls=cls_name,
                                                     header=header,
                                                     sep='-' * len(header)))

    start_time = time.time()

    A = A.reshape(-1, X.shape[1])
    X_embedded = np.dot(X, A.T)  # (n_samples, n_components)
    # Compute softmax distances
    p_ij = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(p_ij, np.inf)
    p_ij = np.exp(-p_ij - logsumexp(-p_ij, axis=1)[:, np.newaxis])
    # (n_samples, n_samples)

    # Compute loss
    masked_p_ij = p_ij * mask
    p = masked_p_ij.sum(axis=1, keepdims=True)  # (n_samples, 1)
    loss = p.sum()

    # Compute gradient of loss w.r.t. `transform`
    weighted_p_ij = masked_p_ij - p_ij * p
    weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
    np.fill_diagonal(weighted_p_ij_sym, - weighted_p_ij.sum(axis=0))
    gradient = 2 * (X_embedded.T.dot(weighted_p_ij_sym)).dot(X)

    if self.verbose:
        start_time = time.time() - start_time
        values_fmt = '[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}'
        print(values_fmt.format(cls=self.__class__.__name__,
                                n_iter=self.n_iter_, loss=loss,
                                start_time=start_time))
        sys.stdout.flush()

    self.n_iter_ += 1
    return sign * loss, sign * gradient.ravel()
