"""
Large Margin Nearest Neighbor Metric learning (LMNN)
"""
import sys
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances
from sklearn.base import TransformerMixin
from sklearn.exceptions import ConvergenceWarning

from ._util import _initialize_components, _check_n_components
from .base_metric import MahalanobisMixin


class LMNN(MahalanobisMixin, TransformerMixin):
  """Large Margin Nearest Neighbor (LMNN)

  LMNN learns a Mahalanobis distance metric in the kNN classification
  setting. The learned metric attempts to keep close k-nearest neighbors
  from the same class, while keeping examples from different classes
  separated by a large margin. This algorithm makes no assumptions about
  the distribution of the data.

  Read more in the :ref:`User Guide <lmnn>`.

  Parameters
  ----------
  n_neighbors : int, optional (default=3)
    Number of neighbors to consider, not including self-edges.

  n_components : int or None, optional (default=None)
    Dimensionality of reduced space (if None, defaults to dimension of X).

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

  neighbors_params : dict, optional (default=None)
    Parameters to pass to a :class:`neighbors.NearestNeighbors` instance -
    apart from ``n_neighbors`` - that will be used to select the target
    neighbors.

  push_loss_weight: float, optional (default=0.5)
    Relative weight between pull and push terms, with 0.5 meaning equal
    weight.

  max_iter : int, optional (default=1000)
    Maximum number of iterations of the optimization procedure.

  tol : float, optional (default=0.00001)
    Tolerance of the optimization procedure. If the objective value varies
    less than `tol`, we consider the algorithm has converged and stop it.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get tuples from indices. If array-like,
    tuples will be formed like this: X[indices].

  verbose : bool, optional (default=False)
    Whether to print the progress of the optimization procedure.

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to initialize the random
    transformation. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the transformation.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_components, n_features)
    The learned linear transformation ``L``.

  n_neighbors_ : int
      The provided ``n_neighbors`` is decreased if it is greater than or
      equal to  min(number of elements in each class).

  n_iter_ : `int`
    The number of iterations the solver has run.


  Examples
  --------

  >>> import numpy as np
  >>> from metric_learn import LMNN
  >>> from sklearn.neighbors import KNeighborsClassifier
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> lmnn = LMNN(n_neighbors=3)
  >>> lmnn.fit(X, Y)
  >>> knn = KNeighborsClassifier(n_neighbors=3)
  >>> knn.fit(lmnn.transform(X), Y)

  References
  ----------
  .. [1] Weinberger, Kilian Q., and Lawrence K. Saul.
         "Distance Metric Learning for Large Margin Nearest Neighbor
         Classification."
         Journal of Machine Learning Research, Vol. 10, Feb. 2009,
         pp. 207-244.
         http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf

  .. [2] Wikipedia entry on Large Margin Nearest Neighbor
         https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor

  """
  def __init__(self, n_neighbors=3, n_components=None, init='auto',
               neighbors_params=None, push_loss_weight=0.5, max_iter=50,
               tol=1e-5, preprocessor=None, verbose=False, random_state=None):

    self.n_neighbors = n_neighbors
    self.n_components = n_components
    self.init = init
    self.neighbors_params = neighbors_params
    self.push_loss_weight = push_loss_weight
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.random_state = random_state
    super(LMNN, self).__init__(preprocessor)

  def fit(self, X, y):
    """ Train Large Margin Nearest Neighbor model.

    :param X: array, shape (n_samples, n_features), the training samples.
    :param y: array, shape (n_samples,), the training labels.

    :return: The trained LMNN model.
    """

    # Check input arrays
    X, y = self._prepare_inputs(X, y, dtype=float, ensure_min_samples=2)

    n_samples, n_features = X.shape
    n_components = _check_n_components(n_features, self.n_components)

    # Initialize transformation
    self.components_ = _initialize_components(n_components, X, y, self.init,
                                              verbose=self.verbose,
                                              random_state=self.random_state)

    # remove singletons after initializing components
    X, y, classes = self._validate_params(X, y)

    # Find the target neighbors of each sample
    target_neighbors = self._select_target_neighbors(X, y, classes)

    # Compute the gradient w.r.t. the target neighbors which remains constant
    tn_graph = _make_knn_graph(target_neighbors)
    const_grad = _sum_weighted_outer_differences(X, tn_graph)

    # Compute the pull loss weight such that the push loss weight becomes 1
    pull_loss_weight = (1 - self.push_loss_weight) / self.push_loss_weight
    const_grad *= pull_loss_weight

    int_verbose = int(self.verbose)
    disp = int_verbose - 2 if int_verbose > 1 else -1
    optimizer_params = {
      'method': 'L-BFGS-B',
      'fun': self._loss_grad_lbfgs,
      'jac': True,
      'args': (X, y, classes, target_neighbors, const_grad),
      'x0': self.components_,
      'tol': self.tol,
      'options': dict(maxiter=self.max_iter, disp=disp),
    }

    # Call the optimizer
    self.n_iter_ = 0
    opt_result = minimize(**optimizer_params)

    # Reshape the solution found by the optimizer
    self.components_ = opt_result.x.reshape(-1, n_features)

    if self.verbose:
      cls_name = self.__class__.__name__

      # Warn the user if the algorithm did not converge
      if not opt_result.success:
        warnings.warn('[{}] LMNN did not converge: {}'.format(
          cls_name, opt_result.message), ConvergenceWarning)

    return self

  def _validate_params(self, X, y):
    """ Validate parameters as soon as :meth:`fit` is called.

    :param X: array, shape (n_samples, n_features), the training samples.
    :param y: array, shape (n_samples,), the training labels.

    :return:
      X: array, shape (n_samples, n_features), the validated training samples.
      y: array, shape (n_samples,), the validated training labels.
      classes: array, shape (n_classes,), the non-singleton classes encoded.
    """

    # Find the appearing classes and the class index of each of the samples
    classes, y = np.unique(y, return_inverse=True)
    classes_inverse = np.arange(len(classes))

    # Ignore classes that have less than two samples (singleton classes)
    class_sizes = np.bincount(y)
    mask_singleton_class = class_sizes == 1
    singleton_classes = np.where(mask_singleton_class)[0]
    if len(singleton_classes):
      warnings.warn('There are {} singleton classes that will be ignored '
                    'during training. A copy of the inputs `X` and `y` will '
                    'be made.'.format(len(singleton_classes)))

      mask_singleton_sample = np.asarray([yi in singleton_classes for
                                          yi in y])
      X = X[~mask_singleton_sample]
      y = y[~mask_singleton_sample]

    # Check that there are at least 2 non-singleton classes
    n_classes_non_singleton = len(classes) - len(singleton_classes)
    if n_classes_non_singleton < 2:
      raise ValueError('LMNN needs at least 2 non-singleton classes, got {}.'
                       .format(n_classes_non_singleton))

    classes_inverse_non_singleton = classes_inverse[~mask_singleton_class]

    # Check the preferred number of neighbors
    min_non_singleton_size = class_sizes[~mask_singleton_class].min()
    if self.n_neighbors >= min_non_singleton_size:
      warnings.warn('`n_neighbors` (={}) is not less than the number of '
           'samples in the smallest non-singleton class (={}). '
           '`n_neighbors_` will be set to {} for estimation.'
           .format(self.n_neighbors, min_non_singleton_size,
                   min_non_singleton_size - 1))

    self.n_neighbors_ = min(self.n_neighbors, min_non_singleton_size - 1)

    return X, y, classes_inverse_non_singleton

  def _loss_grad_lbfgs(self, L, X, y, classes, target_neighbors, const_grad):
    """ Compute loss and gradient after one optimization iteration.

    :param L: array, shape (n_components * n_features,)
              The current (flattened) linear transformation.
    :param X: array, shape (n_samples, n_features), the training samples.
    :param y: array, shape (n_samples,), the training labels.
    :param classes: array, shape (n_classes,), classes encoded as intergers.
    :param target_neighbors: array, shape (n_samples, n_neighbors),
                             the target neighbors of each training sample.
    :param const_grad: array, shape (n_features, n_features),
                       the (weighted) gradient component due to target
                       neighbors that stays fixed throughout the algorithm.

    :return:
      loss: float, the loss given the current solution.
      grad: array, shape (n_components, n_features), the (flattened) gradient.
    """

    # Print headers
    if self.n_iter_ == 0 and self.verbose:
      header_fields = ['Iteration', 'Objective Value',
                       '#Active Triplets', 'Time(s)']
      header_fmt = '{:>10} {:>20} {:>20} {:>10}'
      header = header_fmt.format(*header_fields)
      cls_name = self.__class__.__name__
      print('[{}]'.format(cls_name))
      print('[{}] {}'.format(cls_name, header))
      print('[{}] {}'.format(cls_name, '-' * len(header)))

    # Measure duration of optimization iteration
    start_time = time.time()

    # Reshape current solution
    n_samples, n_features = X.shape
    L = L.reshape(-1, n_features)

    # Transform samples according to the current solution
    X_embedded = X @ L.T

    # Compute squared distances to the target neighbors
    n_neighbors = target_neighbors.shape[1]
    dist_tn = np.zeros((n_samples, n_neighbors))
    for k in range(n_neighbors):
      ind_tn = target_neighbors[:, k]
      diffs = X_embedded - X_embedded[ind_tn]
      dist_tn[:, k] = np.einsum('ij,ij->i', diffs, diffs)  # row norms

    # Add margin to the distances
    dist_tn += 1

    # Find impostors
    impostors_graph = _find_impostors(X_embedded, y, classes, dist_tn[:, -1])

    # Compute the push loss and its gradient
    push_loss, push_loss_grad, n_active = _compute_push_loss(
      X, target_neighbors, dist_tn, impostors_graph)

    # Compute the total loss
    M = L.T @ L
    new_pull_loss = np.dot(const_grad.ravel(), M.ravel())
    loss = push_loss + new_pull_loss

    # Compute the total gradient
    grad = np.dot(L, const_grad + push_loss_grad)
    grad *= 2

    # Report iteration metrics
    self.n_iter_ += 1
    if self.verbose:
      elapsed_time = time.time() - start_time
      values_fmt = '[{}] {:>10} {:>20.6e} {:>20,} {:>10.2f}'
      cls_name = self.__class__.__name__
      print(values_fmt.format(cls_name, self.n_iter_, loss, n_active,
                              elapsed_time))
      sys.stdout.flush()

    return loss, grad.ravel()

  def _select_target_neighbors(self, X, y, classes):
    """ Find the target neighbors of each training sample.

    :param X: array, shape (n_samples, n_features), the training samples.
    :param y: array, shape (n_samples,), the training labels.
    :param classes: array, shape (n_classes,), the classes encoded as integers.

    :return: array, shape (n_samples, n_neighbors),
             the indices of the target neighbors of each training sample.
    """

    start_time = time.time()
    cls_name = self.__class__.__name__
    if self.verbose:
      print('[{}] Finding the target neighbors...'.format(cls_name))
      sys.stdout.flush()

    nn_kwargs = self.neighbors_params or {}
    nn = NearestNeighbors(n_neighbors=self.n_neighbors_, **nn_kwargs)
    target_neighbors = np.empty((X.shape[0], self.n_neighbors_), dtype=int)

    for label in classes:
      ind_class = np.where(y == label)[0]
      nn.fit(X[ind_class])
      ind_neighbors = nn.kneighbors(return_distance=False)
      target_neighbors[ind_class] = ind_class[ind_neighbors]

    if self.verbose:
      elapsed_time = time.time() - start_time
      print('[{}] Found the target neighbors in {:5.2f}s.'.format(
        cls_name, elapsed_time))

    return target_neighbors


def _find_impostors(X_embedded, y, classes, margin_radii):
  """ Find the samples that violate the margin.

  :param X_embedded: array, shape (n_samples, n_components),
                     the training samples in the embedding space.
  :param y: array, shape (n_samples,), training labels.
  :param classes: array, shape (n_classes,), the classes encoded as integers.
  :param margin_radii: array, shape (n_samples,), squared distances of
                       training samples to their farthest target neighbors
                       plus margin.

  :return: coo_matrix, shape (n_samples, n_neighbors),
           If sample i is an impostor to sample j or vice versa, then the
           value of A[i, j] is the squared distance between samples i and j.
           Otherwise A[i, j] is zero.

  """

  # Initialize lists for impostors storage
  imp_row, imp_col, imp_dist = [], [], []

  for label in classes[:-1]:
    ind_a = np.where(y == label)[0]
    ind_b = np.where(y > label)[0]

    dist = euclidean_distances(X_embedded[ind_b], X_embedded[ind_a],
                               squared=True)  # (b, a)

    radii_a = margin_radii[ind_a]  # (a,)
    radii_b = margin_radii[ind_b]  # (b,)

    # Find samples in B that are impostors to A
    imp_ba = np.where((dist < radii_a[None, :]).ravel())[0]

    # Find samples in A that are impostors to B
    imp_ab = np.where((dist < radii_b[:, None]).ravel())[0]

    # merge and filter unique impostors
    ind_impostors = np.unique(np.concatenate((imp_ba, imp_ab)))

    if len(ind_impostors):
      # Map indices back to the original training data indices
      ii, jj = np.unravel_index(ind_impostors, dist.shape)
      imp_row.extend(ind_b[ii])
      imp_col.extend(ind_a[jj])
      imp_dist.extend(dist.ravel()[ind_impostors])

  # turn lists to numpy arrays
  imp_row = np.asarray(imp_row, dtype=np.intp)
  imp_col = np.asarray(imp_col, dtype=np.intp)
  imp_dist = np.asarray(imp_dist)

  # store impostors as a sparse matrix
  n = X_embedded.shape[0]
  impostors_graph = coo_matrix((imp_dist, (imp_row, imp_col)), shape=(n, n))

  return impostors_graph


def _compute_push_loss(X, target_neighbors, inflated_dist_tn, impostors_graph):
  """ Compute the push loss L_push = max(d(a, p) + 1 - d(a, n), 0)

  :param X: array, shape (n_samples, n_features), the training samples.
  :param target_neighbors: array, shape (n_samples, n_neighbors),
                           the target neighbors of each training sample.
  :param inflated_dist_tn: array, shape (n_samples, n_neighbors),
                           squared distances of each sample to their target
                           neighbors plus margin.
  :param impostors_graph: coo_matrix, shape (n_samples, n_samples),

  :return:
    loss: float, the push loss due to the given target neighbors and impostors.
    grad: array, shape (n_features, n_features), the gradient of the push loss.
    n_active_triplets: int, the number of active triplet constraints.

  """

  n_samples, n_neighbors = inflated_dist_tn.shape
  imp_row = impostors_graph.row
  imp_col = impostors_graph.col
  dist_impostors = impostors_graph.data

  loss = 0
  shape = (n_samples, n_samples)
  A0 = csr_matrix(shape)
  sample_range = range(n_samples)
  n_active_triplets = 0
  for k in reversed(range(n_neighbors)):
    # Consider margin violations to the samples in imp_row
    losses1 = np.maximum(inflated_dist_tn[imp_row, k] - dist_impostors, 0)
    ac = np.where(losses1 > 0)[0]
    n_active_triplets += len(ac)
    A1 = csr_matrix((2 * losses1[ac], (imp_row[ac], imp_col[ac])), shape)

    # Consider margin violations to the samples in imp_col
    losses2 = np.maximum(inflated_dist_tn[imp_col, k] - dist_impostors, 0)
    ac = np.where(losses2 > 0)[0]
    n_active_triplets += len(ac)
    A2 = csc_matrix((2 * losses2[ac], (imp_row[ac], imp_col[ac])), shape)

    # Update the loss
    loss += np.dot(losses1, losses1) + np.dot(losses2, losses2)

    # Update the weight matrix for gradient computation
    val = (A1.sum(1).ravel() + A2.sum(0)).getA1()
    A3 = csr_matrix((val, (sample_range, target_neighbors[:, k])), shape)
    A0 = A0 - A1 - A2 + A3

  grad = _sum_weighted_outer_differences(X, A0)

  return loss, grad, n_active_triplets


def _sum_weighted_outer_differences(X, weights):
  """ Compute the sum of weighted outer pairwise differences.

  :param X: array, shape (n_samples, n_features), data samples.
  :param weights: csr_matrix, shape (n_samples, n_samples), sparse weights.

  :return: array, shape (n_features, n_features),
           the sum of all outer weighted differences.
  """

  W = weights + weights.T
  D = W.sum(1).getA()

  # X.T * (D - W) * X
  LX = D * X - W @ X  # D is n x 1, W is n x n
  ret = X.T @ LX

  return ret


def _make_knn_graph(indices):
  """ Convert a dense indices matrix to a sparse adjacency matrix.

  :param indices: array, shape (n_samples, n_neighbors),
                  indices of the k nearest neighbors of each sample.
  :return: csr_matrix, shape (n_samples, n_samples), the adjacency matrix.
  """

  n_samples, n_neighbors = indices.shape
  row = np.repeat(range(n_samples), n_neighbors)
  col = indices.ravel()
  ind = np.ones(indices.size)
  shape = (n_samples, n_samples)
  knn_graph = csr_matrix((ind, (row, col)), shape=shape)

  return knn_graph
