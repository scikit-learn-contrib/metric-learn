"""
Sparse Compositional Metric Learning (SCML)
"""

from __future__ import print_function, absolute_import, division
import numpy as np
from .base_metric import _TripletsClassifierMixin, MahalanobisMixin
from sklearn.base import TransformerMixin
from .constraints import Constraints
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array, check_random_state
import warnings

# hack around lack of where in older numpy versions
try:
  np.sum([[0, 1], [1, 1]], where=[False, True], axis=1)
except TypeError:
  def sum_where(X, where):
    return np.sum(X[where])
else:
  def sum_where(X, where):
    return np.sum(X, where=where)


class _BaseSCML(MahalanobisMixin):

  _tuple_size = 3   # constraints are triplets

  def __init__(self, beta=1e-5, basis='triplet_diffs', n_basis=None,
               gamma=5e-3, max_iter=100000, output_iter=5000, verbose=False,
               preprocessor=None, random_state=None):
    self.beta = beta
    self.basis = basis
    self.n_basis = n_basis
    self.gamma = gamma
    self.max_iter = max_iter
    self.output_iter = output_iter
    self.verbose = verbose
    self.preprocessor = preprocessor
    self.random_state = random_state
    super(_BaseSCML, self).__init__(preprocessor)

  def _fit(self, triplets, basis=None, n_basis=None):
    """
    Optimization procedure to find a sparse vector of weights to
    construct the metric from the basis set. This is based on the
    dual averaging method.
    """

    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO:
    # This algorithm is built to work with indices, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following function. This should be improved
    # in the future.
    triplets, X = self._to_index_points(triplets)

    if basis is None:
      basis, n_basis = self._initialize_basis(triplets, X)

    dist_diff = self._compute_dist_diff(triplets, X, basis)

    n_triplets = triplets.shape[0]

    w = np.zeros((1, n_basis))
    avg_grad_w = np.zeros((1, n_basis))

    best_obj = np.inf

    rng = check_random_state(self.random_state)
    rand_int = rng.randint(low=0, high=n_triplets, size=self.max_iter)
    for iter in range(self.max_iter):
      if iter % self.output_iter == 0:
        # regularization part of obj function
        obj1 = np.sum(w)*self.beta

      # Every triplet distance difference in the space given by L
      # plus a slack of one
        slack_val = 1 + np.matmul(dist_diff, w.T)
      # Mask of places with positive slack
        slack_mask = slack_val > 0

        # loss function of learning task part of obj function
        obj2 = sum_where(slack_val, slack_mask)/n_triplets

        obj = obj1 + obj2
        if self.verbose:
          count = np.sum(slack_mask)
          print("[Global] iter %d\t obj %.6f\t num_imp %d" % (iter,
                obj, count))

        # update the best
        if obj < best_obj:
          best_obj = obj
          best_w = w

      # TODO:
      # Maybe allow the usage of mini-batch opt?

      idx = rand_int[iter]

      slack_val = 1 + np.matmul(dist_diff[idx, :], w.T)

      if slack_val > 0:
        avg_grad_w = (iter * avg_grad_w + dist_diff[idx, :]) / (iter+1)
      else:
        avg_grad_w = iter * avg_grad_w / (iter+1)

      scale_f = -np.sqrt(iter+1) / self.gamma

      # proximal operator with negative trimming equivalent
      w = scale_f * np.minimum(avg_grad_w + self.beta, 0)

    if self.verbose:
      print("max iteration reached.")

    # return L matrix yielded from best weights
    self.components_ = self._components_from_basis_weights(basis, best_w)

    return self

  def _compute_dist_diff(self, triplets, X, basis):
    """
    Helper function to compute the distance difference of every triplet in the
    space yielded by the basis set.
    """
    # Transformation of data by the basis set
    XB = np.matmul(X, basis.T)

    n_triplets = triplets.shape[0]
    # get all positive and negative pairs with lowest index first
    # np.array (2*n_triplets,2)
    triplets_pairs_sorted = np.sort(np.vstack((triplets[:, [0, 1]],
                                               triplets[:, [0, 2]])),
                                    kind='stable')
    # calculate all unique pairs and their indices
    uniqPairs, indices = np.unique(triplets_pairs_sorted, return_inverse=True,
                                   axis=0)
    # calculate L2 distance acording to bases only for unique pairs
    dist = np.square(XB[uniqPairs[:, 0], :] - XB[uniqPairs[:, 1], :])

    # return the diference of distances between all positive and negative
    # pairs
    return dist[indices[:n_triplets]] - dist[indices[n_triplets:]]

  def _components_from_basis_weights(self, basis, w):
    """
    Get components matrix (L) from computed mahalanobis matrix.
    """

    # get rid of inactive bases
    active_idx, = w > 0
    w = w[..., active_idx]
    basis = basis[active_idx, :]

    n_basis, n_features = basis.shape

    if n_basis < n_features:  # if metric is low-rank
      warnings.warn("The number of effective basis is less than the number of"
                    " features of the input, in consequence the learned "
                    "transformation reduces the dimension to %d." % n_basis)
      return np.sqrt(w.T)*basis  # equivalent to np.diag(np.sqrt(w)).dot(basis)

    else:   # if metric is full rank
      return np.linalg.cholesky(np.matmul(basis.T, w.T*basis)).T

  def _to_index_points(self, triplets):
    shape = triplets.shape
    X, triplets = np.unique(np.vstack(triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])
    return triplets, X

  def _initialize_basis(self, triplets, X):
    """ TODO: complete function description
    """
    n_features = X.shape[1]

    # TODO:
    # Add other options passed as string
    authorized_basis = ['triplet_diffs']
    if isinstance(self.basis, np.ndarray):
      # TODO: should copy?
      basis = check_array(self.basis, copy=True)
      if basis.shape[1] != n_features:
        raise ValueError('The dimensionality ({}) of the provided bases must'
                         ' match the dimensionality of the given inputs `X` '
                         '({}).'.format(basis.shape[1], n_features))
    elif self.basis not in authorized_basis:
      raise ValueError(
          "`basis` must be one of the options '{}' "
          "or an array of shape (n_basis, n_features)."
          .format("', '".join(authorized_basis)))
    if self.basis == 'triplet_diffs':
      basis, n_basis = self._generate_bases_dist_diff(triplets, X)

    return basis, n_basis

  def _generate_bases_dist_diff(self, triplets, X):
    """ Bases are generated from triplets as differences of positive or
    negative pairs
    TODO: complete function description
    """

    # TODO: Have a proportion of drawn pos and neg pairs?

    # get all positive and negative pairs with lowest index first
    # np.array (2*lenT,2)
    T_pairs_sorted = np.sort(np.vstack((triplets[:, [0, 1]],
                                        triplets[:, [0, 2]])),
                             kind='stable')
    # calculate all unique pairs and their indices
    uniqPairs = np.unique(T_pairs_sorted, axis=0)

    if self.n_basis is None:
      # TODO: Get a good default n_basis directive
      n_basis = uniqPairs.shape[0]
      warnings.warn('The number of basis will be set to n_basis= %d' % n_basis)

    elif isinstance(self.n_basis, int):
      n_basis = self.n_basis
    else:
      raise ValueError("n_basis should be an integer, instead it is of type %s"
                       % type(self.n_basis))

    if n_basis > uniqPairs.shape[0]:
      n_basis = uniqPairs.shape[0]
      warnings.warn("The selected number of basis is greater than the number "
                    "of points, only n_basis = %d will be generated" %
                    n_basis)

    uniqPairs = X[uniqPairs]

    rng = check_random_state(self.random_state)

    # Select n_basis
    selected_pairs = uniqPairs[rng.choice(uniqPairs.shape[0],
                               size=n_basis, replace=False), :, :]

    basis = selected_pairs[:, 0]-selected_pairs[:, 1]

    return basis, n_basis


class SCML(_BaseSCML, _TripletsClassifierMixin):
  """Sparse Compositional Metric Learning (SCML)

  `SCML` learns a metric from triplet constraints by optimizing sparse
  positive weights assigned to a set of `K` locally discriminative rank-one
  PSD bases. This can be formulated as an optimization problem with only `K`
  parameters, that can be solved with an efficient stochastic composite scheme.

  Read more in the :ref:`User Guide <scml>`.

  Parameters
  ----------
  beta: float (default=1e-5)
      L1 regularization parameter.

  basis : string or array-like, optional (default='triplet_diffs')
      Set of bases to construct the metric. Possible options are
      'triplet_diffs', and an array-like of shape (n_basis, n_features).

       'triplet_diffs'
          The basis set is constructed from the differences between points of
          `n_basis` positive or negative pairs taken from the triplets
          constrains.

       array-like
           A matrix of shape (n_basis, n_features), that will be used as
           the basis set for the metric construction.

  n_basis : int, optional
      Number of basis to be yielded. In case it is not set it will be set based
      on `basis`. If no value is selected a default will be computed based on
      the input.

  gamma: float (default = 5e-3)
      Learning rate for the optimization algorithm.

  max_iter : int (default = 100000)
      Number of iterations for the algorithm.

  output_iter : int (default = 5000)
      Number of iterations to check current weights performance and output this
      information in case verbose is True.

  verbose : bool, optional
      If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get triplets from indices. If array-like,
      triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `_components_from_basis_weights`.)

  Examples
  --------
  >>> from metric_learn import SCML
  >>> triplets = [[[1.2, 7.5], [1.3, 1.5], [6.2, 9.7]],
  >>>             [[1.3, 4.5], [3.2, 4.6], [5.4, 5.4]],
  >>>             [[3.2, 7.5], [3.3, 1.5], [8.2, 9.7]],
  >>>             [[3.3, 4.5], [5.2, 4.6], [7.4, 5.4]]]
  >>> scml = SCML()
  >>> scml.fit(triplets)

  References
  ----------
  .. [1] Y. Shi, A. Bellet and F. Sha. `Sparse Compositional Metric Learning.
         <http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf>`_. \
         (AAAI), 2014.

  .. [2] Adapted from original \
         `Matlab implementation.<https://github.com/bellet/SCML>`_.

  See Also
  --------
  metric_learn.SCML_Supervised : The supervised version of the algorithm.

  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def fit(self, triplets):
    """Learn the SCML model.

    Parameters
    ----------
    triplets : array-like, shape=(n_constraints, 3, n_features) or \
          (n_constraints, 3)
      3D array-like of triplets of points or 2D array of triplets of
      indicators. Triplets are assumed to be ordered such that:
      d(triplets[i, 0],triplets[i, 1]) < d(triplets[i, 0], triplets[i, 2]).

    Returns
    -------
    self : object
      Returns the instance.
    """

    return self._fit(triplets)


class SCML_Supervised(_BaseSCML, TransformerMixin):
  """Supervised version of Sparse Compositional Metric Learning (SCML)

  `SCML_Supervised` creates triplets by taking `k_genuine` neighbours
  of the same class and `k_impostor` neighbours from different classes for each
  point and then runs the SCML algorithm on these triplets.

  Read more in the :ref:`User Guide <scml>`.

  Parameters
  ----------
  beta: float (default=1e-5)
      L1 regularization parameter.

  basis : string or an array-like, optional (default='LDA')
      Set of bases to construct the metric. Possible options are
      'LDA', and an array-like of shape (n_basis, n_features).

      'LDA'
          The `n_basis` basis set is constructed from the LDA of significant
          local regions in the feature space via clustering, for each region
          center k-nearest neighbors are used to obtain the LDA scalings,
          which correspond to the locally discriminative basis.

       array-like
           A matrix of shape (n_basis, n_features), that will be used as
           the basis set for the metric construction.

  n_basis : int, optional
      Number of basis to be yielded. In case it is not set it will be set based
      on `basis`. If no value is selected a default will be computed based on
      the input.

  gamma: float (default = 5e-3)
      Learning rate for the optimization algorithm.

  max_iter : int (default = 100000)
      Number of iterations for the algorithm.

  output_iter : int (default = 5000)
      Number of iterations to check current weights performance and output this
      information in case verbose is True.

  verbose : bool, optional
      If True, prints information while learning.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get triplets from indices. If array-like,
      triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `_components_from_basis_weights`.)

  Examples
  --------
  >>> from metric_learn import SCML
  >>> triplets = np.array([[[1.2, 3.2], [2.3, 5.5], [2.1, 0.6]],
  >>>                      [[4.5, 2.3], [2.1, 2.3], [7.3, 3.4]]])
  >>> scml = SCML(random_state=42)
  >>> scml.fit(triplets)
  SCML(beta=1e-5, B=None, max_iter=100000, verbose=False,
      preprocessor=None, random_state=None)

  References
  ----------
  .. [1] Y. Shi, A. Bellet and F. Sha. `Sparse Compositional Metric Learning.
         <http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf>`_. \
         (AAAI), 2014.

  .. [2] Adapted from original \
         `Matlab implementation.<https://github.com/bellet/SCML>`_.

  See Also
  --------
  metric_learn.SCML_Supervised : The supervised version of this
    algorithm, which construct the triplets from the labels.

  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def __init__(self, k_genuine=3, k_impostor=10, beta=1e-5, basis='LDA',
               n_basis=None, gamma=5e-3, max_iter=100000, output_iter=5000,
               verbose=False, preprocessor=None, random_state=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    _BaseSCML.__init__(self, beta=beta, basis=basis, n_basis=n_basis,
                       max_iter=max_iter, verbose=verbose,
                       preprocessor=preprocessor,
                       random_state=random_state)

  def fit(self, X, y):
    """Create constraints from labels and learn the SCML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    Returns
    -------
    self : object
      Returns the instance.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)

    basis, n_basis = self._initialize_basis_supervised(X, y)

    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, self.k_genuine,
                                                self.k_impostor)

    triplets = X[triplets]

    return self._fit(triplets, basis, n_basis)

  def _initialize_basis_supervised(self, X, y):
    """ TODO: complete function description
    """

    # TODO:
    # Add other options passed as string
    authorized_basis = ['triplet_diffs']
    supervised_basis = ['LDA']
    authorized_basis = supervised_basis + authorized_basis

    if not(isinstance(self.basis, np.ndarray)) \
       and self.basis not in authorized_basis:
      raise ValueError(
          "`basis` must be one of the options '{}' "
          "or an array of shape (n_basis, n_features)."
          .format("', '".join(authorized_basis)))

    if self.basis == 'LDA':
      basis, n_basis = self._generate_bases_LDA(X, y)
    else:
      basis, n_basis = None, None

    return basis, n_basis

  def _generate_bases_LDA(self, X, y):
    """
    Helper function that computes the n_basis basis set constructed from the
    LDA of significant local regions in the feature space via clustering, for
    each region center k-nearest neighbors are used to obtain the LDA scalings,
    which correspond to the locally discriminative basis. Currently this is
    done at two scales `k={10,20}` if `n_feature < 50` or else `k={20,50}`.
    """

    labels, class_count = np.unique(y, return_counts=True)
    n_class = len(labels)

    n_features = X.shape[1]
    # Number of basis yielded from each LDA
    num_eig = min(n_class-1, n_features)

    if self.n_basis is None:
      # TODO: Get a good default n_basis directive
      n_basis = min(20*n_features, X.shape[0]*2*num_eig)
      warnings.warn('The number of basis will be set to n_basis= %d' % n_basis)

    elif isinstance(self.n_basis, int):
      n_basis = self.n_basis
    else:
      raise ValueError("n_basis should be an integer, instead it is of type %s"
                       % type(self.n_basis))

    if n_basis <= n_class:
      raise ValueError("The number of basis should be greater than the"
                       " number of classes")
    elif n_basis >= X.shape[0]*2*num_eig:
      raise ValueError("The selected number of basis needs a greater number of"
                       " clusters than the number of available samples")

    # Number of clusters needed for 2 scales given the number of basis
    # yielded by every LDA
    n_clusters = int(np.ceil(n_basis/(2 * num_eig)))

    # TODO: maybe give acces to Kmeans jobs for faster computation?
    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                    algorithm='elkan').fit(X)
    cX = kmeans.cluster_centers_

    # TODO: find a better way to choose neighborhood size
    if n_features > 50:
        k = 50
    else:
        k = 10

    # In case some class has less elements than k
    k_class = np.minimum(class_count, k)

    # Construct index set with neighbors for every element of every class

    idx_set = np.zeros((n_clusters, sum(k_class)), dtype=np.int)

    start_finish_indices = np.hstack((0, k_class)).cumsum()

    neigh = NearestNeighbors()

    for c in range(n_class):
        sel_c = np.where(y == labels[c])
        kc = k_class[c]

        # get k_class same class neighbours
        neigh.fit(X=X[sel_c])

        start, finish = start_finish_indices[c:c+2]
        idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                           n_neighbors=kc,
                                           return_distance=False))

    # Compute basis for every cluster in first scale
    basis = np.zeros((n_basis, n_features))
    lda = LinearDiscriminantAnalysis()
    for i in range(n_clusters):
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])
        basis[num_eig*i: num_eig*(i+1), :] = normalize(lda.scalings_.T)

    # second scale
    k = 20

    # In case some class has less elements than k
    k_class = np.minimum(class_count, k)

    # Construct index set with neighbors for every element of every class

    idx_set = np.zeros((n_clusters, sum(k_class)), dtype=np.int)

    start_finish_indices = np.hstack((0, k_class)).cumsum()

    for c in range(n_class):
      sel_c = np.where(y == labels[c])
      kc = k_class[c]

      # get k_class genuine neighbours
      neigh.fit(X=X[sel_c])

      start, finish = start_finish_indices[c:c+2]
      idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                         n_neighbors=kc,
                                         return_distance=False))

    # Compute basis for every cluster in second scale
    finish = num_eig * n_clusters

    start_finish_indices = np.arange(num_eig * n_clusters, n_basis, num_eig)
    start_finish_indices = np.append(start_finish_indices, n_basis)

    for i in range(n_clusters):
      try:
        start, finish = start_finish_indices[i:i+2]
      except ValueError:
        # No more clusters to be yielded
        break

      lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])

      basis[start:finish, :] = normalize(lda.scalings_.T[:finish-start])

    return basis, n_basis
