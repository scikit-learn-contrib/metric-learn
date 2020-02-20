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
from sklearn.utils import check_array


class _BaseSCML_global(MahalanobisMixin):

  _tuple_size = 3   # constraints are triplets

  def __init__(self, beta=1e-5, basis=None, n_basis=None, gamma=5e-3,
               max_iter=100000, output_iter=5000, verbose=False,
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
    super(_BaseSCML_global, self).__init__(preprocessor)

  def _fit(self, triplets):
    """
    Optimization procedure to find a sparse vector of weights to
    construct the metric from the basis set. This is based on the
    dual averaging method.
    """

    if self.preprocessor is not None:
      n_features = self.preprocessor.shape[1]
    else:
      n_features = self.triplets.shape[1]

    self._initialize_basis(n_features)

    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO:
    # This algorithm is built to work with indices, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following function. This should be improved
    # in the future.
    triplets, X = self._to_index_points(triplets)

    dist_diff = self._compute_dist_diff(triplets, X)

    sizeT = triplets.shape[0]

    w = np.zeros((1, self.n_basis))
    avg_grad_w = np.zeros((1, self.n_basis))

    best_obj = np.inf

    for iter in range(self.max_iter):
      if (iter % self.output_iter == 0):
        # regularization part of obj function
        obj1 = np.sum(w)*self.beta

      # Every triplet distance difference in the space given by L
      # plus a slack of one
        slack_val = 1 + np.matmul(dist_diff, w.T, order='F')
      # Mask of places with positive slack
        slack_mask = slack_val > 0

        # loss function of learning task part of obj function
        obj2 = np.sum(slack_val, where=slack_mask)/sizeT

        obj = obj1 + obj2
        if(self.verbose):
          count = np.sum(slack_mask)
          print("[Global] iter %d\t obj %.6f\t num_imp %d" % (iter,
                obj, count))

        # update the best
        if (obj < best_obj):
          best_obj = obj
          best_w = w

      # TODO:
      # Maybe allow the usage of mini-batch opt?

      idx = np.random.randint(low=0, high=sizeT)

      slack_val = 1 + dist_diff[idx, :].dot(w.T)

      if (slack_val > 0):
        avg_grad_w = (iter * avg_grad_w + dist_diff[idx, :]) / (iter+1)
      else:
        avg_grad_w = iter * avg_grad_w / (iter+1)

      scale_f = -np.sqrt(iter+1) / self.gamma

      # proximal operator with negative trimming equivalent
      w = scale_f * np.minimum(avg_grad_w + self.beta, 0)

    if(self.verbose):
      print("max iteration reached.")

    # return L matrix yielded from best weights
    self.components_ = self._get_components(best_w)

    return self

  def _compute_dist_diff(self, T, X):
    """
    Helper function to compute the distance difference of every triplet in the
    space yielded by the basis set.
    """
    # Transformation of data by the basis set
    XB = np.matmul(X, self.basis.T)

    lenT = len(T)
    # get all positive and negative pairs with lowest index first
    # np.array (2*lenT,2)
    T_pairs_sorted = np.sort(np.vstack((T[:, [0, 1]], T[:, [0, 2]])),
                             kind='stable')
    # calculate all unique pairs and their indices
    uniqPairs, indices = np.unique(T_pairs_sorted, return_inverse=True,
                                   axis=0)
    # calculate L2 distance acording to bases only for unique pairs
    dist = np.square(XB[uniqPairs[:, 0], :]-XB[uniqPairs[:, 1], :])

    # return the diference of distances between all positive and negative
    # pairs
    return dist[indices[:lenT]]-dist[indices[lenT:]]

  def _get_components(self, w):
    """
    get components matrix (L) from computed mahalanobis matrix
    """

    # get rid of inactive bases
    active_idx = w > 0
    w = w[active_idx]
    basis = self.basis[np.squeeze(active_idx), :]

    K, d = basis.shape

    if(K < d):  # if metric is low-rank
      return basis*np.sqrt(w)[..., None]

    else:   # if metric is full rank
      return np.linalg.cholesky(np.matmul(basis.T * w, basis, order='F')).T

  def _to_index_points(self, triplets):
    shape = triplets.shape
    X, triplets = np.unique(np.vstack(triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])
    return triplets, X

  def _initialize_basis(self, n_features):
    # TODO:
    # Add other options passed as string
    authorized_basis = []
    if isinstance(self.basis, np.ndarray):
      self.basis = check_array(self.basis)
      self.n_basis = self.basis.shape[0]
      if self.basis.shape[1] != n_features:
        raise ValueError('The input dimensionality ({}) of the given '
                         'linear transformation `init` must match the '
                         'dimensionality of the given inputs `X` ({}).'
                         .format(self.basis.shape[1], n_features))
    elif self.basis not in authorized_basis:
      raise ValueError(
          "`basis` must be '{}' "
          "or a numpy array of shape (n_basis, n_features)."
          .format("', '".join(authorized_basis)))


class SCML_global(_BaseSCML_global, _TripletsClassifierMixin):
  """Sparse Compositional Metric Learning (SCML)

  `SCML` builds a metric as the sparse positive combination of a set of locally
  discriminative rank-one PSD basis. This allows an optimization scheme with
  only `K` parameters, that can be yielded with an efficient stochastic
  composite optimization over a set of triplets constraints. Each triplet is
  constructed as a relative distance comparison with respect to the first
  element so that the second element is closer than the last.
  Read more in the :ref:`User Guide <scml>`.
  Parameters
  ----------
  beta: float (default=1e-5)
      L1 regularization parameter.
  basis : None, string or numpy array, optional (default=None)
       Prior to set for the metric. Possible options are
       '', and a numpy array of shape (n_basis, n_features). If
        None an error will be raised as the basis set is esential
        to SCML.
       numpy array
           A matrix of shape (n_basis, n_features), that will be used as
           the basis set for the metric construction.
  n_basis : int, optional
      Number of basis to be yielded. In case it is not set it will be set based
      on the basis numpy array. If an string option is pased to basis an error
      wild be raised as this value will be needed.
  gamma: float (default = 5e-3)
      Learning rate
  max_iter : int (default = 100000)
      Number of iterations for the algorithm
  output_iter : int (default = 5000)
      Number of iterations to check current weights performance and output this
      information in case verbose is True.
  verbose : bool, optional
      if True, prints information while learning
  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get triplets from indices. If array-like,
      triplets will be formed like this: X[indices].
  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.
  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `components_from_metric`.)
  Examples
  --------
  >>> from metric_learn import SCLM_global_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> scml = SCML_global_Supervised(basis='LDA', n_basis=400)
  >>> scml.fit(X, Y)
  References
  ----------
  .. [1] Y. Shi, A. Bellet and F. Sha. `Sparse Compositional Metric Learning.
         <http://researchers.lille.inria.fr/abellet/papers/aaai14.pdf>`_. \
         (AAAI), 2014.
  .. [2] Adapted from original \
         `Matlab implementation.<https://github.com/bellet/SCML>`_.
  See Also
  --------
  metric_learn.SCML_global : The original weakly-supervised algorithm.

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
      indicators. In order to supervise the algorithm in the right way, we
      should have the three samples ordered in a way such that:
      d(triplets[i, 0],triplets[i, 1]) < d(triplets[i, 1], triplets[i, 3])
      for all 0 <= i < n_constraints.

    Returns
    -------
    self : object
      Returns the instance.
    """
    return _BaseSCML_global._fit(triplets)


class SCML_global_Supervised(_BaseSCML_global, TransformerMixin):
  """Supervised version of Sparse Compositional Metric Learning (SCML)

  `SCML_global_Supervised` creates triplets by taking `k_genuine` neighbours
  of the same class and `k_impostor` neighbours from diferent classes for each
  point and then runs the SCML algorithm on these triplets.
  Read more in the :ref:`User Guide <scml>`.
  Parameters
  ----------
  beta: float (default=1e-5)
      L1 regularization parameter.
  basis : None, string or numpy array, optional (default=None)
      Prior to set for the metric. Possible options are
      'LDA', and a numpy array of shape (n_basis, n_features). If
      None an error will be raised as the basis set is esential
      to SCML.
      'LDA'
          The `n_basis` basis set is constructed from the LDA of significant
          local regions in the feature space via clustering, for each region
          center k-nearest neighbors are used to obtain the LDA scalings,
          which correspond to the locally discriminative basis.
       numpy array
           A matrix of shape (n_basis, n_features), that will be used as
           the basis set for the metric construction.
  n_basis : int, optional
      Number of basis to be yielded. In case it is not set it will be set based
      on the basis numpy array. If an string option is pased to basis an error
      wild be raised as this value will be needed.
  gamma: float (default = 5e-3)
      Learning rate
  max_iter : int (default = 100000)
      Number of iterations for the algorithm
  output_iter : int (default = 5000)
      Number of iterations to check current weights performance and output this
      information in case verbose is True.
  verbose : bool, optional
      if True, prints information while learning
  preprocessor : array-like, shape=(n_samples, n_features) or callable
      The preprocessor to call to get triplets from indices. If array-like,
      triplets will be formed like this: X[indices].
  random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.
  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
      The linear transformation ``L`` deduced from the learned Mahalanobis
      metric (See function `components_from_metric`.)
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
  metric_learn.SCML_global_Supervised : The supervised version of this
    algorithm, which construct the triplets from the labels.
  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def __init__(self, k_genuine=3, k_impostor=10, beta=1e-5, basis=None,
               n_basis=None, gamma=5e-3, max_iter=100000, output_iter=5000,
               verbose=False, preprocessor=None, random_state=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    _BaseSCML_global.__init__(self, beta=beta, basis=basis, n_basis=n_basis,
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
    self.preprocessor = X

    # TODO:
    # it can be a problem if fit is called more than once,
    # should that case be handled?

    if(self.basis == "LDA"):
      self._generate_bases_LDA(X, y)

    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, self.k_genuine,
                                                self.k_impostor)

    return self._fit(triplets)

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

    # TODO: maybe a default value for this case?
    if(self.n_basis is None):
      raise ValueError('The number of basis given by n_basis must be set')

    # n_basis must be greater or equal to n_class
    if(self.n_basis < n_class):
        ValueError("The number of basis should be greater than the number of "
                   "classes")

    n_features = np.size(X, 1)
    # Number of basis yielded from each LDA
    num_eig = min(n_class-1, n_features)
    # Number of clusters needed for 2 scales given the number of basis
    # yielded by every LDA
    n_clusters = int(np.ceil(self.n_basis/(2 * num_eig)))

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

    start = 0
    finish = 0
    neigh = NearestNeighbors()

    for c in range(n_class):
        sel_c = np.where(y == labels[c])
        kc = k_class[c]
        # get k_class same class neighbours
        neigh.fit(X=X[sel_c])

        finish += kc
        idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                           n_neighbors=kc,
                                           return_distance=False))
        start = finish

    # Compute basis for every cluster in first scale
    self.basis = np.zeros((self.n_basis, n_features))
    lda = LinearDiscriminantAnalysis()
    for i in range(n_clusters):
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])
        self.basis[num_eig*i: num_eig*(i+1), :] = normalize(lda.scalings_.T)

    # second scale
    k = 20

    # In case some class has less elements than k
    k_class = np.minimum(class_count, k)

    # Construct index set with neighbors for every element of every class

    idx_set = np.zeros((n_clusters, sum(k_class)), dtype=np.int)

    start = 0
    finish = 0

    for c in range(n_class):
        sel_c = np.where(y == labels[c])
        kc = k_class[c]

        # get k_class genuine neighbours
        neigh.fit(X=X[sel_c])
        finish += kc
        idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                           n_neighbors=kc,
                                           return_distance=False))
        start = finish

    # Compute basis for every cluster in second scale
    finish = num_eig * n_clusters
    lda = LinearDiscriminantAnalysis()
    for i in range(n_clusters):
        start = finish
        finish += num_eig

        # handle tail, as n_basis != n_clusters*2*n_eig
        if (finish > self.n_basis):
          finish = self.n_basis

        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])

        self.basis[start:finish, :] = normalize(lda.scalings_.T[:finish-start])

    return
