"""
Sparse Compositional Metric Learning (SCML)
"""

from __future__ import print_function, absolute_import, division
import numpy as np
from .base_metric import _TripletsClassifierMixin, MahalanobisMixin
from sklearn.base import TransformerMixin
from .constraints import generate_knntriplets
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_array


class _BaseSCML_global(MahalanobisMixin):

  _tuple_size = 3   # constraints are triplets

  def __init__(self, beta=1e-5, basis=None, n_basis=None,
               max_iter=100000, verbose=False,  preprocessor=None,
               random_state=None):
    self.beta = beta
    self.basis = basis
    self.n_basis = n_basis
    self.max_iter = max_iter
    self.verbose = verbose
    self.preprocessor = preprocessor
    self.random_state = random_state
    super(_BaseSCML_global, self).__init__(preprocessor)

  def _fit(self, triplets):

    if self.preprocessor is not None:
      n_features = self.preprocessor.shape[1]
    else:
      n_features = self.triplets.shape[1]

    self._initialize_basis(n_features)

    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO:
    # This algorithm is build to work with indeces, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following function. This should be improved
    # in the future.
    triplets, X = self._to_index_points(triplets)

    # TODO: should be given access to gamma?
    gamma = 5e-3
    dist_diff = self._compute_dist_diff(triplets, X)

    sizeT = triplets.shape[0]

    w = np.zeros((1, self.n_basis))
    avg_grad_w = np.zeros((1, self.n_basis))

    output_iter = 5000     # output every output_iter iterations

    best_obj = np.inf

    for iter in range(self.max_iter):
      if (iter % output_iter == 0):

        obj1 = np.sum(w)*self.beta

      # Every triplet distance difference in the space given by L
      # plus a slack of one
        slack_val = 1 + np.matmul(dist_diff, w.T, order='F')
      # Mask of places with positive slack
        slack_mask = slack_val > 0

        obj2 = np.sum(slack_val[slack_mask])/sizeT
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

      scale_f = -np.sqrt(iter+1) / gamma

      # proximal operator and negative trimming equivalent
      w = scale_f * np.minimum(avg_grad_w + self.beta, 0)

    if(self.verbose):
      print("max iteration reached.")

    self.components_ = self._get_components(best_w)

    return self

# should this go to utils?
  def _compute_dist_diff(self, T, X):
    XB = np.matmul(X, self.basis.T)
    T = np.vstack(T)
    lenT = len(T)
    # all positive and negative pairs with lowest index first
    # np.array (2*lenT,2)
    T_pairs_sorted = np.sort(np.vstack((T[:, [0, 1]], T[:, [0, 2]])),
                             kind='stable')
    # calculate all unique pairs
    uniqPairs, indeces = np.unique(T_pairs_sorted, return_inverse=True,
                                   axis=0)
    # calculate L2 distance acording to bases only for unique pairs
    dist = np.square(XB[uniqPairs[:, 0], :]-XB[uniqPairs[:, 1], :])

    # return the diference of distances between all positive and negative
    # pairs
    return dist[indeces[:lenT]]-dist[indeces[lenT:]]

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

    # TODO:
    # Add other options passed as string
    elif type(self.basis) is str:
      ValueError("No option for basis currently supported")


class SCML_global(_BaseSCML_global, _TripletsClassifierMixin):

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

  def __init__(self, k_genuine=3, k_impostor=10, beta=1e-5, basis=None,
               n_basis=None, max_iter=100000, verbose=False,
               preprocessor=None, random_state=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    _BaseSCML_global.__init__(self, beta=beta, basis=basis, n_basis=n_basis,
                              max_iter=max_iter, verbose=verbose,
                              preprocessor=preprocessor,
                              random_state=random_state)

  def fit(self, X, y, random_state=None):
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
      self._generate_bases_LDA(X, y, random_state)

    triplets = generate_knntriplets(X, y, self.k_genuine,
                                    self.k_impostor)

    return self._fit(triplets)

  def _generate_bases_LDA(self, X, y, random_state=None):

    labels, class_count = np.unique(y, return_counts=True)
    n_class = len(labels)

    # TODO: maybe a default value for this case?
    if(self.n_basis is None):
      raise ValueError('The number of basis given by n_basis must be set')

    # n_basis must be greater or equal to n_class
    if(self.n_basis < n_class):
        ValueError("The number of basis should be greater than the number of "
                   "classes")

    dim = np.size(X, 1)
    num_eig = min(n_class-1, dim)
    n_clusters = int(np.ceil(self.n_basis/(2 * num_eig)))

    # TODO: maybe give acces to Kmeans jobs for faster computation?
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    algorithm='elkan').fit(X)
    cX = kmeans.cluster_centers_

    # TODO: find a better way to choose neighbourhood size
    if dim > 50:
        nK = 50
    else:
        nK = 10

    nK_class = np.minimum(class_count, nK)

    idx_set = np.zeros((n_clusters, sum(nK_class)), dtype=np.int)

    start = 0
    finish = 0
    neigh = NearestNeighbors()

    for c in range(n_class):
        sel_c = np.where(y == labels[c])
        nk = nK_class[c]
        # get nK_class same class neighbours
        neigh.fit(X=X[sel_c])

        finish += nk
        idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                           n_neighbors=nk,
                                           return_distance=False))
        start = finish

    self.basis = np.zeros((self.n_basis, dim))
    for i in range(n_clusters):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])
        self.basis[num_eig*i: num_eig*(i+1), :] = normalize(lda.scalings_.T)

    nK = 20

    nK_class = np.minimum(class_count, nK)

    idx_set = np.zeros((n_clusters, sum(nK_class)), dtype=np.int)

    start = 0
    finish = 0

    for c in range(n_class):
        sel_c = np.where(y == labels[c])
        nk = nK_class[c]
        # get nK_class genuine neighbours
        neigh.fit(X=X[sel_c])

        finish += nk
        idx_set[:, start:finish] = np.take(sel_c, neigh.kneighbors(X=cX,
                                           n_neighbors=nk,
                                           return_distance=False))
        start = finish

    finish = num_eig * n_clusters
    n_components = None

    for i in range(n_clusters):
        start = finish
        finish += num_eig
        # handle tail, as n_basis != n_clusters*2*n_eig
        if (finish > self.n_basis):
          finish = self.n_basis
          n_components = finish-start

        lda = LinearDiscriminantAnalysis()
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]], n_components=n_components)

        self.basis[start:finish, :] = normalize(lda.scalings_.T)

    return
