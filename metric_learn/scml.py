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


class _BaseSCML_global(MahalanobisMixin):

  _tuple_size = 3   # constraints are triplets

  def __init__(self, beta=1e-5, B=None,
               max_iter=100000, verbose=False,
               preprocessor=None, random_state=None):
    self.beta = beta
    self.max_iter = max_iter
    self.verbose = verbose
    self.preprocessor = preprocessor
    self.random_state = random_state
    super(_BaseSCML_global, self).__init__(preprocessor)

  def _fit(self, triplets, B, n_basis):

    # TODO: manage B
    # if B is None
    #   error
    # if B is array
    #   pass
    # if option
    #   do something

    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    triplets, X = self._to_index_points(triplets)

    # TODO: should be given access to gamma?
    gamma = 5e-3
    dist_diff = self._compute_dist_diff(triplets, X, B)

    n_basis = B.shape[0]
    sizeT = triplets.shape[0]

    w = np.zeros((1, n_basis))
    avg_grad_w = np.zeros((1, n_basis))

    output_iter = 5000     # output every output_iter iterations

    best_w = np.empty((1, n_basis))
    obj = np.empty((self.max_iter, 1))
    nImp = np.empty((self.max_iter, 1), dtype=int)

    best_obj = np.inf

    for iter in range(self.max_iter):
      if (iter % output_iter == 0):

        obj1 = np.sum(w)*self.beta

        obj2 = 0.0
        count = 0

        for i in range(sizeT):
          slack_val = 1 + dist_diff[i, :].dot(w.T)

          if (slack_val > 0):
            count += 1
            obj2 += slack_val

        obj2 = obj2/sizeT

        obj[iter] = obj1 + obj2
        nImp[iter] = count

        if(self.verbose):
          print("[Global] iter %d\t obj %.6f\t num_imp %d" % (iter,
                obj[iter], nImp[iter]))

        # update the best
        if (obj[iter] < best_obj):
          best_obj = obj[iter]
          best_w = w

      idx = np.random.randint(low=0, high=sizeT)

      slack_val = 1 + dist_diff[idx, :].dot(w.T)

      if (slack_val > 0):
        avg_grad_w = (iter * avg_grad_w + dist_diff[idx, :]) / (iter+1)
      else:
        avg_grad_w = iter * avg_grad_w / (iter+1)

      scale_f = -np.sqrt(iter+1) / gamma

      # TODO: maybe there is a better way to do this?
      w.fill(0)
      pos_mask = avg_grad_w > self.beta
      w[pos_mask] = scale_f * (avg_grad_w[pos_mask] + self.beta)
      neg_mask = avg_grad_w < - self.beta
      w[neg_mask] = scale_f * (avg_grad_w[neg_mask] - self.beta)

      w[w < 0] = 0

    if(self.verbose):
      print("max iteration reached.")

    self.components_ = self._get_components(best_w, B)

    return self

# should this go to utils?
  def _compute_dist_diff(self, T, X, B):
    XB = np.matmul(X, B.T)
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

  def _get_components(self, w, B):
    """
    get components matrix (L) from computed mahalanobis matrix
    """

    # get rid of inactive bases
    active_idx = w > 0
    w = w[active_idx]
    B = B[np.squeeze(active_idx), :]

    K, d = B.shape

    if(K < d):  # if metric is low-rank
      return B*np.sqrt(w)[..., None]

    else:   # if metric is full rank
      return np.dot(B.T * np.sqrt(w), B)

  def _to_index_points(self, triplets):
    shape = triplets.shape
    X, triplets = np.unique(np.vstack(triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])
    return triplets, X


class SCML_global(_BaseSCML_global, _TripletsClassifierMixin):

  def fit(self, triplets, B, n_basis):
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
    B : (n_basis, n_features) array of floats that form the basis set from
      which the metric will be constructed.

    Returns
    -------
    self : object
      Returns the instance.
    """
    return _BaseSCML_global._fit(triplets, B, n_basis)


class SCML_global_Supervised(_BaseSCML_global, TransformerMixin):

  def __init__(self, k_genuine=3, k_impostor=10, beta=1e-5, B=None,
               n_basis=None, max_iter=100000, verbose=False,
               preprocessor=None, random_state=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    _BaseSCML_global.__init__(self, beta=beta, max_iter=max_iter,
                              verbose=verbose, preprocessor=preprocessor,
                              random_state=random_state)

  def fit(self, X, y, B, n_basis, random_state=None):
    """Create constraints from labels and learn the SCML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    B : string or (n_basis x d) array, through this the basis construction
    can be selected or directly given by an array.

    Returns
    -------
    self : object
      Returns the instance.
    """
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    self.preprocessor = X

    if(B == "LDA"):
      B = self._generate_bases_LDA(X, y, n_basis, random_state)
    # this should set super's B

    triplets = generate_knntriplets(X, y, self.k_genuine,
                                    self.k_impostor)

    return self._fit(triplets, B, n_basis)

  def _generate_bases_LDA(self, X, y, n_basis, random_state=None):

    labels, class_count = np.unique(y, return_counts=True)
    n_class = len(labels)

    # n_basis must be greater or equal to n_class
    if(n_basis < n_class):
        ValueError("number of basis should be greater than the number of "
                   "classes")

    dim = np.size(X, 1)
    num_eig = min(n_class-1, dim)
    n_clusters = int(np.ceil(n_basis/(2 * num_eig)))

    # TODO: maybe give acces to Kmeans jobs for faster computation?
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    algorithm='elkan').fit(X)
    cX = kmeans.cluster_centers_

    # TODO: find a better way to choose neighbourhood size
    if dim > 50:
        nK = 50
    else:
        nK = 10

    nK_class = [min(c, nK) for c in class_count]

    idx_set = np.zeros((n_clusters, sum(nK_class)), dtype=np.int)

    start = 0
    finish = 0
    neigh = NearestNeighbors()

    for c in range(n_class):
        sel_c = y == labels[c]
        nk = nK_class[c]
        # get nK_class genuine neighbours
        neigh.fit(X=X[sel_c])

        finish += nk
        idx_set[:, start:finish] = np.take(np.where(sel_c),
                                           neigh.kneighbors(X=cX,
                                           n_neighbors=nk,
                                           return_distance=False))
        start = finish

    B = np.zeros((n_basis, dim))
    for i in range(n_clusters):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])
        B[num_eig*i: num_eig*(i+1), :] = normalize(lda.scalings_.T)

    nK = 20

    nK_class = [min(c, nK) for c in class_count]

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

    for i in range(n_clusters):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X[idx_set[i, :]], y[idx_set[i, :]])
        start = finish
        finish += num_eig
        # TODO: maybe handle tail more elegantly by
        # limiting lda n_components
        if(start == n_basis):
            pass
        elif(finish <= n_basis):
            B[start:finish, :] = normalize(lda.scalings_.T)
        else:
            B[start:, :] = normalize(lda.scalings_.T[:n_basis-start])
            break

    return B
