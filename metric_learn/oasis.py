"""
Online Algorithm for Scalable Image Similarity (OASIS)
"""

from .base_metric import BilinearMixin, _TripletsClassifierMixin
import numpy as np
from sklearn.utils import check_random_state
from .constraints import Constraints
from ._util import _to_index_points, _get_random_indices, \
                   _initialize_similarity_bilinear


class _BaseOASIS(BilinearMixin, _TripletsClassifierMixin):
  def __init__(
          self,
          preprocessor=None,
          n_iter=None,
          c=0.0001,
          random_state=None,
          shuffle=True,
          random_sampling=False,
          init="identity",
          custom_order=None
          ):
    super().__init__(preprocessor=preprocessor)
    self.n_iter = n_iter  # Max iterations
    self.c = c  # Trade-off param
    self.random_state = check_random_state(random_state)
    self.shuffle = shuffle  # Shuffle the trilplets
    self.random_sampling = random_sampling
    self.init = init
    self.custom_order = custom_order

  def _fit(self, triplets):
    """
    Fit OASIS model

    Parameters
    ----------
    triplets : (n x 3 x d) array of samples
    """
    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')
    triplets, X = _to_index_points(triplets)  # Work with indices

    self.n_triplets = triplets.shape[0]  # (n_triplets, 3)
    if self.n_iter is None:
      self.n_iter = self.n_triplets

    M = _initialize_similarity_bilinear(X[triplets],
                                        init=self.init,
                                        strict_pd=False,
                                        random_state=self.random_state)
    self.components_ = M

    self.indices = _get_random_indices(self.n_triplets,
                                       self.n_iter,
                                       shuffle=self.shuffle,
                                       random=self.random_sampling,
                                       random_state=self.random_state,
                                       custom=self.custom_order)
    i = 0
    while i < self.n_iter:
        current_triplet = X[triplets[self.indices[i]]]
        loss = self._loss(current_triplet)
        vi = self._vi_matrix(current_triplet)
        fs = np.linalg.norm(vi, ord='fro') ** 2
        # Global GD or Adjust to tuple
        tau_i = np.minimum(self.c, loss / fs)

        # Update components
        self.components_ = np.add(self.components_, tau_i * vi)
        i = i + 1

    return self

  def partial_fit(self, new_triplets, n_iter, shuffle=True,
                  random_sampling=False, custom_order=None):
    """
    Reuse previous fit, and feed the algorithm with new triplets.
    A new n_iter can be set for these new triplets.

    Parameters
    ----------
    new_ triplets : (n x 3 x d) array of samples

    n_iter: int (default = n_triplets)
      Number of iterations. When n_iter < n_triplets, a random sampling
      takes place without repetition, but preserving the original order.
      When n_iter = n_triplets, all triplets are included with the
      original order. When n_iter > n_triplets, each triplet is included
      at least floor(n_iter/n_triplets) times, while some may have one
      more apparition at most. The order is preserved as well.

    shuffle: bool (default = True)
      Whether the triplets should be shuffled after the sampling process.
      If n_iter > n_triplets, then the suffle happends during the sampling
      and at the end.

    random_sampling: bool (default = False)
      If enabled, the algorithm will sample n_iter triplets from
      the input. This sample can contain duplicates. It does not
      matter if n_iter is lower, equal or greater than the number
      of triplets. The sampling uses uniform distribution.

    custom_order : array-like, optinal (default = None)
      User's custom order of triplets to feed oasis. Might be useful when
      trying to put a bias in the resulting similarity matrix.
    """
    self.n_iter = n_iter
    self.fit(new_triplets, shuffle=shuffle, random_sampling=random_sampling,
             custom_order=custom_order)

  def _loss(self, triplet):
    """
    Loss function in a triplet
    """
    S = self.pair_similarity([[triplet[0], triplet[1]],
                              [triplet[0], triplet[2]]])
    return np.maximum(0, 1 - S[0] + S[1])

  def _vi_matrix(self, triplet):
    """
    Computes V_i, the gradient matrix in a triplet
    """
    # (pi+ - pi-)
    diff = np.subtract(triplet[1], triplet[2])  # Shape (, d)
    result = []

    # For each scalar in first triplet, multiply by the diff of pi+ and pi-
    for v in triplet[0]:
        result.append(v * diff)

    return np.array(result)  # Shape (d, d)


class OASIS(_BaseOASIS):
  """Online Algorithm for Scalable Image Similarity (OASIS)

  `OASIS` learns a bilinear similarity from triplet constraints with an online
  Passive-Agressive (PA) algorithm approach. The bilinear similarity
  between :math:`p_1` and :math:`p_2` is defined as :math:`p_{1}^{T} W p_2`
  where :math:`W` is the learned matrix by OASIS. This particular algorithm
  is fast as it scales linearly with the number of samples.

  Read more in the :ref:`User Guide <oasis>`.

  .. warning::
    OASIS is still a bit experimental, don't hesitate to report if
    something fails/doesn't work as expected.

  Parameters
  ----------
  n_iter: int (default = n_triplets)
    Number of iterations. When n_iter < n_triplets, a random sampling
    takes place without repetition, but preserving the original order.
    When n_iter = n_triplets, all triplets are included with the
    original order. When n_iter > n_triplets, each triplet is included
    at least floor(n_iter/n_triplets) times, while some may have one
    more apparition at most. The order is preserved as well.

  shuffle: bool (default = True)
    Whether the triplets should be shuffled after the sampling process.
    If n_iter > n_triplets, then the suffle happends during the sampling
    and at the end.

  random_sampling: bool (default = False)
    If enabled, the algorithm will sample n_iter triplets from
    the input. This sample can contain duplicates. It does not
    matter if n_iter is lower, equal or greater than the number
    of triplets. The sampling uses uniform distribution.

  c: float (default = 1e-4)
    Passive-agressive param. Controls trade-off bewteen remaining
    close to previous W_i-1 or minimizing loss of the current triplet.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get triplets from indices. If array-like,
    triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.

  custom_order : array-like, optinal (default = None)
    User's custom order of triplets to feed oasis. Might be useful when
    trying to put a bias in the resulting similarity matrix.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The matrix W learned for the bilinear similarity.

  indices : `numpy.ndarray`, shape=(n_iter)
    The final order in which the triplets fed the algorithm. It's the list
    of indices in respect to the original triplet list given as input.

  Examples
  --------
  >>> from metric_learn import OASIS
  >>> triplets = [[[1.2, 7.5], [1.3, 1.5], [6.2, 9.7]],
  >>>             [[1.3, 4.5], [3.2, 4.6], [5.4, 5.4]],
  >>>             [[3.2, 7.5], [3.3, 1.5], [8.2, 9.7]],
  >>>             [[3.3, 4.5], [5.2, 4.6], [7.4, 5.4]]]
  >>> oasis = OASIS()
  >>> oasis.fit(triplets)

  References
  ----------
  .. [1] Chechik, Gal and Sharma, Varun and Shalit, Uri and Bengio, Samy
         `Large Scale Online Learning of Image Similarity Through Ranking.
         <https://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf>`_. \
         , JMLR 2010.

  .. [2] Adapted from original \
         `Matlab implementation.\
           <https://chechiklab.biu.ac.il/~gal/Research/OASIS/index.html>`_.

  See Also
  --------
  metric_learn.OASIS_Supervised : The supervised version of the algorithm.

  :ref:`supervised_version` : The section of the project documentation
    that describes the supervised version of weakly supervised estimators.
  """

  def __init__(self, preprocessor=None, n_iter=None, c=0.0001,
               random_state=None, shuffle=True, random_sampling=False,
               init="identity", custom_order=None):
      super().__init__(preprocessor=preprocessor, n_iter=n_iter, c=c,
                       random_state=random_state, shuffle=shuffle,
                       random_sampling=random_sampling,
                       init=init, custom_order=custom_order)

  def fit(self, triplets):
    """Learn the OASIS model.

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


class OASIS_Supervised(OASIS):
  """Online Algorithm for Scalable Image Similarity (OASIS)

  `OASIS_Supervised` creates triplets by taking `k_genuine` neighbours
  of the same class and `k_impostor` neighbours from different classes for each
  point and then runs the OASIS algorithm on these triplets.

  Read more in the :ref:`User Guide <oasis>`.

  .. warning::
    OASIS is still a bit experimental, don't hesitate to report if
    something fails/doesn't work as expected.

  Parameters
  ----------
  n_iter: int (default = n_triplets)
    Number of iterations. When n_iter < n_triplets, a random sampling
    takes place without repetition, but preserving the original order.
    When n_iter = n_triplets, all triplets are included with the
    original order. When n_iter > n_triplets, each triplet is included
    at least floor(n_iter/n_triplets) times, while some may have one
    more apparition at most. The order is preserved as well.

  shuffle: bool (default = True)
    Whether the triplets should be shuffled after the sampling process.
    If n_iter > n_triplets, then the suffle happends during the sampling
    and at the end.

  random_sampling: bool (default = False)
    If enabled, the algorithm will sample n_iter triplets from
    the input. This sample can contain duplicates. It does not
    matter if n_iter is lower, equal or greater than the number
    of triplets. The sampling uses uniform distribution.

  c: float (default = 1e-4)
    Passive-agressive param. Controls trade-off bewteen remaining
    close to previous W_i-1 or minimizing loss of the current triplet.

  preprocessor : array-like, shape=(n_samples, n_features) or callable
    The preprocessor to call to get triplets from indices. If array-like,
    triplets will be formed like this: X[indices].

  random_state : int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.

  custom_order : array-like, optinal (default = None)
    User's custom order of triplets to feed oasis. Might be useful when
    trying to put a bias in the resulting similarity matrix.

  Attributes
  ----------
  components_ : `numpy.ndarray`, shape=(n_features, n_features)
    The matrix W learned for the bilinear similarity.

  indices : `numpy.ndarray`, shape=(n_iter)
    The final order in which the triplets fed the algorithm. It's the list
    of indices in respect to the original triplet list given as input.

  Examples
  --------
  >>> from metric_learn import OASIS_Supervised
  >>> from sklearn.datasets import load_iris
  >>> iris_data = load_iris()
  >>> X = iris_data['data']
  >>> Y = iris_data['target']
  >>> oasis = OASIS_Supervised()
  >>> oasis.fit(X, Y)
  OASIS_Supervised(n_iter=4500,
                 random_state=RandomState(MT19937) at 0x7FE1B598FA40)
  >>> oasis.pair_similarity([[X[0], X[1]]])
  array([-21.14242072])

  References
  ----------
  .. [1] Chechik, Gal and Sharma, Varun and Shalit, Uri and Bengio, Samy
         `Large Scale Online Learning of Image Similarity Through Ranking.
         <https://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf>`_. \
         , JMLR 2010.

  .. [2] Adapted from original \
         `Matlab implementation.\
           <https://chechiklab.biu.ac.il/~gal/Research/OASIS/index.html>`_.

  See Also
  --------
  metric_learn.OASIS : The weakly supervised version of this
    algorithm.
  """

  def __init__(self, k_genuine=3, k_impostor=10,
               preprocessor=None, n_iter=None, c=0.0001,
               random_state=None, shuffle=True, random_sampling=False,
               init="identity", custom_order=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    super().__init__(preprocessor=preprocessor, n_iter=n_iter, c=c,
                     random_state=random_state, shuffle=shuffle,
                     random_sampling=random_sampling,
                     init=init, custom_order=custom_order)

  def fit(self, X, y):
    """Create constraints from labels and learn the OASIS model.

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
    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, self.k_genuine,
                                                self.k_impostor)
    triplets = X[triplets]

    return self._fit(triplets)
