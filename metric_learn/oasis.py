from .base_metric import BilinearMixin, _TripletsClassifierMixin
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.datasets import make_spd_matrix
from .constraints import Constraints
from ._util import _to_index_points, _get_random_indices


class _BaseOASIS(BilinearMixin, _TripletsClassifierMixin):
  """
  Key params:

  n_iter: Number of iterations. May differ from n_triplets

  c: Passive-agressive param. Controls trade-off bewteen remaining
     close to previous W_i-1 OR minimizing loss of current triplet

  random_state: int or numpy.RandomState or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int.

  shuffle : Whether the triplets should be shuffled beforehand

  random_sampling: Sample triplets, with repetition, uniform probability.
  """

  def __init__(
          self,
          preprocessor=None,
          n_iter=None,
          c=0.0001,
          random_state=None,
          shuffle=True,
          random_sampling=False,
          custom_M="identity",
          custom_order=None
          ):
    super().__init__(preprocessor=preprocessor)
    self.d = 0  # n_features
    self.n_iter = n_iter  # Max iterations
    self.c = c  # Trade-off param
    self.random_state = check_random_state(random_state)
    self.shuffle = shuffle  # Shuffle the trilplets
    self.random_sampling = random_sampling
    self.custom_M = custom_M
    self.custom_order = custom_order

  def _fit(self, triplets):
    """
    Fit OASIS model

    Parameters
    ----------
    triplets : (n x 3 x d) array of samples
    custom_order : User's custom order of triplets to feed oasis.
    """
    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')
    triplets, X = _to_index_points(triplets)  # Work with indices

    self.d = X.shape[1]  # (n_triplets, d)
    self.n_triplets = triplets.shape[0]  # (n_triplets, 3)

    self.components_ = self._check_M(self.custom_M)  # W matrix, needs self.d
    if self.n_iter is None:
      self.n_iter = self.n_triplets

    # Get the order in wich the algoritm will be fed
    if self.custom_order is not None:
      self.indices = self._check_custom_order(self.custom_order)
    else:
      self.indices = _get_random_indices(self.n_triplets, self.n_iter,
                                         self.shuffle, self.random_sampling,
                                         random_state=self.random_state)
    i = 0
    while i < self.n_iter:
        current_triplet = X[triplets[self.indices[i]]]
        loss = self._loss(current_triplet)
        vi = self._vi_matrix(current_triplet)
        fs = self._frobenius_squared(vi)
        # Global GD or Adjust to tuple
        tau_i = np.minimum(self.c, loss / fs)

        # Update components
        self.components_ = np.add(self.components_, tau_i * vi)
        i = i + 1

    return self

  def partial_fit(self, new_triplets, n_iter, shuffle=True,
                  random_sampling=False, custom_order=None):
    """
    Reuse previous fit, and feed the algorithm with new triplets. Shuffle,
    random sampling and custom_order options are available.

    A new n_iter param can be set for the new_triplets.
    """
    self.n_iter = n_iter
    self.fit(new_triplets, shuffle=shuffle, random_sampling=random_sampling,
             custom_order=custom_order)

  def _frobenius_squared(self, v):
    """
    Returns Frobenius norm of a point, squared
    """
    return np.trace(np.dot(v, v.T))

  def _loss(self, triplet):
    """
    Loss function in a triplet
    """
    S = -1 * self.score_pairs([[triplet[0], triplet[1]],
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

  def get_indices(self):
    """
    Returns an array containing indices of triplets, the order in
    which the algorithm was feed.
    """
    return self.indices

  def _check_custom_order(self, custom_order):
    """
    Checks that the custom order is in fact a list or numpy array,
    and has n_iter values in between (0, n_triplets)
    """

    custom_order = check_array(custom_order, ensure_2d=False,
                               allow_nd=True, copy=False,
                               force_all_finite=True, accept_sparse=True,
                               dtype=None, ensure_min_features=self.n_iter,
                               ensure_min_samples=0)
    if len(custom_order) != self.n_iter:
      raise ValueError('The leght of custom_order array ({}), must match '
                       'the number of iterations ({}).'
                       .format(len(custom_order), self.n_iter))

    indices = np.arange(self.n_triplets)
    for i in range(self.n_iter):
      if custom_order[i] not in indices:
        raise ValueError('Found the invalid value {} at index {}'
                         'in custom_order. Use values only between'
                         '0 and n_triplets ({})'
                         .format(custom_order[i], i, self.n_triplets))
    return custom_order

  def _check_M(self, custom_M=None):
    """
    Initiates the matrix M of the bilinear similarity to be learned.
    A custom matrix M can be provided, otherwise an string can be
    provided specifying an alternative: identity, random or spd.
    """
    if isinstance(custom_M, str):
      if custom_M == "identity":
        return np.identity(self.d)
      elif custom_M == "random":
        return self.random_state.rand(self.d, self.d)
      elif custom_M == "spd":
        return make_spd_matrix(self.d, random_state=self.random_state)
      else:
        raise ValueError("Invalid str custom_M for M initialization. "
                         "Strategies availables: identity, random, psd."
                         "Or you can provie a numpy custom matrix M")
    else:
      shape = np.shape(custom_M)
      if shape != (self.d, self.d):
        raise ValueError("The matrix M you provided has shape {}."
                         "You need to provide a matrix with shape "
                         "{}".format(shape, (self.d, self.d)))
      return custom_M


class OASIS(_BaseOASIS):

  def __init__(self, preprocessor=None, n_iter=None, c=0.0001,
               random_state=None, shuffle=True, random_sampling=False,
               custom_M="identity", custom_order=None):
      super().__init__(preprocessor=preprocessor, n_iter=n_iter, c=c,
                       random_state=random_state, shuffle=shuffle,
                       random_sampling=random_sampling,
                       custom_M=custom_M, custom_order=custom_order)

  def fit(self, triplets):
    return self._fit(triplets)


class OASIS_Supervised(OASIS):

  def __init__(self, k_genuine=3, k_impostor=10,
               preprocessor=None, n_iter=None, c=0.0001,
               random_state=None, shuffle=True, random_sampling=False,
               custom_M="identity", custom_order=None):
    self.k_genuine = k_genuine
    self.k_impostor = k_impostor
    super().__init__(preprocessor=preprocessor, n_iter=n_iter, c=c,
                     random_state=random_state, shuffle=shuffle,
                     random_sampling=random_sampling,
                     custom_M=custom_M, custom_order=custom_order)

  def fit(self, X, y):
    X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, self.k_genuine,
                                                self.k_impostor)
    triplets = X[triplets]

    return self._fit(triplets)
