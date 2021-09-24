from .base_metric import BilinearMixin, _TripletsClassifierMixin
import numpy as np
from sklearn.utils import check_random_state


class OASIS(BilinearMixin, _TripletsClassifierMixin):
  """
  Key params:

  max_iter: Max number of iterations. If max_iter > n_triplets,
  a random sampling of seen triplets takes place to feed the model.

  c: Passive-agressive param. Controls trade-off bewteen remaining
        close to previous W_i-1 OR minimizing loss of current triplet

  seed: For random sampling

  shuffle: If True will shuffle the triplets given to fit.
  """

  def __init__(
          self,
          preprocessor=None,
          max_iter=10,
          c=1e-6,
          random_state=None,
          shuffle=False):
    super().__init__(preprocessor=preprocessor)
    self.components_ = None  # W matrix
    self.d = 0  # n_features
    self.max_iter = max_iter  # Max iterations
    self.c = c  # Trade-off param
    self.random_state = random_state  # RNG

  def fit(self, triplets):
    """
    Fit OASIS model

    Parameters
    ----------
    X : (n x d) array of samples
    """
    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO: (Same as SCML)
    # This algorithm is built to work with indices, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following fusnction. This should be improved
    # in the future.
    # Output: indices_to_X, X = unique(triplets)
    triplets, X = self._to_index_points(triplets)

    self.d = X.shape[1]  # (n_triplets, d)
    n_triplets = triplets.shape[0]  # (n_triplets, 3)
    rng = check_random_state(self.random_state)

    self.components_ = np.identity(
        self.d) if self.components_ is None else self.components_

    # Gen max_iter random indices
    random_indices = rng.randint(
        low=0, high=n_triplets, size=(
            self.max_iter))

    i = 0
    while i < self.max_iter:
        current_triplet = X[triplets[random_indices[i]]]
        loss = self._loss(current_triplet)
        vi = self._vi_matrix(current_triplet)
        fs = self._frobenius_squared(vi)
        # Global GD or Adjust to tuple
        tau_i = np.minimum(self.c, loss / fs)

        # Update components
        self.components_ = np.add(self.components_, tau_i * vi)
        i = i + 1

    return self

  def partial_fit(self, new_triplets):
    """
    self.components_ already defined, we reuse previous fit
    """
    self.fit(new_triplets)

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

  def _to_index_points(self, o_triplets):
    """
    Takes the origial triplets, and returns a mapping of the triplets
    to an X array that has all unique point values.

    Returns:

    X: Unique points across all triplets.

    triplets: Triplets-shaped values that represent the indices of X.
    Its guaranteed that shape(triplets) = shape(o_triplets[:-1])
    """
    shape = o_triplets.shape  # (n_triplets, 3, n_features)
    X, triplets = np.unique(np.vstack(o_triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])  # (n_triplets, 3)
    return triplets, X
