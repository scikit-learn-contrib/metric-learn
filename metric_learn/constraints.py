"""
Helper module for generating different types of constraints
from supervised data labels.
"""
import numpy as np
import warnings
from six.moves import xrange
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from numpy.matlib import repmat

__all__ = ['Constraints']


class Constraints(object):
  """
  Class to build constraints from labels.

  See more in the :ref:`User Guide <supervised_version>`
  """
  def __init__(self, partial_labels):
    '''partial_labels : int arraylike, -1 indicating unknown label'''
    partial_labels = np.asanyarray(partial_labels, dtype=int)
    self.partial_labels = partial_labels

  def positive_negative_pairs(self, num_constraints, same_length=False,
                              random_state=None):
    random_state = check_random_state(random_state)
    a, b = self._pairs(num_constraints, same_label=True,
                       random_state=random_state)
    c, d = self._pairs(num_constraints, same_label=False,
                       random_state=random_state)
    if same_length and len(a) != len(c):
      n = min(len(a), len(c))
      return a[:n], b[:n], c[:n], d[:n]
    return a, b, c, d

  def generate_knntriplets(self, X, k_genuine, k_impostor):
    """
    Generates triplets for every point to `k_genuine` neighbors of the same
    class and `k_impostor` neighbors of other classes.

    For every point (X_a) the triplets (X_a, X_b, X_c) are constructed from all
    the combinations of taking `k_genuine` neighbors (X_b) of the same class
    and `k_impostor` neighbors (X_c) of other classes.

    Parameters
    ----------
      X : (n x d) matrix
        Input data, where each row corresponds to a single instance.
      k_genuine : int
        Number of neighbors of the same class to be taken into account.
      k_impostor : int
        Number of neighbors of different classes to be taken into account.

    Returns
    -------
    triplets : array-like, shape=(n_constraints, 3)
      2D array of triplets of indicators.
    """

    labels, labels_count = np.unique(self.partial_labels, return_counts=True)
    n_labels = len(labels)
    len_input = np.size(self.partial_labels, 0)

    # Handle the case where there are too few elements to yield k_genuine or
    # k_impostor neighbors for every class.

    k_genuine_vec = np.ones(n_labels, dtype=np.intp)*k_genuine
    k_impostor_vec = np.ones(n_labels, dtype=np.intp)*k_impostor

    for i in range(n_labels):
      if (k_genuine + 1 > labels_count[i]):
        k_genuine_vec[i] = labels_count[i]-1
        warnings.warn("The class {} has {} elements but a minimum of {},"
                      " which corresponds to k_genuine+1, is expected. "
                      "A lower number of k_genuine will be used for this"
                      "class.\n"
                      .format(labels[i], labels_count[i], k_genuine+1))
      if (k_impostor > len_input - labels_count[i]):
        k_impostor_vec[i] = len_input - labels_count[i]
        warnings.warn("The class {} has {} elements of other classes but a "
                      "minimum of {}, which corresponds to k_impostor, is"
                      " expected. A lower number of k_impostor will be used"
                      " for this class.\n"
                      .format(labels[i], len_input - labels_count[i],
                              k_impostor))

    triplets = np.empty((np.dot(k_genuine_vec*k_impostor_vec, labels_count),
                         3), dtype=np.intp)

    start = 0
    finish = 0
    neigh = NearestNeighbors()

    for i in range(n_labels):

        # generate mask for current label
        gen_mask = self.partial_labels == labels[i]
        gen_indx = np.where(gen_mask)

        # get k_genuine genuine neighbours
        neigh.fit(X=X[gen_indx])
        gen_neigh = np.take(gen_indx, neigh.kneighbors(
                            n_neighbors=k_genuine_vec[i],
                            return_distance=False))

        # generate mask for impostors of current label
        imp_indx = np.where(np.invert(gen_mask))

        # get k_impostor impostor neighbours
        neigh.fit(X=X[imp_indx])
        imp_neigh = np.take(imp_indx, neigh.kneighbors(
                            n_neighbors=k_impostor_vec[i],
                            X=X[gen_mask],
                            return_distance=False))

        # length = len_label*k_genuine*k_impostor
        finish += labels_count[i]*k_genuine_vec[i]*k_impostor_vec[i]

        triplets[start:finish, :] = self._comb(gen_indx, gen_neigh, imp_neigh,
                                               k_genuine_vec[i],
                                               k_impostor_vec[i])
        start = finish

    return triplets

  def _comb(self, A, B, C, sizeB, sizeC):
    # generate_knntripelts helper function
    # generate an array will all combinations of choosing
    # an element from A, B and C
    return np.vstack((repmat(A, sizeB*sizeC, 1).ravel(order='F'),
                      repmat(np.hstack(B), sizeC, 1).ravel(order='F'),
                      repmat(C, 1, sizeB).ravel())).T

  def _pairs(self, num_constraints, same_label=True, max_iter=10,
             random_state=np.random):
    known_label_idx, = np.where(self.partial_labels >= 0)
    known_labels = self.partial_labels[known_label_idx]
    num_labels = len(known_labels)
    ab = set()
    it = 0
    while it < max_iter and len(ab) < num_constraints:
      nc = num_constraints - len(ab)
      for aidx in random_state.randint(num_labels, size=nc):
        if same_label:
          mask = known_labels[aidx] == known_labels
          mask[aidx] = False  # avoid identity pairs
        else:
          mask = known_labels[aidx] != known_labels
        b_choices, = np.where(mask)
        if len(b_choices) > 0:
          ab.add((aidx, random_state.choice(b_choices)))
      it += 1
    if len(ab) < num_constraints:
      warnings.warn("Only generated %d %s constraints (requested %d)" % (
          len(ab), 'positive' if same_label else 'negative', num_constraints))
    ab = np.array(list(ab)[:num_constraints], dtype=int)
    return known_label_idx[ab.T]

  def chunks(self, num_chunks=100, chunk_size=2, random_state=None):
    """
    the random state object to be passed must be a numpy random seed
    """
    random_state = check_random_state(random_state)
    chunks = -np.ones_like(self.partial_labels, dtype=int)
    uniq, lookup = np.unique(self.partial_labels, return_inverse=True)
    unknown_uniq = np.where(uniq < 0)[0]
    all_inds = [set(np.where(lookup == c)[0]) for c in xrange(len(uniq))
                if c not in unknown_uniq]
    max_chunks = int(np.sum([len(s) // chunk_size for s in all_inds]))
    if max_chunks < num_chunks:
      raise ValueError(('Not enough possible chunks of %d elements in each'
                        ' class to form expected %d chunks - maximum number'
                        ' of chunks is %d'
                        ) % (chunk_size, num_chunks, max_chunks))
    idx = 0
    while idx < num_chunks and all_inds:
      if len(all_inds) == 1:
        c = 0
      else:
        c = random_state.randint(0, high=len(all_inds) - 1)
      inds = all_inds[c]
      if len(inds) < chunk_size:
        del all_inds[c]
        continue
      ii = random_state.choice(list(inds), chunk_size, replace=False)
      inds.difference_update(ii)
      chunks[ii] = idx
      idx += 1
    return chunks


def wrap_pairs(X, constraints):
  a = np.array(constraints[0])
  b = np.array(constraints[1])
  c = np.array(constraints[2])
  d = np.array(constraints[3])
  constraints = np.vstack((np.column_stack((a, b)), np.column_stack((c, d))))
  y = np.concatenate([np.ones_like(a), -np.ones_like(c)])
  pairs = X[constraints]
  return pairs, y
