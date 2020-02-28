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
    Generates triplets from labeled data.

    For every point (X_a) the triplets (X_a, X_b, X_c) are constructed from all
    the combinations of taking one of its `k_genuine`-nearest neighbors of the
    same class (X_b) and taking one of its `k_impostor`-nearest neighbors of
    other classes (X_c).

    In the case a class doesn't have enough points in the same class (other
    classes) to yield `k_genuine` (`k_impostor`) neighbors a warning will be
    raised and the maximum value of genuine (impostor) neighbors will be used
    for that class.

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
    # Ignore unlabeled samples
    known_labels_mask = self.partial_labels >= 0
    known_labels = self.partial_labels[known_labels_mask]
    X = X[known_labels_mask]

    labels, labels_count = np.unique(known_labels, return_counts=True)
    len_input = known_labels.shape[0]

    # Handle the case where there are too few elements to yield k_genuine or
    # k_impostor neighbors for every class.

    k_genuine_vec = np.full_like(labels, k_genuine)
    k_impostor_vec = np.full_like(labels, k_impostor)

    for i, count in enumerate(labels_count):
      if k_genuine + 1 > count:
        k_genuine_vec[i] = count-1
        warnings.warn("The class {} has {} elements, which is not sufficient "
                      "to generate {} genuine neighbors as specified by "
                      "k_genuine. Will generate {} genuine neighbors instead."
                      "\n"
                      .format(labels[i], count, k_genuine+1,
                              k_genuine_vec[i]))
      if k_impostor > len_input - count:
        k_impostor_vec[i] = len_input - count
        warnings.warn("The class {} has {} elements of other classes, which is"
                      " not sufficient to generate {} impostor neighbors as "
                      "specified by k_impostor. Will generate {} impostor "
                      "neighbors instead.\n"
                      .format(labels[i], k_impostor_vec[i], k_impostor,
                              k_impostor_vec[i]))

    # The total number of possible triplets combinations per label comes from
    # taking one of the k_genuine_vec[i] genuine neighbors and one of the
    # k_impostor_vec[i] impostor neighbors for the labels_count[i] elements
    comb_per_label = labels_count * k_genuine_vec * k_impostor_vec

    # Get start and finish for later triplet assigning
    # append zero at the begining for start and get cumulative sum
    start_finish_indices = np.hstack((0, comb_per_label)).cumsum()

    # Total number of triplets is the sum of all possible combinations per
    # label
    num_triplets = start_finish_indices[-1]
    triplets = np.empty((num_triplets, 3), dtype=np.intp)

    neigh = NearestNeighbors()

    for i, label in enumerate(labels):

        # generate mask for current label
        gen_mask = known_labels == label
        gen_indx = np.where(gen_mask)

        # get k_genuine genuine neighbors
        neigh.fit(X=X[gen_indx])
        gen_neigh = np.take(gen_indx, neigh.kneighbors(
                            n_neighbors=k_genuine_vec[i],
                            return_distance=False))

        # generate mask for impostors of current label
        imp_indx = np.where(~gen_mask)

        # get k_impostor impostor neighbors
        neigh.fit(X=X[imp_indx])
        imp_neigh = np.take(imp_indx, neigh.kneighbors(
                            n_neighbors=k_impostor_vec[i],
                            X=X[gen_mask],
                            return_distance=False))

        # length = len_label*k_genuine*k_impostor
        start, finish = start_finish_indices[i:i+2]

        triplets[start:finish, :] = comb(gen_indx, gen_neigh, imp_neigh,
                                         k_genuine_vec[i],
                                         k_impostor_vec[i])

    return triplets

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


def comb(A, B, C, sizeB, sizeC):
    # generate_knntriplets helper function
    # generate an array with all combinations of choosing
    # an element from A, B and C
    return np.vstack((repmat(A, sizeB*sizeC, 1).ravel(order='F'),
                      repmat(np.hstack(B), sizeC, 1).ravel(order='F'),
                      repmat(C, 1, sizeB).ravel())).T


def wrap_pairs(X, constraints):
  a = np.array(constraints[0])
  b = np.array(constraints[1])
  c = np.array(constraints[2])
  d = np.array(constraints[3])
  constraints = np.vstack((np.column_stack((a, b)), np.column_stack((c, d))))
  y = np.concatenate([np.ones_like(a), -np.ones_like(c)])
  pairs = X[constraints]
  return pairs, y
