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


def _comb(A, B, C, sizeB, sizeC):
  # generate an array will all combinations of choosing
  # an element from A, B and C
  return np.vstack((repmat(A, sizeB*sizeC, 1).ravel(order='F'),
                    repmat(np.hstack(B), sizeC, 1).ravel(order='F'),
                    repmat(C, 1, sizeB).ravel())).T


def generate_knntriplets(X, y, k_genuine, k_impostor):

  labels = np.unique(y)
  L = len(labels)
  len_input = np.size(y, 0)
  triplets = np.empty((len_input*k_genuine*k_impostor, 3), dtype=np.intp)

  start = 0
  finish = 0
  neigh = NearestNeighbors()

  for i in range(L):

      # generate mask for current label
      gen_mask = y == labels[i]
      gen_indx = np.where(gen_mask)

      # get k_genuine genuine neighbours
      neigh.fit(X=X[gen_indx])
      gen_neigh = np.take(gen_indx, neigh.kneighbors(n_neighbors=k_genuine,
                          return_distance=False))

      # generate mask for impostors of current label
      imp_indx = np.where(np.invert(gen_mask))

      # get k_impostor impostor neighbours
      neigh.fit(X=X[imp_indx])
      imp_neigh = np.take(imp_indx, neigh.kneighbors(
                          n_neighbors=k_impostor,
                          X=X[gen_mask],
                          return_distance=False))

      # lenght = len_label*k_genuine*k_impostor
      finish += np.sum(gen_mask)*k_genuine*k_impostor

      triplets[start:finish, :] = _comb(gen_indx, gen_neigh,
                                        imp_neigh, k_genuine,
                                        k_impostor)
      start = finish

      # TODO: deal with too litle elements for k neighbors to be yielded

  return triplets
