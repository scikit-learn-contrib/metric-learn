"""
Helper module for generating different types of constraints
from supervised data labels.
"""
import numpy as np
import warnings
from six.moves import xrange
from scipy.sparse import coo_matrix

__all__ = ['Constraints']


class Constraints(object):
  def __init__(self, partial_labels):
    '''partial_labels : int arraylike, -1 indicating unknown label'''
    partial_labels = np.asanyarray(partial_labels)
    self.num_points, = partial_labels.shape
    self.known_label_idx, = np.where(partial_labels >= 0)
    self.known_labels = partial_labels[self.known_label_idx]

  def adjacency_matrix(self, num_constraints, random_state=np.random):
    a, b, c, d = self.positive_negative_pairs(num_constraints,
                                              random_state=random_state)
    row = np.concatenate((a, c))
    col = np.concatenate((b, d))
    data = np.ones_like(row, dtype=int)
    data[len(a):] = -1
    adj = coo_matrix((data, (row, col)), shape=(self.num_points,)*2)
    # symmetrize
    return adj + adj.T

  def positive_negative_pairs(self, num_constraints, same_length=False,
                              random_state=np.random):
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
    num_labels = len(self.known_labels)
    ab = set()
    it = 0
    while it < max_iter and len(ab) < num_constraints:
      nc = num_constraints - len(ab)
      for aidx in random_state.randint(num_labels, size=nc):
        if same_label:
          mask = self.known_labels[aidx] == self.known_labels
          mask[aidx] = False  # avoid identity pairs
        else:
          mask = self.known_labels[aidx] != self.known_labels
        b_choices, = np.where(mask)
        if len(b_choices) > 0:
          ab.add((aidx, random_state.choice(b_choices)))
      it += 1
    if len(ab) < num_constraints:
      warnings.warn("Only generated %d %s constraints (requested %d)" % (
          len(ab), 'positive' if same_label else 'negative', num_constraints))
    ab = np.array(list(ab)[:num_constraints], dtype=int)
    return self.known_label_idx[ab.T]

  def chunks(self, num_chunks=100, chunk_size=2, random_state=np.random):
    """
    the random state object to be passed must be a numpy random seed
    """
    chunks = -np.ones_like(self.known_label_idx, dtype=int)
    uniq, lookup = np.unique(self.known_labels, return_inverse=True)
    all_inds = [set(np.where(lookup==c)[0]) for c in xrange(len(uniq))]
    idx = 0
    while idx < num_chunks and all_inds:
      c = random_state.randint(0, high=len(all_inds)-1)
      inds = all_inds[c]
      if len(inds) < chunk_size:
        del all_inds[c]
        continue
      ii = random_state.choice(list(inds), chunk_size, replace=False)
      inds.difference_update(ii)
      chunks[ii] = idx
      idx += 1
    if idx < num_chunks:
      raise ValueError('Unable to make %d chunks of %d examples each' %
                       (num_chunks, chunk_size))
    return chunks

  @staticmethod
  def random_subset(all_labels, num_preserved=np.inf, random_state=np.random):
    """
    the random state object to be passed must be a numpy random seed
    """
    n = len(all_labels)
    num_ignored = max(0, n - num_preserved)
    idx = random_state.randint(n, size=num_ignored)
    partial_labels = np.array(all_labels, copy=True)
    partial_labels[idx] = -1
    return Constraints(partial_labels)
