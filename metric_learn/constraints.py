"""
Helper module for generating different types of constraints
from supervised data labels.
"""
import numpy as np
import warnings
from six.moves import xrange
from scipy.sparse import coo_matrix
from sklearn.utils import check_array

__all__ = ['Constraints', 'ConstrainedDataset']


class Constraints(object):
  def __init__(self, partial_labels):
    '''partial_labels : int arraylike, -1 indicating unknown label'''
    partial_labels = np.asanyarray(partial_labels, dtype=int)
    self.num_points, = partial_labels.shape
    self.known_label_idx, = np.where(partial_labels >= 0)
    self.known_labels = partial_labels[self.known_label_idx]

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
      if len(all_inds) == 1:
        c = 0
      else:
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


class ConstrainedDataset(object):

  def __init__(self, X, c):
    # we convert the data to a suitable format
    self.X = check_array(X, accept_sparse=True, dtype=None, warn_on_dtype=True)
    self.c = check_array(c, dtype=['int'] + np.sctypes['int']
                                  + np.sctypes['uint'],
                         # we add 'int' at the beginning to tell it is the
                         # default format we want in case of conversion
                         ensure_2d=False, ensure_min_samples=False,
                         ensure_min_features=False, warn_on_dtype=True)
    self._check_index(self.X.shape[0], self.c)
    self.shape = (len(c) if hasattr(c, '__len__') else 0, self.X.shape[1])

  def __getitem__(self, item):
    return ConstrainedDataset(self.X, self.c[item])

  def __len__(self):
    return self.shape

  def __str__(self):
    return self.toarray().__str__()

  def __repr__(self):
    return self.toarray().__repr__()

  def toarray(self):
    return self.X[self.c]

  @staticmethod
  def _check_index(length, indices):
    max_index = np.max(indices)
    min_index = np.min(indices)
    pb_index = None
    if max_index >= length:
      pb_index = max_index
    elif min_index > length + 1:
      pb_index = min_index
    if pb_index is not None:
      raise IndexError("ConstrainedDataset cannot be created: the length of "
                       "the dataset is {}, so index {} is out of range."
                       .format(length, pb_index))

  @staticmethod
  def pairs_from_labels(y):
    # TODO: to be implemented
    raise NotImplementedError

  @staticmethod
  def triplets_from_labels(y):
    # TODO: to be implemented
    raise NotImplementedError


def unwrap_pairs(constrained_dataset, y):
  a = constrained_dataset.c[(y == 0)[:, 0]][:, 0]
  b = constrained_dataset.c[(y == 0)[:, 0]][:, 1]
  c = constrained_dataset.c[(y == 1)[:, 0]][:, 0]
  d = constrained_dataset.c[(y == 1)[:, 0]][:, 1]
  X = constrained_dataset.X
  return X, [a, b, c, d]

def wrap_pairs(X, constraints):
  a = np.array(constraints[0])
  b = np.array(constraints[1])
  c = np.array(constraints[2])
  d = np.array(constraints[3])
  constraints = np.vstack([np.hstack([a[:, None], b[:, None]]),
                           np.hstack([c[:, None], d[:, None]])])
  y = np.vstack([np.zeros((len(a), 1)), np.ones((len(c), 1))])
  constrained_dataset = ConstrainedDataset(X, constraints)
  return constrained_dataset, y

def unwrap_to_graph(constrained_dataset, y):

  X, [a, b, c, d] = unwrap_pairs(constrained_dataset, y)
  row = np.concatenate((a, c))
  col = np.concatenate((b, d))
  data = np.ones_like(row, dtype=int)
  data[len(a):] = -1
  adj = coo_matrix((data, (row, col)), shape=(constrained_dataset.X.shape[0],)
                                             * 2)
  return constrained_dataset.X, adj + adj.T