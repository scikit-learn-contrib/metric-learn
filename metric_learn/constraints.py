"""
Helper module for generating different types of constraints
from supervised data labels.
"""
import numpy as np
import warnings
from six.moves import xrange
from sklearn.utils import check_random_state

__all__ = ['Constraints']


class Constraints(object):
  """
  Class to build constraints from labeled data.

  See more in the :ref:`User Guide <supervised_version>`.

  Parameters
  ----------
  partial_labels : `numpy.ndarray` of ints, shape=(n_samples,)
    Array of labels, with -1 indicating unknown label.

  Attributes
  ----------
  partial_labels : `numpy.ndarray` of ints, shape=(n_samples,)
    Array of labels, with -1 indicating unknown label.
  """

  def __init__(self, partial_labels):
    partial_labels = np.asanyarray(partial_labels, dtype=int)
    self.partial_labels = partial_labels

  def positive_negative_pairs(self, num_constraints, same_length=False,
                              random_state=None):
    """
    Generates positive pairs and negative pairs from labeled data.

    Positive pairs are formed by randomly drawing ``num_constraints`` pairs of
    points with the same label. Negative pairs are formed by randomly drawing
    ``num_constraints`` pairs of points with different label.

    In the case where it is not possible to generate enough positive or
    negative pairs, a smaller number of pairs will be returned with a warning.

    Parameters
    ----------
    num_constraints : int
      Number of positive and negative constraints to generate.

    same_length : bool, optional (default=False)
      If True, forces the number of positive and negative pairs to be
      equal by ignoring some pairs from the larger set.

    random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.

    Returns
    -------
    a : array-like, shape=(n_constraints,)
      1D array of indicators for the left elements of positive pairs.

    b : array-like, shape=(n_constraints,)
      1D array of indicators for the right elements of positive pairs.

    c : array-like, shape=(n_constraints,)
      1D array of indicators for the left elements of negative pairs.

    d : array-like, shape=(n_constraints,)
      1D array of indicators for the right elements of negative pairs.
    """
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
    Generates chunks from labeled data.

    Each of ``num_chunks`` chunks is composed of ``chunk_size`` points from
    the same class drawn at random. Each point can belong to at most 1 chunk.

    In the case where there is not enough points to generate ``num_chunks``
    chunks of size ``chunk_size``, a ValueError will be raised.

    Parameters
    ----------
    num_chunks : int, optional (default=100)
      Number of chunks to generate.

    chunk_size : int, optional (default=2)
      Number of points in each chunk.

    random_state : int or numpy.RandomState or None, optional (default=None)
      A pseudo random number generator object or a seed for it if int.

    Returns
    -------
    chunks : array-like, shape=(n_samples,)
      1D array of chunk indicators, where -1 indicates that the point does not
      belong to any chunk.
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
