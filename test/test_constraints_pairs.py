import warnings

import numpy as np
import pytest

from metric_learn import Constraints
from metric_learn.constraints import wrap_pairs


@pytest.mark.parametrize('same_length', [False, True])
@pytest.mark.parametrize('labels, expected_lengths', [
    ([-1, 2, 5, -1, 9], (0, 2)),
    ([-1, 7, 7, -1, 7], (2, 0)),
    ([-1, 3, -1], (0, 0)),
])
def test_pairs_with_unavailable_pair_type(labels, expected_lengths,
                                          same_length):
  labels = np.array(labels)
  with pytest.warns(UserWarning, match='Only generated 0'):
    pairs = Constraints(labels).positive_negative_pairs(
        2, same_length=same_length, random_state=42)

  if same_length:
    expected_lengths = (0, 0)
  assert tuple(len(pairs[i]) for i in (0, 2)) == expected_lengths
  for indices in pairs:
    assert indices.ndim == 1
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all(labels[indices] >= 0)
  a, b, c, d = pairs
  np.testing.assert_array_equal(labels[a], labels[b])
  assert np.all(a != b)
  assert np.all(labels[c] != labels[d])

  X = np.arange(len(labels) * 2).reshape(-1, 2)
  formed_pairs, pair_labels = wrap_pairs(X, pairs)
  assert formed_pairs.shape == (sum(expected_lengths), 2, 2)
  np.testing.assert_array_equal(
      pair_labels, [1] * expected_lengths[0] + [-1] * expected_lengths[1])


@pytest.mark.parametrize('same_length', [False, True])
def test_zero_requested_pairs(same_length):
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    pairs = Constraints([0, 0, 1, 1]).positive_negative_pairs(
        0, same_length=same_length, random_state=42)
  assert len(pairs) == 4
  for indices in pairs:
    assert indices.shape == (0,)
    assert np.issubdtype(indices.dtype, np.integer)


@pytest.mark.parametrize('same_length', [False, True])
@pytest.mark.parametrize('labels', [[0, 0, 1, 1], [-1, 3, 3, -1, 8, 8]])
def test_nonempty_pair_generation_control(labels, same_length):
  labels = np.array(labels)
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    pairs = Constraints(labels).positive_negative_pairs(
        2, same_length=same_length, random_state=42)
  assert len(pairs) == 4
  for indices in pairs:
    assert indices.shape == (2,)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all(labels[indices] >= 0)
  a, b, c, d = pairs
  np.testing.assert_array_equal(labels[a], labels[b])
  assert np.all(a != b)
  assert np.all(labels[c] != labels[d])
