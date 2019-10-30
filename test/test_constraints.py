import pytest
import numpy as np
from sklearn.utils import shuffle
from metric_learn.constraints import Constraints

SEED = 42


def gen_labels_for_chunks(num_chunks, chunk_size,
                          n_classes=10, n_unknown_labels=5):
  """Generates num_chunks*chunk_size labels that split in num_chunks chunks,
  that are homogeneous in the label."""
  classes = shuffle(np.arange(n_classes))
  n_per_class = chunk_size * (num_chunks // n_classes)
  n_maj_class = chunk_size * num_chunks - n_per_class * n_classes
  most_labels = [[k] * n_per_class for k in classes[1:]]
  labels = [classes[0] * n_maj_class] + most_labels + [-1] * n_unknown_labels
  return labels


@pytest.mark.parametrize('num_chunks, chunk_size', [(11, 5), (115, 12)])
def test_chunk_case_exact_num_points(num_chunks, chunk_size,
                                     n_classes=10, n_unknown_labels=5):
  """Checks that the chunk generation works well with just enough points."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size,
                                 n_classes=n_classes,
                                 n_unknown_labels=n_unknown_labels)
  constraints = Constraints(shuffle(labels, random_state=SEED))
  chunks = constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                              random_state=SEED)
  return chunks


@pytest.mark.parametrize('num_chunks, chunk_size', [(5, 10), (10, 50)])
def test_chunk_case_one_miss_point(num_chunks, chunk_size,
                                   n_classes=10, n_unknown_labels=5):
  """Checks that the chunk generation breaks when one point is missing."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size,
                                 n_classes=n_classes,
                                 n_unknown_labels=n_unknown_labels)
  assert len(labels) >= 1
  labels = shuffle(labels)[1:]
  constraints = Constraints(shuffle(labels, random_state=SEED))
  with pytest.raises(ValueError) as e:
    constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                       random_state=SEED)

  expected_message = (('Not enough examples in each class to form %d chunks '
                       'of %d examples - maximum number of chunks is %d'
                       ) % (num_chunks, chunk_size, num_chunks - 1))

  assert e.value.message == expected_message
