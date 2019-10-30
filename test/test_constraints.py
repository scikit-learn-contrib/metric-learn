import pytest
import numpy as np
from sklearn.utils import shuffle
from metric_learn.constraints import Constraints

SEED = 42


def gen_labels_for_chunks(num_chunks, chunk_size,
                          n_classes=10, n_unknown_labels=5):
  """Generates num_chunks*chunk_size labels that split in num_chunks chunks,
  that are homogeneous in the label."""
  assert min(num_chunks, chunk_size) > 0
  classes = shuffle(np.arange(n_classes), random_state=SEED)
  n_per_class = chunk_size * (num_chunks // n_classes)
  n_maj_class = n_per_class + chunk_size * num_chunks - n_per_class * n_classes

  first_labels = classes[0] * np.ones(n_maj_class, dtype=int)
  remaining_labels = np.concatenate([k * np.ones(n_per_class, dtype=int)
                                     for k in classes[1:]])
  unknown_labels = -1 * np.ones(n_unknown_labels, dtype=int)

  labels = np.concatenate([first_labels, remaining_labels, unknown_labels])
  return shuffle(labels, random_state=SEED)


@pytest.mark.parametrize('num_chunks, chunk_size', [(11, 5), (115, 12)])
def test_chunk_case_exact_num_points(num_chunks, chunk_size,
                                     n_classes=10, n_unknown_labels=5):
  """Checks that the chunk generation works well with just enough points."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size,
                                 n_classes=n_classes,
                                 n_unknown_labels=n_unknown_labels)
  constraints = Constraints(labels)
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
  constraints = Constraints(labels[1:])
  with pytest.raises(ValueError) as e:
    constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                       random_state=SEED)

  expected_message = (('Not enough examples in each class to form %d chunks '
                       'of %d examples - maximum number of chunks is %d'
                       ) % (num_chunks, chunk_size, num_chunks - 1))

  assert str(e.value) == expected_message
