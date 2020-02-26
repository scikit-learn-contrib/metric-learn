import pytest
import numpy as np
from sklearn.utils import shuffle
from metric_learn.constraints import Constraints
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

SEED = 42


def gen_labels_for_chunks(num_chunks, chunk_size,
                          n_classes=10, n_unknown_labels=5):
  """Generates num_chunks*chunk_size labels that split in num_chunks chunks,
  that are homogeneous in the label."""
  assert min(num_chunks, chunk_size) > 0
  classes = shuffle(np.arange(n_classes), random_state=SEED)
  n_per_class = chunk_size * (num_chunks // n_classes)
  n_maj_class = chunk_size * num_chunks - n_per_class * (n_classes - 1)

  first_labels = classes[0] * np.ones(n_maj_class, dtype=int)
  remaining_labels = np.concatenate([k * np.ones(n_per_class, dtype=int)
                                     for k in classes[1:]])
  unknown_labels = -1 * np.ones(n_unknown_labels, dtype=int)

  labels = np.concatenate([first_labels, remaining_labels, unknown_labels])
  return shuffle(labels, random_state=SEED)


@pytest.mark.parametrize("num_chunks, chunk_size", [(5, 10), (10, 50)])
def test_exact_num_points_for_chunks(num_chunks, chunk_size):
  """Checks that the chunk generation works well with just enough points."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size)

  constraints = Constraints(labels)
  chunks = constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                              random_state=SEED)

  chunk_no, size_each_chunk = np.unique(chunks[chunks >= 0],
                                        return_counts=True)

  np.testing.assert_array_equal(size_each_chunk, chunk_size)
  assert chunk_no.shape[0] == num_chunks


@pytest.mark.parametrize("num_chunks, chunk_size", [(5, 10), (10, 50)])
def test_chunk_case_one_miss_point(num_chunks, chunk_size):
  """Checks that the chunk generation breaks when one point is missing."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size)

  assert len(labels) >= 1
  constraints = Constraints(labels[1:])
  with pytest.raises(ValueError) as e:
    constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                       random_state=SEED)

  expected_message = (('Not enough possible chunks of %d elements in each'
                       ' class to form expected %d chunks - maximum number'
                       ' of chunks is %d'
                       ) % (chunk_size, num_chunks, num_chunks - 1))

  assert str(e.value) == expected_message


@pytest.mark.parametrize("num_chunks, chunk_size", [(5, 10), (10, 50)])
def test_unknown_labels_not_in_chunks(num_chunks, chunk_size):
  """Checks that unknown labels are not assigned to any chunk."""
  labels = gen_labels_for_chunks(num_chunks, chunk_size)

  constraints = Constraints(labels)
  chunks = constraints.chunks(num_chunks=num_chunks, chunk_size=chunk_size,
                              random_state=SEED)

  assert np.all(chunks[labels < 0] < 0)


def test_generate_knntriplets():
  """Toy example validation of knn triplets construction"""
  k = 1
  X = np.array([[0, 0], [2, 2], [4, 4], [8, 8], [16, 16], [32, 32], [64, 64],
                [128, 128], [256, 256]])
  y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

  T_test = np.array([[0, 1, 3], [1, 0, 3], [2, 1, 3], [3, 4, 2], [4, 3, 2],
                     [5, 4, 2], [6, 7, 5], [7, 6, 5], [8, 7, 5]])
  T = Constraints(y).generate_knntriplets(X, k, k)

  assert len(list(set(map(tuple, T)) - set(map(tuple, T_test)))) == 0


@pytest.mark.parametrize("delta_genuine, delta_impostor", [(1, 1), (1, 2),
                                                           (2, 1), (2, 2)])
def test_generate_knntriplets_k(delta_genuine, delta_impostor):
  """Checks edge cases of knn triplet construction"""
  X, y = shuffle(*make_blobs(random_state=SEED),
                 random_state=SEED)

  label, labels_count = np.unique(y, return_counts=True)
  labels_count_min = np.min(labels_count)
  k_genuine = labels_count_min - delta_genuine

  length = len(y)
  labels_count_max = np.max(labels_count)
  k_impostor = length - labels_count_max + 1 - delta_impostor

  T = Constraints(y).generate_knntriplets(X, k_genuine, k_impostor)
  T_test = naive_generate_knntriplets(X, y, k_genuine, k_impostor)

  assert len(list(set(map(tuple, T)) - set(map(tuple, T_test)))) == 0


def naive_generate_knntriplets(X, y, k_genuine, k_impostor):
  """
  Generates triplets from labeled data. Naive implementation
  intended for testing.

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

  labels, labels_count = np.unique(y, return_counts=True)
  n_labels = len(labels)
  len_input = np.size(y, 0)

  triplets = np.empty((len_input*k_genuine*k_impostor, 3),
                      dtype=np.intp)

  j = 0
  neigh = NearestNeighbors()

  for i in range(n_labels):

      # generate mask for current label
      gen_mask = y == labels[i]
      gen_indx = np.where(gen_mask)

      # get k_genuine genuine neighbours
      neigh.fit(X=X[gen_indx])
      gen_neigh = np.take(gen_indx, neigh.kneighbors(
                          n_neighbors=k_genuine,
                          return_distance=False))

      # generate mask for impostors of current label
      imp_indx = np.where(np.invert(gen_mask))

      # get k_impostor impostor neighbours
      neigh.fit(X=X[imp_indx])
      imp_neigh = np.take(imp_indx, neigh.kneighbors(
                          n_neighbors=k_impostor,
                          X=X[gen_mask],
                          return_distance=False))

      for a, k in zip(gen_indx[0], range(len(gen_indx[0]))):
        for b in gen_neigh[k, :]:
          for c in imp_neigh[k, :]:
            triplets[j, :] = np.array([a, b, c])
            j += 1

  return triplets


def test_generate_knntriplets_k_genuine():
  """Checks the correct error raised when k_genuine is too big """
  X, y = shuffle(*make_blobs(random_state=SEED),
                 random_state=SEED)

  label, labels_count = np.unique(y, return_counts=True)
  labels_count_min = np.min(labels_count)
  idx_smallest_label = np.where(labels_count == labels_count_min)
  k_genuine = labels_count_min

  warn_msgs = []
  for idx in idx_smallest_label[0]:
    warn_msgs.append("The class {} has {} elements, which is not sufficient "
                     "to generate {} genuine neighbors as specified by "
                     "k_genuine. Will generate {} genuine neighbors instead."
                     "\n"
                     .format(label[idx], k_genuine, k_genuine+1, k_genuine-1))

  with pytest.warns(UserWarning) as raised_warning:
    Constraints(y).generate_knntriplets(X, k_genuine, 1)
  for warn in raised_warning:
    assert str(warn.message) in warn_msgs


def test_generate_knntriplets_k_impostor():
  """Checks the correct error raised when k_impostor is too big """
  X, y = shuffle(*make_blobs(random_state=SEED),
                 random_state=SEED)

  length = len(y)
  label, labels_count = np.unique(y, return_counts=True)
  labels_count_max = np.max(labels_count)
  idx_biggest_label = np.where(labels_count == labels_count_max)
  k_impostor = length - labels_count_max + 1

  warn_msgs = []
  for idx in idx_biggest_label[0]:
    warn_msgs.append("The class {} has {} elements of other classes, which is"
                     " not sufficient to generate {} impostor neighbors as "
                     "specified by k_impostor. Will generate {} impostor "
                     "neighbors instead.\n"
                     .format(label[idx], k_impostor-1, k_impostor,
                             k_impostor-1))

  with pytest.warns(UserWarning) as raised_warning:
    Constraints(y).generate_knntriplets(X, 1, k_impostor)
  for warn in raised_warning:
    assert str(warn.message) in warn_msgs
