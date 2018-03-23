import unittest
import numpy as np
from metric_learn.constraints import ConstrainedDataset
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_raise_message

num_points = 20
num_features = 5
num_constraints = 15
RNG = check_random_state(0)

X = RNG.randn(num_points, num_features)
c = RNG.randint(0, num_points, (num_constraints, 2))
X_constrained = ConstrainedDataset(X, c)
y = RNG.randint(0, 2, num_constraints)
group = RNG.randint(0, 3, num_constraints)

class TestConstrainedDataset(unittest.TestCase):

    @staticmethod
    def check_indexing(idx):
        # checks that an indexing returns the data we expect
        np.testing.assert_array_equal(X_constrained[idx].c, c[idx])
        np.testing.assert_array_equal(X_constrained[idx].toarray(), X[c[idx]])
        np.testing.assert_array_equal(X_constrained[idx].toarray(), X[c][idx])

    def test_allowed_inputs(self):
        # test the allowed ways to create a ConstrainedDataset
        ConstrainedDataset(X, c)

    def test_invalid_inputs(self):
        # test the invalid ways to create a ConstrainedDataset
        two_points = [[1, 2], [3, 5]]
        out_of_range_constraints = [[1, 2], [0, 1]]
        msg = ("ConstrainedDataset cannot be created: the length of "
              "the dataset is 2, so index 2 is out of "
              "range.")
        assert_raise_message(IndexError, msg, ConstrainedDataset, two_points,
                             out_of_range_constraints)

    def test_getitem(self):
        # test different types of slicing
        i = RNG.randint(1, num_constraints - 1)
        begin = RNG.randint(1, num_constraints - 1)
        end = RNG.randint(begin + 1, num_constraints)
        fancy_index = RNG.randint(0, num_constraints, 20)
        binary_index = RNG.randint(0, 2, num_constraints)
        boolean_index = binary_index.astype(bool)
        items = [0, num_constraints - 1, i, slice(i), slice(0, begin),
                 slice(begin, end), slice(end, num_constraints),
                 slice(0, num_constraints), fancy_index, binary_index,
                 boolean_index]
        for item in items:
            self.check_indexing(item)

    def test_repr(self):
        self.assertEqual(repr(X_constrained), repr(X[c]))

    def test_str(self):
        self.assertEqual(str(X_constrained), str(X[c]))

    def test_shape(self):
        self.assertEqual(X_constrained.shape, (c.shape[0], c.shape[1],
                                               X.shape[1]))
        self.assertEqual(X_constrained[0, 0].shape, (0, 0, X.shape[1]))

    def test_len(self):
        self.assertEqual(len(X_constrained),c.shape[0])

    def test_toarray(self):
        assert_array_equal(X_constrained.toarray(), X_constrained.X[c])

    def test_folding(self):
        # test that ConstrainedDataset is compatible with scikit-learn folding
        shuffle_list = [True, False]
        groups_list = [group, None]
        for alg in [KFold, StratifiedKFold]:
            for shuffle_i in shuffle_list:
                for group_i in groups_list:
                    for train_idx, test_idx \
                            in alg(shuffle=shuffle_i).split(X_constrained, y,
                                                            group_i):
                        self.check_indexing(train_idx)
                        self.check_indexing(test_idx)
