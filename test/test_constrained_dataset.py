import unittest
import numpy as np
from metric_learn.constraints import ConstrainedDataset
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.testing import assert_raise_message

X = np.random.randn(20, 5)
c = np.random.randint(0, X.shape[0], (15, 2))
cd = ConstrainedDataset(X, c)
y = np.random.randint(0, 2, c.shape[0])
group = np.random.randint(0, 3, c.shape[0])

c_shape = c.shape[0]


class TestConstrainedDataset(unittest.TestCase):

    @staticmethod
    def check_indexing(idx):
        # checks that an indexing returns the data we expect
        np.testing.assert_array_equal(cd[idx].c, c[idx])
        np.testing.assert_array_equal(cd[idx].toarray(), X[c[idx]])
        np.testing.assert_array_equal(cd[idx].toarray(), X[c][idx])

    def test_inputs(self):
        # test the allowed and forbidden ways to create a ConstrainedDataset
        ConstrainedDataset(X, c)
        two_points = [[1, 2], [3, 5]]
        out_of_range_constraints = [[1, 2], [0, 1]]
        msg = "ConstrainedDataset cannot be created: the length of " \
              "the dataset is 2, so index 2 is out of " \
              "range."
        assert_raise_message(IndexError, msg, ConstrainedDataset, two_points,
                             out_of_range_constraints)

    def test_getitem(self):
        # test different types of slicing
        i = np.random.randint(1, c_shape - 1)
        begin = np.random.randint(1, c_shape - 1)
        end = np.random.randint(begin + 1, c_shape)
        fancy_index = np.random.randint(0, c_shape, 20)
        binary_index = np.random.randint(0, 2, c_shape)
        boolean_index = binary_index.astype(bool)
        items = [0, c_shape - 1, i, slice(i), slice(0, begin), slice(begin,
                 end), slice(end, c_shape), slice(0, c_shape), fancy_index,
                 binary_index, boolean_index]
        for item in items:
            self.check_indexing(item)

    def test_repr(self):
        assert repr(cd) == repr(X[c])

    def test_str(self):
        assert str(cd) == str(X[c])

    def test_shape(self):
        assert cd.shape == (c.shape[0], X.shape[1])
        assert cd[0, 0].shape == (0, X.shape[1])

    def test_toarray(self):
        assert_array_equal(cd.toarray(), cd.X[c])

    def test_folding(self):
        # test that ConstrainedDataset is compatible with scikit-learn folding
        shuffle_list = [True, False]
        groups_list = [group, None]
        for alg in [KFold, StratifiedKFold]:
            for shuffle_i in shuffle_list:
                for group_i in groups_list:
                    for train_idx, test_idx in alg(
                            shuffle=shuffle_i).split(cd, y, group_i):
                        self.check_indexing(train_idx)
                        self.check_indexing(test_idx)
