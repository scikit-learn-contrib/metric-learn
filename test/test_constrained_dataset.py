import unittest
import numpy as np
import scipy
from metric_learn.constraints import ConstrainedDataset
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.mocking import MockDataFrame


class _BaseTestConstrainedDataset(object):

    def setUp(self):
        self.num_points = 20
        self.num_features = 5
        self.num_constraints = 15
        self.RNG = check_random_state(0)

        self.c = self.RNG.randint(0, self.num_points,
                                  (self.num_constraints, 2))
        self.y = self.RNG.randint(0, 2, self.num_constraints)
        self.group = self.RNG.randint(0, 3, self.num_constraints)

    def check_indexing(self, idx):
        # checks that an indexing returns the data we expect
        np.testing.assert_array_equal(self.X_constrained[idx].c, self.c[idx])
        np.testing.assert_array_equal(self.X_constrained[idx].toarray(),
                                      self.X[self.c[idx]])
        np.testing.assert_array_equal(self.X_constrained[idx].toarray(),
                                      self.X[self.c][idx])
        # checks that slicing does not copy the initial X
        self.assertTrue(self.X_constrained[idx].X is self.X_constrained.X)

    def test_allowed_inputs(self):
        # test the allowed ways to create a ConstrainedDataset
        ConstrainedDataset(self.X, self.c)

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
        i = self.RNG.randint(1, self.num_constraints - 1)
        begin = self.RNG.randint(1, self.num_constraints - 1)
        end = self.RNG.randint(begin + 1, self.num_constraints)
        fancy_index = self.RNG.randint(0, self.num_constraints, 20)
        binary_index = self.RNG.randint(0, 2, self.num_constraints)
        boolean_index = binary_index.astype(bool)
        items = [0, self.num_constraints - 1, i, slice(i), slice(0, begin),
                 slice(begin, end), slice(end, self.num_constraints),
                 slice(0, self.num_constraints), fancy_index,
                 fancy_index.tolist(), binary_index, binary_index.tolist(),
                 boolean_index, boolean_index.tolist()]
        for item in items:
            self.check_indexing(item)

    def test_repr(self):
        self.assertEqual(repr(self.X_constrained), repr(self.X[self.c]))

    def test_str(self):
        self.assertEqual(str(self.X_constrained), str(self.X[self.c]))

    def test_shape(self):
        self.assertEqual(self.X_constrained.shape, (self.c.shape[0],
                                                    self.c.shape[1],
                                                    self.X.shape[1]))
        self.assertEqual(self.X_constrained[0, 0].shape,
                         (0, 0, self.X.shape[1]))

    def test_len(self):
        self.assertEqual(len(self.X_constrained), self.c.shape[0])

    def test_toarray(self):
        X = self.X_constrained.X
        assert_array_equal(self.X_constrained.toarray(), X[self.c])

    def test_folding(self):
        # test that ConstrainedDataset is compatible with scikit-learn folding
        shuffle_list = [True, False]
        groups_list = [self.group, None]
        for alg in [KFold, StratifiedKFold]:
            for shuffle_i in shuffle_list:
                for group_i in groups_list:
                    for train_idx, test_idx \
                            in alg(shuffle=shuffle_i).split(self.X_constrained,
                                                            self.y,
                                                            group_i):
                        self.check_indexing(train_idx)
                        self.check_indexing(test_idx)


class TestDenseConstrainedDataset(_BaseTestConstrainedDataset,
                                  unittest.TestCase):

    def setUp(self):
        super(TestDenseConstrainedDataset, self).setUp()
        self.X = self.RNG.randn(self.num_points, self.num_features)
        self.X_constrained = ConstrainedDataset(self.X, self.c)

    def test_init(self):
        """
        Test alternative ways to initialize a ConstrainedDataset
        (where the remaining X will stay dense)
        """
        X_list = [self.X, self.X.tolist(), list(self.X), MockDataFrame(self.X)]
        c_list = [self.c, self.c.tolist(), list(self.c), MockDataFrame(self.c)]
        for X in X_list:
            for c in c_list:
                X_constrained = ConstrainedDataset(X, c)


class TestSparseConstrainedDataset(_BaseTestConstrainedDataset,
                                   unittest.TestCase):

    def setUp(self):
        super(TestSparseConstrainedDataset, self).setUp()
        self.X = scipy.sparse.random(self.num_points, self.num_features,
                                     format='csr', random_state=self.RNG)
        # todo: for now we test only csr but we should test all sparse types
        #  in the future
        self.X_constrained = ConstrainedDataset(self.X, self.c)

    def check_indexing(self, idx):
        # checks that an indexing returns the data we expect
        np.testing.assert_array_equal(self.X_constrained[idx].c, self.c[idx])
        np.testing.assert_array_equal(self.X_constrained[idx].toarray(),
                                      self.X.A[self.c[idx]])
        np.testing.assert_array_equal(self.X_constrained[idx].toarray(),
                                      self.X.A[self.c][idx])
        # checks that slicing does not copy the initial X
        self.assertTrue(self.X_constrained[idx].X is self.X_constrained.X)

    def test_repr(self):
        self.assertEqual(repr(self.X_constrained), repr(self.X.A[self.c]))

    def test_str(self):
        self.assertEqual(str(self.X_constrained), str(self.X.A[self.c]))

    def test_toarray(self):
        X = self.X_constrained.X
        assert_array_equal(self.X_constrained.toarray(), X.A[self.c])

    def test_folding(self):
        # test that ConstrainedDataset is compatible with scikit-learn folding
        shuffle_list = [True, False]
        groups_list = [self.group, None]
        for alg in [KFold, StratifiedKFold]:
            for shuffle_i in shuffle_list:
                for group_i in groups_list:
                    for train_idx, test_idx \
                            in alg(shuffle=shuffle_i).split(self.X_constrained,
                                                            self.y,
                                                            group_i):
                        self.check_indexing(train_idx)
                        self.check_indexing(test_idx)

if __name__=='__main__':
    unittest.main()