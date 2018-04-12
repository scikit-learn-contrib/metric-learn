import unittest
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import is_public_parameter
from sklearn.utils.testing import set_random_state, assert_true, \
    assert_allclose_dense_sparse, assert_dict_equal, assert_false

from metric_learn import ITML, LSML, MMC, SDML
from metric_learn.constraints import ConstrainedDataset, Constraints, \
    wrap_pairs
from sklearn.utils import check_random_state, shuffle
import numpy as np

class _TestWeaklySupervisedBase(object):

    def setUp(self):
        self.RNG = check_random_state(0)
        set_random_state(self.estimator)
        dataset = load_iris()
        self.X, y = shuffle(dataset.data, dataset.target, random_state=self.RNG)
        self.X, y = self.X[:20], y[:20]
        num_constraints = 20
        constraints = Constraints.random_subset(y, random_state=self.RNG)
        self.pairs = constraints.positive_negative_pairs(num_constraints,
                        same_length=True,
                        random_state=self.RNG)

    def test_cross_validation(self):
        # test that you can do cross validation on a ConstrainedDataset with
        #  a WeaklySupervisedMetricLearner
        estimator = clone(self.estimator)
        self.assertTrue(np.isfinite(cross_val_score(estimator,
                                    self.X_constrained, self.y)).all())

    def check_score(self, estimator, X_constrained, y):
        score = estimator.score(X_constrained, y)
        self.assertTrue(np.isfinite(score))

    def check_predict(self, estimator, X_constrained):
        y_predicted = estimator.predict(X_constrained)
        self.assertEqual(len(y_predicted), len(X_constrained))

    def check_transform(self, estimator, X_constrained):
        X_transformed = estimator.transform(X_constrained)
        self.assertEqual(len(X_transformed), len(X_constrained.X))

    def test_simple_estimator(self):
        estimator = clone(self.estimator)
        estimator.fit(self.X_constrained_train, self.y_train)
        self.check_score(estimator, self.X_constrained_test, self.y_test)
        self.check_predict(estimator, self.X_constrained_test)
        self.check_transform(estimator, self.X_constrained_test)

    def test_pipelining_with_transformer(self):
        """
        Test that weakly supervised estimators fit well into pipelines
        """
        # test in a pipeline with KMeans
        estimator = clone(self.estimator)
        pipe = make_pipeline(estimator, KMeans())
        pipe.fit(self.X_constrained_train, self.y_train)
        self.check_score(pipe, self.X_constrained_test, self.y_test)
        self.check_transform(pipe, self.X_constrained_test)
        # we cannot use check_predict because in this case the shape of the
        # output is the shape of X_constrained.X, not X_constrained
        y_predicted = pipe.predict(self.X_constrained)
        self.assertEqual(len(y_predicted), len(self.X_constrained.X))

        # test in a pipeline with PCA
        estimator = clone(self.estimator)
        pipe = make_pipeline(estimator, PCA())
        pipe.fit(self.X_constrained_train, self.y_train)
        self.check_transform(pipe, self.X_constrained_test)

    def test_no_fit_attributes_set_in_init(self):
        """Check that Estimator.__init__ doesn't set trailing-_ attributes."""
        # From scikit-learn
        estimator = clone(self.estimator)
        for attr in dir(estimator):
            if attr.endswith("_") and not attr.startswith("__"):
                # This check is for properties, they can be listed in dir
                # while at the same time have hasattr return False as long
                # as the property getter raises an AttributeError
                assert_false(
                    hasattr(estimator, attr),
                    "By convention, attributes ending with '_' are "
                    'estimated from data in scikit-learn. Consequently they '
                    'should not be initialized in the constructor of an '
                    'estimator but in the fit method. Attribute {!r} '
                    'was found in estimator {}'.format(
                        attr, type(estimator).__name__))

    def test_estimators_fit_returns_self(self):
        """Check if self is returned when calling fit"""
        # From scikit-learn
        estimator = clone(self.estimator)
        assert_true(estimator.fit(self.X_constrained, self.y) is estimator)

    def test_pipeline_consistency(self):
        # From scikit learn
        # check that make_pipeline(est) gives same score as est
        estimator = clone(self.estimator)
        pipeline = make_pipeline(estimator)
        estimator.fit(self.X_constrained, self.y)
        pipeline.fit(self.X_constrained, self.y)

        funcs = ["score", "fit_transform"]

        for func_name in funcs:
            func = getattr(estimator, func_name, None)
            if func is not None:
                func_pipeline = getattr(pipeline, func_name)
                result = func(self.X_constrained, self.y)
                result_pipe = func_pipeline(self.X_constrained, self.y)
                assert_allclose_dense_sparse(result, result_pipe)

    def test_dict_unchanged(self):
        # From scikit-learn
        estimator = clone(self.estimator)
        if hasattr(estimator, "n_components"):
            estimator.n_components = 1
        estimator.fit(self.X_constrained, self.y)
        for method in ["predict", "transform", "decision_function",
                       "predict_proba"]:
            if hasattr(estimator, method):
                dict_before = estimator.__dict__.copy()
                getattr(estimator, method)(self.X_constrained)
                assert_dict_equal(estimator.__dict__, dict_before,
                                  'Estimator changes __dict__ during %s'
                                  % method)

    def test_dont_overwrite_parameters(self):
        # From scikit-learn
        # check that fit method only changes or sets private attributes
        estimator = clone(self.estimator)
        if hasattr(estimator, "n_components"):
            estimator.n_components = 1
        dict_before_fit = estimator.__dict__.copy()

        estimator.fit(self.X_constrained, self.y)
        dict_after_fit = estimator.__dict__

        public_keys_after_fit = [key for key in dict_after_fit.keys()
                                 if is_public_parameter(key)]

        attrs_added_by_fit = [key for key in public_keys_after_fit
                              if key not in dict_before_fit.keys()]

        # check that fit doesn't add any public attribute
        assert_true(not attrs_added_by_fit,
                    ('Estimator adds public attribute(s) during'
                     ' the fit method.'
                     ' Estimators are only allowed to add private '
                     'attributes'
                     ' either started with _ or ended'
                     ' with _ but %s added' % ', '.join(
                        attrs_added_by_fit)))

        # check that fit doesn't change any public attribute
        attrs_changed_by_fit = [key for key in public_keys_after_fit
                                if (dict_before_fit[key]
                                    is not dict_after_fit[key])]

        assert_true(not attrs_changed_by_fit,
                    ('Estimator changes public attribute(s) during'
                     ' the fit method. Estimators are only allowed'
                     ' to change attributes started'
                     ' or ended with _, but'
                     ' %s changed' % ', '.join(attrs_changed_by_fit)))


class _TestPairsBase(_TestWeaklySupervisedBase):

    def setUp(self):
        super(_TestPairsBase, self).setUp()
        self.X_constrained, self.y = wrap_pairs(self.X, self.pairs)
        self.X_constrained, self.y = shuffle(self.X_constrained, self.y,
                                             random_state=self.RNG)
        self.X_constrained_train, self.X_constrained_test, self.y_train, \
            self.y_test = train_test_split(self.X_constrained, self.y)


class _TestQuadrupletsBase(_TestWeaklySupervisedBase):

    def setUp(self):
        super(_TestQuadrupletsBase, self).setUp()
        c = np.column_stack(self.pairs)
        self.X_constrained = ConstrainedDataset(self.X, c)
        self.X_constrained = shuffle(self.X_constrained)
        self.y, self.y_train, self.y_test = None, None, None
        self.X_constrained_train, self.X_constrained_test = train_test_split(
            self.X_constrained)


class TestITML(_TestPairsBase, unittest.TestCase):
    
    def setUp(self):
        self.estimator = ITML()
        super(TestITML, self).setUp()


class TestLSML(_TestQuadrupletsBase, unittest.TestCase):

    def setUp(self):
        self.estimator = LSML()
        super(TestLSML, self).setUp()


class TestMMC(_TestPairsBase, unittest.TestCase):
    
    def setUp(self):
        self.estimator = MMC()
        super(TestMMC, self).setUp()

        
class TestSDML(_TestPairsBase, unittest.TestCase):
    
    def setUp(self):
        self.estimator = SDML()
        super(TestSDML, self).setUp()


if __name__ == '__main__':
    unittest.main()
