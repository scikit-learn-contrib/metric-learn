"""This file is for fixing imports due to different APIs
depending on the scikit-learn version"""
import sklearn
from packaging import version
SKLEARN_AT_LEAST_0_22 = (version.parse(sklearn.__version__)
                         >= version.parse('0.22.0'))
if SKLEARN_AT_LEAST_0_22:
    from sklearn.utils._testing import (set_random_state,
                                        assert_warns_message,
                                        ignore_warnings,
                                        assert_allclose_dense_sparse,
                                        _get_args)
    from sklearn.utils.estimator_checks import (_is_public_parameter
                                                as is_public_parameter)
    from sklearn.metrics._scorer import get_scorer
else:
    from sklearn.utils.testing import (set_random_state,
                                       assert_warns_message,
                                       ignore_warnings,
                                       assert_allclose_dense_sparse,
                                       _get_args)
    from sklearn.utils.estimator_checks import is_public_parameter
    from sklearn.metrics.scorer import get_scorer

__all__ = ['set_random_state', 'assert_warns_message', 'set_random_state',
           'ignore_warnings', 'assert_allclose_dense_sparse', '_get_args',
           'is_public_parameter', 'get_scorer']
