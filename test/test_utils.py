import pytest
import numpy as np
from sklearn.exceptions import DataConversionWarning
from metric_learn import NCA
from metric_learn._util import check_tuples, make_context


@pytest.fixture
def X_prep():
    """Basic array for testing when using a preprocessor"""
    X = np.array([[1, 2],
                  [2, 3]])
    return X


@pytest.fixture
def X_no_prep():
    """Basic array for testing when using no preprocessor"""
    X = np.array([[[1., 2.3], [2.3, 5.3]],
                  [[2.3, 4.3], [0.2, 0.4]]])
    return X


@pytest.mark.parametrize('estimator, expected',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_make_context(estimator, expected):
    """test the make_name function"""
    assert make_context(estimator) == expected


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_X, preprocessor',
                         [(X_prep, True), (X_no_prep, False)])
def test_check_tuples_invalid_t(estimator, context, load_X, preprocessor):
  """Checks that the exception are raised if t is not the one expected"""
  X = load_X()
  expected_msg = ("Tuples of 3 element(s) expected{}. Got tuples of 2 "
                  "element(s) instead (shape={}):\ninput={}.\n"
                  .format(context, X.shape, X))
  with pytest.raises(ValueError) as raised_error:
    check_tuples(X, t=3, preprocessor=preprocessor, estimator=estimator)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('X, found, expected, preprocessor',
                         [(5, '0', '2', True),
                          (5, '0', '3', False),
                          ([1, 2], '1', '2', True),
                          ([1, 2], '1', '3', False),
                          ([[[[5]]]], '4', '2', True),
                          ([[[[5]]]], '4', '3', False),
                          ([[1], [3]], '2', '3', False),
                          ([[[1], [3]]], '3', '2', True)])
def test_check_tuples_invalid_shape(estimator, context, X, found, expected,
                                    preprocessor):
  """Checks that a value error with the appropriate message is raised if
  shape is invalid (not 2D with preprocessor or 3D with no preprocessor)
  """
  X = np.array(X)
  msg = ("{}D array expected{} when using {} preprocessor. Found {}D array "
         "instead:\ninput={}.\n"
         .format(expected, context, 'a' if preprocessor else 'no', found, X))
  with pytest.raises(ValueError) as raised_error:
      check_tuples(X, preprocessor=preprocessor, ensure_min_samples=0,
                   estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_check_tuples_invalid_n_features(estimator, context, X_no_prep):
  """Checks that the right warning is printed if not enough features
  Here we only test if no preprocessor (otherwise we don't ensure this)
  """
  msg = ("Found array with 2 feature(s) (shape={}) while"
         " a minimum of 3 is required{}.".format(X_no_prep.shape, context))
  with pytest.raises(ValueError) as raised_error:
      check_tuples(X_no_prep, preprocessor=False, ensure_min_features=3,
                   estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_X, preprocessor',
                         [(X_prep, True), (X_no_prep, False)])
def test_check_tuples_invalid_n_samples(estimator, context, load_X,
                                        preprocessor):
  """Checks that the right warning is printed if n_samples is too small"""
  X = load_X()
  msg = ("Found array with 2 sample(s) (shape={}) while a minimum of 3 "
         "is required{}.".format(X.shape, context))
  with pytest.raises(ValueError) as raised_error:
    check_tuples(X, preprocessor=preprocessor, ensure_min_samples=3,
                 estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_X, preprocessor',
                         [(X_prep, True), (X_no_prep, False)])
def test_check_tuples_invalid_dtype_convertible(estimator, context, load_X,
                                                preprocessor):
  """Checks that a warning is raised if a convertible input is converted to
  float"""
  X = load_X().astype(object)
  msg = ("Data with input dtype object was converted to float64{}."
         .format(context))
  with pytest.warns(DataConversionWarning) as raised_warning:
    check_tuples(X, preprocessor=preprocessor, dtype=np.float64,
                 warn_on_dtype=True, estimator=estimator)
  assert str(raised_warning[0].message) == msg


@pytest.mark.parametrize('preprocessor, X',
                         [(True, np.array([['a', 'b'],
                                           ['e', 'b']])),
                          (False, np.array([[['b', 'v'], ['a', 'd']],
                                            [['x', 'u'], ['c', 'a']]]))])
def test_check_tuples_invalid_dtype_not_convertible(preprocessor, X):
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float
  """
  with pytest.raises(ValueError):
    check_tuples(X, preprocessor=preprocessor, dtype=np.float64)


@pytest.mark.parametrize('t', [2, None])
def test_check_tuples_valid_t(t, X_prep, X_no_prep):
  """For inputs that have the right matrix dimension (2D or 3D for instance),
  checks that checking the number of tuples (pairs, quadruplets, etc) raises
  no warning
  """
  with pytest.warns(None) as record:
    check_tuples(X_prep, preprocessor=True, t=t)
    check_tuples(X_no_prep, preprocessor=False, t=t)
  assert len(record) == 0


@pytest.mark.parametrize('X',
                         [np.array([[2.5, 0.1, 2.6],
                                    [1.6, 4.8, 9.1]]),
                          np.array([[2, 0, 2],
                                    [1, 4, 9]]),
                          np.array([["img1.png", "img3.png"],
                                    ["img2.png", "img4.png"]]),
                          [[2, 0, 2],
                           [1, 4, 9]],
                          [np.array([2, 0, 2]),
                           np.array([1, 4, 9])],
                          ((2, 0, 2),
                           (1, 4, 9))
                          ])
def test_check_tuples_valid_with_preprocessor(X):
  """Test that valid inputs when using a preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_tuples(X, preprocessor=True)
  assert len(record) == 0


@pytest.mark.parametrize('X',
                         [np.array([[[2.5], [0.1], [2.6]],
                                    [[1.6], [4.8], [9.1]],
                                    [[5.6], [2.8], [6.1]]]),
                          np.array([[[2], [0], [2]],
                                    [[1], [4], [9]],
                                    [[1], [5], [3]]]),
                          [[[2], [0], [2]],
                           [[1], [4], [9]],
                           [[3], [4], [29]]],
                          (((2, 1), (0, 2), (2, 3)),
                           ((1, 2), (4, 4), (9, 3)),
                           ((3, 1), (4, 4), (29, 4)))])
def test_check_tuples_valid_without_preprocessor(X):
  """Test that valid inputs when using no preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_tuples(X, preprocessor=False)
  assert len(record) == 0


def test_check_tuples_behaviour_auto_dtype(X_no_prep):
  """Checks that check_tuples allows by default every type if using a
  preprocessor, and numeric types if using no preprocessor"""
  X_prep = [['img1.png', 'img2.png'], ['img3.png', 'img5.png']]
  with pytest.warns(None) as record:
    check_tuples(X_prep, preprocessor=True)
  assert len(record) == 0

  with pytest.warns(None) as record:
      check_tuples(X_no_prep)  # numeric type
  assert len(record) == 0

  # not numeric type
  X_no_prep = np.array([[['img1.png'], ['img2.png']],
                        [['img3.png'], ['img5.png']]])
  X_no_prep = X_no_prep.astype(object)
  with pytest.raises(ValueError):
      check_tuples(X_no_prep)


def test_check_tuples_invalid_complex_data():
  """Checks that the right error message is thrown if given complex data (
  this comes from sklearn's check_array's message)"""
  X = np.array([[[1 + 2j, 3 + 4j], [5 + 7j, 5 + 7j]],
                [[1 + 3j, 2 + 4j], [5 + 8j, 1 + 7j]]])
  msg = ("Complex data not supported\n"
         "{}\n".format(X))
  with pytest.raises(ValueError) as raised_error:
    check_tuples(X)
  assert str(raised_error.value) == msg
