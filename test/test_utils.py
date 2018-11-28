import pytest
from collections import namedtuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.testing import set_random_state
from sklearn.base import clone
from metric_learn._util import (check_input, make_context, preprocess_tuples,
                                make_name, preprocess_points,
                                check_collapsed_pairs)
from metric_learn import (ITML, LSML, MMC, RCA, SDML, Covariance, LFDA,
                          LMNN, MLKR, NCA, ITML_Supervised, LSML_Supervised,
                          MMC_Supervised, RCA_Supervised, SDML_Supervised)
from metric_learn.base_metric import ArrayIndexer
from metric_learn.exceptions import PreprocessorError
from sklearn.datasets import make_regression, make_blobs
from .test_sklearn_compat import build_pairs, build_quadruplets


def mock_preprocessor(indices):
  """A preprocessor for testing purposes that returns an all ones 3D array
  """
  return np.ones((indices.shape[0], 3))


@pytest.mark.parametrize('type_of_inputs', ['other', 'tuple', 'classics', 2,
                                            int, NCA()])
def test_check_input_invalid_type_of_inputs(type_of_inputs):
  """Tests that an invalid type of inputs in check_inputs raises an error."""
  with pytest.raises(ValueError) as e:
    check_input([[0.2, 2.1], [0.2, .8]], type_of_inputs=type_of_inputs)
  msg = ("Unknown value {} for type_of_inputs. Valid values are "
         "'classic' or 'tuples'.".format(type_of_inputs))
  assert str(e.value) == msg


#  ---------------- test check_input with 'tuples' type_of_input' ------------


@pytest.fixture
def tuples_prep():
  """Basic array for testing when using a preprocessor"""
  tuples = np.array([[1, 2],
                     [2, 3]])
  return tuples


@pytest.fixture
def tuples_no_prep():
  """Basic array for testing when using no preprocessor"""
  tuples = np.array([[[1., 2.3], [2.3, 5.3]],
                     [[2.3, 4.3], [0.2, 0.4]]])
  return tuples


@pytest.mark.parametrize('estimator, expected',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_make_context(estimator, expected):
  """test the make_name function"""
  assert make_context(estimator) == expected


@pytest.mark.parametrize('estimator, expected',
                         [(NCA(), "NCA"), ('NCA', "NCA"), (None, None)])
def test_make_name(estimator, expected):
  """test the make_name function"""
  assert make_name(estimator) == expected


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_tuples, preprocessor',
                         [(tuples_prep, mock_preprocessor),
                          (tuples_no_prep, None),
                          (tuples_no_prep, mock_preprocessor)])
def test_check_tuples_invalid_tuple_size(estimator, context, load_tuples,
                                         preprocessor):
  """Checks that the exception are raised if tuple_size is not the one
  expected"""
  tuples = load_tuples()
  preprocessed_tuples = (preprocess_tuples(tuples, preprocessor)
                         if (preprocessor is not None and
                         tuples.ndim == 2) else tuples)
  expected_msg = ("Tuples of 3 element(s) expected{}. Got tuples of 2 "
                  "element(s) instead (shape={}):\ninput={}.\n"
                  .format(context, preprocessed_tuples.shape,
                          preprocessed_tuples))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples', tuple_size=3,
                preprocessor=preprocessor, estimator=estimator)
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('tuples, found, expected, preprocessor',
                         [(5, '0', '2D array of indicators or 3D array of '
                                   'formed tuples', mock_preprocessor),
                          (5, '0', '3D array of formed tuples', None),
                          ([1, 2], '1', '2D array of indicators or 3D array '
                                        'of formed tuples', mock_preprocessor),
                          ([1, 2], '1', '3D array of formed tuples', None),
                          ([[[[5]]]], '4', '2D array of indicators or 3D array'
                                           ' of formed tuples',
                           mock_preprocessor),
                          ([[[[5]]]], '4', '3D array of formed tuples', None),
                          ([[1], [3]], '2', '3D array of formed '
                                            'tuples', None)])
def test_check_tuples_invalid_shape(estimator, context, tuples, found,
                                    expected, preprocessor):
  """Checks that a value error with the appropriate message is raised if
  shape is invalid (not 2D with preprocessor or 3D with no preprocessor)
  """
  tuples = np.array(tuples)
  msg = ("{} expected{}{}. Found {}D array instead:\ninput={}. Reshape your "
         "data{}.\n"
         .format(expected, context, ' when using a preprocessor'
                 if preprocessor else '', found, tuples,
                 ' and/or use a preprocessor' if
                 (not preprocessor and tuples.ndim == 2) else ''))
  with pytest.raises(ValueError) as raised_error:
      check_input(tuples, type_of_inputs='tuples',
                  preprocessor=preprocessor, ensure_min_samples=0,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_check_tuples_invalid_n_features(estimator, context, tuples_no_prep):
  """Checks that the right warning is printed if not enough features
  Here we only test if no preprocessor (otherwise we don't ensure this)
  """
  msg = ("Found array with 2 feature(s) (shape={}) while"
         " a minimum of 3 is required{}.".format(tuples_no_prep.shape,
                                                 context))
  with pytest.raises(ValueError) as raised_error:
      check_input(tuples_no_prep, type_of_inputs='tuples',
                  preprocessor=None, ensure_min_features=3,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_tuples, preprocessor',
                         [(tuples_prep, mock_preprocessor),
                          (tuples_no_prep, None),
                          (tuples_no_prep, mock_preprocessor)])
def test_check_tuples_invalid_n_samples(estimator, context, load_tuples,
                                        preprocessor):
  """Checks that the right warning is printed if n_samples is too small"""
  tuples = load_tuples()
  msg = ("Found array with 2 sample(s) (shape={}) while a minimum of 3 "
         "is required{}.".format((preprocess_tuples(tuples, preprocessor)
                                 if (preprocessor is not None and
                                 tuples.ndim == 2) else tuples).shape,
                                 context))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=preprocessor,
                ensure_min_samples=3, estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_tuples, preprocessor',
                         [(tuples_prep, mock_preprocessor),
                          (tuples_no_prep, None),
                          (tuples_no_prep, mock_preprocessor)])
def test_check_tuples_invalid_dtype_convertible(estimator, context,
                                                load_tuples, preprocessor):
  """Checks that a warning is raised if a convertible input is converted to
  float"""
  tuples = load_tuples().astype(object)  # here the object conversion is
  # useless for the tuples_prep case, but this allows to test the
  # tuples_prep case

  if preprocessor is not None:  # if the preprocessor is not None we
    # overwrite it to have a preprocessor that returns objects
    def preprocessor(indices):  #
      # preprocessor that returns objects
      return np.ones((indices.shape[0], 3)).astype(object)

  msg = ("Data with input dtype object was converted to float64{}."
         .format(context))
  with pytest.warns(DataConversionWarning) as raised_warning:
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=preprocessor, dtype=np.float64,
                warn_on_dtype=True, estimator=estimator)
  assert str(raised_warning[0].message) == msg


def test_check_tuples_invalid_dtype_not_convertible_with_preprocessor(
        tuples_prep):
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float, when using a preprocessor
  """

  def preprocessor(indices):
    # preprocessor that returns objects
    return np.full((indices.shape[0], 3), 'a')

  with pytest.raises(ValueError):
    check_input(tuples_prep, type_of_inputs='tuples',
                preprocessor=preprocessor, dtype=np.float64)


def test_check_tuples_invalid_dtype_not_convertible_without_preprocessor(
        tuples_no_prep):
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float, when using no preprocessor
  """
  tuples = np.full_like(tuples_no_prep, 'a', dtype=object)
  with pytest.raises(ValueError):
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=None, dtype=np.float64)


@pytest.mark.parametrize('tuple_size', [2, None])
def test_check_tuples_valid_tuple_size(tuple_size, tuples_prep, tuples_no_prep):
  """For inputs that have the right matrix dimension (2D or 3D for instance),
  checks that checking the number of tuples (pairs, quadruplets, etc) raises
  no warning if there is the right number of points in a tuple.
  """
  with pytest.warns(None) as record:
    check_input(tuples_prep, type_of_inputs='tuples',
                preprocessor=mock_preprocessor, tuple_size=tuple_size)
    check_input(tuples_no_prep, type_of_inputs='tuples', preprocessor=None,
                tuple_size=tuple_size)
  assert len(record) == 0


@pytest.mark.parametrize('tuples',
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
                           (1, 4, 9)),
                          np.array([[[1.2, 2.2], [1.4, 3.3]],
                                    [[2.6, 2.3], [3.4, 5.0]]])])
def test_check_tuples_valid_with_preprocessor(tuples):
  """Test that valid inputs when using a preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(tuples, type_of_inputs='tuples',
                preprocessor=mock_preprocessor)
  assert len(record) == 0


@pytest.mark.parametrize('tuples',
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
def test_check_tuples_valid_without_preprocessor(tuples):
  """Test that valid inputs when using no preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(tuples, type_of_inputs='tuples', preprocessor=None)
  assert len(record) == 0


def test_check_tuples_behaviour_auto_dtype(tuples_no_prep):
  """Checks that check_tuples allows by default every type if using a
  preprocessor, and numeric types if using no preprocessor"""
  tuples_prep = [['img1.png', 'img2.png'], ['img3.png', 'img5.png']]
  with pytest.warns(None) as record:
    check_input(tuples_prep, type_of_inputs='tuples',
                preprocessor=mock_preprocessor)
  assert len(record) == 0

  with pytest.warns(None) as record:
      check_input(tuples_no_prep, type_of_inputs='tuples')  # numeric type
  assert len(record) == 0

  # not numeric type
  tuples_no_prep = np.array([[['img1.png'], ['img2.png']],
                             [['img3.png'], ['img5.png']]])
  tuples_no_prep = tuples_no_prep.astype(object)
  with pytest.raises(ValueError):
      check_input(tuples_no_prep, type_of_inputs='tuples')


def test_check_tuples_invalid_complex_data():
  """Checks that the right error message is thrown if given complex data (
  this comes from sklearn's check_array's message)"""
  tuples = np.array([[[1 + 2j, 3 + 4j], [5 + 7j, 5 + 7j]],
                     [[1 + 3j, 2 + 4j], [5 + 8j, 1 + 7j]]])
  msg = ("Complex data not supported\n"
         "{}\n".format(tuples))
  with pytest.raises(ValueError) as raised_error:
    check_input(tuples, type_of_inputs='tuples')
  assert str(raised_error.value) == msg


# ------------- test check_input with 'classic' type_of_inputs ----------------


@pytest.fixture
def points_prep():
  """Basic array for testing when using a preprocessor"""
  points = np.array([1, 2])
  return points


@pytest.fixture
def points_no_prep():
  """Basic array for testing when using no preprocessor"""
  points = np.array([[1., 2.3],
                     [2.3, 4.3]])
  return points


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('points, found, expected, preprocessor',
                         [(5, '0', '1D array of indicators or 2D array of '
                                   'formed points', mock_preprocessor),
                          (5, '0', '2D array of formed points', None),
                          ([1, 2], '1', '2D array of formed points', None),
                          ([[[5]]], '3', '1D array of indicators or 2D '
                                         'array of formed points',
                           mock_preprocessor),
                          ([[[5]]], '3', '2D array of formed points', None)])
def test_check_classic_invalid_shape(estimator, context, points, found,
                                     expected, preprocessor):
  """Checks that a value error with the appropriate message is raised if
  shape is invalid (valid being 1D or 2D with preprocessor or 2D with no
  preprocessor)
  """
  points = np.array(points)
  msg = ("{} expected{}{}. Found {}D array instead:\ninput={}. Reshape your "
         "data{}.\n"
         .format(expected, context, ' when using a preprocessor'
                 if preprocessor else '', found, points,
                 ' and/or use a preprocessor' if
                 (not preprocessor and points.ndim == 1) else ''))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic', preprocessor=preprocessor,
                ensure_min_samples=0,
                estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
def test_check_classic_invalid_n_features(estimator, context,
                                          points_no_prep):
  """Checks that the right warning is printed if not enough features
  Here we only test if no preprocessor (otherwise we don't ensure this)
  """
  msg = ("Found array with 2 feature(s) (shape={}) while"
         " a minimum of 3 is required{}.".format(points_no_prep.shape,
                                                 context))
  with pytest.raises(ValueError) as raised_error:
      check_input(points_no_prep, type_of_inputs='classic', preprocessor=None,
                  ensure_min_features=3,
                  estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_points, preprocessor',
                         [(points_prep, mock_preprocessor),
                          (points_no_prep, None),
                          (points_no_prep, mock_preprocessor)])
def test_check_classic_invalid_n_samples(estimator, context, load_points,
                                         preprocessor):
  """Checks that the right warning is printed if n_samples is too small"""
  points = load_points()
  msg = ("Found array with 2 sample(s) (shape={}) while a minimum of 3 "
         "is required{}.".format((preprocess_points(points,
                                                    preprocessor)
                                 if preprocessor is not None and
                                 points.ndim == 1 else
                                 points).shape,
                                 context))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic', preprocessor=preprocessor,
                ensure_min_samples=3,
                estimator=estimator)
  assert str(raised_error.value) == msg


@pytest.mark.parametrize('estimator, context',
                         [(NCA(), " by NCA"), ('NCA', " by NCA"), (None, "")])
@pytest.mark.parametrize('load_points, preprocessor',
                         [(points_prep, mock_preprocessor),
                          (points_no_prep, None),
                          (points_no_prep, mock_preprocessor)])
def test_check_classic_invalid_dtype_convertible(estimator, context,
                                                 load_points,
                                                 preprocessor):
  """Checks that a warning is raised if a convertible input is converted to
  float"""
  points = load_points().astype(object)  # here the object conversion is
  # useless for the points_prep case, but this allows to test the
  # points_prep case

  if preprocessor is not None:  # if the preprocessor is not None we
    # overwrite it to have a preprocessor that returns objects
    def preprocessor(indices):
      # preprocessor that returns objects
      return np.ones((indices.shape[0], 3)).astype(object)

  msg = ("Data with input dtype object was converted to float64{}."
         .format(context))
  with pytest.warns(DataConversionWarning) as raised_warning:
    check_input(points, type_of_inputs='classic',
                preprocessor=preprocessor, dtype=np.float64,
                warn_on_dtype=True, estimator=estimator)
  assert str(raised_warning[0].message) == msg


@pytest.mark.parametrize('preprocessor, points',
                         [(mock_preprocessor, np.array([['a', 'b'],
                                                        ['e', 'b']])),
                          (None, np.array([[['b', 'v'], ['a', 'd']],
                                           [['x', 'u'], ['c', 'a']]]))])
def test_check_classic_invalid_dtype_not_convertible(preprocessor, points):
  """Checks that a value error is thrown if attempting to convert an
  input not convertible to float
  """
  with pytest.raises(ValueError):
    check_input(points, type_of_inputs='classic',
                preprocessor=preprocessor, dtype=np.float64)


@pytest.mark.parametrize('points',
                         [["img1.png", "img3.png", "img2.png"],
                          np.array(["img1.png", "img3.png", "img2.png"]),
                          [2, 0, 2, 1, 4, 9],
                          range(10),
                          np.array([2, 0, 2]),
                          (2, 0, 2),
                          np.array([[1.2, 2.2],
                                    [2.6, 2.3]])])
def test_check_classic_valid_with_preprocessor(points):
  """Test that valid inputs when using a preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(points, type_of_inputs='classic',
                preprocessor=mock_preprocessor)
  assert len(record) == 0


@pytest.mark.parametrize('points',
                         [np.array([[2.5, 0.1, 2.6],
                                    [1.6, 4.8, 9.1],
                                    [5.6, 2.8, 6.1]]),
                          np.array([[2, 0, 2],
                                    [1, 4, 9],
                                    [1, 5, 3]]),
                          [[2, 0, 2],
                           [1, 4, 9],
                           [3, 4, 29]],
                          ((2, 1, 0, 2, 2, 3),
                           (1, 2, 4, 4, 9, 3),
                           (3, 1, 4, 4, 29, 4))])
def test_check_classic_valid_without_preprocessor(points):
  """Test that valid inputs when using no preprocessor raises no warning"""
  with pytest.warns(None) as record:
    check_input(points, type_of_inputs='classic', preprocessor=None)
  assert len(record) == 0


def test_check_classic_by_default():
  """Checks that 'classic' is the default behaviour of check_input"""
  assert (check_input([[2, 3], [3, 2]]) ==
          check_input([[2, 3], [3, 2]], type_of_inputs='classic')).all()


def test_check_classic_behaviour_auto_dtype(points_no_prep):
  """Checks that check_input (for points) allows by default every type if
  using a preprocessor, and numeric types if using no preprocessor"""
  points_prep = ['img1.png', 'img2.png', 'img3.png', 'img5.png']
  with pytest.warns(None) as record:
    check_input(points_prep, type_of_inputs='classic',
                preprocessor=mock_preprocessor)
  assert len(record) == 0

  with pytest.warns(None) as record:
      check_input(points_no_prep, type_of_inputs='input')  # numeric type
  assert len(record) == 0

  # not numeric type
  points_no_prep = np.array(['img1.png', 'img2.png', 'img3.png',
                             'img5.png'])
  points_no_prep = points_no_prep.astype(object)
  with pytest.raises(ValueError):
      check_input(points_no_prep, type_of_inputs='classic')


def test_check_classic_invalid_complex_data():
  """Checks that the right error message is thrown if given complex data (
  this comes from sklearn's check_array's message)"""
  points = np.array([[[1 + 2j, 3 + 4j], [5 + 7j, 5 + 7j]],
                     [[1 + 3j, 2 + 4j], [5 + 8j, 1 + 7j]]])
  msg = ("Complex data not supported\n"
         "{}\n".format(points))
  with pytest.raises(ValueError) as raised_error:
    check_input(points, type_of_inputs='classic')
  assert str(raised_error.value) == msg


# ----------------------------- Test preprocessor -----------------------------


X = np.array([[0.89, 0.11, 1.48, 0.12],
              [2.63, 1.08, 1.68, 0.46],
              [1.00, 0.59, 0.62, 1.15]])


class MockFileLoader:
  """Preprocessor that takes a root file path at construction and simulates
  fetching the file in the specific root folder when given the name of the
  file"""

  def __init__(self, root):
    self.root = root
    self.folders = {'fake_root': {'img0.png': X[0],
                                  'img1.png': X[1],
                                  'img2.png': X[2]
                                  },
                    'other_folder': {}  # empty folder
                    }

  def __call__(self, path_list):
    images = list()
    for path in path_list:
      images.append(self.folders[self.root][path])
    return np.array(images)


def mock_id_loader(list_of_indicators):
  """A preprocessor as a function that takes indicators (strings) and
  returns the corresponding samples"""
  points = []
  for indicator in list_of_indicators:
      points.append(X[int(indicator[2:])])
  return np.array(points)


tuples_list = [np.array([[0, 1],
                         [2, 1]]),

               np.array([['img0.png', 'img1.png'],
                         ['img2.png', 'img1.png']]),

               np.array([['id0', 'id1'],
                         ['id2', 'id1']])
               ]

points_list = [np.array([0, 1, 2, 1]),

               np.array(['img0.png', 'img1.png', 'img2.png', 'img1.png']),

               np.array(['id0', 'id1', 'id2', 'id1'])
               ]

preprocessors = [X, MockFileLoader('fake_root'), mock_id_loader]


@pytest.fixture
def y_tuples():
  y = [-1, 1]
  return y


@pytest.fixture
def y_points():
  y = [0, 1, 0, 0]
  return y


@pytest.mark.parametrize('preprocessor, tuples', zip(preprocessors,
                                                     tuples_list))
def test_preprocessor_weakly_supervised(preprocessor, tuples, y_tuples):
  """Tests different ways to use the preprocessor argument: an array,
  a class callable, and a function callable, with a weakly supervised
  algorithm
  """
  nca = ITML(preprocessor=preprocessor)
  nca.fit(tuples, y_tuples)


@pytest.mark.parametrize('preprocessor, points', zip(preprocessors,
                                                     points_list))
def test_preprocessor_supervised(preprocessor, points, y_points):
  """Tests different ways to use the preprocessor argument: an array,
  a class callable, and a function callable, with a supervised algorithm
  """
  lfda = LFDA(preprocessor=preprocessor)
  lfda.fit(points, y_points)


@pytest.mark.parametrize('estimator', ['NCA', NCA(), None])
def test_preprocess_tuples_invalid_message(estimator):
  """Checks that if the preprocessor does some weird stuff, the preprocessed
  input is detected as weird. Checks this for preprocess_tuples."""

  context = make_context(estimator) + (' after the preprocessor '
                                       'has been applied')

  def preprocessor(sequence):
    return np.ones((len(sequence), 2, 2))  # returns a 3D array instead of 2D

  with pytest.raises(ValueError) as raised_error:
      check_input(np.ones((3, 2)), type_of_inputs='tuples',
                  preprocessor=preprocessor, estimator=estimator)
  expected_msg = ("3D array of formed tuples expected{}. Found 4D "
                  "array instead:\ninput={}. Reshape your data{}.\n"
                  .format(context, np.ones((3, 2, 2, 2)),
                          ' and/or use a preprocessor' if preprocessor
                          is not None else ''))
  assert str(raised_error.value) == expected_msg


@pytest.mark.parametrize('estimator', ['NCA', NCA(), None])
def test_preprocess_points_invalid_message(estimator):
  """Checks that if the preprocessor does some weird stuff, the preprocessed
  input is detected as weird."""

  context = make_context(estimator) + (' after the preprocessor '
                                       'has been applied')

  def preprocessor(sequence):
    return np.ones((len(sequence), 2, 2))  # returns a 3D array instead of 2D

  with pytest.raises(ValueError) as raised_error:
    check_input(np.ones((3,)), type_of_inputs='classic',
                preprocessor=preprocessor, estimator=estimator)
  expected_msg = ("2D array of formed points expected{}. "
                  "Found 3D array instead:\ninput={}. Reshape your data{}.\n"
                  .format(context, np.ones((3, 2, 2)),
                          ' and/or use a preprocessor' if preprocessor
                          is not None else ''))
  assert str(raised_error.value) == expected_msg


def test_progress_message_preprocessor_points(capsys):
  """Tests that when using a preprocessor on points, a message is printed
  """
  points = np.array([1, 2, 4])

  def fun(row):
    return [[1, 1], [3, 3], [4, 4]]

  preprocess_points(points, preprocessor=fun)
  out, _ = capsys.readouterr()
  assert out == "Preprocessing points...\n"


def test_progress_message_preprocessor_tuples(capsys):
  """Tests that when using a preprocessor on tuples, a message is printed
  """
  tuples = np.array([[1, 2],
                     [2, 3],
                     [4, 5]])

  def fun(row):
    return np.array([[1, 1], [3, 3], [4, 4]])

  preprocess_tuples(tuples, preprocessor=fun)
  out, _ = capsys.readouterr()
  assert out == "Preprocessing tuples...\n"


def test_preprocessor_error_message():
  """Tests whether the preprocessor returns a preprocessor error when there
  is a problem using the preprocessor
  """
  preprocessor = ArrayIndexer(np.array([[1.2, 3.3], [3.1, 3.2]]))

  # with tuples
  X = np.array([[[2, 3], [3, 3]], [[2, 3], [3, 2]]])
  # There are less samples than the max index we want to preprocess
  with pytest.raises(PreprocessorError):
    preprocess_tuples(X, preprocessor)

  # with points
  X = np.array([[1], [2], [3], [3]])
  with pytest.raises(PreprocessorError):
    preprocess_points(X, preprocessor)


@pytest.mark.parametrize('estimator', [ITML(), LSML(), MMC(), SDML()],
                         ids=['ITML', 'LSML', 'MMC', 'SDML'])
def test_error_message_tuple_size(estimator):
  """Tests that if a tuples learner is not given the good number of points
  per tuple, it throws an error message"""
  estimator = clone(estimator)
  set_random_state(estimator)
  invalid_pairs = np.array([[[1.3, 6.3], [3., 6.8], [6.5, 4.4]],
                            [[1.9, 5.3], [1., 7.8], [3.2, 1.2]]])
  y = [1, 1]
  with pytest.raises(ValueError) as raised_err:
    estimator.fit(invalid_pairs, y)
  expected_msg = ("Tuples of {} element(s) expected{}. Got tuples of 3 "
                  "element(s) instead (shape=(2, 3, 2)):\ninput={}.\n"
                  .format(estimator._tuple_size, make_context(estimator),
                          invalid_pairs))
  assert str(raised_err.value) == expected_msg


@pytest.mark.parametrize('estimator', [ITML(), LSML(), MMC(), RCA(), SDML(),
                                       Covariance(), LFDA(), LMNN(), MLKR(),
                                       NCA(), ITML_Supervised(),
                                       LSML_Supervised(), MMC_Supervised(),
                                       RCA_Supervised(), SDML_Supervised()],
                         ids=['ITML', 'LSML', 'MMC', 'RCA', 'SDML',
                              'Covariance', 'LFDA', 'LMNN', 'MLKR', 'NCA',
                              'ITML_Supervised', 'LSML_Supervised',
                              'MMC_Supervised', 'RCA_Supervised',
                              'SDML_Supervised'])
def test_error_message_t_score_pairs(estimator):
  """tests that if you want to score_pairs on triplets for instance, it returns
  the right error message
  """
  estimator = clone(estimator)
  set_random_state(estimator)
  estimator.check_preprocessor()
  triplets = np.array([[[1.3, 6.3], [3., 6.8], [6.5, 4.4]],
                       [[1.9, 5.3], [1., 7.8], [3.2, 1.2]]])
  with pytest.raises(ValueError) as raised_err:
    estimator.score_pairs(triplets)
  expected_msg = ("Tuples of 2 element(s) expected{}. Got tuples of 3 "
                  "element(s) instead (shape=(2, 3, 2)):\ninput={}.\n"
                  .format(make_context(estimator), triplets))
  assert str(raised_err.value) == expected_msg


def test_preprocess_tuples_simple_example():
  """Test the preprocessor on a very simple example of tuples to ensure the
  result is as expected"""
  array = np.array([[1, 2],
                    [2, 3],
                    [4, 5]])

  def fun(row):
    return np.array([[1, 1], [3, 3], [4, 4]])

  expected_result = np.array([[[1, 1], [1, 1]],
                              [[3, 3], [3, 3]],
                              [[4, 4], [4, 4]]])

  assert (preprocess_tuples(array, fun) == expected_result).all()


def test_preprocess_points_simple_example():
  """Test the preprocessor on very simple examples of points to ensure the
  result is as expected"""
  array = np.array([1, 2, 4])

  def fun(row):
    return [[1, 1], [3, 3], [4, 4]]

  expected_result = np.array([[1, 1],
                              [3, 3],
                              [4, 4]])

  assert (preprocess_points(array, fun) == expected_result).all()


# ----------------------------------------------------------------------------
# test that supervised algorithms using a preprocessor behave consistently
# with their no-preprocessor equivalent


Dataset = namedtuple('Dataset', 'formed_points points_indicators labels data')


@pytest.fixture
def build_classification(rng):
  """Basic array for testing when using a preprocessor"""
  X, y = shuffle(*make_blobs(random_state=rng),
                 random_state=rng)
  indices = shuffle(np.arange(X.shape[0]), random_state=rng)
  indices = indices.astype(int)
  return Dataset(X[indices], indices, y, X)


@pytest.fixture
def build_regression(rng):
  """Basic array for testing when using a preprocessor"""
  X, y = shuffle(*make_regression(n_samples=100, n_features=5,
                                  random_state=rng),
                 random_state=rng)
  indices = shuffle(np.arange(X.shape[0]), random_state=rng)
  indices = indices.astype(int)
  return Dataset(X[indices], indices, y, X)


RNG = check_random_state(0)

classifiers = [Covariance(),
               LFDA(),
               LMNN(),
               NCA(),
               RCA(),
               ITML_Supervised(max_iter=5),
               LSML_Supervised(),
               MMC_Supervised(max_iter=5),
               RCA_Supervised(num_chunks=10),  # less chunks because we only
               # have a few data in the test
               SDML_Supervised()]

regressors = [MLKR()]

estimators = [(classifier, build_classification(RNG)) for classifier in
              classifiers]
estimators += [(regressor, build_regression(RNG)) for regressor in
               regressors]

ids_estimators = list(map(lambda x: x.__class__.__name__, classifiers +
                          regressors))


@pytest.mark.parametrize('estimator, dataset', estimators,
                         ids=ids_estimators)
def test_same_with_or_without_preprocessor_classic(estimator, dataset):
  """Test that supervised algorithms using a preprocessor behave consistently
  with their no-preprocessor equivalent.
  """
  (formed_points_train, formed_points_test,
   y_train, y_test, points_indicators_train,
   points_indicators_test) = train_test_split(dataset.formed_points,
                                              dataset.labels,
                                              dataset.points_indicators,
                                              random_state=RNG)

  def make_random_state(estimator):
    rs = {}
    if estimator.__class__.__name__[-11:] == '_Supervised':
      rs['random_state'] = check_random_state(0)
    return rs

  estimator_without_prep = clone(estimator)
  set_random_state(estimator_without_prep)
  estimator_without_prep.set_params(preprocessor=None)
  estimator_without_prep.fit(formed_points_train, y_train,
                             **make_random_state(estimator))
  embedding_without_prep = estimator_without_prep.transform(formed_points_test)

  estimator_with_prep = clone(estimator)
  set_random_state(estimator_with_prep)
  estimator_with_prep.set_params(preprocessor=dataset.data)
  estimator_with_prep.fit(points_indicators_train, y_train,
                          **make_random_state(estimator))
  embedding_with_prep = estimator_with_prep.transform(points_indicators_test)

  estimator_with_prep_formed = clone(estimator)
  set_random_state(estimator_with_prep_formed)
  estimator_with_prep_formed.set_params(preprocessor=dataset.data)
  estimator_with_prep_formed.fit(formed_points_train, y_train,
                                 **make_random_state(estimator))
  embedding_with_prep_formed = estimator_with_prep_formed.transform(
      formed_points_test)

  # test transform
  assert (embedding_with_prep == embedding_without_prep).all()
  assert (embedding_with_prep == embedding_with_prep_formed).all()

  # test score_pairs
  assert (estimator_without_prep.score_pairs(
      formed_points_test[np.array([[0, 2], [5, 3]])]) ==
      estimator_with_prep.score_pairs(
          points_indicators_test[np.array([[0, 2], [5, 3]])])).all()

  assert (
      estimator_with_prep.score_pairs(
          points_indicators_test[np.array([[0, 2], [5, 3]])]) ==
      estimator_with_prep_formed.score_pairs(
          formed_points_test[np.array([[0, 2], [5, 3]])])).all()


@pytest.mark.parametrize('estimator, build_dataset',
                         [(ITML(), build_pairs),
                          (LSML(), build_quadruplets),
                          (MMC(max_iter=2), build_pairs),
                          (SDML(), build_pairs)],
                         ids=['itml', 'lsml', 'mmc', 'sdml'])
def test_same_with_or_without_preprocessor_tuples(estimator, build_dataset):
  """For weakly supervised algorithms, test that using a preprocessor or not
  (with the appropriate corresponding inputs) give the same result.
  """
  (X, tuples, y, tuples_train, tuples_test, y_train,
   y_test, _) = build_dataset(preprocessor=mock_preprocessor)
  formed_tuples_train = X[tuples_train]
  formed_tuples_test = X[tuples_test]

  estimator_with_preprocessor = clone(estimator)
  set_random_state(estimator_with_preprocessor)
  estimator_with_preprocessor.set_params(preprocessor=X)
  estimator_with_preprocessor.fit(tuples_train, y_train)

  estimator_without_preprocessor = clone(estimator)
  set_random_state(estimator_without_preprocessor)
  estimator_without_preprocessor.set_params(preprocessor=None)
  estimator_without_preprocessor.fit(formed_tuples_train, y_train)

  estimator_with_prep_formed = clone(estimator)
  set_random_state(estimator_with_prep_formed)
  estimator_with_prep_formed.set_params(preprocessor=X)
  estimator_with_prep_formed.fit(tuples_train, y_train)

  # test prediction methods
  for method in ["predict", "decision_function"]:
    if hasattr(estimator, method):
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(tuples_test)
      output_without_prep = getattr(estimator_without_preprocessor,
                                    method)(formed_tuples_test)
      assert np.array(output_with_prep == output_without_prep).all()
      output_with_prep = getattr(estimator_with_preprocessor,
                                 method)(tuples_test)
      output_with_prep_formed = getattr(estimator_with_prep_formed,
                                        method)(formed_tuples_test)
      assert np.array(output_with_prep == output_with_prep_formed).all()

  # test score_pairs
  output_with_prep = estimator_with_preprocessor.score_pairs(
      tuples_test[:, :2])
  output_without_prep = estimator_without_preprocessor.score_pairs(
      formed_tuples_test[:, :2])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.score_pairs(
      tuples_test[:, :2])
  output_without_prep = estimator_with_prep_formed.score_pairs(
      formed_tuples_test[:, :2])
  assert np.array(output_with_prep == output_without_prep).all()

  # test transform
  output_with_prep = estimator_with_preprocessor.transform(
      tuples_test[:, 0])
  output_without_prep = estimator_without_preprocessor.transform(
      formed_tuples_test[:, 0])
  assert np.array(output_with_prep == output_without_prep).all()

  output_with_prep = estimator_with_preprocessor.transform(
      tuples_test[:, 0])
  output_without_prep = estimator_with_prep_formed.transform(
      formed_tuples_test[:, 0])
  assert np.array(output_with_prep == output_without_prep).all()


def test_check_collapsed_pairs_raises_no_error():
  """Checks that check_collapsed_pairs raises no error if no collapsed pairs
  is present"""
  pairs_ok = np.array([[[0.1, 3.3], [3.3, 0.1]],
                       [[0.1, 3.3], [3.3, 0.1]],
                       [[2.5, 8.1], [0.1, 3.3]]])
  check_collapsed_pairs(pairs_ok)


def test_check_collapsed_pairs_raises_error():
  """Checks that check_collapsed_pairs raises no error if no collapsed pairs
  is present"""
  pairs_not_ok = np.array([[[0.1, 3.3], [0.1, 3.3]],
                           [[0.1, 3.3], [3.3, 0.1]],
                           [[2.5, 8.1], [2.5, 8.1]]])
  with pytest.raises(ValueError) as e:
    check_collapsed_pairs(pairs_not_ok)
  assert str(e.value) == ("2 collapsed pairs found (where the left element is "
                          "the same as the right element), out of 3 pairs in"
                          " total.")
