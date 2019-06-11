"""
The :mod:`metric_learn.exceptions` module includes all custom warnings and
error classes used across metric-learn.
"""
from numpy.linalg import LinAlgError


class PreprocessorError(Exception):

  def __init__(self, original_error):
    err_msg = ("An error occurred when trying to use the "
               "preprocessor: {}").format(repr(original_error))
    super(PreprocessorError, self).__init__(err_msg)


class NonPSDError(LinAlgError):

    def __init__(self):
      err_msg = "Matrix is not positive semidefinite (PSD)."
      super(LinAlgError, self).__init__(err_msg)
