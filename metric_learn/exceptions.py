"""
The :mod:`metric_learn.exceptions` module includes all custom warnings and
error classes used across metric-learn.
"""


class PreprocessorError(Exception):

  def __init__(self, original_error):
    err_msg = ("An error occurred when trying to use the "
               "preprocessor: {}").format(repr(original_error))
    super(PreprocessorError, self).__init__(err_msg)
