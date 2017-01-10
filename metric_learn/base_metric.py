from numpy.linalg import inv,cholesky


class BaseMetricLearner(object):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = inv(cholesky(M))

    Returns
    -------
    L : (d x d) matrix
    """
    return inv(cholesky(self.metric()))

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X
    L = self.transformer()
    return X.dot(L.T)
  
  def fit_transform(self, *args, **kwargs):
    """
    Function calls .fit() and returns the result of .transform()
    Essentially, it runs the relevant Metric Learning algorithm with .fit()
    and returns the metric-transformed input data.

    Paramters
    ---------
    
    Since all the parameters passed to fit_transform are passed on to
    fit(), the parameters to be passed must be noted from the corresponding
    Metric Learning algorithm's fit method.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`

    """
    self.fit(*args, **kwargs)
    return self.transform()

  def get_params(self, deep=False):
    """Get parameters for this metric learner.

    Parameters
    ----------
    deep: boolean, optional
        @WARNING doesn't do anything, only exists because
        scikit-learn has this on BaseEstimator.

    Returns
    -------
    params : mapping of string to any
        Parameter names mapped to their values.
    """
    return self.params

  def set_params(self, **kwarg):
    """Set the parameters of this metric learner.

    Overwrites any default parameters or parameters specified in constructor.

    Returns
    -------
    self
    """
    self.params.update(kwarg)
    return self
  
  # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L287
  def __repr__(self):
    class_name = self.__class__.__name__
    return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

###############################################################################

# _iteritems taken from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/externals/six.py#L437
def _iteritems(d, **kw):
    """Return an iterator over the (key, value) pairs of a dictionary."""
    import sys
    return iter(getattr(d, "items" if sys.version_info[0] == 3 else "iteritems")(**kw))

# __pprint taken from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L124
def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    import numpy as np
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(_iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
###############################################################################