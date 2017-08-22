from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class MatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        raise NotImplementedError(
            'MatrixTransformer should not be instantiated')

    def duplicate_instance(self):
        return self.__class__(**self.get_params())

    def individual_size(self, input_dim):
        raise NotImplementedError('individual_size() is not implemented')

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('fit() is not implemented')

    def transformer(self):
        """Returns the evolved transformation matrix from the Mahalanobis matrix.

        Returns
        -------
        L : (d x d) matrix
        """
        return self.L

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
            X = self.X_
        else:
            X = check_array(X, accept_sparse=True)
        L = self.transformer()
        return X.dot(L.T)

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.

        .. math:: M = L^{\\top} L

        Returns
        -------
        M : (d x d) matrix
        """
        L = self.transformer()
        return L.T.dot(L)
