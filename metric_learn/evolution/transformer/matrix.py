from sklearn.utils.validation import check_array

from .base_transformer import BaseTransformer


class MatrixTransformer(BaseTransformer):
    def __init__(self):
        raise NotImplementedError(
            'MatrixTransformer should not be instantiated')

    def transformer(self):
        """Returns the evolved transformation matrix from the Mahalanobis matrix.

        Returns
        -------
        L : (d x d) matrix
        """
        return self.L

    def transform(self, X):
        """Applies the metric transformation.

        Parameters
        ----------
        X : (n x d) matrix, optional
            Data to transform.

        Returns
        -------
        transformed : (n x d) matrix
            Input data transformed to the metric space by :math:`XL^{\\top}`
        """
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
