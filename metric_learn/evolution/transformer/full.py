import numpy as np

from .matrix import MatrixTransformer


class FullMatrixTransformer(MatrixTransformer):
    def __init__(self, num_dims=None):
        self.num_dims = num_dims

    def individual_size(self, input_dim):
        if self.num_dims is None:
            return input_dim**2

        return input_dim * self.num_dims

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]
        if self.individual_size(input_dim) != len(flat_weights):
            raise Exception(
                '`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(self.individual_size(input_dim), len(flat_weights))
            )

        self.L = np.reshape(
            flat_weights,
            (len(flat_weights) // input_dim, input_dim)
        )
        return self
