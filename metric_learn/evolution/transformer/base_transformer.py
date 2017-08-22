from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        raise NotImplementedError('BaseTransformer should not be instantiated')

    def duplicate_instance(self):
        return self.__class__(**self.get_params())

    def individual_size(self, input_dim):
        raise NotImplementedError('individual_size() is not implemented')

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('fit() is not implemented')

    def transform(self, X):
        raise NotImplementedError('transform() is not implemented')

    def transformer(self):
        raise NotImplementedError('transformer() is not implemented')

    def metric(self):
        raise NotImplementedError('metric() is not implemented')
