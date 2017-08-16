import numpy as np

from scipy.spatial import distance

from sklearn.cluster import KMeans

from .matrix import MatrixTransformer


class KMeansTransformer(MatrixTransformer, BaseBuilder):
    def __init__(self, transformer='full', n_clusters='classes',
                 function='distance', n_init=1, random_state=None, **kwargs):
        self._transformer = None
        self.params = {
            'transformer': transformer,
            'n_clusters': n_clusters,
            'function': function,
            'n_init': n_init,
            'random_state': random_state,
        }
        self.params.update(**kwargs)

    def individual_size(self, input_dim):
        if self._transformer is None:
            self._transformer = self.build_transformer()

        return self._transformer.individual_size(input_dim)

    def fit(self, X, y, flat_weights):
        self._transformer = self.build_transformer()
        self._transformer.fit(X, y, flat_weights)

        if self.params['n_clusters'] == 'classes':
            n_clusters = np.unique(y).size
        elif self.params['n_clusters'] == 'same':
            n_clusters = X.shape[1]
        else:
            n_clusters = int(self.params['n_clusters'])

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.params['random_state'],
            n_init=self.params['n_init'],
        )
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_

        return self

    def transform(self, X):
        Xt = self._transformer.transform(X)

        if self.params['function'] == 'distance':
            return distance.cdist(Xt, self.centers)
        elif self.params['function'] == 'product':
            return np.dot(Xt, self.centers.T)

        raise ValueError('Invalid function param.')

    def transformer(self):
        return self._transformer
