import numpy as np

from scipy.spatial import distance

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .base import BaseFitness


class WeightedPurityFitness(BaseFitness):
    def __init__(self, sig=5, kmeans__n_init=1):
        self.params = {
            'sig': sig,
            'kmeans__n_init': kmeans__n_init,
        }

    @staticmethod
    def available(method):
        return method in ['wpur']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        le = LabelEncoder()
        y = le.fit_transform(y)

        kmeans = KMeans(
            n_clusters=len(np.unique(y)),
            n_init=self.params['kmeans__n_init']
        )
        kmeans.fit(X)

        r = distance.cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
        h = np.exp(-r / (self.params['sig']**2))

        N = confusion_matrix(y, kmeans.labels_)

        wN = np.zeros(h.shape)
        for l in range(wN.shape[0]):  # label
            for c in range(wN.shape[0]):  # cluster
                for j in range(wN.shape[0]):
                    wN[l, c] += h[l, c] * N[l, j]

        return wN.max(axis=0).sum() / wN.sum()
