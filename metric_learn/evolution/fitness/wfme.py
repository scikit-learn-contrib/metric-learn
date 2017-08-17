import numpy as np

from scipy.spatial import distance

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from .base import BaseFitness


class WeightedFMeasureFitness(BaseFitness):
    def __init__(self, weighted=False, sig=5, kmeans__n_init=1,
                 random_state=None):
        self.params = {
            'weighted': weighted,
            'sig': sig,
            'kmeans__n_init': kmeans__n_init,
            'random_state': random_state,
        }

    @staticmethod
    def available(method):
        return method in ['wfme']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        le = LabelEncoder()
        y = le.fit_transform(y)

        kmeans = KMeans(
            n_clusters=len(np.unique(y)),
            n_init=self.params['kmeans__n_init'],
            random_state=self.params['random_state'],
        )
        kmeans.fit(X)

        if not self.params['weighted']:
            return f1_score(y, kmeans.labels_, average='weighted')

        r = distance.cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
        h = np.exp(-r / (self.params['sig']**2))

        N = confusion_matrix(y, kmeans.labels_)

        wN = np.zeros(h.shape)
        for l in range(wN.shape[0]):  # label
            for c in range(wN.shape[0]):  # cluster
                for j in range(wN.shape[0]):
                    wN[l, c] += h[l, c] * N[l, j]

        Prec = wN / wN.sum(axis=0)
        Rec = wN / N.sum(axis=1)[:, None]
        F = (2 * Prec * Rec) / (Prec + Rec)

        wFme = 0
        for l in range(F.shape[0]):
            wFme += (N[l, :].sum() / N.sum()) * F[l, :].max()

        return wFme
