import numpy as np

from sklearn.metrics import pairwise_distances

from .base_fitness import BaseFitness


class ClassSeparationFitness(BaseFitness):

    def __init__(self, **kwargs):
        super(ClassSeparationFitness, self).__init__(**kwargs)

    @staticmethod
    def available(method):
        return method in ['class_separation']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        unique_labels, label_inds = np.unique(y, return_inverse=True)
        ratio = 0
        for li in range(len(unique_labels)):
            Xc = X[label_inds == li]
            Xnc = X[label_inds != li]
            ratio += pairwise_distances(Xc).mean() \
                / pairwise_distances(Xc, Xnc).mean()

        return -ratio / len(unique_labels)
