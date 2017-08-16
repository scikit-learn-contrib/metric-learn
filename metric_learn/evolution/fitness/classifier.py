from sklearn.base import ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from .base import BaseFitness


class ClassifierFitness(BaseFitness):
    def __init__(self, classifier, **kwargs):
        self.params = {
            'classifier': classifier,
        }
        self.classifier_params = kwargs

    @staticmethod
    def available(method):
        return (method in ['knn', 'scv', 'lsvc']) \
            or isinstance(method, ClassifierMixin)

    def __call__(self, X_train, X_test, y_train, y_test):
        classifier = self._build_classifier(
            self.params['classifier'],
            self.classifier_params,
        )
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)

    def _build_classifier(self, classifier, params):
        if isinstance(classifier, ClassifierMixin):
            return clone(classifier)
        elif classifier == 'svc':
            return SVC(**params)
        elif classifier == 'lsvc':
            return LinearSVC(**params)
        elif classifier == 'knn':
            return KNeighborsClassifier(**params)

        raise ValueError('Invalid `classifier` parameter value.')
